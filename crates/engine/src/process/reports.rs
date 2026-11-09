//! What this process states to its daemon: the schedulers' and devices' counters once a second,
//! the program-layer reports, and the engine's state and description on the request interface.

use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use ipc::ProgramStatus;
use worker::control::Control;

use super::MainProcess;

/// The name of a program-layer status, for the line that says which step was answered.
pub(super) fn status_name(status: &ProgramStatus) -> &'static str {
    match status {
        ProgramStatus::Precopied { .. } => "precopied",
        ProgramStatus::PrecopyFailed { .. } => "precopy_failed",
        ProgramStatus::Switched { .. } => "switched",
        ProgramStatus::Committed { .. } => "committed",
        ProgramStatus::Reverted { .. } => "reverted",
        ProgramStatus::Purged { .. } => "purged",
        ProgramStatus::PolicyReverted { .. } => "policy_reverted",
        ProgramStatus::Failed { .. } => "failed",
    }
}

/// The report one program-layer status states.
///
/// A status the daemon learns another way - none today - answers `None`, and the caller logs it
/// instead of inventing a report.
fn report_of(status: &ProgramStatus) -> Option<proto::process::Report> {
    use proto::process::report::Kind;

    let kind = match status {
        ProgramStatus::Precopied {
            generation,
            uncompared,
        } => Kind::Precopied(proto::process::Precopied {
            generation: *generation,
            uncompared: uncompared.clone(),
        }),
        ProgramStatus::PrecopyFailed { failure, .. } => {
            Kind::PrecopyFailed(proto::process::PrecopyFailed {
                reason: failure.reason(),
            })
        }
        ProgramStatus::Switched { generation } => Kind::Switched(proto::process::Switched {
            generation: *generation,
        }),
        ProgramStatus::Committed { .. } => Kind::Committed(proto::process::Committed {}),
        ProgramStatus::Reverted { reason, .. } => Kind::Reverted(proto::process::Reverted {
            reason: reason.clone(),
        }),
        ProgramStatus::Purged { generation } => Kind::Purged(proto::process::Purged {
            generation: *generation,
        }),
        // A policy that failed is reported with the backend and revision it came from, so the
        // daemon's record takes the previous policy back.
        ProgramStatus::PolicyReverted {
            scheduler,
            backend,
            revision,
            reason,
            fatal,
        } => Kind::PolicyReverted(proto::process::PolicyReverted {
            scheduler: Some(*scheduler),
            backend: backend.clone(),
            revision: *revision,
            reason: reason.clone(),
            fatal: *fatal,
        }),
        // A probation failure is not a fault of the device: the revert that follows it states the
        // generation it left and the first error, so reporting it twice would name a device the
        // failure did not happen on.
        ProgramStatus::Failed { .. } => return None,
    };
    Some(proto::process::Report { kind: Some(kind) })
}

/// States a main process's state on the request interface.
pub(super) fn announce(
    control: &mut Control,
    state: proto::request::wire::EngineState,
    info: Option<proto::request::wire::EngineInfo>,
) -> Result<()> {
    let message = proto::request::wire::FromEngine {
        kind: Some(proto::request::wire::from_engine::Kind::State(
            proto::request::wire::State {
                state: state as i32,
                info,
                error: String::new(),
            },
        )),
    };

    // Every live front is told: a handover closes the launching daemon's front and leaves the
    // attached ones serving, so a state stated to that front alone would end this process for a
    // daemon that left.
    let mut delivered = 0;
    let mut failure = None;
    for front in control.report_fronts() {
        let Some(connection) = control.front_mut(front) else {
            continue;
        };
        match connection.send(&message) {
            Ok(()) => delivered += 1,
            Err(error) => failure = Some(error),
        }
    }
    match failure {
        Some(error) if delivered == 0 => {
            Err(error).context("stating what this process is to the daemon")
        }
        _ => Ok(()),
    }
}

/// States to the daemon that launched this process why its start failed.
pub(super) fn state_failure(control: &mut Control, reason: &str) {
    let sent = control
        .daemon_mut()
        .send(&proto::request::wire::FromEngine {
            kind: Some(proto::request::wire::from_engine::Kind::State(
                proto::request::wire::State {
                    state: proto::request::wire::EngineState::Failed as i32,
                    info: None,
                    error: reason.to_owned(),
                },
            )),
        });
    if let Err(error) = sent {
        tracing::warn!(%error, "the daemon was not told why this process's start failed");
    }
}

/// How long a metrics pass waits on one scheduler's counters: the report is sent once a second and a
/// pass that waited longer than this would be reporting what the next pass reports anyway.
const METRICS_WAIT: Duration = Duration::from_millis(50);

impl MainProcess {
    /// Sends the engine's counters to the daemon, one report a second and a
    /// final flush when the process ends.
    ///
    /// The worker states its own figures: the launches and device nanoseconds
    /// its runners counted, the jobs in flight the control thread's watchdog
    /// reads, and the pools the arena books against the rank's budget - which is
    /// the Storage Server's ledger invariant, a device never mapping more than its budget
    /// allows. The scheduler-owned counters (`steps`, `batches`, `tokens`, the
    /// policy's fuel and calls) come from the schedulers' handle and stay zero
    /// until it exists.
    pub(super) fn send_metrics(&mut self) {
        // The driver calls this thread waited in, and the fence records an idle device left unread,
        // reach the next batch result.
        let current = platform::cupti::blocking::Calls::current();
        worker::profiling::counters::add_blocking("control", &current.since(&self.control_calls));
        self.control_calls = current;
        if let Some(fences) = worker::profiling::fences::Fences::installed() {
            fences.poll(false);
        }
        let mut kernel_launches = 0_u64;
        let mut device_ns = 0_u64;
        let mut queue = Vec::with_capacity(self.workers.len());
        let mut memory = Vec::with_capacity(self.workers.len());
        for (rank, worker) in self.workers.iter().enumerate() {
            let counters = worker.counters();
            kernel_launches += counters.kernel_launches;
            device_ns += counters.device_ns;
            queue.push(proto::request::wire::QueueDepth {
                rank: rank as u32,
                depth: worker.watchdog().jobs.len() as u32,
            });
            // One allocator books the device's three pools (`arena::DeviceMemory`), so the three
            // figures sum against one budget and item 5's invariant is reported as it is decided.
            let pools = worker.memory();
            memory.push(proto::request::wire::DeviceMemory {
                device: worker.device().0 as u32,
                budget: pools.budget,
                pools: ["graph", "scratch", "resident"]
                    .into_iter()
                    .zip([pools.graph, pools.scratch, pools.resident])
                    .map(|(pool, mapped)| proto::request::wire::PoolMemory {
                        pool: pool.to_owned(),
                        mapped,
                    })
                    .collect(),
            });
        }
        // The schedulers state their own figures; a node that holds none reports the workers'
        // and a step count of zero, because no step happened here.
        let (mut steps, mut batches, mut tokens, mut policy_fuel, mut policy_calls) =
            (0, 0, 0, 0, 0);
        for scheduler in &self.schedulers {
            // A scheduler answers its counters on its own thread: a metrics report is not worth
            // stalling the control thread for, so the pass takes what is there.
            let Ok(counters) = scheduler.counters().recv_timeout(METRICS_WAIT) else {
                continue;
            };
            steps += counters.steps;
            batches += counters.batches;
            tokens += counters.tokens;
            policy_fuel += counters.policy_fuel;
            policy_calls += counters.policy_calls;
            // The launches and the device time are the workers', counted where they are issued:
            // a scheduler's copy of them would be a second source for one figure.
        }
        let metrics = proto::request::wire::Metrics {
            steps,
            batches,
            tokens,
            kernel_launches,
            device_ns,
            queue,
            policy_fuel,
            policy_calls,
            memory,
        };
        // The scheduler statistics travel beside the counters: a daemon records
        // `uniserve:num_requests_running` and its kind from this snapshot, so a process that
        // sent only the allocator's figures would report an idle deployment forever.
        let mut snapshots = Vec::with_capacity(self.schedulers.len());
        for (index, (scheduler, reporter)) in self
            .schedulers
            .iter()
            .zip(&mut self.stats_reporters)
            .enumerate()
        {
            let snapshot = reporter.snapshot(scheduler.stats());
            if let Ok(snapshot) = snapshot.encode() {
                snapshots.push(proto::request::wire::Stats {
                    scheduler: index as u32,
                    snapshot,
                });
            }
        }
        // Every live front is told: the daemon that launched this process, and each daemon
        // that attached to it. A handover closes the launching daemon's front
        // and leaves the successor's serving, so a report aimed at index 0 alone would stop
        // exactly when the successor owns the engine - and the successor's `/metrics` would
        // hold no figure of what it inherited. Each daemon keeps its own registry, so a report
        // on each front is one figure per registry.
        let fronts = self.control.report_fronts();
        let mut delivered = 0;
        let mut failure = None;
        for front in fronts {
            let Some(connection) = self.control.front_mut(front) else {
                continue;
            };
            let sent = connection
                .send(&proto::request::wire::FromEngine {
                    kind: Some(proto::request::wire::from_engine::Kind::Metrics(
                        metrics.clone(),
                    )),
                })
                .and_then(|()| {
                    for stats in &snapshots {
                        connection.send(&proto::request::wire::FromEngine {
                            kind: Some(proto::request::wire::from_engine::Kind::Stats(
                                stats.clone(),
                            )),
                        })?;
                    }
                    Ok(())
                });
            match sent {
                Ok(()) => delivered += 1,
                Err(error) => failure = Some(error),
            }
        }
        if delivered == 0
            && let Some(error) = failure
        {
            tracing::warn!(
                %error,
                "reporting the engine's metrics failed: no front of this process is live"
            );
        }
    }

    /// Reports a program-layer status to the daemon.
    pub(super) fn report(&mut self, status: ProgramStatus) {
        // The program layer's answers, in the vocabulary the daemon's epoch record reads: a precopy
        // that finished, the switchover, the commit, the revert and the purge, each naming its
        // generation, and a failure carrying the first error. The daemon needs no answer to any of
        // them; it marks the epoch's state with what it is told, and the probation's rules are
        // applied on these reports.
        let Some(report) = report_of(&status) else {
            tracing::debug!(?status, "a program-layer status needs no report");
            return;
        };
        self.send_report(report);
    }

    /// States one report to the daemon.
    ///
    /// A report is unasked, so a failure to state it costs the daemon's record and not this
    /// process's service: it is logged, and the epoch's own transition has already happened.
    /// States what this process is on the request interface.
    ///
    /// The daemon's launch path waits for `Ready` (or `Standby`, for a process started beside a
    /// serving one) and records the info it carries for the epoch. Nothing else on either side moves
    /// before it: the first `Precopy` carries the scheduling policy and the first `Switchover` fills
    /// the generation, so a process that waited to be ready until it had a policy would wait for the
    /// command that waits for it - the deadlock the caller's stuck run shows.
    pub(super) fn announce(&mut self, state: proto::request::wire::EngineState) -> Result<()> {
        self.reported_state = state as i32;
        let info = match state {
            proto::request::wire::EngineState::Ready
            | proto::request::wire::EngineState::Standby => Some(self.engine_info()?),
            _ => None,
        };
        // A parked process states no info and keeps what it announced: its cache's description
        // holds until it serves again.
        if info.is_some() {
            self.announced = info.clone();
        }
        tracing::info!(
            state = ?state,
            schedulers = self.schedulers.len(),
            devices = self.workers.len(),
            "stating what this process is to the daemon"
        );
        if matches!(
            state,
            proto::request::wire::EngineState::Ready | proto::request::wire::EngineState::Standby
        ) {
            self.announced_at.get_or_insert_with(Instant::now);
        }
        announce(&mut self.control, state, info)
    }

    /// What an attaching daemon is told this process is: the info it announced, for a state that
    /// carries one, and nothing otherwise.
    pub(super) fn announced_info(&mut self) -> Option<proto::request::wire::EngineInfo> {
        // The value announced, not a fresh one: describing the cache asks every scheduler for
        // what it holds, and a scheduler that is busy does not answer for as long as the stall
        // bound - which is a control thread that cannot serve the handover's own attach
        // (`501fa5e6f`: the engine read the attach and then said nothing for sixty seconds).
        self.announced.clone()
    }

    /// What the daemon records of this process: the limits every request is checked against, the
    /// batch the workers were started for, and the KV cache segments a successor on the same
    /// devices maps.
    ///
    /// The figures are the workers' own (`SchedulerHandle::info`), except the cache segments, which
    /// each scheduler describes for itself: a successor needs what the workers hold now, not what
    /// this process planned.
    pub(super) fn engine_info(&self) -> Result<proto::request::wire::EngineInfo> {
        let first = self
            .schedulers
            .first()
            .context("this process runs no scheduler, so it states no engine info")?;
        let info = first.info();
        let mut segments = Vec::new();
        for scheduler in &self.schedulers {
            // A group that cannot describe its cache is reported with what the others stated rather
            // than failing the start: the segments are what a *successor* maps, and a successor that
            // finds one missing refuses by name, while a process that refused to state what it is
            // over this would hang the epoch before its first policy - the deadlock this report
            // exists to end.
            let groups = match scheduler
                .describe_cache()
                .recv_timeout(self.watchdog.stall_bound())
            {
                Ok(Ok(groups)) => groups,
                Ok(Err(error)) => {
                    tracing::warn!(%error, "a worker group did not describe its cache");
                    continue;
                }
                Err(error) => {
                    tracing::warn!(%error, "a scheduler did not describe its cache in time");
                    continue;
                }
            };
            segments.extend(groups.into_iter().map(|group| {
                proto::request::wire::CacheSegments {
                    worker_group: group.worker_group,
                    segments: group
                        .segments
                        .into_iter()
                        .map(|segment| proto::request::wire::CacheSegment {
                            epoch: segment.epoch,
                            ordinal: segment.ordinal,
                            units: segment.units,
                        })
                        .collect(),
                    held: group.held,
                    units: group.units,
                    scratch: group.scratch,
                }
            }));
        }
        Ok(proto::request::wire::EngineInfo {
            limits: Some((&first.generation_limits()).into()),
            video_denoiser: info
                .video_denoiser
                .as_ref()
                .map(|denoiser| denoiser.description.clone())
                .unwrap_or_default(),
            latent_encoding_units: first
                .media_units(ipc::MediaCall::LatentEncoding)
                .unwrap_or(0),
            schedulers: u32::try_from(self.schedulers.len()).unwrap_or(u32::MAX),
            capacity: Some(proto::request::wire::BatchCapacity {
                max_batch: info.max_batch_calls,
                max_num_batched_tokens: u64::from(info.max_batch_tokens),
                // The launch's own figure, not the worker's row count: the rows include the
                // guidance prefix, which is not a request.
                max_running_requests: self.max_running_requests,
            }),
            segments,
            prefix_seeds: info.prefix_cache.as_ref().map_or_else(Vec::new, |cache| {
                cache
                    .seeds()
                    .into_iter()
                    .zip(&cache.groups)
                    .filter_map(|(seed, group)| {
                        (!matches!(group.kind, ipc::CacheGroupKind::Draft)).then_some(seed)
                    })
                    .collect()
            }),
            latent_seed: first.latent_seed(),
        })
    }

    pub(super) fn send_report(&mut self, report: proto::process::Report) {
        if let Err(error) = self.process.report(report) {
            tracing::warn!(%error, "reporting this epoch's state to the daemon failed");
        }
    }
}
