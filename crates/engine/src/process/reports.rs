//! What this process states to its daemon: the schedulers' and devices' counters once a second,
//! its reports on the process interface, and its lifecycle state and description on the request
//! interface.

use std::time::Duration;

use anyhow::{Context as _, Result};
use worker::control::Control;

use super::MainProcess;

/// States an engine's lifecycle state on the request interface.
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

/// States on every live front that this engine failed, and why: the process ends next, and a
/// connection that closes says only that it went.
pub(super) fn state_failure(control: &mut Control, reason: &str) {
    let message = proto::request::wire::FromEngine {
        kind: Some(proto::request::wire::from_engine::Kind::State(
            proto::request::wire::State {
                state: proto::request::wire::EngineState::Failed as i32,
                info: None,
                error: reason.to_owned(),
            },
        )),
    };
    for front in control.report_fronts() {
        let Some(connection) = control.front_mut(front) else {
            continue;
        };
        if let Err(error) = connection.send(&message) {
            tracing::warn!(%error, front, "a daemon was not told why this engine failed");
        }
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
        // The driver calls this thread waited in, and the range records an idle device left
        // unread, reach the next batch output.
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
                depth: u32::from(worker.running(0).is_some()),
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
            // A scheduler states its counters on its own thread: a metrics report is not worth
            // stalling the control thread for, so the report takes what is there.
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

    /// States the lifecycle state this engine is in on the request interface, to every live
    /// front, `described` with what it serves: the limits, the batch capacity and the KV cache
    /// segments a successor maps.
    ///
    /// The daemon's launch path waits for the state a start ends in - `SERVING`, or `PARKED` for
    /// an engine started beside a serving one - and records the description it carries. A state
    /// that changes nothing of what the engine holds (a drain's `DRAINING` and `PARKED`) is
    /// stated without one and keeps what was announced: describing the cache asks every scheduler
    /// for what it holds.
    pub(super) fn announce(&mut self, described: bool) -> Result<()> {
        let state = self.state.lifecycle().state();
        let info = if described {
            let info = self.engine_info()?;
            self.announced = Some(info.clone());
            Some(info)
        } else {
            None
        };
        tracing::info!(
            state = ?state,
            schedulers = self.schedulers.len(),
            devices = self.workers.len(),
            "stating what this engine is to the daemon"
        );
        announce(&mut self.control, state, info)
    }

    /// What an attaching daemon is told this process is: the info it announced, for a state that
    /// carries one, and nothing otherwise.
    pub(super) fn announced_info(&mut self) -> Option<proto::request::wire::EngineInfo> {
        // The value announced, not a fresh one: describing the cache asks every scheduler for
        // what it holds, and a scheduler that is busy does not respond for as long as the stall
        // bound - which is a control thread that cannot serve the handover's own attach.
        self.announced.clone()
    }

    /// What the daemon records of this process: the limits every request is checked against, the
    /// batch the workers were started for, and the KV cache segments a successor on the same
    /// devices maps.
    ///
    /// The figures are the workers' own (`SchedulerHandle::info`), except the cache segments, which
    /// each scheduler describes for itself: a successor needs what the workers hold now, not what
    /// this process sized at its start.
    pub(super) fn engine_info(&self) -> Result<proto::request::wire::EngineInfo> {
        // A peer of an epoch over several hosts holds ranks and no scheduler: it admits no request
        // and holds no cache a successor maps, so it states that it schedules nothing.
        if self.schedulers.is_empty() && matches!(self.peers, super::Peers::Peer { .. }) {
            return Ok(proto::request::wire::EngineInfo::default());
        }
        let first = self
            .schedulers
            .first()
            .context("this process runs no scheduler, so it states no engine info")?;
        let info = first.info();
        let mut segments = Vec::new();
        for scheduler in &self.schedulers {
            // A group that cannot describe its cache is reported with what the others stated rather
            // than failing the start: the segments are what a *successor* maps, and a successor that
            // finds one missing refuses by name.
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
                scheduler::prefix_seeds(cache)
                    .into_iter()
                    .zip(&cache.groups)
                    .filter_map(|(seed, group)| {
                        (!matches!(group.kind, ipc::CacheGroupKind::Draft))
                            .then_some(seed.0.to_vec())
                    })
                    .collect()
            }),
            latent_seed: first.latent_seed(),
        })
    }

    /// States one report to the daemon on the process interface.
    ///
    /// A report is unasked, so a failure to state it costs the daemon's record and not this
    /// engine's service: it is logged, and the engine's own transition has already happened.
    pub(super) fn send_report(&mut self, report: proto::process::Report) {
        tracing::info!(report = ?report.kind, "reporting to the daemon");
        if let Err(error) = self.process.report(report) {
            tracing::warn!(%error, "reporting this engine's state to the daemon failed");
        }
    }
}
