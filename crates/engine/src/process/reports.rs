//! What the process states to its daemon: its lifecycle state and what it serves on every
//! request front, the schedulers' statistics and the process's metrics once a second, the
//! watchdog's stalls and sticky errors, and the failures and faults it observed on the process
//! interface.

use std::time::{Duration, Instant};

use ipc::batch::BatchId;
use proto::process::report;
use proto::report::SchedulerStats;
use proto::request::RequestId;
use proto::request::wire::{self, from_engine};

use super::MainProcess;
use super::lifecycle::{self, Lifecycle};

/// How often the process states its metrics.
pub(super) const METRICS_EVERY: Duration = Duration::from_secs(1);

impl MainProcess {
    /// States the lifecycle state the engine is in on every live front, `described` with what it
    /// serves ([`Self::engine_info`]), which the daemon records from the state a start or a
    /// promotion ends in. A state that changes nothing the engine serves (`Draining`, a drained
    /// `Parked`, `Stopped`) is stated without it.
    pub(super) fn announce(&mut self, described: bool) {
        let info = if described {
            self.announced = self.engine_info();
            self.announced.clone()
        } else {
            None
        };
        let state = wire::State {
            state: self.lifecycle.state() as i32,
            info,
            error: String::new(),
        };
        tracing::info!(state = ?self.lifecycle.state(), "stating what this engine is");
        self.broadcast(&from_engine::Kind::State(state));
    }

    /// What the daemon records of this engine: the limits every request is checked against, its
    /// schedulers, the batch its executors were started for and the page seeds of its cache
    /// groups. `None` before the start ended.
    ///
    /// A process with no scheduler (a peer) states its limits and no scheduler. It states no KV
    /// cache segments: nothing of the cache is handed to another epoch, which learns the buffer
    /// pool's cache by `List`.
    pub(super) fn engine_info(&self) -> Option<wire::EngineInfo> {
        let engine = self.engine.as_ref()?;

        // The seeds of the cache groups every rank computes; a draft cache's pages are its own
        // and reach no other epoch.
        let prefix_seeds = engine
            .capacities
            .iter()
            .map(|(_, capacity)| capacity)
            .find(|capacity| !capacity.seeds.is_empty())
            .map(|capacity| {
                capacity
                    .seeds
                    .iter()
                    .zip(&capacity.cache_groups)
                    .filter(|(_, (kind, _))| !matches!(kind, ipc::CacheGroupKind::Draft))
                    .map(|(seed, _)| seed.0.to_vec())
                    .collect()
            })
            .unwrap_or_default();

        // A bound no captured bucket states binds nothing, which the wire states as zero.
        let max_batch = u32::try_from(engine.ceilings.max_batch).unwrap_or(0);
        let max_num_batched_tokens = match engine.ceilings.max_num_batched_tokens {
            u32::MAX => 0,
            tokens => u64::from(tokens),
        };
        Some(wire::EngineInfo {
            limits: Some((&engine.limits).into()),
            video_denoiser: Vec::new(),
            latent_encoding_units: 0,
            schedulers: u32::try_from(engine.schedulers.len()).unwrap_or(u32::MAX),
            capacity: Some(wire::BatchCapacity {
                max_batch,
                max_num_batched_tokens,
                max_running_requests: u32::try_from(
                    self.launch.engine_config.max_running_requests.get(),
                )
                .unwrap_or(u32::MAX),
            }),
            segments: Vec::new(),
            prefix_seeds,
            latent_seed: 0,
        })
    }

    /// One scheduler's statistics of the last second: forwarded to every front, and counted into
    /// the metrics. The scheduler's step, batch and policy counts are lifetime values, of which
    /// the metrics state the sum over the schedulers; its tokens are the interval's, which the
    /// metrics sum since the start.
    pub(super) fn stats(&mut self, scheduler: usize, stats: SchedulerStats) {
        let lifetime = [
            stats.steps,
            stats.step_counter,
            stats.policy_calls,
            stats.policy_fuel,
        ];
        if self.lifetime.len() <= scheduler {
            self.lifetime.resize(scheduler + 1, [0; 4]);
        }
        let before = std::mem::replace(&mut self.lifetime[scheduler], lifetime);
        let grown = |index: usize| lifetime[index].saturating_sub(before[index]);
        self.metrics.steps += grown(0);
        self.metrics.batches += grown(1);
        self.metrics.policy_calls += grown(2);
        self.metrics.policy_fuel += grown(3);
        self.metrics.tokens += stats.batch_tokens;

        let rank = u32::try_from(scheduler).unwrap_or(u32::MAX);
        let depth = u32::try_from(stats.num_waiting_reqs).unwrap_or(u32::MAX);
        match self
            .metrics
            .queue
            .iter_mut()
            .find(|queue| queue.rank == rank)
        {
            Some(queue) => queue.depth = depth,
            None => self.metrics.queue.push(wire::QueueDepth { rank, depth }),
        }

        match stats.encode() {
            Ok(snapshot) => self.broadcast(&from_engine::Kind::Stats(wire::Stats {
                scheduler: rank,
                snapshot,
            })),
            Err(error) => tracing::warn!(%error, "a scheduler's statistics did not encode"),
        }
    }

    /// States the process's metrics on every front: the schedulers' sums since the start; the
    /// device time the executors spent in forwards and the kernels their serving forwards
    /// launched (`Executor::counters`); and what each device maps now, against the limit of its
    /// GPU's quota: each pool of its allocators and its KV regions' planes (`DataPlane::memory`),
    /// which the daemon exports as `uniserve:memory_budget` and `uniserve:memory_mapped`.
    pub(super) fn metrics(&mut self) {
        let Some(engine) = self.engine.as_ref() else {
            return;
        };
        let (mut device_ns, mut kernel_launches) = (0, 0);
        for executor in &engine.executors {
            let (_, forward_ns, launches) = executor.counters();
            device_ns += forward_ns;
            kernel_launches += launches;
        }
        self.metrics.device_ns = device_ns;
        self.metrics.kernel_launches = kernel_launches;
        self.metrics.memory = engine.data.memory();

        let metrics = self.metrics.clone();
        self.broadcast(&from_engine::Kind::Metrics(metrics));
    }

    /// The watchdog's rules at `now` and the metrics' second: a job an executor has been inside
    /// for the stall bound is a stall, and while a job has been in flight for a second every
    /// device's context is checked once a second for a sticky error; both fail the epoch. A
    /// change of the ranks' weights that failed part of the way fails it too.
    pub(super) fn watch(&mut self, now: Instant) {
        if now >= self.next_metrics {
            self.metrics();
            self.next_metrics = now + METRICS_EVERY;
        }
        let Some(engine) = self.engine.as_ref() else {
            return;
        };

        let in_flight = self.in_flight();
        let mut failure = None;
        if let Some((rank, stall)) = self.watchdog.stalls(now, &in_flight).into_iter().next() {
            let job = engine
                .executors
                .get(rank)
                .and_then(|executor| executor.performing())
                .unwrap_or_default();
            let reason = format!(
                "a job of rank {rank} ({job}) ran {} ms, past the stall bound",
                stall.running_ms
            );
            let stalled = lifecycle::report(report::Kind::Stalled(proto::process::Stalled {
                generation: self.generation,
                batch: stall
                    .batch_id
                    .map(|batch| batch.to_string())
                    .unwrap_or_default(),
                job,
                elapsed_ms: stall.running_ms,
            }));
            failure = Some((reason, stalled));
        }

        if failure.is_none() && self.watchdog.check_due(now, &in_flight) {
            self.watchdog.checked(now);
            for executor in &engine.executors {
                let device = executor.device().map_or(0, |device| device.0);
                let device = u32::try_from(device).unwrap_or_default();
                match executor.sticky_error() {
                    Ok(None) => {}
                    Ok(Some(error)) => {
                        let reason = format!("device {device} reports a sticky error: {error}");
                        let fault = lifecycle::fault("sticky", error.to_string(), device);
                        failure = Some((reason, fault));
                        break;
                    }
                    Err(error) => {
                        tracing::warn!(device, %error, "a device's context could not be read");
                    }
                }
            }
        }
        self.watchdog.settled(&in_flight);

        if failure.is_none()
            && let Some(reason) = engine.data.poisoned()
        {
            let reason = format!("a change of the ranks' weights failed part of the way: {reason}");
            failure = Some((reason.clone(), lifecycle::fault("weights", reason, 0)));
        }
        if let Some((reason, report)) = failure {
            self.fail(reason, report);
        }
    }

    /// What each executor of the process has in flight, in rank order, as the watchdog reads it.
    pub(super) fn in_flight(&self) -> Vec<Option<(Instant, Option<BatchId>)>> {
        self.engine.as_ref().map_or_else(Vec::new, |engine| {
            engine
                .executors
                .iter()
                .map(|executor| executor.in_flight())
                .collect()
        })
    }

    /// The epoch fails for `reason`: every request this engine holds ends as migratable, every
    /// front is told why, the daemon receives `report`, and the process ends without its orderly
    /// close. A failure after the first changes nothing more.
    pub(super) fn fail(&mut self, reason: String, report: proto::process::Report) {
        if self.lifecycle == Lifecycle::Failed {
            return;
        }
        tracing::error!(%reason, "the epoch fails");
        self.lifecycle = Lifecycle::Failed;
        self.lose();
        self.broadcast(&from_engine::Kind::State(wire::State {
            state: wire::EngineState::Failed as i32,
            info: None,
            error: reason.clone(),
        }));
        self.report(report);
        self.ending = Some(Err(anyhow::anyhow!(reason)));
    }

    /// A batch computed non-finite logits for `requests`: each is reported, and a generation on
    /// trial rolls back by itself; outside a trial the daemon decides.
    pub(super) fn numerical_fault(&mut self, batch: BatchId, requests: Vec<RequestId>) {
        tracing::warn!(
            batch = batch.0,
            ?requests,
            "a batch computed non-finite logits"
        );
        for request in &requests {
            self.report(lifecycle::report(report::Kind::NonFiniteLogits(
                proto::process::NonFiniteLogits {
                    request: request.0.to_string(),
                },
            )));
        }
        if let Some(generation) = lifecycle::fault_on_trial(self.lifecycle, &mut self.update) {
            let reason = format!(
                "batch {} of generation {generation} computed non-finite logits",
                batch.0
            );
            self.roll_back(generation, true, reason);
        }
    }

    /// The scheduling policy of scheduler `scheduler` (the route instance for `None`) faulted:
    /// the previous policy serves again unless `fatal`, when none is left. Reported with the
    /// serving policy package; a generation on trial rolls back by itself, and a fatal fault
    /// outside a trial fails the epoch.
    pub(super) fn policy_rolled_back(
        &mut self,
        scheduler: Option<usize>,
        reason: String,
        fatal: bool,
    ) {
        let policy = self.serving.get(&super::policy_slot());
        let report = lifecycle::report(report::Kind::PolicyRolledBack(
            proto::process::PolicyRolledBack {
                scheduler: scheduler.map(|index| u32::try_from(index).unwrap_or(u32::MAX)),
                backend: policy
                    .map(|package| package.backend.clone())
                    .unwrap_or_default(),
                revision: policy.map_or(0, |package| package.revision),
                reason: reason.clone(),
                fatal,
            },
        ));
        self.report(report);

        if let Some(generation) = lifecycle::fault_on_trial(self.lifecycle, &mut self.update) {
            let reason = format!("the policy of generation {generation} faulted: {reason}");
            self.roll_back(generation, true, reason);
        } else if fatal {
            let reason =
                format!("the scheduling policy faulted with none to roll back to: {reason}");
            self.fail(reason.clone(), lifecycle::fault("policy", reason, 0));
        }
    }

    /// Sends one report on the process interface. A report is unasked, so one the daemon cannot
    /// read costs its record and not the engine's service.
    pub(super) fn report(&mut self, report: proto::process::Report) {
        tracing::info!(report = ?report.kind, "reporting to the daemon");
        let Some(process) = self.process.as_mut() else {
            tracing::debug!("the process interface closed; a report goes nowhere");
            return;
        };
        if let Err(error) = super::daemon::report(process, report) {
            tracing::warn!(%error, "reporting to the daemon failed");
        }
    }

    /// Sends one frame to every live front: every daemon keeps its own record of the engine.
    pub(super) fn broadcast(&mut self, kind: &from_engine::Kind) {
        for front in 0..self.fronts.len() {
            self.send(front, kind.clone(), None);
        }
    }
}
