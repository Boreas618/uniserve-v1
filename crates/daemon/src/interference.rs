//! How much a world change's transfers may slow the serving world down
//! (`update_config.transfer_slowdown`, `specs/online-operations.md` §4.5.3).
//!
//! While a successor prepares, the Transfer Servers of the deployment's
//! nodes copy beside the serving world (weights, warmed pages), on copy
//! engines and links whose byte budgets a head may halve. The leader reads
//! each scheduler's step time of the serving world from the statistics the
//! schedulers report once a second ([`Observer`]) and compares it with that
//! scheduler's steps before the change:
//!
//! - **Step time** of one report is its host-observed batch time over its
//!   batches (`batch_roundtrip_us / batch_count`), the step time the trial's
//!   regression thresholds read.
//! - **Baseline.** A step's length grows with the tokens it computes, and a
//!   change of the load moves a scheduler between batch sizes, so a step is
//!   compared only with steps of its own size: each report falls in the
//!   power-of-two range of its query tokens per batch, and a scheduler's
//!   baseline of a range is the batch time over the batches of its reports
//!   in that range during the [`BASELINE`] before the preparation began,
//!   once they hold [`MIN_BATCHES`]. A report of a range without a baseline
//!   decides nothing.
//! - **Eager forwards.** A serving world that runs its forwards eagerly
//!   while its successor prepares (`update_config.eager_prepare`) steps
//!   slower than its graphs did whatever the transfers do, so its baseline
//!   holds only reports that cover eager steps alone: those that arrived
//!   more than an [`INTERVAL`] after the switch and before the preparation
//!   began. A switch that precedes the preparation by less leaves no
//!   baseline, and the transfers keep their budgets.
//! - **Decision.** Once per interval, the reports that arrived since the
//!   previous decision are read: a scheduler whose step takes more than
//!   `1 + transfer_slowdown` times its baseline makes the leader halve the
//!   budgets of the links the deployment's transfers use, on every node,
//!   again each interval while it lasts (the Transfer Servers keep a 1/64
//!   floor); once every scheduler's steps are back within the ratio, the
//!   leader restores them, which the Transfer Servers do gradually, and it
//!   restores them when the preparation ends.
//!
//! The control acts through the serving engine, whose claims cover every
//! node of the deployment ([`EngineClient::throttle_transfers`]).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::engine::client::EngineClient;
use crate::worlds::{Observer, Sample};

/// The span before a world change whose steps make the schedulers'
/// baselines: sixty of their once-a-second reports.
pub(crate) const BASELINE: Duration = Duration::from_secs(60);

/// How often the control reads the schedulers' reports, which they make
/// once a second.
pub(crate) const INTERVAL: Duration = Duration::from_secs(1);

/// The fewest batches of one batch-size range that make its baseline, so
/// that a range the load visited for a few steps does not set it.
const MIN_BATCHES: u64 = 20;

/// The power-of-two range of `tokens` query tokens per batch: 0 for one
/// token, `k` for `(2^(k-1), 2^k]`.
fn range(tokens_per_batch: u64) -> u32 {
    tokens_per_batch.max(1).next_power_of_two().trailing_zeros()
}

/// One report's batch time per batch and batch-size range, `None` for a
/// report of no batch.
fn step(sample: &Sample) -> Option<(f64, u32)> {
    (sample.batches > 0).then(|| {
        (
            sample.batch_us as f64 / sample.batches as f64,
            range(sample.tokens / sample.batches),
        )
    })
}

/// Each scheduler's step time before a change, per batch-size range.
#[derive(Debug, Default)]
pub(crate) struct Baseline {
    /// Batch microseconds and batches, by scheduler and range.
    steps: BTreeMap<(u32, u32), (u64, u64)>,
}

impl Baseline {
    /// `world`'s baseline from its reports in `[from, until)`.
    pub(crate) fn of(observer: &Observer, world: u64, from: Instant, until: Instant) -> Self {
        let mut baseline = Self::default();
        observer.samples(world, from, until, |sample| {
            if let Some((_, range)) = step(sample) {
                let entry = baseline.steps.entry((sample.scheduler, range)).or_default();
                entry.0 += sample.batch_us;
                entry.1 += sample.batches;
            }
        });
        baseline
    }

    /// Scheduler `scheduler`'s baseline step time in `range`, once its
    /// batches there make one.
    fn step_us(&self, scheduler: u32, range: u32) -> Option<f64> {
        let &(batch_us, batches) = self.steps.get(&(scheduler, range))?;
        (batches >= MIN_BATCHES).then(|| batch_us as f64 / batches as f64)
    }
}

/// What one interval's reports ask of the transfers' budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Decision {
    Halve,
    Restore,
}

/// The control of one world change's preparation: the serving world's
/// baseline and whether the budgets stand halved.
pub(crate) struct Throttling {
    world: u64,
    /// The slowdown tolerated: a step may take `1 + slowdown` times its
    /// baseline.
    slowdown: f64,
    baseline: Baseline,
    /// Where the previous decision stopped reading.
    read: Instant,
    halved: bool,
}

impl Throttling {
    /// The control of `world`, the serving world, whose successor began
    /// preparing at `preparing`, tolerating steps `1 + slowdown` times their
    /// baseline. `eager_since` is when the serving world switched to eager
    /// forwards for the preparation, if it did: its steps before the switch
    /// ran other kernels and make no baseline.
    pub(crate) fn new(
        observer: &Observer,
        world: u64,
        slowdown: f64,
        preparing: Instant,
        eager_since: Option<Instant>,
    ) -> Self {
        // A report covers the interval before it arrives, so the first one
        // of eager steps alone arrives an interval after the switch.
        let from = match eager_since {
            Some(since) => since + INTERVAL,
            None => preparing.checked_sub(BASELINE).unwrap_or(preparing),
        };
        Self {
            world,
            slowdown,
            baseline: Baseline::of(observer, world, from, preparing),
            read: preparing,
            halved: false,
        }
    }

    /// The decision of the reports that arrived since the previous one,
    /// with the largest step-to-baseline ratio among them: a halving while a
    /// scheduler's step is past the tolerated slowdown, a restoration once
    /// none is after a halving, and nothing otherwise or without a report
    /// that has a baseline.
    pub(crate) fn decide(&mut self, observer: &Observer) -> Option<(Decision, f64)> {
        let now = Instant::now();
        let mut slowest: Option<f64> = None;
        observer.samples(self.world, self.read, now, |sample| {
            let Some((step_us, range)) = step(sample) else {
                return;
            };
            let Some(baseline) = self.baseline.step_us(sample.scheduler, range) else {
                return;
            };
            if baseline > 0.0 {
                let ratio = step_us / baseline;
                slowest = Some(slowest.map_or(ratio, |slowest| slowest.max(ratio)));
            }
        });
        self.read = now;
        let ratio = slowest?;
        if ratio > 1.0 + self.slowdown {
            self.halved = true;
            Some((Decision::Halve, ratio))
        } else if self.halved {
            self.halved = false;
            Some((Decision::Restore, ratio))
        } else {
            None
        }
    }

    /// Whether the budgets stand halved, which the end of the preparation
    /// restores.
    pub(crate) fn halved(&self) -> bool {
        self.halved
    }
}

/// Runs `throttling` every `interval` until `done`, sending its decisions
/// through `running`, the serving engine, and restores the budgets at the
/// end when they stand halved. A throttle the engine cannot pass on is
/// logged: the preparation goes on, at the budgets the Transfer Servers
/// keep.
pub(crate) async fn control(
    running: Arc<EngineClient>,
    observer: Arc<Observer>,
    mut throttling: Throttling,
    interval: Duration,
    done: CancellationToken,
) {
    let mut ticks = tokio::time::interval(interval);
    ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // The first tick completes at once.
    ticks.tick().await;
    loop {
        tokio::select! {
            () = done.cancelled() => break,
            _ = ticks.tick() => {}
        }
        if let Some((decision, ratio)) = throttling.decide(&observer) {
            let halve = decision == Decision::Halve;
            match running.throttle_transfers(halve).await {
                Ok(hosts) => info!(
                    ?decision,
                    ratio,
                    ?hosts,
                    "the serving world's steps against their baseline changed the transfers' \
                     budgets"
                ),
                Err(error) => warn!(?decision, %error, "the transfers' budgets did not change"),
            }
        }
    }
    if throttling.halved()
        && let Err(error) = running.throttle_transfers(false).await
    {
        warn!(%error, "the transfers' budgets were not restored after the preparation");
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use engine::{EngineConfig, SchedulerHandle, SimEngine, SimExecutor};
    use proto::report::{ForwardStats, SchedulerStats};

    use super::*;

    /// One scheduler's report of `batches` batches of `tokens_per_batch`
    /// query tokens each taking `step_us`.
    fn report(step_us: u64, batches: u64, tokens_per_batch: u64) -> SchedulerStats {
        SchedulerStats {
            batch_roundtrip_us: step_us * batches,
            batch_count: batches,
            worker_forward_stats: Some(ForwardStats {
                mode_tokens: BTreeMap::from([(
                    "graph_replay".to_owned(),
                    tokens_per_batch * batches,
                )]),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    /// A step is compared with its scheduler's steps of its own batch-size
    /// range: larger batches that take longer are not a slowdown, and a
    /// range without a baseline decides nothing.
    #[test]
    fn steps_are_compared_within_their_batch_size_range() {
        let observer = Observer::default();
        for _ in 0..4 {
            observer.record(1, 0, &report(10_000, 10, 32));
            observer.record(1, 0, &report(30_000, 10, 256));
        }
        let mut throttling = Throttling::new(&observer, 1, 0.2, Instant::now(), None);

        observer.record(1, 0, &report(30_000, 10, 250));
        assert_eq!(throttling.decide(&observer), None);
        observer.record(1, 0, &report(90_000, 10, 2048));
        assert_eq!(throttling.decide(&observer), None);
        observer.record(1, 0, &report(13_000, 10, 30));
        assert_eq!(
            throttling.decide(&observer).map(|(decision, _)| decision),
            Some(Decision::Halve)
        );
        // Another scheduler's baseline is its own.
        observer.record(1, 1, &report(13_000, 10, 30));
        observer.record(1, 0, &report(10_500, 10, 30));
        assert_eq!(
            throttling.decide(&observer).map(|(decision, _)| decision),
            Some(Decision::Restore)
        );
    }

    /// A serving world that switched to eager forwards for the preparation
    /// is compared with its eager steps alone: its slower eager steps are no
    /// slowdown, and without eager steps before the preparation nothing is
    /// decided.
    #[test]
    fn eager_steps_are_compared_with_eager_steps() {
        let observer = Observer::default();
        for _ in 0..4 {
            observer.record(1, 0, &report(10_000, 10, 32));
        }
        // Steps of 10 ms by graphs, then a switch an interval back, so the
        // reports from now on cover eager steps (14 ms) alone.
        let switched = Instant::now()
            .checked_sub(INTERVAL)
            .expect("the clock runs past an interval");
        for _ in 0..4 {
            observer.record(1, 0, &report(14_000, 10, 32));
        }
        let mut throttling = Throttling::new(&observer, 1, 0.2, Instant::now(), Some(switched));
        observer.record(1, 0, &report(15_000, 10, 32));
        assert_eq!(throttling.decide(&observer), None);
        observer.record(1, 0, &report(20_000, 10, 32));
        assert_eq!(
            throttling.decide(&observer).map(|(decision, _)| decision),
            Some(Decision::Halve)
        );

        // A switch just before the preparation leaves no baseline.
        let mut throttling =
            Throttling::new(&observer, 1, 0.2, Instant::now(), Some(Instant::now()));
        observer.record(1, 0, &report(20_000, 10, 32));
        assert_eq!(throttling.decide(&observer), None);
    }

    /// The serving world's slowed scheduler makes the leader halve the
    /// budgets of the deployment's transfers through the engine, once each
    /// interval while it lasts; its recovery makes it restore them once.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_slowed_scheduler_halves_the_transfers_budgets_and_its_recovery_restores_them() {
        let mut executor = SimExecutor::new(SimEngine::new());
        let throttles = executor.observe_throttles();
        let core = SchedulerHandle::with_executor(
            EngineConfig::sim(crate::test_support::SERVED_MODEL),
            Box::new(executor),
        )
        .expect("start a sim scheduler");
        let running = Arc::new(crate::test_support::sim_client(vec![core]));
        let observer = Arc::new(Observer::default());

        // A minute of steps of 10 ms at 32 tokens per batch before the
        // change.
        for _ in 0..60 {
            observer.record(1, 0, &report(10_000, 25, 32));
        }
        let throttling = Throttling::new(&observer, 1, 0.2, Instant::now(), None);
        let interval = Duration::from_millis(100);
        let done = CancellationToken::new();
        let control = tokio::spawn(control(
            Arc::clone(&running),
            Arc::clone(&observer),
            throttling,
            interval,
            done.clone(),
        ));
        let next = |wait: Duration| throttles.recv_timeout(wait).ok();

        // Steps within the tolerated slowdown change nothing.
        observer.record(1, 0, &report(11_000, 25, 32));
        assert_eq!(next(interval * 5), None);

        // Steps 50% longer halve the budgets each interval they last.
        observer.record(1, 0, &report(15_000, 25, 32));
        assert_eq!(next(Duration::from_secs(10)), Some(true));
        observer.record(1, 0, &report(15_000, 25, 32));
        assert_eq!(next(Duration::from_secs(10)), Some(true));

        // Steps back within it restore them, once.
        observer.record(1, 0, &report(10_200, 25, 32));
        assert_eq!(next(Duration::from_secs(10)), Some(false));
        observer.record(1, 0, &report(10_100, 25, 32));
        assert_eq!(next(interval * 5), None);

        // A preparation that ends with the budgets halved restores them.
        observer.record(1, 0, &report(20_000, 25, 32));
        assert_eq!(next(Duration::from_secs(10)), Some(true));
        done.cancel();
        control.await.expect("the control ended");
        assert_eq!(next(Duration::from_secs(10)), Some(false));
    }
}
