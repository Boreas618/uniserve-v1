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
//!   began. The successor starts once each scheduler reported
//!   [`EAGER_REPORTS`] of them with steps ([`eager_baseline`]), or ran no
//!   step for [`IDLE`]: an idle scheduler has no steps to protect, and its
//!   reports during the preparation decide nothing.
//! - **Decision.** Once per interval, the reports that arrived since the
//!   previous decision are read: a scheduler whose step takes more than
//!   `1 + transfer_slowdown` times its baseline makes the leader halve the
//!   budgets of the links the deployment's transfers use, on every node,
//!   again each interval while it lasts (the Transfer Servers keep a 1/64
//!   floor); once every scheduler's steps are back within the ratio, the
//!   leader restores them, which the Transfer Servers do gradually, and it
//!   restores them when the preparation ends.
//!
//! The control acts through the serving engine, which throttles every node
//! of the deployment as the leader knows it at each decision
//! ([`EngineClient::throttle_transfers`]): the copies of a world change run
//! on the Transfer Servers of the successor's nodes, a node that joined the
//! deployment after the serving engine started among them, and a node that
//! holds no claim of the deployment has none to throttle.

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

/// The reports of eager steps alone, with steps, each scheduler of a
/// serving world that switched to eager forwards makes before its successor
/// starts: its baseline.
const EAGER_REPORTS: usize = 2;

/// How long a scheduler may run no step before the wait for its eager
/// baseline ends without it.
pub(crate) const IDLE: Duration = Duration::from_millis(2500);

/// How often the wait for an eager baseline reads the reports, which
/// arrive once a second.
const POLL: Duration = Duration::from_millis(100);

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

/// Waits until every one of `world`'s `schedulers` schedulers either
/// reported [`EAGER_REPORTS`] reports of eager steps alone with steps since
/// the switch to eager forwards at `eager_since`, or ran no step for
/// [`IDLE`], and returns those that ran none, which the preparation starts
/// without a baseline of. The wait ends within [`IDLE`] of each scheduler's
/// latest step, so a world that stops stepping does not hold it.
pub(crate) async fn eager_baseline(
    observer: &Observer,
    world: u64,
    schedulers: u32,
    eager_since: Instant,
) -> Vec<u32> {
    loop {
        if let Some(idle) = eager_waited(observer, world, schedulers, eager_since, Instant::now()) {
            return idle;
        }
        tokio::time::sleep(POLL).await;
    }
}

/// The schedulers [`eager_baseline`] stops waiting for without a baseline
/// at `now`, once every other one has its baseline; `None` while one still
/// steps without it.
fn eager_waited(
    observer: &Observer,
    world: u64,
    schedulers: u32,
    eager_since: Instant,
    now: Instant,
) -> Option<Vec<u32>> {
    let count = schedulers as usize;
    // Per scheduler: its reports of eager steps alone with steps, and the
    // arrival of its latest report with steps (the switch before any).
    let mut eager = vec![0_usize; count];
    let mut stepped = vec![eager_since; count];
    observer.samples(world, eager_since, now, |sample| {
        let index = sample.scheduler as usize;
        if index >= count || sample.batches == 0 {
            return;
        }
        stepped[index] = stepped[index].max(sample.at);
        if sample.at >= eager_since + INTERVAL {
            eager[index] += 1;
        }
    });

    let mut idle = Vec::new();
    for index in 0..count {
        if eager[index] >= EAGER_REPORTS {
            continue;
        }
        if now.saturating_duration_since(stepped[index]) < IDLE {
            return None;
        }
        idle.push(index as u32);
    }
    Some(idle)
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
    /// The largest step-to-baseline ratio of the latest interval whose
    /// reports had a baseline.
    ratio: Option<f64>,
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
            ratio: None,
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
        self.ratio = Some(ratio);
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

/// The throttles one preparation sent, which its change reports.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub(crate) struct Throttled {
    pub(crate) halvings: u32,
    pub(crate) restorations: u32,
}

/// The serving world's steps against their baseline while a successor
/// prepares, as the control last read them, which `GET /admin/config`
/// reports of the change in progress.
#[derive(Debug, Default, Clone, Copy, PartialEq, serde::Serialize)]
pub(crate) struct StepTime {
    /// The largest ratio of a scheduler's step time to its baseline of the
    /// same batch size among the reports of the latest interval that had a
    /// baseline; `None` before any had.
    pub(crate) ratio: Option<f64>,
    /// The slowdown tolerated (`update_config.transfer_slowdown`): a step
    /// may take `1 + slowdown` times its baseline.
    pub(crate) slowdown: f64,
    /// Whether the transfers' link budgets stand halved.
    pub(crate) halved: bool,
    #[serde(flatten)]
    pub(crate) throttled: Throttled,
}

/// Runs `throttling` every `interval` until `done`, sending its decisions
/// through `running`, the serving engine, and restores the budgets at the
/// end when they stand halved; returns the throttles the engine passed on.
/// A throttle the engine cannot pass on is logged: the preparation goes on,
/// at the budgets the Transfer Servers keep. After each interval `steps`
/// holds the serving world's steps against their baseline as read.
pub(crate) async fn control(
    running: Arc<EngineClient>,
    mut observed: Vec<(Arc<Observer>, Throttling)>,
    interval: Duration,
    done: CancellationToken,
    steps: Arc<std::sync::Mutex<StepTime>>,
) -> Throttled {
    let mut throttled = Throttled::default();
    crate::leader::locked(&steps).slowdown = observed
        .iter()
        .map(|(_, throttling)| throttling.slowdown)
        .fold(0.0, f64::max);
    let mut ticks = tokio::time::interval(interval);
    ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // The first tick completes at once.
    ticks.tick().await;
    loop {
        tokio::select! {
            () = done.cancelled() => break,
            _ = ticks.tick() => {}
        }
        let decisions: Vec<_> = observed
            .iter_mut()
            .filter_map(|(observer, throttling)| throttling.decide(observer))
            .collect();
        let decision = if decisions
            .iter()
            .any(|(decision, _)| *decision == Decision::Halve)
        {
            Some(Decision::Halve)
        } else if decisions
            .iter()
            .any(|(decision, _)| *decision == Decision::Restore)
            && observed.iter().all(|(_, throttling)| !throttling.halved())
        {
            Some(Decision::Restore)
        } else {
            None
        };
        if let Some(decision) = decision {
            let ratio = decisions
                .iter()
                .map(|(_, ratio)| *ratio)
                .fold(0.0, f64::max);
            let halve = decision == Decision::Halve;
            match running.throttle_transfers(halve).await {
                Ok(hosts) => {
                    if halve {
                        throttled.halvings += 1;
                    } else {
                        throttled.restorations += 1;
                    }
                    info!(
                        ?decision,
                        ratio,
                        ?hosts,
                        "the serving world's steps against their baseline changed the \
                         transfers' budgets"
                    );
                }
                Err(error) => warn!(?decision, %error, "the transfers' budgets did not change"),
            }
        }
        let mut read = crate::leader::locked(&steps);
        if let Some(ratio) = observed
            .iter()
            .filter_map(|(_, throttling)| throttling.ratio)
            .reduce(f64::max)
        {
            read.ratio = Some(ratio);
        }
        read.halved = observed.iter().any(|(_, throttling)| throttling.halved());
        read.throttled = throttled;
    }
    if observed.iter().any(|(_, throttling)| throttling.halved()) {
        match running.throttle_transfers(false).await {
            Ok(_) => throttled.restorations += 1,
            Err(error) => {
                warn!(%error, "the transfers' budgets were not restored after the preparation");
            }
        }
    }
    throttled
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

    /// The successor waits for two reports of eager steps alone from each
    /// stepping scheduler, and for no scheduler that runs no step for
    /// [`IDLE`].
    #[test]
    fn the_eager_baseline_waits_for_two_reports_of_steps_or_an_idle_scheduler() {
        let observer = Observer::default();
        // A switch an interval back, so the reports from now on cover eager
        // steps alone.
        let switched = Instant::now()
            .checked_sub(INTERVAL)
            .expect("the clock runs past an interval");

        observer.record(1, 0, &report(14_000, 10, 32));
        observer.record(1, 1, &report(0, 0, 0));
        assert_eq!(
            eager_waited(&observer, 1, 2, switched, Instant::now()),
            None
        );

        observer.record(1, 0, &report(14_000, 10, 32));
        observer.record(1, 1, &report(0, 0, 0));
        assert_eq!(
            eager_waited(&observer, 1, 2, switched, Instant::now()),
            None
        );

        // Scheduler 1 ran no step since the switch: once that lasts the idle
        // span, the wait ends without its baseline.
        let later = switched + IDLE;
        assert_eq!(
            eager_waited(&observer, 1, 2, switched, later),
            Some(vec![1])
        );

        // A world that runs no step at all starts without any.
        let idle = Observer::default();
        assert_eq!(eager_waited(&idle, 1, 2, switched, Instant::now()), None);
        assert_eq!(eager_waited(&idle, 1, 2, switched, later), Some(vec![0, 1]));
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
        let steps = Arc::new(std::sync::Mutex::new(StepTime::default()));
        let control = tokio::spawn(control(
            Arc::clone(&running),
            vec![(Arc::clone(&observer), throttling)],
            interval,
            done.clone(),
            Arc::clone(&steps),
        ));
        let next = |wait: Duration| throttles.recv_timeout(wait).ok();

        // Steps within the tolerated slowdown change nothing.
        observer.record(1, 0, &report(11_000, 25, 32));
        assert_eq!(next(interval * 5), None);

        // Steps 50% longer halve the budgets each interval they last, which
        // the change in progress reports.
        observer.record(1, 0, &report(15_000, 25, 32));
        assert_eq!(next(Duration::from_secs(10)), Some(true));
        tokio::time::sleep(interval * 2).await;
        let read = *crate::leader::locked(&steps);
        assert!(
            read.ratio.is_some_and(|ratio| (ratio - 1.5).abs() < 1e-9),
            "{read:?}"
        );
        assert!(read.halved, "{read:?}");
        assert!((read.slowdown - 0.2).abs() < 1e-12, "{read:?}");
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
        let throttled = control.await.expect("the control ended");
        assert_eq!(next(Duration::from_secs(10)), Some(false));
        assert_eq!(
            throttled,
            Throttled {
                halvings: 3,
                restorations: 2
            }
        );
    }
}
