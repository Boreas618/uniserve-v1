//! How much an epoch switch's transfers may slow the serving epoch down
//! (`update_config.transfer_slowdown`, `specs/online-operations.md` §4.5.3).
//!
//! While a target prepares, the Transfer Servers of the deployment's
//! nodes copy beside the serving epoch (weights, warmed pages), on copy
//! engines and links whose byte budgets a head may halve. The leader reads
//! each scheduler's step time of the serving epoch from the statistics the
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
//! - **Decision.** Once per interval, the reports that arrived since the
//!   previous decision are read: a scheduler whose step takes more than
//!   `1 + transfer_slowdown` times its baseline makes the leader halve the
//!   budgets of the links the deployment's transfers use, on every node,
//!   again each interval while it lasts (the Transfer Servers keep a 1/64
//!   floor); once every scheduler's steps are back within the ratio, the
//!   leader restores them, which the Transfer Servers do gradually, and it
//!   restores them when the preparation ends.
//!
//! The control acts through the deployment's daemons ([`Throttle`]), which
//! throttle every node as the leader knows it at each decision: the copies of
//! an epoch switch run on the Transfer Servers of the target's nodes, and a
//! node that holds no claim of the deployment has none to throttle.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::observe::{Observer, Sample};

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
    /// The largest step-to-baseline ratio of the latest interval whose
    /// reports had a baseline.
    ratio: Option<f64>,
}

impl Throttling {
    /// The control of `world`, the serving epoch's, whose target began
    /// preparing at `preparing`, tolerating steps `1 + slowdown` times their
    /// baseline.
    pub(crate) fn new(observer: &Observer, world: u64, slowdown: f64, preparing: Instant) -> Self {
        let from = preparing.checked_sub(BASELINE).unwrap_or(preparing);
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

/// Who passes a decision on to the nodes' Transfer Servers.
pub(crate) trait Throttle: Send + Sync + 'static {
    /// Halves, or restores when not `halve`, the byte budgets of the links the
    /// deployment's transfers use on every node, and answers the hosts that
    /// took it. Blocks until every node's daemon answered.
    fn throttle(&self, halve: bool) -> Result<Vec<String>, String>;
}

/// Passes a decision on off the runtime's threads.
async fn pass_on(throttle: &Arc<dyn Throttle>, halve: bool) -> Result<Vec<String>, String> {
    let throttle = Arc::clone(throttle);
    tokio::task::spawn_blocking(move || throttle.throttle(halve))
        .await
        .map_err(|error| error.to_string())?
}

/// Runs `throttling` every `interval` until `done`, sending its decisions
/// through `throttle`, and restores the budgets at the end when they stand
/// halved; returns the throttles that were passed on. A throttle that cannot
/// be passed on is logged: the preparation goes on, at the budgets the
/// Transfer Servers keep. After each interval `steps` holds the serving
/// epoch's steps against their baseline as read.
pub(crate) async fn control(
    throttle: Arc<dyn Throttle>,
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
            match pass_on(&throttle, halve).await {
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
        match pass_on(&throttle, false).await {
            Ok(_) => throttled.restorations += 1,
            Err(error) => {
                warn!(%error, "the transfers' budgets were not restored after the preparation");
            }
        }
    }
    let mut read = crate::leader::locked(&steps);
    read.halved = false;
    read.throttled = throttled;
    drop(read);
    throttled
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::{Receiver, Sender, channel};

    use proto::report::SchedulerStats;

    use super::*;

    /// One scheduler's report of `batches` batches of `tokens_per_batch`
    /// query tokens each taking `step_us`.
    fn report(step_us: u64, batches: u64, tokens_per_batch: u64) -> SchedulerStats {
        SchedulerStats {
            batch_roundtrip_us: step_us * batches,
            batch_count: batches,
            batch_tokens: tokens_per_batch * batches,
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
        let mut throttling = Throttling::new(&observer, 1, 0.2, Instant::now());

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

    /// A throttle that records what it was told.
    struct Recorded(std::sync::Mutex<Sender<bool>>);

    impl Throttle for Recorded {
        fn throttle(&self, halve: bool) -> Result<Vec<String>, String> {
            crate::leader::locked(&self.0)
                .send(halve)
                .map_err(|error| error.to_string())?;
            Ok(vec!["rank-0".to_owned()])
        }
    }

    fn recorded() -> (Arc<dyn Throttle>, Receiver<bool>) {
        let (decisions, told) = channel();
        (Arc::new(Recorded(std::sync::Mutex::new(decisions))), told)
    }

    /// The serving world's slowed scheduler makes the leader halve the
    /// budgets of the deployment's transfers through the engine, once each
    /// interval while it lasts; its recovery makes it restore them once.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_slowed_scheduler_halves_the_transfers_budgets_and_its_recovery_restores_them() {
        let (throttle, throttles) = recorded();
        let observer = Arc::new(Observer::default());

        // A minute of steps of 10 ms at 32 tokens per batch before the
        // change.
        for _ in 0..60 {
            observer.record(1, 0, &report(10_000, 25, 32));
        }
        let throttling = Throttling::new(&observer, 1, 0.2, Instant::now());
        let interval = Duration::from_millis(100);
        let done = CancellationToken::new();
        let steps = Arc::new(std::sync::Mutex::new(StepTime::default()));
        let control = tokio::spawn(control(
            throttle,
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
