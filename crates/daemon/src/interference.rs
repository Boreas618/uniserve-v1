//! How much an epoch switch's transfers may slow the serving epoch down
//! (`update_config.transfer_slowdown`).
//!
//! While a target prepares, the Transfer Servers of the deployment's
//! nodes copy beside the serving epoch (weights, warmed pages), on copy
//! engines and links whose byte budgets a head may halve. The leader reads
//! each scheduler's step time of the serving epoch from the statistics the
//! schedulers report ([`Observer`]) and compares it with that scheduler's
//! steps before the change:
//!
//! - **Step time** of one report is its host-observed batch time over its
//!   batches (`batch_roundtrip_us / batch_count`), the step time the trial's
//!   regression thresholds read.
//! - **Baseline.** A step's length grows with the tokens it computes, and a
//!   change of the load moves a scheduler between batch sizes, so a step is
//!   compared only with steps of its own size: each report falls in the
//!   power-of-two range of its query tokens per batch, and a scheduler's
//!   baseline of a range is the batch time over the batches of its reports
//!   in that range during the `update_config.baseline` before the preparation
//!   began, once they hold `update_config.baseline_batches`. A report of a
//!   range without a baseline decides nothing.
//! - **Decision.** As each report arrives, the reports that arrived since the
//!   previous decision are read: a scheduler whose step takes more than
//!   `1 + transfer_slowdown` times its baseline makes the leader halve the
//!   budgets of the links the deployment's transfers use, on every node,
//!   again with each report while it lasts (the Transfer Servers keep their
//!   own floor); once every scheduler's steps are back within the ratio, the
//!   leader restores them, which the Transfer Servers do gradually, and it
//!   restores them when the preparation ends.
//!
//! The control acts through the deployment's daemons ([`Throttle`]), which
//! throttle every node as the leader knows it at each decision: the copies of
//! an epoch switch run on the Transfer Servers of the target's nodes, and a
//! node that holds no claim of the deployment has none to throttle.

use std::collections::BTreeMap;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Instant;

use proto::admin::{StepTime, Throttled};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::observe::{Observer, Sample};
use crate::options::UpdateConfig;

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
#[derive(Debug)]
pub(crate) struct Baseline {
    /// Batch microseconds and batches, by scheduler and range.
    steps: BTreeMap<(u32, u32), (u64, u64)>,
    /// The fewest batches of a range that make its baseline.
    batches: NonZeroU32,
}

impl Baseline {
    /// `epoch`'s baseline from its reports in `[from, until)`, of the ranges whose reports hold
    /// `batches` batches.
    pub(crate) fn of(
        observer: &Observer,
        epoch: u64,
        from: Instant,
        until: Instant,
        batches: NonZeroU32,
    ) -> Self {
        let mut baseline = Self {
            steps: BTreeMap::new(),
            batches,
        };
        observer.samples(epoch, from, until, |sample| {
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
        (batches >= u64::from(self.batches.get())).then(|| batch_us as f64 / batches as f64)
    }
}

/// What the reports read at one decision ask of the transfers' budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Decision {
    Halve,
    Restore,
}

/// The control of one epoch change's preparation: the serving epoch's
/// baseline and whether the budgets stand halved.
pub(crate) struct Throttling {
    epoch: u64,
    /// The slowdown tolerated: a step may take `1 + slowdown` times its
    /// baseline.
    slowdown: f64,
    baseline: Baseline,
    /// Where the previous decision stopped reading.
    read: Instant,
    halved: bool,
    /// The largest step-to-baseline ratio of the latest decision whose
    /// reports had a baseline.
    ratio: Option<f64>,
}

impl Throttling {
    /// The control of `epoch`, the serving epoch's, whose target began
    /// preparing at `preparing`, tolerating steps `1 + slowdown` times their
    /// baseline, which `options` state the span and batches of
    /// (`update_config`).
    pub(crate) fn new(
        observer: &Observer,
        epoch: u64,
        slowdown: f64,
        options: &UpdateConfig,
        preparing: Instant,
    ) -> Self {
        let from = preparing.checked_sub(options.baseline).unwrap_or(preparing);
        Self {
            epoch,
            slowdown,
            baseline: Baseline::of(observer, epoch, from, preparing, options.baseline_batches),
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
        observer.samples(self.epoch, self.read, now, |sample| {
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

/// Who passes a decision on to the nodes' Transfer Servers.
pub(crate) trait Throttle: Send + Sync + 'static {
    /// Halves, or restores when not `halve`, the byte budgets of the links the
    /// deployment's transfers use on every node, and returns the hosts that
    /// took it. Blocks until every node's daemon replied.
    ///
    /// # Errors
    ///
    /// The reason a node's daemon did not take it.
    fn throttle(&self, halve: bool) -> Result<Vec<String>, String>;
}

/// Passes a decision on off the runtime's threads.
async fn pass_on(throttle: &Arc<dyn Throttle>, halve: bool) -> Result<Vec<String>, String> {
    let throttle = Arc::clone(throttle);
    tokio::task::spawn_blocking(move || throttle.throttle(halve))
        .await
        .map_err(|error| error.to_string())?
}

/// Runs `throttling` as each report of its observers arrives until `done`,
/// sending its decisions through `throttle`, and restores the budgets at the
/// end when they stand halved; returns the throttles that were passed on. A
/// throttle that cannot be passed on is logged: the preparation goes on, at
/// the budgets the Transfer Servers keep, and the change's report counts only
/// the throttles that were taken. After each decision `steps` holds the serving
/// epoch's steps against their baseline as read.
pub(crate) async fn control(
    throttle: Arc<dyn Throttle>,
    mut observed: Vec<(Arc<Observer>, Throttling)>,
    done: CancellationToken,
    steps: Arc<std::sync::Mutex<StepTime>>,
) -> Throttled {
    let mut throttled = Throttled::default();
    service::lock(&steps).slowdown = observed
        .iter()
        .map(|(_, throttling)| throttling.slowdown)
        .fold(0.0, f64::max);
    let mut records: Vec<_> = observed
        .iter()
        .map(|(observer, _)| observer.records())
        .collect();
    // The first decision reads the reports that arrived since the preparation began, before the
    // control watched for them; each later one, the reports since the decision before it.
    let mut first = true;
    loop {
        if !std::mem::take(&mut first) {
            tokio::select! {
                () = done.cancelled() => break,
                () = crate::observe::next_record(&mut records) => {}
            }
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
                        "the serving epoch's steps against their baseline changed the \
                         transfers' budgets"
                    );
                }
                Err(error) => warn!(?decision, %error, "the transfers' budgets did not change"),
            }
        }
        let mut read = service::lock(&steps);
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
    let mut read = service::lock(&steps);
    read.halved = false;
    read.throttled = throttled;
    drop(read);
    throttled
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use proto::report::SchedulerStats;
    use tokio::sync::mpsc;

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

    /// The slowdown the controls tolerate.
    const SLOWDOWN: f64 = 0.2;

    /// A step is compared with its scheduler's steps of its own batch-size
    /// range: larger batches that take longer are not a slowdown, and a
    /// range without a baseline decides nothing.
    #[test]
    fn steps_are_compared_within_their_batch_size_range() {
        let observer = Observer::new(Duration::from_secs(600));
        for _ in 0..4 {
            observer.record(1, 0, &report(10_000, 10, 32));
            observer.record(1, 0, &report(30_000, 10, 256));
        }
        let mut throttling = Throttling::new(
            &observer,
            1,
            SLOWDOWN,
            &UpdateConfig::default(),
            Instant::now(),
        );

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

    /// A throttle that sends what it was told.
    struct Recorded(mpsc::UnboundedSender<bool>);

    impl Throttle for Recorded {
        fn throttle(&self, halve: bool) -> Result<Vec<String>, String> {
            self.0.send(halve).map_err(|error| error.to_string())?;
            Ok(vec!["rank-0".to_owned()])
        }
    }

    /// The serving epoch's slowed scheduler makes the leader halve the budgets of the
    /// deployment's transfers with each slowed report, its recovery restores them once, steps
    /// within the tolerated slowdown change nothing, and a preparation that ends halved restores
    /// them: the throttles the control passed on are exactly that sequence.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_slowed_scheduler_halves_the_transfers_budgets_and_its_recovery_restores_them() {
        let (told, mut throttles) = mpsc::unbounded_channel();
        let observer = Arc::new(Observer::new(Duration::from_secs(600)));

        // A minute of steps of 10 ms at 32 tokens per batch before the change.
        for _ in 0..60 {
            observer.record(1, 0, &report(10_000, 25, 32));
        }
        let throttling = Throttling::new(
            &observer,
            1,
            SLOWDOWN,
            &UpdateConfig::default(),
            Instant::now(),
        );
        let done = CancellationToken::new();
        let steps = Arc::new(std::sync::Mutex::new(StepTime::default()));
        let control = tokio::spawn(control(
            Arc::new(Recorded(told)),
            vec![(Arc::clone(&observer), throttling)],
            done.clone(),
            Arc::clone(&steps),
        ));
        // Each report is decided before the next is recorded: the throttle of the report before
        // is awaited, and a report that changes nothing is followed by one that does.
        let mut next = async || throttles.recv().await.expect("the control throttles");

        // Steps within the tolerated slowdown change nothing, and the 50% slower ones after
        // them halve the budgets, which the change in progress reports.
        observer.record(1, 0, &report(11_000, 25, 32));
        observer.record(1, 0, &report(15_000, 25, 32));
        assert!(next().await);
        observer.record(1, 0, &report(15_000, 25, 32));
        assert!(next().await);
        let read = *service::lock(&steps);
        assert!(
            read.ratio.is_some_and(|ratio| (ratio - 1.5).abs() < 1e-9),
            "{read:?}"
        );
        assert!(read.halved, "{read:?}");
        assert!((read.slowdown - 0.2).abs() < 1e-12, "{read:?}");

        // Steps back within it restore them, once; the next slowed step halves them again.
        observer.record(1, 0, &report(10_200, 25, 32));
        assert!(!next().await);
        observer.record(1, 0, &report(10_100, 25, 32));
        observer.record(1, 0, &report(20_000, 25, 32));
        assert!(next().await);

        // A preparation that ends with the budgets halved restores them.
        done.cancel();
        let throttled = control.await.expect("the control ended");
        assert!(!next().await);
        assert_eq!(
            throttled,
            Throttled {
                halvings: 3,
                restorations: 2
            }
        );
        assert!(
            throttles.try_recv().is_err(),
            "no other throttle was passed on"
        );
    }
}
