//! The trial an update stands while it is on probation: the thresholds of
//! `update_config.rollback`, read against what the main processes report
//! (`specs/worker.md` §7.3 step 4, §7.4 step 3).
//!
//! A fault of the target reverts an update by itself, through the reports the
//! switch reads. This module adds what no process can report about itself: that
//! its steps are slower, or its throughput lower, than the steps the deployment
//! served before the promotion, or that the requests ending on it fail too
//! often.
//!
//! - **Step time and throughput** are the [`Rates`] of the once-a-second
//!   scheduler reports ([`Observer`]): the host-observed batch time over the
//!   batches, and the query tokens the batches computed over the span. The
//!   baseline is what serving reported over `rollback.window` up to the
//!   promotion; the trial reads the same `window` up to now, and only once a
//!   whole `window` of it passed, so a target is judged on as long a span as
//!   the epoch it is compared with.
//! - **The failure rate** is the share of the requests that ended since the
//!   trial began that failed (`crate::leader::outcomes`): it needs no window,
//!   and it decides nothing before a request ended.
//!
//! A threshold left unset decides nothing, so a deployment that states none has
//! the trial of the operator's commit alone. An epoch with no batch in its
//! baseline has nothing to be compared with: its regressions decide nothing, and
//! the status shows the empty baseline.
//!
//! The trial only decides. [`Trial::breach`] names the first threshold it finds
//! passed as the update's [`Failure`], and the caller reverts through the path a
//! fault takes, so the update ends unavailable with that reason.

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::epoch::Failure;
use crate::observe::{Observer, Rates};
use crate::options::RollbackConfig;

/// How often a trial reads its thresholds and its switch's end: a report
/// arrives once a second, and a revert waits for the hold to name the epoch
/// that serves next, so this is part of the pause a revert costs.
pub(crate) const POLL: Duration = Duration::from_millis(100);

/// One replica's reports a trial reads: the world the update serves it from,
/// and what serving reported before the promotion.
pub(crate) struct Observed {
    pub(crate) observer: Arc<Observer>,
    pub(crate) world: u64,
    pub(crate) baseline: Option<Rates>,
}

impl Observed {
    /// `world`'s reports through `observer`, with the baseline of `previous`,
    /// the world that served, over `window` up to `promotion`.
    pub(crate) fn new(
        observer: Arc<Observer>,
        world: u64,
        previous: u64,
        window: Option<Duration>,
        promotion: Instant,
    ) -> Self {
        let baseline = window.and_then(|window| {
            let from = promotion.checked_sub(window).unwrap_or(promotion);
            observer.rates_between(previous, from, promotion)
        });
        Self {
            observer,
            world,
            baseline,
        }
    }
}

/// An update on probation, as far as its thresholds read it.
pub(crate) struct Trial {
    thresholds: RollbackConfig,
    observed: Vec<Observed>,
    /// When the trial began: reports and outcomes count from here.
    began: Instant,
    /// The request outcomes counted when the trial began.
    outcomes: (u64, u64),
}

impl Trial {
    /// A trial of `thresholds` over `observed`, beginning now with `outcomes`
    /// counted so far.
    pub(crate) fn begin(
        thresholds: RollbackConfig,
        observed: Vec<Observed>,
        outcomes: (u64, u64),
    ) -> Self {
        Self {
            thresholds,
            observed,
            began: Instant::now(),
            outcomes,
        }
    }

    /// The requests that ended and failed since the trial began, of `outcomes`
    /// counted now.
    fn since(&self, outcomes: (u64, u64)) -> (u64, u64) {
        (
            outcomes.0.saturating_sub(self.outcomes.0),
            outcomes.1.saturating_sub(self.outcomes.1),
        )
    }

    /// The first threshold the trial passed, `outcomes` being the requests
    /// counted now: what the update reverts with, or `None` while none is.
    pub(crate) fn breach(&self, outcomes: (u64, u64)) -> Option<Failure> {
        let failure = |kind: &str, reason: String| Failure {
            kind: kind.to_owned(),
            reason,
            device: None,
        };
        if let Some(rate) = self.thresholds.failure_rate {
            let (ended, failed) = self.since(outcomes);
            if ended > 0 && failed as f64 / ended as f64 > rate {
                return Some(failure(
                    "failure_rate",
                    format!(
                        "{failed} of the {ended} requests that ended on trial failed, past the \
                         failure rate {rate}"
                    ),
                ));
            }
        }
        let window = self.thresholds.window?;
        if self.began.elapsed() < window {
            return None;
        }
        for observed in &self.observed {
            let (Some(baseline), Some(rates)) = (
                observed.baseline,
                observed.observer.rates(observed.world, self.began, window),
            ) else {
                continue;
            };
            if let Some(regression) = self.thresholds.step_time_regression
                && baseline.step_us > 0.0
                && rates.step_us > baseline.step_us * (1.0 + regression)
            {
                return Some(failure(
                    "step_time_regression",
                    format!(
                        "the epoch on trial's steps take {:.0} us against {:.0} us before its \
                         promotion, past the regression {regression}",
                        rates.step_us, baseline.step_us
                    ),
                ));
            }
            if let Some(regression) = self.thresholds.throughput_regression
                && baseline.tokens_per_second > 0.0
                && rates.tokens_per_second < baseline.tokens_per_second * (1.0 - regression)
            {
                return Some(failure(
                    "throughput_regression",
                    format!(
                        "the epoch on trial computes {:.0} tokens/s against {:.0} before its \
                         promotion, past the regression {regression}",
                        rates.tokens_per_second, baseline.tokens_per_second
                    ),
                ));
            }
        }
        None
    }

    /// The trial as `GET /admin/config` reports it.
    pub(crate) fn report(&self, outcomes: (u64, u64)) -> Value {
        let (ended, failed) = self.since(outcomes);
        let window = self.thresholds.window;
        let rates: Vec<_> = self
            .observed
            .iter()
            .map(|observed| {
                let rates = window
                    .and_then(|window| observed.observer.rates(observed.world, self.began, window));
                json!({
                    "world": observed.world,
                    "baseline": observed.baseline,
                    "rates": rates,
                })
            })
            .collect();
        json!({
            "thresholds": self.thresholds,
            "seconds": self.began.elapsed().as_secs_f64(),
            "requests_ended": ended,
            "requests_failed": failed,
            "replicas": rates,
        })
    }
}

/// Reads `trial` for as long as `standing` says its update is on probation, and
/// hands the first threshold it passes to `revert`, which ends the watch.
pub(crate) async fn watch<Revert, Reverted>(
    trial: &Trial,
    outcomes: impl Fn() -> (u64, u64),
    standing: impl Fn() -> bool,
    revert: Revert,
) where
    Revert: FnOnce(Failure) -> Reverted,
    Reverted: std::future::Future<Output = ()>,
{
    while standing() {
        if let Some(failure) = trial.breach(outcomes()) {
            revert(failure).await;
            return;
        }
        tokio::time::sleep(POLL).await;
    }
}

#[cfg(test)]
mod tests {
    use proto::report::SchedulerStats;

    use super::*;

    const WINDOW: Duration = Duration::from_millis(40);

    fn stats(step_us: u64, batches: u64, tokens: u64) -> SchedulerStats {
        SchedulerStats {
            batch_roundtrip_us: step_us * batches,
            batch_count: batches,
            batch_tokens: tokens,
            ..Default::default()
        }
    }

    /// A trial of `thresholds` whose world 2 is compared with world 1's steps
    /// of 1000 us at 100 tokens a batch, and a whole window of it already past.
    fn trial(thresholds: RollbackConfig, observer: &Arc<Observer>) -> Trial {
        observer.record(1, 0, &stats(1_000, 100, 10_000));
        let observed = Observed::new(
            Arc::clone(observer),
            2,
            1,
            thresholds.window,
            Instant::now(),
        );
        let trial = Trial::begin(thresholds, vec![observed], (10, 1));
        std::thread::sleep(WINDOW);
        trial
    }

    fn regressions(step: Option<f64>, throughput: Option<f64>) -> RollbackConfig {
        RollbackConfig {
            step_time_regression: step,
            throughput_regression: throughput,
            window: Some(WINDOW),
            ..Default::default()
        }
    }

    /// A step past the regression reverts with that reason, and steps within
    /// it do not.
    #[test]
    fn a_slowed_target_breaches_the_step_time_regression() {
        let observer = Arc::new(Observer::default());
        let trial = trial(regressions(Some(0.2), None), &observer);
        observer.record(2, 0, &stats(1_100, 100, 10_000));
        assert_eq!(trial.breach((10, 1)), None, "10% slower is within 20%");
        observer.record(2, 0, &stats(2_000, 100, 10_000));
        let failure = trial.breach((10, 1)).expect("the average is past 20%");
        assert_eq!(failure.kind, "step_time_regression");
        assert!(
            failure.reason.contains("1550 us against 1000 us"),
            "{failure:?}"
        );
    }

    /// The same slowed target states no breach when no threshold names it: a
    /// trial without thresholds is the operator's alone.
    #[test]
    fn no_threshold_decides_nothing() {
        let observer = Arc::new(Observer::default());
        let trial = trial(RollbackConfig::default(), &observer);
        observer.record(2, 0, &stats(9_000, 100, 10));
        assert_eq!(trial.breach((1_000, 1_000)), None);
    }

    /// A lower throughput than the baseline's breaches its own threshold, and
    /// the failure names it, not the step time.
    #[test]
    fn a_target_that_computes_less_breaches_the_throughput_regression() {
        let observer = Arc::new(Observer::default());
        let observed_before = stats(1_000, 100, 10_000);
        observer.record(1, 0, &observed_before);
        let thresholds = regressions(None, Some(0.5));
        let observed = Observed::new(
            Arc::clone(&observer),
            2,
            1,
            thresholds.window,
            Instant::now(),
        );
        let baseline = observed.baseline.expect("a report before the promotion");
        let trial = Trial::begin(thresholds, vec![observed], (0, 0));
        std::thread::sleep(WINDOW);
        let close = (baseline.tokens_per_second * 0.6) as u64;
        observer.record(2, 0, &stats(1_000, 100, close));
        assert_eq!(trial.breach((0, 0)), None, "60% of it is within the half");
        std::thread::sleep(WINDOW);
        observer.record(2, 0, &stats(1_000, 100, close / 4));
        let failure = trial.breach((0, 0)).expect("a quarter is past the half");
        assert_eq!(failure.kind, "throughput_regression");
    }

    /// The requests that ended before the trial do not count, and a rate past
    /// the threshold breaches at the first report of it, before any window.
    #[test]
    fn failed_requests_past_the_rate_breach_it() {
        let observer = Arc::new(Observer::default());
        let thresholds = RollbackConfig {
            failure_rate: Some(0.25),
            ..Default::default()
        };
        let trial = Trial::begin(
            thresholds,
            vec![Observed::new(observer, 2, 1, None, Instant::now())],
            (100, 50),
        );
        assert_eq!(trial.breach((100, 50)), None, "nothing ended on trial");
        assert_eq!(trial.breach((104, 51)), None, "a quarter is not past it");
        let failure = trial.breach((104, 52)).expect("half is past it");
        assert_eq!(failure.kind, "failure_rate");
        assert!(failure.reason.contains("2 of the 4"), "{failure:?}");
    }

    /// A program-layer update's watch hands the passed threshold over once, and
    /// a watch whose update ended reverts nothing.
    #[tokio::test]
    async fn the_watch_reverts_once_and_ends_with_its_update() {
        let observer = Arc::new(Observer::default());
        let thresholds = RollbackConfig {
            failure_rate: Some(0.5),
            ..Default::default()
        };
        let trial = Trial::begin(
            thresholds,
            vec![Observed::new(observer, 2, 1, None, Instant::now())],
            (0, 0),
        );
        let reverted = std::sync::Mutex::new(Vec::new());
        watch(
            &trial,
            || (2, 2),
            || true,
            |failure| {
                reverted.lock().expect("the list").push(failure.kind);
                std::future::ready(())
            },
        )
        .await;
        assert_eq!(*reverted.lock().expect("the list"), ["failure_rate"]);

        watch(
            &trial,
            || (2, 2),
            || false,
            |_| async { panic!("an update that ended reverts nothing") },
        )
        .await;
    }

    /// A window that has not passed decides nothing, however slow the target.
    #[test]
    fn a_window_decides_only_once_it_passed() {
        let observer = Arc::new(Observer::default());
        observer.record(1, 0, &stats(1_000, 100, 10_000));
        let thresholds = RollbackConfig {
            step_time_regression: Some(0.1),
            window: Some(Duration::from_secs(3600)),
            ..Default::default()
        };
        let observed = Observed::new(
            Arc::clone(&observer),
            2,
            1,
            thresholds.window,
            Instant::now(),
        );
        let trial = Trial::begin(thresholds, vec![observed], (0, 0));
        observer.record(2, 0, &stats(9_000, 100, 10_000));
        assert_eq!(trial.breach((0, 0)), None);
    }

    /// An epoch that ran no batch before the promotion has no baseline, so its
    /// regressions decide nothing, and the status shows it empty.
    #[test]
    fn a_trial_without_a_baseline_decides_nothing_and_reports_it() {
        let observer = Arc::new(Observer::default());
        let thresholds = regressions(Some(0.1), Some(0.1));
        let observed = Observed::new(
            Arc::clone(&observer),
            2,
            1,
            thresholds.window,
            Instant::now(),
        );
        let trial = Trial::begin(thresholds, vec![observed], (0, 0));
        std::thread::sleep(WINDOW);
        observer.record(2, 0, &stats(9_000, 100, 1));
        assert_eq!(trial.breach((0, 0)), None);
        let report = trial.report((3, 1));
        assert_eq!(report["replicas"][0]["baseline"], Value::Null);
        assert_eq!(report["requests_ended"], 3);
        assert_eq!(report["requests_failed"], 1);
    }
}
