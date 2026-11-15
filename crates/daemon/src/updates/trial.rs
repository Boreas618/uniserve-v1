//! The trial an update stands in its `Trial` phase: the thresholds of
//! `update_config.rollback`, read against what the engines report.
//!
//! A fault of the target rolls an update back by itself, through the reports the
//! switch reads. This module adds what no process can report about itself: that
//! its steps are slower, or its throughput lower, than the steps the deployment
//! served before the switch, or that the requests ending on it fail too often.
//!
//! - **The trial begins when the switch ended** (`specs/updates.md` §3.7):
//!   its follower begins it once the update entered its `Trial` phase. What
//!   the target computed while the update switched is not its trial's: its
//!   warm-up at its first opening, the first steps of the requests migrated to
//!   it at the drain's deadline, and steps that shared each GPU with the
//!   draining epoch.
//! - **Step time and throughput** are the [`Rates`] of the scheduler reports
//!   ([`Observer`]): the host-observed batch time over the batches, and the
//!   query tokens the batches computed over the span their reports cover. The
//!   baseline is what serving reported over `rollback.window` up to the
//!   switch; the trial reads the same `window` up to each reading, from its
//!   beginning on, and decides only once a whole `window` of it passed, so a
//!   target is judged on as long a span as the epoch it is compared with. A
//!   step regression is a trial's average step longer than
//!   `1 + step_time_regression` times the baseline's; a throughput regression
//!   is a throughput below `1 - throughput_regression` times the baseline's.
//! - **The failure rate** is the share of the requests that ended since the
//!   trial began that failed (`crate::leader::request_counts`): it needs no
//!   window, and it decides nothing before a request ended.
//!
//! A threshold left unset decides nothing, so a deployment that states none has
//! the trial of the operator's commit alone. An epoch with no batch in its
//! baseline has nothing to be compared with: its regressions decide nothing, and
//! the status shows the empty baseline.
//!
//! The trial only decides. [`Trial::read`] reads the reports once and names the
//! first threshold the reading passed as the update's [`Failure`]; the update's
//! follower reads it as each report of the trial's replicas arrives
//! ([`Observer::records`]), the updater only while the update stands on trial,
//! and rolls back through the path a fault takes, so the update ends
//! unavailable with that reason. The
//! status states the latest reading ([`Trial::report`]): for a trial a
//! threshold ended, the very rates that decided it.

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::epoch::Failure;
use crate::observe::{Observer, Rates};
use crate::options::RollbackConfig;

/// One replica's reports a trial reads: the epoch the update serves it from,
/// and what serving reported before the switch.
pub(crate) struct Baseline {
    pub(crate) observer: Arc<Observer>,
    pub(crate) epoch: u64,
    pub(crate) baseline: Option<Rates>,
}

impl Baseline {
    /// `epoch`'s reports through `observer`, with the baseline of `previous`,
    /// the epoch that served, over `window` up to `switch`, when the switch
    /// began; no baseline without an epoch that served before.
    pub(crate) fn new(
        observer: Arc<Observer>,
        epoch: u64,
        previous: Option<u64>,
        window: Option<Duration>,
        switch: Instant,
    ) -> Self {
        let baseline = window.zip(previous).and_then(|(window, previous)| {
            let from = switch.checked_sub(window).unwrap_or(switch);
            observer.rates_between(previous, from, switch)
        });
        Self {
            observer,
            epoch,
            baseline,
        }
    }
}

/// What one reading of a trial read.
struct Reading {
    /// When it read.
    at: Instant,
    /// The requests that ended on trial up to it, and those of them that
    /// failed.
    requests: (u64, u64),
    /// Each replica's rates over the window up to it, in the order of the
    /// trial's baselines; `None` for a replica that ran no batch in it.
    rates: Vec<Option<Rates>>,
}

/// An update on trial, as far as its thresholds read it.
pub(crate) struct Trial {
    thresholds: RollbackConfig,
    baselines: Vec<Baseline>,
    /// When the trial began: reports and request counts start from here.
    began: Instant,
    /// The requests that had ended, and those of them that had failed, when the trial began.
    counts: (u64, u64),
    /// The latest reading, `None` before the first.
    reading: Option<Reading>,
}

impl Trial {
    /// A trial of `thresholds` over `baselines`, beginning now with the
    /// `counts` of requests so far. Its caller begins it when the update
    /// entered its `Trial` phase.
    pub(crate) fn begin(
        thresholds: RollbackConfig,
        baselines: Vec<Baseline>,
        counts: (u64, u64),
    ) -> Self {
        Self {
            thresholds,
            baselines,
            began: Instant::now(),
            counts,
            reading: None,
        }
    }

    /// Reads the engines' reports once, `counts` being the requests counted
    /// now, and returns the first threshold the reading passed: what the
    /// update rolls back with, or `None` while none is. The reading is what
    /// [`Self::report`] states until the next one.
    pub(crate) fn read(&mut self, counts: (u64, u64)) -> Option<Failure> {
        let window = self.thresholds.window;
        let rates = self
            .baselines
            .iter()
            .map(|replica| {
                window.and_then(|window| replica.observer.rates(replica.epoch, self.began, window))
            })
            .collect();
        let reading = Reading {
            at: Instant::now(),
            requests: (
                counts.0.saturating_sub(self.counts.0),
                counts.1.saturating_sub(self.counts.1),
            ),
            rates,
        };

        let failure = self.breach(&reading);
        self.reading = Some(reading);
        failure
    }

    /// The first threshold `reading` passed.
    fn breach(&self, reading: &Reading) -> Option<Failure> {
        let failure = |kind: &str, reason: String| Failure {
            kind: kind.to_owned(),
            reason,
            device: None,
        };

        if let Some(rate) = self.thresholds.failure_rate {
            let (ended, failed) = reading.requests;
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

        // The regressions compare a whole window of the trial with one of the
        // baseline.
        let window = self.thresholds.window?;
        if reading.at.saturating_duration_since(self.began) < window {
            return None;
        }
        for (replica, rates) in self.baselines.iter().zip(&reading.rates) {
            let (Some(baseline), Some(rates)) = (replica.baseline, rates) else {
                continue;
            };
            if let Some(regression) = self.thresholds.step_time_regression
                && baseline.step_us > 0.0
                && rates.step_us > baseline.step_us * (1.0 + regression)
            {
                return Some(failure(
                    "step_time_regression",
                    format!(
                        "the epoch on trial's steps take {:.0} us against {:.0} us before the \
                         switch, past the regression {regression}",
                        rates.step_us, baseline.step_us
                    ),
                ));
            }
            if let Some(regression) = self.thresholds.throughput_regression
                && let (Some(before), Some(now)) =
                    (baseline.tokens_per_second, rates.tokens_per_second)
                && before > 0.0
                && now < before * (1.0 - regression)
            {
                return Some(failure(
                    "throughput_regression",
                    format!(
                        "the epoch on trial computes {now:.0} tokens/s against {before:.0} \
                         before the switch, past the regression {regression}"
                    ),
                ));
            }
        }
        None
    }

    /// The trial as `GET /admin/config` reports it: its thresholds and its
    /// latest reading, each replica's rates beside its baseline.
    pub(crate) fn report(&self) -> Value {
        let (ended, failed) = self
            .reading
            .as_ref()
            .map_or((0, 0), |reading| reading.requests);
        let replicas: Vec<_> = self
            .baselines
            .iter()
            .enumerate()
            .map(|(index, replica)| {
                let rates = self
                    .reading
                    .as_ref()
                    .and_then(|reading| reading.rates.get(index).copied().flatten());
                json!({
                    "epoch": replica.epoch,
                    "baseline": replica.baseline,
                    "rates": rates,
                })
            })
            .collect();
        json!({
            "thresholds": self.thresholds,
            "seconds": self.began.elapsed().as_secs_f64(),
            "requests_ended": ended,
            "requests_failed": failed,
            "replicas": replicas,
        })
    }
}

#[cfg(test)]
mod tests {
    use proto::report::SchedulerStats;

    use super::*;

    const WINDOW: Duration = Duration::from_millis(40);

    /// A report of `batches` batches of `step_us` each, computing `tokens`
    /// query tokens together.
    fn stats(step_us: u64, batches: u64, tokens: u64) -> SchedulerStats {
        SchedulerStats {
            batch_roundtrip_us: step_us * batches,
            batch_count: batches,
            batch_tokens: tokens,
            ..Default::default()
        }
    }

    /// A trial of `thresholds` whose epoch 2 is compared with epoch 1's steps
    /// of 1000 us at 100 tokens a batch, and a whole window of it already past.
    fn trial(thresholds: RollbackConfig, observer: &Arc<Observer>) -> Trial {
        observer.record(1, 0, &stats(1_000, 100, 10_000));
        let replica = Baseline::new(
            Arc::clone(observer),
            2,
            Some(1),
            thresholds.window,
            Instant::now(),
        );
        let trial = Trial::begin(thresholds, vec![replica], (10, 1));
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

    /// A step past the regression rolls back with that reason, and steps within
    /// it do not.
    #[test]
    fn a_slowed_target_breaches_the_step_time_regression() {
        let observer = Arc::new(Observer::new(Duration::from_secs(60)));
        let mut trial = trial(regressions(Some(0.2), None), &observer);
        observer.record(2, 0, &stats(1_100, 100, 10_000));
        assert_eq!(trial.read((10, 1)), None, "10% slower is within 20%");
        observer.record(2, 0, &stats(2_000, 100, 10_000));
        let failure = trial.read((10, 1)).expect("the average is past 20%");
        assert_eq!(failure.kind, "step_time_regression");
        assert!(
            failure.reason.contains("1550 us against 1000 us"),
            "{failure:?}"
        );
    }

    /// The steps of a deployment that served at 10666 us and a target that
    /// steps at 9829 us once it serves alone, after a switch in which the
    /// target's warm-up and first migrated steps averaged 29744 us: the
    /// switch is not the trial's, so the faster target stands, and the
    /// status states the rates the reading decided on. A target slower than
    /// the regression allows rolls back, and its reason and its status state
    /// the same rates.
    #[test]
    fn the_trial_reads_its_target_from_the_switchs_end_and_states_what_it_decided_on() {
        let thresholds = regressions(Some(0.05), None);
        let serve = |target_us: u64| {
            let observer = Arc::new(Observer::new(Duration::from_secs(60)));
            observer.record(1, 0, &stats(10_666, 400, 400));
            let replica = Baseline::new(
                Arc::clone(&observer),
                2,
                Some(1),
                thresholds.window,
                Instant::now(),
            );
            // The switch: the target's first batches, both epochs on one GPU.
            observer.record(2, 0, &stats(29_744, 3, 4));
            std::thread::sleep(Duration::from_millis(2));

            let mut trial = Trial::begin(thresholds.clone(), vec![replica], (0, 0));
            std::thread::sleep(WINDOW);
            observer.record(2, 0, &stats(target_us, 400, 400));
            let failure = trial.read((0, 0));
            (failure, trial.report())
        };

        let (failure, report) = serve(9_829);
        assert_eq!(failure, None, "{report}");
        let replica = &report["replicas"][0];
        assert_eq!(replica["rates"]["step_us"], 9_829.0);
        assert_eq!(replica["baseline"]["step_us"], 10_666.0);

        // 11,500 us is 7.8% past the baseline's 10,666 us.
        let (failure, report) = serve(11_500);
        let failure = failure.expect("a regression past 5%");
        assert_eq!(failure.kind, "step_time_regression");
        assert!(
            failure.reason.contains("11500 us against 10666 us"),
            "{failure:?}"
        );
        assert_eq!(report["replicas"][0]["rates"]["step_us"], 11_500.0);
    }

    /// The same slowed target states no breach when no threshold names it: a
    /// trial without thresholds is the operator's alone.
    #[test]
    fn no_threshold_decides_nothing() {
        let observer = Arc::new(Observer::new(Duration::from_secs(60)));
        let mut trial = trial(RollbackConfig::default(), &observer);
        observer.record(2, 0, &stats(9_000, 100, 10));
        assert_eq!(trial.read((1_000, 1_000)), None);
    }

    /// A lower throughput than the baseline's breaches its own threshold, and
    /// the failure names it, not the step time.
    #[test]
    fn a_target_that_computes_less_breaches_the_throughput_regression() {
        let observer = Arc::new(Observer::new(Duration::from_secs(60)));
        // A scheduler's first report covers no known interval; the second covers a window.
        observer.record(1, 0, &stats(1_000, 100, 10_000));
        std::thread::sleep(WINDOW);
        observer.record(1, 0, &stats(1_000, 100, 10_000));
        let thresholds = regressions(None, Some(0.5));
        let replica = Baseline::new(
            Arc::clone(&observer),
            2,
            Some(1),
            thresholds.window,
            Instant::now(),
        );
        let baseline = replica.baseline.expect("a report before the switch");
        assert!(baseline.tokens_per_second.is_some(), "{baseline:?}");
        let mut trial = Trial::begin(thresholds, vec![replica], (0, 0));
        observer.record(2, 0, &stats(1_000, 100, 10_000));
        std::thread::sleep(WINDOW);
        observer.record(2, 0, &stats(1_000, 100, 9_000));
        assert_eq!(trial.read((0, 0)), None, "90% of it is within the half");
        std::thread::sleep(WINDOW);
        observer.record(2, 0, &stats(1_000, 100, 1_000));
        let failure = trial.read((0, 0)).expect("a tenth is past the half");
        assert_eq!(failure.kind, "throughput_regression");
    }

    /// The requests that ended before the trial do not count, and a rate past
    /// the threshold breaches at the first report of it, before any window.
    #[test]
    fn failed_requests_past_the_rate_breach_it() {
        let observer = Arc::new(Observer::new(Duration::from_secs(60)));
        let thresholds = RollbackConfig {
            failure_rate: Some(0.25),
            ..Default::default()
        };
        let mut trial = Trial::begin(
            thresholds,
            vec![Baseline::new(observer, 2, Some(1), None, Instant::now())],
            (100, 50),
        );
        assert_eq!(trial.read((100, 50)), None, "nothing ended on trial");
        assert_eq!(trial.read((104, 51)), None, "a quarter is not past it");
        let failure = trial.read((104, 52)).expect("half is past it");
        assert_eq!(failure.kind, "failure_rate");
        assert!(failure.reason.contains("2 of the 4"), "{failure:?}");
    }

    /// A window that has not passed decides nothing, however slow the target.
    #[test]
    fn a_window_decides_only_once_it_passed() {
        let observer = Arc::new(Observer::new(Duration::from_secs(60)));
        observer.record(1, 0, &stats(1_000, 100, 10_000));
        let thresholds = RollbackConfig {
            step_time_regression: Some(0.1),
            window: Some(Duration::from_secs(3600)),
            ..Default::default()
        };
        let replica = Baseline::new(
            Arc::clone(&observer),
            2,
            Some(1),
            thresholds.window,
            Instant::now(),
        );
        let mut trial = Trial::begin(thresholds, vec![replica], (0, 0));
        observer.record(2, 0, &stats(9_000, 100, 10_000));
        assert_eq!(trial.read((0, 0)), None);
    }

    /// An epoch that ran no batch before the switch has no baseline, so its
    /// regressions decide nothing, and the status shows it empty.
    #[test]
    fn a_trial_without_a_baseline_decides_nothing_and_reports_it() {
        let observer = Arc::new(Observer::new(Duration::from_secs(60)));
        let thresholds = regressions(Some(0.1), Some(0.1));
        let replica = Baseline::new(
            Arc::clone(&observer),
            2,
            Some(1),
            thresholds.window,
            Instant::now(),
        );
        let mut trial = Trial::begin(thresholds, vec![replica], (0, 0));
        std::thread::sleep(WINDOW);
        observer.record(2, 0, &stats(9_000, 100, 1));
        assert_eq!(trial.read((3, 1)), None);
        let report = trial.report();
        assert_eq!(report["replicas"][0]["baseline"], Value::Null);
        assert_eq!(report["requests_ended"], 3);
        assert_eq!(report["requests_failed"], 1);
    }
}
