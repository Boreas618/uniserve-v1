//! The instrumentation an epoch change and a profiling session read: the schedulers' statistics
//! kept over the spans their consumers read back, the rates an epoch computes at, and the samples
//! they are read from. It is not part of the change mechanism; the mechanism reads it, and
//! profiling reports through it.
//!
//! Every record notifies the readers ([`Observer::records`]): a consumer decides as each report
//! arrives, never at an interval of its own. The span a report covers is the time since its
//! scheduler's previous report, as the reports arrive, so no report period is assumed.

use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};

use proto::report::{ProfileProgress, SchedulerStats};
use serde::{Deserialize, Serialize};
use tokio::sync::watch;

/// Average step time and throughput of one epoch over a span.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rates {
    /// Microseconds per batch.
    pub step_us: f64,
    /// Tokens per second the forwards computed over the span the reports cover; `None` when
    /// every report of the span is its scheduler's first, whose span is unknown.
    pub tokens_per_second: Option<f64>,
}

/// One statistics report of one scheduler of one epoch: the interval's host-observed batch
/// time, batches and query tokens.
pub(crate) struct Sample {
    pub(crate) at: Instant,
    /// The scheduler's previous report of the epoch, where this one's interval began; `None`
    /// for its first.
    pub(crate) since: Option<Instant>,
    pub(crate) epoch: u64,
    pub(crate) scheduler: u32,
    pub(crate) batch_us: u64,
    pub(crate) batches: u64,
    pub(crate) tokens: u64,
}

/// What the observer keeps between records.
struct Kept {
    samples: VecDeque<Sample>,
    /// Each epoch's schedulers' latest report, where the next one's interval begins.
    latest: BTreeMap<(u64, u32), Instant>,
    /// How long a sample is kept after it arrived.
    retention: Duration,
}

/// The schedulers' statistics of every epoch of one replica, kept for the longest span a reader
/// reads back ([`Observer::retain`]), from which a trial reads the previous epoch's baseline and
/// its own rates, the transfers' control their step times, and a profiling session its guard.
pub(crate) struct Observer {
    kept: std::sync::Mutex<Kept>,
    /// The profiling sessions' progress the workers reported, by session, summed.
    profile: std::sync::Mutex<BTreeMap<u64, ProfileProgress>>,
    /// Counts the records, which every reader waits on to decide as each one arrives.
    records: watch::Sender<u64>,
}

impl Observer {
    /// An observer that keeps each report for `retention` after it arrived.
    pub(crate) fn new(retention: Duration) -> Self {
        Self {
            kept: std::sync::Mutex::new(Kept {
                samples: VecDeque::new(),
                latest: BTreeMap::new(),
                retention,
            }),
            profile: std::sync::Mutex::new(BTreeMap::new()),
            records: watch::Sender::new(0),
        }
    }

    /// Keeps reports for `span` at least from now on: a reader that reads back further than the
    /// observer kept reads what it kept.
    pub(crate) fn retain(&self, span: Duration) {
        let mut kept = service::lock(&self.kept);
        kept.retention = kept.retention.max(span);
    }

    /// How long a report is kept after it arrived.
    pub(crate) fn retention(&self) -> Duration {
        service::lock(&self.kept).retention
    }

    /// The records as they arrive: the receiver changes with every one.
    pub(crate) fn records(&self) -> watch::Receiver<u64> {
        self.records.subscribe()
    }

    /// Records the report of scheduler `scheduler` of `epoch`, and notifies the readers.
    pub(crate) fn record(&self, epoch: u64, scheduler: u32, stats: &SchedulerStats) {
        if let Some(forward) = &stats.worker_forward_stats {
            let mut profile = service::lock(&self.profile);
            for (session, progress) in &forward.profile {
                profile.entry(*session).or_default().add(progress);
            }
        }
        {
            // The sample is stamped under the lock, so the samples stand in the order of their
            // stamps and a reader that reads up to an instant it took before locking misses none
            // it will not read next time: a stamp taken before the lock could fall before that
            // instant while the sample is pushed after the reader's pass.
            let mut kept = service::lock(&self.kept);
            let now = Instant::now();
            let since = kept.latest.insert((epoch, scheduler), now);
            kept.samples.push_back(Sample {
                at: now,
                since,
                epoch,
                scheduler,
                batch_us: stats.batch_roundtrip_us,
                batches: stats.batch_count,
                tokens: stats.batch_tokens,
            });
            let retention = kept.retention;
            while kept
                .samples
                .front()
                .is_some_and(|sample| now.duration_since(sample.at) > retention)
            {
                kept.samples.pop_front();
            }
            kept.latest
                .retain(|_, latest| now.duration_since(*latest) <= retention);
        }
        self.records.send_modify(|records| *records += 1);
    }

    /// `epoch`'s rates over the `window` before now, from `since` on; `None` without a batch in
    /// it.
    pub(crate) fn rates(&self, epoch: u64, since: Instant, window: Duration) -> Option<Rates> {
        let now = Instant::now();
        let from = since.max(now.checked_sub(window).unwrap_or(since));
        self.rates_between(epoch, from, now)
    }

    /// `epoch`'s rates over the reports it made in `[from, until)`; `None` without a batch among
    /// them. The throughput is the tokens of the reports whose interval is known over the span
    /// those intervals cover: from the earliest interval's start to the latest report.
    pub(crate) fn rates_between(&self, epoch: u64, from: Instant, until: Instant) -> Option<Rates> {
        let (mut batch_us, mut batches) = (0_u64, 0_u64);
        let mut tokens = 0_u64;
        let mut covered: Option<(Instant, Instant)> = None;
        self.samples(epoch, from, until, |sample| {
            batch_us += sample.batch_us;
            batches += sample.batches;
            if let Some(since) = sample.since {
                tokens += sample.tokens;
                covered = Some(match covered {
                    Some((start, end)) => (start.min(since), end.max(sample.at)),
                    None => (since, sample.at),
                });
            }
        });
        if batches == 0 {
            return None;
        }
        let tokens_per_second = covered
            .map(|(start, end)| end.saturating_duration_since(start))
            .filter(|span| !span.is_zero())
            .map(|span| tokens as f64 / span.as_secs_f64());
        Some(Rates {
            step_us: batch_us as f64 / batches as f64,
            tokens_per_second,
        })
    }

    /// Calls `visit` with every report of `epoch` made in `[from, until)`, in the order they
    /// arrived.
    pub(crate) fn samples(
        &self,
        epoch: u64,
        from: Instant,
        until: Instant,
        mut visit: impl FnMut(&Sample),
    ) {
        for sample in service::lock(&self.kept)
            .samples
            .iter()
            .filter(|sample| sample.epoch == epoch && sample.at >= from && sample.at < until)
        {
            visit(sample);
        }
    }

    /// Profiling session `session`'s progress the workers reported: the steps their windows
    /// took, the bytes of their files, and the workers whose window ended.
    pub(crate) fn profile(&self, session: u64) -> ProfileProgress {
        service::lock(&self.profile)
            .get(&session)
            .copied()
            .unwrap_or_default()
    }
}

/// The span an observer keeps for a deployment whose updates `options` configure: the longest a
/// reader reads back from a report's arrival. A trial's baseline is the `rollback.window` before
/// the switch began, read once the switch's drain ended; the transfers' control reads its
/// `baseline` before the preparation began.
pub(crate) fn span(options: &crate::options::UpdateConfig) -> Duration {
    let trial =
        options.rollback.window.unwrap_or_default() + Duration::from_secs_f64(options.drain_s);
    trial.max(options.baseline)
}

/// Waits for the next record of any of `records`; never when there is none to wait on, or once
/// every observer they read is gone.
pub(crate) async fn next_record(records: &mut [watch::Receiver<u64>]) {
    let waits: Vec<_> = records
        .iter_mut()
        .map(|records| Box::pin(records.changed()))
        .collect();
    if waits.is_empty() {
        return std::future::pending().await;
    }
    let (changed, _, _) = futures::future::select_all(waits).await;
    if changed.is_err() {
        std::future::pending::<()>().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proto::report::SchedulerStats;

    fn stats(batch_us: u64, batches: u64, tokens: u64) -> SchedulerStats {
        SchedulerStats {
            batch_roundtrip_us: batch_us,
            batch_count: batches,
            batch_tokens: tokens,
            ..Default::default()
        }
    }

    /// A report counts toward the span it arrived in: the steps of a span that ended are read
    /// apart from the ones after it.
    #[test]
    fn rates_between_read_the_reports_of_one_span() {
        let observer = Observer::new(Duration::from_secs(60));
        let started = Instant::now();
        observer.record(1, 0, &stats(1_000, 10, 500));
        let ended = Instant::now();
        observer.record(1, 0, &stats(9_000, 10, 500));
        let after = Instant::now();

        let rates = observer
            .rates_between(1, started, ended)
            .expect("the span holds a report");
        assert!((rates.step_us - 100.0).abs() < 1e-9, "{rates:?}");
        let rates = observer
            .rates_between(1, ended, after)
            .expect("the span holds a report");
        assert!((rates.step_us - 900.0).abs() < 1e-9, "{rates:?}");
    }

    /// An epoch's rates average its own batches over the window, apart from another epoch's,
    /// and its throughput is read over the intervals its reports cover: a scheduler's first
    /// report covers no known interval.
    #[test]
    fn rates_average_one_epochs_batches_over_the_intervals_its_reports_cover() {
        let observer = Observer::new(Duration::from_secs(60));
        let since = Instant::now();
        observer.record(1, 0, &stats(1_000, 10, 500));
        let first = observer
            .rates(1, since, Duration::from_secs(60))
            .expect("epoch 1 ran batches");
        assert_eq!(first.tokens_per_second, None, "{first:?}");

        observer.record(1, 0, &stats(3_000, 10, 500));
        observer.record(2, 0, &stats(90_000, 1, 1));
        let rates = observer
            .rates(1, since, Duration::from_secs(60))
            .expect("epoch 1 ran batches");
        assert!((rates.step_us - 200.0).abs() < 1e-9, "{rates:?}");
        assert!(
            rates.tokens_per_second.is_some_and(|rate| rate > 0.0),
            "{rates:?}"
        );
        assert!(observer.rates(3, since, Duration::from_secs(60)).is_none());
    }

    /// Every record changes what a reader waits on, whichever observer of several it reads.
    #[tokio::test]
    async fn a_reader_wakes_on_each_record_of_any_observer() {
        let (first, second) = (
            Observer::new(Duration::from_secs(60)),
            Observer::new(Duration::from_secs(60)),
        );
        let mut records = vec![first.records(), second.records()];

        second.record(1, 0, &stats(1_000, 1, 1));
        next_record(&mut records).await;
        assert_eq!(*records[1].borrow_and_update(), 1);
        first.record(1, 0, &stats(1_000, 1, 1));
        next_record(&mut records).await;
        assert_eq!(*records[0].borrow_and_update(), 1);
    }
}
