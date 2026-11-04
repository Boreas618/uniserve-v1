//! The instrumentation a world change reads: the schedulers' statistics kept
//! over the longest span a trial measures, the rates a world computes at, and
//! the samples they are read from. It is not part of the change mechanism; the
//! mechanism reads it, and profiling reports through it.

use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};

use proto::report::SchedulerStats;
use serde::{Deserialize, Serialize};

use crate::leader::locked;

/// Average step time and throughput of one world over a span.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rates {
    /// Microseconds per batch.
    pub step_us: f64,
    /// Tokens per second the forwards computed.
    pub tokens_per_second: f64,
}

/// One statistics report of one scheduler of one world: the interval's
/// host-observed batch time, batches, query tokens and numerical faults.
pub(crate) struct Sample {
    pub(crate) at: Instant,
    pub(crate) world: u64,
    pub(crate) scheduler: u32,
    pub(crate) batch_us: u64,
    pub(crate) batches: u64,
    pub(crate) tokens: u64,
    numerical_faults: u64,
}

/// The schedulers' statistics of every world, kept over the longest span a
/// trial measures, from which a trial reads the previous world's baseline
/// and its own rates and faults.
#[derive(Default)]
pub(crate) struct Observer {
    samples: std::sync::Mutex<VecDeque<Sample>>,
    /// The profiling sessions' progress the workers reported, keyed
    /// `{session}/{counter}` (`ForwardStats::profile`), summed.
    profile: std::sync::Mutex<BTreeMap<String, u64>>,
}

/// The longest span statistics are kept for.
const OBSERVED: Duration = Duration::from_secs(600);

impl Observer {
    /// Records the report of scheduler `scheduler` of `world`.
    pub(crate) fn record(&self, world: u64, scheduler: u32, stats: &SchedulerStats) {
        let tokens = stats
            .worker_forward_stats
            .as_ref()
            .map_or(0, |forward| forward.mode_tokens.values().sum());
        if let Some(forward) = &stats.worker_forward_stats {
            let mut profile = locked(&self.profile);
            for (key, value) in &forward.profile {
                *profile.entry(key.clone()).or_default() += value;
            }
        }
        let now = Instant::now();
        let mut samples = locked(&self.samples);
        samples.push_back(Sample {
            at: now,
            world,
            scheduler,
            batch_us: stats.batch_roundtrip_us,
            batches: stats.batch_count,
            tokens,
            numerical_faults: stats.numerical_faults,
        });
        while samples
            .front()
            .is_some_and(|sample| now.duration_since(sample.at) > OBSERVED)
        {
            samples.pop_front();
        }
    }

    /// `world`'s rates over the `window` before now, from `since` on; `None`
    /// without a batch in it.
    pub(crate) fn rates(&self, world: u64, since: Instant, window: Duration) -> Option<Rates> {
        let now = Instant::now();
        let from = since.max(now.checked_sub(window).unwrap_or(since));
        let samples = locked(&self.samples);
        let (mut batch_us, mut batches, mut tokens) = (0_u64, 0_u64, 0_u64);
        let mut first: Option<Instant> = None;
        for sample in samples
            .iter()
            .filter(|sample| sample.world == world && sample.at >= from)
        {
            batch_us += sample.batch_us;
            batches += sample.batches;
            tokens += sample.tokens;
            first.get_or_insert(sample.at);
        }
        if batches == 0 {
            return None;
        }
        let span = now
            .duration_since(first.unwrap_or(from))
            .max(Duration::from_secs(1));
        Some(Rates {
            step_us: batch_us as f64 / batches as f64,
            tokens_per_second: tokens as f64 / span.as_secs_f64(),
        })
    }

    /// Calls `visit` with every report of `world` made in `[from, until)`,
    /// in the order they arrived.
    pub(crate) fn samples(
        &self,
        world: u64,
        from: Instant,
        until: Instant,
        mut visit: impl FnMut(&Sample),
    ) {
        for sample in locked(&self.samples)
            .iter()
            .filter(|sample| sample.world == world && sample.at >= from && sample.at < until)
        {
            visit(sample);
        }
    }

    /// Profiling session `session`'s progress the workers reported: the
    /// steps their windows took, the bytes of their files, and the workers
    /// whose window ended.
    pub(crate) fn profile(&self, session: u64) -> (u64, u64, u64) {
        let profile = locked(&self.profile);
        let counter = |name: &str| {
            profile
                .get(&format!("{session}/{name}"))
                .copied()
                .unwrap_or_default()
        };
        (counter("steps"), counter("bytes"), counter("ended"))
    }

    /// The calls of `world` that produced non-finite logits since `since`.
    pub(crate) fn numerical_faults(&self, world: u64, since: Instant) -> u64 {
        locked(&self.samples)
            .iter()
            .filter(|sample| sample.world == world && sample.at >= since)
            .map(|sample| sample.numerical_faults)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proto::report::SchedulerStats;
    use std::collections::BTreeMap;

    fn stats(batch_us: u64, batches: u64, tokens: u64, faults: u64) -> SchedulerStats {
        SchedulerStats {
            batch_roundtrip_us: batch_us,
            batch_count: batches,
            numerical_faults: faults,
            worker_forward_stats: Some(proto::report::ForwardStats {
                mode_tokens: BTreeMap::from([("decode".to_owned(), tokens)]),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    /// A world's rates average its own batches over the window, apart from
    /// another world's.
    #[test]
    fn rates_average_one_worlds_batches() {
        let observer = Observer::default();
        let since = Instant::now();
        observer.record(1, 0, &stats(1_000, 10, 500, 0));
        observer.record(1, 0, &stats(3_000, 10, 500, 0));
        observer.record(2, 0, &stats(90_000, 1, 1, 2));
        let rates = observer
            .rates(1, since, Duration::from_secs(60))
            .expect("world 1 ran batches");
        assert!((rates.step_us - 200.0).abs() < 1e-9, "{rates:?}");
        assert!(rates.tokens_per_second > 0.0);
        assert_eq!(observer.numerical_faults(1, since), 0);
        assert_eq!(observer.numerical_faults(2, since), 2);
        assert!(observer.rates(3, since, Duration::from_secs(60)).is_none());
    }
}
