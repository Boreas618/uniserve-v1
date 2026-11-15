//! Periodic logging of generation throughput and scheduler occupancy.
//!
//! The statistics sources feed the log as they record their families: every finished request's
//! token usage ([`StatsLog::usage`], from `stats::StatLogger`) and every scheduler's interval
//! report ([`StatsLog::scheduler`], from the leader's statistics sink). The log keeps one
//! interval's sums and each scheduler's latest occupancy; the logger takes them once per
//! `logging_config.stats_interval` and logs one line of structured fields. It reads no metric
//! family back.
//!
//! An interval's throughput counts the tokens of the requests that finished in it, so a request
//! whose stream ended before its usage report adds none. Occupancy is the latest report of every
//! scheduler that reported in the interval: a retired scheduler stops reporting and leaves the
//! next interval's line.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio_util::task::AbortOnDropHandle;
use tracing::{debug, info};

use proto::report::SchedulerStats;

use crate::metrics::{Metrics, SchedulerLabels};

/// The sums of one interval of the statistics log.
#[derive(Debug, Default)]
struct Interval {
    /// Prompt tokens of the requests that finished.
    prompt_tokens: u64,
    /// Generated tokens of the requests that finished.
    generation_tokens: u64,
    /// Prefix-cache query tokens the schedulers reported.
    prefix_cache_queries: u64,
    /// Prefix-cache hit tokens the schedulers reported.
    prefix_cache_hits: u64,
    /// Each scheduler's latest occupancy: running and waiting requests, and
    /// its cache usage.
    occupancy: BTreeMap<(String, u32, u32), (u64, u64, f64)>,
}

/// What the statistics sources feed the statistics log.
#[derive(Debug, Default)]
pub(crate) struct StatsLog {
    interval: Mutex<Interval>,
}

impl StatsLog {
    /// Adds a finished request's tokens.
    pub(crate) fn usage(&self, prompt_tokens: u64, generation_tokens: u64) {
        let mut interval = service::lock(&self.interval);
        interval.prompt_tokens += prompt_tokens;
        interval.generation_tokens += generation_tokens;
    }

    /// Adds one scheduler's interval report.
    pub(crate) fn scheduler(&self, labels: &SchedulerLabels, stats: &SchedulerStats) {
        let mut interval = service::lock(&self.interval);
        interval.prefix_cache_queries += stats.prefix_cache_stats.base.queries;
        interval.prefix_cache_hits += stats.prefix_cache_stats.base.hits;
        interval.occupancy.insert(
            (labels.model_name.clone(), labels.replica, labels.scheduler),
            (
                stats.num_running_reqs,
                stats.num_waiting_reqs,
                stats.prefix_cache_usage,
            ),
        );
    }

    /// Takes the interval's sums, starting the next interval from none.
    fn take(&self) -> Interval {
        std::mem::take(&mut *service::lock(&self.interval))
    }
}

/// Periodic logger for throughput and scheduler state.
///
/// Spawns a background task that logs one line per interval. When idle (both current and
/// previous throughputs are zero), it logs at DEBUG level; when load drops to zero, it emits one
/// final INFO-level line before going quiet. Dropping the logger aborts the task.
pub(crate) struct StatsLogger {
    _task: AbortOnDropHandle<()>,
}

impl StatsLogger {
    /// Starts the background task that logs `metrics`' statistics log every `period`.
    ///
    /// Must be called within a Tokio runtime, since it spawns onto the current one.
    pub(crate) fn start(metrics: Arc<Metrics>, period: Duration) -> Self {
        let task = AbortOnDropHandle::new(tokio::spawn(run(metrics, period)));
        Self { _task: task }
    }
}

/// Logs `metrics`' statistics log every `period` until its task is aborted.
async fn run(metrics: Arc<Metrics>, period: Duration) {
    let log = &metrics.log;
    let mut ticks = tokio::time::interval(period);
    ticks.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    // The first tick fires at once and starts the first interval.
    ticks.tick().await;
    drop(log.take());
    let mut started = Instant::now();
    let mut was_idle = true;
    loop {
        ticks.tick().await;
        let interval = log.take();
        let now = Instant::now();
        let seconds = now.duration_since(started).as_secs_f64();
        started = now;

        let prompt_throughput = interval.prompt_tokens as f64 / seconds;
        let generation_throughput = interval.generation_tokens as f64 / seconds;
        let idle = interval.prompt_tokens == 0 && interval.generation_tokens == 0;
        let (running, waiting, usage) = interval.occupancy.values().fold(
            (0, 0, 0.0),
            |(running, waiting, usage), &(more_running, more_waiting, more_usage)| {
                (
                    running + more_running,
                    waiting + more_waiting,
                    usage + more_usage,
                )
            },
        );
        // Cache usage is each scheduler's fraction, averaged over the schedulers.
        let cache_usage = if interval.occupancy.is_empty() {
            0.0
        } else {
            usage / interval.occupancy.len() as f64
        };
        let prefix_cache_hit_rate = if interval.prefix_cache_queries > 0 {
            interval.prefix_cache_hits as f64 / interval.prefix_cache_queries as f64
        } else {
            0.0
        };

        // An idle interval after an idle one logs at DEBUG; the first idle
        // interval after load still logs at INFO.
        if idle && was_idle {
            debug!(
                prompt_throughput,
                generation_throughput,
                running,
                waiting,
                cache_usage,
                prefix_cache_hit_rate,
                "engine statistics"
            );
        } else {
            info!(
                prompt_throughput,
                generation_throughput,
                running,
                waiting,
                cache_usage,
                prefix_cache_hit_rate,
                "engine statistics"
            );
        }
        was_idle = idle;
    }
}
