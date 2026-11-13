//! The always-on counters CUPTI gives a serving process: the watch of the driver calls that make
//! its threads wait for the device and the record of its streams' device time, reported as
//! `ForwardStats::blocking_calls` and `ForwardStats::stream_us` ([`super::forward_stats`]).
//!
//! The process subscribes to CUPTI as it starts ([`subscribe`]), before it loads a module or
//! captures a graph, so that collectors observe their CUDA resources' lifetimes. The counters
//! start once the process's ranks serve ([`Counters::start`]) and stay on: they share the
//! subscription with the session collectors. A profiler that holds CUPTI's one subscription
//! (Nsight Systems) leaves the process without them.

use std::collections::BTreeMap;
use std::sync::Mutex;

use platform::Stream;
use platform::cupti::activity::StreamActivity;
use platform::cupti::blocking::BlockingWatch;

use service::lock;

/// Names the CUPTI of the Python environment `VIRTUAL_ENV` states ([`platform::cupti::packaged`])
/// as the process's NVTX injection library, unless the environment names one already. NVTX reaches
/// CUPTI only through that library, and the process then loads the same one
/// ([`platform::cupti::library`]).
///
/// Called once at the process's start, before it starts any thread.
pub(crate) fn inject() {
    if std::env::var_os("NVTX_INJECTION64_PATH").is_some() {
        return;
    }
    let Some(library) = platform::cupti::packaged() else {
        return;
    };
    // SAFETY: the process's entry calls this before it starts any thread, so no other thread reads
    // the environment.
    unsafe { std::env::set_var("NVTX_INJECTION64_PATH", library) };
}

/// Subscribe the process to CUPTI; a failure (another CUPTI client holds the subscription, or the
/// process loaded a CUPTI release the subscription refuses) is logged, and the process serves
/// without CUPTI.
pub(crate) fn subscribe() {
    match platform::cupti::subscriber::subscribe() {
        Ok(()) => tracing::info!("CUPTI is subscribed ({})", platform::cupti::library()),
        Err(error) => tracing::warn!(
            "CUPTI is not subscribed, and no CUPTI counter or profiling session is \
             available: {error}"
        ),
    }
}

/// The process's counters. The activity record drops before the watch.
pub(crate) struct Counters {
    /// `None` when CUPTI refused the activity records.
    _activity: Option<StreamActivity>,
    _watch: BlockingWatch,
}

impl Counters {
    /// Start the counters, recording the device time of `streams` by name (`device`, `copy`), a
    /// graph's launch as one record unless the deployment keeps PC sampling enabled, which refuses
    /// that record; `None` when the process is not subscribed.
    pub(crate) fn start(streams: &[(&'static str, Stream)], whole_graphs: bool) -> Option<Self> {
        if !platform::cupti::subscriber::subscribed() {
            return None;
        }
        let watch = BlockingWatch::start()
            .inspect_err(|error| {
                tracing::warn!("the blocking driver calls are not counted: {error}");
            })
            .ok()?;
        let activity = StreamActivity::start(
            streams.iter().map(|(name, stream)| (*name, stream)),
            whole_graphs,
        )
        .inspect_err(|error| {
            tracing::warn!("the device time of the process's streams is not recorded: {error}");
        })
        .ok();
        if activity.is_some() && whole_graphs {
            super::trace_whole_graphs();
        }
        Some(Self {
            _activity: activity,
            _watch: watch,
        })
    }
}

/// The driver calls that made the process's serving threads wait for the device, under
/// `{thread}/{call}`, since the last take. They travel with the next batch result.
static BLOCKING: Mutex<BTreeMap<String, u64>> = Mutex::new(BTreeMap::new());

/// Counts `calls` (by call name) that `thread` made.
pub(crate) fn add_blocking(thread: &str, calls: &[(&'static str, u64)]) {
    if calls.is_empty() {
        return;
    }
    let mut counts = lock(&BLOCKING);
    for &(call, count) in calls {
        *counts.entry(format!("{thread}/{call}")).or_default() += count;
    }
}

/// The counts since the last take.
pub(crate) fn take_blocking() -> BTreeMap<String, u64> {
    std::mem::take(&mut *lock(&BLOCKING))
}
