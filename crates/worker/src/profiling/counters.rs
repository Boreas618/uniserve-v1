//! The always-on counters CUPTI gives a serving rank: the watch of the
//! driver calls that make its threads wait for the device
//! ([`super::blocking`]) and the record of its streams' device time
//! ([`cuda::cupti::activity`], reported as `ForwardStats::stream_us`).
//!
//! A rank subscribes the process to CUPTI as it starts ([`subscribe`]),
//! before loading modules and capturing graphs so that collectors observe
//! their CUDA resource lifetimes, and makes whole-graph records its standing
//! graph tracing mode then: a graph launch counts as one interval of its
//! stream, and CUPTI instruments no graph node, which would not survive the
//! rank withdrawing the storage its graphs address
//! ([`cuda::cupti::timeline::trace_graphs_whole`]). The counters
//! start once the rank serves ([`Counters::start`]) and stay on: they share
//! the subscription with the session collectors. A profiler that holds
//! CUPTI's one subscription (Nsight Systems) leaves the rank without them.

use cuda::cupti::activity::StreamActivity;
use cuda::cupti::blocking::BlockingWatch;

/// Subscribe the process to CUPTI and trace its graphs whole; a failure
/// (another CUPTI client holds the subscription) is logged, and the rank
/// serves without CUPTI. Returns whether graphs are traced whole, the mode
/// the rank's graph owners instantiate their executables in.
pub(crate) fn subscribe() -> bool {
    if let Err(error) = cuda::cupti::subscriber::subscribe() {
        tracing::warn!(
            "CUPTI is not subscribed, and no CUPTI counter or profiling session is \
             available: {error}"
        );
        return false;
    }
    tracing::info!("CUPTI is subscribed ({})", cuda::cupti::library());
    match cuda::cupti::timeline::trace_graphs_whole() {
        Ok(()) => true,
        Err(error) => {
            tracing::warn!("CUPTI does not trace graphs whole; it records their nodes: {error}");
            false
        }
    }
}

/// The rank's counters. The activity record drops before the watch.
pub(crate) struct Counters {
    /// `None` when CUPTI refused the activity records.
    _activity: Option<StreamActivity>,
    _watch: BlockingWatch,
}

impl Counters {
    /// Start the counters, recording the device time of `streams` by name
    /// (`device`, `copy`); `None` when the process is not subscribed.
    pub(crate) fn start(streams: &[(&'static str, cuda::Stream)]) -> Option<Self> {
        if !cuda::cupti::subscriber::subscribed() {
            return None;
        }
        let watch = BlockingWatch::start()
            .inspect_err(|error| {
                tracing::warn!("the blocking driver calls are not counted: {error}")
            })
            .ok()?;
        let activity = StreamActivity::start(streams.iter().map(|(name, stream)| (*name, stream)))
            .inspect_err(|error| {
                tracing::warn!("the device time of the rank's streams is not recorded: {error}");
            })
            .ok();
        Some(Self {
            _activity: activity,
            _watch: watch,
        })
    }
}
