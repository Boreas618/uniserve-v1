//! Observability of an engine: NVTX ranges, CUPTI's always-on counters, the range markers that
//! time the model's modules, and profiling sessions (`docs/profiling.md`).
//!
//! - [`ranges`]: the `uniserve.worker` NVTX ranges of the worker's host phases, switched at run
//!   time.
//! - [`counters`]: the process's CUPTI subscription, and the watch of its threads' waits for the
//!   device and the record of its streams' device time.
//! - [`markers`]: the range markers at the entry and exit of the model's profiler ranges
//!   (`nn::profiler::range`), the process's `ops::Profiler`, exported as per-step module
//!   times.
//! - [`pc_sampling`]: CUPTI's PC sampling, enabled for a rank's life when the deployment profiles
//!   its CUDA graphs' kernels.
//! - [`session`]: a profiling session's window on one rank, its collectors and its file.
//!
//! ## What a worker and its runners call
//!
//! Nothing here knows the worker: each call takes the plain values it needs.
//!
//! - A runner that captures a graph gives its context the process's profiler:
//!   `context.set_profiler(markers::profiler(&stream))`.
//! - The worker keeps one `Option<Session>` per rank and performs each [`session::Change`] of
//!   its queue between two batches with [`session::Session::change`].
//! - At a step's begin it tells the session ([`session::Session::begin`]) and launches the step's
//!   begin marker ([`markers::Markers::begin_step`]); right behind the step's last launch it
//!   launches the end marker ([`markers::Markers::end_step`]); once the step completed it tells
//!   the session ([`session::Session::end`]). When the session returns `true` the worker waits
//!   for the rank's streams and calls [`session::Session::apply`].
//! - With each batch result it sends what [`forward_stats`] returns
//!   (`BatchOutput::forward_stats`).
//! - A graph compares [`graph_renewals`] with the number its executable was instantiated at, and
//!   the owner of the storage graphs address calls [`storage_remapped`] after it unmaps or maps
//!   that storage again.

pub(crate) mod artifact;
pub(crate) mod collectors;
pub(crate) mod counters;
pub(crate) mod markers;
pub(crate) mod pc_sampling;
pub(crate) mod ranges;
pub(crate) mod session;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use platform::DeviceId;
use platform::cupti::blocking::Calls;

/// Whether CUPTI records whole graphs now, which the executables replayed must be instantiated
/// for.
static WHOLE_GRAPHS: AtomicBool = AtomicBool::new(false);

/// Whether it does so between timelines: the counters' record of a graph's launch.
static WHOLE_BY_DEFAULT: AtomicBool = AtomicBool::new(false);

/// How many times every graph executable of the process was declared stale.
static GRAPH_RENEWALS: AtomicU64 = AtomicU64::new(0);

/// A timeline changes how CUPTI records a graph's launch: an executable is instantiated in its
/// tracing mode, and running a whole-graph executable in node mode leaks CUPTI's timestamp slots,
/// so every graph replaces its executable before its next replay.
pub(crate) fn set_graph_trace(whole: bool) {
    if WHOLE_GRAPHS.swap(whole, Ordering::AcqRel) != whole {
        GRAPH_RENEWALS.fetch_add(1, Ordering::AcqRel);
    }
}

/// The counters record a graph's launch whole for as long as the process serves.
fn trace_whole_graphs() {
    WHOLE_BY_DEFAULT.store(true, Ordering::Release);
    set_graph_trace(true);
}

/// How CUPTI records a graph's launch with no timeline running.
pub(crate) fn graphs_whole_by_default() -> bool {
    WHOLE_BY_DEFAULT.load(Ordering::Acquire)
}

/// Storage graphs address was unmapped or mapped again.
///
/// CUPTI's per-node instrumentation of an executable does not survive that: with
/// `nvidia-cuda-cupti` 13.0.85 and driver 580.82.07 on GB200, the executable's next launch faults
/// inside CUPTI in `cuGraphLaunch`, while one instantiated after the remapping replays correctly.
/// So a process subscribed to CUPTI replaces each executable before its next replay.
pub(crate) fn storage_remapped() {
    if platform::cupti::subscriber::subscribed() {
        GRAPH_RENEWALS.fetch_add(1, Ordering::AcqRel);
    }
}

/// The number a graph compares with the one its executable was instantiated at to know the
/// executable is current: it grows whenever every executable of the process became stale.
pub(crate) fn graph_renewals() -> u64 {
    GRAPH_RENEWALS.load(Ordering::Acquire)
}

/// The runtime mode of a forward that replayed a captured graph, as the forward counts name it
/// (`ForwardStats::mode_counts`, `cuda_graph_runtime_mode_counts`).
const GRAPH_REPLAY: &str = "graph_replay";

/// The runtime mode of a forward run eagerly, with no graph.
const EAGER: &str = "eager";

/// What the rank of `device` observed since its last batch result, for the result that leaves
/// now, or `None` when it observed nothing:
///
/// - the forwards of the batch, `forwards` being the graphs it replayed and the forwards it ran
///   eagerly: each counted under its runtime mode (`mode_counts`), a replay also as a graph
///   dispatch (`cuda_graph_runtime_mode_counts`, `cuda_graph_replays`), and the step's time,
///   `step_us`, under the batch's mode (`mode_us`): a replay's when it replayed a graph, the
///   eager mode's when it ran its forwards eagerly only;
/// - the driver calls that made a serving thread wait for the device (`blocking_calls`): the
///   calling worker's since `blocking`, its count when its last result left, which is advanced
///   to now, and what other threads added ([`counters::add_blocking`]);
/// - the device time of the process's observed streams (`stream_us`);
/// - the per-step module times the device's range markers measured, as histograms by
///   `{stage}/{class}`, and the stamps its reader lost (`fence_us`, `fence_records_lost`);
/// - the progress of `session`, the rank's profiling session (`profile`).
///
/// Called by the rank's worker, on its own thread, once per batch result. It never waits for
/// the device.
pub(crate) fn forward_stats(
    device: DeviceId,
    session: Option<&mut session::Session>,
    blocking: &mut Calls,
    forwards: (u64, u64),
    step_us: u64,
) -> Option<proto::report::ForwardStats> {
    let mut stats = proto::report::ForwardStats::default();

    let (replays, eager) = forwards;
    for (mode, count) in [(GRAPH_REPLAY, replays), (EAGER, eager)] {
        if count > 0 {
            stats.mode_counts.insert(mode.to_owned(), count);
        }
    }
    if replays > 0 {
        stats
            .cuda_graph_runtime_mode_counts
            .insert(GRAPH_REPLAY.to_owned(), replays);
        stats.cuda_graph_replays = replays;
    }
    if replays > 0 || eager > 0 {
        let mode = if replays > 0 { GRAPH_REPLAY } else { EAGER };
        stats.mode_us.insert(mode.to_owned(), step_us);
    }

    let current = Calls::current();
    counters::add_blocking("device", &current.since(blocking));
    *blocking = current;
    stats.blocking_calls = counters::take_blocking();

    let activity = platform::cupti::activity::take();
    if activity.dropped_records > 0 {
        tracing::warn!(
            "CUPTI dropped {} activity records; the streams' device time misses their operations",
            activity.dropped_records
        );
    }
    for stream in &activity.streams {
        for (state, us) in [
            ("busy", stream.busy_us),
            ("gap", stream.gap_us),
            ("late", stream.late_us),
        ] {
            if us > 0 {
                *stats
                    .stream_us
                    .entry(format!("{}/{state}", stream.name))
                    .or_default() += us;
            }
        }
    }

    if let Some(markers) = markers::Markers::installed() {
        markers.read_device(device, false);
        (stats.fence_us, stats.fence_records_lost) = markers.take(device);
    }
    if let Some(session) = session {
        stats.profile = session.progress();
    }

    let observed = !stats.mode_counts.is_empty()
        || !stats.blocking_calls.is_empty()
        || !stats.stream_us.is_empty()
        || !stats.fence_us.is_empty()
        || stats.fence_records_lost > 0
        || !stats.profile.is_empty();
    observed.then_some(stats)
}

#[cfg(test)]
mod tests {
    use platform::DeviceId;
    use platform::cupti::blocking::Calls;

    use super::forward_stats;

    /// The modes and time of a batch's forwards as one result reports them.
    fn modes(forwards: (u64, u64), step_us: u64) -> Option<proto::report::ForwardStats> {
        forward_stats(DeviceId(0), None, &mut Calls::default(), forwards, step_us)
    }

    #[test]
    fn a_batch_s_forwards_are_counted_by_how_they_ran() {
        // Three graphs replayed and one forward run eagerly: the step is a replayed one.
        let stats = modes((3, 1), 500).unwrap();
        let counts: Vec<(&str, u64)> = stats
            .mode_counts
            .iter()
            .map(|(mode, count)| (mode.as_str(), *count))
            .collect();
        assert_eq!(counts, [("eager", 1), ("graph_replay", 3)]);
        assert_eq!(
            stats.cuda_graph_runtime_mode_counts.get("graph_replay"),
            Some(&3)
        );
        assert_eq!(stats.cuda_graph_replays, 3);
        assert_eq!(stats.mode_us.get("graph_replay"), Some(&500));
        assert_eq!(stats.mode_us.get("eager"), None);

        // Forwards run eagerly only: no graph dispatch, and the step's time is the eager mode's.
        let stats = modes((0, 2), 70).unwrap();
        assert_eq!(stats.mode_counts.get("eager"), Some(&2));
        assert!(stats.cuda_graph_runtime_mode_counts.is_empty());
        assert_eq!(stats.cuda_graph_replays, 0);
        assert_eq!(stats.mode_us.get("eager"), Some(&70));
    }
}
