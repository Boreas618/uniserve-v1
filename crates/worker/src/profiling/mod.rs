//! Observability of the main process: NVTX ranges, CUPTI's always-on counters, fence probes and
//! profiling sessions (`docs/profiling.md`).
//!
//! - [`ranges`]: the `uniserve.worker` NVTX ranges, switched at run time.
//! - [`counters`]: the process's CUPTI subscription, and the watch of its threads' waits for the
//!   device and the record of its streams' device time.
//! - [`fences`]: the fence probes at the model's components, exported as per-step component
//!   durations.
//! - [`pc_sampling`]: CUPTI's PC sampling, enabled for a rank's life when the deployment profiles
//!   its CUDA graphs' kernels.
//! - [`session`]: a profiling session's window on one rank, its collectors and its file.

pub(crate) mod artifact;
pub(crate) mod collectors;
pub mod counters;
pub mod fences;
pub mod pc_sampling;
pub(crate) mod ranges;
pub mod session;

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

pub use ranges::{enabled, range, set_enabled};

/// Whether CUPTI records whole graphs now, which the executables replayed must be instantiated for.
static WHOLE_GRAPHS: AtomicBool = AtomicBool::new(false);

/// Whether it does so between timelines: the counters' record of a graph's launch.
static WHOLE_BY_DEFAULT: AtomicBool = AtomicBool::new(false);

/// How many times every graph executable of the process was declared stale.
static GRAPH_RENEWALS: AtomicU64 = AtomicU64::new(0);

/// A timeline changes how CUPTI records a graph's launch: an executable is instantiated in its
/// tracing mode, and running a whole-graph executable in node mode leaks CUPTI's timestamp slots, so
/// every graph replaces its executable before its next replay.
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
/// inside CUPTI in `cuGraphLaunch`, while one instantiated after the remapping replays correctly. So
/// a process subscribed to CUPTI replaces each executable before its next replay.
pub(crate) fn storage_remapped() {
    if platform::cupti::subscriber::subscribed() {
        GRAPH_RENEWALS.fetch_add(1, Ordering::AcqRel);
    }
}

/// The number an executable compares with its own to know it is current.
pub(crate) fn graph_renewals() -> u64 {
    GRAPH_RENEWALS.load(Ordering::Acquire)
}
