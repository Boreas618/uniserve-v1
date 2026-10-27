//! Observability of a rank: NVTX ranges, selected kernels, the torch
//! profiler window, CUPTI's always-on counters, fence probes and profiling
//! sessions.
//!
//! - [`counters`]: the process's CUPTI subscription, and the watch of the
//!   rank's waits for the device and record of its streams' device time.
//! - [`fences`]: the fence probes at the model's components, always on,
//!   exported as per-step component durations.
//! - [`session`]: a profiling session's window, collectors and file.

pub(crate) mod blocking;
pub(crate) mod counters;
pub(crate) mod fences;
pub(crate) mod kernels;
pub(crate) mod profiler;
pub(crate) mod ranges;
pub(crate) mod session;

pub(crate) use ranges::range;

/// A boolean environment flag (`1`, `true`, `yes`, `on`).
pub(crate) fn flag(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}
