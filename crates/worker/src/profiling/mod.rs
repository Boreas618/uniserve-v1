//! Observability of a rank: NVTX ranges, selected kernels and profiler
//! windows.

pub mod kernels;
pub mod profiler;
mod ranges;

pub(crate) use ranges::range;

/// A boolean environment flag (`1`, `true`, `yes`, `on`).
pub fn flag(value: Option<&str>) -> bool {
    value.is_some_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

/// Whether device timing events are recorded for completions: under a
/// profiler window, NVTX ranges or an external CUDA profiler.
pub fn timing_events_enabled() -> bool {
    std::env::var("UNISERVE_TORCH_PROFILER_DIR").is_ok_and(|value| !value.is_empty())
        || ["UNISERVE_NVTX", "UNISERVE_CUDA_PROFILER"]
            .iter()
            .any(|name| flag(std::env::var(name).ok().as_deref()))
}
