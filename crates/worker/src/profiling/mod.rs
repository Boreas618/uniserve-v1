//! Observability of a rank: NVTX ranges, selected kernels, profiler
//! windows, and CUPTI's watch of its waits for the device and record of its
//! lanes' device time.

pub(crate) mod blocking;
pub(crate) mod cupti;
pub(crate) mod kernels;
pub(crate) mod profiler;
mod ranges;

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
