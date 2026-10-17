//! Observability of a rank: NVTX ranges, selected kernels and profiler
//! windows.

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
