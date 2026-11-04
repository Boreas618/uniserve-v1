//! NVTX ranges of the `uniserve.executor` domain, switched at run time.
//!
//! The design's profiling story (`docs/profiling.md`) wants a timeline that shows each phase of a
//! batch and each storage job beside the kernels they issue. Ranges are cheap and switched: the
//! switch starts from `UNISERVE_NVTX` (`1`, `true`, `yes` or `on`) and a timeline session changes
//! it while the process serves ([`set_enabled`]).
//!
//! NVTX reaches a tool through the injection library named in `NVTX_INJECTION64_PATH` when the
//! process makes its first NVTX call; an administrator naming a tool there (Nsight Systems) is
//! enough, and the executor starts no injection library itself because the main process has no
//! Python environment to take CUPTI from.

use std::ffi::CStr;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use nvtx::{Domain, domain::LocalRange};

/// Whether ranges are emitted; set from `UNISERVE_NVTX` at the first range.
static ENABLED: OnceLock<AtomicBool> = OnceLock::new();

fn switch() -> &'static AtomicBool {
    ENABLED.get_or_init(|| AtomicBool::new(flag(std::env::var("UNISERVE_NVTX").ok().as_deref())))
}

/// Whether a switch value turns ranges on.
pub(crate) fn flag(value: Option<&str>) -> bool {
    let Some(value) = value else {
        return false;
    };
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Whether ranges are emitted now.
pub fn enabled() -> bool {
    switch().load(Ordering::Acquire)
}

/// Emit ranges from now on, or stop; returns whether they were emitted.
pub fn set_enabled(value: bool) -> bool {
    switch().swap(value, Ordering::AcqRel)
}

/// Marks a synchronous phase on its executing thread: a batch being staged, a capture, a rebind,
/// a storage job. The range closes when the returned guard drops.
///
/// `batch` is the batch the phase belongs to, so a timeline can line the ranges up with the
/// scheduler's steps; a phase with no single batch owner passes `None`.
pub fn range(name: &'static CStr, batch: Option<u64>) -> Option<LocalRange<'static>> {
    static DOMAIN: OnceLock<Domain> = OnceLock::new();

    if !enabled() {
        return None;
    }
    let domain = DOMAIN.get_or_init(|| Domain::new(c"uniserve.executor"));
    let mut attributes = domain.event_attributes_builder().message(name);
    if let Some(batch) = batch {
        attributes = attributes.payload(batch);
    }
    Some(domain.local_range(attributes.build()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_switch_reads_its_spellings() {
        assert!(flag(Some("1")));
        assert!(flag(Some("true")));
        assert!(flag(Some("YES")));
        assert!(flag(Some(" on ")));
        assert!(!flag(Some("0")));
        assert!(!flag(Some("off")));
        assert!(!flag(None));
    }
}
