//! NVTX ranges of the `uniserve.worker` domain, switched at run time.
//!
//! A timeline (`docs/profiling.md`) shows each host phase of a batch and each storage job beside
//! the kernels they issue, and when the scheduler applies a batch's output and the control thread
//! writes its events to the daemon. These ranges mark what a thread of the engine does on the
//! host; the time a module of the model takes on the device is measured by the range markers
//! ([`super::markers`]). Ranges are cheap and switched: the switch starts from `UNISERVE_NVTX`
//! (`1`, `true`, `yes` or `on`) and a timeline session changes it while the process serves
//! ([`set_enabled`]).
//!
//! NVTX reaches a tool through the injection library named in `NVTX_INJECTION64_PATH` when the
//! process makes its first NVTX call: the process names CUPTI there as it starts
//! ([`super::counters::inject`]) unless a profiler (Nsight Systems) named its own, so a timeline
//! session's CUPTI records the ranges.

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
pub(crate) fn enabled() -> bool {
    switch().load(Ordering::Acquire)
}

/// Emits ranges from now on, or stops; returns whether they were emitted.
pub(crate) fn set_enabled(value: bool) -> bool {
    switch().swap(value, Ordering::AcqRel)
}

/// Marks a synchronous phase on its executing thread: a turn of a worker's job, which prepares a
/// batch's inputs and launches it, captures a bucket, rebinds a generation or runs a control
/// operation; a scheduler's application of a batch's output or step; the control thread's write
/// of one request's event. The range closes when the returned guard drops.
///
/// `batch` is the payload: the batch the phase belongs to, so a timeline can line the ranges up
/// with the scheduler's steps, or for an event the request it is of; a phase with neither passes
/// `None`.
pub(crate) fn range(name: &'static CStr, batch: Option<u64>) -> Option<LocalRange<'static>> {
    static DOMAIN: OnceLock<Domain> = OnceLock::new();

    if !enabled() {
        return None;
    }
    let domain = DOMAIN.get_or_init(|| Domain::new(c"uniserve.worker"));
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
