//! NVTX ranges of the `uniserve.worker` domain, switched at run time.
//!
//! The switch starts as `UNISERVE_NVTX` states (`1`, `true`, `yes` or `on`)
//! and changes while the rank serves ([`set_enabled`]): a timeline session
//! turns it on for its span and restores it after. The Python library's
//! ranges follow the same switch (`uniserve.profiling.set_nvtx`), which the
//! rank sets with this one. No language callback or device operation is
//! involved.
//!
//! NVTX reaches a tool through the injection library the process names in
//! `NVTX_INJECTION64_PATH` when it makes its first NVTX call: the worker names
//! CUPTI ([`inject`]) unless a profiler (Nsight Systems) named its own, so a
//! timeline session's CUPTI records the ranges.

use std::ffi::CStr;
use std::path::Path;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use nvtx::{Domain, domain::LocalRange};

/// Whether ranges are emitted; set from `UNISERVE_NVTX` at the first range.
static ENABLED: OnceLock<AtomicBool> = OnceLock::new();

fn switch() -> &'static AtomicBool {
    ENABLED.get_or_init(|| {
        AtomicBool::new(super::flag(std::env::var("UNISERVE_NVTX").ok().as_deref()))
    })
}

/// Whether ranges are emitted now.
pub(crate) fn enabled() -> bool {
    switch().load(Ordering::Acquire)
}

/// Emit ranges from now on, or stop; returns whether they were emitted.
pub(crate) fn set_enabled(enabled: bool) -> bool {
    switch().swap(enabled, Ordering::AcqRel)
}

/// Mark a synchronous phase on its executing thread. Batch phases carry the
/// scheduler's batch ID; shared resource reaping has no single batch owner.
pub(crate) fn range(name: &'static CStr, batch_id: Option<u64>) -> Option<LocalRange<'static>> {
    static DOMAIN: OnceLock<Domain> = OnceLock::new();

    if !enabled() {
        return None;
    }
    let domain = DOMAIN.get_or_init(|| Domain::new(c"uniserve.worker"));
    let mut attributes = domain.event_attributes_builder().message(name);
    if let Some(batch_id) = batch_id {
        attributes = attributes.payload(batch_id);
    }
    Some(domain.local_range(attributes.build()))
}

/// Name the CUPTI of the Python environment at `python` (the environment's
/// interpreter) as the process's NVTX injection library, unless the
/// environment names one already. Called once at the worker's start, before
/// any thread or NVTX call.
pub(crate) fn inject(python: &Path) {
    if std::env::var_os("NVTX_INJECTION64_PATH").is_some() {
        return;
    }
    let Some(library) = environment_cupti(python) else {
        tracing::warn!(
            "no CUPTI in the Python environment of {}: NVTX ranges reach no timeline session",
            python.display()
        );
        return;
    };
    // SAFETY: the worker's entry calls this before it starts any thread, so
    // no other thread reads the environment.
    unsafe { std::env::set_var("NVTX_INJECTION64_PATH", library) };
}

/// `libcupti.so.N` of the NVIDIA CUDA packages of the Python environment
/// whose interpreter is `python`: `<prefix>/lib/pythonX.Y/site-packages/
/// nvidia/cuN/lib`, the directory of the CUDA runtime PyTorch loads there.
fn environment_cupti(python: &Path) -> Option<std::path::PathBuf> {
    let prefix = python.parent()?.parent()?;
    for entry in std::fs::read_dir(prefix.join("lib")).ok()?.flatten() {
        let name = entry.file_name();
        if !name.to_string_lossy().starts_with("python") {
            continue;
        }
        let nvidia = entry.path().join("site-packages").join("nvidia");
        for package in std::fs::read_dir(&nvidia).into_iter().flatten().flatten() {
            if !package.file_name().to_string_lossy().starts_with("cu") {
                continue;
            }
            let lib = package.path().join("lib");
            for library in std::fs::read_dir(&lib).into_iter().flatten().flatten() {
                if library
                    .file_name()
                    .to_string_lossy()
                    .starts_with("libcupti.so.")
                {
                    return Some(library.path());
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_environments_cupti_is_found_beside_its_interpreter() {
        let prefix = tempfile::tempdir().unwrap();
        let bin = prefix.path().join("bin");
        let lib = prefix
            .path()
            .join("lib/python3.12/site-packages/nvidia/cu13/lib");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::create_dir_all(&lib).unwrap();
        assert_eq!(environment_cupti(&bin.join("python")), None);
        std::fs::write(lib.join("libcupti.so.13"), b"").unwrap();
        assert_eq!(
            environment_cupti(&bin.join("python")),
            Some(lib.join("libcupti.so.13"))
        );
    }
}
