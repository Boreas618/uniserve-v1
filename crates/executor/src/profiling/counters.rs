//! The always-on counters CUPTI gives a serving process: the watch of the driver calls that make its
//! threads wait for the device and the record of its streams' device time, reported as
//! `ForwardStats::blocking_calls` and `ForwardStats::stream_us`.
//!
//! The process subscribes to CUPTI as it starts ([`subscribe`]), before it loads a module or captures
//! a graph, so that collectors observe their CUDA resources' lifetimes. The counters start once the
//! process's ranks serve ([`Counters::start`]) and stay on: they share the subscription with the
//! session collectors. A profiler that holds CUPTI's one subscription (Nsight Systems) leaves the
//! process without them.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use platform::Stream;
use platform::cupti::activity::StreamActivity;
use platform::cupti::blocking::BlockingWatch;

use crate::lock;

/// Why a collector is unavailable on a process that loaded the CUPTI [`cupti_serves`] refuses.
pub(crate) const REFUSED: &str = "the process loaded the CUDA toolkit's CUPTI 2025.3.0, which \
     stalls a process that replays CUDA graphs while kernels are recorded; name the \
     `nvidia-cuda-cupti` package's library in NVTX_INJECTION64_PATH";

/// Names the CUPTI of the Python environment `VIRTUAL_ENV` states as the process's NVTX injection
/// library, unless the environment names one already. NVTX reaches CUPTI only through that library,
/// and the process then loads the same one ([`platform::cupti::library`]).
///
/// Called once at the process's start, before it starts any thread.
pub fn inject() {
    if std::env::var_os("NVTX_INJECTION64_PATH").is_some() {
        return;
    }
    let Some(environment) = std::env::var_os("VIRTUAL_ENV") else {
        return;
    };
    let Some(library) = environment_cupti(std::path::Path::new(&environment)) else {
        return;
    };
    // SAFETY: the process's entry calls this before it starts any thread, so no other thread reads
    // the environment.
    unsafe { std::env::set_var("NVTX_INJECTION64_PATH", library) };
}

/// `libcupti.so.N` of the NVIDIA CUDA packages of the Python environment at `prefix`:
/// `<prefix>/lib/pythonX.Y/site-packages/nvidia/cuN/lib`.
fn environment_cupti(prefix: &std::path::Path) -> Option<std::path::PathBuf> {
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

/// Whether the CUPTI this process loads may record it while it serves.
///
/// The toolkit's CUPTI 2025.3.0 stalled a worker that replayed CUDA graphs while kernels were
/// recorded, within a minute of load, and an always-on counter would do exactly that. The library's
/// real file names its release, whichever link the loader followed.
pub(crate) fn cupti_serves() -> bool {
    static SERVES: OnceLock<bool> = OnceLock::new();
    *SERVES.get_or_init(|| {
        if platform::cupti::version().is_err() {
            return false;
        }
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
        !maps.contains("libcupti.so.2025.3.0")
    })
}

/// Subscribe the process to CUPTI; a failure (another CUPTI client holds the subscription, or the
/// process loaded a CUPTI that stalls it) is logged, and the process serves without CUPTI.
pub fn subscribe() {
    if !cupti_serves() {
        tracing::warn!("no CUPTI counter or profiling session is available: {REFUSED}");
        return;
    }
    match platform::cupti::subscriber::subscribe() {
        Ok(()) => tracing::info!("CUPTI is subscribed ({})", platform::cupti::library()),
        Err(error) => tracing::warn!(
            "CUPTI is not subscribed, and no CUPTI counter or profiling session is \
             available: {error}"
        ),
    }
}

/// The process's counters. The activity record drops before the watch.
pub struct Counters {
    /// `None` when CUPTI refused the activity records.
    _activity: Option<StreamActivity>,
    _watch: BlockingWatch,
}

impl Counters {
    /// Start the counters, recording the device time of `streams` by name (`device`, `copy`); `None`
    /// when the process is not subscribed.
    pub fn start(streams: &[(&'static str, Stream)]) -> Option<Self> {
        if !platform::cupti::subscriber::subscribed() {
            return None;
        }
        let watch = BlockingWatch::start()
            .inspect_err(|error| {
                tracing::warn!("the blocking driver calls are not counted: {error}");
            })
            .ok()?;
        let activity = StreamActivity::start(streams.iter().map(|(name, stream)| (*name, stream)))
            .inspect_err(|error| {
                tracing::warn!("the device time of the process's streams is not recorded: {error}");
            })
            .ok();
        Some(Self {
            _activity: activity,
            _watch: watch,
        })
    }
}

/// The driver calls that made the process's serving threads wait for the device, under
/// `{thread}/{call}`, since the last take. They travel with the next batch result.
static BLOCKING: Mutex<BTreeMap<String, u64>> = Mutex::new(BTreeMap::new());

/// Count `calls` (by call name) that `thread` made.
pub fn add_blocking(thread: &str, calls: &[(&'static str, u64)]) {
    if calls.is_empty() {
        return;
    }
    let mut counts = lock(&BLOCKING);
    for &(call, count) in calls {
        *counts.entry(format!("{thread}/{call}")).or_default() += count;
    }
}

/// The counts since the last take.
pub(crate) fn take_blocking() -> BTreeMap<String, u64> {
    std::mem::take(&mut *lock(&BLOCKING))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_environments_cupti_is_found_beside_its_packages() {
        let prefix = tempfile::tempdir().unwrap();
        let lib = prefix
            .path()
            .join("lib/python3.12/site-packages/nvidia/cu13/lib");
        std::fs::create_dir_all(&lib).unwrap();
        assert_eq!(environment_cupti(prefix.path()), None);
        std::fs::write(lib.join("libcupti.so.13"), b"").unwrap();
        assert_eq!(
            environment_cupti(prefix.path()),
            Some(lib.join("libcupti.so.13"))
        );
    }
}
