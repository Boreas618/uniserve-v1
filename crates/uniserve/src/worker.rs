//! `uniserve worker`: the process of one worker.
//!
//! The `uniserve` executable is installed into a Python environment's
//! `bin/`, beside the environment's interpreter, and its `worker` command
//! takes the arguments a node's daemon starts a worker with. It
//! does not link the worker runtime. It finds its environment from its own
//! path, asks the
//! environment's interpreter where three files are, loads them and calls the
//! worker library's `worker_main` on this, the process's main thread:
//!
//! 1. the interpreter's shared `libpython`, into the global scope, so that
//!    the interpreter the library hosts and every extension module it
//!    imports resolve the C API from it;
//! 2. `libtvm_ffi` of the environment's `tvm_ffi` package, into the global
//!    scope, which the library and `tvm_ffi`'s extension module then share;
//! 3. the worker library `uniserve/_worker`, exactly the file Python
//!    resolves `uniserve._worker` to, so the process holds one instance of
//!    the runtime when Python loads that library again (the video rules it
//!    exports).
//!
//! The process exits with the status `worker_main` returns, without
//! running exit handlers or Python finalization: an aborted worker retains
//! device resources, and finalizers could wait on a failed peer.

use std::ffi::{CString, OsStr, OsString, c_char, c_int};
use std::io::Write;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_LOCAL, RTLD_NOW};

/// The type of the worker library's entry (`worker::worker_main`).
type WorkerMain = unsafe extern "C" fn(c_int, *const *const c_char, *const c_char) -> c_int;

/// What the environment's interpreter prints, NUL-separated: the path of
/// its shared libpython (empty for an interpreter built without one), of
/// libtvm_ffi, and of the worker library. Resolving a module's spec imports
/// its parent package, `uniserve`, which imports nothing; `-P` keeps the
/// working directory off `sys.path`, as it is off the path of the
/// interpreter the library hosts.
const LOCATE: &str = r#"
import importlib.util
import os
import sys
import sysconfig


def package_directory(name):
    spec = importlib.util.find_spec(name)
    if spec is None or not spec.submodule_search_locations:
        sys.exit(f"the environment has no {name} package")
    return spec.submodule_search_locations[0]


libpython = ""
soname = sysconfig.get_config_var("INSTSONAME")
if sysconfig.get_config_var("Py_ENABLE_SHARED") and soname:
    for directory in (
        os.path.join(sys.base_prefix, "lib"),
        sysconfig.get_config_var("LIBDIR"),
    ):
        if directory and os.path.isfile(os.path.join(directory, soname)):
            libpython = os.path.join(directory, soname)
            break

package_directory("uniserve")
worker = importlib.util.find_spec("uniserve._worker")
if worker is None or worker.origin is None:
    sys.exit("the worker library uniserve._worker is not installed")
libtvm_ffi = os.path.join(package_directory("tvm_ffi"), "lib", "libtvm_ffi.so")
sys.stdout.write("\0".join((libpython, libtvm_ffi, worker.origin)))
"#;

/// Runs one worker with `arguments`, the worker's command line after
/// `worker`, and exits the process with its status.
pub(crate) fn run(arguments: Vec<OsString>) -> ! {
    let status = start(arguments).unwrap_or_else(|message| {
        eprintln!("uniserve worker: error: {message}");
        1
    });
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    // SAFETY: `_exit` ends the process without running any handler.
    unsafe { libc::_exit(status) }
}

/// Raise the soft limit on open descriptors to the hard limit: a worker
/// holds the descriptor of every retained storage it maps, one per large
/// weight of its model.
fn raise_descriptor_limit() -> Result<(), String> {
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: getrlimit fills the struct it is given.
    if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) } != 0 {
        return Err(format!(
            "reading the descriptor limit: {}",
            std::io::Error::last_os_error()
        ));
    }
    limit.rlim_cur = limit.rlim_max;
    // SAFETY: setrlimit reads the struct it is given.
    if unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &limit) } != 0 {
        return Err(format!(
            "raising the descriptor limit: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(())
}

/// The variable FlashInfer's JIT builds and PyTorch's C++ extensions take
/// their parallel compilation jobs from. Without it their ninja runs one
/// job per CPU plus two.
const COMPILE_JOBS_VARIABLE: &str = "MAX_JOBS";

/// The peak host memory of one compilation job: one `cicc` of FlashInfer's
/// fused-MoE build reaches 6.1 GiB.
const COMPILE_JOB_BYTES: u64 = 61 * (1 << 30) / 10;

/// Bound the parallel jobs of the kernels the worker compiles on first use,
/// unless its environment sets `MAX_JOBS` itself.
///
/// The ranks of the worker's launch on its host (`LOCAL_WORLD_SIZE`, one
/// when unset) start and compile together, so each takes an equal share of
/// half of the host's available CPUs and of the memory available as it
/// starts, at [`COMPILE_JOB_BYTES`] per job, and at least one job. The
/// variable is set before the interpreter starts, so every build of the
/// worker reads it.
fn bound_compile_jobs() -> Result<(), String> {
    if std::env::var_os(COMPILE_JOBS_VARIABLE).is_some() {
        return Ok(());
    }

    let ranks = match std::env::var("LOCAL_WORLD_SIZE") {
        Ok(value) => value
            .parse::<u64>()
            .ok()
            .filter(|&ranks| ranks > 0)
            .ok_or_else(|| format!("LOCAL_WORLD_SIZE={value} is not a positive rank count"))?,
        Err(std::env::VarError::NotPresent) => 1,
        Err(error) => return Err(format!("reading LOCAL_WORLD_SIZE: {error}")),
    };
    let cpus = std::thread::available_parallelism()
        .map_err(|error| format!("counting the available CPUs: {error}"))?
        .get() as u64;
    let available = available_memory()?;

    let by_cpus = cpus / 2 / ranks;
    let by_memory = available / (COMPILE_JOB_BYTES * ranks);
    let jobs = by_cpus.min(by_memory).max(1);
    eprintln!(
        "uniserve worker: kernel compilations run up to {jobs} parallel jobs \
         ({cpus} CPUs and {} GiB available for {ranks} ranks of this host; \
         set {COMPILE_JOBS_VARIABLE} to choose)",
        available >> 30
    );
    // SAFETY: no other thread exists yet (see `main`): the worker loads the
    // Python runtime and starts its threads after this returns.
    unsafe { std::env::set_var(COMPILE_JOBS_VARIABLE, jobs.to_string()) };
    Ok(())
}

/// The host memory available to new allocations without swapping, in bytes:
/// `MemAvailable` of `/proc/meminfo`.
fn available_memory() -> Result<u64, String> {
    let meminfo = std::fs::read_to_string("/proc/meminfo")
        .map_err(|error| format!("reading /proc/meminfo: {error}"))?;
    meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|value| value.trim().strip_suffix("kB"))
        .and_then(|kilobytes| kilobytes.trim().parse::<u64>().ok())
        .map(|kilobytes| kilobytes << 10)
        .ok_or_else(|| "/proc/meminfo states no MemAvailable in kB".to_owned())
}

/// Load the environment's runtime and run the worker; returns its exit
/// status, or why it could not start.
fn start(arguments: Vec<OsString>) -> Result<c_int, String> {
    raise_descriptor_limit()?;
    bound_compile_jobs()?;
    let python = environment_interpreter()?;
    let [libpython, libtvm_ffi, library] = locate(&python)?;
    if libpython.as_os_str().is_empty() {
        return Err(format!(
            "{} has no shared libpython; a worker hosts the interpreter, which needs a \
             Python built with --enable-shared",
            python.display()
        ));
    }

    // The libraries stay loaded until the process exits.
    let _libpython = open(&libpython, RTLD_NOW | RTLD_GLOBAL)?;
    let _libtvm_ffi = open(&libtvm_ffi, RTLD_NOW | RTLD_GLOBAL)?;
    let library = open(&library, RTLD_NOW | RTLD_LOCAL)?;
    // SAFETY: `worker_main` is the library's entry and has this type.
    let entry = unsafe { library.get::<WorkerMain>(b"worker_main\0") }
        .map_err(|error| format!("the worker library has no entry: {error}"))?;

    let arguments = arguments
        .iter()
        .map(|argument| CString::new(argument.as_bytes()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| "an argument holds a NUL byte".to_owned())?;
    let pointers: Vec<*const c_char> = arguments.iter().map(|argument| argument.as_ptr()).collect();
    let count = c_int::try_from(pointers.len()).map_err(|_| "too many arguments".to_owned())?;
    let interpreter = CString::new(python.as_os_str().as_bytes())
        .map_err(|_| format!("{} holds a NUL byte", python.display()))?;

    // SAFETY: the arguments and the interpreter path are NUL-terminated and
    // outlive the call, which this process makes once.
    Ok(unsafe { entry(count, pointers.as_ptr(), interpreter.as_ptr()) })
}

/// The interpreter of the environment whose `bin/` holds this executable.
fn environment_interpreter() -> Result<PathBuf, String> {
    let executable = std::env::current_exe()
        .map_err(|error| format!("this executable's path is unknown: {error}"))?;
    let bin = executable
        .parent()
        .ok_or_else(|| format!("{} has no directory", executable.display()))?;
    ["python", "python3"]
        .into_iter()
        .map(|name| bin.join(name))
        .find(|candidate| candidate.is_file())
        .ok_or_else(|| {
            format!(
                "{} is not installed in a Python environment: {} holds no python",
                executable.display(),
                bin.display()
            )
        })
}

/// The shared libpython, libtvm_ffi and worker library `python` resolves.
fn locate(python: &Path) -> Result<[PathBuf; 3], String> {
    let located = Command::new(python)
        .args(["-P", "-c", LOCATE])
        .stdin(Stdio::null())
        .stderr(Stdio::inherit())
        .output()
        .map_err(|error| format!("running {}: {error}", python.display()))?;
    if !located.status.success() {
        return Err(format!(
            "{} could not locate the worker runtime ({})",
            python.display(),
            located.status
        ));
    }
    let paths: Vec<PathBuf> = located
        .stdout
        .split(|&byte| byte == 0)
        .map(|path| PathBuf::from(OsStr::from_bytes(path)))
        .collect();
    <[PathBuf; 3]>::try_from(paths).map_err(|paths| {
        format!(
            "{} located {} paths instead of three",
            python.display(),
            paths.len()
        )
    })
}

/// Load the shared library at `path` with `flags`.
fn open(path: &Path, flags: c_int) -> Result<Library, String> {
    // SAFETY: loading runs the library's initializers; these are the
    // environment's libpython, libtvm_ffi and the worker library, whose
    // initializers start nothing.
    unsafe { Library::open(Some(path), flags) }
        .map_err(|error| format!("loading {}: {error}", path.display()))
}
