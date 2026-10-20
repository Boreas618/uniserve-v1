//! The CPython interpreter a worker hosts on its Python thread.
//!
//! The `uniserve-worker` executable loads the environment's shared
//! `libpython` into the process's global scope before it loads this library,
//! and the library initializes the interpreter on a thread it created, which
//! Python then treats as its main thread. The C API functions used here are
//! resolved from the process when the interpreter starts rather than linked:
//! the same library is also loaded into ordinary Python processes (tests,
//! harnesses, the video rules it exports), where Python is already running
//! and none of this is called.
//!
//! The interpreter is never finalized. A worker ends with `_exit`, as an
//! aborted worker must: finalizers could wait on device work or on a failed
//! peer.

use std::ffi::{CStr, CString, c_char, c_int};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

use libc::wchar_t;
use libloading::{Library, Symbol};
use pyo3_ffi::{PyConfig, PyStatus, PyThreadState};

use crate::ffi::Interpreter;
use crate::{Error, Result};

/// The global function through which the library reaches every Python
/// callable by its import path, registered by `uniserve.worker.attach`.
const IMPORT_PATH: &str = "uniserve.worker.import_path";

/// What the Python thread runs once the interpreter is up: the library's
/// worker entry, which configures Python logging and diagnostic dumps and
/// registers [`IMPORT_PATH`].
const ATTACH: &CStr = c"import uniserve.worker\nuniserve.worker.attach()\n";

/// The C API functions that host the interpreter.
struct CApi<'process> {
    config_init: Symbol<'process, unsafe extern "C" fn(*mut PyConfig)>,
    config_set_bytes_string: Symbol<
        'process,
        unsafe extern "C" fn(*mut PyConfig, *mut *mut wchar_t, *const c_char) -> PyStatus,
    >,
    config_clear: Symbol<'process, unsafe extern "C" fn(*mut PyConfig)>,
    initialize: Symbol<'process, unsafe extern "C" fn(*const PyConfig) -> PyStatus>,
    status_exception: Symbol<'process, unsafe extern "C" fn(PyStatus) -> c_int>,
    run_simple_string: Symbol<'process, unsafe extern "C" fn(*const c_char) -> c_int>,
    save_thread: Symbol<'process, unsafe extern "C" fn() -> *mut PyThreadState>,
}

impl<'process> CApi<'process> {
    /// Resolve every function from the process's global scope.
    fn resolve(process: &'process Library) -> Result<Self> {
        fn symbol<'process, T>(
            process: &'process Library,
            name: &'static CStr,
        ) -> Result<Symbol<'process, T>> {
            // SAFETY: each name is a CPython C API function whose type is
            // the one the caller states, as `Python.h` declares it.
            unsafe { process.get(name.to_bytes_with_nul()) }.map_err(|error| {
                Error::Unsupported(format!(
                    "no CPython is loaded into this process ({name:?}: {error}); a worker \
                     starts as the `uniserve-worker` executable, which loads the \
                     environment's shared libpython"
                ))
            })
        }
        Ok(Self {
            config_init: symbol(process, c"PyConfig_InitPythonConfig")?,
            config_set_bytes_string: symbol(process, c"PyConfig_SetBytesString")?,
            config_clear: symbol(process, c"PyConfig_Clear")?,
            initialize: symbol(process, c"Py_InitializeFromConfig")?,
            status_exception: symbol(process, c"PyStatus_Exception")?,
            run_simple_string: symbol(process, c"PyRun_SimpleString")?,
            save_thread: symbol(process, c"PyEval_SaveThread")?,
        })
    }

    /// Fail with the status's message when it reports an error or an exit.
    fn check(&self, status: PyStatus, what: &str) -> Result<()> {
        // SAFETY: the status was returned by the C API just now.
        if unsafe { (self.status_exception)(status) } == 0 {
            return Ok(());
        }
        let message = if status.err_msg.is_null() {
            "no message".into()
        } else {
            // SAFETY: a set error message is a static C string.
            unsafe { CStr::from_ptr(status.err_msg) }.to_string_lossy()
        };
        Err(Error::Interface(format!("{what}: {message}")))
    }
}

/// Make the calling thread the process's Python thread: initialize CPython
/// with `python`, the environment's interpreter, as `PyConfig.executable`,
/// so that `sys.prefix` names the environment and subprocesses that torch
/// and Triton start run its interpreter; run `uniserve.worker.attach()`;
/// install its import function ([`Interpreter::install_import`]); and
/// release the GIL.
///
/// Python installs no signal handlers: the worker owns SIGINT and SIGTERM
/// (`super::Signals`), and `attach` registers only faulthandler's
/// diagnostic dumps. From here on every call into Python is a TVM-FFI
/// callback, which takes the GIL itself.
///
/// Fails when no CPython is loaded into the process, when the interpreter
/// does not initialize, or when `attach` raises; its traceback is printed to
/// stderr.
pub(super) fn host(python: &Path) -> Result<()> {
    let process = Library::from(libloading::os::unix::Library::this());
    let api = CApi::resolve(&process)?;
    let executable = CString::new(python.as_os_str().as_bytes())
        .map_err(|_| Error::Invalid(format!("{} holds a NUL byte", python.display())))?;

    let mut config = std::mem::MaybeUninit::<PyConfig>::uninit();
    // SAFETY: `PyConfig_InitPythonConfig` initializes every field of the
    // configuration it is given.
    let mut config = unsafe {
        (api.config_init)(config.as_mut_ptr());
        config.assume_init()
    };
    config.install_signal_handlers = 0;
    // The process ends without finalization, which would flush buffered
    // standard streams: Python writes them through at once instead.
    config.buffered_stdio = 0;
    let configured = {
        let config = &raw mut config;
        // SAFETY: both fields belong to the initialized configuration, and
        // the string is copied into it.
        unsafe {
            (api.config_set_bytes_string)(
                config,
                &raw mut (*config).executable,
                executable.as_ptr(),
            )
        }
    };
    let initialized = api
        .check(configured, "configuring the worker's interpreter")
        .and_then(|()| {
            // SAFETY: the configuration is initialized, and this is the
            // process's only interpreter.
            let status = unsafe { (api.initialize)(&raw const config) };
            api.check(status, "initializing the worker's interpreter")
        });
    // SAFETY: the configuration was initialized above and is cleared once.
    unsafe { (api.config_clear)(&raw mut config) };
    initialized?;

    // SAFETY: the interpreter is initialized and this thread holds the GIL.
    if unsafe { (api.run_simple_string)(ATTACH.as_ptr()) } != 0 {
        return Err(Error::Interface(
            "uniserve.worker.attach() raised; its traceback precedes".into(),
        ));
    }
    let import = tvm_ffi::Function::get_global(IMPORT_PATH).map_err(|error| {
        Error::Interface(format!(
            "uniserve.worker.attach() registered no {IMPORT_PATH}: {}",
            error.message()
        ))
    })?;
    Interpreter::install_import(import);

    // The thread keeps its thread state, which every callback's
    // `PyGILState_Ensure` on this thread takes up again.
    // SAFETY: this thread holds the GIL, which it releases.
    unsafe { (api.save_thread)() };
    Ok(())
}
