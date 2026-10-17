//! The rank's log output: one line per event on stderr, in the format the
//! numerical library's Python logging uses (`time LEVEL target: message`).

use std::backtrace::Backtrace;
use std::panic::PanicHookInfo;

use tracing_subscriber::EnvFilter;

/// Install the process logger and the panic hook once; later calls keep the
/// first.
pub(crate) fn init() {
    let filter =
        EnvFilter::try_from_env("UNISERVE_WORKER_LOG").unwrap_or_else(|_| EnvFilter::new("info"));
    let installed = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .try_init();
    if installed.is_ok() {
        std::panic::set_hook(Box::new(report_panic));
    }
}

/// Log a panic as one error event of the rank's log, with its thread,
/// location and a backtrace.
///
/// Every build profile sets `panic = "abort"`, so the process aborts as soon
/// as the hook returns: this event is the rank's last word and the only
/// record of where it failed. It replaces the default hook, whose message
/// would bypass the rank's log format and carry a backtrace only under
/// `RUST_BACKTRACE`.
fn report_panic(info: &PanicHookInfo<'_>) {
    let thread = std::thread::current();
    let message = info.payload_as_str().unwrap_or("a panic without a message");
    let location = info.location().map_or_else(
        || "an unknown location".to_owned(),
        |location| location.to_string(),
    );
    tracing::error!(
        "thread {} panicked at {location}: {message}\n{}",
        thread.name().unwrap_or("<unnamed>"),
        Backtrace::force_capture()
    );
}
