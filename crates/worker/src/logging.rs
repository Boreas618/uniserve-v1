//! The rank's log output: one line per event on stderr, in the format the
//! numerical library's Python logging uses (`time LEVEL target: message`).

use tracing_subscriber::EnvFilter;

/// Install the process logger once; later calls keep the first.
pub(crate) fn init() {
    let filter =
        EnvFilter::try_from_env("UNISERVE_WORKER_LOG").unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .try_init();
}
