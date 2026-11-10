//! The scheduler's wall-clock and monotonic clocks.

use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Returns wall-clock time in fractional seconds since the Unix epoch.
///
/// Times before the epoch clamp to zero.
pub(crate) fn now_unix_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

/// Returns process-local monotonic time in whole microseconds from a stable,
/// process-wide epoch fixed by the first call. Used for lifecycle phase stamps, whose differences
/// stay valid across wall-clock adjustments.
pub(crate) fn now_monotonic_us() -> u64 {
    EPOCH.get_or_init(Instant::now).elapsed().as_micros() as u64
}

/// Process-wide origin shared by the monotonic clocks, set on first use.
static EPOCH: OnceLock<Instant> = OnceLock::new();

/// The wall-clock time, in fractional Unix seconds, of the monotonic clock's
/// origin, read once: the one offset that maps a wall-clock instant onto the
/// monotonic clock, so a given instant always maps to the same monotonic one.
pub(crate) fn monotonic_origin_unix_secs() -> f64 {
    *ORIGIN_UNIX.get_or_init(|| now_unix_secs() - now_monotonic_us() as f64 / 1e6)
}

static ORIGIN_UNIX: OnceLock<f64> = OnceLock::new();
