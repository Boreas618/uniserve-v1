//! The driver calls that made a rank's serving threads wait for the device,
//! counted by thread and call while the rank serves ([`cuda::cupti`]).
//!
//! The builder thread watches each serving call of the executor (a call so
//! counted fails, `crate::ffi::MethodCall::serving`), a lane's execution
//! thread each job it performs, and the control thread its serve loop. Their
//! counts accumulate here under `{thread}/{call}` and travel with the next
//! batch result (`ForwardStats::blocking_calls`).

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::lock;

static COUNTS: Mutex<BTreeMap<String, u64>> = Mutex::new(BTreeMap::new());

/// Count `calls` (by call name) that `thread` made.
pub(crate) fn add(thread: &str, calls: &[(&'static str, u64)]) {
    if calls.is_empty() {
        return;
    }
    let mut counts = lock(&COUNTS);
    for &(call, count) in calls {
        *counts.entry(format!("{thread}/{call}")).or_default() += count;
    }
}

/// The counts since the last take.
pub(crate) fn take() -> BTreeMap<String, u64> {
    std::mem::take(&mut *lock(&COUNTS))
}
