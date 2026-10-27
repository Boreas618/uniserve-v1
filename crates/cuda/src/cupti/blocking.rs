//! Driver calls that make the host wait for the device, observed on marked
//! threads through CUPTI's callback API.
//!
//! [`BlockingWatch::start`] enables, on the process's subscription
//! ([`super::subscriber`]), the callbacks of every driver call that blocks its
//! calling thread until device work completes: stream, event and context
//! synchronization, synchronous copies, `cuMemFree` and `cuMemFreeHost`
//! (which the driver documents as possibly synchronizing implicitly) and
//! `cuMemUnmap`. Runtime API calls, PyTorch's included, reach these driver
//! calls too, so the driver domain covers both.
//!
//! CUPTI invokes the callback on the calling thread as the call is entered.
//! A thread inside a [`Watched`] scope counts the call in its own counters
//! by call, which the scope reads ([`Watched::blocking_calls`],
//! [`Watched::calls`]) and a thread can snapshot at any time
//! ([`Calls::current`]); a thread outside any scope is not counted. The
//! callback only increments thread-local counters.
//!
//! Dropping the watch releases its callbacks; a later
//! [`BlockingWatch::start`] enables them again.

use std::cell::Cell;
use std::ffi::c_void;
use std::marker::PhantomData;

use super::subscriber::{self, API_ENTER, DRIVER_DOMAIN};
use crate::Result;

/// The watched driver calls: `CUpti_driver_api_trace_cbid` and name. Every
/// call here returns only once the device work it waits for completed.
const WATCHED: &[(u32, &str)] = &[
    (17, "cuCtxSynchronize"),
    (800, "cuCtxSynchronize_v2"),
    (121, "cuEventSynchronize"),
    (126, "cuStreamSynchronize"),
    (440, "cuStreamSynchronize_ptsz"),
    (38, "cuMemFreeHost"),
    (245, "cuMemFree_v2"),
    (552, "cuMemUnmap"),
    (43, "cuMemcpyHtoD"),
    (45, "cuMemcpyDtoH"),
    (47, "cuMemcpyDtoD"),
    (49, "cuMemcpyDtoA"),
    (51, "cuMemcpyAtoD"),
    (53, "cuMemcpyHtoA"),
    (54, "cuMemcpyAtoH"),
    (55, "cuMemcpyAtoA"),
    (56, "cuMemcpy2D"),
    (57, "cuMemcpy2DUnaligned"),
    (58, "cuMemcpy3D"),
    (248, "cuMemcpy_v2"),
    (276, "cuMemcpyHtoD_v2"),
    (278, "cuMemcpyDtoH_v2"),
    (280, "cuMemcpyDtoD_v2"),
    (282, "cuMemcpyAtoH_v2"),
    (284, "cuMemcpyAtoD_v2"),
    (285, "cuMemcpyDtoA_v2"),
    (286, "cuMemcpyAtoA_v2"),
    (287, "cuMemcpy2D_v2"),
    (288, "cuMemcpy2DUnaligned_v2"),
    (290, "cuMemcpy3D_v2"),
    (292, "cuMemcpyHtoA_v2"),
    (305, "cuMemcpy"),
    (318, "cuMemcpyPeer"),
    (320, "cuMemcpy3DPeer"),
    (397, "cuMemcpyHtoD_v2_ptds"),
    (398, "cuMemcpyDtoH_v2_ptds"),
    (399, "cuMemcpyDtoD_v2_ptds"),
    (408, "cuMemcpy_ptds"),
];

thread_local! {
    /// How many [`Watched`] scopes the thread is inside.
    static DEPTH: Cell<u32> = const { Cell::new(0) };
    /// The watched calls the thread made inside a scope, by entry of
    /// [`WATCHED`].
    static CALLS: [Cell<u64>; WATCHED.len()] = const { [const { Cell::new(0) }; WATCHED.len()] };
}

/// The process's watch of the blocking driver calls; releases its callbacks
/// on drop.
pub struct BlockingWatch {
    _private: (),
}

impl BlockingWatch {
    /// Watch the blocking driver calls on the process's subscription.
    ///
    /// # Errors
    ///
    /// Fails when the process could not subscribe to CUPTI
    /// ([`subscriber::subscribe`]).
    pub fn start() -> Result<Self> {
        subscriber::subscribe()?;
        for (index, &(cbid, _)) in WATCHED.iter().enumerate() {
            if let Err(error) = subscriber::enable_callback(DRIVER_DOMAIN, cbid) {
                for &(enabled, _) in &WATCHED[..index] {
                    subscriber::disable_callback(DRIVER_DOMAIN, enabled);
                }
                return Err(error);
            }
        }
        Ok(Self { _private: () })
    }
}

impl Drop for BlockingWatch {
    fn drop(&mut self) {
        for &(cbid, _) in WATCHED {
            subscriber::disable_callback(DRIVER_DOMAIN, cbid);
        }
    }
}

/// The dispatch's part: counts a watched call entered on a watched thread.
///
/// # Safety
/// `data` is the `CUpti_CallbackData` of a driver API callback.
pub(super) unsafe fn on_driver_call(cbid: u32, data: *const c_void) {
    if DEPTH.get() == 0 {
        return;
    }
    // SAFETY: the callback data's first field is the `CUpti_ApiCallbackSite`.
    if unsafe { *data.cast::<i32>() } != API_ENTER {
        return;
    }
    if let Some(index) = WATCHED.iter().position(|&(watched, _)| watched == cbid) {
        CALLS.with(|calls| calls[index].set(calls[index].get() + 1));
    }
}

/// A snapshot of the calling thread's counters of watched calls.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Calls([u64; WATCHED.len()]);

impl Default for Calls {
    fn default() -> Self {
        Self([0; WATCHED.len()])
    }
}

impl Calls {
    /// The calling thread's counters now.
    pub fn current() -> Self {
        Self(CALLS.with(|calls| std::array::from_fn(|index| calls[index].get())))
    }

    /// The calls counted between `earlier` and this snapshot of the same
    /// thread, by call name; calls without any are left out.
    pub fn since(&self, earlier: &Self) -> Vec<(&'static str, u64)> {
        WATCHED
            .iter()
            .zip(self.0.iter().zip(&earlier.0))
            .filter_map(|(&(_, name), (&now, &then))| {
                let count = now.saturating_sub(then);
                (count > 0).then_some((name, count))
            })
            .collect()
    }
}

/// A scope in which the calling thread's watched driver calls are counted.
/// It stays on the thread that entered it.
pub struct Watched {
    /// The thread's counters when the scope was entered.
    entered: Calls,
    _thread: PhantomData<*const ()>,
}

impl Watched {
    /// Count the calling thread's watched calls until the scope ends.
    pub fn enter() -> Self {
        DEPTH.set(DEPTH.get() + 1);
        Self {
            entered: Calls::current(),
            _thread: PhantomData,
        }
    }

    /// Watched calls this thread made since the scope was entered.
    pub fn blocking_calls(&self) -> u64 {
        self.calls().iter().map(|&(_, count)| count).sum()
    }

    /// The watched calls this thread made since the scope was entered, by
    /// call name; calls it did not make are left out.
    pub fn calls(&self) -> Vec<(&'static str, u64)> {
        Calls::current().since(&self.entered)
    }
}

impl Drop for Watched {
    fn drop(&mut self) {
        DEPTH.set(DEPTH.get().saturating_sub(1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cupti::activity::StreamActivity;
    use crate::{DeviceAllocation, DeviceId, Stream, lock, memset_d8};

    #[test]
    #[ignore = "requires a CUDA device"]
    fn only_a_watched_thread_counts_its_waits_on_the_device() -> Result<()> {
        let _serial = lock(&crate::cupti::tests::SERIAL);
        let watch = BlockingWatch::start()?;
        let stream = Stream::new(DeviceId(0))?;

        // Outside a scope a wait is not counted.
        let before = Calls::current();
        stream.synchronize()?;
        assert_eq!(Calls::current(), before);

        let scope = Watched::enter();
        stream.synchronize()?;
        assert_eq!(scope.blocking_calls(), 1);
        assert_eq!(scope.calls(), vec![("cuStreamSynchronize", 1)]);
        drop(scope);
        drop(watch);
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn a_stopped_watch_counts_nothing_until_a_new_one_starts() -> Result<()> {
        let _serial = lock(&crate::cupti::tests::SERIAL);
        let stream = Stream::new(DeviceId(0))?;
        let scope = Watched::enter();
        let watch = BlockingWatch::start()?;
        drop(StreamActivity::start([("device", &stream)])?);
        drop(watch);
        stream.synchronize()?;
        assert_eq!(scope.blocking_calls(), 0);

        let watch = BlockingWatch::start()?;
        let recording = StreamActivity::start([("device", &stream)])?;
        let bytes = 256 << 20;
        let memory = DeviceAllocation::linear(DeviceId(0), bytes)?;
        // SAFETY: the allocation holds `bytes` and outlives the fill, which
        // the synchronization waits for.
        unsafe { memset_d8(memory.address(), 0, bytes, &stream)? };
        stream.synchronize()?;
        assert_eq!(scope.calls(), vec![("cuStreamSynchronize", 1)]);
        let taken = crate::cupti::activity::tests::taken_until("device", |taken| taken.busy_us > 0);
        assert!(taken.busy_us > 0, "{taken:?}");
        drop(recording);
        drop(watch);
        Ok(())
    }
}
