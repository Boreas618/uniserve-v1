//! Deferred release of device and pinned host storage.
//!
//! Dropping a [`crate::DeviceAllocation`], [`crate::PeerMapping`] or
//! [`crate::PinnedBuffer`] makes no CUDA call on the dropping thread. The
//! release (`cuMemFree`, `cuMemUnmap` + `cuMemAddressFree` + `cuMemRelease`,
//! `cuMemFreeHost`) is queued to one process-wide reclaimer thread instead,
//! because:
//!
//! - the last reference may drop inside a DLPack deleter while the Python GIL
//!   is held, and `cuMemFree`/`cuMemFreeHost` may synchronize the device,
//!   which can wait on a host function that needs the GIL;
//! - a free issued from a thread that is capturing a CUDA graph in global
//!   mode would invalidate the capture. The reclaimer thread runs in relaxed
//!   capture mode, so its calls never interact with other threads' captures.
//!
//! The reclaimer does not order releases after device work. Owners guarantee
//! that no queued or replayable device work still addresses storage when they
//! drop its last reference (the worker retires storage by events and drains
//! its streams before shutdown).
//!
//! [`flush`] waits for every release queued so far and reports failures; the
//! worker calls it last during shutdown.

use std::sync::mpsc::{self, Sender, SyncSender};
use std::sync::{Mutex, OnceLock};
use std::thread;

use crate::driver::driver;
use crate::{Error, Result, lock};

const CU_STREAM_CAPTURE_MODE_RELAXED: i32 = 2;

type Release = Box<dyn FnOnce() -> Result<()> + Send>;

enum Message {
    Release(Release),
    Flush(SyncSender<()>),
}

/// Failures since the previous flush: their count and the first one.
static FAILURES: Mutex<(usize, Option<Error>)> = Mutex::new((0, None));

fn record(error: Error) {
    let mut failures = lock(&FAILURES);
    failures.0 += 1;
    failures.1.get_or_insert(error);
}

fn run(release: Release) {
    if let Err(error) = release() {
        record(error);
    }
}

fn reclaimer() -> Option<&'static Sender<Message>> {
    static RECLAIMER: OnceLock<Option<Sender<Message>>> = OnceLock::new();

    RECLAIMER
        .get_or_init(|| {
            let (sender, receiver) = mpsc::channel::<Message>();
            thread::Builder::new()
                .name("cuda-reclaim".into())
                .spawn(move || {
                    if let Ok(driver) = driver() {
                        let mut mode = CU_STREAM_CAPTURE_MODE_RELAXED;
                        // SAFETY: sets this thread's capture interaction mode;
                        // the driver writes back the previous mode.
                        let _ = unsafe { (driver.thread_exchange_capture_mode)(&mut mode) };
                    }
                    for message in receiver {
                        match message {
                            Message::Release(release) => run(release),
                            // Messages are handled in order, so every release
                            // queued before the flush has completed.
                            Message::Flush(done) => {
                                let _ = done.send(());
                            }
                        }
                    }
                })
                .ok()
                .map(|_| sender)
        })
        .as_ref()
}

/// Queues `release` on the reclaimer thread. If the thread could not be
/// started, the release runs on the calling thread instead of leaking.
pub(crate) fn defer(release: impl FnOnce() -> Result<()> + Send + 'static) {
    let release: Release = Box::new(release);
    match reclaimer() {
        Some(sender) => {
            if let Err(mpsc::SendError(Message::Release(release))) =
                sender.send(Message::Release(release))
            {
                run(release);
            }
        }
        None => run(release),
    }
}

/// Waits until every release queued before this call has run.
///
/// Returns [`Error::Reclaim`] if any deferred release failed since the
/// previous flush; the failed storage remains allocated until process exit.
pub fn flush() -> Result<()> {
    if let Some(sender) = reclaimer() {
        let (done, completed) = mpsc::sync_channel(1);
        if sender.send(Message::Flush(done)).is_ok() {
            let _ = completed.recv();
        }
    }

    let mut failures = lock(&FAILURES);
    let (failed, first) = std::mem::take(&mut *failures);
    match first {
        Some(first) => Err(Error::Reclaim {
            failed,
            first: Box::new(first),
        }),
        None => Ok(()),
    }
}
