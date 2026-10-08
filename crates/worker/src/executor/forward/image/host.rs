//! Host work of the image runtime: input preparation and image encoding.
//!
//! Inline request images are decoded and transformed on a host lane while
//! other batches execute (`image.prepare_input`, which releases the GIL for
//! most of its work); decoded images are encoded as base64 PNG on a host
//! lane once their pixels reached host memory. Both run on the runtime's
//! [`HostLane`]; completing a task wakes the rank's service loop.

use std::sync::{Arc, Mutex, PoisonError};

use crate::ffi::{Function, Handle};
use crate::host::image::png_base64;
use crate::host::{HostAction, HostTask};
use crate::runner::encoder::{PreparedImage, Tower, prepare_input};
use crate::storage::OutputLease;
use crate::storage::memory::StorageView;
use crate::{Error, Outcome, Result};

/// One unit of image host work.
pub(crate) enum ImageWork {
    /// Decode and transform one inline image of a request.
    Prepare {
        function: Function,
        loaded: Handle,
        tower: Tower,
        payload: Arc<str>,
        input_images: u32,
        pin: bool,
    },
    /// Encode one image's quantized pixels, read back into `lease`, as a
    /// base64 PNG of at most `maximum` bytes.
    Encode {
        pixels: StorageView,
        lease: OutputLease,
        height: u32,
        width: u32,
        maximum: usize,
    },
}

/// What image host work produced.
pub(crate) enum ImageResult {
    Prepared(PreparedImage),
    Encoded(Vec<u8>),
}

/// A host-lane action of the image runtime.
pub(crate) struct ImageTask {
    work: Mutex<ImageWork>,
}

impl ImageTask {
    pub(crate) fn new(work: ImageWork) -> Self {
        Self {
            work: Mutex::new(work),
        }
    }
}

impl HostAction for ImageTask {
    type Output = ImageResult;
    type Error = Error;
    type Callback = Box<dyn FnOnce() + Send>;
    type Wake = uniserve_ipc::Wake;

    fn ready(&self) -> Result<bool> {
        let work = self.work.lock().unwrap_or_else(PoisonError::into_inner);
        match &*work {
            ImageWork::Prepare { .. } => Ok(true),
            // Encoding reads host bytes only after their copy completed.
            ImageWork::Encode { lease, .. } => Ok(lease
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .ready_at()?
                .is_some()),
        }
    }

    fn run(&self) -> Result<ImageResult> {
        let work = self.work.lock().unwrap_or_else(PoisonError::into_inner);
        match &*work {
            ImageWork::Prepare {
                function,
                loaded,
                tower,
                payload,
                input_images,
                pin,
            } => prepare_input(function, loaded, *tower, payload, *input_images, *pin)
                .map(ImageResult::Prepared),
            ImageWork::Encode {
                pixels,
                height,
                width,
                maximum,
                ..
            } => {
                let bytes = pixels.nbytes() as usize;
                // SAFETY: the lease's device-to-host copy of these bytes
                // completed (`ready`), and the task holds the lease's CPU
                // reader until it finishes.
                let pixels =
                    unsafe { std::slice::from_raw_parts(pixels.address() as *const u8, bytes) };
                png_base64(pixels, *height, *width, *maximum).map(ImageResult::Encoded)
            }
        }
    }

    fn release(&self) -> Result<()> {
        let work = self.work.lock().unwrap_or_else(PoisonError::into_inner);
        if let ImageWork::Encode { lease, .. } = &*work {
            lease
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .release_reader()?;
        }
        Ok(())
    }

    fn input_outcome(&self) -> Option<Outcome<Error>> {
        // Inputs are host bytes the task owns or a completed copy.
        Some(Outcome::Success(()))
    }

    fn defer_release(_: Arc<HostTask<Self>>) -> Result<()> {
        Err(Error::Invariant(
            "image host work has no deferred input".into(),
        ))
    }

    fn notify(callbacks: Vec<Self::Callback>) {
        for callback in callbacks {
            callback();
        }
    }

    fn wake(wake: &uniserve_ipc::Wake) {
        wake.wake();
    }

    fn report(error: Error) {
        tracing::error!(target: "uniserve_worker.execution", "image host work failed: {error}");
    }

    fn error(error: Error) -> Error {
        error
    }

    fn note_cleanup(_: &mut Error, cleanup: Error) {
        tracing::error!(
            target: "uniserve_worker.execution",
            "image host work cleanup also failed: {cleanup}"
        );
    }
}
