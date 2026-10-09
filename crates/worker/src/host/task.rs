//! Media work on a host lane: request preparation, condition reading,
//! codecs and MP4 assembly.
//!
//! Each [`MediaTask`] is one host-lane action. It holds every input it
//! reads (product views, borrowed shared-memory rows, the request's
//! [`MuxSession`]) until the lane retires it, so the executor can retire
//! the batch that configured it independently of the task's thread. Python
//! runs only inside the interface functions a task calls.

use std::sync::{Arc, Mutex, PoisonError};

use tvm_ffi::{Any, AnyView};
use uniserve_core::VideoCondition;

use super::mux::{MuxSession, frame_unit};
use super::reader::{ConditionProducts, ConditionReader};
use super::{HostAction, HostTask};
use crate::ffi::{Backing, DType, Device, Function, Handle};
use crate::storage::memory::StorageView;
use crate::transfer::{SharedMapping, SharedRead};
use crate::{Error, Outcome};

/// A borrowed shared-memory payload exposed as host bytes. The mapping
/// stays alive through every tensor viewing it.
struct MappedBytes {
    mapping: Arc<SharedMapping>,
    offset: usize,
    bytes: usize,
}

impl Backing for MappedBytes {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn address(&self) -> u64 {
        (self.mapping.address() + self.offset) as u64
    }

    fn len(&self) -> usize {
        self.bytes
    }
}

/// The RGB24 frames of one media unit a codec encodes.
pub(crate) enum UnitFrames {
    /// Read in place from the decoder's shared-memory export; released
    /// (acknowledged) when the task retires.
    Borrowed(Mutex<SharedRead>),
    /// A host view of an imported product row.
    Host(StorageView),
}

impl UnitFrames {
    /// The leading `bytes` of the unit as `[frames, height, width, 3]`.
    fn view(&self, frames: u32, height: u32, width: u32) -> crate::Result<StorageView> {
        let shape = [i64::from(frames), i64::from(height), i64::from(width), 3];
        let bytes = u64::from(frames) * u64::from(height) * u64::from(width) * 3;
        match self {
            Self::Borrowed(read) => {
                let read = read.lock().unwrap_or_else(PoisonError::into_inner);
                if (read.nbytes() as u64) < bytes {
                    return Err(Error::Invalid(
                        "decoded media unit holds fewer bytes than its frames".into(),
                    ));
                }
                let backing = MappedBytes {
                    mapping: read.mapping()?,
                    offset: read.offset(),
                    bytes: bytes as usize,
                };
                StorageView::contiguous(Arc::new(backing) as Arc<dyn Backing>, 0, DType::U8, &shape)
            }
            Self::Host(view) => {
                if view.nbytes() < bytes || !view.is_contiguous() {
                    return Err(Error::Invalid(
                        "decoded media unit holds fewer bytes than its frames".into(),
                    ));
                }
                StorageView::contiguous(Arc::clone(view.memory()), view.offset(), DType::U8, &shape)
            }
        }
    }

    fn release(&self) {
        if let Self::Borrowed(read) = self {
            read.lock()
                .unwrap_or_else(PoisonError::into_inner)
                .release();
        }
    }
}

/// One host-lane action.
pub(crate) enum MediaTask {
    /// Draw a video request's noise and fill its slot's host sources
    /// (`prepare_request` of the denoiser binding).
    Prepare {
        function: Function,
        arguments: Vec<Handle>,
    },
    /// Decode a conditioned request's media into its products.
    Read {
        reader: Arc<ConditionReader>,
        conditions: Arc<[VideoCondition]>,
        products: ConditionProducts,
    },
    /// Encode one media unit and frame it into its `encoded_units` row.
    EncodeUnit {
        function: Function,
        frames: UnitFrames,
        shape: (u32, u32, u32),
        frame_rate: u32,
        row: StorageView,
    },
    /// Encode the request's soundtrack into its MP4 session.
    EncodeAudio {
        function: Function,
        pcm: StorageView,
        samples: u64,
        rate: u32,
        session: Arc<MuxSession>,
    },
    /// Append encoded units to the request's MP4 in unit order.
    Append {
        session: Arc<MuxSession>,
        units: Vec<Vec<u8>>,
    },
    /// Finish the request's MP4.
    Finalize { session: Arc<MuxSession> },
}

/// What a finished task produced.
pub(crate) enum TaskOutput {
    Done,
    /// The framed bytes written into an encoded unit's row.
    Framed(usize),
    /// A finished MP4.
    Media(Vec<u8>),
}

/// Observers of a task's completion.
pub(crate) type TaskCallback = Box<dyn FnOnce() + Send>;

impl MediaTask {
    fn execute(&self) -> crate::Result<TaskOutput> {
        match self {
            Self::Prepare {
                function,
                arguments,
            } => {
                let views: Vec<AnyView<'_>> = arguments.iter().map(Handle::view).collect();
                function.invoke(&views)?;
                Ok(TaskOutput::Done)
            }
            Self::Read {
                reader,
                conditions,
                products,
            } => {
                reader.read(conditions, products)?;
                Ok(TaskOutput::Done)
            }
            Self::EncodeUnit {
                function,
                frames,
                shape: (count, height, width),
                frame_rate,
                row,
            } => {
                let rgb = frames.view(*count, *height, *width)?;
                let encoded: tvm_ffi::Bytes = function.call_as(&[
                    AnyView::from(&Any::from(rgb.to_tensor()?)),
                    AnyView::from(&Any::from(i64::from(*frame_rate))),
                ])?;
                if row.device() != Device::Cpu || !row.is_contiguous() {
                    return Err(Error::Invariant(
                        "an encoded unit row is not contiguous host memory".into(),
                    ));
                }
                // SAFETY: the row is this call's reserved host product row;
                // no reader exists until the task's call publishes it.
                let target = unsafe {
                    std::slice::from_raw_parts_mut(row.address() as *mut u8, row.nbytes() as usize)
                };
                Ok(TaskOutput::Framed(frame_unit(encoded.as_slice(), target)?))
            }
            Self::EncodeAudio {
                function,
                pcm,
                samples,
                rate,
                session,
            } => {
                let encoded: tvm_ffi::Bytes = function.call_as(&[
                    AnyView::from(&Any::from(pcm.to_tensor()?)),
                    AnyView::from(&Any::from(*samples as i64)),
                    AnyView::from(&Any::from(i64::from(*rate))),
                ])?;
                session.set_audio(encoded)?;
                Ok(TaskOutput::Done)
            }
            Self::Append { session, units } => {
                let units: Vec<&[u8]> = units.iter().map(Vec::as_slice).collect();
                session.append(&units)?;
                Ok(TaskOutput::Done)
            }
            Self::Finalize { session } => {
                Ok(TaskOutput::Media(session.finish()?.as_slice().to_vec()))
            }
        }
    }
}

impl HostAction for MediaTask {
    type Output = TaskOutput;
    type Error = Arc<Error>;
    type Callback = TaskCallback;
    type Wake = uniserve_ipc::Wake;

    fn ready(&self) -> Result<bool, Self::Error> {
        Ok(true)
    }

    fn run(&self) -> Result<TaskOutput, Self::Error> {
        let _range = crate::profiling::range(c"uniserve.host.media", None);
        self.execute().map_err(Arc::new)
    }

    fn release(&self) -> Result<(), Self::Error> {
        if let Self::EncodeUnit { frames, .. } = self {
            frames.release();
        }
        Ok(())
    }

    fn input_outcome(&self) -> Option<Outcome<Self::Error>> {
        // Inputs are host memory read on the task's own thread: they are
        // released as soon as the action returns or is withdrawn.
        Some(Outcome::Success(()))
    }

    fn defer_release(_: Arc<HostTask<Self>>) -> Result<(), Self::Error> {
        Err(Arc::new(Error::Invariant(
            "a media task's inputs are released with its action".into(),
        )))
    }

    fn notify(callbacks: Vec<Self::Callback>) {
        for callback in callbacks {
            callback();
        }
    }

    fn wake(wake: &Self::Wake) {
        wake.wake();
    }

    fn report(error: Self::Error) {
        tracing::error!(target: "uniserve_worker.media", "host media task failed: {error}");
    }

    fn error(error: Error) -> Self::Error {
        Arc::new(error)
    }

    fn note_cleanup(_error: &mut Self::Error, cleanup: Self::Error) {
        tracing::warn!(target: "uniserve_worker.media", "host media cleanup failed: {cleanup}");
    }
}
