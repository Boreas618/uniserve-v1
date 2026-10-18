//! Media work on a host lane: request preparation, condition reading,
//! codecs and MP4 assembly.
//!
//! Each [`MediaTask`] is one host-lane action. It holds every input it
//! reads (product views, borrowed shared-memory rows, the request's
//! [`MuxSession`]) until the lane retires it, so the executor can retire
//! the batch that configured it independently of the task's thread. Python
//! runs only inside the library functions a task calls.

use std::sync::{Arc, Mutex};

use tvm_ffi::{Any, AnyView};
use uniserve_core::VideoCondition;

use uniserve_cuda::{CopyBatch, SourceOrder};

use super::HostAction;
use super::mux::{MuxSession, frame_unit};
use super::reader::{ConditionProducts, ConditionReader};
use crate::ffi::{Backing, DType, Function, Interpreter, MemoryKind};
use crate::storage::memory::StorageView;
use crate::stream::CUDAStream;
use crate::transfer::{SharedMapping, SharedRead};
use crate::{Error, Outcome, Result, lock};

/// A borrowed shared-memory payload exposed as host bytes. The mapping
/// stays alive through every tensor viewing it.
struct MappedBytes {
    mapping: Arc<SharedMapping>,
    offset: usize,
    bytes: usize,
}

impl Backing for MappedBytes {
    fn device(&self) -> MemoryKind {
        MemoryKind::Host
    }

    fn address(&self) -> u64 {
        (self.mapping.address() + self.offset) as u64
    }

    fn bytes(&self) -> usize {
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
                let read = lock(read);
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
            lock(read).release();
        }
    }
}

/// One host-lane action.
pub(crate) enum MediaTask {
    /// Decode a conditioned request's media into its products, or into
    /// page-locked staging that `upload` then copies to device products.
    Read {
        reader: Arc<ConditionReader>,
        conditions: Arc<[VideoCondition]>,
        products: ConditionProducts,
        upload: Option<Upload>,
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

/// The copies that move a device rank's decoded condition media from its
/// page-locked staging into its device products.
pub(crate) struct Upload {
    /// The reader's copy stream, whose only issuer is the host lane.
    pub(crate) stream: Arc<CUDAStream>,
    /// `(staging, product)` views of equal bytes, both contiguous.
    pub(crate) copies: Vec<(StorageView, StorageView)>,
}

impl Upload {
    /// Copy every staging view into its product and wait for the copies, so
    /// the products are complete when the task finishes.
    fn perform(&self) -> Result<()> {
        let mut batch = CopyBatch::new(SourceOrder::Stream);
        for (staging, product) in &self.copies {
            if staging.nbytes() != product.nbytes()
                || !staging.is_contiguous()
                || !product.is_contiguous()
            {
                return Err(Error::Invariant(
                    "condition staging disagrees with its device product".into(),
                ));
            }
            batch.push(
                product.address(),
                staging.address(),
                product.nbytes() as usize,
            );
        }
        let stream = self.stream.stream()?;
        // SAFETY: each product is this call's reserved device product, which
        // no reader sees until the call publishes it after this task, and
        // its staging is page-locked memory the task holds until the copies
        // drained below.
        unsafe { batch.submit(stream) }?;
        self.stream.synchronize()
    }
}

/// What a finished task produced.
pub(crate) enum TaskOutput {
    Done,
    /// The framed bytes written into an encoded unit's row.
    Framed(usize),
    /// A finished MP4.
    Media(Vec<u8>),
}

impl MediaTask {
    fn execute(&self) -> crate::Result<TaskOutput> {
        match self {
            Self::Read {
                reader,
                conditions,
                products,
                upload,
            } => {
                reader.read(conditions, products)?;
                if let Some(upload) = upload {
                    upload.perform()?;
                }
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
                // The view's wrapping and the call take one handoff.
                let encoded: tvm_ffi::Bytes = Interpreter::execute(|| {
                    function.call_as(&[
                        rgb.handle()?.view(),
                        AnyView::from(&Any::from(i64::from(*frame_rate))),
                    ])
                })?;
                if row.device() != MemoryKind::Host || !row.is_contiguous() {
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
                let encoded: tvm_ffi::Bytes = Interpreter::execute(|| {
                    function.call_as(&[
                        pcm.handle()?.view(),
                        AnyView::from(&Any::from(*samples as i64)),
                        AnyView::from(&Any::from(i64::from(*rate))),
                    ])
                })?;
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

    fn run(&self) -> Result<TaskOutput> {
        let _range = crate::profiling::range(c"uniserve.host.media", None);
        self.execute()
    }

    fn release(&self) -> Result<()> {
        if let Self::EncodeUnit { frames, .. } = self {
            frames.release();
        }
        Ok(())
    }

    fn input_outcome(&self) -> Outcome {
        // Inputs are host memory read on the task's own thread: they are
        // released as soon as the action returns or is withdrawn.
        Outcome::Success(())
    }
}
