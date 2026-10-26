//! The image host resource of an image model's lanes ([`Images`]) and the
//! host preparation of a batch's inline input images ([`InlineImages`]).
//!
//! A lane whose manifest names the `images` resource reads request images
//! the engine sent inline (base64 payloads) or encodes the images it
//! decodes to PNG. Both run on the rank's image host lane beside device
//! work: an encoding batch's inline images are decoded and transformed by
//! the model's image policy ([`ImagePreparation`]) into page-locked blocks
//! before the batch's forward begins ([`prepare_inputs`]), so the
//! executor's preparation knows each image's facts and its execution reads
//! the prepared pixels; a decoding batch's PNG encodings run once its pixels
//! reached the host (`crate::executor::image`).

use std::sync::{Arc, Weak};

use cuda::DeviceId;
use ipc::{CallKind, MediaCall};
use tvm_ffi::Any;

use crate::error::WorkerError;
use crate::executor::Submission;
use crate::executor::backend::{WorkerBackend, WorkerBatch};
use crate::ffi::{DType, Function, Handle, Interpreter};
use crate::host::image::{ImageResult, ImageTask, ImageWork};
use crate::host::{HostLane, HostTask};
use crate::memory::PinnedPool;
use crate::memory::allocation::StorageView;
use crate::{Error, Outcome, Result};

/// The Python class whose static methods prepare request images.
const ENCODER: &str = "uniserve.runner.encoder.EncoderRunner";

/// Host workers preparing and encoding images.
const HOST_WORKERS: usize = 4;

/// The image transform a preparation applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Tower {
    /// The understanding tower (`ImageProcessor.vit`): vision encoding.
    Vit,
    /// The latent codec tower (`ImageProcessor.vae`): latent encoding.
    Vae,
}

impl Tower {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Vit => "vit",
            Self::Vae => "vae",
        }
    }

    /// The tower an encoding call kind runs; `None` for other kinds.
    pub(crate) fn of(kind: CallKind) -> Option<Self> {
        match kind {
            CallKind::Media(MediaCall::VisionEncoding) => Some(Self::Vit),
            CallKind::Media(MediaCall::LatentEncoding) => Some(Self::Vae),
            _ => None,
        }
    }
}

/// What preparing an image produces besides its pixels
/// (`EncoderRunner.image_facts`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ImageFacts {
    /// `[patches, row]` for a patch tower, `[channels, height, width]`
    /// otherwise.
    pub shape: Vec<i64>,
    /// The `[rows, columns]` patch grid of a patch tower.
    pub grid: Option<[u32; 2]>,
    /// The canvas the encoder's features describe.
    pub height: u32,
    pub width: u32,
}

impl ImageFacts {
    /// The facts `function` (`EncoderRunner.image_facts`) returned:
    /// `[height, width, rows, columns, *shape]`.
    fn parse(function: &Function, values: &[i64]) -> Result<Self> {
        let [height, width, rows, columns, ref shape @ ..] = values[..] else {
            return Err(Error::Interface(format!(
                "{} reports a canvas, a grid and a pixel shape",
                function.name()
            )));
        };
        let dimension = |extent: i64, name: &str| -> Result<u32> {
            u32::try_from(extent)
                .ok()
                .filter(|&extent| extent > 0)
                .ok_or_else(|| {
                    Error::Interface(format!("{} reports an invalid {name}", function.name()))
                })
        };
        // A patch tower reports its grid; other towers report zeros.
        let grid = match (rows, columns) {
            (0, 0) => None,
            (rows, columns) if rows > 0 && columns > 0 => Some([rows as u32, columns as u32]),
            _ => {
                return Err(Error::Interface(format!(
                    "{} reports a grid that is not two positive extents",
                    function.name()
                )));
            }
        };
        if shape.is_empty() || shape.iter().any(|&extent| extent < 1) {
            return Err(Error::Interface(format!(
                "{} reports an invalid pixel shape",
                function.name()
            )));
        }
        Ok(Self {
            shape: shape.to_vec(),
            grid,
            height: dimension(height, "height")?,
            width: dimension(width, "width")?,
        })
    }

    /// Values of the prepared pixels.
    pub(crate) fn elements(&self) -> i64 {
        self.shape.iter().product()
    }

    /// The facts as the executor's preparation reads them (`image.input.
    /// inline` of the prepared step).
    pub(crate) fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "shape": self.shape,
            "grid": self.grid,
            "height": self.height,
            "width": self.width,
        })
    }
}

/// One inline image prepared on the host: its facts and its page-locked
/// pixels, of the policy's pixel dtype.
#[derive(Clone, Debug)]
pub(crate) struct PreparedImage {
    pub(crate) facts: ImageFacts,
    pub(crate) pixels: StorageView,
}

/// The image policy's host preparation over a loaded model:
/// `EncoderRunner.image_size`, `image_facts` and `prepare_image`.
#[derive(Clone)]
pub(crate) struct ImagePreparation {
    loaded: Handle,
    size: Function,
    facts: Function,
    prepare: Function,
}

impl ImagePreparation {
    pub(crate) fn new(loaded: Handle) -> Result<Self> {
        let function = |name: &str| Interpreter::import(&format!("{ENCODER}.{name}"));
        Ok(Self {
            loaded,
            size: function("image_size")?,
            facts: function("image_facts")?,
            prepare: function("prepare_image")?,
        })
    }

    /// Decode and transform one inline request image on the calling (host)
    /// thread with the model's image policy, into a block of `pool` in the
    /// policy's pixel `dtype`: read the image's size from its header, plan
    /// its preparation, lease its pixels and write them.
    ///
    /// `input_images` is the request's input image count, which a shared
    /// pixel budget divides.
    pub(crate) fn prepare(
        &self,
        tower: Tower,
        payload: &str,
        input_images: u32,
        pool: &Arc<PinnedPool>,
        dtype: DType,
    ) -> Result<PreparedImage> {
        if input_images == 0 {
            return Err(Error::Invalid(
                "request admission declares no input images".into(),
            ));
        }
        let payload = Any::from(tvm_ffi::String::from(payload));
        let size = crate::ffi::integers(self.size.call_values(std::slice::from_ref(&payload))?)
            .map_err(|error| Error::numerical(self.size.name(), &error))?;
        let [height, width] = size[..] else {
            return Err(Error::Interface(format!(
                "{} reports a height and a width",
                self.size.name()
            )));
        };
        let tower = Any::from(tvm_ffi::String::from(tower.name()));
        let input_images = Any::from(i64::from(input_images));
        let values = crate::ffi::integers(self.facts.call_values(&[
            self.loaded.to_any(),
            tower.clone(),
            Any::from(height),
            Any::from(width),
            input_images.clone(),
        ])?)
        .map_err(|error| Error::numerical(self.facts.name(), &error))?;
        let facts = ImageFacts::parse(&self.facts, &values)?;

        let bytes = facts.elements() as usize * dtype.bytes();
        let pixels = StorageView::contiguous(pool.lease(bytes)?, 0, dtype, &facts.shape)?;
        self.prepare.call_values(&[
            self.loaded.to_any(),
            tower,
            payload,
            input_images,
            pixels.handle()?.to_any(),
        ])?;
        Ok(PreparedImage { facts, pixels })
    }
}

/// The rank's image host resource: the host lane that prepares inline
/// images and encodes decoded ones, the image policy and the page-locked
/// blocks prepared pixels occupy.
pub(crate) struct Images {
    pub(crate) host: HostLane<ImageTask>,
    pub(crate) preparation: ImagePreparation,
    pub(crate) pinned: Arc<PinnedPool>,
    /// The element type prepared pixels take (`EncoderRunner.pixel_dtype`).
    pub(crate) dtype: DType,
}

impl Images {
    /// The resource of a rank whose model is `loaded`, on `device`, with
    /// host tasks for at least `capacity` images at once.
    pub(crate) fn new(loaded: &Handle, device: Option<DeviceId>, capacity: usize) -> Result<Self> {
        let function = Interpreter::import(&format!("{ENCODER}.pixel_dtype"))?;
        let name: tvm_ffi::String = function.call_as(&[loaded.view()])?;
        let dtype = DType::from_name(name.as_str()).ok_or_else(|| {
            Error::Interface(format!(
                "{} reports the unknown dtype {:?}",
                function.name(),
                name.as_str()
            ))
        })?;
        Ok(Self {
            host: HostLane::new(capacity.max(HOST_WORKERS), HOST_WORKERS, "uniserve-image")?,
            preparation: ImagePreparation::new(loaded.clone())?,
            pinned: PinnedPool::new(device),
            dtype,
        })
    }

    /// Stop the host lane once the lanes drained; an aborted close leaves
    /// running tasks to the process's exit.
    pub(crate) fn close(&self, aborted: bool) -> Result<()> {
        if aborted {
            self.host.abort();
            return Ok(());
        }
        match self.host.close().into_iter().next() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }
}

/// The host-prepared inline images of an encoding batch, by call index.
pub(crate) struct InlineImages {
    tasks: Vec<(usize, Arc<HostTask<ImageTask>>)>,
}

impl InlineImages {
    fn ready(&self) -> bool {
        self.tasks.iter().all(|(_, task)| task.completion.done())
    }

    /// The prepared image of call `index`, once every preparation
    /// completed; a failed preparation fails the batch.
    pub(crate) fn image(&self, index: usize) -> Result<Option<PreparedImage>> {
        let Some((_, task)) = self.tasks.iter().find(|(call, _)| *call == index) else {
            return Ok(None);
        };
        match task.completion.outcome() {
            Some(Outcome::Success(result)) => match &*result {
                ImageResult::Prepared(image) => Ok(Some(image.clone())),
                ImageResult::Encoded(_) => Err(Error::Invariant(
                    "an image preparation produced encoded bytes".into(),
                )),
            },
            Some(Outcome::Failed(error)) => Err(Error::Invalid(format!(
                "inline image preparation failed: {error}"
            ))),
            Some(Outcome::Cancelled) | None => {
                Err(Error::State("inline image preparation did not complete"))
            }
        }
    }

    fn cancel(&self) {
        for (_, task) in &self.tasks {
            let _ = task.cancel(false);
        }
    }
}

/// Start the host preparation of an encoding batch's inline images on the
/// rank's image host lane, and report whether it completed. Notifies
/// `submission` when it does. A batch of another kind, or of a rank whose
/// lanes keep no image host resource, prepares nothing.
pub(crate) fn prepare_inputs(
    backend: &mut WorkerBackend,
    batch: &mut WorkerBatch,
    submission: &Arc<Submission>,
) -> std::result::Result<bool, WorkerError> {
    if let Some(inputs) = batch.image_inputs() {
        return Ok(inputs.ready());
    }
    let plan = Arc::clone(batch.shared_plan());
    let Some(tower) = plan.calls.first().and_then(|call| Tower::of(call.kind)) else {
        return Ok(true);
    };
    if plan.calls.iter().all(|call| call.input_image.is_none()) {
        return Ok(true);
    }
    let Some(images) = backend.lanes().and_then(|lanes| lanes.images()) else {
        return Ok(true);
    };
    let requests = backend.requests();
    let mut tasks = Vec::new();
    let reserved = (|| -> Result<()> {
        for (index, call) in plan.calls.iter().enumerate() {
            let Some(payload) = &call.input_image else {
                continue;
            };
            let input_images = requests
                .get(call.request_key.request_id.0)?
                .admission()
                .input_images;
            let task = images.host.reserve()?;
            tasks.push((index, Arc::clone(&task)));
            task.configure(ImageTask::new(ImageWork::Prepare {
                preparation: images.preparation.clone(),
                tower,
                payload: Arc::clone(payload),
                input_images,
                pool: Arc::clone(&images.pinned),
                dtype: images.dtype,
            }))?;
            let weak: Weak<Submission> = Arc::downgrade(submission);
            let notify: Box<dyn FnOnce() + Send> = Box::new(move || {
                if let Some(submission) = weak.upgrade() {
                    submission.notify_ready();
                }
            });
            if let Some(notify) = task.completion.subscribe(notify) {
                notify();
            }
            task.submit()?;
        }
        Ok(())
    })();
    let inputs = InlineImages { tasks };
    if let Err(error) = reserved {
        inputs.cancel();
        return Err(WorkerError::classify(error, Some("image preparation")));
    }
    let ready = inputs.ready();
    batch.set_image_inputs(inputs);
    Ok(ready)
}
