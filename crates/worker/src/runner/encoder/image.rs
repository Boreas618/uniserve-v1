//! Images for encoders: an image's facts and prepared pixels
//! ([`PreparedImage`]), the image policy's host preparation functions
//! ([`ImagePreparation`]), and the encoding of an image batch's vision and
//! latent encoding calls, resumed between the jobs it records
//! ([`EncodeResume`]).

use std::collections::HashMap;
use std::sync::Arc;

use cuda::{DeviceId, Stream};
use ipc::{Batch as BatchPlan, CallKind, ForwardStats, MediaCall};
use tvm_ffi::Any;

use super::{ENCODER, Encoded, Recorded};
use crate::executor::image::ImageParts;
use crate::executor::image::{ImageBatch, Progress, Resume, allocations};
use crate::executor::products::Publications;
use crate::ffi::{DType, Function, Handle, Interpreter};
use crate::lane::{Completion, Job};
use crate::memory::allocation::StorageView;
use crate::memory::{Metadata, OutputPlacement, PinnedPool, ProductBuffer, ValueRange};
use crate::runner::methods::{Argument, MethodCall};
use crate::stream::Issue;
use crate::{Error, Result};

/// The image transform a preparation applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tower {
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
}

/// What preparing an image produces besides its pixels, known before them
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
}

/// Where an image's pixels are.
#[derive(Clone, Debug)]
pub(crate) enum ImagePixels {
    /// Prepared on the host into a block of the encoder's pinned pool
    /// ([`ImagePreparation::prepare`]).
    Host(StorageView),
    /// A resident image product, which the encode prepares on its device;
    /// `signed_unit` declares values in `[-1, 1]` rather than `[0, 1]`.
    Resident {
        image: StorageView,
        signed_unit: bool,
    },
}

/// One image ready for an encoder: its facts and its pixels.
#[derive(Clone, Debug)]
pub struct PreparedImage {
    pub(crate) facts: ImageFacts,
    pub(crate) pixels: ImagePixels,
}

/// The image policy's preparation functions over a loaded model:
/// `EncoderRunner.image_size`, `image_facts` and `prepare_image`.
#[derive(Clone)]
pub struct ImagePreparation {
    loaded: Handle,
    size: Function,
    facts: Function,
    prepare: Function,
}

impl ImagePreparation {
    pub fn new(loaded: Handle) -> Result<Self> {
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
    pub fn prepare(
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
        Ok(PreparedImage {
            facts,
            pixels: ImagePixels::Host(pixels),
        })
    }

    /// Record the facts call of a resident image of `height` x `width`
    /// pixels for `tower` into `job`: a generated image, so with no input
    /// images. The job's completion returns them
    /// ([`Self::returned_facts`]).
    pub(crate) fn record_facts(&self, tower: Tower, height: u32, width: u32, job: &Job) {
        job.call_returning(MethodCall::new(
            &self.facts,
            vec![
                Argument::Handle(self.loaded.clone()),
                Argument::Text(tower.name().to_owned()),
                Argument::Integer(i64::from(height)),
                Argument::Integer(i64::from(width)),
                Argument::Integer(0),
            ],
        ));
    }

    /// The facts a recorded facts call returned.
    pub(crate) fn returned_facts(&self, values: &[i64]) -> Result<ImageFacts> {
        ImageFacts::parse(&self.facts, values)
    }
}

/// The tower an encoding kind runs.
pub(crate) fn tower(call: MediaCall) -> Result<Tower> {
    match call {
        MediaCall::VisionEncoding => Ok(Tower::Vit),
        MediaCall::LatentEncoding => Ok(Tower::Vae),
        other => Err(Error::Invalid(format!(
            "{} is not an image encoding",
            other.as_str()
        ))),
    }
}

/// The encoder serving a batch, with its device and stream.
pub(crate) fn image_placement(
    image: &ImageParts<'_>,
    plan: &BatchPlan,
) -> Result<(Option<DeviceId>, Option<Stream>, usize)> {
    let first = plan
        .calls
        .first()
        .ok_or_else(|| Error::Invariant("an image batch has calls".into()))?;
    let runner = &image.encoders.image[image.encoders.route(&first.component, first.kind)?];
    let stream = runner.stream().cloned();
    Ok((runner.device().cuda(), stream, 0))
}

/// An image encoding between two of its jobs.
pub(crate) enum EncodeResume {
    /// The job on the lane reads the resident images' facts.
    Facts {
        call: MediaCall,
        /// Each active call's image; none yet for a resident one.
        images: Vec<Option<PreparedImage>>,
        /// The resident images by position in `images`: the image product
        /// and whether its values are in `[-1, 1]`.
        resident: Vec<(usize, StorageView, bool)>,
    },
    /// The job on the lane holds an unpacked encoder's calls: its
    /// completion carries the shapes they wrote.
    Shapes {
        images: Vec<PreparedImage>,
        writes: Vec<ProductBuffer>,
        recorded: Recorded,
    },
}

impl ImageBatch<'_> {
    /// Record the encoding of every active call's image into its feature
    /// product, up to its first job boundary. `prepared` holds the
    /// host-prepared inline images by call index.
    ///
    /// The encoding's jobs: the read of the resident images' facts, when
    /// the batch has any; the encode calls; and, after an unpacked
    /// encoder's calls returned the shapes they wrote, the features' fences
    /// and exports.
    pub(crate) fn encode(
        &mut self,
        call: MediaCall,
        prepared: HashMap<usize, PreparedImage>,
    ) -> Result<Progress> {
        // The forward records into the batch's job, taken for its length so
        // that the batch's owners stay borrowable beside it.
        let job = std::mem::take(&mut self.job);
        let progress = self.prepare_images(&job, call, prepared);
        self.job = job;
        progress
    }

    /// Continue the encoding from the `completion` of the job its last
    /// boundary ended.
    pub(crate) fn resume_encode(
        &mut self,
        resume: EncodeResume,
        completion: &mut Completion,
    ) -> Result<Progress> {
        let job = std::mem::take(&mut self.job);
        let progress = self.continue_encode(&job, resume, completion);
        self.job = job;
        progress
    }

    fn continue_encode(
        &mut self,
        job: &Job,
        resume: EncodeResume,
        completion: &mut Completion,
    ) -> Result<Progress> {
        match resume {
            EncodeResume::Facts {
                call,
                mut images,
                resident,
            } => {
                let preparation = self.parts.policy.preparation();
                for (position, image, signed_unit) in resident {
                    images[position] = Some(PreparedImage {
                        facts: preparation.returned_facts(&completion.returned()?)?,
                        pixels: ImagePixels::Resident { image, signed_unit },
                    });
                }
                self.encode_images(job, call, images)
            }
            EncodeResume::Shapes {
                images,
                writes,
                recorded,
            } => {
                let encoded = recorded.finish(completion)?;
                self.publish_features(job, &images, &writes, &encoded)
                    .map(Progress::done)
            }
        }
    }

    /// The image encoder serving the batch's calls of `call`.
    fn encoder_route(&self, call: MediaCall) -> Result<usize> {
        let first = self
            .active()
            .first()
            .copied()
            .ok_or_else(|| Error::Invariant("an encoding batch lost its calls".into()))?;
        self.parts
            .encoders
            .route(&self.plan.calls[first].component, CallKind::Media(call))
    }

    /// Lease the resident images and record the reads of their facts, then
    /// go on to the encode when every image is prepared on the host.
    fn prepare_images(
        &mut self,
        job: &Job,
        call: MediaCall,
        mut prepared: HashMap<usize, PreparedImage>,
    ) -> Result<Progress> {
        let tower = tower(call)?;
        let plan = self.plan;
        let active = self.active();
        if active.is_empty() {
            return Ok(Progress::done(ForwardStats::default()));
        }
        let target = self.target();
        let stream = self.batch.stream.clone();

        // Every image, inline or resident, ready for the encoder. A
        // resident image's facts are a call of the job; the encode prepares
        // its pixels.
        let mut images: Vec<Option<PreparedImage>> = Vec::with_capacity(active.len());
        let mut resident = Vec::new();
        for (position, &index) in active.iter().enumerate() {
            let call = &plan.calls[index];
            if call.encoder_output.is_none() {
                return Err(Error::Invalid(
                    "image encoding requires an encoder output".into(),
                ));
            }
            match (call.input_image.is_some(), &call.image_input) {
                (true, None) => {
                    images.push(Some(prepared.remove(&index).ok_or_else(|| {
                        Error::Invariant("an inline image was not prepared".into())
                    })?))
                }
                (false, Some(reference)) => {
                    let read = self.storage.tensors.consume(
                        reference,
                        call.call_id,
                        Some(target),
                        job.on(stream.as_ref()),
                    )?;
                    let range = match read.metadata() {
                        Some(Metadata::Image {
                            value_range: Some(range),
                            ..
                        }) => range,
                        _ => {
                            self.batch.reads.push(read);
                            return Err(Error::Invalid(
                                "resident image encoding requires an image value range".into(),
                            ));
                        }
                    };
                    let value = read.value().clone();
                    self.batch.reads.push(read);
                    // An image product is [3, height, width], or a batch of
                    // one of them.
                    let (height, width) = match value.shape() {
                        [.., 3, height, width] => (*height, *width),
                        _ => {
                            return Err(Error::Invalid(
                                "a resident image is [3, height, width]".into(),
                            ));
                        }
                    };
                    let extent = |side: i64| {
                        u32::try_from(side)
                            .map_err(|_| Error::Invalid("a resident image is too large".into()))
                    };
                    self.parts.policy.preparation().record_facts(
                        tower,
                        extent(height)?,
                        extent(width)?,
                        job,
                    );
                    resident.push((position, value, range == ValueRange::SignedUnit));
                    images.push(None);
                }
                _ => {
                    return Err(Error::Invalid(
                        "image encoding requires exactly one inline or resident image".into(),
                    ));
                }
            }
        }
        if resident.is_empty() {
            return self.encode_images(job, call, images);
        }

        // The resident images' facts come back with the job that read them.
        Ok(Progress::Boundary(Resume::Encode(EncodeResume::Facts {
            call,
            images,
            resident,
        })))
    }

    /// Reserve the feature products and record the encode of the prepared
    /// `images`, then the features' publication when their shapes are
    /// known already.
    fn encode_images(
        &mut self,
        job: &Job,
        call: MediaCall,
        images: Vec<Option<PreparedImage>>,
    ) -> Result<Progress> {
        let plan = self.plan;
        let active = self.active();
        let route = self.encoder_route(call)?;
        let target = self.target();
        let images: Vec<PreparedImage> = images
            .into_iter()
            .map(|image| image.ok_or_else(|| Error::Invariant("an image was not prepared".into())))
            .collect::<Result<_>>()?;

        // The feature products, reserved in their scheduler allocations.
        let allocations = allocations(plan);
        let bindings: Vec<_> = active
            .iter()
            .filter_map(|&index| plan.calls[index].encoder_output.clone())
            .map(|reference| (reference, target))
            .collect();
        let writes = self.storage.tensors.reserve_features(
            &bindings,
            &allocations,
            OutputPlacement::default(),
        )?;
        self.batch.writes.extend(writes.iter().cloned());

        // A latent image's posterior noise is drawn at its request's seed
        // and the request's latent encodings before it.
        let draws = match call {
            MediaCall::LatentEncoding => active
                .iter()
                .map(|&index| {
                    let request = self
                        .requests
                        .get(plan.calls[index].request_key.request_id.0)?;
                    let seed = request
                        .admission()
                        .image
                        .as_ref()
                        .and_then(|image| image.seed)
                        .unwrap_or(0);
                    Ok((seed, request.posterior_draw()?))
                })
                .collect::<Result<Vec<_>>>()?,
            _ => Vec::new(),
        };
        let runner = &mut self.parts.encoders.image[route];
        let products: Vec<StorageView> = writes.iter().map(|write| write.view()).collect();
        let borrowed: Vec<&PreparedImage> = images.iter().collect();
        let recorded = runner.record(&borrowed, &draws, &products, job)?;
        self.retain_pixels(&images, job)?;

        // An unpacked encoder's calls return the shapes they wrote, which
        // the features are fenced and exported with: the next job's work.
        if recorded.returns() {
            return Ok(Progress::Boundary(Resume::Encode(EncodeResume::Shapes {
                images,
                writes,
                recorded,
            })));
        }
        let encoded = recorded.finish(&mut Completion::default())?;
        self.publish_features(job, &images, &writes, &encoded)
            .map(Progress::done)
    }

    /// Keep the host-prepared pixels of `images` from their pool until the
    /// stream passed the work recorded in `job` that reads them. A host job
    /// reads them before it completes: its calls hold their arguments and
    /// its copies take their sources as they are recorded.
    fn retain_pixels(&self, images: &[PreparedImage], job: &Job) -> Result<()> {
        let pixels: Vec<StorageView> = images
            .iter()
            .filter_map(|image| match &image.pixels {
                ImagePixels::Host(pixels) => Some(pixels.clone()),
                ImagePixels::Resident { .. } => None,
            })
            .collect();
        let issue = job.on(self.batch.stream.as_ref());
        if pixels.is_empty() || issue.stream().is_none() {
            return Ok(());
        }
        let events = &self.storage.events;
        let fence = events.fence(issue, false)?;
        issue.defer(
            events,
            vec![fence],
            Box::new(move || {
                drop(pixels);
                Ok(())
            }),
        )
    }

    /// Fence each feature at the shape its encode wrote and issue its
    /// export; the batch's commit locates the exports.
    fn publish_features(
        &mut self,
        job: &Job,
        images: &[PreparedImage],
        writes: &[ProductBuffer],
        encoded: &Encoded,
    ) -> Result<ForwardStats> {
        let plan = self.plan;
        let active = self.active();
        let stream = self.batch.stream.clone();
        let issue = job.on(stream.as_ref());
        for (((write, image), shape), &index) in
            writes.iter().zip(images).zip(&encoded.shapes).zip(&active)
        {
            let shape: Vec<u64> = shape.iter().map(|&extent| extent as u64).collect();
            self.storage.tensors.produced(
                write,
                &shape,
                issue,
                Some(Metadata::Feature {
                    height: image.facts.height,
                    width: image.facts.width,
                }),
            )?;
            // Other ranks and workers import the feature from its export,
            // located when the batch's commit is applied.
            let export = Publications::issue_feature(
                &self.storage.tensors,
                &self.storage.transports,
                &plan.calls[index],
                write,
                self.batch.reports,
                issue,
            )?;
            if let Some(export) = export {
                self.features.push((index, write.clone(), export));
            }
        }

        let mut stats = ForwardStats::default();
        stats.cuda_graph_replays += encoded.replays;
        if encoded.replays > 0 {
            *stats.mode_counts.entry("graph_replay".into()).or_default() += encoded.replays;
        }
        if encoded.eager > 0 {
            *stats.mode_counts.entry("eager".into()).or_default() += encoded.eager;
        }
        Ok(stats)
    }
}
