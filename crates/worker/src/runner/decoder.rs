//! The image decoder: completed latents to delivered pixels.
//!
//! A [`DecoderRunner`] is the Rust half of the Python
//! `uniserve.runner.decoder.DecoderRunner` for an image model's decoder (a
//! video model's decoders are the executor's, `uniserve.executor.video`).
//! An image decoder of a bound component decodes images
//! whose latent rows the caller gathered into the latent pool's step
//! workspace, optionally publishes the decoded values as image products,
//! and quantizes every image to packed 8-bit HWC pixels in its staging
//! bytes. The caller copies each
//! image's staged pixels to host memory on the runner's stream and encodes
//! them there ([`crate::host::image::png_base64`]).
//!
//! Staging holds the pixels of as many of the largest images as fit its
//! bytes; a batch with more images decodes in consecutive sub-batches.
//! Images of one size decode together within a sub-batch.

mod image;
pub(crate) mod runners;
pub(crate) mod warmup;

use std::sync::Arc;

use cuda::{CopyBatch, SourceOrder, Stream};
use tvm_ffi::Any;

use crate::config::Device;
use crate::cuda_graph::GraphPool;
use crate::ffi::DType;
use crate::lane::{Completion, Job};
use crate::memory::allocation::{Storage, StorageView};
use crate::runner::ImageSetup;
use crate::runner::methods::{MethodCall, kernels_call};
use crate::stream::Issue;
use crate::{Error, Result};

use image::ImageDecoder;
pub(crate) use image::image_placement;

const DECODER: &str = "uniserve.runner.decoder.DecoderRunner";

/// One image to decode.
#[derive(Clone, Debug)]
pub struct DecodeImage {
    /// The image's first latent row in the step workspace and its rows.
    pub sample: u32,
    pub units: u32,
    pub height: u32,
    pub width: u32,
    /// The image product receiving the decoded values, if published.
    pub product: Option<StorageView>,
}

impl DecodeImage {
    /// Bytes of the image's quantized pixels.
    pub(crate) fn pixel_bytes(&self) -> usize {
        self.height as usize * self.width as usize * 3
    }
}

/// The runner of an image model's decoder.
pub struct DecoderRunner {
    image: Box<ImageDecoder>,
}

impl DecoderRunner {
    /// The runner of an image model's decoder over the latent pool's step
    /// workspace `latents` and `staging_bytes` of staging. Under CUDA graphs
    /// ([`Self::serve_image_graphs`]) it decodes images of at most
    /// `input_units` latent rows and `values` elements, each at a size it
    /// captures at startup (`DecoderRunners::capture_image`).
    pub fn image(
        setup: &ImageSetup<'_>,
        range: (f32, f32),
        latents: StorageView,
        staging_bytes: usize,
        input_units: u32,
        values: usize,
    ) -> Result<Self> {
        let image = ImageDecoder::new(setup, range, latents, staging_bytes, input_units, values)?;
        Ok(Self {
            image: Box::new(image),
        })
    }

    /// Serve the image decoder under CUDA graphs, capturing into `pools`.
    pub(crate) fn serve_image_graphs(&mut self, pools: Vec<Arc<GraphPool>>) -> Result<()> {
        self.image.set_pools(pools);
        Ok(())
    }

    /// The image decoder's graph pool identity.
    pub(crate) fn image_graph_key(&self) -> Result<usize> {
        Ok(self.image_decoder()?.graph_key())
    }

    /// Whether the image decoder serves under CUDA graphs.
    pub(crate) fn image_graphs(&self) -> Result<bool> {
        Ok(self.image_decoder()?.graphs())
    }

    /// Record in `job` the warmup and the capture of an image decoder's
    /// decode of a `height` x `width` image of `units` latent rows and its
    /// quantization of a resident image of that size; the keys it captures,
    /// in order, for [`Self::complete_image`].
    pub(crate) fn capture_image(
        &self,
        units: u32,
        height: u32,
        width: u32,
        job: &Job,
        returns: &mut Vec<Vec<i64>>,
    ) -> Result<Vec<image::OneImage>> {
        self.image_decoder()?
            .capture_size(units, height, width, job, returns)
    }

    /// Store the graph a capture of [`Self::capture_image`] took for `key`.
    pub(crate) fn complete_image(
        &mut self,
        key: image::OneImage,
        completion: &mut Completion,
    ) -> Result<()> {
        self.image.complete_capture(key, completion)
    }

    fn image_decoder(&self) -> Result<&ImageDecoder> {
        Ok(&self.image)
    }

    pub(crate) fn device(&self) -> Device {
        self.image.device
    }

    /// The runner's stream; `None` on the host.
    pub(crate) fn stream(&self) -> Option<&Stream> {
        self.image.stream.as_ref()
    }

    /// An image decoder's pixel range, `(low, high)`.
    pub(crate) fn range(&self) -> Result<(f32, f32)> {
        Ok(self.image_decoder()?.range)
    }

    /// Decode `images` at once, for a caller outside a rank, which has no
    /// lane: the decode as serving records it (`Self::record_decode`),
    /// each image's staged pixels copied out to host memory behind its
    /// sub-batch, before the next sub-batch reuses the staging bytes, all
    /// performed on the calling thread, which then waits for the stream.
    /// Returns each image's staged pixels, in input order, readable on the
    /// host.
    pub fn decode(&self, images: &[DecodeImage]) -> Result<Vec<StorageView>> {
        let job = Job::new();
        let issue = job.on(self.stream());
        let device = self.device();
        let mut staged = Vec::with_capacity(images.len());
        let mut returns = Vec::new();
        self.record_decode(
            images,
            |_, pixels| {
                let bytes = pixels.nbytes() as usize;
                let memory: Arc<dyn crate::ffi::Backing> = match device {
                    Device::Cuda(index) => Storage::pinned(index, bytes)?,
                    Device::Cpu => Storage::host(bytes),
                };
                let copy = StorageView::contiguous(memory, 0, DType::U8, pixels.shape())?;
                let mut out = CopyBatch::new(SourceOrder::Stream);
                out.push(copy.address(), pixels.address(), bytes);
                // SAFETY: the staged pixels and their fresh copy both hold
                // `bytes`; the copy follows its sub-batch's call in the job
                // and precedes the next one, and the copy's memory lives in
                // the returned view.
                unsafe { issue.copy(&out) }?;
                staged.push(copy);
                Ok(())
            },
            &job,
            &mut returns,
        )?;
        // Performing the job issues the copies; the returned host pixels
        // are complete once the stream finished them.
        issue.synchronize()?;
        let mut completion = job.perform().check().map_err(|failure| failure.error)?;
        self.check_staged(returns, &mut completion)?;
        Ok(staged)
    }

    /// Check the byte counts the decode calls of a performed job returned
    /// against what [`Self::record_decode`] expected of them, in order.
    pub(crate) fn check_staged(
        &self,
        returns: Vec<Vec<i64>>,
        completion: &mut Completion,
    ) -> Result<()> {
        let decoder = self.image_decoder()?;
        for expected in returns {
            let counts = completion.returned()?;
            if counts != expected {
                return Err(Error::Interface(format!(
                    "{} staged {counts:?} bytes of images of {expected:?} bytes",
                    decoder.decode.name()
                )));
            }
        }
        Ok(())
    }

    /// Record the decoding of `images` into `job`: each sub-batch's call
    /// and what `stage` issues for each image's staged pixels. `returns`
    /// receives the byte counts each call must return.
    pub(crate) fn record_decode(
        &self,
        images: &[DecodeImage],
        stage: impl FnMut(usize, StorageView) -> Result<()>,
        job: &Job,
        returns: &mut Vec<Vec<i64>>,
    ) -> Result<()> {
        self.image_decoder()?
            .record_decode(images, stage, job, returns)
    }

    /// Record the quantization of a resident image product holding values
    /// in `range` into `job` and what `stage` issues for its staged pixels.
    /// `returns` receives the byte count the call must return.
    pub(crate) fn quantize(
        &self,
        image: &StorageView,
        range: (f32, f32),
        stage: impl FnOnce(StorageView) -> Result<()>,
        job: &Job,
        returns: &mut Vec<Vec<i64>>,
    ) -> Result<()> {
        self.image_decoder()?
            .quantize(image, range, stage, job, returns)
    }

    /// Refuse further captures.
    pub(crate) fn seal(&mut self) {
        self.image.seal();
    }

    /// The calls that list the kernels the operators of the decoder's
    /// Python runners selected.
    pub(crate) fn kernel_calls(&self) -> Vec<MethodCall> {
        vec![kernels_call(&self.image.kernels, &self.image.runner)]
    }

    /// Release the decoder's graphs, then the execution context its Python
    /// half constructed.
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        let image = &mut self.image;
        let graphs = image.graphs.close(aborted);
        let flag = Any::from(aborted);
        graphs.and(image.close.invoke(&[image.runner.view(), (&flag).into()]))
    }
}
