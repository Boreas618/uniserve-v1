//! The decoder runners of a rank.

use ipc::{CallKind, MediaCall};

use super::{DecodeImage, DecoderRunner};
use crate::Result;
use crate::lane::{BLOCK_ON_TIMEOUT, Completion, Job, JobQueues};
use crate::memory::LatentCache;
use crate::runner::denoiser::image_model::ImageDenoiserLayout;
use crate::stream::Issue;

/// The lane kind of the image decoder's jobs.
const IMAGE: CallKind = CallKind::Media(MediaCall::ImageDecoding);

/// Perform a startup `job` on the lane serving `kind` and wait for its
/// completion.
fn perform(lanes: &JobQueues, kind: CallKind, job: Job) -> Result<Completion> {
    lanes
        .block_on(kind, job, BLOCK_ON_TIMEOUT)?
        .check()
        .map_err(|failure| failure.error)
}

/// The decoder runners of a rank.
#[derive(Default)]
pub(crate) struct DecoderRunners {
    /// An image model's decoder, with its component.
    pub(crate) image: Option<(String, DecoderRunner)>,
    /// The image sizes the image decoder captures under graphs, each with
    /// its latent rows: `(units, height, width)`.
    image_sizes: Vec<(u32, u32, u32)>,
    /// Output words of an image decoding batch's pixels at most.
    pub(crate) max_pixel_words: usize,
}

impl DecoderRunners {
    /// The decoder of an image model and the most output words a decoding
    /// batch's pixels take.
    /// Under graphs the decoder decodes the images of `sizes` alone,
    /// `(units, height, width)` each.
    pub(crate) fn image(
        decoder: Option<(String, DecoderRunner)>,
        sizes: Vec<(u32, u32, u32)>,
        max_pixel_words: usize,
    ) -> Self {
        Self {
            image: decoder,
            image_sizes: sizes,
            max_pixel_words,
        }
    }

    /// Capture the image decoder's decode and quantization at every size
    /// it serves, each warmed first, as jobs of the lane serving image
    /// decoding on `lanes`; then seal it. Without graphs it captures
    /// nothing.
    pub(crate) fn capture_image(&mut self, lanes: &JobQueues) -> Result<()> {
        let Some((_, decoder)) = &mut self.image else {
            return Ok(());
        };
        if !decoder.image_graphs()? {
            return Ok(());
        }
        for &(units, height, width) in &self.image_sizes {
            let job = Job::new();
            let mut returns = Vec::new();
            let keys = decoder.capture_image(units, height, width, &job, &mut returns)?;
            job.on(decoder.stream()).synchronize()?;
            let mut completion = perform(lanes, IMAGE, job)?;
            decoder.check_staged(returns, &mut completion)?;
            for key in keys {
                decoder.complete_image(key, &mut completion)?;
            }
        }
        decoder.seal();
        tracing::info!(
            sizes = self.image_sizes.len(),
            "captured the image decoder's decode and quantization"
        );
        Ok(())
    }

    /// Warm the image decoder on one black `side` x `side` image whose
    /// latent rows are the head of the step workspace of `latents`: one job
    /// of the lane on `lanes` serving image decoding zeroes the rows,
    /// decodes the image as serving records the decode and waits for the
    /// stream.
    pub(crate) fn warm_image(
        &self,
        latents: Option<&LatentCache>,
        layout: Option<ImageDenoiserLayout>,
        side: u32,
        lanes: &JobQueues,
    ) -> Result<()> {
        let (Some((_, decoder)), Some(latents), Some(layout)) = (&self.image, latents, layout)
        else {
            return Ok(());
        };
        // Under graphs every capture warms its size first.
        if decoder.image_graphs()? {
            return Ok(());
        }
        let units = layout.latent_rows(side, side)?;
        let workspace = latents.workspace()?;
        let job = Job::new();
        let issue = job.on(decoder.stream());
        // SAFETY: the latent cache keeps its workspace past this warmup,
        // whose jobs are performed before it returns; nothing else uses the
        // workspace before serving starts.
        unsafe { workspace.narrow(0, 0, i64::from(units))?.zero(issue) }?;
        let mut returns = Vec::new();
        // The staged pixels are discarded.
        decoder.record_decode(
            &[DecodeImage {
                sample: 0,
                units,
                height: side,
                width: side,
                product: None,
            }],
            |_, _| Ok(()),
            &job,
            &mut returns,
        )?;
        issue.synchronize()?;
        let mut completion = perform(lanes, IMAGE, job)?;
        decoder.check_staged(returns, &mut completion)
    }
}
