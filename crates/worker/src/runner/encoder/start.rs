//! Startup of a video model's encoder runners on a rank.

use ipc::MediaCall;
use models::video::plan::ProcessorConfig;

use super::runners::EncoderRunners;
use super::{ENCODER, EncoderConfig, EncoderRunner, TextSetup};
use crate::config::placement::Placed;
use crate::executor::video::extent;
use crate::ffi::{DType, Interpreter, texts};
use crate::runner::RunnerSetup;
use crate::runner::component::{
    Component, LATENT_GRAPHS, TEXT_GRAPHS, VideoSetup, call_path, cuda_device,
};
use crate::{Error, Result};

impl EncoderRunners {
    /// Build the text encoder of a placed component and, when the component
    /// encodes vision patches, the vision tower bound with it.
    pub(crate) fn start_text(&mut self, setup: &mut VideoSetup<'_>, placed: &Placed) -> Result<()> {
        let (loaded, config) = (setup.loaded, setup.config);
        let plan = &setup.capacity.media;
        let attention = setup.attention.clone();
        let device = placed.device;
        // The media reader packs a request's vision blocks into the
        // rows of `vision_pixels`; a deployment without conditioned
        // tasks declares none.
        let vision_pixels = setup
            .capacity
            .outputs
            .get("media_reader")
            .and_then(|outputs| outputs.iter().find(|output| output.name == "vision_pixels"));
        let vision_rows = vision_pixels
            .and_then(|output| output.shape_bound.dims.first())
            .map_or(0, |dim| extent(dim) as usize);
        let path = placed
            .calls
            .iter()
            .find(|call| call.is(crate::config::placement::Capability::TextEncoder))
            .map(|call| call.path.clone())
            .unwrap_or_default();
        let (stream, scratch) = setup.stream(MediaCall::TextEncoding, device)?;
        self.text_pool = setup.graph_pool(TEXT_GRAPHS, device)?;
        let (mut runner, mut tower) = EncoderRunner::text(TextSetup {
            loaded,
            component: &placed.name,
            path: &path,
            device: cuda_device(device)?,
            stream,
            scratch,
            attention: attention.clone(),
            dtype: crate::runner::causal_lm::model_dtype(&config.model_dtype)?,
            config: EncoderConfig {
                // The text encoder derives its vision blocks'
                // bound from the row budget.
                max_images: 1,
                row_budget: vision_rows.max(1),
                text_capacities: plan
                    .text_capacities
                    .iter()
                    .map(|&capacity| capacity as usize)
                    .collect(),
                max_text_tokens: plan.max_text_tokens as usize,
            },
            vision: placed.has(crate::config::placement::Capability::PatchEncoder),
        })?;
        // Under graphs the vision tower encodes only blocks of the grids it
        // captured: an fl2va keyframe's at each admitted canvas, which the
        // server plans with the checkpoint's processor configs.
        if let (Some(tower), Some(_), Some(pixels)) = (&mut tower, &self.text_pool, vision_pixels) {
            let processor = texts(
                Interpreter::import(&format!("{ENCODER}.vision_processor"))?
                    .call_values(&[loaded.to_any()])?,
            )
            .map_err(|error| Error::numerical("vision_processor", &error))?;
            let [image, video] = &processor[..] else {
                return Err(Error::Unsupported(
                    "under CUDA graphs the vision tower captures the keyframe grids its \
                     processor configs plan, which the model does not keep"
                        .into(),
                ));
            };
            let processor = ProcessorConfig::parse(image, video)
                .map_err(|error| Error::Interface(format!("the vision processor: {error}")))?;
            let grids = plan
                .admitted
                .iter()
                .map(|served| {
                    processor
                        .image_grid(served.canvas.height, served.canvas.width)
                        .map(|grid| [grid.t, grid.h, grid.w])
                        .map_err(|error| Error::Invalid(error.to_string()))
                })
                .collect::<Result<Vec<_>>>()?;
            let width = pixels
                .shape_bound
                .dims
                .get(1)
                .map(extent)
                .ok_or_else(|| Error::Interface("vision pixels are rows".into()))?;
            tower.capture_blocks(grids, width as i64, DType::from(pixels.dtype))?;
            // A prompt holding the blocks' features encodes through its
            // capacity's graph of prompts with vision.
            if let super::EncoderInfo::Vision(vision) = tower.info {
                runner.capture_vision(i64::from(vision.output_size), vision.output_dtype)?;
            }
        }
        self.text = Some(Component {
            name: placed.name.clone(),
            config: placed.config.clone(),
            device,
            runner,
        });
        self.vision = tower;
        Ok(())
    }

    /// Build the condition latent encoders of a placed component.
    pub(crate) fn start_conditions(
        &mut self,
        setup: &mut VideoSetup<'_>,
        placed: &Placed,
    ) -> Result<()> {
        let loaded = setup.loaded;
        let attention = setup.attention.clone();
        let device = placed.device;
        let canvases: Vec<models::video::Canvas> = setup
            .capacity
            .media
            .admitted
            .iter()
            .map(|served| served.canvas)
            .collect();
        // The rows a call of the video encoder writes.
        let rows = setup
            .capacity
            .outputs
            .get(placed.name.as_str())
            .and_then(|outputs| {
                outputs
                    .iter()
                    .find(|output| output.name == "condition_video_latents")
            })
            .and_then(|output| {
                let width = output.shape_bound.dims.get(1).map(extent)?;
                Some((width as i64, DType::from(output.dtype)))
            });
        let (stream, scratch) = setup.stream(MediaCall::LatentEncoding, device)?;
        // The pool the still-frame tile encoding is captured into.
        self.latent_pool = setup.graph_pool(LATENT_GRAPHS, device)?;
        let setup = |capability| -> Result<RunnerSetup<'_>> {
            Ok(RunnerSetup {
                loaded,
                path: call_path(placed, capability)?,
                stream: stream.clone(),
                device: cuda_device(device)?,
                attention: attention.clone(),
                scratch: scratch.clone(),
            })
        };
        let runner = EncoderRunner::conditions(
            setup(crate::config::placement::Capability::VideoEncoder)?,
            setup(crate::config::placement::Capability::AudioEncoder)?,
        )?;
        // Under graphs the encoders serve only the keyframe encodings they
        // capture: an fl2va request's keyframes, at most two, fitted to an
        // admitted canvas, as many in one call as a rank takes units.
        if let (Some(_), Some(rows)) = (&self.latent_pool, rows) {
            let per_call = if placed.config.distribution.is_some() {
                placed.config.units_per_rank.min(2) as u32
            } else {
                2
            };
            self.keyframes = Some((canvases, per_call, rows));
        }
        self.latent = Some(Component {
            name: placed.name.clone(),
            config: placed.config.clone(),
            device,
            runner,
        });
        Ok(())
    }
}
