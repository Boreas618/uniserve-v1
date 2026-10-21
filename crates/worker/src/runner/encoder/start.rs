//! Startup of a video model's encoder runners on a rank.

use ipc::MediaCall;

use super::runners::EncoderRunners;
use super::{EncoderConfig, EncoderRunner, TextSetup};
use crate::Result;
use crate::config::placement::Placed;
use crate::executor::video::extent;
use crate::runner::RunnerSetup;
use crate::runner::component::{
    Component, LATENT_GRAPHS, TEXT_GRAPHS, VideoSetup, call_path, cuda_device,
};

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
        let vision_rows = setup
            .capacity
            .outputs
            .get("media_reader")
            .and_then(|outputs| outputs.iter().find(|output| output.name == "vision_pixels"))
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
        let (runner, tower) = EncoderRunner::text(TextSetup {
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
        self.latent = Some(Component {
            name: placed.name.clone(),
            config: placed.config.clone(),
            device,
            runner,
        });
        Ok(())
    }
}
