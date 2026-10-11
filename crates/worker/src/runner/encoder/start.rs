//! Startup of a video model's encoder runners on a rank.

use uniserve_ipc::MediaCall;

use super::runners::EncoderRunners;
use super::{EncoderConfig, EncoderRunner, TextSetup};
use crate::Result;
use crate::executor::forward::batch::extent;
use crate::model::registry::Placed;
use crate::runner::RunnerSetup;
use crate::runner::component::{
    Component, TEXT_GRAPHS, VideoSetup, call_path, device_index, placed_device,
};

impl EncoderRunners {
    /// Build the text encoder of a placed component and, when the component
    /// encodes vision patches, the vision tower bound with it.
    pub(crate) fn start_text(&mut self, setup: &mut VideoSetup<'_>, placed: &Placed) -> Result<()> {
        let (interface, loaded, config) = (setup.interface, setup.loaded, setup.config);
        let plan = &setup.capacity.media;
        let attention = setup.attention.clone();
        let device = placed_device(placed);
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
            .find(|call| call.is(crate::model::components::Capability::TextEncoder))
            .map(|call| call.path.clone())
            .unwrap_or_default();
        let (stream, scratch) = setup.stream(MediaCall::TextEncoding, device)?;
        self.text_pool = setup.graph_pool(TEXT_GRAPHS, &placed.name, device)?;
        let (runner, tower) = EncoderRunner::text(TextSetup {
            interface,
            loaded,
            component: &placed.name,
            path: &path,
            device: device_index(device)?,
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
            vision: placed.has(crate::model::components::Capability::PatchEncoder),
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
        let (interface, loaded) = (setup.interface, setup.loaded);
        let attention = setup.attention.clone();
        let device = placed_device(placed);
        let (stream, scratch) = setup.stream(MediaCall::LatentEncoding, device)?;
        let setup = |capability| -> Result<RunnerSetup<'_>> {
            Ok(RunnerSetup {
                interface,
                loaded,
                path: call_path(placed, capability)?,
                stream: stream.clone(),
                device: device_index(device)?,
                attention: attention.clone(),
                scratch: scratch.clone(),
            })
        };
        let runner = EncoderRunner::conditions(
            setup(crate::model::components::Capability::VideoEncoder)?,
            setup(crate::model::components::Capability::AudioEncoder)?,
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
