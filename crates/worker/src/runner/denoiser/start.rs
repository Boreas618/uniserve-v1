//! Startup of a video model's denoiser runner on a rank.

use uniserve_ipc::MediaCall;

use super::runners::DenoiserRunners;
use super::{Conditioning, DenoiserRunner};
use crate::executor::forward::batch::extent;
use crate::ffi::{DType, Device};
use crate::model::registry::Placed;
use crate::runner::component::{Component, DENOISER_GRAPHS, VideoSetup, placed_device};
use crate::{Error, Result};

impl DenoiserRunners {
    /// Build the video denoiser of a placed component.
    pub(crate) fn start_video(
        &mut self,
        setup: &mut VideoSetup<'_>,
        placed: &Placed,
    ) -> Result<()> {
        let info = setup.info;
        let (interface, loaded, model, config) =
            (setup.interface, setup.loaded, setup.model, setup.config);
        let plan = &setup.capacity.media;
        let attention = setup.attention.clone();
        let device = placed_device(placed);
        // The text features the denoiser's conditioner reads.
        let conditioning = info
            .components
            .iter()
            .flat_map(|component| &component.outputs)
            .find(|output| output.name == "conditioning")
            .map(|output| -> Result<(u64, DType)> {
                let width = output
                    .shape_bound
                    .dims
                    .get(1)
                    .map(extent)
                    .ok_or_else(|| Error::Interface("text features are rows".into()))?;
                Ok((
                    width,
                    crate::storage::memory::parse_dtype(output.dtype.storage_name())?,
                ))
            })
            .transpose()?;
        let envelope = setup
            .capacity
            .envelope
            .clone()
            .ok_or_else(|| Error::Invariant("a denoising rank has no slot state".into()))?;
        let (stream, scratch) = setup.stream(MediaCall::Denoising, device)?;
        let graph_pool = setup.graph_pool(DENOISER_GRAPHS, &placed.name, device)?;
        let conditioner = placed.calls.iter().find(|call| {
            call.method == "encode" && call.is(crate::model::components::Capability::Encoder)
        });
        let conditioning = match (conditioner, conditioning) {
            (Some(call), Some((width, dtype))) => Some(Conditioning {
                path: call.path.clone(),
                capacities: plan.text_capacities.clone(),
                width,
                dtype,
            }),
            (Some(_), None) => {
                return Err(Error::Unsupported(
                    "the text conditioner has no text encoder features".into(),
                ));
            }
            (None, _) => None,
        };
        let Device::Cuda(index) = device else {
            return Err(Error::Unsupported(
                "the video denoiser computes on CUDA".into(),
            ));
        };
        let runner = DenoiserRunner::new(crate::runner::denoiser::DenoiserSetup {
            interface,
            loaded,
            component: &placed.name,
            stream,
            device: index,
            maximum: plan.maximum,
            envelope,
            pages: setup.capacity.pages,
            slots: info.request_slots,
            conditioning,
            conditioning_width: u64::from(model.facts.text_condition_width),
            attention: attention.clone(),
            scratch,
            graph_pool,
            transient_layouts: config.max_request_pool_size,
        })?;
        self.video = Some(Component {
            name: placed.name.clone(),
            config: placed.config.clone(),
            device,
            runner,
        });
        Ok(())
    }
}
