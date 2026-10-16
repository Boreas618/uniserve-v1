//! Startup of a video model's denoiser runner on a rank.

use tvm_ffi::Any;
use uniserve_ipc::MediaCall;

use super::runners::DenoiserRunners;
use super::{Conditioning, DenoiserRunner};
use crate::config::placement::Placed;
use crate::executor::video::extent;
use crate::ffi::{Device, Interpreter};
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
        let (loaded, model, config) = (setup.loaded, setup.model, setup.config);
        let plan = &setup.capacity.media;
        let attention = setup.attention.clone();
        let device = placed_device(placed);
        let envelope = setup
            .capacity
            .envelope
            .clone()
            .ok_or_else(|| Error::Invariant("a denoising rank has no slot state".into()))?;
        let (stream, scratch) = setup.stream(MediaCall::Denoising, device)?;
        let graph_pool = setup.graph_pool(DENOISER_GRAPHS, &placed.name, device)?;
        // The conditioner's call runs on one pipeline stage; only a rank of
        // that stage holds its module and constructs its runners.
        let holds = Interpreter::import("uniserve.runner.encoder.EncoderRunner.holds")?;
        let mut conditioner = None;
        for call in &placed.calls {
            let encodes =
                call.method == "encode" && call.is(crate::config::placement::Capability::Encoder);
            let path = Any::from(tvm_ffi::String::from(call.path.as_str()));
            if encodes && holds.call_as::<bool>(&[loaded.view(), (&path).into()])? {
                conditioner = Some(call);
                break;
            }
        }
        let conditioning = conditioner
            .map(|call| -> Result<Conditioning> {
                // The conditioner reads the text encoder's prompt features.
                // Every rank plans the products of all the deployment's
                // components, whichever its worker holds (the capacity's
                // `outputs`), so a denoiser placed on a worker apart from
                // the text encoder sizes its input from the same declaration
                // as one beside it; the features reach it as a product.
                let features = setup
                    .capacity
                    .outputs
                    .values()
                    .flatten()
                    .find(|output| output.name == "conditioning")
                    .ok_or_else(|| {
                        Error::Unsupported(
                            "the text conditioner has no text encoder features".into(),
                        )
                    })?;
                let width = features
                    .shape_bound
                    .dims
                    .get(1)
                    .map(extent)
                    .ok_or_else(|| Error::Interface("text features are rows".into()))?;
                Ok(Conditioning {
                    path: call.path.clone(),
                    capacities: plan.text_capacities.clone(),
                    width,
                    dtype: crate::storage::memory::parse_dtype(features.dtype.storage_name())?,
                })
            })
            .transpose()?;
        let Device::Cuda(index) = device else {
            return Err(Error::Unsupported(
                "the video denoiser computes on CUDA".into(),
            ));
        };
        let runner = DenoiserRunner::video(crate::runner::denoiser::DenoiserSetup {
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
