//! Startup of a video model's decoder runners on a rank.

use uniserve_ipc::MediaCall;

use super::runners::DecoderRunners;
use super::{DecodeCapacity, DecoderRunner};
use crate::ffi::DType;
use crate::model::components::Capability;
use crate::model::registry::Placed;
use crate::request::video::VideoSize;
use crate::runner::RunnerSetup;
use crate::runner::component::{
    AUDIO_GRAPHS, Component, DECODER_GRAPHS, VideoSetup, call_path, device_index, placed_device,
};
use crate::{Error, Result};

impl DecoderRunners {
    /// Build the video decoder of a placed component.
    pub(crate) fn start_video(
        &mut self,
        setup: &mut VideoSetup<'_>,
        placed: &Placed,
    ) -> Result<()> {
        let (loaded, model) = (setup.loaded, setup.model);
        let plan = &setup.capacity.media;
        let attention = setup.attention.clone();
        let device = placed_device(placed);
        let (stream, scratch) = setup.stream(MediaCall::VideoDecoding, device)?;
        let graph_pool = setup.graph_pool(DECODER_GRAPHS, &placed.name, device)?;
        let mut capacity = DecodeCapacity {
            latent: [0, 0],
            frames: [0, 0, 0],
            overlap: (0, DType::F16),
        };
        let longest = model
            .unit_frames(plan.max_frames())?
            .into_iter()
            .max()
            .unwrap_or(0);
        for &canvas in &plan.canvases {
            let size = model.overlap(canvas)?;
            capacity.overlap = (capacity.overlap.0.max(size.0), size.1);
            let layout = model.layout_size(&VideoSize::plain(
                plan.max_frames(),
                canvas,
                plan.max_text_tokens,
            ))?;
            let latent = model.latent_shape("video", &layout)?;
            let [rows, width] = latent[..] else {
                return Err(Error::Interface("a video latent is [rows, width]".into()));
            };
            capacity.latent = [capacity.latent[0].max(rows), width];
            capacity.frames = [
                u64::from(longest),
                capacity.frames[1].max(u64::from(canvas.height)),
                capacity.frames[2].max(u64::from(canvas.width)),
            ];
        }
        let runner = DecoderRunner::video(
            RunnerSetup {
                loaded,
                path: call_path(placed, Capability::VideoDecoder)?,
                stream,
                device: device_index(device)?,
                attention: attention.clone(),
                scratch,
            },
            capacity,
            setup.info.request_slots,
            graph_pool,
        )?;
        self.video = Some(Component {
            name: placed.name.clone(),
            config: placed.config.clone(),
            device,
            runner,
        });
        Ok(())
    }

    /// Build the audio decoder of a placed component.
    pub(crate) fn start_audio(
        &mut self,
        setup: &mut VideoSetup<'_>,
        placed: &Placed,
    ) -> Result<()> {
        let (loaded, model) = (setup.loaded, setup.model);
        let plan = &setup.capacity.media;
        let attention = setup.attention.clone();
        let device = placed_device(placed);
        let (stream, scratch) = setup.stream(MediaCall::AudioDecoding, device)?;
        let graph_pool = setup.graph_pool(AUDIO_GRAPHS, &placed.name, device)?;
        // A track's latent does not depend on the canvas.
        let layout = model.layout_size(&VideoSize::plain(
            plan.max_frames(),
            plan.canvases[0],
            plan.max_text_tokens,
        ))?;
        let latent = model.latent_shape("audio", &layout)?;
        let [rows, width] = latent[..] else {
            return Err(Error::Interface("an audio latent is [rows, width]".into()));
        };
        let runner = DecoderRunner::audio(
            RunnerSetup {
                loaded,
                path: call_path(placed, Capability::AudioDecoder)?,
                stream,
                device: device_index(device)?,
                attention: attention.clone(),
                scratch,
            },
            [rows, width],
            model.track_samples(plan.max_frames())?,
            graph_pool,
        )?;
        self.audio = Some(Component {
            name: placed.name.clone(),
            config: placed.config.clone(),
            device,
            runner,
        });
        Ok(())
    }
}
