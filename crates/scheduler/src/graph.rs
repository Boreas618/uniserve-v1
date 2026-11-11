//! The call graph of one video request.
//!
//! Admission builds a [`VideoGraph`] from the request's conditions: which
//! media calls the request runs, the products each one publishes with their
//! exact leading extents, and which calls read them. A request without
//! conditions is text encoding, latent preparation, denoising, both
//! decoders, both encoders and the muxer. A conditioned request first reads its
//! condition media on a CPU worker (`MediaReading`), encodes the vision blocks
//! the presentation names (`VisionEncoding`) ahead of the text encoding that
//! splices them in, and encodes its visual conditions by temporal units and
//! its audio tracks separately (`LatentEncoding`); latent preparation then
//! reads the text features and every condition latent.
//!
//! Products are named as the loaded stages declare them, so admission
//! finds each product's output on the stage serving its call by name.
//! A product laid out at the request's raster declares its raster axes, which
//! admission binds to the graph's resolution. The graph holds sizes only: it never
//! inspects media.

use crate::DiffusionRequest;
use ipc::MediaCall;

/// A video raster in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Resolution {
    pub(crate) width: u32,
    pub(crate) height: u32,
}

pub(crate) use ipc::product;

/// One product a request reserves at admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GraphProduct {
    /// The call that publishes it.
    pub(crate) call: MediaCall,
    /// The name its stage declares it under.
    pub(crate) name: &'static str,
    /// The request's exact extent of the product's device-sized axis, or
    /// `None` for a product whose every extent is static.
    pub(crate) extent: Option<u32>,
}

/// Where one call of a request's graph stands.
///
/// A call the request runs once (media reading, vision, text and audio
/// encoding, latent preparation, audio decoding) moves `Pending` →
/// `InFlight` when it is scheduled and `InFlight` → `Done` when its valid
/// result lands. The muxer runs one round at a time: each round moves it to
/// `InFlight` and back to `Pending`, and the final round, which carries no
/// media units, to `Done`. The calls that run in unit rounds (latent
/// encoding of condition units, video decoding and encoding, denoising)
/// stay `Pending`: their progress is the request's unit cursors and its
/// `Denoising` trajectory, which name the range of each round.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NodeState {
    /// Not scheduled yet (or, for the muxer, between rounds).
    Pending,
    /// Scheduled; its result has not landed.
    InFlight,
    /// Its result landed; the call is not offered again.
    Done,
}

/// The media calls of one video request and the products connecting them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VideoGraph {
    calls: Vec<MediaCall>,
    /// Each call's state, by its index in `calls`.
    states: Vec<NodeState>,
    products: Vec<GraphProduct>,
    /// The generated video's raster.
    resolution: Resolution,
    /// Denoiser rows of each temporal unit the visual latent encoding
    /// covers, condition by condition in request order.
    condition_units: Vec<u32>,
    /// Whether any condition carries an audio track to encode.
    condition_audio: bool,
}

/// The media calls every video request runs, whatever its task.
const GENERATION_CALLS: [MediaCall; 8] = [
    MediaCall::TextEncoding,
    MediaCall::LatentPreparation,
    MediaCall::Denoising,
    MediaCall::VideoDecoding,
    MediaCall::AudioDecoding,
    MediaCall::VideoEncoding,
    MediaCall::AudioEncoding,
    MediaCall::Muxing,
];

/// The media calls a request runs: the generation calls, and for a request
/// with conditions the calls reading and encoding them. Every conditioned
/// request presents vision blocks, so it always encodes vision.
pub(crate) fn video_calls(conditioned: bool) -> Vec<MediaCall> {
    let conditions: &[MediaCall] = if conditioned {
        &[
            MediaCall::MediaReading,
            MediaCall::VisionEncoding,
            MediaCall::LatentEncoding,
        ]
    } else {
        &[]
    };
    conditions
        .iter()
        .chain(GENERATION_CALLS.iter())
        .copied()
        .collect()
}

/// Whether a call's node moves through [`NodeState`]s; the calls that run
/// in unit rounds or denoising steps track their progress by cursors.
const fn staged(call: MediaCall) -> bool {
    !matches!(
        call,
        MediaCall::LatentEncoding
            | MediaCall::Denoising
            | MediaCall::VideoDecoding
            | MediaCall::VideoEncoding
    )
}

/// A request whose product extent exceeds the protocol's `u32` dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("the request's {quantity} exceed the protocol's extents")]
pub(crate) struct ExtentError {
    /// What the extent counts, in the request's terms ("condition pixels").
    quantity: &'static str,
}

impl VideoGraph {
    /// Builds the graph of a request that passed `DiffusionRequest::validate`.
    ///
    /// # Errors
    ///
    /// Returns [`ExtentError`] when a product extent does not fit the
    /// protocol's `u32` dimensions.
    pub(crate) fn new(request: &DiffusionRequest) -> Result<Self, ExtentError> {
        let extent = |quantity: &'static str, value: u64| {
            u32::try_from(value).map_err(|_| ExtentError { quantity })
        };
        let conditions = &request.conditions;
        let pixels: u64 = conditions
            .iter()
            .map(|condition| condition.pixel_bytes / 3)
            .sum();
        let samples: u64 = conditions
            .iter()
            .map(|condition| condition.audio_samples)
            .sum();
        let patches: u64 = conditions
            .iter()
            .map(|condition| condition.vision_patches)
            .sum();
        let tokens: u64 = conditions
            .iter()
            .map(|condition| u64::from(condition.vision_tokens))
            .sum();
        let video_rows: u64 = conditions.iter().map(|c| c.video_rows()).sum();
        let audio_rows: u64 = conditions.iter().map(|c| u64::from(c.audio_rows)).sum();

        let mut calls = Vec::new();
        let mut products = Vec::new();
        let mut produce = |call: MediaCall, name: &'static str, extent: Option<u32>| {
            products.push(GraphProduct { call, name, extent });
        };
        if !conditions.is_empty() {
            calls.push(MediaCall::MediaReading);
            if pixels > 0 {
                produce(
                    MediaCall::MediaReading,
                    product::CONDITION_PIXELS,
                    Some(extent("condition pixels", pixels)?),
                );
            }
            if samples > 0 {
                produce(
                    MediaCall::MediaReading,
                    product::CONDITION_SAMPLES,
                    Some(extent("condition samples", samples)?),
                );
            }
            if patches > 0 {
                calls.push(MediaCall::VisionEncoding);
                produce(
                    MediaCall::MediaReading,
                    product::VISION_PIXELS,
                    Some(extent("vision patches", patches)?),
                );
                produce(
                    MediaCall::VisionEncoding,
                    product::VISION_FEATURES,
                    Some(extent("vision tokens", tokens)?),
                );
            }
            calls.push(MediaCall::LatentEncoding);
            if video_rows > 0 {
                produce(
                    MediaCall::LatentEncoding,
                    product::CONDITION_VIDEO_LATENTS,
                    Some(extent("condition video rows", video_rows)?),
                );
            }
            if audio_rows > 0 {
                produce(
                    MediaCall::LatentEncoding,
                    product::CONDITION_AUDIO_LATENTS,
                    Some(extent("condition audio rows", audio_rows)?),
                );
            }
        }
        calls.extend(GENERATION_CALLS);
        produce(
            MediaCall::TextEncoding,
            product::CONDITIONING,
            Some(extent(
                "prompt tokens",
                request.prompt_token_ids.len() as u64,
            )?),
        );
        produce(MediaCall::Denoising, product::VIDEO_LATENTS, None);
        produce(MediaCall::Denoising, product::AUDIO_LATENTS, None);
        produce(
            MediaCall::VideoDecoding,
            product::VIDEO_UNITS,
            Some(request.sampling.video_units),
        );
        produce(
            MediaCall::VideoEncoding,
            product::ENCODED_UNITS,
            Some(request.sampling.video_units),
        );
        produce(MediaCall::AudioDecoding, product::AUDIO_SAMPLES, None);

        Ok(Self {
            states: vec![NodeState::Pending; calls.len()],
            calls,
            products,
            resolution: Resolution {
                width: request.sampling.width,
                height: request.sampling.height,
            },
            condition_units: conditions
                .iter()
                .flat_map(|condition| condition.latent_units.iter().copied())
                .collect(),
            condition_audio: audio_rows > 0,
        })
    }

    /// The media calls the request runs, in graph order.
    pub(crate) fn calls(&self) -> &[MediaCall] {
        &self.calls
    }

    /// Whether the request runs calls of this kind.
    pub(crate) fn runs(&self, call: MediaCall) -> bool {
        self.calls.contains(&call)
    }

    /// The state of the request's call of kind `call`; `Pending` for a call
    /// the request does not run.
    pub(crate) fn state(&self, call: MediaCall) -> NodeState {
        self.calls
            .iter()
            .position(|candidate| *candidate == call)
            .map_or(NodeState::Pending, |index| self.states[index])
    }

    /// Whether the call of kind `call` is still to be scheduled.
    pub(crate) fn pending(&self, call: MediaCall) -> bool {
        self.state(call) == NodeState::Pending
    }

    /// Whether the result of the call of kind `call` has landed for good.
    pub(crate) fn done(&self, call: MediaCall) -> bool {
        self.state(call) == NodeState::Done
    }

    /// Records that a call of kind `call` was scheduled: a call that runs
    /// once, or a muxing round, is in flight from here on.
    pub(crate) fn schedule(&mut self, call: MediaCall) {
        if staged(call) {
            self.set(call, NodeState::InFlight);
        }
    }

    /// Records the valid result of a call of kind `call`: a call that runs
    /// once is done, and so is the muxer after its `final_round`; an earlier
    /// muxing round leaves the muxer free for the next.
    pub(crate) fn land(&mut self, call: MediaCall, final_round: bool) {
        if !staged(call) {
            return;
        }
        let state = if call == MediaCall::Muxing && !final_round {
            NodeState::Pending
        } else {
            NodeState::Done
        };
        self.set(call, state);
    }

    /// Moves the call of kind `call` to `state`. A call the request does
    /// not run has no state to move.
    fn set(&mut self, call: MediaCall, state: NodeState) {
        if let Some(index) = self.calls.iter().position(|candidate| *candidate == call) {
            self.states[index] = state;
        }
    }

    /// The calls of this request that read the products of a call of kind
    /// `call`, in the order the graph consumes them (`MediaCall::readers`);
    /// empty for a call the request does not run.
    pub(crate) fn consumers(&self, call: MediaCall) -> Vec<MediaCall> {
        if !self.runs(call) {
            return Vec::new();
        }
        call.readers()
            .iter()
            .copied()
            .filter(|reader| self.runs(*reader))
            .collect()
    }

    /// Every product the request reserves at admission.
    pub(crate) fn products(&self) -> &[GraphProduct] {
        &self.products
    }

    /// The generated video's raster, which binds every raster axis.
    pub(crate) fn resolution(&self) -> Resolution {
        self.resolution
    }

    /// The product a call publishes under `name`.
    pub(crate) fn product(&self, name: &str) -> Option<&GraphProduct> {
        self.products.iter().find(|product| product.name == name)
    }

    /// Whether the request reads condition media before encoding.
    pub(crate) fn reads_media(&self) -> bool {
        self.runs(MediaCall::MediaReading)
    }

    /// Whether the presentation holds vision blocks to encode.
    pub(crate) fn encodes_vision(&self) -> bool {
        self.runs(MediaCall::VisionEncoding)
    }

    /// Denoiser rows of each visual condition unit, in encoding order.
    pub(crate) fn condition_units(&self) -> &[u32] {
        &self.condition_units
    }

    /// Whether any condition carries an audio track to encode.
    pub(crate) fn encodes_condition_audio(&self) -> bool {
        self.condition_audio
    }
}

#[cfg(test)]
mod tests {
    use ipc::{DiffusionSamplingParams, RequestId};

    use super::*;
    use crate::ConditionExtent;

    fn request(conditions: Vec<ConditionExtent>) -> DiffusionRequest {
        DiffusionRequest {
            request_id: RequestId(1),
            prompt_token_ids: vec![1; 40],
            conditioning: b"{}".to_vec(),
            priority: 0,
            sampling: DiffusionSamplingParams {
                num_frames: 124,
                video_units: 7,
                seed: 42,
                width: 1344,
                height: 768,
                condition_rows: conditions
                    .iter()
                    .map(|condition| condition.video_rows() as u32 + condition.audio_rows)
                    .sum(),
            },
            conditions,
            trajectory: Default::default(),
            encoder_inputs: None,
        }
    }

    /// A text request runs the generation calls alone and reserves only the
    /// prompt features, the latents and the media output products.
    #[test]
    fn a_text_request_runs_the_generation_calls_alone() {
        let graph = VideoGraph::new(&request(Vec::new())).unwrap();
        assert!(!graph.reads_media());
        assert!(!graph.runs(MediaCall::LatentEncoding));
        assert_eq!(
            graph.consumers(MediaCall::TextEncoding),
            [MediaCall::LatentPreparation]
        );
        let names = graph
            .products()
            .iter()
            .map(|product| product.name)
            .collect::<Vec<_>>();
        assert_eq!(
            names,
            [
                product::CONDITIONING,
                product::VIDEO_LATENTS,
                product::AUDIO_LATENTS,
                product::VIDEO_UNITS,
                product::ENCODED_UNITS,
                product::AUDIO_SAMPLES,
            ]
        );
        assert_eq!(
            graph.product(product::CONDITIONING).unwrap().extent,
            Some(40)
        );
    }

    /// A video reference with its soundtrack is read, its vision blocks
    /// encoded before the text, its clips encoded by units and its
    /// soundtrack separately; the products carry the request's exact sizes.
    #[test]
    fn a_reference_request_reads_and_encodes_its_conditions() {
        // 124 frames of 1344x768 with a soundtrack, sampled by the
        // conditioner on a 6x42x24 grid, and one 1344x768 image.
        let video = ConditionExtent {
            pixel_bytes: 124 * 1344 * 768 * 3,
            audio_samples: 165_334,
            vision_patches: 6 * 42 * 24,
            vision_tokens: 1512,
            latent_units: [vec![5040; 7], vec![2016]].concat(),
            audio_rows: 414,
        };
        let image = ConditionExtent {
            pixel_bytes: 1344 * 768 * 3,
            vision_patches: 48 * 84,
            vision_tokens: 1008,
            latent_units: vec![1008],
            ..ConditionExtent::default()
        };
        let graph = VideoGraph::new(&request(vec![video, image])).unwrap();

        assert_eq!(
            graph.calls()[..3],
            [
                MediaCall::MediaReading,
                MediaCall::VisionEncoding,
                MediaCall::LatentEncoding
            ]
        );
        assert_eq!(
            graph.consumers(MediaCall::MediaReading),
            [MediaCall::VisionEncoding, MediaCall::LatentEncoding]
        );
        assert_eq!(
            graph.consumers(MediaCall::VisionEncoding),
            [MediaCall::TextEncoding]
        );
        assert_eq!(
            graph.consumers(MediaCall::LatentEncoding),
            [MediaCall::LatentPreparation]
        );
        assert_eq!(graph.condition_units().len(), 9);
        assert!(graph.encodes_condition_audio());

        let extent = |name| graph.product(name).unwrap().extent.unwrap();
        assert_eq!(extent(product::CONDITION_PIXELS), (124 + 1) * 1344 * 768);
        assert_eq!(extent(product::CONDITION_SAMPLES), 165_334);
        assert_eq!(extent(product::VISION_PIXELS), 6 * 42 * 24 + 48 * 84);
        assert_eq!(extent(product::VISION_FEATURES), 1512 + 1008);
        assert_eq!(
            extent(product::CONDITION_VIDEO_LATENTS),
            7 * 5040 + 2016 + 1008
        );
        assert_eq!(extent(product::CONDITION_AUDIO_LATENTS), 414);
    }

    /// An audio reference adds only audio products: no pixels, no vision.
    #[test]
    fn an_audio_reference_adds_no_visual_products() {
        let audio = ConditionExtent {
            audio_samples: 165_600,
            audio_rows: 414,
            ..ConditionExtent::default()
        };
        let graph = VideoGraph::new(&request(vec![audio])).unwrap();
        assert!(!graph.encodes_vision());
        assert!(graph.product(product::CONDITION_PIXELS).is_none());
        assert!(graph.product(product::CONDITION_VIDEO_LATENTS).is_none());
        assert_eq!(
            graph.consumers(MediaCall::MediaReading),
            [MediaCall::LatentEncoding]
        );
    }
}
