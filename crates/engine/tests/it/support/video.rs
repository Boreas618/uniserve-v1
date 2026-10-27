//! A simulated video worker and its requests, shared by the tests of video
//! scheduling: the worker's components, their products and the ranks each
//! component is launched on.

use std::collections::BTreeMap;

use engine::{
    ComponentConfig, ComponentDistribution, DiffusionRequest, Request, SimEngine,
    VideoDenoiserInfo, VideoTaskInfo,
};
use ipc::{
    CallKind, ComponentInfo, DType, DiffusionSamplingParams, DimBound, MediaCall, OutputInfo,
    RasterAxes, RequestId, ShapeBound,
};
use models::video::Canvas;

/// Denoising steps the simulated model advertises and every request follows.
pub(crate) const STEPS: u32 = 3;

/// Every request carries the same short prompt.
pub(crate) const PROMPT: [u32; 3] = [11, 12, 13];

/// Declares one loaded component: the outputs its ranks report and the
/// component config the pool is launched with, assigning it `ranks`.
///
/// A distributed component reconstructs one media unit per rank per round
/// (`units_per_rank` of one) and keeps the default parallel config, whose
/// world size of one is what a temporal-unit distribution requires.
pub(crate) fn component(
    name: &str,
    ranks: Vec<usize>,
    distributed: bool,
    outputs: Vec<OutputInfo>,
) -> (ComponentInfo, ComponentConfig) {
    let mut config = ComponentConfig::parallel(ranks, Default::default());
    if distributed {
        config.distribution = Some(ComponentDistribution::TemporalUnits);
    }

    let info = ComponentInfo {
        name: name.to_owned(),
        outputs,
    };
    (info, config)
}

/// A small tensor result, indexed by media unit or row along a device-actual
/// leading axis when `units` bounds it.
pub(crate) fn output(name: &str, units: Option<u32>) -> OutputInfo {
    let mut dims = vec![DimBound::Static(4)];
    if let Some(max) = units {
        dims.insert(0, DimBound::Device { max });
    }

    OutputInfo {
        name: name.to_owned(),
        dtype: DType::BF16,
        shape_bound: ShapeBound { dims },
        raster_axes: None,
    }
}

/// Rows of a condition product, bounded at `max` rows of `width` elements.
pub(crate) fn rows(name: &str, max: u32, width: u32, dtype: DType) -> OutputInfo {
    OutputInfo {
        name: name.to_owned(),
        dtype,
        shape_bound: ShapeBound {
            dims: vec![DimBound::Device { max }, DimBound::Static(width)],
        },
        raster_axes: None,
    }
}

/// Bounds of the condition products the fixture's components declare.
pub(crate) const MAX_CONDITION_ROWS: u32 = 4_096;
pub(crate) const MAX_CONDITION_PIXELS: u32 = 1 << 20;
pub(crate) const MAX_VISION_PATCHES: u32 = 4_096;

/// The largest raster the fixture's video decoder admits.
pub(crate) const MAX_RASTER: Canvas = Canvas {
    width: 32,
    height: 32,
};

/// A decoded media unit result as a video decoder declares it: units along a
/// device-actual leading axis, then frames, the raster's height and width at
/// their largest admitted extents, and RGB channels.
pub(crate) fn media_units(name: &str, units: u32) -> OutputInfo {
    OutputInfo {
        name: name.to_owned(),
        dtype: DType::U8,
        shape_bound: ShapeBound {
            dims: vec![
                DimBound::Device { max: units },
                DimBound::Static(4),
                DimBound::Static(MAX_RASTER.height),
                DimBound::Static(MAX_RASTER.width),
                DimBound::Static(3),
            ],
        },
        raster_axes: Some(RasterAxes {
            height: 2,
            width: 3,
        }),
    }
}

/// A video worker whose text encoder, denoiser, audio decoder and muxer live on
/// rank 0 and whose video decoder and video codec are distributed over
/// `decoder_ranks` ranks, each reconstructing and encoding one media unit per
/// round.
///
/// Calls route to components the way a loaded model reports them: the video
/// encoder component encodes the media units the decoder produced, and the
/// muxer encodes audio and assembles the artifact.
pub(crate) fn video_worker(decoder_ranks: usize, host_task_capacity: u32) -> SimEngine {
    let mut sim = SimEngine::new();
    // A deep executor queue keeps queue capacity from bounding which calls
    // are in flight together, leaving that to the occupancy rules under test.
    sim.set_queue_depth(64);
    sim.set_results_on_wait(true);

    let info = sim.mut_info_for_test();
    info.supported_calls = MediaCall::ALL
        .iter()
        .filter(|call| **call != MediaCall::ImageDecoding)
        .map(|call| CallKind::Media(*call))
        .collect();
    info.video_denoiser = Some(VideoDenoiserInfo {
        tasks: [("t2va", false), ("fl2va", true), ("ref2va", true)]
            .map(|(name, conditioned)| VideoTaskInfo {
                name: name.to_owned(),
                conditioned,
            })
            .to_vec(),
        denoise_steps: STEPS,
        description: b"{}".to_vec(),
    });
    // The denoiser's latent pool holds the reserved sentinel page plus two
    // pages of samples per request slot.
    info.latent_page_units = 256;
    info.latent_pages = 2 * info.request_slots + 1;
    info.host_task_capacity = host_task_capacity;
    let components = vec![
        component(
            "media_reader",
            vec![0],
            false,
            vec![
                rows("condition_pixels", MAX_CONDITION_PIXELS, 3, DType::U8),
                rows("condition_samples", MAX_CONDITION_ROWS * 400, 2, DType::F32),
                rows("vision_pixels", MAX_VISION_PATCHES, 1536, DType::F32),
            ],
        ),
        component(
            "text_encoder",
            vec![0],
            false,
            vec![
                output("conditioning", Some(64)),
                rows(
                    "vision_features",
                    MAX_VISION_PATCHES / 4,
                    20_480,
                    DType::BF16,
                ),
            ],
        ),
        component(
            "latent_encoder",
            (0..decoder_ranks).collect(),
            true,
            vec![
                rows(
                    "condition_video_latents",
                    MAX_CONDITION_ROWS,
                    96,
                    DType::F32,
                ),
                rows(
                    "condition_audio_latents",
                    MAX_CONDITION_ROWS,
                    32,
                    DType::F32,
                ),
            ],
        ),
        component(
            "denoiser",
            vec![0],
            false,
            vec![output("video_latents", None), output("audio_latents", None)],
        ),
        component(
            "video_decoder",
            (0..decoder_ranks).collect(),
            true,
            vec![media_units("video_units", 32)],
        ),
        component(
            "video_codec",
            (0..decoder_ranks).collect(),
            true,
            vec![output("encoded_units", Some(32))],
        ),
        component(
            "audio_decoder",
            vec![0],
            true,
            vec![output("audio_samples", None)],
        ),
        component("muxer", vec![0], false, Vec::new()),
    ];
    info.components = components.iter().map(|(info, _)| info.clone()).collect();
    info.media_components = BTreeMap::from([
        (MediaCall::MediaReading, "media_reader".to_owned()),
        (MediaCall::VisionEncoding, "text_encoder".to_owned()),
        (MediaCall::LatentEncoding, "latent_encoder".to_owned()),
        (MediaCall::TextEncoding, "text_encoder".to_owned()),
        (MediaCall::LatentPreparation, "denoiser".to_owned()),
        (MediaCall::Denoising, "denoiser".to_owned()),
        (MediaCall::VideoDecoding, "video_decoder".to_owned()),
        (MediaCall::VideoEncoding, "video_codec".to_owned()),
        (MediaCall::AudioDecoding, "audio_decoder".to_owned()),
        (MediaCall::AudioEncoding, "muxer".to_owned()),
        (MediaCall::Muxing, "muxer".to_owned()),
    ]);
    sim.place_components(
        components
            .into_iter()
            .map(|(info, config)| (info.name, config))
            .collect(),
    );

    sim
}

/// A `t2va` request decoded in `video_units` media units.
pub(crate) fn video_request(id: u64, video_units: u32) -> Request {
    Request::Diffusion(DiffusionRequest {
        request_id: RequestId(id),
        task: "t2va".to_owned(),
        prompt_token_ids: PROMPT.to_vec(),
        conditioning: b"{}".to_vec(),
        conditions: Vec::new(),
        priority: 0,
        sampling: DiffusionSamplingParams {
            num_frames: 16,
            video_units,
            seed: id,
            // The fixture decoder's raster (`video_worker`).
            width: 24,
            height: 16,
        },
        trajectory: Default::default(),
    })
}
