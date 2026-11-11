//! The video pool and requests shared by the tests of video scheduling: the
//! pool's stages, their products and the ranks each stage is
//! launched on, served by `testkit::FakeBatches` with `testkit::Media`
//! responding to its media calls.

use std::collections::BTreeMap;
use std::sync::Arc;

use ipc::execution::{Inventory, PoolLayout};
use ipc::{
    CallKind, DiffusionSamplingParams, DimBound, MainProcessInfo, MediaCall, OutputInfo,
    RasterAxes, RequestId, ShapeBound, StageInfo,
};
use proto::tensor::DType;
use proto::video::Canvas;
use scheduler::{BlockConfig, BlockDistribution, DiffusionRequest, Request, VideoDenoiserInfo};
use testkit::{FakeBatches, Media};

/// Denoising steps the model advertises and every request follows.
pub(crate) const STEPS: u32 = 3;

/// Every request carries the same short prompt.
pub(crate) const PROMPT: [u32; 3] = [11, 12, 13];

/// Declares one loaded stage: the outputs its ranks report and the
/// stage config the pool is launched with, assigning it `ranks`.
///
/// A distributed stage reconstructs one media unit per rank per round
/// (`units_per_rank` of one) and keeps the default parallel config, whose
/// world size of one is what a temporal-unit distribution requires.
pub(crate) fn stage(
    name: &str,
    ranks: Vec<usize>,
    distributed: bool,
    outputs: Vec<OutputInfo>,
) -> (StageInfo, BlockConfig) {
    let mut config = BlockConfig::parallel(ranks);
    if distributed {
        config.distribution = Some(BlockDistribution::TemporalUnits);
    }

    let info = StageInfo {
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

/// Bounds of the condition products the fixture's stages declare.
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

/// The fixture's stages: the media reader, text encoder, audio decoder
/// and muxer on rank 0; the denoiser on the first `denoising_ranks` ranks, as
/// a sequence-parallel denoiser runs; and the latent encoder, video decoder
/// and video codec distributed over `decoder_ranks` ranks, each
/// reconstructing and encoding one media unit per round.
pub(crate) fn video_stages(
    decoder_ranks: usize,
    denoising_ranks: usize,
) -> Vec<(StageInfo, BlockConfig)> {
    vec![
        stage(
            "media_reader",
            vec![0],
            false,
            vec![
                rows("condition_pixels", MAX_CONDITION_PIXELS, 3, DType::U8),
                rows("condition_samples", MAX_CONDITION_ROWS * 400, 2, DType::F32),
                rows("vision_pixels", MAX_VISION_PATCHES, 1536, DType::F32),
            ],
        ),
        stage(
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
        stage(
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
        stage(
            "denoiser",
            (0..denoising_ranks).collect(),
            false,
            vec![output("video_latents", None), output("audio_latents", None)],
        ),
        stage(
            "video_decoder",
            (0..decoder_ranks).collect(),
            true,
            vec![media_units("video_units", 32)],
        ),
        stage(
            "video_codec",
            (0..decoder_ranks).collect(),
            true,
            vec![output("encoded_units", Some(32))],
        ),
        stage(
            "audio_decoder",
            vec![0],
            true,
            vec![output("audio_samples", None)],
        ),
        stage("muxer", vec![0], false, Vec::new()),
    ]
}

/// What every rank of the fixture's video pool reports: every media call, a
/// denoiser of [`STEPS`] steps that takes conditions, `host_task_capacity`
/// host tasks, and `stages`.
///
/// Calls route to stages the way a loaded model reports them: the video
/// encoder stage encodes the media units the decoder produced, and the
/// muxer encodes audio and assembles the artifact.
pub(crate) fn video_info(
    host_task_capacity: u32,
    stages: &[(StageInfo, BlockConfig)],
) -> MainProcessInfo {
    MainProcessInfo {
        supported_calls: MediaCall::ALL
            .iter()
            .map(|call| CallKind::Media(*call))
            .collect(),
        buffer_pool_bytes: 257_u64 * (256 << 20),
        encoder_cache_entries: 256,
        encoder_entry_bytes: 256 << 20,
        max_logprobs: 20,
        max_batch_calls: 1024,
        max_unresolved_calls: 2,
        video_denoiser: Some(VideoDenoiserInfo {
            takes_conditions: true,
            denoise_steps: STEPS,
            description: b"{}".to_vec(),
        }),
        host_task_capacity,
        stages: stages.iter().map(|(info, _)| info.clone()).collect(),
        media_stages: BTreeMap::from([
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
        ]),
        ..MainProcessInfo::default()
    }
}

/// The fixture's video pool, named `sim`, on one host: [`video_stages`]
/// over as many ranks as they span, each reporting [`video_info`].
///
/// A deep queue keeps queue capacity from bounding which calls are in flight
/// together, leaving that to the occupancy rules under test.
pub(crate) fn video_pool(
    decoder_ranks: usize,
    host_task_capacity: u32,
    denoising_ranks: usize,
) -> Inventory {
    let stages = video_stages(decoder_ranks, denoising_ranks);
    let info = video_info(host_task_capacity, &stages);
    let ranks = decoder_ranks.max(denoising_ranks).max(1);

    Inventory {
        group: "sim".to_owned(),
        ranks: vec![info; ranks],
        layout: PoolLayout {
            queue_depth: 64,
            ranks,
            stages: stages
                .into_iter()
                .map(|(info, config)| (info.name, config))
                .collect(),
        },
        max_model_len: 32_768,
    }
}

/// A fake worker serving `pools` whose media calls `media` responses.
pub(crate) fn video_fake(pools: Vec<Inventory>, media: &Arc<Media>) -> Arc<FakeBatches> {
    Arc::new(
        FakeBatches::new(pools)
            .with_output(Arc::clone(media).output_fn())
            .with_media(Arc::new(Media::serves)),
    )
}

/// A `t2va` request decoded in `video_units` media units.
pub(crate) fn video_request(id: u64, video_units: u32) -> Request {
    Request::Diffusion(DiffusionRequest {
        request_id: RequestId(id),
        prompt_token_ids: PROMPT.to_vec(),
        conditioning: b"{}".to_vec(),
        conditions: Vec::new(),
        priority: 0,
        sampling: DiffusionSamplingParams {
            num_frames: 16,
            video_units,
            seed: id,
            // The fixture decoder's raster (`video_stages`).
            width: 24,
            height: 16,
            condition_rows: 0,
        },
        trajectory: Default::default(),
        encoder_inputs: None,
    })
}
