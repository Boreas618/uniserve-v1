//! Profile runs of the denoiser on the meta device, over the released checkpoints' headers.
//!
//! Each run opens a checkpoint through the family, loads the network a deployment places from
//! the shard headers alone (`nn::MetaWeights`) and runs its forwards under a meta
//! context over meta tensors: the tables filled after the load, the conditioning of a prompt,
//! and one denoising step of a small and of a large layout. A meta forward computes nothing; it
//! checks every operand and states the kernel each launch would run, so a run yields the
//! stage's kernel keys and its allocator peak, which the tests print.
//!
//! The tests read the releases of [`crate::releases`] through the Hugging Face cache, so they
//! are ignored by default and fail, naming the release, when asked for without it:
//!
//! ```text
//! cargo test -p models --test it minimax_h3::denoiser_profile -- --ignored --nocapture
//! ```

use std::collections::BTreeSet;
use std::sync::Arc;

use models::minimax_h3::stage::Stage;
use nn::family::{
    ConditionMedia, DenoiserSplit, LayoutCapacity, Model, Projection, SequenceSplit, VideoModel,
    VideoPlacement,
};
use nn::vae::Tiling;
use nn::{
    Clip, ConditionShape, Conditioning, DType, MetaWeights, Split, VideoRequest, WeightLoader,
};
use nn_testing::{Recording, Release, meta, record, report};
use proto::config::{PrecisionMode, Quantization, WeightFormat};

use crate::releases::{FASTH3_4_STEP, FASTH3_4_STEP_NVFP4, FASTH3_8_STEP, MINIMAX_H3};

/// The resolution the profiled deployments serve: 16:9.
const CANVAS: (u32, u32) = (1344, 768);

/// The longest prompt the profiled deployments admit, in rows.
const TEXT_ROWS: u32 = 2048;

/// The least condition rung an engine captures, in rows.
const LEAST_RUNG: u32 = 1024;

/// The pipeline of `release` as a group that places `network` at `sequence`'s rank holds it,
/// loaded from the shard headers, in the checkpoint's own representation.
fn load(
    release: Release,
    network: Stage,
    sequence: DenoiserSplit,
) -> nn::Result<Arc<dyn VideoModel>> {
    load_under(release, network, sequence, &Quantization::default())
}

/// [`load`] under the deployment's precision policy `quantization`.
fn load_under(
    release: Release,
    network: Stage,
    sequence: DenoiserSplit,
    quantization: &Quantization,
) -> nn::Result<Arc<dyn VideoModel>> {
    let Model::Video(family) = models::minimax_h3::FAMILY.model else {
        panic!("MiniMax-H3 is a video family");
    };
    let (checkpoint, opened) = family.open(&release.sources(), quantization)?;

    // The layouts hold the longest prompt and the condition rows the network's tasks are sized
    // for; the key bound is taken over 1024 frames, as an engine sizes them.
    let stages: BTreeSet<String> = [network.name().to_owned()].into();
    let placement = VideoPlacement {
        capacity: LayoutCapacity {
            text_rows: TEXT_ROWS,
            condition_rows: opened.condition_rows(Some(&stages))?,
            frames: 1024,
        },
        stages: Some(stages),
        denoiser: sequence,
        tensor: Split::WHOLE,
        dtype: DType::BF16,
        tiling: Tiling {
            tile: 256,
            overlap: 64,
        },
        posterior_seed: 42,
    };
    let store = MetaWeights::new();
    let mut loader = WeightLoader::new(&checkpoint, &store, None);

    opened.load(&checkpoint, &mut loader, &placement)
}

/// The condition rows the layouts of a deployment of `release` placing `network` hold: those of
/// the conditions the network's tasks are sized for, as the checkpoint states them.
fn condition_rows(release: Release, network: Stage) -> u32 {
    let Model::Video(family) = models::minimax_h3::FAMILY.model else {
        panic!("MiniMax-H3 is a video family");
    };
    let (_, opened) = family
        .open(&release.sources(), &Quantization::default())
        .unwrap();
    let stages: BTreeSet<String> = [network.name().to_owned()].into();

    opened.condition_rows(Some(&stages)).unwrap()
}

/// The smallest layout the profiled deployments capture: the shortest output a request is
/// admitted at, of the network's first kind of request, on the least condition rung when the
/// kind carries conditions; it states none of its conditions.
fn small(model: &dyn VideoModel) -> VideoRequest {
    let kind = model.requests().swap_remove(0);
    let condition_rows = (!kind.roles.is_empty()).then_some(LEAST_RUNG);

    VideoRequest {
        width: CANVAS.0,
        height: CANVAS.1,
        frames: model.timelines(4.0).unwrap()[0],
        fps: model.frame_rate(),
        roles: kind.roles,
        text_tokens: TEXT_ROWS,
        conditions: Vec::new(),
        vision_spans: Vec::new(),
        condition_rows,
    }
}

/// The largest layout the profiled deployments capture: a fifteen-second output on the widest
/// condition rung, `rows` rows, of the network's last kind of request, opened with every
/// keyframe at the canvas when the kind anchors keyframes, or with the largest image reference
/// the rung holds when it takes references.
fn large(model: &dyn VideoModel, rows: u32) -> VideoRequest {
    let kind = model.requests().pop().unwrap();
    let frames = model.timeline(15.0).unwrap();
    let conditions: Vec<ConditionShape> = match kind.media {
        ConditionMedia::Canvas => kind
            .roles
            .iter()
            .map(|role| ConditionShape {
                role: *role,
                frames: 1,
                height: CANVAS.1,
                width: CANVAS.0,
                audio_samples: 0,
            })
            .collect(),
        ConditionMedia::Client => model
            .largest_conditions(CANVAS, frames, rows)
            .unwrap()
            .into_iter()
            .filter(|reference| reference.frames == 1)
            .collect(),
        ConditionMedia::None => Vec::new(),
    };

    VideoRequest {
        width: CANVAS.0,
        height: CANVAS.1,
        frames,
        fps: model.frame_rate(),
        roles: conditions.iter().map(|condition| condition.role).collect(),
        text_tokens: TEXT_ROWS,
        conditions,
        vision_spans: Vec::new(),
        condition_rows: Some(rows),
    }
}

/// Runs one denoising step of `request` on the meta device and returns its recording.
///
/// The latents and the conditioning are meta tensors of the layout's shapes: the FP32 video
/// and audio latents, and the stack-typed conditioning rows.
fn step(model: &dyn VideoModel, sequence: DenoiserSplit, request: &VideoRequest) -> Recording {
    let denoiser = model.denoiser().unwrap();
    let layout = denoiser.layout(request).unwrap();
    let schedule = denoiser.schedule(request).unwrap();
    let width = denoiser.text_width().unwrap() as usize;

    let (result, recording) = record(&sequence.axes(), || {
        let shape = layout.video;
        let video = meta(
            (
                shape.channels as usize,
                shape.frames as usize,
                shape.height as usize,
                shape.width as usize,
            ),
            model.sample_type(),
        );
        let track = layout.audio.unwrap();
        let audio = meta(
            (track.channels as usize, track.frames as usize),
            model.sample_type(),
        );
        let cond = Conditioning {
            text: meta(
                (nn::capability::prefix_rows(&layout) as usize, width),
                DType::BF16,
            ),
            source: None,
            media: Vec::new(),
            images: None,
            audio: None,
        };

        denoiser.forward(schedule[0], request, &video, Some(&audio), &cond)
    });
    result.unwrap();

    recording
}

/// Profiles the network `network` of `release` at `sequence`: the tables its load leaves to
/// fill, a prompt's conditioning, and a step of the smallest and of the largest layout the
/// deployment captures, the largest one with every keyframe when the network takes them.
fn profile(release: Release, network: Stage, sequence: DenoiserSplit) {
    let model = load(release, network, sequence).unwrap();
    let denoiser = model.denoiser().unwrap();
    let name = format!(
        "{} {} {sequence:?}",
        release.repository,
        model.denoiser_stage(),
    );

    let (result, recording) = record(&sequence.axes(), || denoiser.initialize());
    result.unwrap();
    report(&format!("{name}: initialize"), &recording);

    let small = &small(model.as_ref());
    let rows = condition_rows(release, network);

    // The prompt's conditioning: the refiner over the text capacity's rows.
    let layout = denoiser.layout(small).unwrap();
    let (result, recording) = record(&sequence.axes(), || {
        let cond = Conditioning {
            text: meta(
                (
                    nn::capability::prefix_rows(&layout) as usize,
                    denoiser.text_width().unwrap() as usize,
                ),
                DType::BF16,
            ),
            source: Some(meta(
                (layout.text_rows as usize, model.text_width() as usize),
                DType::BF16,
            )),
            media: Vec::new(),
            images: None,
            audio: None,
        };
        denoiser.prepare_conditioning(small, &cond)
    });
    result.unwrap();
    report(&format!("{name}: prepare_conditioning"), &recording);

    // The smallest layout: the shortest output, no condition.
    let layout = denoiser.layout(small).unwrap();
    assert_eq!(
        (layout.video.frames, layout.video.height, layout.video.width),
        (32, 48, 84)
    );
    report(
        &format!("{name}: step of {} frames", small.frames),
        &step(model.as_ref(), sequence, small),
    );

    // The largest layout: the longest output on the widest condition rung, opened with the
    // largest condition set of its kind.
    let large = large(model.as_ref(), rows);
    let layout = denoiser.layout(&large).unwrap();
    assert_eq!(layout.video.frames, 107);
    assert_eq!(layout.audio.unwrap().frames, 603);
    report(
        &format!(
            "{name}: step of {} frames with {} condition(s)",
            large.frames,
            large.conditions.len()
        ),
        &step(model.as_ref(), sequence, &large),
    );
}

#[test]
#[ignore = "reads the Hugging Face release MiniMaxAI/MiniMax-H3"]
fn the_base_networks_profile_a_small_and_a_large_layout() {
    profile(MINIMAX_H3, Stage::Transformer, DenoiserSplit::WHOLE);
    profile(MINIMAX_H3, Stage::TransformerRef, DenoiserSplit::WHOLE);
}

#[test]
#[ignore = "reads the Hugging Face release FastVideo/FastVideo-FastH3-8-Step-V2"]
fn the_eight_step_student_profiles_a_small_and_a_large_layout() {
    profile(FASTH3_8_STEP, Stage::Transformer, DenoiserSplit::WHOLE);
}

#[test]
#[ignore = "reads the Hugging Face release FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"]
fn the_four_step_student_profiles_a_small_and_a_large_layout() {
    profile(FASTH3_4_STEP, Stage::Transformer, DenoiserSplit::WHOLE);
}

/// A rank of a sequence split of four, under each projection: the dense network returns its
/// rows in one exchange, the sparse one interval by interval, and under the gather projection
/// each block's rows are gathered while the block before it computes.
#[test]
#[ignore = "reads the Hugging Face releases MiniMaxAI/MiniMax-H3 and \
            FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"]
fn a_rank_of_a_sequence_split_profiles_under_each_projection() {
    for projection in [Projection::Gather, Projection::Exchange] {
        let sequence = DenoiserSplit {
            ulysses: SequenceSplit {
                split: Split::new(4, 1).unwrap(),
                projection,
            },
            ..DenoiserSplit::WHOLE
        };

        profile(MINIMAX_H3, Stage::Transformer, sequence);
        profile(FASTH3_4_STEP, Stage::Transformer, sequence);
    }
}

/// A rank of each four-rank partition the deployment page offers beside Ulysses alone: the
/// tensor split, the all-gather split, the hybrid of both sequence splits, the tensor split of a
/// Ulysses split and the layer split, for the dense network and the sparse one; each rank loads
/// its share and profiles its step over its own rows.
#[test]
#[ignore = "reads the Hugging Face releases MiniMaxAI/MiniMax-H3 and \
            FastVideo/FastVideo-FastH3-8-Step-V2"]
fn a_rank_of_each_four_rank_partition_profiles_its_share() {
    let partitions = [
        DenoiserSplit {
            tensor: Split::new(4, 2).unwrap(),
            ..DenoiserSplit::WHOLE
        },
        DenoiserSplit {
            gather: Split::new(4, 1).unwrap(),
            ..DenoiserSplit::WHOLE
        },
        DenoiserSplit {
            ulysses: SequenceSplit {
                split: Split::new(2, 1).unwrap(),
                projection: Projection::Gather,
            },
            gather: Split::new(2, 1).unwrap(),
            ..DenoiserSplit::WHOLE
        },
        DenoiserSplit {
            ulysses: SequenceSplit {
                split: Split::new(2, 1).unwrap(),
                projection: Projection::Exchange,
            },
            tensor: Split::new(2, 1).unwrap(),
            ..DenoiserSplit::WHOLE
        },
        DenoiserSplit {
            layers: Split::new(4, 1).unwrap(),
            ..DenoiserSplit::WHOLE
        },
    ];
    for partition in partitions {
        profile(FASTH3_8_STEP, Stage::Transformer, partition);
        profile(MINIMAX_H3, Stage::Transformer, partition);
    }
}

/// The NVFP4 export's profile: the network's feed-forward projections are packed, so a step
/// encodes their activations and multiplies the packed weights (`nvfp4_quantize`,
/// `nvfp4_gemm`), block by block above the packed product's largest row class. The text
/// encoder, which the export's manifest does not calibrate, loads dense beside it.
#[test]
#[ignore = "reads the Hugging Face release \
            FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree-NVFP4"]
fn the_nvfp4_export_profiles_a_small_and_a_large_layout() {
    profile(
        FASTH3_4_STEP_NVFP4,
        Stage::Transformer,
        DenoiserSplit::WHOLE,
    );

    let encoder = load(
        FASTH3_4_STEP_NVFP4,
        Stage::TextEncoder,
        DenoiserSplit::WHOLE,
    )
    .unwrap();
    assert!(encoder.text_encoder().is_some());
}

/// The operators a step of the smallest layout the deployment captures launches.
fn small_step_operators(model: &dyn VideoModel) -> BTreeSet<&'static str> {
    let recording = step(model, DenoiserSplit::WHOLE, &small(model));

    recording
        .launches
        .into_iter()
        .map(|(operator, _)| operator)
        .collect()
}

/// The operators one decode of the shortest clip at the smallest served size launches, on the
/// video decoder `model` places.
fn decode_operators(model: &dyn VideoModel) -> BTreeSet<&'static str> {
    let decoder = model.video_decoder().unwrap();
    let (width, height) = CANVAS;
    // One temporal unit: the clip's seven latent frames, decoded whole.
    let clip = Clip {
        frames: 7 * 4,
        height,
        width,
    };
    let layout = decoder.layout_of(clip).unwrap();
    let rows = decoder.latent_rows(&layout).unwrap().unwrap().dims()[0];

    let block = meta((rows, 24), DType::F32);
    let raster = (3, clip.frames as usize, height as usize, width as usize);
    let video = meta(raster, decoder.element_type());
    let (decoded, recording) = record(&[], || decoder.decode(&layout, &block, Some(&video), None));
    decoded.unwrap();

    recording
        .launches
        .into_iter()
        .map(|(operator, _)| operator)
        .collect()
}

/// The precision policy, read against the released headers of the eight-step student. Each
/// mode loads: `quality` (stated or not) and `balanced` multiply the denoiser's projections
/// densely, `performance` multiplies its feed-forward in FP8 and `maximum` in NVFP4, each
/// measuring its activation's scale (`dynamic_scale`); every mode but `quality` decodes with the
/// video decoder's blocks in NVFP4, and `maximum` loads the text encoder's language layers in
/// FP8. An NVFP4 attention under an eight-way Ulysses split, whose head shares are not whole
/// tiles of the packed product, is refused naming the projection. The NVFP4 export serves its
/// packed feed-forward products without a policy, and refuses one.
#[test]
#[ignore = "reads the Hugging Face releases FastVideo/FastVideo-FastH3-8-Step-V2 and \
            FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree-NVFP4"]
fn the_precision_policy_reads_against_the_released_headers() {
    let mode = |mode| Quantization {
        mode: Some(mode),
        modules: Default::default(),
    };
    let fp8 = BTreeSet::from(["fp8_quantize", "fp8_gemm", "dynamic_scale"]);
    let nvfp4 = BTreeSet::from(["nvfp4_quantize", "nvfp4_gemm", "dynamic_scale"]);
    let none = BTreeSet::new();

    for (policy, network, decoder) in [
        (Quantization::default(), &none, &none),
        (mode(PrecisionMode::Quality), &none, &none),
        (mode(PrecisionMode::Balanced), &none, &nvfp4),
        (mode(PrecisionMode::Performance), &fp8, &nvfp4),
        (mode(PrecisionMode::Maximum), &nvfp4, &nvfp4),
    ] {
        let load = |stage| {
            load_under(FASTH3_8_STEP, stage, DenoiserSplit::WHOLE, &policy)
                .unwrap_or_else(|error| panic!("{policy}: {error}"))
        };
        let encoded = |operators: BTreeSet<&'static str>| -> BTreeSet<&'static str> {
            let all = [
                "fp8_quantize",
                "fp8_gemm",
                "nvfp4_quantize",
                "nvfp4_gemm",
                "dynamic_scale",
            ];
            operators
                .into_iter()
                .filter(|operator| all.contains(operator))
                .collect()
        };

        let operators = small_step_operators(load(Stage::Transformer).as_ref());
        assert!(operators.contains("gemm"), "{policy}: {operators:?}");
        assert_eq!(&encoded(operators), network, "{policy}: the denoiser");

        let operators = decode_operators(load(Stage::VideoDecoder).as_ref());
        assert_eq!(&encoded(operators), decoder, "{policy}: the video decoder");

        assert!(
            load(Stage::TextEncoder).text_encoder().is_some(),
            "{policy}"
        );
    }

    // Eight members' shares of 56 heads are 896 channels, and the packed product computes whole
    // tiles of 256.
    let policy = Quantization {
        mode: None,
        modules: [("attention".to_owned(), WeightFormat::Nvfp4)].into(),
    };
    let ulysses8 = DenoiserSplit {
        ulysses: SequenceSplit {
            split: Split::new(8, 3).unwrap(),
            projection: Projection::Exchange,
        },
        ..DenoiserSplit::WHOLE
    };
    let error = load_under(FASTH3_8_STEP, Stage::Transformer, ulysses8, &policy)
        .err()
        .unwrap_or_else(|| panic!("{policy} was served under eight members"));
    let message = error.to_string();
    assert!(
        message.contains("attn.to_q.weight") && message.contains("not whole tiles of 256"),
        "{message}"
    );

    let packed = load_under(
        FASTH3_4_STEP_NVFP4,
        Stage::Transformer,
        DenoiserSplit::WHOLE,
        &Quantization::default(),
    )
    .unwrap();
    let operators = small_step_operators(packed.as_ref());
    assert!(
        operators.contains("nvfp4_quantize") && operators.contains("nvfp4_gemm"),
        "{operators:?}"
    );

    let policy = mode(PrecisionMode::Quality);
    let error = load_under(
        FASTH3_4_STEP_NVFP4,
        Stage::Transformer,
        DenoiserSplit::WHOLE,
        &policy,
    )
    .err()
    .unwrap_or_else(|| panic!("the NVFP4 export took {policy}"));
    assert!(
        error
            .to_string()
            .contains("checkpoint format 'modelopt_nvfp4' owns its numerical configuration"),
        "{error}"
    );
}
