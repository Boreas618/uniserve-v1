//! Profile runs of the denoiser on the meta device, over the released checkpoints' headers.
//!
//! Each run opens a checkpoint through the family, loads the network a deployment places from
//! the shard headers alone (`nn::MetaWeights`) and runs its forwards under a meta
//! context over meta tensors: the tables filled after the load, the conditioning of a prompt,
//! and one denoising step of a small and of a large layout. A meta forward computes nothing; it
//! checks every operand and states the kernel each launch would run, so a run yields the
//! stage's kernel keys and its allocator peak, which the tests print.
//!
//! The checkpoints are not part of the repository, so the tests are ignored by default and
//! fail when a checkpoint is missing:
//!
//! ```text
//! cargo test -p models --test it minimax_h3::denoiser_profile -- --ignored --nocapture
//! ```

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
use super::support;

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use nn::family::{
    DenoiserSplit, Model, Projection, SequenceSplit, Stage, VideoModel, VideoPlacement,
    VideoServing,
};
use nn::{
    Checkpoint, Clip, ConditionRole, ConditionShape, Conditioning, DType, MetaWeights, Split,
    VideoRequest, WeightLoader,
};
use proto::config::{PrecisionMode, Quantization, WeightFormat};
use support::{Recording, meta, recorded, recorded_at, report};

/// The released base checkpoint: dense attention, the 50-point uniform grid, both networks.
const BASE: &str = "/workspace/models/MiniMax-H3";

/// The eight-step FastH3 student: a DMD ladder under video sparse attention.
const FAST_8_STEP: &str = "/workspace/models/FastVideo-FastH3-8-Step-V2";

/// The four-step FastH3 student: a DMD ladder under video sparse attention.
const FAST_4_STEP: &str = "/workspace/models/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree";

/// The four-step student's ModelOpt NVFP4 export.
const FAST_4_STEP_NVFP4: &str =
    "/workspace/models/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree-NVFP4";

/// What the profiled deployments serve: the 16:9 resolution, outputs of four to fifteen
/// seconds and prompts of up to 2048 rows.
fn serving() -> VideoServing {
    VideoServing {
        sizes: vec![(1344, 768)],
        max_seconds: 15.0,
        text_capacities: vec![2048],
        condition_rows: None,
    }
}

/// The pipeline of the checkpoint at `root` as a group that places `network` at `sequence`'s
/// rank holds it, loaded from the shard headers, in the checkpoint's own representation.
fn load(root: &str, network: Stage, sequence: DenoiserSplit) -> nn::Result<Arc<dyn VideoModel>> {
    load_under(root, network, sequence, &Quantization::default())
}

/// [`load`] under the deployment's precision policy `quantization`.
fn load_under(
    root: &str,
    network: Stage,
    sequence: DenoiserSplit,
    quantization: &Quantization,
) -> nn::Result<Arc<dyn VideoModel>> {
    assert!(
        Path::new(root).is_dir(),
        "the profile runs read the headers of {root}, which is missing"
    );
    let checkpoint = Checkpoint::open(Path::new(root))?;
    let Model::Video(family) = models::minimax_h3::FAMILY.model else {
        panic!("MiniMax-H3 is a video family");
    };
    let opened = family.open(&checkpoint, quantization)?;

    let stages: BTreeSet<String> = [network.name().to_owned()].into();
    let placement = VideoPlacement {
        stages: Some(stages),
        denoiser: sequence,
        tensor: Split::WHOLE,
        dtype: DType::BF16,
        serving: serving(),
    };
    let store = MetaWeights::new();
    let mut loader = WeightLoader::new(&checkpoint, &store, None);

    opened.load(&checkpoint, &mut loader, &placement)
}

/// The requests the deployment captures for, with the keyframe sets each may open with.
fn candidates(model: &dyn VideoModel) -> Vec<(VideoRequest, Vec<Vec<ConditionShape>>)> {
    model.candidates(&serving()).unwrap()
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

    let (result, recording) = recorded_at(sequence, || {
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

/// Profiles the network `network` of the checkpoint at `root` at `sequence`: the tables its
/// load leaves to fill, a prompt's conditioning, and a step of the smallest and of the largest
/// layout the deployment captures, the largest one with every keyframe when the network takes
/// them.
fn profile(root: &str, network: Stage, sequence: DenoiserSplit) {
    let model = load(root, network, sequence).unwrap();
    let denoiser = model.denoiser().unwrap();
    let name = format!(
        "{} {} {sequence:?}",
        Path::new(root).file_name().unwrap().to_string_lossy(),
        model.denoiser_stage(),
    );

    let (result, recording) = recorded_at(sequence, || denoiser.initialize());
    result.unwrap();
    report(&format!("{name}: initialize"), &recording);

    let requests = candidates(model.as_ref());
    let (small, _) = requests.first().unwrap();
    let (large, openings) = requests.last().unwrap();

    // The prompt's conditioning: the refiner over the text capacity's rows.
    let layout = denoiser.layout(small).unwrap();
    let (result, recording) = recorded_at(sequence, || {
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
    // largest keyframe set the rung serves, or with an image and a video reference with its
    // soundtrack when the network takes references.
    let mut large = large.clone();
    if let Some(keyframes) = openings.last() {
        large.conditions.clone_from(keyframes);
        large.first_frames = 1;
        large.last_frames = u32::from(keyframes.len() > 1);
    } else if large.references > 0 {
        let reference = |frames: u32, side: u32, audio_samples: u32| ConditionShape {
            role: ConditionRole::Reference,
            frames,
            height: side,
            width: side,
            audio_samples,
        };
        large.conditions = vec![reference(1, 512, 0), reference(30, 384, 40_000)];
        large.references = 2;
    }
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
#[ignore = "requires /workspace/models/MiniMax-H3"]
fn the_base_networks_profile_a_small_and_a_large_layout() {
    profile(BASE, Stage::Transformer, DenoiserSplit::WHOLE);
    profile(BASE, Stage::TransformerRef, DenoiserSplit::WHOLE);
}

#[test]
#[ignore = "requires /workspace/models/FastVideo-FastH3-8-Step-V2"]
fn the_eight_step_student_profiles_a_small_and_a_large_layout() {
    profile(FAST_8_STEP, Stage::Transformer, DenoiserSplit::WHOLE);
}

#[test]
#[ignore = "requires /workspace/models/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"]
fn the_four_step_student_profiles_a_small_and_a_large_layout() {
    profile(FAST_4_STEP, Stage::Transformer, DenoiserSplit::WHOLE);
}

/// A rank of a sequence split of four, under each projection: the dense network returns its
/// rows in one exchange, the sparse one interval by interval, and under the gather projection
/// each block's rows are gathered while the block before it computes.
#[test]
#[ignore = "requires /workspace/models/MiniMax-H3 and the four-step FastH3 student"]
fn a_rank_of_a_sequence_split_profiles_under_each_projection() {
    for projection in [Projection::Gather, Projection::Exchange] {
        let sequence = DenoiserSplit {
            ulysses: SequenceSplit {
                split: Split { size: 4, rank: 1 },
                projection,
            },
            ..DenoiserSplit::WHOLE
        };

        profile(BASE, Stage::Transformer, sequence);
        profile(FAST_4_STEP, Stage::Transformer, sequence);
    }
}

/// A rank of each four-rank partition the deployment page offers beside Ulysses alone: the
/// tensor split, the all-gather split, the hybrid of both sequence splits, the tensor split of a
/// Ulysses split and the layer split, for the dense network and the sparse one; each rank loads
/// its share and profiles its step over its own rows.
#[test]
#[ignore = "requires /workspace/models/MiniMax-H3 and the eight-step FastH3 student"]
fn a_rank_of_each_four_rank_partition_profiles_its_share() {
    let partitions = [
        DenoiserSplit {
            tensor: Split { size: 4, rank: 2 },
            ..DenoiserSplit::WHOLE
        },
        DenoiserSplit {
            gather: Split { size: 4, rank: 1 },
            ..DenoiserSplit::WHOLE
        },
        DenoiserSplit {
            ulysses: SequenceSplit {
                split: Split { size: 2, rank: 1 },
                projection: Projection::Gather,
            },
            gather: Split { size: 2, rank: 1 },
            ..DenoiserSplit::WHOLE
        },
        DenoiserSplit {
            ulysses: SequenceSplit {
                split: Split { size: 2, rank: 1 },
                projection: Projection::Exchange,
            },
            tensor: Split { size: 2, rank: 1 },
            ..DenoiserSplit::WHOLE
        },
        DenoiserSplit {
            layers: Split { size: 4, rank: 1 },
            ..DenoiserSplit::WHOLE
        },
    ];
    for partition in partitions {
        profile(FAST_8_STEP, Stage::Transformer, partition);
        profile(BASE, Stage::Transformer, partition);
    }
}

/// The NVFP4 export's profile: the network's feed-forward projections are packed, so a step
/// encodes their activations and multiplies the packed weights (`nvfp4_quantize`,
/// `nvfp4_gemm`), block by block above the packed product's largest row class. The text
/// encoder, which the export's manifest does not calibrate, loads dense beside it.
#[test]
#[ignore = "requires /workspace/models/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree-NVFP4"]
fn the_nvfp4_export_profiles_a_small_and_a_large_layout() {
    profile(FAST_4_STEP_NVFP4, Stage::Transformer, DenoiserSplit::WHOLE);

    let encoder = load(FAST_4_STEP_NVFP4, Stage::TextEncoder, DenoiserSplit::WHOLE).unwrap();
    assert!(encoder.text_encoder().is_some());
}

/// The operators a step of the smallest layout the deployment captures launches.
fn small_step_operators(model: &dyn VideoModel) -> BTreeSet<&'static str> {
    let requests = candidates(model);
    let (small, _) = requests.first().unwrap();
    let (launches, _) = step(model, DenoiserSplit::WHOLE, small);

    launches.into_iter().map(|(operator, _)| operator).collect()
}

/// The operators one decode of the shortest clip at the smallest served size launches, on the
/// video decoder `model` places.
fn decode_operators(model: &dyn VideoModel) -> BTreeSet<&'static str> {
    let decoder = model.video_decoder().unwrap();
    let (width, height) = serving().sizes[0];
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
    let (decoded, (launches, _)) = recorded(|| decoder.decode(&layout, &block, Some(&video), None));
    decoded.unwrap();

    launches.into_iter().map(|(operator, _)| operator).collect()
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
#[ignore = "requires the eight-step FastH3 student and the four-step student's NVFP4 export"]
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
            load_under(FAST_8_STEP, stage, DenoiserSplit::WHOLE, &policy)
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
            split: Split { size: 8, rank: 3 },
            projection: Projection::Exchange,
        },
        ..DenoiserSplit::WHOLE
    };
    let error = load_under(FAST_8_STEP, Stage::Transformer, ulysses8, &policy)
        .err()
        .unwrap_or_else(|| panic!("{policy} was served under eight members"));
    let message = error.to_string();
    assert!(
        message.contains("attn.to_q.weight") && message.contains("not whole tiles of 256"),
        "{message}"
    );

    let packed = load_under(
        FAST_4_STEP_NVFP4,
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
        FAST_4_STEP_NVFP4,
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
