//! Profile runs of the denoiser on the meta device, over the released checkpoints' headers.
//!
//! Each run opens a checkpoint through the family, loads the network a deployment places from
//! the shard headers alone (`nn::WeightLoader::meta`) and runs its forwards under a meta
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
    Model, Projection, SequenceSplit, Stage, VideoModel, VideoPlacement, VideoServing,
};
use nn::{
    Checkpoint, ConditionRole, ConditionShape, Conditioning, DType, Split, VideoRequest,
    WeightLoader,
};
use support::{Recording, meta, recorded, report};

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
/// rank holds it, loaded from the shard headers.
fn load(root: &str, network: Stage, sequence: SequenceSplit) -> nn::Result<Arc<dyn VideoModel>> {
    assert!(
        Path::new(root).is_dir(),
        "the profile runs read the headers of {root}, which is missing"
    );
    let checkpoint = Checkpoint::open(Path::new(root))?;
    let Model::Video(family) = models::minimax_h3::FAMILY.model else {
        panic!("MiniMax-H3 is a video family");
    };
    let opened = family.open(&checkpoint)?;

    let stages: BTreeSet<String> = [network.name().to_owned()].into();
    let placement = VideoPlacement {
        stages: Some(stages),
        sequence,
        tensor: Split::WHOLE,
        dtype: DType::BF16,
        serving: serving(),
    };
    let mut loader = WeightLoader::meta(&checkpoint, None, None);

    opened.load(&checkpoint, &mut loader, &placement)
}

/// [`recorded`], with the context stating this rank's place in `sequence`'s group.
fn recorded_at<T>(
    sequence: SequenceSplit,
    forward: impl FnOnce() -> nn::Result<T>,
) -> (nn::Result<T>, Recording) {
    if sequence.split.size == 1 {
        return recorded(forward);
    }

    let context = backend::Context::meta(None);
    context
        .set_group(
            None,
            u64::from(sequence.split.rank),
            u64::from(sequence.split.size),
        )
        .unwrap();
    context.record_launches();

    let output = {
        let _guard = context.enter();
        forward()
    };
    let launches = context
        .take_launches()
        .into_iter()
        .map(|(operator, key, _)| (operator, key))
        .collect();

    (output, (launches, context.peak()))
}

/// The requests the deployment captures for, with the keyframe sets each may open with.
fn candidates(model: &dyn VideoModel) -> Vec<(VideoRequest, Vec<Vec<ConditionShape>>)> {
    model.candidates(&serving()).unwrap()
}

/// Runs one denoising step of `request` on the meta device and returns its recording.
///
/// The latents and the conditioning are meta tensors of the layout's shapes: the FP32 video
/// and audio latents, and the stack-typed conditioning rows.
fn step(model: &dyn VideoModel, sequence: SequenceSplit, request: &VideoRequest) -> Recording {
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
fn profile(root: &str, network: Stage, sequence: SequenceSplit) {
    let model = load(root, network, sequence).unwrap();
    let denoiser = model.denoiser().unwrap();
    let name = format!(
        "{} {} rank {} of {} ({:?})",
        Path::new(root).file_name().unwrap().to_string_lossy(),
        model.denoiser_stage(),
        sequence.split.rank,
        sequence.split.size,
        sequence.projection
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
    profile(BASE, Stage::Transformer, SequenceSplit::WHOLE);
    profile(BASE, Stage::TransformerRef, SequenceSplit::WHOLE);
}

#[test]
#[ignore = "requires /workspace/models/FastVideo-FastH3-8-Step-V2"]
fn the_eight_step_student_profiles_a_small_and_a_large_layout() {
    profile(FAST_8_STEP, Stage::Transformer, SequenceSplit::WHOLE);
}

#[test]
#[ignore = "requires /workspace/models/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"]
fn the_four_step_student_profiles_a_small_and_a_large_layout() {
    profile(FAST_4_STEP, Stage::Transformer, SequenceSplit::WHOLE);
}

/// A rank of a sequence split of four, under each projection: the dense network returns its
/// rows in one exchange, the sparse one interval by interval, and under the gather projection
/// each block's rows are gathered while the block before it computes.
#[test]
#[ignore = "requires /workspace/models/MiniMax-H3 and the four-step FastH3 student"]
fn a_rank_of_a_sequence_split_profiles_under_each_projection() {
    for projection in [Projection::Gather, Projection::Exchange] {
        let sequence = SequenceSplit {
            split: Split { size: 4, rank: 1 },
            projection,
        };

        profile(BASE, Stage::Transformer, sequence);
        profile(FAST_4_STEP, Stage::Transformer, sequence);
    }
}

/// The NVFP4 export's profile: the network's feed-forward projections are packed, so a step
/// encodes their activations and multiplies the packed weights (`nvfp4_quantize`,
/// `nvfp4_gemm`), block by block above the packed product's largest row class. The text
/// encoder, which the export's manifest does not calibrate, loads dense beside it.
#[test]
#[ignore = "requires /workspace/models/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree-NVFP4"]
fn the_nvfp4_export_profiles_a_small_and_a_large_layout() {
    profile(FAST_4_STEP_NVFP4, Stage::Transformer, SequenceSplit::WHOLE);

    let encoder = load(FAST_4_STEP_NVFP4, Stage::TextEncoder, SequenceSplit::WHOLE).unwrap();
    assert!(encoder.text_encoder().is_some());
}
