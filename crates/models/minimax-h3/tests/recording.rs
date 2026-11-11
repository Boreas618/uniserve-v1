//! A recorded walk of a deployment over a real checkpoint: the pipeline is loaded from the
//! checkpoint's headers alone (`nn::WeightLoader::meta`) and its forwards run on a recording
//! forward, which checks every launch's views and uploads nothing to a device. It exercises the
//! loading boundary, every shape of a denoising step and of a clip decode, and the host values a
//! step uploads, on the extents a served request has.

use std::collections::BTreeSet;
use std::path::PathBuf;

use nn::capability::{Clip, Conditioning, DecodedMedia, Latents, VideoRequest};
use nn::family::{SequenceSplit, Split, Stage, VideoPlacement, VideoServing};
use nn::{Checkpoint, DType, Shape, WeightLoader};

fn root() -> PathBuf {
    std::env::var_os("UNISERVE_H3_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("UNISERVE_H3_MODEL names a MiniMax-H3 checkpoint"))
}

#[test]
#[ignore = "requires UNISERVE_H3_MODEL naming a MiniMax-H3 checkpoint"]
fn a_served_request_steps_and_decodes_on_a_recording_forward() {
    walk(SequenceSplit::WHOLE, true);
}

/// Rank 0 of two under the gather projection: each layer reads the whole sequence's normalized
/// rows, gathered interval by interval while the layer before it computes, and returns its
/// attended rows interval by interval.
#[test]
#[ignore = "requires UNISERVE_H3_MODEL naming a MiniMax-H3 checkpoint"]
fn a_sequence_split_step_gathers_its_rows_on_a_recording_forward() {
    walk(
        SequenceSplit {
            split: Split { size: 2, rank: 0 },
            projection: nn::family::Projection::Gather,
        },
        false,
    );
}

/// Load the checkpoint's pipeline at `sequence` from its headers and record one denoising step
/// of a served request, and, when `decode`, one clip decode of the same canvas.
fn walk(sequence: SequenceSplit, decode: bool) {
    let checkpoint =
        Checkpoint::open(&root()).unwrap_or_else(|error| panic!("the checkpoint: {error}"));
    let nn::family::Model::Video(family) = minimax_h3::family::FAMILY.model else {
        panic!("MiniMax-H3 is a video family");
    };
    let opened = family
        .open(&checkpoint)
        .unwrap_or_else(|error| panic!("the configuration: {error}"));
    let stages: BTreeSet<String> = [
        Stage::TextEncoder,
        Stage::Transformer,
        Stage::VideoDecoder,
        Stage::AudioDecoder,
        Stage::VideoPostprocessor,
    ]
    .iter()
    .map(|stage| stage.name().to_owned())
    .collect();
    let placement = VideoPlacement {
        stages: Some(stages),
        sequence,
        tensor: Split::WHOLE,
        dtype: DType::BF16,
        serving: VideoServing {
            sizes: vec![(1344, 768)],
            max_seconds: 4.0,
            text_capacities: vec![512],
            condition_rows: None,
        },
    };
    let mut loader = WeightLoader::meta(&checkpoint, None, None);
    let model = opened
        .load(&checkpoint, &mut loader, &placement)
        .unwrap_or_else(|error| panic!("the meta load: {error}"));

    let arena = nn::Arena::counting();
    let staging = nn::Staging::counting();
    let f = nn::Forward::recording(&arena).with_staging(&staging);
    let f = if sequence.split.size > 1 {
        f.with_split(
            u64::from(sequence.split.rank),
            u64::from(sequence.split.size),
        )
        .unwrap_or_else(|error| panic!("the split: {error}"))
    } else {
        f
    };
    let tensor = |dtype: DType, dims: &[u64]| {
        arena
            .tensor(dtype, Shape::new(dims))
            .unwrap_or_else(|error| panic!("a counting arena holds any reservation: {error}"))
    };

    // One text-to-video step of a 4-second request at 1344x768.
    let denoiser = model.denoiser().unwrap_or_else(|| panic!("a denoiser"));
    let request = VideoRequest {
        width: 1344,
        height: 768,
        frames: model
            .timeline(4.0)
            .unwrap_or_else(|error| panic!("{error}")),
        fps: model.frame_rate(),
        first_frames: 0,
        last_frames: 0,
        references: 0,
        text_tokens: 100,
        conditions: &[],
        vision_spans: &[],
        condition_rows: None,
    };
    let layout = denoiser
        .layout(&request)
        .unwrap_or_else(|error| panic!("the layout: {error}"));
    let schedule = denoiser
        .schedule(&request)
        .unwrap_or_else(|error| panic!("the schedule: {error}"));
    let video = layout.video;
    let audio = layout.audio.unwrap_or_else(|| panic!("an audio stream"));
    let sample = model.sample_type();
    let latents = Latents {
        video: tensor(
            sample,
            &[
                u64::from(video.channels),
                u64::from(video.frames),
                u64::from(video.height),
                u64::from(video.width),
            ],
        ),
        audio: Some(tensor(
            sample,
            &[u64::from(audio.channels), u64::from(audio.frames)],
        )),
    };
    let width = u64::from(denoiser.text_width().unwrap_or(model.text_width()));
    let prefix = u64::from(layout.text_rows + layout.condition_rows + 1);
    let cond = Conditioning {
        text: tensor(DType::BF16, &[prefix, width]),
        source: Some(tensor(
            DType::BF16,
            &[u64::from(layout.text_rows), u64::from(model.text_width())],
        )),
        media: &[],
        images: None,
        audio: None,
    };
    denoiser
        .initialize(&f)
        .unwrap_or_else(|error| panic!("the AdaLN cache: {error}"));
    arena.reset();
    staging.reset();
    denoiser
        .forward(&f, schedule[0], &request, latents, cond)
        .unwrap_or_else(|error| panic!("the recorded step: {error}"));
    assert!(
        staging.peak() > 0,
        "a step uploads its layout's host values"
    );

    if !decode {
        return;
    }

    // One decode window of the same canvas.
    let decoder = model
        .video_decoder()
        .unwrap_or_else(|| panic!("a video decoder"));
    let clip = Clip {
        frames: model.decode_window(),
        height: 768,
        width: 1344,
    };
    let clip_layout = decoder
        .layout_of(clip)
        .unwrap_or_else(|error| panic!("the clip's layout: {error}"));
    let rows = decoder
        .latent_rows(&clip_layout)
        .unwrap_or_else(|error| panic!("the clip's rows: {error}"))
        .map_or(0, |rows| rows.len() as u64);
    arena.reset();
    staging.reset();
    decoder
        .decode(
            &f,
            &clip_layout,
            tensor(
                decoder.latent_type(),
                &[rows, u64::from(clip_layout.video.channels)],
            ),
            DecodedMedia {
                video: Some(tensor(
                    decoder.element_type(),
                    &[3, u64::from(clip.frames), 768, 1344],
                )),
                audio: None,
            },
        )
        .unwrap_or_else(|error| panic!("the recorded decode: {error}"));
}
