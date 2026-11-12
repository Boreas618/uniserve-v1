//! Profile runs of the VAE stages over the released checkpoint's headers.
//!
//! Each run loads a stage through a meta loader, which reads the shard headers and no tensor
//! payload, and computes one forward on meta tensors under a meta context: the forward checks
//! its shapes, states the kernel of every launch and computes nothing. A run asserts its output
//! shape and that every launch names an operator and a key, and prints what a deployment sizes
//! the stage by: the launches, the distinct kernel keys, which operators are generic tensor
//! primitives and which are fused operators of `nn`, and the allocator's peak.
//!
//! The runs need the checkpoint at [`CHECKPOINT`] and are ignored without it:
//!
//! ```text
//! cargo test -p minimax-h3 --test vae_profile -- --ignored --nocapture
//! ```

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]

mod support;

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use minimax_h3::audio_vae::{AudioDecoder, AudioEncoder, padded_samples};
use minimax_h3::video_vae::{self, VideoDecoder, VideoEncoder};
use minimax_h3::{AudioVaeConfig, VideoVaeConfig};
use nn::{Checkpoint, Clip, DType, Decoder, LatentShape, Layout, WeightLoader};
use processing::minimax_h3::{
    AUDIO_LATENTS_PER_SECOND, FRAMES, MAX_SECONDS, MIN_SECONDS, named_canvases,
};
use support::{Recording, meta, recorded};

/// The released checkpoint the runs read the headers of.
const CHECKPOINT: &str = "/workspace/models/MiniMax-H3";

/// The operators of the device backend's generic tensor primitives (`backend::primitives`);
/// every other operator a forward launches is a fused operator of `nn`.
const PRIMITIVES: [&str; 8] = [
    "cast",
    "copy_strided",
    "elementwise_binary",
    "elementwise_unary",
    "elementwise_where",
    "fill",
    "index",
    "reduce",
];

/// The stage `name` of the released checkpoint, opened by its headers.
fn stage(name: &str) -> Checkpoint {
    let directory = Path::new(CHECKPOINT).join(name);
    assert!(
        directory.is_dir(),
        "the profile runs read the headers of {CHECKPOINT}, which holds no {name} stage"
    );

    Checkpoint::open(&directory).unwrap()
}

/// The video VAE's stage and its config.
fn video_stage() -> (Checkpoint, VideoVaeConfig) {
    let checkpoint = stage("vae");
    let sidecar = checkpoint.sidecar("config.json").unwrap().unwrap();
    let config = VideoVaeConfig::parse(&sidecar).unwrap();

    (checkpoint, config)
}

/// The audio VAE's stage and its config.
fn audio_stage() -> (Checkpoint, AudioVaeConfig) {
    let checkpoint = stage("audio_vae");
    let sidecar = checkpoint.sidecar("config.json").unwrap().unwrap();
    let config = AudioVaeConfig::parse(&sidecar).unwrap();

    (checkpoint, config)
}

/// The smallest and the largest resolution the family names, `(width, height)` in pixels.
fn served_resolutions() -> [(u32, u32); 2] {
    let mut resolutions: Vec<(u32, u32)> = named_canvases()
        .unwrap()
        .iter()
        .map(|served| (served.canvas.width, served.canvas.height))
        .collect();
    resolutions.sort_by_key(|(width, height)| (width * height, *width));

    [resolutions[0], resolutions[resolutions.len() - 1]]
}

/// Checks that every launch of `recording` names an operator and a key, and prints the run's
/// launches, kernel keys, operators by kind and allocator peak.
fn report(run: &str, (launches, peak): &Recording) {
    assert!(!launches.is_empty(), "{run} launched nothing");
    for (index, (operator, key)) in launches.iter().enumerate() {
        assert!(
            !operator.is_empty() && !key.is_empty(),
            "{run}: launch {index} states operator {operator:?} and key {key:?}"
        );
    }

    // Each operator's launches and its distinct keys.
    let mut operators: BTreeMap<&str, (usize, BTreeSet<&str>)> = BTreeMap::new();
    for (operator, key) in launches {
        let entry = operators.entry(operator).or_default();
        entry.0 += 1;
        entry.1.insert(key);
    }
    let keys: usize = operators.values().map(|(_, keys)| keys.len()).sum();

    println!(
        "{run}: {} launches, {keys} kernel keys, allocator peak {peak} bytes ({:.1} MiB)",
        launches.len(),
        *peak as f64 / (1 << 20) as f64
    );
    for (kind, generic) in [("generic primitives", true), ("fused operators", false)] {
        let listed: Vec<String> = operators
            .iter()
            .filter(|(operator, _)| PRIMITIVES.contains(operator) == generic)
            .map(|(operator, (count, keys))| format!("{operator} {count}/{}", keys.len()))
            .collect();

        println!("  {kind} (launches/keys): {}", listed.join(", "));
    }
    for (operator, (_, keys)) in &operators {
        for key in keys {
            println!("    {operator}: {key}");
        }
    }
}

#[test]
#[ignore = "requires /workspace/models/MiniMax-H3"]
fn the_video_decoder_profiles_one_clip_at_a_small_and_a_large_resolution() {
    let (checkpoint, config) = video_stage();
    let loader = WeightLoader::meta(&checkpoint, None, None);
    let decoder = VideoDecoder::load(&config, loader.root()).unwrap();
    let clips = video_vae::temporal_clips(&config);

    for (width, height) in served_resolutions() {
        // One temporal unit: the clip's window of latent frames, decoded whole.
        let clip = Clip {
            frames: clips.window() * config.temporal_compression(),
            height,
            width,
        };
        let layout = decoder.layout_of(clip).unwrap();
        assert_eq!(layout.video.frames, 7);

        let rows = decoder.latent_rows(&layout).unwrap().unwrap().dims()[0];
        let timeline = 2 * clips.advance + clips.window();
        let voxels = (timeline * layout.video.height * layout.video.width) as usize;

        let trajectory = meta((24, voxels), DType::F32);
        let block = meta((rows, 24), DType::F32);
        let (gathered, recording) =
            recorded(|| decoder.gather_latents(&layout, &trajectory, clips.start(2), &block));
        gathered.unwrap();
        report(&format!("video gather {width}x{height}"), &recording);

        let raster = (3, clip.frames as usize, height as usize, width as usize);
        let video = meta(raster, decoder.element_type());
        let (decoded, recording) = recorded(|| decoder.decode(&layout, &block, Some(&video), None));
        decoded.unwrap();
        report(
            &format!("video decode {width}x{height}, {rows} latent rows"),
            &recording,
        );
    }
}

#[test]
#[ignore = "requires /workspace/models/MiniMax-H3"]
fn the_video_encoder_profiles_a_still_and_a_clip() {
    let (checkpoint, config) = video_stage();
    let loader = WeightLoader::meta(&checkpoint, None, None);
    let encoder = VideoEncoder::load(&config, loader.root()).unwrap();
    let [(small_width, small_height), (large_width, large_height)] = served_resolutions();

    // A keyframe at the smallest resolution, and a reference video of the fewest frames the
    // frame rule generates at the largest: one whole clip and the tail.
    for (frames, width, height) in [
        (1, small_width, small_height),
        (FRAMES.min_frames, large_width, large_height),
    ] {
        let pixels = meta(
            (3, frames as usize, height as usize, width as usize),
            DType::F32,
        );
        let (latent, recording) = recorded(|| encoder.encode(&pixels));
        let latent = latent.unwrap();

        assert_eq!(
            latent.dims(),
            [
                24,
                FRAMES.latent_frames(frames) as usize,
                height as usize / 16,
                width as usize / 16
            ]
        );
        assert_eq!(latent.dtype(), DType::F32);
        report(
            &format!("video encode {width}x{height}, {frames} frame(s)"),
            &recording,
        );
    }
}

#[test]
#[ignore = "requires /workspace/models/MiniMax-H3"]
fn the_audio_decoder_profiles_the_shortest_and_the_longest_track() {
    let (checkpoint, config) = audio_stage();
    let loader = WeightLoader::meta(&checkpoint, None, None);
    let decoder = AudioDecoder::load(&config, loader.root()).unwrap();
    assert_eq!(decoder.sampling_rate(), 32_000);

    for seconds in [MIN_SECONDS, MAX_SECONDS] {
        let frames = (seconds * f64::from(AUDIO_LATENTS_PER_SECOND)) as u32;
        let track = LatentShape {
            channels: 2 * config.latent_channels,
            frames,
            height: 1,
            width: 1,
        };
        let layout = Layout {
            video: LatentShape {
                channels: 0,
                frames: 0,
                height: 0,
                width: 0,
            },
            audio: Some(track),
            text_rows: 0,
            condition_rows: 0,
            condition_elements: 0,
        };

        // A latent frame is 800 samples at 32 kHz, in two channels.
        let (samples, channels) = decoder.pcm_window(&track).unwrap();
        assert_eq!((samples, channels), (u64::from(frames) * 800, 2));

        let latents = meta((track.channels as usize, frames as usize), DType::F32);
        let audio = meta((2, samples as usize), DType::F32);
        let (decoded, recording) =
            recorded(|| decoder.decode(&layout, &latents, None, Some(&audio)));
        decoded.unwrap();
        report(
            &format!("audio decode {seconds} s, {frames} latent frames"),
            &recording,
        );
    }
}

#[test]
#[ignore = "requires /workspace/models/MiniMax-H3"]
fn the_audio_encoder_profiles_a_reference_track() {
    let (checkpoint, config) = audio_stage();
    let loader = WeightLoader::meta(&checkpoint, None, None);
    let encoder = AudioEncoder::load(&config, loader.root()).unwrap();
    assert_eq!(encoder.hop(), 800);

    // A track one sample longer than four seconds is padded to the next latent frame.
    let length = padded_samples(4 * 32_000 + 1, encoder.hop()).unwrap();
    let frames = (length / encoder.hop()) as usize;
    assert_eq!(frames, 161);

    let samples = meta((2, 1, length as usize), DType::F32);
    let (rows, recording) = recorded(|| encoder.encode(&samples));
    let rows = rows.unwrap();

    assert_eq!(rows.dims(), [2 * frames, 32]);
    assert_eq!(rows.dtype(), DType::F32);
    report(
        &format!("audio encode {length} samples, {frames} latent frames"),
        &recording,
    );
}
