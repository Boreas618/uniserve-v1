//! The denoiser on CPU tensors over a tiny checkpoint of stated values: one step's result does
//! not depend on the layout its request is laid out in.
//!
//! A layout decides where a request's rows sit (the dense layout leads with the generated rows,
//! the tiled layout holds whole tiles) and how many padding rows surround them; the velocity of
//! a request is a function of its document alone. So the same request, stepped
//!
//! - in dense layouts of two text capacities, and
//! - in the dense layout and in the tiled layout of a sparse attention that keeps every tile
//!   and whose compression gate is negligible,
//!
//! advances its latents to the same values. That exercises, through the public capability, the
//! packing's gathers, the modulation rows, the rotary coordinates, the key bounds of the dense
//! attention and the tile tables of the sparse one.
//!
//! ## Tolerance
//!
//! The two runs of a pair evaluate the same FP32 formulas with their reductions in another
//! order: the attention sums over the same keys at other row positions, and the sparse
//! attention reduces tile by tile. A reduction of `K` terms differs by at most `K · 2⁻²⁴`
//! relative between two orders, and a step chains `L` such stages, so the bound is
//! `L · K · 2⁻²⁴` of a value's magnitude: here `K = 384` (the widest tiled layout's rows) and
//! `L = 2 · 8 + 6` (eight reductions a block, the refiner's and the heads'), about `5.0e-4`.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
use super::support;

use std::path::Path;

use models::minimax_h3::config::{DenoiserConfig, ScheduleConfig, TransformerConfig, VsaConfig};
use models::minimax_h3::denoiser::Denoiser;
use models::minimax_h3::{VideoTask, packing};
use nn::family::SequenceSplit;
use nn::{
    Checkpoint, ConditionRole, ConditionShape, Conditioning, CpuWeights, DType, Denoiser as _,
    Device, Tensor, VideoRequest, WeightLoader,
};
use support::{assert_close, f32_shard, sequence, values};

/// Stages times the longest reduction times the FP32 unit roundoff (module documentation).
const TOLERANCE: f32 = 22.0 * 384.0 / 16_777_216.0;

/// The magnitude of the compression gate's weights in the tiny checkpoint.
const GATE_SCALE: f32 = 1e-9;

/// A network small enough to step on the CPU: two blocks of two heads, twelve of a head's
/// sixteen channels rotated.
fn transformer() -> TransformerConfig {
    TransformerConfig {
        hidden_size: 32,
        num_attention_heads: 2,
        head_dim: 16,
        num_hidden_layers: 2,
        num_refiner_layers: 1,
        intermediate_size: 48,
        video_channels: 4,
        audio_channels: 8,
        text_dim: 24,
        frequency_dim: 16,
        time_hidden_dim: 32,
        time_dim: 16,
        rope_frequency_dim: 2,
        rope_theta: 10_000.0,
        norm_eps: 1e-5,
        qk_norm_eps: 1e-5,
        output_heads: 1,
    }
}

/// The network's configuration with a four-rung ladder, attending densely or through `vsa`.
fn config(vsa: Option<VsaConfig>) -> DenoiserConfig {
    DenoiserConfig::new(
        transformer(),
        ScheduleConfig::ladder(&[999, 749, 500, 250], 1000.0, 12.0, 3.0).unwrap(),
        vsa,
        vec![
            VideoTask::TextToVideoAudio,
            VideoTask::FirstLastToVideoAudio,
        ],
    )
    .unwrap()
}

/// Writes the tiny network's checkpoint into `directory`: every tensor the released checkpoint
/// names, with values that differ from tensor to tensor, and a negligible compression gate.
fn write_checkpoint(directory: &Path) {
    let config = transformer();
    let (hidden, inner) = (
        config.hidden_size as usize,
        (config.num_attention_heads * config.head_dim) as usize,
    );
    let ffn = config.intermediate_size as usize;
    let time = config.time_dim as usize;

    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    let mut seed = 0;
    let mut tensor = |name: String, shape: &[usize], scale: f32, offset: f32| {
        seed += 1;
        let values = sequence(shape.iter().product(), seed)
            .into_iter()
            .map(|value| value * scale + offset)
            .collect();
        tensors.push((name, shape.to_vec(), values));
    };

    // A projection's weight is scaled by its fan-in so activations stay near one, and a norm's
    // gain sits near one.
    let mut linear = |name: &str, outputs: usize, inputs: usize, bias: bool| {
        let scale = 1.0 / (inputs as f32).sqrt();
        tensor(format!("{name}.weight"), &[outputs, inputs], scale, 0.0);
        if bias {
            tensor(format!("{name}.bias"), &[outputs], 0.1, 0.0);
        }
    };
    linear("proj_in", hidden, config.patch_width() as usize, true);
    linear(
        "audio_proj_in",
        hidden,
        config.audio_channels as usize,
        true,
    );
    linear("proj_out", config.patch_width() as usize, hidden, true);
    linear(
        "audio_proj_out",
        config.audio_channels as usize,
        hidden,
        true,
    );
    linear("context_embedder", hidden, config.text_dim as usize, true);
    linear(
        "time_embedder.linear_1",
        config.time_hidden_dim as usize,
        config.frequency_dim as usize,
        true,
    );
    linear(
        "time_embedder.linear_2",
        time,
        config.time_hidden_dim as usize,
        true,
    );
    linear("norm_out.linear", 2 * hidden, time, true);

    let blocks = (0..config.num_refiner_layers)
        .map(|index| format!("token_refiner.refiner_blocks.{index}"))
        .chain((0..config.num_hidden_layers).map(|index| format!("transformer_blocks.{index}")));
    for block in blocks {
        for projection in ["to_q", "to_k", "to_v"] {
            linear(&format!("{block}.attn.{projection}"), inner, hidden, false);
        }
        linear(&format!("{block}.attn.to_out.0"), hidden, inner, false);
        linear(&format!("{block}.ff.net.0.proj"), 2 * ffn, hidden, false);
        linear(&format!("{block}.ff.net.2"), hidden, ffn, false);

        if block.starts_with("transformer_blocks") {
            linear(
                &format!("{block}.adaln_proj.linear"),
                18 * hidden,
                time,
                true,
            );
        }
    }

    let mut gain = |name: String, width: usize| tensor(name, &[width], 0.1, 1.0);
    gain("norm_out.norm.weight".to_owned(), hidden);
    gain("token_refiner.final_norm.weight".to_owned(), hidden);
    let blocks = (0..config.num_refiner_layers)
        .map(|index| format!("token_refiner.refiner_blocks.{index}"))
        .chain((0..config.num_hidden_layers).map(|index| format!("transformer_blocks.{index}")));
    for block in blocks {
        gain(format!("{block}.norm1.weight"), hidden);
        gain(format!("{block}.norm2.weight"), hidden);
        gain(
            format!("{block}.attn.norm_q.weight"),
            config.head_dim as usize,
        );
        gain(
            format!("{block}.attn.norm_k.weight"),
            config.head_dim as usize,
        );
    }

    // The loader refuses an all-zero projection, so the compression gate is `GATE_SCALE`
    // small instead: what it folds of the compressed branch into a row is that fraction of the
    // row, far below the tolerance, and a sparse attention that keeps every tile is then the
    // dense attention.
    for index in 0..config.num_hidden_layers {
        tensor(
            format!("transformer_blocks.{index}.attn.to_gate_compress.weight"),
            &[inner, hidden],
            GATE_SCALE,
            0.0,
        );
    }

    f32_shard(directory, "model.safetensors", &tensors);
}

/// A request of 22 frames at 64x64 pixels with a prompt of five rows, and `conditions`.
fn request(conditions: Vec<ConditionShape>, condition_rows: Option<u32>) -> VideoRequest {
    VideoRequest {
        width: 64,
        height: 64,
        frames: 22,
        fps: 24,
        first_frames: conditions.len() as u32,
        last_frames: 0,
        references: 0,
        text_tokens: 5,
        conditions,
        vision_spans: vec![(1, 2)],
        condition_rows,
    }
}

/// The request's latents after its conditioning is prepared and `steps` denoising steps, in a
/// layout of `capacity`'s text rows and condition capacity, attending densely or through
/// `vsa`: `(video, audio)`.
fn stepped(
    vsa: Option<VsaConfig>,
    capacity: (u32, u32),
    request: &VideoRequest,
    steps: usize,
) -> (Vec<f32>, Vec<f32>) {
    let directory = tempfile::tempdir().unwrap();
    write_checkpoint(directory.path());
    let checkpoint = Checkpoint::open(directory.path()).unwrap();
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None, None);

    let config = config(vsa);
    let denoiser =
        Denoiser::load(&config, capacity, SequenceSplit::WHOLE, None, loader.root()).unwrap();
    denoiser.initialize().unwrap();

    let layout = denoiser.layout(request).unwrap();
    packing::check(request, &layout).unwrap();
    let transformer = transformer();

    // The prompt's features, padded with zeros to the text capacity, and the conditioning the
    // denoiser reads: the refined text rows, the condition rows, the row of zeros.
    let prompt = request.text_tokens as usize;
    let width = transformer.text_dim as usize;
    let mut features = sequence(prompt * width, 1000);
    features.resize(layout.text_rows as usize * width, 0.0);
    let source =
        Tensor::from_vec(features, (layout.text_rows as usize, width), &Device::Cpu).unwrap();

    let rows = nn::capability::prefix_rows(&layout) as usize;
    let hidden = transformer.hidden_size as usize;
    let text = Tensor::zeros((rows, hidden), DType::F32, &Device::Cpu).unwrap();

    // A condition's rows are written after the text capacity, as the encoders would write
    // them: stated values stand for an encoded keyframe.
    let taken: usize = request
        .conditions
        .iter()
        .map(|condition| packing::condition_rows(condition).1)
        .sum();
    if taken > 0 {
        let encoded = Tensor::from_vec(
            sequence(taken * hidden, 2000),
            (taken, hidden),
            &Device::Cpu,
        )
        .unwrap();
        text.slice_set(&encoded, 0, layout.text_rows as usize)
            .unwrap();
    }

    let cond = Conditioning {
        text,
        source: Some(source),
        media: Vec::new(),
        images: None,
        audio: None,
    };
    denoiser.prepare_conditioning(request, &cond).unwrap();

    let shape = layout.video;
    let video = Tensor::zeros(
        (
            shape.channels as usize,
            shape.frames as usize,
            shape.height as usize,
            shape.width as usize,
        ),
        DType::F32,
        &Device::Cpu,
    )
    .unwrap();
    let track = layout.audio.unwrap();
    let audio = Tensor::zeros(
        (track.channels as usize, track.frames as usize),
        DType::F32,
        &Device::Cpu,
    )
    .unwrap();
    denoiser
        .draw_noise(7, &layout, &video, Some(&audio))
        .unwrap();
    denoiser
        .prepare_latents(&cond, &video, Some(&audio))
        .unwrap();

    let schedule = denoiser.schedule(request).unwrap();
    assert_eq!(schedule.len(), 4);
    for step in &schedule[..steps] {
        denoiser
            .forward(*step, request, &video, Some(&audio), &cond)
            .unwrap();
    }

    (values(&video), values(&audio))
}

#[test]
fn a_step_moves_the_latents_and_keeps_them_finite() {
    let request = request(Vec::new(), None);
    let (noise, _) = stepped(None, (64, 64), &request, 0);
    let (video, audio) = stepped(None, (64, 64), &request, 2);

    // 4 channels of 7 latent frames of 4 x 4 cells, and 37 latent steps of two channels of 8.
    assert_eq!(video.len(), 4 * 7 * 4 * 4);
    assert_eq!(audio.len(), 16 * 37);
    assert!(video.iter().chain(&audio).all(|value| value.is_finite()));
    assert!(
        video
            .iter()
            .zip(&noise)
            .any(|(stepped, drawn)| (stepped - drawn).abs() > 1e-3),
        "two steps advance the sample"
    );
}

#[test]
fn a_step_does_not_depend_on_the_text_capacity() {
    let request = request(Vec::new(), None);
    let narrow = stepped(None, (64, 64), &request, 2);
    let wide = stepped(None, (128, 64), &request, 2);

    assert_close(&wide.0, &narrow.0, TOLERANCE);
    assert_close(&wide.1, &narrow.1, TOLERANCE);
}

#[test]
fn a_sparse_attention_that_keeps_every_tile_is_the_dense_attention() {
    let vsa = VsaConfig::new(64, 0.0, None).unwrap();

    // Without conditions, and with a first-frame keyframe on the generated video's grid.
    let keyframe = ConditionShape {
        role: ConditionRole::FirstFrame,
        frames: 1,
        height: 64,
        width: 64,
        audio_samples: 0,
    };
    for request in [request(Vec::new(), None), request(vec![keyframe], Some(64))] {
        let dense = stepped(None, (64, 64), &request, 2);
        let tiled = stepped(Some(vsa), (64, 64), &request, 2);

        assert_close(&tiled.0, &dense.0, TOLERANCE);
        assert_close(&tiled.1, &dense.1, TOLERANCE);
    }
}

#[test]
fn a_request_of_a_task_the_network_does_not_serve_is_refused_by_name() {
    let directory = tempfile::tempdir().unwrap();
    write_checkpoint(directory.path());
    let checkpoint = Checkpoint::open(directory.path()).unwrap();
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None, None);
    let denoiser = Denoiser::load(
        &config(None),
        (64, 64),
        SequenceSplit::WHOLE,
        None,
        loader.root(),
    )
    .unwrap();

    let mut reference = request(Vec::new(), None);
    reference.references = 1;
    let error = denoiser.schedule(&reference).unwrap_err().to_string();
    assert!(error.contains("ReferenceToVideoAudio"), "{error}");

    // A request without conditions reserves no condition row, whatever the capacity.
    let layout = denoiser.layout(&request(Vec::new(), None)).unwrap();
    assert_eq!((layout.text_rows, layout.condition_rows), (64, 0));
    assert_eq!(denoiser.condition_ladder(), [64]);
}
