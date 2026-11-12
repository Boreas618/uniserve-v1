//! The VAE modules' arithmetic on CPU tensors against the reference implementation, and the
//! video decoder's tiles against its own single-tile decode.
//!
//! The fixtures of `tests/fixtures/vae/` are written by `tests/fixtures/generate_vae.py`, which
//! runs the reference's classes on the CPU in FP32: each holds a tiny checkpoint under the
//! released checkpoint's names, the inputs and the reference's output.
//!
//! ## Tolerances
//!
//! Every comparison with the reference is of two FP32 evaluations of one formula in different
//! orders. An inner product of `K` terms is within `K · u` of exact relative to the sum of its
//! terms' magnitudes, `u = 2⁻²⁴` (Higham, *Accuracy and Stability of Numerical Algorithms*,
//! §3.1), and the fixtures' activations and fan-in-scaled weights keep that sum of order one.
//! The errors of the `L` rounded stages along a path add to first order, so a case's bound is
//! `L · K · u` of each element's magnitude, or of one below it ([`bound`]). Each test states its
//! `L` and `K`.
//!
//! Nothing here touches a GPU.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]

mod support;

use std::path::Path;

use minimax_h3::audio_vae::{AudioDecoder, AudioEncoder};
use minimax_h3::video_vae::{self, TransformerBlock, VideoDecoder, VideoEncoder};
use minimax_h3::{AudioVaeConfig, VideoVaeConfig};
use nn::functional::{self, Padding};
use nn::modules::{ResnetBlock3d, ResnetBlock3dConfig, Rotary};
use nn::{
    Checkpoint, Clip, CpuWeights, DType, Decoder, Device, LatentShape, Layout, Tensor, WeightLoader,
};
use support::{assert_close, f32_shard, f32_values, sequence, tensor, values};

/// The unit roundoff of FP32.
const UNIT_ROUNDOFF: f32 = f32::EPSILON / 2.0;

/// The spacing of FP16 values relative to their magnitude: one part in 2¹⁰.
const F16_SPACING: f32 = 1.0 / 1024.0;

/// The bound of a path of `stages` rounded stages whose longest reduction has `terms` terms
/// (module documentation).
fn bound(stages: usize, terms: usize) -> f32 {
    (stages * terms) as f32 * UNIT_ROUNDOFF
}

/// The fixture `case`: a tiny checkpoint with its inputs and the reference's output.
fn fixture(case: &str) -> Checkpoint {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/vae")
        .join(case);

    Checkpoint::open(&directory).unwrap()
}

/// The FP32 CPU tensor the fixture stores as `fixture.<name>`.
fn stored(checkpoint: &Checkpoint, name: &str) -> Tensor {
    let name = format!("fixture.{name}");
    let shape = checkpoint.locate(&name).unwrap().shape.clone();

    tensor(
        &f32_values(&checkpoint.read(&name).unwrap()),
        shape,
        DType::F32,
    )
}

/// A tiny video VAE: an encoder of four halving stages of one block each, widths of four and
/// eight channels in two groups, and a decoder of two heads of eight channels.
fn video_config() -> VideoVaeConfig {
    VideoVaeConfig::parse(&serde_json::json!({
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": 2,
        "block_out_channels": [4, 4, 8, 8],
        "layers_per_block": 1,
        "spatial_downsample_factors": [2, 2, 2, 2],
        "temporal_downsample_factors": [1, 2, 2, 1],
        "norm_num_groups": 2,
        "norm_eps": 1e-6,
        "spatial_padding_mode": "reflect",
        "decoder_num_layers": 2,
        "decoder_num_attention_heads": 2,
        "decoder_attention_head_dim": 8,
        "decoder_num_register_tokens": 4,
        "decoder_ffn_mult": 4,
        "decoder_rope_theta": 100.0,
        "decoder_rope_dim_ratio": 0.75,
        "decoder_norm_eps": 1e-5,
        "clip_length": 17,
        "token_drop": 3,
        "latents_mean": [0.25, -0.5],
        "latents_std": [1.5, 2.0],
    }))
    .unwrap()
}

/// A tiny audio VAE: two stages of stride two each way, sixteen features, four latent
/// channels and two attention heads, with the statistics the generator states.
fn audio_config() -> AudioVaeConfig {
    AudioVaeConfig::parse(&serde_json::json!({
        "encoder_dim": 4,
        "encoder_rates": [2, 2],
        "latent_dim": 16,
        "latent_channels": 4,
        "decoder_dim": 8,
        "decoder_rates": [2, 2],
        "decoder_kernel_sizes": [4, 4],
        "num_attention_heads": 2,
        "resblock_kernel_sizes": [3, 5],
        "resblock_dilation_sizes": [[1, 3], [1, 3]],
        "sampling_rate": 32000,
        "latents_mean": [0.25, -0.5, 0.125, 0.75],
        "latents_std": [1.5, 0.75, 2.0, 1.25],
    }))
    .unwrap()
}

// ---- Video encoder ----------------------------------------------------------------------------

#[test]
fn the_encoders_residual_block_computes_the_references() {
    let checkpoint = fixture("resnet_block");
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None, None);

    let block = ResnetBlock3d::load(
        &ResnetBlock3dConfig {
            in_channels: 4,
            out_channels: 8,
            groups: 2,
            eps: 1e-6,
            spatial: [1; 4],
            front: 2,
            padding: Padding::Reflect,
            kernel: 3,
            tf32: true,
        },
        loader.root(),
    )
    .unwrap();

    let expected = stored(&checkpoint, "expected");
    let actual = block.forward(&stored(&checkpoint, "input")).unwrap();
    assert_eq!(actual.dims(), expected.dims());

    // Two normalizations, two convolutions, the shortcut and the sum; a convolution of eight
    // channels sums 8 · 27 products.
    assert_close(&values(&actual), &values(&expected), bound(6, 8 * 27));
}

#[test]
fn the_video_encoder_computes_the_references_moments() {
    let checkpoint = fixture("video_encoder");
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None, None);
    let encoder = VideoEncoder::load(&video_config(), loader.root()).unwrap();

    // Five frames of 32 by 32 pixels: two latent frames of two by two cells.
    let pixels = stored(&checkpoint, "input");
    let expected = stored(&checkpoint, "expected");
    let moments = encoder.forward(&pixels).unwrap();
    assert_eq!(moments.dims(), [1, 4, 2, 2, 2]);

    // The input convolution, four stages of a residual block (six stages each) and a
    // downsampler, the final normalization and the output convolution; the widest
    // convolution sums 8 · 27 products.
    assert_close(
        &values(&moments),
        &values(&expected),
        bound(1 + 4 * 7 + 2, 8 * 27),
    );
}

#[test]
fn a_condition_is_the_posteriors_seeded_sample_normalized() {
    let checkpoint = fixture("video_encoder");
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None, None);
    let config = video_config();
    let encoder = VideoEncoder::load(&config, loader.root()).unwrap();

    // One clip of five frames inside one tile: `[3, 5, 32, 32]` to `[2, 2, 2, 2]`.
    let pixels = stored(&checkpoint, "input").squeeze(0).unwrap();
    let latent = encoder.encode(&pixels).unwrap();
    assert_eq!(latent.dims(), [2, 2, 2, 2]);
    assert_eq!(latent.dtype(), DType::F32);

    // The reference's projected moments, `quant_conv(encoder(x))`: the mean's two channels,
    // then the log-variance's. The sample is `mean + exp(0.5 · clamp(logvar)) · noise` under
    // the posterior's seed, rounded to FP16.
    let moments = values(&stored(&checkpoint, "moments"));
    let (mean, logvar) = moments.split_at(16);
    let like = Tensor::zeros((2, 2, 2, 2), DType::F32, &Device::Cpu).unwrap();
    let noise =
        values(&functional::philox_normal_like(&like, video_vae::POSTERIOR_SEED, 0, 0, 0).unwrap());
    assert!(noise.iter().any(|sample| *sample != 0.0));

    let sample: Vec<f32> = (0..16)
        .map(|index| mean[index] + (0.5 * logvar[index].clamp(-30.0, 20.0)).exp() * noise[index])
        .collect();
    let expected = values(&tensor(&sample, 16, DType::F16));

    // The encoder's latent restored from its normalization is that sample. An FP32-level
    // difference in the moments moves the FP16-rounded sample by at most one FP16 spacing.
    let restored: Vec<f32> = values(&latent)
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let channel = index / 8;
            value * config.latents_std[channel] + config.latents_mean[channel]
        })
        .collect();
    assert_close(&restored, &expected, F16_SPACING);

    // A still is one frame and one latent frame; a frame count whose last clip the frame rule
    // does not encode whole, and a raster that is not whole latent cells, are refused.
    let still = encoder.encode(&pixels.narrow(1, 0, 1).unwrap()).unwrap();
    assert_eq!(still.dims(), [2, 1, 2, 2]);

    let frames = |count: usize| Tensor::zeros((3, count, 32, 32), DType::F32, &Device::Cpu);
    assert!(encoder.encode(&frames(30).unwrap()).is_err());
    assert!(
        encoder
            .encode(&pixels.narrow(2, 0, 24).unwrap().contiguous().unwrap())
            .is_err()
    );
    assert!(
        encoder
            .encode(&pixels.to_dtype(DType::F16).unwrap())
            .is_err()
    );
}

// ---- Video decoder ----------------------------------------------------------------------------

#[test]
fn a_decoder_block_computes_the_references() {
    let checkpoint = fixture("transformer_block");
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None, None);
    let block = TransformerBlock::load(&video_config(), loader.root().pp("block")).unwrap();

    // Two sequences of seventeen tokens, each reading the same positions.
    let (sequences, rows) = (2, 17);
    let over_sequences = |name: &str| {
        let table = stored(&checkpoint, name);
        Tensor::cat(&vec![table; sequences], 0).unwrap()
    };
    let (cos, sin) = (over_sequences("cos"), over_sequences("sin"));
    let cu_seqlens = Tensor::new(&[0_u32, 17, 34], &Device::Cpu).unwrap();

    let (hidden, update) = block
        .forward(
            stored(&checkpoint, "input"),
            None,
            (&cos, &sin),
            &cu_seqlens,
            rows,
        )
        .unwrap();

    // The block's output is the stream with the feed-forward's update joined at `scale2`,
    // which the next normalization folds in.
    let scale2 = tensor(
        &f32_values(&checkpoint.read("block.scale2").unwrap()),
        16,
        DType::F32,
    );
    let actual = (hidden + update.broadcast_mul(&scale2).unwrap()).unwrap();
    let expected = stored(&checkpoint, "expected");
    assert_eq!(actual.dims(), expected.dims());

    // Two normalizations, three projections with the per-head normalization and rotation,
    // the attention's scores, softmax and sum, the output projection, the feed-forward's two
    // projections and its gate, and two scaled sums; the feed-forward's output projection
    // sums 64 products.
    assert_close(&values(&actual), &values(&expected), bound(14, 64));
}

#[test]
fn the_decoders_rotary_tables_are_the_references() {
    let checkpoint = fixture("transformer_block");
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None, None);

    let pos_embed = Rotary::load(
        &video_vae::rotary_config(&video_config()),
        loader.root().pp("decoder").pp("pos_embed"),
    )
    .unwrap();
    let positions = video_vae::token_positions(2, 2, 3, 5).unwrap();
    let (cos, sin) = pos_embed.forward(&positions).unwrap();
    assert_eq!(cos.dims(), [17, 3]);

    // An angle is `2π · centre · frequency`: the centre's three operations, the turn, the
    // frequency's power and reciprocal and the product, seven roundings of an angle of at
    // most one turn, and the factor's own.
    let tolerance = std::f32::consts::TAU * 7.0 * UNIT_ROUNDOFF + UNIT_ROUNDOFF;
    assert_close(
        &values(&cos),
        &values(&stored(&checkpoint, "cos")),
        tolerance,
    );
    assert_close(
        &values(&sin),
        &values(&stored(&checkpoint, "sin")),
        tolerance,
    );
}

/// The tensors of a decoder of `config`, as its checkpoint names them, with fixed values: each
/// projection scaled over its fan-in, each gain near one, and the rest small.
fn decoder_tensors(config: &VideoVaeConfig) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let channels = config.latent_channels as usize;
    let width = (config.decoder_num_attention_heads * config.decoder_attention_head_dim) as usize;
    let inner = width * config.decoder_ffn_mult as usize;
    let patch = config.out_channels as usize * 4 * 16 * 16;

    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    let mut add = |name: String, shape: Vec<usize>, offset: f32, scale: f32| {
        let seed = tensors.len() as u32;
        let values = sequence(shape.iter().product(), seed)
            .into_iter()
            .map(|value| offset + scale * value)
            .collect();
        tensors.push((name, shape, values));
    };
    let mut linear = |name: &str, outputs: usize, inputs: usize, filter: &[usize]| {
        let mut shape = vec![outputs, inputs];
        shape.extend_from_slice(filter);

        add(
            format!("{name}.weight"),
            shape,
            0.0,
            (inputs as f32).sqrt().recip(),
        );
        add(format!("{name}.bias"), vec![outputs], 0.0, 0.1);
    };

    linear("post_quant_conv", channels, channels, &[1, 1, 1]);
    linear("decoder.proj_in", width, channels, &[]);
    linear("decoder.proj_out", patch, width, &[]);
    for index in 0..config.decoder_num_layers {
        let block = format!("decoder.transformer_blocks.{index}");

        for projection in ["to_q", "to_k", "to_v", "to_out.0"] {
            linear(&format!("{block}.attn.{projection}"), width, width, &[]);
        }
        linear(&format!("{block}.ff.net.0.proj"), 2 * inner, width, &[]);
        linear(&format!("{block}.ff.net.2"), width, inner, &[]);
    }

    for index in 0..config.decoder_num_layers {
        let block = format!("decoder.transformer_blocks.{index}");

        add(format!("{block}.norm1.weight"), vec![width], 1.0, 0.1);
        add(format!("{block}.norm2.weight"), vec![width], 1.0, 0.1);
        add(format!("{block}.scale1"), vec![width], 0.0, 0.5);
        add(format!("{block}.scale2"), vec![width], 0.0, 0.5);
    }
    add("decoder.norm_out.weight".to_owned(), vec![width], 1.0, 0.1);
    add("decoder.norm_out.bias".to_owned(), vec![width], 0.0, 0.1);

    let registers = config.decoder_num_register_tokens as usize;
    add(
        "decoder.register_tokens".to_owned(),
        vec![1, registers, width],
        0.0,
        0.5,
    );
    tensors
}

/// One clip of `trajectory`, `[channels, voxels]` of a `[frames, height, width]` latent
/// timeline, decoded as execution decodes it: the decoder's layout of the raster, its rows
/// gathered from the trajectory, and its raster.
fn decoded(decoder: &VideoDecoder, trajectory: &Tensor, clip: Clip) -> Tensor {
    let layout = decoder.layout_of(clip).unwrap();
    let rows = decoder.latent_rows(&layout).unwrap().unwrap();
    let channels = trajectory.dims()[0];

    let block = Tensor::zeros((rows.dims()[0], channels), DType::F32, &Device::Cpu).unwrap();
    decoder
        .gather_latents(&layout, trajectory, 0, &block)
        .unwrap();

    let raster = (
        3,
        clip.frames as usize,
        clip.height as usize,
        clip.width as usize,
    );
    let video = Tensor::zeros(raster, decoder.element_type(), &Device::Cpu).unwrap();
    decoder.decode(&layout, &block, Some(&video), None).unwrap();
    video
}

#[test]
fn a_tiled_decode_is_each_tiles_own_decode_outside_the_seam() {
    let directory = tempfile::tempdir().unwrap();
    let config = video_config();
    f32_shard(
        directory.path(),
        "model.safetensors",
        &decoder_tensors(&config),
    );
    let checkpoint = Checkpoint::open(directory.path()).unwrap();
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None, None);
    let decoder = VideoDecoder::load(&config, loader.root()).unwrap();

    assert_eq!(decoder.element_type(), DType::F16);
    assert_eq!(decoder.latent_type(), DType::F32);

    // One latent frame of 17 by 17 cells: 272 by 272 pixels, two tiles of 256 pixels along
    // each axis that share the pixels 16..256, so each tile holds all but one latent row and
    // one latent column.
    let side = 17;
    let clip = Clip {
        frames: 4,
        height: 272,
        width: 272,
    };
    let layout = decoder.layout_of(clip).unwrap();
    assert_eq!(
        (layout.video.frames, layout.video.height, layout.video.width),
        (1, 17, 17)
    );
    assert_eq!(layout.video.channels, 2);
    let rows = decoder.latent_rows(&layout).unwrap().unwrap();
    assert_eq!(
        rows.dims(),
        [4 * 16 * 16],
        "a voxel of a seam has one row per tile that holds it"
    );

    let trajectory = tensor(
        &sequence(2 * side * side, 900),
        (2, side * side),
        DType::F32,
    );
    let whole = decoded(&decoder, &trajectory, clip);
    assert_eq!(whole.dims(), [3, 4, 272, 272]);
    assert!(values(&whole).iter().all(|pixel| pixel.is_finite()));

    // The first and the last tile decoded as rasters of their own: the latent rows and
    // columns 0..16, and 1..17.
    let tile = Clip {
        frames: 4,
        height: 256,
        width: 256,
    };
    let corner = |first: usize| {
        let latents = trajectory
            .reshape((2, side, side))
            .unwrap()
            .narrow(1, first, 16)
            .unwrap()
            .narrow(2, first, 16)
            .unwrap()
            .contiguous()
            .unwrap()
            .reshape((2, 16 * 16))
            .unwrap();

        decoded(&decoder, &latents, tile)
    };
    let (first, last) = (corner(0), corner(1));

    // Outside the seams the raster is the tile's own pixels: a tile's rows are computed as a
    // sequence of their own, so the values are those of its own decode, up to one FP16
    // spacing where a product over more rows sums in another order.
    let region = |raster: &Tensor, rows: (usize, usize), columns: (usize, usize)| {
        values(
            &raster
                .narrow(2, rows.0, rows.1)
                .unwrap()
                .narrow(3, columns.0, columns.1)
                .unwrap(),
        )
    };
    assert_close(
        &region(&whole, (0, 16), (0, 16)),
        &region(&first, (0, 16), (0, 16)),
        F16_SPACING,
    );
    assert_close(
        &region(&whole, (256, 16), (256, 16)),
        &region(&last, (240, 16), (240, 16)),
        F16_SPACING,
    );

    // A fade starts at a weight of zero: the first row of the seam below the first tile, and
    // the first column of the seam to its right, are still the first tile's own.
    assert_close(
        &region(&whole, (16, 1), (0, 16)),
        &region(&first, (16, 1), (0, 16)),
        F16_SPACING,
    );
    assert_close(
        &region(&whole, (0, 16), (16, 1)),
        &region(&first, (0, 16), (16, 1)),
        F16_SPACING,
    );

    // A raster that is not whole latent cells has no layout, and a trajectory or a block of
    // another element type is refused.
    assert!(
        decoder
            .layout_of(Clip {
                height: 264,
                ..clip
            })
            .is_err()
    );
    assert!(decoder.layout_of(Clip { frames: 5, ..clip }).is_err());

    let narrow = Tensor::zeros((4 * 16 * 16, 2), DType::F16, &Device::Cpu).unwrap();
    assert!(
        decoder
            .gather_latents(&layout, &trajectory, 0, &narrow)
            .is_err()
    );
    let video = Tensor::zeros((3, 4, 272, 272), DType::F16, &Device::Cpu).unwrap();
    assert!(
        decoder
            .decode(&layout, &narrow, Some(&video), None)
            .is_err()
    );
}

// ---- Audio ------------------------------------------------------------------------------------

/// The layout of a track of `frames` latent frames of the tiny audio VAE, as a denoiser states
/// it: the stereo pair's channels as the latent's channels.
fn track_layout(frames: u32) -> Layout {
    let empty = LatentShape {
        channels: 0,
        frames: 0,
        height: 0,
        width: 0,
    };

    Layout {
        video: empty,
        audio: Some(LatentShape {
            channels: 2 * 4,
            frames,
            height: 1,
            width: 1,
        }),
        text_rows: 0,
        condition_rows: 0,
        condition_elements: 0,
    }
}

#[test]
fn the_audio_decoder_computes_the_references_waveform() {
    let checkpoint = fixture("audio_decoder");
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None, None);
    let decoder = AudioDecoder::load(&audio_config(), loader.root()).unwrap();
    assert_eq!(decoder.sampling_rate(), 32000);
    assert_eq!(decoder.element_type(), DType::F32);

    // Five latent frames at a hop of four: twenty samples in each of two channels.
    let layout = track_layout(5);
    let window = decoder.pcm_window(&layout.audio.unwrap()).unwrap();
    assert_eq!(window, (20, 2));

    let rows = stored(&checkpoint, "input");
    let audio = Tensor::zeros((2, 20), DType::F32, &Device::Cpu).unwrap();
    decoder.decode(&layout, &rows, None, Some(&audio)).unwrap();

    // The restore, the input projection and the pre-convolution, two stages of an upsampling
    // and two dilation steps of two activations (a filter, a Snake and a filter each) and two
    // convolutions, their sum and average, then the last activation and convolution; the
    // pre-convolution sums 16 · 7 products.
    let expected = stored(&checkpoint, "expected");
    assert_close(
        &values(&audio),
        &values(&expected),
        bound(3 + 2 * (1 + 2 * 8 + 2) + 4, 16 * 7),
    );

    // The track is read by its element count: the denoiser's `[2 · frames, channels]` rows in
    // the latent's stated shape decode to the same waveform.
    let stated = rows.reshape((8, 5)).unwrap();
    let again = Tensor::zeros((2, 20), DType::F32, &Device::Cpu).unwrap();
    decoder
        .decode(&layout, &stated, None, Some(&again))
        .unwrap();
    assert_eq!(values(&again), values(&audio));

    // A waveform of another length, a track of another frame count and a raster are refused.
    let short = Tensor::zeros((2, 16), DType::F32, &Device::Cpu).unwrap();
    assert!(decoder.decode(&layout, &rows, None, Some(&short)).is_err());
    assert!(
        decoder
            .decode(&track_layout(4), &rows, None, Some(&audio))
            .is_err()
    );
    assert!(
        decoder
            .decode(&layout, &rows, Some(&audio), Some(&audio))
            .is_err()
    );
}

#[test]
fn the_audio_encoder_computes_the_references_latent_rows() {
    let checkpoint = fixture("audio_encoder");
    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None, None);
    let encoder = AudioEncoder::load(&audio_config(), loader.root()).unwrap();
    assert_eq!(encoder.hop(), 4);

    // The stereo pair of 24 samples: six latent frames a channel, the first channel's first.
    let samples = stored(&checkpoint, "input");
    let rows = encoder.encode(&samples).unwrap();
    assert_eq!(rows.dims(), [2 * 6, 4]);

    // The input convolution, two stages of three residual units (four stages each), a Snake
    // and a strided convolution, the output Snake and convolution, the projection's fourteen
    // stages, the mean and the normalization; a stage's dilated convolution sums 8 · 7
    // products.
    let expected = stored(&checkpoint, "expected");
    assert_close(
        &values(&rows),
        &values(&expected),
        bound(1 + 2 * 14 + 2 + 14 + 3, 8 * 7),
    );

    // A mono track, a track that is not whole latent frames and another element type are
    // refused.
    let mono = samples.narrow(0, 0, 1).unwrap().contiguous().unwrap();
    assert!(encoder.encode(&mono).is_err());

    let ragged = samples.narrow(2, 0, 23).unwrap().contiguous().unwrap();
    assert!(encoder.encode(&ragged).is_err());
    assert!(
        encoder
            .encode(&samples.to_dtype(DType::F16).unwrap())
            .is_err()
    );
}
