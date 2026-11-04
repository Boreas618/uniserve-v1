//! The descriptor keys the H3 deployment compiles, one `operator<TAB>key` line each.
//!
//! Run it with
//!
//! ```text
//! cargo run -p ops --example minimax_h3_keys
//! ```
//!
//! The output is `notes/programs/minimax-h3-keys.txt`: what a kernel program must package for
//! `crates/models/minimax-h3`, so no package has to guess a variant.
//!
//! ## Where the numbers come from
//!
//! Every geometry below is the checkpoint's, read from
//! `/opt/uv/models/MiniMax-H3/{transformer,audio_vae,vae}/config.json`, never inferred from prose:
//!
//! - transformer: `hidden_size 5376`, `num_attention_heads 56`, `attention_head_dim 128` (so
//!   attention is 56 heads of 128, not grouped), `ffn_dim 14336`, `num_layers 50`, `norm_eps` and
//!   `qk_norm_eps` 1e-5, `rope_freq_dim 16` per axis over the three axes with the leading 96
//!   channels of each head rotated and the trailing 32 left as they are
//!   (`uniserve/models/minimax_h3/attention.py`), `in_channels 24`, `audio_in_channels 32`,
//!   `text_dim 5120`, `patch_size [1, 2, 2]`;
//! - audio VAE: `latent_dim 2048`, `latent_channels 32`, `encoder_dim 64`, `encoder_rates
//!   [2, 4, 4, 5, 5]`, `decoder_dim 1024`, `decoder_rates [5, 5, 2, 2, 2, 2, 2]`,
//!   `decoder_kernel_sizes [9, 9, 4, 4, 4, 4, 4]`, `num_attention_heads 8` over the 2048-wide latent
//!   (so 8 heads of 256, FP32);
//! - video VAE: `in_channels 3`, `latent_channels 24`, `block_out_channels [128, 256, 256, 512,
//!   512, 1024]`, `norm_num_groups 32`, `norm_eps 1e-6`, spatial padding `reflect`,
//!   `decoder_num_attention_heads 32` of `decoder_attention_head_dim 64`.
//!
//! The BigVGAN decoder's channel schedule follows from `decoder_dim` and `decoder_rates`: the
//! upsamplers halve the width at each of the seven stages (`1024 // 2**(i+1)`), and the AMP blocks,
//! their activations and `activation_post` all sit at those widths
//! (`diffusers/models/autoencoders/autoencoder_kl_minimax_h3_audio.py:457-470`). The encoder doubles
//! from `encoder_dim` at each of its five rates, so its `Snake` runs at 128 through 2048.
//!
//! The tile and block constants belong to the layer crate, which `ops` sits below and cannot call;
//! they are mirrored here the way the Qwen3 fixture mirrors them
//! (`crates/nn/src/layers/mod.rs`, module `launch`; `ops::residual::RESIDUAL_BLOCK`,
//! `ops::noise::NOISE_BLOCK`). If the layer crate and this file disagree, the layer crate is right.
//!
//! ## What is deliberately not here yet
//!
//! Four families depend on numbers only `crates/models/minimax-h3` will fix, and a guess would cost
//! a package a wrong variant:
//!
//! - the attention caps: `max_kv_len` for the dense variants and the paged KV length. H3 packs
//!   frames × latent rows × (height/16) × (width/16) for video plus text and audio tokens, so the
//!   sequence is tens of thousands of rows at the deployment's resolutions — the cap is the model's
//!   scheduling decision, not a constant of the checkpoint;
//! - `vsa_select_tiles`, whose `block` is the number of key tiles one query tile may select;
//! - `gated_residual`, whose `elements` bakes `rows × width` into the descriptor and therefore has
//!   one variant per token count — the deployment's buckets are the model's;
//! - the `frame` operators, whose tile rule (`FramePadDescriptor::tiles`) follows the decoding
//!   resolution's pixel extents.
//!
//! They are added to this generator the hour the model names those numbers; nothing else waits.

use ops::DType;
use ops::modulation::ModulatedRmsNormDescriptor;
use ops::noise::NoiseDescriptor;
use ops::residual::{ScaledResidualDescriptor, WeightedRmsNormDescriptor};

/// Whether the layers' descriptors are the programmatic-dependent-launch variants.
const PDL: bool = false;

/// Elements one elementwise program handles: `snake`, `snake_beta`, `phased_residual`, `philox_noise`.
const ELEMENTWISE_BLOCK: u32 = 1024;

/// Samples one `audio_resample` program produces (`crates/nn/src/layers/`'s resampler tile).
const RESAMPLE_BLOCK: u32 = 256;

/// The audio VAE decoder's initial width (`decoder_dim`).
const DECODER_DIM: u32 = 1024;

/// The audio VAE decoder's upsampling rates: seven stages of the 800× total.
const DECODER_RATES: [u32; 7] = [5, 5, 2, 2, 2, 2, 2];

/// The audio VAE encoder's initial width (`encoder_dim`).
const ENCODER_DIM: u32 = 64;

/// The audio VAE encoder's strides: five blocks.
const ENCODER_RATES: [u32; 5] = [2, 4, 4, 5, 5];

/// The alias-free activation wrapper's upsampling ratio (`Activation1d(ratio=2)`).
const ACTIVATION_RATIO: u32 = 2;

/// The alias-free activation wrapper's Kaiser-sinc filter length (`Activation1d(kernel_size=12)`).
const ACTIVATION_KERNEL: u32 = 12;

/// The transformer's hidden width (`hidden_size`).
const HIDDEN: u32 = 5376;

/// The transformer's normalizer epsilon (`norm_eps`, `qk_norm_eps`).
const NORM_EPSILON: f32 = 1e-5;

/// The row width a row-wise kernel loads at a time: the width rounded up to a power of two
/// (`crates/nn/src/layers/mod.rs::row_tile`).
fn row_tile(columns: u32) -> u32 {
    columns.max(1).next_power_of_two()
}

fn main() {
    let mut lines: Vec<(String, String)> = Vec::new();
    let mut push = |operator: &str, descriptor: String| {
        lines.push((operator.to_owned(), descriptor));
    };

    // ---- Audio: the BigVGAN decoder's activations, at every width the seven stages produce.
    //
    // `snake_beta` is the decoder's: `MiniMaxH3AudioAMPBlock` and `activation_post` wrap
    // `MiniMaxH3AudioSnakeBeta` (`autoencoder_kl_minimax_h3_audio.py:412`, `:470`), whose two
    // parameters are log-space.
    for stage in 0..DECODER_RATES.len() as u32 {
        let channels = DECODER_DIM >> (stage + 1);
        push(
            "snake_beta",
            ops::snake_beta::Descriptor {
                dtype: DType::F32,
                channels,
                block: ELEMENTWISE_BLOCK,
                pdl: PDL,
            }
            .to_string(),
        );
    }

    // `snake` is the encoder's: one linear parameter, used by the DAC residual units and encoder
    // blocks (`:229`, `:231`, `:252`, `:278`, and `audio_vae.py:321`-`471`). The encoder doubles its
    // width at each rate.
    let mut encoder_channels = ENCODER_DIM;
    for rate in ENCODER_RATES {
        let _ = rate;
        encoder_channels *= 2;
        push(
            "snake",
            ops::snake::Descriptor {
                dtype: DType::F32,
                channels: encoder_channels,
                block: ELEMENTWISE_BLOCK,
                pdl: PDL,
            }
            .to_string(),
        );
    }

    // The decoder's upsamplers are the reference's `Activation1d`: a replicate pad, a transposed
    // convolution of stride 2, a `SnakeBeta`, then a low-pass of the same ratio — which is our
    // `audio_resample` in both directions at the same width, ratio 2 and a 12-tap filter.
    for stage in 0..DECODER_RATES.len() as u32 {
        let channels = DECODER_DIM >> (stage + 1);
        for mode in [
            ops::audio_resample::Mode::Upsample,
            ops::audio_resample::Mode::Downsample,
        ] {
            push(
                "audio_resample",
                ops::audio_resample::Descriptor {
                    mode,
                    dtype: DType::F32,
                    channels,
                    kernel: ACTIVATION_KERNEL,
                    ratio: ACTIVATION_RATIO,
                    block: RESAMPLE_BLOCK,
                    pdl: PDL,
                }
                .to_string(),
            );
        }
    }

    // ---- Media: the two programs whose variants do not depend on a resolution.
    //
    // The audio path pads the ends of a waveform with one axis (`leading = batch · channels`).
    push(
        "replicate_pad",
        ops::replicate_pad::Descriptor {
            dtype: DType::F32,
            axes: 1,
            block: ELEMENTWISE_BLOCK,
            pdl: PDL,
        }
        .to_string(),
    );
    // The video path's antialiased downscale of a packed uint8 frame, three colour channels.
    push(
        "antialiased_resize_uint8",
        ops::resize::Descriptor {
            mode: ops::resize::Mode::Bicubic,
            channels: 3,
            block: ELEMENTWISE_BLOCK,
            pdl: PDL,
        }
        .to_string(),
    );

    // ---- The transformer's per-row programs, at `hidden_size`.
    push(
        "modulated_rms_norm",
        ModulatedRmsNormDescriptor {
            dtype: DType::BF16,
            output: DType::BF16,
            indices: DType::I32,
            width: HIDDEN,
            block: row_tile(HIDDEN),
            epsilon: ops::Float::from(NORM_EPSILON),
            // The modulation vectors are `[rows, hidden]`: their row stride is the width.
            gate_stride: u64::from(HIDDEN),
            shift_stride: u64::from(HIDDEN),
            scale_stride: u64::from(HIDDEN),
            update: true,
            retain: false,
        }
        .to_string(),
    );
    push(
        "scaled_residual",
        ScaledResidualDescriptor {
            dtype: DType::BF16,
            columns: HIDDEN,
            block: ops::residual::RESIDUAL_BLOCK,
            update_bias: false,
        }
        .to_string(),
    );
    push(
        "weighted_rms_norm",
        WeightedRmsNormDescriptor {
            dtype: DType::BF16,
            columns: HIDDEN,
            tile: row_tile(HIDDEN),
            absmax: false,
        }
        .to_string(),
    );
    push(
        "philox_noise",
        NoiseDescriptor {
            dtype: DType::BF16,
            block: ops::noise::NOISE_BLOCK,
            // The initial latent noise: one standard normal sample per element.
            kind: ops::noise::NoiseKind::Normal,
        }
        .to_string(),
    );

    // One key per line, the same key once: a variant two layers launch is packaged once.
    lines.sort();
    lines.dedup();
    for (operator, descriptor) in lines {
        println!("{operator}\t{descriptor}");
    }
}
