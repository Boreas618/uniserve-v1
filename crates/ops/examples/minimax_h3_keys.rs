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

/// The rotary geometry: 16 frequencies per position axis over three axes (time, height, width),
/// so the leading 96 channels of each 128-channel head rotate and the trailing 32 do not.
const ROPE_FREQ_DIM: u32 = 16;
const ROPE_AXES: u32 = 3;
const ROTATED: u32 = ROPE_FREQ_DIM * ROPE_AXES * 2;

/// Elements one `rotary_factors` program handles (`launch::ROTARY_BLOCK`).
const ROTARY_BLOCK: u32 = 256;

/// Elements one `rows` program places: the layer's own `BLOCK` in `crates/nn/src/layers/rows.rs`.
const ROWS_BLOCK: u32 = 1024;

/// The video VAE decoder's token width: `decoder_num_attention_heads · decoder_attention_head_dim`.
const VAE_WIDTH: u32 = 32 * 64;

/// Tokens one `qk_norm_rope` program handles (`launch::QK_NORM_TOKENS`).
const QK_NORM_TOKENS: u32 = 1;

/// The attention projections' inner width, `num_attention_heads · attention_head_dim = 7168`: the
/// contraction length of the output projection (`attention.py`'s `RowParallelLinear(inner, hidden)`).
const HEADS_TIMES_HEAD_DIM: u32 = 7168;

/// The attention geometry: 56 heads of 128 features (`num_attention_heads`, `attention_head_dim`).
const HEADS: u32 = 56;
const HEAD_DIM: u32 = 128;

/// Rows of one VSA tile (`vsa_tile_size` in `fastvideo_inference.json`: 64 for the FastH3 students,
/// 128 for OmniRef). `uniserve/models/minimax_h3/packing.py` binds a 64-row tile to a `4×4×4`
/// spatiotemporal block and refuses any size it does not pack, so this is the checkpoint's number,
/// not a scheduling choice. The `vsa_sparsity: 0.8` beside it gives the selection's keep count, which
/// is a launch value: it follows the clip's tile count, so it is in no key.
const VSA_TILE_ROWS: u32 = 64;

/// Rows one VSA program handles (`_BLOCK_ROWS` in `uniserve_kernels/attention/vsa_segments.py`): the
/// kernel's own tiling inside a tile, compiled in.
const VSA_BLOCK_ROWS: u32 = 16;

/// The key tiles of the deployment's longest sparse reduction: 40960 keys at `VSA_TILE_ROWS`.
const VSA_MAX_TILES: u32 = 640;

/// The transformer's feed-forward width (`ffn_dim`).
const FFN: u32 = 14336;

/// The audio encoder's attention: 8 heads over the 2048-wide latent, so 256 features per head.
const AUDIO_ATTENTION_HEADS: u32 = 8;
const AUDIO_LATENT_DIM: u32 = 2048;

/// The audio VAE's latent channels (`latent_channels`): the encoder's narrow width.
const AUDIO_LATENT_CHANNELS: u32 = 32;

/// The modulation table's chunks per row: shift, scale and gate for each of the two sublayers
/// (`transformer.rs`'s `chunks` refuses a table wider or narrower than `6 · width`).
const MODULATION_CHUNKS: u32 = 6;

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
    // Padding is not one of them: the video VAE's replicate and reflect pads are `frame_pad`'s
    // `reflect` flag (`crates/ops/src/frame.rs`), and the audio decoder's pads are fused into
    // `audio_resample` and its convolutions. A general ATen `F.pad(mode="replicate")` over trailing
    // axes has no caller in the reference, so it is not declared.
    //
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

    // ---- The DiT's projections: one key per distinct contraction length.
    //
    // The transformer stores its weights as Hugging Face checkpoints do (`[n, k]`, `nxk`), contracts
    // in FP32 and adds no bias: `MergedColumnParallelLinear(..., bias=False)` for the fused q/k/v and
    // `RowParallelLinear` for the output, `ffn_dim` for the feed-forward (`attention.py`,
    // `transformer.py`). The three lengths are the attention projections' `hidden_size`,
    // the output projection's `heads · head_dim = 7168`, and the feed-forward's `ffn_dim = 14336`.
    for k in [HIDDEN, HEADS_TIMES_HEAD_DIM, FFN] {
        push(
            "gemm",
            ops::gemm::Descriptor {
                input: DType::BF16,
                weight: DType::BF16,
                accumulate: DType::F32,
                output: DType::BF16,
                order: ops::gemm::WeightOrder::NxK,
                epilogue: ops::gemm::Epilogue::None,
                k,
            }
            .to_string(),
        );
    }

    // The audio encoder is FP32 throughout (`Model.forward` casts, and the whole autoencoder runs in
    // FP32): its pre-block contracts at the latent width for the fused q/k/v and the shortcut, at
    // the head width for the attended rows, and at the latent channels for the GeGLU's two halves
    // and its output (`audio_vae.py:403-440`: `Linear(width, 3·width)`, `Linear(width, channels)`,
    // `Linear(channels, channels)`, `GeGLU(channels, 2·channels)`).
    for k in [
        AUDIO_LATENT_DIM,
        AUDIO_LATENT_DIM / AUDIO_ATTENTION_HEADS,
        AUDIO_LATENT_CHANNELS,
        2 * AUDIO_LATENT_CHANNELS,
    ] {
        push(
            "gemm",
            ops::gemm::Descriptor {
                input: DType::F32,
                weight: DType::F32,
                accumulate: DType::F32,
                output: DType::F32,
                order: ops::gemm::WeightOrder::NxK,
                epilogue: ops::gemm::Epilogue::None,
                k,
            }
            .to_string(),
        );
    }

    // ---- The DiT's rotation: three position axes of 16 frequencies each, cut into three sections
    // of the head's rotated columns, and the per-head normalization fused with it.
    //
    // `attention.py` normalizes every query and key head with a *learned* RMSNorm (`affine`) and
    // rotates the leading 96 channels with neox RoPE over (time, height, width).
    push(
        "rotary_factors",
        ops::rope::FactorsDescriptor {
            dtype: DType::F32,
            positions: DType::I32,
            columns: ROTATED / 2,
            block: ROTARY_BLOCK,
            // The layer's scale, not the rope base: every H3 `Rotary` is built with 1.0 and the
            // 1e4 base lives in the frequencies table the host fills.
            scale: ops::Float::from(1.0_f32),
            pdl: PDL,
            axes: ROPE_AXES,
            sections: {
                let mut sections = [0; ops::rope::MAX_AXES];
                sections[0] = ROPE_FREQ_DIM;
                sections[1] = ROPE_FREQ_DIM;
                sections[2] = ROPE_FREQ_DIM;
                sections
            },
            // H3's text encoder is Qwen3-VL: its M-RoPE cycles the axes over the frequencies.
            interleaved: true,
        }
        .to_string(),
    );
    push(
        "qk_norm_rope",
        ops::rope::QkNormRopeDescriptor {
            dtype: DType::BF16,
            output: DType::BF16,
            head_dim: HEAD_DIM,
            q_heads: HEADS,
            k_heads: HEADS,
            // "neox RoPE": adjacent pairs rotate together.
            rotation: ops::rope::Rotation::Interleaved,
            epsilon: ops::Float::from(NORM_EPSILON),
            rotated: ROTATED,
            rows: QK_NORM_TOKENS,
            block: row_tile(HEAD_DIM).min(1024),
            affine: true,
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
    // The gated residual that closes each block: `hidden + gate[i] · update`. The block's
    // modulation table is `[rows, 6 · hidden]` and the gate is the sixth chunk of a row
    // (`transformer.rs`'s `chunks(modulation, width, 3)`, which takes the last of the three MLP
    // chunks), so the gate's rows sit `6 · hidden` elements apart. The row count is the launch's,
    // not the key's: it changes with the clip length.
    push(
        "gated_residual",
        ops::modulation::GatedResidualDescriptor {
            dtype: DType::BF16,
            indices: DType::I32,
            width: HIDDEN,
            gate_stride: u64::from(MODULATION_CHUNKS) * u64::from(HIDDEN),
        }
        .to_string(),
    );
    // The solver's step flattens the latent to `[elements, 1]` and steps it by one scalar, so the
    // residual's width there is one column and the kernel's `WIDTH` is one.
    push(
        "scaled_residual",
        ScaledResidualDescriptor {
            dtype: DType::BF16,
            columns: 1,
            block: ops::residual::RESIDUAL_BLOCK,
            update_bias: false,
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
    // The DiT's attention: one dense program over the packed sequence. `max_kv_len` is a declared
    // bound outside the key (a padded-token bucket up to 174,598 for the widest layout), so the key
    // names only the geometry. The batch path carries `[batch + 1]` offsets.
    push(
        "dense_attention",
        ops::attention::dense::DenseDescriptor {
            query: DType::BF16,
            key_value: DType::BF16,
            output: DType::BF16,
            head_dim: HEAD_DIM,
            q_heads: HEADS,
            kv_heads: HEADS,
            mask: ops::attention::Mask::Dense,
            windowed: false,
            max_kv_len: 174_598,
            cu_seq_lens: true,
        }
        .to_string(),
    );
    // The VSA branch's own attention: one program per tile size, and the key extent (tiles per
    // segment per layout) is a declared bound outside the key.
    push(
        "sparse_attention",
        ops::attention::sparse::SparseDescriptor {
            block: 64,
            dtype: DType::BF16,
            output: DType::BF16,
            head_dim: HEAD_DIM,
            lse: false,
            keys: 65_536,
        }
        .to_string(),
    );
    // The DiT's row mapping: one key for every raster, because the program reads the canvas as a
    // launch word. A canvas is a multiple of the model's own step and the VAE compresses it 16x, so
    // every latent side a served canvas produces is even — that check is the operator's, at launch.
    push(
        "patch",
        ops::patch::Descriptor {
            dtype: DType::BF16,
            block: ops::patch::BLOCK,
            pdl: PDL,
        }
        .to_string(),
    );
    // The two `unpatchify_video` forms the decoder needs: the latents, whose per-channel scale and
    // bias denormalize, and the decoder's own output, which does not. The token grid, the channels
    // and the patch are the request's and travel at launch, so the two forms are the whole set.
    for affine in [true, false] {
        push(
            "unpatchify_video",
            ops::unpatchify::Descriptor {
                dtype: DType::F16,
                affine,
                block: 1024,
                pdl: PDL,
            }
            .to_string(),
        );
    }
    // The register prefix and the zero tail around the caller's rows.
    push(
        "sequence_append",
        ops::sequence_append::Descriptor {
            dtype: DType::BF16,
            width: HIDDEN,
            registers: 4,
            tail: 12,
            block: ops::sequence_append::BLOCK,
            pdl: PDL,
        }
        .to_string(),
    );
    // The seam between two decoded tiles, five voxels of overlap along the height.
    push(
        "cross_fade",
        ops::cross_fade::Descriptor {
            dtype: DType::F16,
            extent: 5,
            axis: 2,
            block: ops::cross_fade::BLOCK,
            pdl: PDL,
        }
        .to_string(),
    );
    // The latent reorder: the video decoder writes channel-first and the media stage reads
    // channel-last, so one strided copy moves the block. Two rows, because the two consumers work in
    // different precisions — the VAE's latent is FP32 (its native precision) and the VSA compose's
    // pooled block is BF16 — and the key names the element type and nothing else: the strides are
    // launch values, so one program serves every clip geometry.
    for dtype in [DType::F32, DType::BF16] {
        push(
            "copy_block",
            ops::copy_block::Descriptor {
                source: dtype,
                out: dtype,
            }
            .to_string(),
        );
    }
    // The condition encoder's last numerical step: the VAE encoder's moments to the sample a
    // conditioner reads, through its FP16 round trip. One row: the chain is fixed and in f32 by the
    // reference, so the only thing the key can name is the operator's own tiling -- and the two
    // launches the conditioned path makes (it does not use programmatic dependent launch) share it.
    push(
        "posterior",
        ops::posterior::Descriptor {
            block: ops::posterior::BLOCK,
            pdl: PDL,
        }
        .to_string(),
    );
    // The audio decoder's fused BigVGAN block, at the decoder's own rate and taps.
    push(
        "audio_activation",
        ops::audio_activation::Descriptor {
            dtype: DType::F32,
            channels: DECODER_DIM,
            kernel: ACTIVATION_KERNEL,
            ratio: ACTIVATION_RATIO,
            block: 1024,
            pdl: PDL,
        }
        .to_string(),
    );
    // The timestep embedding's activation, over the transformer's hidden width.
    push(
        "silu",
        ops::activation::SiluDescriptor {
            dtype: DType::BF16,
            columns: HIDDEN,
            block: 1024,
            pdl: PDL,
        }
        .to_string(),
    );
    // The solver's Euler step: an FP32 source added into a BF16 running sample.
    push(
        "scaled_add",
        ops::residual::ScaledAddDescriptor {
            target: DType::BF16,
            source: DType::F32,
            columns: HIDDEN,
            block: 1024,
        }
        .to_string(),
    );
    // The conditioner's anchoring mix runs over `[condition rows, video_channels * 4]`, which is
    // narrower than any hidden width, in the conditioner's own dtype and in the trace path's.
    for (target, source) in [(DType::F32, DType::F32), (DType::BF16, DType::BF16)] {
        push(
            "scaled_add",
            ops::residual::ScaledAddDescriptor {
                target,
                source,
                columns: 96,
                block: 1024,
            }
            .to_string(),
        );
    }
    // The audio decoder's last step: two float channels to interleaved 16-bit PCM. The clip's length
    // is a launch value, so the key holds only the element type and the block.
    push(
        "pcm_interleave",
        ops::pcm::Descriptor {
            dtype: DType::F32,
            block: ops::pcm::BLOCK,
            pdl: PDL,
        }
        .to_string(),
    );
    push(
        "philox_noise",
        NoiseDescriptor {
            dtype: DType::BF16,
            block: ops::noise::NOISE_BLOCK,
            // The initial latent noise: one standard normal sample per element, its stream keyed by
            // the request's seed as a launch scalar (the model has no key view to upload).
            kind: ops::noise::NoiseKind::Normal,
            source: ops::noise::KeySource::Scalar,
        }
        .to_string(),
    );

    // ---- The audio encoder's head pool, and the video path's last step.
    //
    // The pool is the projection's two fused reductions: `heads = num_attention_heads 8`, the head
    // width `latent_dim / heads = 256`, pooled to `latent_channels = 32` (`audio_vae.py:431-440`,
    // `audio_vae/config.json`).
    push(
        "adaptive_pool",
        ops::adaptive_pool::Descriptor {
            dtype: DType::F32,
            heads: AUDIO_ATTENTION_HEADS,
            width: AUDIO_LATENT_DIM / AUDIO_ATTENTION_HEADS,
            output: AUDIO_LATENT_CHANNELS,
            block: ELEMENTWISE_BLOCK,
            pdl: PDL,
        }
        .to_string(),
    );
    // The video path's last step: the VAE's raster is `float16`
    // (`uniserve/models/minimax_h3/output.py`'s own state buffer), three channels, packed bytes.
    push(
        "rgb_bytes",
        ops::rgb::Descriptor {
            dtype: DType::F16,
            channels: 3,
            block: ops::rgb::BLOCK,
            pdl: PDL,
        }
        .to_string(),
    );

    // ---- The text encoder (Qwen3-VL, `text_encoder/config.json`): its norms and its two audited
    // activations. `hidden_size 5120` for the language model, `intermediate_size` 25600 gated, and
    // the vision tower's `gelu_pytorch_tanh` over its own `intermediate_size 4304`.
    //
    // `rms_norm`/`add_rms_norm` are compiled per column count and tile, the tile being the layer's
    // `next_power_of_two(columns)`; the element type is the checkpoint's `bfloat16`.
    let text_columns = 5120_u32;
    push(
        "rms_norm",
        ops::norm::RmsNormDescriptor {
            dtype: DType::BF16,
            columns: text_columns,
            tile: text_columns.next_power_of_two(),
            pdl: PDL,
        }
        .to_string(),
    );
    push(
        "add_rms_norm",
        ops::norm::AddRmsNormDescriptor {
            dtype: DType::BF16,
            columns: text_columns,
            tile: text_columns.next_power_of_two(),
            pdl: PDL,
        }
        .to_string(),
    );
    // The audio encoder's GeGLU applies tanh-GELU over its hidden width, `2 · latent_channels`
    // (`audio_vae.py:417`, `GeGLU(channels, 2 * channels)`), in FP32.
    push(
        "gelu",
        ops::gelu::Descriptor {
            dtype: DType::F32,
            columns: AUDIO_LATENT_CHANNELS * 2,
            block: ELEMENTWISE_BLOCK,
            approximate: ops::gelu::Approximation::Tanh,
            pdl: PDL,
        }
        .to_string(),
    );
    // Its norms are `[patches, 1152]`, so the tile is the layer's `next_power_of_two`.
    push(
        "layer_norm",
        ops::norm::LayerNormDescriptor {
            dtype: DType::BF16,
            columns: 1152,
            tile: 1152_u32.next_power_of_two(),
            bias: true,
            pdl: PDL,
        }
        .to_string(),
    );
    // Its 2-D rotary turns each head's leading half against the trailing half (`Rotation::Half`,
    // `vision.rs`'s `Rotate::new(head_dim, heads, Rotation::Half)`), 8 rows and 128 lanes per program
    // (`nn::layers::rotate`'s own constants); the whole 72-element head rotates, so no narrower
    // `rotated` is appended.
    push(
        "rope_apply",
        ops::rope::RopeApplyDescriptor {
            dtype: DType::BF16,
            output: DType::BF16,
            head_dim: 72,
            heads: 16,
            rotation: ops::rope::Rotation::Half,
            rows: 8,
            block: 128,
            pdl: PDL,
            rotated: 72,
        }
        .to_string(),
    );
    // Its learned position table, bilinearly resampled from the 48 × 48 it was trained on to the
    // patch grid a request's frame count produces. `channels` and `in_side` are compiled in; the
    // output's grid is the args' extent.
    push(
        "position_interpolate",
        ops::position::Descriptor {
            dtype: DType::BF16,
            channels: 1152,
            in_side: 48,
            block: 1024,
            pdl: PDL,
        }
        .to_string(),
    );

    // The vision tower is `Qwen3VLVisionModel` (`text_encoder/config.json`'s `vision_config`):
    // `hidden_size 1152`, `num_heads 16` — so `head_dim 72` — 27 blocks, `num_position_embeddings
    // 2304` (a 48 × 48 learned table), `patch_size 16`, `temporal_patch_size 2`, and both weight and
    // bias on every LayerNorm (the checkpoint's `model.visual.*.norm{1,2}.bias` entries). Its MLP
    // activation is `gelu_pytorch_tanh` over `intermediate_size 4304`.
    push(
        "gelu",
        ops::gelu::Descriptor {
            dtype: DType::BF16,
            columns: 4304,
            block: ELEMENTWISE_BLOCK,
            approximate: ops::gelu::Approximation::Tanh,
            pdl: PDL,
        }
        .to_string(),
    );

    // ---- The vision tower's products, all BF16 against FP32 accumulation with no epilogue.
    //
    // The key names the contraction, not the output width, so four rows cover nine call sites: the
    // patch projection reads the flat tubelet (`in_channels · temporal_patch_size · patch²` = 3 · 2 ·
    // 16² = 1536, `vision.rs`'s `Geometry::tubelet`); the fused qkv branches and the attention
    // projection contract over `hidden_size` 1152; the MLP's first matrix does too and its second
    // contracts over `intermediate_size` 4304; and both merger matrices read the merged row
    // (`hidden_size · spatial_merge_size²` = 4 · 1152 = 4608, `Geometry::merged_width`), the deepstack
    // mergers included.
    for k in [1536_u32, 1152, 4304, 4608] {
        push(
            "gemm",
            ops::gemm::Descriptor {
                input: DType::BF16,
                weight: DType::BF16,
                accumulate: DType::F32,
                output: DType::BF16,
                order: ops::gemm::WeightOrder::NxK,
                epilogue: ops::gemm::Epilogue::None,
                k,
            }
            .to_string(),
        );
    }

    // Its attention and its residual add, at the tower's own geometry: 16 heads of 72, a dense mask
    // with no window, and **one sequence per time step** — `vision.rs` calls `attend_dense` with the
    // offsets the grid's time steps give it, so the key carries no `cu_seq_lens` suffix (that name is
    // written only for the form without offsets).
    push(
        "dense_attention",
        ops::attention::DenseDescriptor {
            query: DType::BF16,
            key_value: DType::BF16,
            output: DType::BF16,
            head_dim: 72,
            q_heads: 16,
            kv_heads: 16,
            mask: ops::attention::Mask::Dense,
            windowed: false,
            // A declared bound outside the key, sized for the tower's longest sequence: its learned
            // position table is `num_position_embeddings 2304`, and no attention sequence is longer
            // than the patches that table covers.
            max_kv_len: 2304,
            cu_seq_lens: true,
        }
        .to_string(),
    );
    // The tower's residual stream is `[patches, 1152]` with a plain scale and no bias update.
    push(
        "scaled_residual",
        ops::residual::ScaledResidualDescriptor {
            dtype: DType::BF16,
            columns: 1152,
            block: ELEMENTWISE_BLOCK,
            update_bias: false,
        }
        .to_string(),
    );

    // ---- The video VAE's encoder, `vae/config.json` and `encoding.rs`'s own construction.
    //
    // The encoder is FP32 and every convolution passes `tf32 = true` at `Conv3d::new`, so the flag is
    // the model's declaration rather than a preset's inference. Its geometry is cubic with zero
    // padding and unit dilation (`encoding.rs::geometry`), and the downsampler of a stage strides by
    // that stage's factors: `spatial_downsample_factors [2,2,2,2,1,1]` against
    // `temporal_downsample_factors [1,2,2,1,1,1]` gives three distinct pairs — `[1,2,2]`, `[2,2,2]`
    // and the two stages whose product is 1, which carry no downsampler at all — so four rows cover
    // every convolution of the encoder and the whole of `quant_conv`.
    for (kernel, stride) in [
        ([3_u32, 3, 3], [1_u32, 1, 1]),
        ([3, 3, 3], [1, 2, 2]),
        ([3, 3, 3], [2, 2, 2]),
        ([1, 1, 1], [1, 1, 1]),
    ] {
        push(
            "conv3d",
            ops::conv::Descriptor {
                input: DType::F32,
                weight: DType::F32,
                output: DType::F32,
                geometry: ops::conv::Geometry {
                    kernel,
                    stride,
                    padding: [0, 0, 0],
                    dilation: [1, 1, 1],
                    groups: 1,
                },
                bias: true,
                tf32: true,
                rank: 3,
                transposed: false,
            }
            .to_string(),
        );
    }
    // Its per-frame group statistics over `norm_num_groups 32` with `norm_eps 1e-6`.
    push(
        "frame_moments",
        ops::frame::FrameMomentsDescriptor {
            dtype: DType::F32,
            // The encoder's widest frame: four channels per group over a 1344x768 raster. The class
            // is a declared bound and not a compilation, so it does not reach the key - the launch
            // refuses a frame outside it.
            reduction: ops::frame::frame_reduction_class(4, 1344 * 768),
            epsilon: ops::descriptor::Float::from(1e-6_f32),
        }
        .to_string(),
    );
    // Its normalize-then-pad, with the affine parameters the checkpoint carries (`centered` is the
    // form without them) and `spatial_padding_mode reflect`. `single_pixel` is the descriptor's form
    // for a one-pixel raster, which needs no padding at all: both are listed because which one a
    // request takes depends on the clip's latent grid, not on a compiled choice.
    for single_pixel in [false, true] {
        push(
            "frame_norm_pad",
            ops::frame::FrameNormPadDescriptor {
                dtype: DType::F32,
                reduction: ops::frame::frame_reduction_class(4, 1344 * 768),
                channels_last: false,
                centered: false,
                single_pixel,
            }
            .to_string(),
        );
    }

    // ---- The text encoder's MLP and its token embedding, and the two packed gated activations.
    //
    // The language layers' MLP is a `GatedActivation` over two *separate* contiguous projections —
    // `text_encoder.rs` runs its own `gate` and `up` Linears into their own `[tokens, intermediate]`
    // buffers before the activation — so both row strides are the intermediate width, 25600. The
    // activation is `silu` (`hidden_act`), the layer's block is 1024, and `wide` is false because the
    // output is in bytes far below the u32 element count.
    push(
        "act_and_mul",
        ops::activation::ActAndMulDescriptor {
            dtype: DType::BF16,
            width: 25600,
            block: 1024,
            activation: ops::activation::Activation::Silu,
            gate_stride: 25600,
            value_stride: 25600,
            wide: false,
            pdl: PDL,
            stepwise: false,
        }
        .to_string(),
    );
    // The two packed forms: the DiT's block MLP and the conditioning projection both contract to
    // `ffn_dim 14336` in BF16 from a packed `[rows, 2 · width]` row, and the video VAE decoder's to
    // its own `mlp` width 8192 in FP16 (`decoders.md`'s arena: `mlp [rows, 8192]`). The row stride is
    // twice the width because the two halves are packed beside each other, and no call site passes a
    // bias.
    for (dtype, width) in [(DType::BF16, 14336_u32), (DType::F16, 8192)] {
        push(
            "value_first_swiglu",
            ops::activation::ValueFirstSwigluDescriptor {
                dtype,
                width,
                block: 1024,
                input_stride: 2 * u64::from(width),
                bias: false,
                absmax: false,
            }
            .to_string(),
        );
    }
    // The text encoder's token embedding, at the single-partition shape: the whole vocabulary
    // (151936 rows of 5120) with i32 ids, eight tokens per program. A sharded deployment re-keys it,
    // because `first` and `rows` are the rank's slice of the vocabulary.
    push(
        "embedding",
        ops::embedding::Descriptor {
            dtype: DType::BF16,
            ids: DType::I32,
            hidden: 5120,
            first: 0,
            rows: 151936,
            block: 8,
            pdl: PDL,
        }
        .to_string(),
    );

    // The encoder's causal frame padding, in FP32 with reflect edges (`spatial_padding_mode`).
    //
    // **Its tile is a function of the raster's width**, not a constant: `FramePadDescriptor::tiles`
    // takes `min(ROW_PIXELS 512, next_power_of_two(columns))` pixels per program and divides
    // `ROW_ELEMENTS 2048` by it, so a raster at least 512 pixels wide gives the 512-pixel tile with
    // four channels per program — every realistic clip width — and a narrower one re-keys to its own
    // pixel block. This row is the wide case; a request whose width is under 512 compiles another.
    push(
        "frame_pad",
        ops::frame::FramePadDescriptor {
            dtype: DType::F32,
            reflect: true,
            block_pixels: 512,
            block_channels: 4,
        }
        .to_string(),
    );

    // ---- The audio decoder's and encoder's convolutions, in FP32 with TF32 declared on.
    //
    // The decoder's seven upsamplers are `ConvTranspose1d` with `kernel = decoder_kernel_sizes`,
    // `stride = decoder_rates`, `padding = (kernel - rate) / 2` — two distinct geometries for the
    // seven stages (`autoencoder_kl_minimax_h3_audio.py:447-459`): `(9, 5, 2)` twice and `(4, 2, 1)`
    // five times. The key names the geometry, not the channels, so two rows cover all seven.
    // The key builder: the operator's name and the descriptor, pushed by the caller so that one
    // closure does not hold `push` for the rest of the section.
    let convolution = |kernel: [u32; 3],
                       stride: [u32; 3],
                       padding: [u32; 3],
                       dilation: [u32; 3],
                       transposed: bool|
     -> (&'static str, String) {
        (
            if transposed {
                "conv_transpose1d"
            } else {
                "conv1d"
            },
            ops::conv::Descriptor {
                input: DType::F32,
                weight: DType::F32,
                output: DType::F32,
                geometry: ops::conv::Geometry {
                    kernel,
                    stride,
                    padding,
                    dilation,
                    groups: 1,
                },
                bias: true,
                // The preset's declaration for a model convolution in FP32 (the video VAE keeps it
                // on, and the audio VAEs are the same FP32 numerical path).
                tf32: true,
                rank: 1,
                transposed,
            }
            .to_string(),
        )
    };
    let (operator, key) = convolution([9, 1, 1], [5, 1, 1], [2, 1, 1], [1, 1, 1], true);
    push(operator, key);
    let (operator, key) = convolution([4, 1, 1], [2, 1, 1], [1, 1, 1], [1, 1, 1], true);
    push(operator, key);
    // The decoder's entry and exit convolutions, and the pointwise narrowing inside a residual unit.
    let (operator, key) = convolution([7, 1, 1], [1, 1, 1], [3, 1, 1], [1, 1, 1], false);
    push(operator, key);
    let (operator, key) = convolution([1, 1, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1], false);
    push(operator, key);
    // The decoder's residual units: the three resblock kernels, each at the three dilations
    // (`resblock_kernel_sizes [3, 7, 11]`, `resblock_dilation_sizes [1, 3, 5]`).
    for kernel in [3_u32, 7, 11] {
        for dilation in [1_u32, 3, 5] {
            push(
                "conv1d",
                ops::conv::Descriptor {
                    input: DType::F32,
                    weight: DType::F32,
                    output: DType::F32,
                    geometry: ops::conv::Geometry {
                        kernel: [kernel, 1, 1],
                        stride: [1, 1, 1],
                        padding: [(dilation * (kernel - 1)) / 2, 1, 1],
                        dilation: [dilation, 1, 1],
                        groups: 1,
                    },
                    bias: true,
                    tf32: true,
                    rank: 1,
                    transposed: false,
                }
                .to_string(),
            );
        }
    }
    // The encoder's residual units are a seven-tap convolution at dilations 1, 3 and 9
    // (`audio_vae.py:352-354`, padding `3 · dilation`), and its exit convolution is three taps at
    // padding 1. Two of these rows coincide with the decoder's seven-tap rows above.
    for dilation in [1_u32, 3, 9] {
        let (operator, key) = convolution(
            [7, 1, 1],
            [1, 1, 1],
            [3 * dilation, 1, 1],
            [dilation, 1, 1],
            false,
        );
        push(operator, key);
    }
    let (operator, key) = convolution([3, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], false);
    push(operator, key);
    // The encoder's strided convolutions: `kernel = 2 · stride`, `padding = ceil(stride / 2)`
    // (`audio_vae.py:355-363`), for the three distinct rates of `encoder_rates [2, 4, 4, 5, 5]`.
    for stride in [2_u32, 4, 5] {
        let (operator, key) = convolution(
            [2 * stride, 1, 1],
            [stride, 1, 1],
            [stride.div_ceil(2), 1, 1],
            [1, 1, 1],
            false,
        );
        push(operator, key);
    }

    // ---- The VSA selection's pooled-score product: the stage-1 video attention's per-head tile
    // similarity, over the transformer's own head geometry (56 heads of 128). The operands are the
    // pooled means in FP32 and so are the scores, so no element type is in the key; the lanes one
    // program uses for the contraction are the head width, and the pooling itself (the tile
    // reduction and its stride) is `vsa_prepare_tiles`' launch values.
    push(
        "vsa_scores",
        ops::segments::ScoresDescriptor::new(HEADS, HEAD_DIM, HEAD_DIM.next_power_of_two(), PDL)
            .to_string(),
    );

    // The compression the gate adds to every query row: one softmax over the key tiles. `max_tiles`
    // is the deployment's longest sparse reduction — 40960 keys at the 64-key tile — because it is
    // the block the softmax reduces in, and so the reduction length the tolerance is written
    // against. The pooled scores and values are FP32, so the result's type is the whole of `dtype`.
    push(
        "vsa_compress_pooled",
        ops::segments::CompressPooledDescriptor::new(
            DType::BF16,
            HEADS,
            HEAD_DIM,
            VSA_MAX_TILES,
            HEAD_DIM.next_power_of_two(),
            PDL,
        )
        .to_string(),
    );

    // The pooling that feeds it, and the two tile passes around the attention, all at the tile the
    // contract names. Q, K and V are contiguous `[rows, 56, 128]` views of one projected interval, so
    // a row is 7168 elements and a head 128; stage-1 Q and K are normalized and rotated over the
    // leading 96 of their 128 channels (`attention.py`'s `vsa.NormRope`), which is `ROTATED`.
    push(
        "vsa_prepare_tiles",
        ops::segments::PrepareTilesDescriptor {
            dtype: DType::BF16,
            heads: HEADS,
            width: HEAD_DIM,
            rotated: ROTATED,
            tile_rows: VSA_TILE_ROWS,
            block_rows: VSA_BLOCK_ROWS,
            epsilon: ops::descriptor::Float::from(NORM_EPSILON),
            norm: true,
            query_row_stride: u64::from(HEADS_TIMES_HEAD_DIM),
            query_head_stride: u64::from(HEAD_DIM),
            key_row_stride: u64::from(HEADS_TIMES_HEAD_DIM),
            key_head_stride: u64::from(HEAD_DIM),
            value_row_stride: u64::from(HEADS_TIMES_HEAD_DIM),
            value_head_stride: u64::from(HEAD_DIM),
            // A split-half factor table holds `rotated / 2` values a row.
            factor_stride: u64::from(ROTATED / 2),
        }
        .to_string(),
    );
    // A tile's rows past its valid size hold zeros, on both sides of the attention.
    push(
        "vsa_zero_padding",
        ops::segments::ZeroPaddingDescriptor {
            dtype: DType::BF16,
            heads: HEADS,
            tile_rows: VSA_TILE_ROWS,
            width: HEAD_DIM,
            block_rows: VSA_BLOCK_ROWS,
        }
        .to_string(),
    );
    // The compressed gated tiles that give back the sparse rows.
    push(
        "vsa_add_gated_tiles",
        ops::segments::AddGatedTilesDescriptor {
            dtype: DType::BF16,
            tile_rows: VSA_TILE_ROWS,
            width: HEAD_DIM,
            block_rows: VSA_BLOCK_ROWS,
        }
        .to_string(),
    );

    // ---- The audio encoder's frame/channel swap, in the audio VAEs' own FP32.
    //
    // `AudioVAEEncoder` holds `[batch, channels, frames]` and its attention attends over frames, so
    // frames move to the leading axis before the qkv projection and back after it. Both extents are
    // launch values, so one row covers every clip length.
    push(
        "transpose",
        ops::transpose::Descriptor {
            dtype: DType::F32,
            block: ops::transpose::TILE,
            pdl: PDL,
        }
        .to_string(),
    );

    // Placing the rows a tower contributes into the sequence that reserved them. Four call sites
    // exist and each is in a layer of `crates/models/minimax-h3`, all of them at the layer crate's
    // one element block (its `rows.rs`'s `BLOCK`, 1024) and none on a dependent launch:
    //
    // - the text encoder's prompt reserves placeholder rows for its images and the tower's vision
    //   rows *replace* them (`text_encoder.rs`, `Rows::replace`), at the language width;
    // - a DeepStack tap *adds* a tower layer's rows into rows that already hold a residual
    //   (`text_encoder.rs`, `Rows::add`), at the same width;
    // - the denoiser scatters its packed video rows and its audio rows into the one sequence
    //   (`denoiser.rs`, `Rows::replace`), at the transformer's hidden width — one key for both
    //   modalities, because the operator reads only rows, columns and the index view;
    // - the video VAE decoder starts its stream by placing the projected voxels, its register rows
    //   and its zero row into the token rows (`video_vae.rs`, `Rows::replace`), at the decoder's
    //   `decoder_num_attention_heads · decoder_attention_head_dim = 32 · 64` and in FP16.
    //
    // The merge that makes a language-width token out of a `2x2` block of patches is *not* here: it
    // is a reshape of the contiguous `[patches, 1152]` hidden state into `[patches/4, 4608]`, so no
    // rows are gathered and no program reads them.
    for (dtype, mode, columns) in [
        (DType::BF16, ops::rows::Mode::Replace, text_columns),
        (DType::BF16, ops::rows::Mode::Add, text_columns),
        (DType::BF16, ops::rows::Mode::Replace, HIDDEN),
        (DType::F16, ops::rows::Mode::Replace, VAE_WIDTH),
    ] {
        push(
            "rows",
            ops::rows::Descriptor {
                dtype,
                mode,
                columns,
                block: ROWS_BLOCK,
                pdl: PDL,
            }
            .to_string(),
        );
    }

    // Gaps this file does not fill, named rather than invented. Each is a number the deployment
    // owns, not one this file could read off a config:
    // - `bias_add`: the audio decoder's channels are derivable and the video VAE's are not yet, and
    //   in both cases the two flags belong to the layer that calls it. The audio decoder halves
    //   `decoder_dim` once per upsample (`diffusers`' `MiniMaxH3AudioBigVGANDecoder.ups`:
    //   `upsample_initial_channel // 2**i` into `// 2**(i+1)`, and its resblocks sit at the output
    //   width), so its call sites are 1024, 512, 256, 128, 64, 32, 16 and 8; the video VAE's widths
    //   come from its own stage list once that decoder is composed. `residual` and `residual_bias`
    //   decide the key, and they are how the block folds its residual — the layer's, not the config's.
    // - `paged_attention`: the decode attention is trtllm-gen, and its key follows the executor's
    //   page geometry (page size, pages per sequence), which is scheduling and not the model's.
    // - `vsa_select_tiles`: the number of key tiles one query tile may select is the deployment's
    //   sparsity choice; `vsa_prepare_tiles`, `vsa_scores`, `vsa_add_gated_tiles` and
    //   `vsa_zero_padding` are above, because their geometry is the checkpoint's.
    // - `softmax`, `swiglu`, `scaled_residual_rms_norm`, `scaled_residual_layer_norm`,
    //   `vsa_compose_gated_rows`: declared, and nothing in either model's code calls them yet, so
    //   there is no geometry to key. They are named here rather than left to be found: the first two
    //   belong to the sampler's chain and the Qwen3 slice, the two residuals to a stream that folds
    //   its normalization in one launch, and the composition to the VSA chain's last step.

    // One key per line, the same key once: a variant two layers launch is packaged once.
    lines.sort();
    lines.dedup();
    for (operator, descriptor) in lines {
        println!("{operator}\t{descriptor}");
    }
}
