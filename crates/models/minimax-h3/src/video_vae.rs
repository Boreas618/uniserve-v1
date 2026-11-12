//! The video VAE: the transformer decoder that turns a clip of latents into frames, and the
//! causal convolutional encoder that turns a condition's pixels into its latent.
//!
//! ## The decoder
//!
//! The decoder is a non-causal ViT (the reference's `ViT3DDecoder`,
//! `refs/sglang/python/sglang/multimodal_gen/runtime/models/vaes/minimax_h3_video_vae/vae_vit.py:237-461`).
//! `post_quant_conv` projects the restored latent channels, `proj_in` embeds each latent voxel
//! as one token, the learned register tokens and one zero token trail the voxels, the
//! transformer blocks attend over the whole sequence, and `proj_out` writes
//! `out_channels · temporal_compression · spatial_compression²` elements per token: the
//! voxel's block of pixels.
//!
//! A raster wider than one tile is decoded in overlapping tiles (`klvae.py:449-501`), each tile
//! a sequence of its own, so nothing crosses between tiles until the stitch. Every tile's rows
//! sit in one stream: each projection is one product over all of them, and the attention is
//! one launch over as many sequences as there are tiles.
//!
//! ## Precision
//!
//! The reference runs the decoder under `torch.autocast(dtype=FP16)`, and what autocast does to
//! each operation is the table below. It is the model's own declaration, stated here in code,
//! so it takes no preset and no switch; every parameter the table lists as stored stays in the
//! checkpoint's own FP32.
//!
//! ```text
//! latents, the restored latents        FP32   (`latents.float() * std + mean`)
//! post_quant_conv                      FP16 weights; the restored latents round to FP16 on the
//!                                      way in, and the product rounds before its bias joins it
//! proj_in                              FP16 weights and output
//! register tokens, norms, scales       stored (FP32)
//! the residual stream                  FP32   (`cat` of an FP16 projection with FP32 registers
//!                                              promotes; every sum after it is FP32)
//! the normalized rows                  FP16   (FP32 statistics of the unrounded sum, one rounding)
//! rotary factors                       FP32 tables of FP32 positions
//! to_q, to_k, to_v, to_out, ff         FP16 weights, bias and output
//! per-head norm and rotation           FP16 rows, FP32 arithmetic, one rounding
//! attention                            FP16 operands, FP32 softmax and accumulators
//! the gated activation                 FP16 rows, FP32 arithmetic, one rounding
//! proj_out                             FP16; the raster and the seam fades stay FP16
//! ```
//!
//! The residual stream is the one wide tensor: a projection's output joins it in FP32, and
//! everything a projection reads is a rounding of it.
//!
//! ## The encoder
//!
//! The encoder is the reference's `EncoderFCN3D` (`vae_cnn.py:160-268`): causal 3-d
//! convolutions over per-frame group normalizations, every convolution FP32 with TF32 allowed.
//! A stride-`s` temporal convolution pads two zero frames in front and none behind, so `T`
//! frames become `ceil(T / s)`; a spatial downsampling pads the bottom and the right by one
//! pixel and strides by two. A condition is cut into the clips of the frame rule
//! ([`FRAMES`]), each clip encoded alone in overlapping tiles whose moments cross-fade where
//! they meet ([`nn::vae::tiled_encode`]).

use std::ops::Range;

use nn::functional::{self, Activation, Padding, Rounding};
use nn::modules::{
    Attention, AttentionConfig, Conv3d, ConvConfig, FrameNorm, FrameNormConfig, GateUp, Linear,
    LinearConfig, Mlp, MlpConfig, Parallel, Precision, QkNorm, QkvBias, QkvLinearConfig, QkvNames,
    ResnetBlock3d, ResnetBlock3dConfig, Rotary, RotaryConfig, Spectrum,
};
use nn::vae::{self, TemporalClips};
use nn::{
    Clip, DType, Decoder, Device, LatentShape, Layout, PixelNorm, Splice, Split, Tensor, VarBuilder,
};
use processing::minimax_h3::{FRAMES, LATENT_COMPRESSION};

use crate::{Error, Result, VideoVaeConfig};

/// The ImageNet constants a video raster is normalized by, in red, green, blue order: the
/// decoder's output and the encoder's condition pixels alike (the reference's
/// `pixel_norm_type="imagenet"`). `value · std + mean` restores the pixel intensity.
pub const PIXEL_NORM: PixelNorm = PixelNorm {
    mean: [0.485, 0.456, 0.406],
    std: [0.229, 0.224, 0.225],
};

/// The seed the encoder samples every condition's posterior under.
///
/// It is the checkpoint's recipe and not the request's (the reference's
/// `MINIMAX_H3_KEYFRAME_ENCODE_SEED`): the posterior's noise is its own stream from element
/// zero, so a condition always encodes to the same latent, whatever the request's seed draws.
pub const POSTERIOR_SEED: u64 = 42;

/// The extent of one tile, in pixels (the reference's `vae_tile_size`).
const TILE_SIZE: usize = 256;

/// The fewest pixels two neighbouring tiles share (the reference's `vae_tile_overlap_min`).
const TILE_OVERLAP_MIN: usize = 64;

/// The element type of the decoder's projections, of the rows they read and write, and of the
/// raster (module documentation).
const ACTIVATION: DType = DType::F16;

/// The element type of the latents a decode reads and of the residual stream.
const STREAM: DType = DType::F32;

/// The zero frames a causal convolution of three taps reads before a clip's first frame.
const CAUSAL_FRONT: usize = 2;

/// The temporal clips the checkpoint's decoder decodes a timeline in, by the frame rule
/// ([`FRAMES`]).
///
/// A clip keeps the rule's clip frames, skips the padding frames the final clip drops, and
/// ends in the rule's tail, which the next clip fades its first frames against (the
/// reference's `frame_overlap`): 17 + 3 + 5 = 25 frames for the released checkpoint, so a clip
/// reads `ceil(25 / 4) = 7` latent frames from `5k` and its decode's leading surplus is
/// cropped.
pub fn temporal_clips(config: &VideoVaeConfig) -> TemporalClips {
    TemporalClips {
        body: FRAMES.clip_frames,
        padding: FRAMES.token_drop(),
        overlap: FRAMES.tail_frames(),
        temporal: config.temporal_compression(),
        advance: FRAMES.clip_latent_frames,
    }
}

/// The splices a timeline of `frames` frames is decoded and written in: one per clip of the
/// frame rule.
///
/// # Errors
///
/// A frame count the frame rule does not generate.
pub fn splices(clips: &TemporalClips, frames: u32) -> Result<Vec<Splice>> {
    if !FRAMES.is_legal(frames) {
        return Err(Error::request(format!(
            "a timeline is a frame count of the frame rule {FRAMES:?}, and {frames} is not"
        )));
    }

    Ok(clips.splices(FRAMES.units(frames))?)
}

/// The rotary embedding of the decoder's attention.
///
/// `decoder_rope_dim_ratio` of a head rotates (48 of 64 channels), split evenly over the frame,
/// row and column axes; each axis repeats one spectrum of base `decoder_rope_theta`, evaluated
/// in FP32 as the reference's `1 / base ** arange(0, 1, 2n / dim, float32)` evaluates it
/// (`base_module.py:125-144`). The trailing channels of a head pass through.
pub fn rotary_config(config: &VideoVaeConfig) -> RotaryConfig {
    let rotated =
        f64::from(config.decoder_attention_head_dim) * f64::from(config.decoder_rope_dim_ratio);

    RotaryConfig {
        theta: config.decoder_rope_theta,
        dim: rotated.round() as u32,
        spectrum: Spectrum::Repeated { axes: 3 },
        precision: Precision::F32,
    }
}

/// The rotary coordinates of one sequence of the decoder: a `[3, rows]` FP32 CPU tensor,
/// axis-major, of the `frames · height · width` voxel tokens in raster order and then
/// `suffix` trailing tokens.
///
/// A voxel sits at its cell's centre in `[-1, 1)` along each axis, times `2π` (the reference's
/// `create_token_ids`, `vit_utils.py:9-46`, and `angle_scale`, `base_module.py:136-159`). The
/// register tokens and the zero token are not placed in space: their coordinates are zero.
///
/// # Errors
///
/// A raster with an empty axis.
pub fn token_positions(frames: u32, height: u32, width: u32, suffix: u32) -> Result<Tensor> {
    let voxels = functional::grid_positions(&[frames, height, width], std::f32::consts::TAU)?;
    let trailing = Tensor::zeros((3, suffix as usize), DType::F32, &Device::Cpu)?;

    Ok(Tensor::cat(&[voxels, trailing], 1)?)
}

/// One axis of `pixels` pixels split into the checkpoint's tiles, each tile's range in units
/// of `unit` pixels: pixels for a decoded raster, latent cells for a latent one.
///
/// The split is the reference's `split_tiles` (`klvae.py:222-250`) on the grid of the spatial
/// compression, so every tile and every seam is whole latent cells.
fn tile_ranges(pixels: usize, spatial: usize, unit: usize) -> Result<Vec<Range<usize>>> {
    let tiles = vae::split_tiles(pixels, TILE_SIZE, TILE_OVERLAP_MIN, spatial)?;

    Ok(tiles
        .into_iter()
        .map(|tile| tile.start / unit..tile.end / unit)
        .collect())
}

/// The rasters of a clip's tiles from their tokens: `tokens`, `[tiles, voxels, patch]` with a
/// tile's voxels in raster order over `[frames, height, width]` latent cells, as
/// `[tiles, channels, frames · temporal, height · spatial, width · spatial]`.
///
/// A token holds its voxel's block channel-major, `[channels, temporal, spatial, spatial]`
/// (the reference's `_unpack_tensors_3d`, `vae_vit.py:129-145`). The interleave of the three
/// block axes with the three raster axes is two permutations, each within the six axes a
/// device tensor has: the frames first, then the rows and the columns.
fn unpatchify(
    tokens: &Tensor,
    [frames, height, width]: [usize; 3],
    channels: usize,
    temporal: usize,
    spatial: usize,
) -> Result<Tensor> {
    let tiles = tokens.dims()[0];

    // [tiles, t, h·w, c, a, b·d] -> [tiles, c, t, a, h·w, b·d]
    let by_frame = tokens
        .reshape((
            tiles,
            frames,
            height * width,
            channels,
            temporal,
            spatial * spatial,
        ))?
        .permute((0, 3, 1, 4, 2, 5))?;

    // [tiles·c·t·a, h, w, b, d] -> [tiles·c·t·a, h, b, w, d]
    let by_pixel = by_frame
        .reshape((
            tiles * channels * frames * temporal,
            height,
            width,
            spatial,
            spatial,
        ))?
        .permute((0, 1, 3, 2, 4))?;

    Ok(by_pixel.reshape((
        tiles,
        channels,
        frames * temporal,
        height * spatial,
        width * spatial,
    ))?)
}

/// One transformer block of the decoder (the reference's `TransformerBlock`,
/// `base_module.py:170-289`): RMS normalization, attention with a per-head q/k normalization
/// and a 3-axis rotation, RMS normalization and a gated SiLU feed-forward, each update scaled
/// per channel before it joins the residual stream.
///
/// ```text
/// h = h + scale1 · attn(norm1(h))
/// h = h + scale2 · ff(norm2(h))
/// ```
///
/// Each scaled update is folded into the normalization that reads the sum, as the reference's
/// fused blocks fold it (`vae_vit.py:70-88`): [`Self::forward`] returns the stream with the
/// attention's update folded in and the feed-forward's update still pending, which the next
/// block's first normalization, or the decoder's final one, folds with `scale2`.
///
/// The projections are read in the element type the block's builder pins, and the rows they
/// read are normalized into it; the normalizations' weights and the scales are the stream's
/// FP32.
pub struct TransformerBlock {
    /// `norm1.weight`, `[width]`.
    norm1: Tensor,
    attn: Attention,
    /// `scale1`, `[width]`: the attention update's per-channel scale.
    scale1: Tensor,
    /// `norm2.weight`, `[width]`.
    norm2: Tensor,
    ff: Mlp,
    /// `scale2`, `[width]`: the feed-forward update's per-channel scale.
    scale2: Tensor,
    eps: f32,
}

impl TransformerBlock {
    /// The block at `vb` (`decoder.transformer_blocks.<i>`): `norm1`, `attn.{to_q, to_k, to_v,
    /// to_out.0}`, `scale1`, `norm2`, `ff.net.{0.proj, 2}` and `scale2`.
    ///
    /// The checkpoint stores `ff.net.0.proj` value half first and gate half second, and no
    /// gain for the per-head q/k normalization.
    ///
    /// # Errors
    ///
    /// A tensor the checkpoint lacks or holds at another extent, or a head of no width.
    pub fn load(config: &VideoVaeConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let (heads, head_dim) = (
            config.decoder_num_attention_heads,
            config.decoder_attention_head_dim,
        );
        let width = heads * head_dim;
        let eps = config.decoder_norm_eps;
        let stream = vb.to_dtype(STREAM);

        let attn = Attention::load(
            &AttentionConfig {
                qkv: QkvLinearConfig {
                    hidden: width,
                    heads,
                    kv_heads: heads,
                    head_dim,
                    gated: false,
                    names: QkvNames::Separate(["to_q", "to_k", "to_v"]),
                    bias: QkvBias::Present,
                    split: Split::WHOLE,
                },
                output: "to_out.0",
                output_bias: true,
                qk_norm: QkNorm::Fused {
                    gains: None,
                    eps,
                    centred: false,
                },
                scale: (head_dim as f32).sqrt().recip(),
                causal: false,
                window: None,
                // The keys a launch reduces over are the clip's rows, which each forward
                // states from its layout; the module adds no bound of its own.
                max_kv_len: u32::MAX,
            },
            vb.pp("attn"),
        )?;

        let ff = Mlp::load(
            &MlpConfig {
                hidden: width,
                intermediate: width * config.decoder_ffn_mult,
                activation: Activation::Silu,
                rounding: Rounding::ValueProduct,
                gate_up: GateUp::ValueFirst("net.0.proj"),
                down: "net.2",
                bias: true,
                split: Split::WHOLE,
            },
            vb.pp("ff"),
        )?;

        let width = width as usize;
        Ok(Self {
            norm1: stream.pp("norm1").get(width, "weight")?,
            attn,
            scale1: stream.get(width, "scale1")?,
            norm2: stream.pp("norm2").get(width, "weight")?,
            ff,
            scale2: stream.get(width, "scale2")?,
            eps,
        })
    }

    /// One block over `hidden`, the `[rows, width]` FP32 residual stream of every sequence:
    /// `(hidden, update)`, the stream with `pending` and the attention's update folded in, and
    /// the feed-forward's update, which joins the stream scaled by this block's `scale2`.
    ///
    /// `pending` is the preceding block's feed-forward update and its `scale2`, absent for the
    /// first block. `rope` is the rows' factor tables `(cos, sin)`, FP32 `[rows, pairs]`.
    /// `cu_seqlens` is the `[sequences + 1]` U32 cumulative row counts of the sequences, each
    /// attending itself, and `max_seqlen` the longest of them.
    ///
    /// # Errors
    ///
    /// A stream of another width, tables that are not one row per token, or an operator's
    /// refusal.
    pub fn forward(
        &self,
        hidden: Tensor,
        pending: Option<(&Tensor, &Tensor)>,
        rope: (&Tensor, &Tensor),
        cu_seqlens: &Tensor,
        max_seqlen: u32,
    ) -> Result<(Tensor, Tensor)> {
        let rows = u32::try_from(hidden.dims().first().copied().unwrap_or(0))
            .map_err(|_| Error::request("a stream of more rows than a launch addresses"))?;

        // The rows the projections read are normalized into the projections' own type.
        let activation = self.attn.query().weight().dtype();
        let (normalized, hidden) = match pending {
            None => (
                functional::rms_norm(&hidden, &self.norm1, self.eps, false, activation)?,
                hidden,
            ),
            Some((update, scale)) => functional::scaled_residual_rms_norm_(
                hidden,
                update,
                None,
                scale,
                &self.norm1,
                self.eps,
            )?,
        };

        let attended = self.attn.forward_varlen(
            &normalized,
            Some(rope),
            cu_seqlens,
            cu_seqlens,
            Some(max_seqlen),
            rows,
        )?;

        let (normalized, hidden) = functional::scaled_residual_rms_norm_(
            hidden,
            &attended,
            None,
            &self.scale1,
            &self.norm2,
            self.eps,
        )?;

        Ok((hidden, self.ff.forward(&normalized)?))
    }
}

/// The video decoder: one clip's latent voxels become the clip's pixels.
///
/// A decode reads a clip's latents as `[rows, latent_channels]` FP32, every tile's voxels tile
/// after tile ([`Decoder::latent_rows`]), and writes the clip's raster
/// `[out_channels, frames · temporal_compression, height · spatial_compression,
/// width · spatial_compression]` in FP16, normalized by [`PIXEL_NORM`]. The tiles, the token
/// positions and the sequence offsets are arithmetic over the layout's extents, so one decoder
/// serves every resolution.
pub struct VideoDecoder {
    /// `[latent_channels]` FP32: the latent space's per-channel mean.
    latents_mean: Tensor,
    /// `[latent_channels]` FP32: the latent space's per-channel deviation.
    latents_std: Tensor,
    /// `post_quant_conv`, a one-by-one convolution, as the projection it is; its bias joins
    /// the rounded product.
    post_quant_conv: Linear,
    /// `post_quant_conv.bias`, `[latent_channels]` FP16.
    post_quant_bias: Tensor,
    proj_in: Linear,
    /// `[registers, width]` FP32: the learned tokens that trail each sequence's voxels.
    register_tokens: Tensor,
    transformer_blocks: Vec<TransformerBlock>,
    /// `norm_out.weight` and `norm_out.bias`, `[width]` FP32 each.
    norm_out: (Tensor, Tensor),
    proj_out: Linear,
    pos_embed: Rotary,
    config: VideoVaeConfig,
    /// The path the decoder was loaded under: its profiler range's.
    path: String,
}

impl VideoDecoder {
    /// The decoder at `vb`, the root of the checkpoint's `vae` stage: `post_quant_conv` and
    /// `decoder.{proj_in, register_tokens, transformer_blocks.<i>, norm_out, proj_out}`.
    ///
    /// The projections are read in FP16 and everything else in FP32 (module documentation).
    /// The latent statistics and the rotary frequencies are constants the config states,
    /// published as `latents_mean`, `latents_std` and `decoder.pos_embed.inv_freq`.
    ///
    /// # Errors
    ///
    /// A config with no layer or with statistics of another count than its channels, or a
    /// tensor the checkpoint lacks or holds at another extent.
    pub fn load(config: &VideoVaeConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let channels = config.latent_channels;
        if config.decoder_num_layers == 0 {
            return Err(Error::config("the video decoder has no transformer block"));
        }
        if config.latents_mean.len() != channels as usize
            || config.latents_std.len() != channels as usize
        {
            return Err(Error::config(format!(
                "the video latent statistics hold {} means and {} deviations for {channels} \
                 channels",
                config.latents_mean.len(),
                config.latents_std.len()
            )));
        }

        let width = config.decoder_num_attention_heads * config.decoder_attention_head_dim;
        let patch = config.out_channels
            * config.temporal_compression()
            * config.spatial_compression()
            * config.spatial_compression();
        let linear = |inputs: u32, outputs: u32, bias: bool| LinearConfig {
            inputs,
            outputs,
            bias,
            parallel: Parallel::None,
        };

        let decoder = vb.pp("decoder");
        let projections = decoder.to_dtype(ACTIVATION);
        let stream = decoder.to_dtype(STREAM);
        let post_quant = vb.pp("post_quant_conv").to_dtype(ACTIVATION);

        let transformer_blocks = (0..config.decoder_num_layers)
            .map(|index| {
                TransformerBlock::load(config, projections.pp("transformer_blocks").pp(index))
            })
            .collect::<Result<Vec<_>>>()?;

        let norm_out = stream.pp("norm_out");
        let (channels, width) = (channels as usize, width as usize);

        Ok(Self {
            latents_mean: vb.constant("latents_mean", STREAM, channels, &config.latents_mean)?,
            latents_std: vb.constant("latents_std", STREAM, channels, &config.latents_std)?,
            post_quant_conv: Linear::load(
                &linear(config.latent_channels, config.latent_channels, false),
                post_quant.clone(),
            )?,
            post_quant_bias: post_quant.get(channels, "bias")?,
            proj_in: Linear::load(
                &linear(config.latent_channels, width as u32, true),
                projections.pp("proj_in"),
            )?,
            register_tokens: stream.get(
                (config.decoder_num_register_tokens as usize, width),
                "register_tokens",
            )?,
            transformer_blocks,
            norm_out: (norm_out.get(width, "weight")?, norm_out.get(width, "bias")?),
            proj_out: Linear::load(
                &linear(width as u32, patch, true),
                projections.pp("proj_out"),
            )?,
            pos_embed: Rotary::load(&rotary_config(config), decoder.pp("pos_embed"))?,
            config: config.clone(),
            path: decoder.path().to_owned(),
        })
    }

    /// The clip of `layout` decoded from `latents` into `video` ([`Decoder::decode`]).
    fn decode_clip(&self, layout: &Layout, latents: &Tensor, video: &Tensor) -> Result<()> {
        let config = &self.config;
        let (spatial, temporal) = (
            config.spatial_compression() as usize,
            config.temporal_compression() as usize,
        );
        let (channels, out_channels) = (
            config.latent_channels as usize,
            config.out_channels as usize,
        );
        let LatentShape {
            frames,
            height,
            width,
            ..
        } = layout.video;

        // The clip's tiles, in pixels: every tile is one extent, so one tile's positions and
        // one sequence length serve them all.
        let rows = tile_ranges(height as usize * spatial, spatial, 1)?;
        let columns = tile_ranges(width as usize * spatial, spatial, 1)?;
        let tiles = rows.len() * columns.len();
        let tile = [
            frames as usize,
            rows[0].len() / spatial,
            columns[0].len() / spatial,
        ];
        let voxels = tile.iter().product::<usize>();
        let suffix = config.decoder_num_register_tokens as usize + 1;
        let sequence = voxels + suffix;

        if latents.dims() != [tiles * voxels, channels] || latents.dtype() != STREAM {
            return Err(Error::request(format!(
                "the decoder reads a clip's latents as {STREAM:?} [{}, {channels}], {tiles} \
                 tile(s) of {voxels} voxels, not {:?} {:?}",
                tiles * voxels,
                latents.dtype(),
                latents.dims()
            )));
        }
        let raster = [
            out_channels,
            frames as usize * temporal,
            height as usize * spatial,
            width as usize * spatial,
        ];
        if video.dims() != raster || video.dtype() != ACTIVATION {
            return Err(Error::request(format!(
                "the clip's raster is {ACTIVATION:?} {raster:?}, not {:?} {:?}",
                video.dtype(),
                video.dims()
            )));
        }

        // The latents restore in FP32 and round once on their way into the FP16 projections;
        // the post-quantization product rounds before its bias joins it, as the reference's
        // convolution under autocast does.
        let restored = latents
            .broadcast_mul(&self.latents_std)?
            .broadcast_add(&self.latents_mean)?
            .to_dtype(ACTIVATION)?;
        let quantized = self
            .post_quant_conv
            .forward(&restored)?
            .broadcast_add(&self.post_quant_bias)?;
        let embedded = self.proj_in.forward(&quantized)?;

        // A tile's sequence is its voxels, the register tokens and one zero token
        // (`vae_vit.py:373-381`); joining the FP32 registers promotes the stream to FP32.
        let hidden_width = self.register_tokens.dims()[1];
        let registers = suffix - 1;
        let mut hidden = Tensor::cat(
            &[
                embedded
                    .to_dtype(STREAM)?
                    .reshape((tiles, voxels, hidden_width))?,
                self.register_tokens.unsqueeze(0)?.broadcast_as((
                    tiles,
                    registers,
                    hidden_width,
                ))?,
                Tensor::zeros((tiles, 1, hidden_width), STREAM, latents.device())?,
            ],
            1,
        )?
        .reshape((tiles * sequence, hidden_width))?;

        // Host values, whose shapes follow from the layout's extents alone: one tile's token
        // positions, and the row offsets of the tiles' sequences.
        let positions = token_positions(frames, tile[1] as u32, tile[2] as u32, suffix as u32)?
            .to_device(latents.device())?;
        let cu_seqlens = Tensor::arange(0_u32, tiles as u32 + 1, &Device::Cpu)?
            .broadcast_mul(&Tensor::new(sequence as u32, &Device::Cpu)?)?
            .to_device(latents.device())?;

        // Every tile reads the same positions: one tile's factors, repeated over the tiles.
        let (cos, sin) = self.pos_embed.forward(&positions)?;
        let pairs = cos.dims()[1];
        let over_tiles = |table: Tensor| -> Result<Tensor> {
            Ok(table
                .unsqueeze(0)?
                .broadcast_as((tiles, sequence, pairs))?
                .reshape((tiles * sequence, pairs))?)
        };
        let (cos, sin) = (over_tiles(cos)?, over_tiles(sin)?);

        // Each block folds its predecessor's scaled update into its first normalization, and
        // the final normalization folds the last block's.
        let mut pending: Option<(Tensor, &Tensor)> = None;
        for block in &self.transformer_blocks {
            let (stream, update) = block.forward(
                hidden,
                pending.as_ref().map(|(update, scale)| (update, *scale)),
                (&cos, &sin),
                &cu_seqlens,
                sequence as u32,
            )?;

            hidden = stream;
            pending = Some((update, &block.scale2));
        }
        let Some((update, scale)) = pending else {
            return Err(Error::config("the video decoder has no transformer block"));
        };
        let normalized = functional::scaled_residual_layer_norm(
            &hidden,
            &update,
            None,
            scale,
            &self.norm_out.0,
            &self.norm_out.1,
            config.decoder_norm_eps,
        )?;

        // Each voxel token's block of pixels, laid out as its tile's raster; the trailing
        // tokens of a sequence decode to nothing (`vae_vit.py:452-459`).
        let tokens = self
            .proj_out
            .forward(&normalized)?
            .reshape((tiles, sequence, ()))?
            .narrow(1, 0, voxels)?;
        let rasters = unpatchify(&tokens, tile, out_channels, temporal, spatial)?;

        // The tiles fade over their seams in FP16 and are set in place (`klvae.py:295-333`).
        let blocks = (0..tiles)
            .map(|index| rasters.get(index))
            .collect::<uniserve_tensor::Result<Vec<Tensor>>>()?;
        let stitched = vae::stitch_tiles(&blocks, &rows, &columns)?;

        Ok(video.slice_set(&stitched, 0, 0)?)
    }
}

impl Decoder for VideoDecoder {
    /// FP16: the type the reference casts its decoded clip to and blends it in.
    fn element_type(&self) -> DType {
        ACTIVATION
    }

    /// FP32: the latents stay as a denoiser publishes them until they are restored.
    fn latent_type(&self) -> DType {
        STREAM
    }

    /// One row per voxel of every tile: a tile's rectangle of every frame of the clip, tile
    /// after tile.
    fn gather_latents(
        &self,
        layout: &Layout,
        trajectory: &Tensor,
        first: u32,
        block: &Tensor,
    ) -> nn::Result<()> {
        let channels = self.config.latent_channels as usize;
        if trajectory.rank() != 2
            || trajectory.dims()[0] != channels
            || trajectory.dtype() != STREAM
            || block.dtype() != STREAM
        {
            return Err(Error::request(format!(
                "the decoder gathers {STREAM:?} latents of {channels} channels, and the \
                 trajectory is {:?} {:?} and the block {:?}",
                trajectory.dtype(),
                trajectory.dims(),
                block.dtype()
            ))
            .into());
        }

        // The clip's voxels follow the `first` latent frames before it in the trajectory.
        let Some(rows) = self.latent_rows(layout)? else {
            return Err(Error::request("the video decoder lists its tiles' voxels").into());
        };
        let frame = layout.video.height * layout.video.width;
        let voxels = rows
            .broadcast_add(&Tensor::new(first * frame, &Device::Cpu)?)?
            .to_device(trajectory.device())?;

        // A voxel's channels lie a whole timeline apart in the trajectory and are one row of
        // the block.
        let gathered = trajectory.index_select(&voxels, 1)?.t()?;
        Ok(block.slice_set(&gathered, 0, 0)?)
    }

    fn latent_rows(&self, layout: &Layout) -> nn::Result<Option<Tensor>> {
        let spatial = self.config.spatial_compression() as usize;
        let LatentShape {
            frames,
            height,
            width,
            ..
        } = layout.video;
        let (height, width) = (height as usize, width as usize);

        let rows = tile_ranges(height * spatial, spatial, spatial)?;
        let columns = tile_ranges(width * spatial, spatial, spatial)?;

        Ok(Some(vae::tile_voxels(
            &rows,
            &columns,
            frames as usize,
            height,
            width,
        )?))
    }

    /// Clip `k` decodes the latent frames from `5k`: five that advance the timeline, and two
    /// the next clip's window also reads.
    fn clip_start(&self, clip: u32) -> u32 {
        temporal_clips(&self.config).start(clip)
    }

    /// The raster's extents over the checkpoint's compressions: `temporal_compression` frames
    /// a latent frame and `spatial_compression` pixels a latent cell.
    fn layout_of(&self, clip: Clip) -> nn::Result<Layout> {
        let (spatial, temporal) = (
            self.config.spatial_compression(),
            self.config.temporal_compression(),
        );
        let whole = |extent: u32, ratio: u32| extent > 0 && extent.is_multiple_of(ratio);

        if !whole(clip.frames, temporal)
            || !whole(clip.height, spatial)
            || !whole(clip.width, spatial)
        {
            return Err(Error::request(format!(
                "a {}x{} raster of {} frame(s) is not whole latent cells of {spatial} pixels and \
                 {temporal} frames",
                clip.width, clip.height, clip.frames
            ))
            .into());
        }

        Ok(Layout {
            video: LatentShape {
                channels: self.config.latent_channels,
                frames: clip.frames / temporal,
                height: clip.height / spatial,
                width: clip.width / spatial,
            },
            audio: None,
            text_rows: 0,
            condition_rows: 0,
            condition_elements: 0,
        })
    }

    /// Decodes one clip: every tile of its raster, stitched into `video`.
    ///
    /// `latents` is the clip's `[rows, latent_channels]` FP32 block in [`Self::latent_rows`]'
    /// order and `video` the clip's FP16 raster, whose frames are the clip's whole decode:
    /// the surplus the splice crops is still in it. `audio` must be absent.
    fn decode(
        &self,
        layout: &Layout,
        latents: &Tensor,
        video: Option<&Tensor>,
        audio: Option<&Tensor>,
    ) -> nn::Result<()> {
        let (Some(video), None) = (video, audio) else {
            return Err(Error::request("a video decode writes a raster and no waveform").into());
        };

        nn::profiler::range("VideoDecoder", &self.path, || {
            Ok(self.decode_clip(layout, latents, video)?)
        })
    }
}

/// One stage of the encoder (the reference's `down[i]`, `vae_cnn.py:201-234`): its residual
/// blocks, then the convolution that halves space and strides time, when the stage has one.
struct DownBlock {
    resnets: Vec<ResnetBlock3d>,
    /// `downsamplers.0.conv`: three taps along every axis, strided, over an input padded by
    /// one row below and one column to the right when it halves space.
    downsampler: Option<Conv3d>,
    /// The rows below and the columns to the right the downsampler's input gains.
    edge: usize,
    padding: Padding,
}

impl DownBlock {
    /// The stage's output for `x`, `[n, channels, d, h, w]` in FP32.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut hidden = x.clone();
        for resnet in &self.resnets {
            hidden = resnet.forward(&hidden)?;
        }

        match &self.downsampler {
            None => Ok(hidden),
            Some(conv) => {
                let padded = functional::causal_pad(
                    &hidden,
                    [0, self.edge, 0, self.edge],
                    CAUSAL_FRONT,
                    self.padding,
                )?;

                Ok(conv.forward(&padded)?)
            }
        }
    }
}

/// The video encoder: the causal convolutional network that turns a condition's pixels into
/// the posterior moments of its latent, and the sample a condition reads of them.
pub struct VideoEncoder {
    conv_in: Conv3d,
    down_blocks: Vec<DownBlock>,
    /// The final normalization, which writes its output activated and padded for `conv_out`.
    norm_out: FrameNorm,
    conv_out: Conv3d,
    /// The one-by-one projection of the moments.
    quant_conv: Conv3d,
    /// `[latent_channels, 1, 1, 1]` FP32: the latent space's per-channel mean.
    latents_mean: Tensor,
    /// `[latent_channels, 1, 1, 1]` FP32: the latent space's per-channel deviation.
    latents_std: Tensor,
    in_channels: usize,
    latent_channels: usize,
    spatial: usize,
    temporal: usize,
    padding: Padding,
    /// The path the encoder was loaded under: its profiler range's.
    path: String,
}

/// Refuses a config whose encoder stages do not form an encoder of the family's latent space:
/// no stage, no residual block, a zero stride, or a spatial compression other than
/// [`LATENT_COMPRESSION`].
fn check_stages(config: &VideoVaeConfig) -> Result<()> {
    let stages = config.block_out_channels.len();
    if stages == 0
        || config.spatial_downsample_factors.len() != stages
        || config.temporal_downsample_factors.len() != stages
        || config.layers_per_block == 0
        || config.latent_channels == 0
    {
        return Err(Error::config(format!(
            "the video encoder has {stages} stage(s) of {} block(s) and {} latent channel(s), \
             with {} spatial and {} temporal strides",
            config.layers_per_block,
            config.latent_channels,
            config.spatial_downsample_factors.len(),
            config.temporal_downsample_factors.len()
        )));
    }

    let strides = config
        .spatial_downsample_factors
        .iter()
        .chain(&config.temporal_downsample_factors);
    if strides.clone().any(|stride| *stride == 0) {
        return Err(Error::config("a stride of the video encoder is zero"));
    }

    let compression = config
        .spatial_downsample_factors
        .iter()
        .try_fold(1_u32, |product, stride| product.checked_mul(*stride));
    if compression != Some(LATENT_COMPRESSION) {
        return Err(Error::config(format!(
            "the video encoder compresses space {compression:?}-fold, and the latent space is \
             {LATENT_COMPRESSION}-fold"
        )));
    }

    if !config.norm_eps.is_finite() || config.norm_eps <= 0.0 {
        return Err(Error::config(format!(
            "the video encoder's normalization epsilon is {}",
            config.norm_eps
        )));
    }
    Ok(())
}

impl VideoEncoder {
    /// The encoder at `vb`, the root of the checkpoint's `vae` stage: `encoder.{conv_in,
    /// down_blocks.<i>.{resnets.<j>, downsamplers.0.conv}, norm_out, conv_out}` and
    /// `quant_conv`, every tensor read in FP32.
    ///
    /// # Errors
    ///
    /// Stages that do not form an encoder of the latent space, or a tensor the checkpoint
    /// lacks or holds at another extent.
    pub fn load(config: &VideoVaeConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        check_stages(config)?;

        let vb = vb.to_dtype(DType::F32);
        let encoder = vb.pp("encoder");
        let padding = config.spatial_padding_mode;
        let moments = 2 * config.latent_channels;

        // No convolution pads with zeros of its own: a causal pad, alone or written by the
        // normalization before it, supplies the padded input. The reference keeps cuDNN's
        // TF32 for these FP32 convolutions.
        let conv = |from: u32, to: u32, kernel: u32, stride: [u32; 3]| ConvConfig {
            stride,
            tf32: true,
            ..ConvConfig::new(from, to, [kernel; 3])
        };

        let mut width = config.block_out_channels[0];
        let conv_in = Conv3d::load(
            &conv(config.in_channels, width, 3, [1; 3]),
            encoder.pp("conv_in"),
        )?;

        let mut down_blocks = Vec::with_capacity(config.block_out_channels.len());
        for (index, &out) in config.block_out_channels.iter().enumerate() {
            let stage = encoder.pp("down_blocks").pp(index);

            let resnets = (0..config.layers_per_block)
                .map(|block| {
                    ResnetBlock3d::load(
                        &ResnetBlock3dConfig {
                            in_channels: if block == 0 { width } else { out },
                            out_channels: out,
                            groups: config.norm_num_groups,
                            eps: config.norm_eps,
                            spatial: [1; 4],
                            front: CAUSAL_FRONT,
                            padding,
                            kernel: 3,
                            tf32: true,
                        },
                        stage.pp("resnets").pp(block),
                    )
                })
                .collect::<nn::Result<Vec<_>>>()?;

            // A stage whose strides are all one has no convolution (`vae_cnn.py:217-232`), and
            // only a stride of two pads the bottom and the right (`vae_cnn.py:72-75`).
            let (space, time) = (
                config.spatial_downsample_factors[index],
                config.temporal_downsample_factors[index],
            );
            let downsampler = if space * time > 1 {
                Some(Conv3d::load(
                    &conv(out, out, 3, [time, space, space]),
                    stage.pp("downsamplers").pp(0).pp("conv"),
                )?)
            } else {
                None
            };

            down_blocks.push(DownBlock {
                resnets,
                downsampler,
                edge: usize::from(space == 2),
                padding,
            });
            width = out;
        }

        let channels = config.latent_channels as usize;
        let statistic =
            |name: &str, values: &[f32]| vb.constant(name, DType::F32, (channels, 1, 1, 1), values);

        Ok(Self {
            conv_in,
            down_blocks,
            norm_out: FrameNorm::load(
                &FrameNormConfig {
                    channels: width,
                    groups: config.norm_num_groups,
                    eps: config.norm_eps,
                    affine: true,
                    spatial: [1; 4],
                    front: CAUSAL_FRONT,
                    padding,
                },
                encoder.pp("norm_out"),
            )?,
            conv_out: Conv3d::load(&conv(width, moments, 3, [1; 3]), encoder.pp("conv_out"))?,
            quant_conv: Conv3d::load(&conv(moments, moments, 1, [1; 3]), vb.pp("quant_conv"))?,
            latents_mean: statistic("latents_mean", &config.latents_mean)?,
            latents_std: statistic("latents_std", &config.latents_std)?,
            in_channels: config.in_channels as usize,
            latent_channels: channels,
            spatial: config.spatial_compression() as usize,
            temporal: config.temporal_compression() as usize,
            padding,
            path: encoder.path().to_owned(),
        })
    }

    /// The encoder network over one raster (the reference's `EncoderFCN3D.forward`,
    /// `vae_cnn.py:256-268`): `pixels`, `[n, in_channels, frames, height, width]` in FP32, to
    /// the moments `[n, 2 · latent_channels, ceil(frames / temporal_compression),
    /// height / spatial_compression, width / spatial_compression]` before `quant_conv`.
    ///
    /// # Errors
    ///
    /// A raster the convolutions or the normalizations refuse.
    pub fn forward(&self, pixels: &Tensor) -> Result<Tensor> {
        // The input convolution reads padded pixels; it has no normalization of its own.
        let padded = functional::causal_pad(pixels, [1; 4], CAUSAL_FRONT, self.padding)?;

        let mut hidden = self.conv_in.forward(&padded)?;
        for block in &self.down_blocks {
            hidden = block.forward(&hidden)?;
        }

        Ok(self.conv_out.forward(&self.norm_out.forward(&hidden)?)?)
    }

    /// A condition's normalized latent: `pixels`, `[in_channels, frames, height, width]` in
    /// FP32 normalized by [`PIXEL_NORM`] (one frame for a still), to
    /// `[latent_channels, latent_frames, height / 16, width / 16]` in FP32.
    ///
    /// Each clip of the frame rule ([`FRAMES`]) is encoded alone, in the checkpoint's tiles,
    /// whose moments are stitched ([`nn::vae::tiled_encode`]). `quant_conv` then projects the
    /// stitched moments of the whole condition once, the posterior is sampled under
    /// [`POSTERIOR_SEED`] as `mean + exp(0.5 · clamp(logvar, −30, 20)) · noise` rounded
    /// through FP16 (the reference's `DiagonalGaussianDistribution.sample` and
    /// `use_fp16_latent`, `klvae.py:71-92`), and the sample is normalized per channel,
    /// `(sample − mean) / std`.
    ///
    /// # Errors
    ///
    /// Pixels of another rank, element type or channel count, a raster that is not whole
    /// latent cells, a frame count whose clips the frame rule does not encode, or what a
    /// module refuses.
    pub fn encode(&self, pixels: &Tensor) -> Result<Tensor> {
        nn::profiler::range("VideoEncoder", &self.path, || self.encode_condition(pixels))
    }

    /// [`Self::encode`] inside its profiler range.
    fn encode_condition(&self, pixels: &Tensor) -> Result<Tensor> {
        let (spatial, channels) = (self.spatial, self.latent_channels);
        let &[_, frames, height, width] = pixels.dims() else {
            return Err(Error::request(format!(
                "a condition's pixels are [channels, frames, height, width], not {:?}",
                pixels.dims()
            )));
        };
        if pixels.dims()[0] != self.in_channels
            || pixels.dtype() != DType::F32
            || frames == 0
            || !height.is_multiple_of(spatial)
            || !width.is_multiple_of(spatial)
        {
            return Err(Error::request(format!(
                "the video encoder reads FP32 pixels of {} channels over whole {spatial}-pixel \
                 latent cells, not {:?} {:?}",
                self.in_channels,
                pixels.dtype(),
                pixels.dims()
            )));
        }

        // The tiles of the raster, in latent cells.
        let rows = tile_ranges(height, spatial, spatial)?;
        let columns = tile_ranges(width, spatial, spatial)?;

        let mut clips = Vec::new();
        for span in FRAMES.clips(frames as u32) {
            // A clip's frames encode to the latent frames the rule keeps of it only for the
            // frame counts the rule generates, and for a still.
            let encoded = (span.frames as usize).div_ceil(self.temporal);
            if encoded != span.latents as usize {
                return Err(Error::request(format!(
                    "a condition of {frames} frames ends in a clip of {} that encodes to \
                     {encoded} latent frames, of which the frame rule keeps {}",
                    span.frames, span.latents
                )));
            }
            let clip = pixels.narrow(1, span.first as usize, span.frames as usize)?;

            let moments = vae::tiled_encode(
                &rows,
                &columns,
                &[2 * channels, encoded],
                DType::F32,
                |index| {
                    let (row, column) = (
                        &rows[index / columns.len()],
                        &columns[index % columns.len()],
                    );
                    let tile = clip
                        .narrow(2, row.start * spatial, row.len() * spatial)?
                        .narrow(3, column.start * spatial, column.len() * spatial)?
                        .unsqueeze(0)?;

                    Ok(self.forward(&tile)?.squeeze(0)?)
                },
            )?;
            clips.push(moments);
        }

        // The posterior of the whole condition: its mean and its clamped log-variance.
        let moments = self
            .quant_conv
            .forward(&Tensor::cat(&clips, 1)?.unsqueeze(0)?)?
            .squeeze(0)?;
        let mean = moments.narrow(0, 0, channels)?;
        let std = (moments.narrow(0, channels, channels)?.clamp(-30.0, 20.0)? * 0.5)?.exp()?;

        // The sample, rounded through FP16 as a step of its own, then the latent space's
        // normalization in FP32.
        let noise = functional::philox_normal_like(&mean, POSTERIOR_SEED, 0, 0, 0)?;
        let sample = ((noise * std)? + mean)?
            .to_dtype(DType::F16)?
            .to_dtype(DType::F32)?;

        Ok(sample
            .broadcast_sub(&self.latents_mean)?
            .broadcast_div(&self.latents_std)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::tests_support;

    fn config() -> VideoVaeConfig {
        VideoVaeConfig::parse(&tests_support::video_vae()).expect("the released sidecar")
    }

    /// The elements of a CPU tensor, widened to FP32, in row-major order.
    fn values(tensor: &Tensor) -> Vec<f32> {
        tensor
            .to_dtype(DType::F32)
            .and_then(|tensor| tensor.flatten_all())
            .and_then(|tensor| tensor.to_vec1())
            .expect("a CPU tensor")
    }

    #[test]
    fn a_clip_decodes_seven_latent_frames_into_twenty_five() {
        let config = config();
        let clips = temporal_clips(&config);

        // The decode emits 28 frames and the splice keeps 25 of them: 25 = 17 + 3 + 5.
        assert_eq!(clips.window(), 7);
        assert_eq!(clips.native(), 25);
        assert_eq!(
            clips.window() * config.temporal_compression() - clips.native(),
            3
        );
        assert_eq!(clips.start(2), 10);
    }

    #[test]
    fn a_timeline_is_spliced_one_clip_per_unit_of_the_frame_rule() {
        let clips = temporal_clips(&config());

        // 22 frames are one clip, which keeps its tail; 39 are two.
        let one = splices(&clips, 22).expect("one clip");
        assert_eq!(one.len(), 1);
        assert_eq!((one[0].keep, one[0].overlap, one[0].first), (22, 0, 0));

        let two = splices(&clips, 39).expect("two clips");
        assert_eq!(two.len(), 2);
        assert_eq!((two[0].keep, two[0].overlap, two[0].first), (17, 0, 0));
        assert_eq!((two[1].keep, two[1].overlap, two[1].first), (22, 5, 17));

        assert!(splices(&clips, 30).is_err(), "30 is not 17n + 5");
    }

    #[test]
    fn three_quarters_of_a_head_rotate_over_three_axes() {
        let rotary = rotary_config(&config());
        assert_eq!(rotary.dim, 48);
        assert_eq!(rotary.spectrum, Spectrum::Repeated { axes: 3 });

        // One spectrum of sixteen channels, eight pairs, repeated per axis.
        let frequencies = rotary.inverse_frequencies().expect("a spectrum");
        assert_eq!(frequencies.len(), 24);
        assert_eq!(frequencies[..8], frequencies[8..16]);
        assert_eq!(frequencies[0], 1.0);
    }

    #[test]
    fn the_token_positions_are_the_cell_centres_of_the_raster_and_zeros() {
        let table = token_positions(2, 2, 3, 5).expect("positions");
        assert_eq!(table.dims(), [3, 2 * 2 * 3 + 5]);
        let table = values(&table);
        let rows = 17;

        // Frame and row axes of two: -0.5, 0.5. Column axis of three: -2/3, 0, 2/3. Times one
        // turn, one leading row per axis, the voxels in frame, row, column order.
        let centres = [
            [-0.5, 0.5, 0.0],
            [-0.5, 0.5, 0.0],
            [-2.0 / 3.0, 0.0, 2.0 / 3.0],
        ];
        for voxel in 0..12 {
            let cell = [voxel / 6, voxel / 3 % 2, voxel % 3];

            for axis in 0..3 {
                let expected = std::f64::consts::TAU * centres[axis][cell[axis]];
                let actual = f64::from(table[axis * rows + voxel]);
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "voxel {voxel} axis {axis}: {actual} against {expected}"
                );
            }
        }

        // The register tokens and the zero token are not placed.
        for axis in 0..3 {
            assert!(
                table[axis * rows + 12..(axis + 1) * rows]
                    .iter()
                    .all(|x| *x == 0.0)
            );
        }
    }

    #[test]
    fn a_raster_splits_into_uniform_tiles_whose_seams_take_the_slack() {
        // 768 by 1344 pixels are 48 by 84 latent cells: four tile rows and seven columns of
        // sixteen cells, with seams of 6, 5, 5 cells down and 5, 5, 5, 5, 4, 4 across.
        let rows = tile_ranges(768, 16, 16).expect("rows");
        let columns = tile_ranges(1344, 16, 16).expect("columns");
        assert_eq!((rows.len(), columns.len()), (4, 7));
        assert!(rows.iter().chain(&columns).all(|tile| tile.len() == 16));

        let seams = |tiles: &[Range<usize>]| -> Vec<usize> {
            tiles
                .windows(2)
                .map(|pair| pair[0].end - pair[1].start)
                .collect()
        };
        assert_eq!(seams(&rows), [6, 5, 5]);
        assert_eq!(seams(&columns), [5, 5, 5, 5, 4, 4]);
        assert_eq!((rows[3].end, columns[6].end), (48, 84));

        // A raster inside one tile is one tile with no seam, in pixels and in cells alike.
        let pixels = tile_ranges(256, 16, 1).expect("one tile");
        assert_eq!((pixels.len(), pixels[0].clone()), (1, 0..256));
        let cells = tile_ranges(192, 16, 16).expect("one tile");
        assert_eq!((cells.len(), cells[0].clone()), (1, 0..12));

        // A raster that is not whole latent cells is refused.
        assert!(tile_ranges(264, 16, 16).is_err());
    }

    #[test]
    fn a_token_is_its_voxels_block_of_pixels() {
        // Two tiles of [frames 2, height 2, width 3] voxels, two channels, a block of two
        // frames by two by two pixels. Each element holds its own flat index, so the raster
        // names the token element every pixel came from.
        let (tiles, [frames, height, width]) = (2, [2, 2, 3]);
        let (channels, temporal, spatial) = (2, 2, 2);
        let patch = channels * temporal * spatial * spatial;
        let voxels = frames * height * width;
        let count = tiles * voxels * patch;

        let tokens = Tensor::arange(0_f32, count as f32, &Device::Cpu)
            .and_then(|tokens| tokens.reshape((tiles, voxels, patch)))
            .expect("tokens");
        let rasters = unpatchify(
            &tokens,
            [frames, height, width],
            channels,
            temporal,
            spatial,
        )
        .expect("rasters");
        assert_eq!(
            rasters.dims(),
            [
                tiles,
                channels,
                frames * temporal,
                height * spatial,
                width * spatial
            ]
        );
        let rasters = values(&rasters);

        let mut pixel = 0;
        for tile in 0..tiles {
            for channel in 0..channels {
                for frame in 0..frames * temporal {
                    for row in 0..height * spatial {
                        for column in 0..width * spatial {
                            // The voxel the pixel lies in, and its place in the voxel's block.
                            let voxel = (frame / temporal * height + row / spatial) * width
                                + column / spatial;
                            let element = ((channel * temporal + frame % temporal) * spatial
                                + row % spatial)
                                * spatial
                                + column % spatial;
                            let expected = (tile * voxels + voxel) * patch + element;

                            assert_eq!(rasters[pixel], expected as f32, "pixel {pixel}");
                            pixel += 1;
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn stages_that_are_not_an_encoder_of_the_latent_space_are_refused() {
        assert!(check_stages(&config()).is_ok());

        let mut no_stage = config();
        no_stage.block_out_channels.clear();
        assert!(check_stages(&no_stage).is_err());

        let mut short = config();
        short.spatial_downsample_factors.pop();
        assert!(check_stages(&short).is_err());

        let mut zero = config();
        zero.temporal_downsample_factors[1] = 0;
        assert!(check_stages(&zero).is_err());

        // Eight-fold is another latent space than the one a request is laid out in.
        let mut coarse = config();
        coarse.spatial_downsample_factors[3] = 1;
        assert!(check_stages(&coarse).is_err());
    }
}
