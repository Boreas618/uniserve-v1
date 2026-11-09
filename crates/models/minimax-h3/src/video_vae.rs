//! The H3 video VAE decoder: its raster and tile arithmetic, and the numerical model that decodes one
//! clip of latents into RGB frames.
//!
//! The decoder is a non-causal ViT: one
//! latent voxel becomes one token, four learned register tokens and one zero token trail the voxels,
//! and each token expands into a patch of RGB. `post_quant_conv` projects the latent channels, 36
//! transformer layers attend over the whole token sequence, and the output projection writes
//! `out_channels · temporal_compression · spatial_compression²` elements per token, which the
//! unpacking step lays out as a planar clip of frames.
//!
//! ## What this module owns
//!
//! * the arithmetic that turns a raster into token counts ([`Canvas`]) and a canvas into its
//!   overlapping decode tiles ([`grid`], over `nn::tiling::Grid`, the reference's `split_tiles`),
//!   so a tile's extent and every seam's overlap are host values a captured bucket is planned from;
//!   and the temporal clips a timeline is decoded in ([`clips`]);
//! * the per-layer composition and the element type each layer declares (`torch.autocast` is
//!   replaced by a declaration, not by an implicit conversion).
//!
//! ## Precision
//!
//! The reference runs this decoder under `torch.autocast(dtype=FP16)` with the checkpoint's weights
//! converted per its precision preset, and what autocast does to each operation is the table below.
//! Every row is stated in code here, and no conversion happens that a row does not name. `dtype =
//! "auto"` still decides everything this table does not: the parameters it lists as stored stay in
//! the checkpoint's own representation, which is FP32. This is the model's own declaration, so it
//! takes no preset and no switch.
//!
//! ```text
//! latents, the restored latents        FP32        (`latents.float() * std + mean`)
//! post_quant_conv                      FP16 weights; the restored latents round to FP16 on the way in,
//!                                      and the product rounds again before its bias joins it
//! the input projection                 FP16 weights and output
//! register tokens, norms, scales       stored (FP32)
//! the residual stream                  FP32        (`cat` of an FP16 projection with FP32 registers
//!                                                   promotes; every sum after it is FP32)
//! the normalized rows                  FP16        (FP32 statistics of the unrounded sum, one rounding)
//! rotary factors                       FP16        (FP32 trigonometry, `.to(compute_dtype)`)
//! qkv, output, gate/up, down           FP16 weights, FP16 bias in the epilogue, FP16 output
//! per-head norm and rotation           FP16 rows, FP32 arithmetic, one rounding
//! attention                            FP16 operands, FP32 softmax and accumulators (FlashAttention 4)
//! the gated activation                 FP16 rows, FP32 arithmetic, one rounding
//! the output projection                FP16; the raster and the seam fades stay FP16
//! ```
//!
//! So the residual is the one wide tensor: a projection's output joins it in FP32, and everything a
//! projection reads is a rounding of it.
//!
//! ## One launch per layer for the whole clip
//!
//! The reference decodes a clip's tiles as one batch, and so does this decoder: every tile's rows sit
//! in one stream, so each projection is one product over all of them and the attention is one launch
//! over as many packed sequences as there are tiles. A tile is a sequence of its own, so nothing
//! crosses between tiles until the stitch.

use crate::{Error, Result, VideoVaeConfig};
use nn::tiling::{Grid, TemporalClips, TiledDecode};

/// Frames of a decoded clip's tail, which the next clip fades its first body frames against.
///
/// The checkpoint's decode constant (the reference's `frame_overlap`); its config does not state it.
pub const OVERLAP_FRAMES: u32 = 5;

/// The temporal clips the checkpoint's decoder decodes a timeline in.
///
/// A clip keeps `clip_length` frames, skips the `token_drop` padding frames the final clip drops,
/// and ends in an [`OVERLAP_FRAMES`] tail: 25 native frames, so a clip reads `ceil(25 / 4) = 7`
/// latent frames from `5k` ([`crate::CLIP_LATENTS`] per clip) and its decode's leading surplus is
/// cropped.
pub fn clips(config: &VideoVaeConfig) -> TemporalClips {
    TemporalClips {
        body: config.clip_length,
        padding: config.token_drop,
        overlap: OVERLAP_FRAMES,
        temporal: config.temporal_compression(),
        advance: crate::CLIP_LATENTS,
    }
}

/// The latent raster of one decode call, and the token count it produces.
///
/// One clip decodes `latent_frames` latent frames of `height` × `width` latent voxels. Every voxel is
/// one token; [`Canvas::rows`] adds the register tokens and the one zero token the reference appends.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Canvas {
    /// Latent frames decoded in this call (7 for a clip of the H3 checkpoint).
    pub frames: u32,
    /// Latent rows.
    pub height: u32,
    /// Latent columns.
    pub width: u32,
    /// Learned register tokens trailing the voxels.
    pub registers: u32,
    /// Zero tokens trailing the register tokens (one, as the checkpoint's layout states).
    pub tail: u32,
}

impl Canvas {
    /// The canvas of one clip of `config`: the [`clips`] window of latent frames at the given latent
    /// extents.
    pub fn clip(config: &VideoVaeConfig, height: u32, width: u32) -> Self {
        Self {
            frames: clips(config).window(),
            height,
            width,
            registers: config.decoder_num_register_tokens,
            tail: 1,
        }
    }

    /// The latent voxels: the token rows before the trailing ones.
    pub fn voxels(&self) -> u64 {
        u64::from(self.frames) * u64::from(self.height) * u64::from(self.width)
    }

    /// Every token the transformer's sequence holds.
    pub fn rows(&self) -> u64 {
        self.voxels() + u64::from(self.registers) + u64::from(self.tail)
    }

    /// The clip's decoded frame count: latent frames times the temporal compression.
    pub fn frames_out(&self, config: &VideoVaeConfig) -> u32 {
        self.frames * config.temporal_compression()
    }

    /// The clip's decoded pixel extents.
    pub fn pixels(&self, config: &VideoVaeConfig) -> (u32, u32) {
        let ratio = config.spatial_compression();
        (self.height * ratio, self.width * ratio)
    }
}

/// The decode tile grid of `canvas`, in output pixels: the checkpoint's [`TILE_PIXELS`] tiles with
/// at least [`MINIMUM_OVERLAP`] shared, aligned to the spatial compression.
///
/// # Errors
///
/// A canvas the split refuses.
pub fn grid(config: &VideoVaeConfig, canvas: &Canvas) -> Result<Grid> {
    let (height, width) = canvas.pixels(config);
    Ok(Grid::split(
        height,
        width,
        TILE_PIXELS,
        MINIMUM_OVERLAP,
        config.spatial_compression(),
    )?)
}

/// The token canvas of one decode tile of `canvas`: a split keeps every tile one extent, so this is
/// every tile's, and the position table every tile reads is its.
///
/// # Errors
///
/// A canvas the split refuses.
pub fn tile(config: &VideoVaeConfig, canvas: &Canvas) -> Result<Canvas> {
    let (height, width) = grid(config, canvas)?.tile(config.spatial_compression());
    Ok(Canvas {
        height,
        width,
        ..*canvas
    })
}

/// The width of a decoder head the rotary embedding rotates, and its three shares.
///
/// A head is `decoder_attention_head_dim` wide and `decoder_rope_dim_ratio` of it is rotated — 64 and
/// 0.75 in the checkpoint, so 48 elements — split evenly over the frame, row and column axes: sixteen
/// each, eight rotation pairs per axis. The trailing elements of a head pass through unrotated. The
/// base is `decoder_rope_theta` (100), stated by the layer that builds the factors.
pub fn rotary(config: &VideoVaeConfig) -> Result<(u32, [u32; 3])> {
    let head = config.decoder_attention_head_dim;
    let ratio = config.decoder_rope_dim_ratio;
    if head == 0 || !(0.0..=1.0).contains(&ratio) || ratio == 0.0 {
        return Err(Error::config(format!(
            "a {head}-wide head and a rotated fraction of {ratio} do not make a rotation"
        )));
    }
    let width = (head as f64 * f64::from(ratio)).round();
    if (width - head as f64 * f64::from(ratio)).abs() > 1e-6 || width < 1.0 {
        return Err(Error::config(format!(
            "a {ratio} fraction of a {head}-wide head is not a whole width"
        )));
    }
    let width = width as u32;
    if !width.is_multiple_of(6) || !width.is_multiple_of(2) {
        return Err(Error::config(format!(
            "a rotated width of {width} does not split evenly over three axes of pairs"
        )));
    }
    let share = width / 3;
    Ok((width, [share; 3]))
}

/// The cell-centre rotary coordinates of the transformer's token rows, as `[3, rows]` FP32.
///
/// Along an axis of `size`, row `i` sits at `2 * (i + 0.5) / size - 1`, in `[-1, 1)`: the reference's
/// cell-centre grid. The rows are the canvas's latent voxels in
/// raster order — frame, then row, then column — followed by the register tokens and the trailing zero
/// token, whose coordinates are all zero because they are not placed in space.
///
/// The table is `[3, rows]` F32, axis-major, because the rotation reads one leading coordinate per
/// axis (the decoder's repeated spectrum): a row-major `[rows, 3]` table would be read transposed and
/// silently rotate every row by another row's angles. Coordinates are in radians — the reference's
/// cell-centre grid times `2π` — so the raw inverse frequencies give the reference's angle.
///
/// The table is a per-layout constant: the runner uploads it once per captured graph
/// (`video_positions`).
pub fn positions(canvas: &Canvas) -> Result<Vec<f32>> {
    if canvas.frames == 0 || canvas.height == 0 || canvas.width == 0 {
        return Err(Error::config(
            "a decoder canvas of no frame, row or column has no position".to_owned(),
        ));
    }
    let rows = canvas.rows() as usize;
    let mut table = vec![0.0f32; 3 * rows];
    let mut row = 0;
    for frame in 0..canvas.frames {
        for height in 0..canvas.height {
            for column in 0..canvas.width {
                let coordinates = [
                    axis(frame, canvas.frames),
                    axis(height, canvas.height),
                    axis(column, canvas.width),
                ];
                for (channel, value) in coordinates.iter().enumerate() {
                    table[channel * rows + row] = *value;
                }
                row += 1;
            }
        }
    }
    Ok(table)
}

/// One cell-centre coordinate of an axis of `size`, in radians.
///
/// The reference builds cell-centre coordinates in `[-1, 1)` and multiplies them by `2π` at its
/// rotary call; the frequency table here is the raw inverse frequencies, so the angle comes out as
/// `2π · position · frequency` with the `2π` folded into the table the runner uploads.
fn axis(index: u32, size: u32) -> f32 {
    // The reference's own arithmetic: the cell centre and the `2π` turn are both float32 (its
    // meshgrid is float32 and its caller multiplies by `2π` in the same type), so one rounding
    // stands between the reference and this table rather than two.
    let centre = (index as f32 + 0.5) / size as f32;
    std::f32::consts::TAU * (2.0 * centre - 1.0)
}

/// The tile extent of the checkpoint's decoder, in output pixels.
pub const TILE_PIXELS: u32 = 256;

/// The minimum overlap between two adjacent tiles, in output pixels.
pub const MINIMUM_OVERLAP: u32 = 64;

// ---------------------------------------------------------------------------------------------
// The video decoder's transformer.
// ---------------------------------------------------------------------------------------------

use nn::Forward;
use nn::blocks::{GateUp, GatedMlpSpec, PreNormBlock, PreNormSpec, ResidualForm};
use nn::family::Split;
use nn::layers::{
    Bias, Cast, ChannelAffine, DenseAttention, DenseSpec, Epilogue, IndexRows, LayerNorm, Linear,
    MergedColumnLinear, Parallel, Precision, QkNorm, ResidualStream, Rotary, RotarySpec, Rounding,
    Spectrum, Varlen,
};
use nn::weights::{Weight, WeightLoader};
use nn::{DType, DecodedMedia, Decoder, Shape, StateTable, TableValues, TensorView};
use ops::gemm::WeightOrder;

/// The element type of the decoder's projections and of the rows they read and write, and the type
/// its rasters are written in: the model's own declaration of the reference's `quality` row (see the
/// module's table).
pub const ACTIVATION: DType = DType::F16;

/// The element type of the latents a decode reads and of the residual stream, which the reference
/// keeps wider than the rows its projections read.
pub const STREAM: DType = DType::F32;

/// One transformer layer of the decoder: a pre-norm block with layer scales over the decoder's
/// attention.
///
/// The residual updates are folded into the normalization that reads them, as the reference's
/// `_residual_norm` does ([`ResidualForm::Fused`]): the attention's output joins the stream and
/// normalizes for the feed-forward in one launch, and the feed-forward's output joins it and
/// normalizes for the next layer, or for the final norm.
struct Layer {
    /// The query, key and value projections as one `[3 · width, width]` product: one read of the
    /// normalized rows, and the three outputs are column ranges of one buffer, which the rotation
    /// turns in place through its row stride.
    query_key_value: MergedColumnLinear,
    output: Linear,
    /// The norms, the layer scales and the value-first SwiGLU.
    block: PreNormBlock,
}

/// The video decoder's latent stage: one clip's latent voxels become the clip's pixels.
///
/// The decoder reads a clip's latents `[frames, height, width]` of [`Canvas`] (seven latent frames of
/// 24 channels for the checkpoint), runs the 36-layer transformer over every tile's voxel rows plus
/// their learned register rows and zero row, and writes one 3×4×16×16 pixel block per voxel. The
/// per-head normalization before the rotation has no learned weight — the checkpoint stores none,
/// which is what `Rotary::apply`'s absent gains express.
pub struct VideoDecoder {
    /// Every raster this decoder serves: the deployment's canvases, in the order it named them.
    canvases: Vec<Canvas>,
    /// The tiled decode's stitch, built at load so that no layer is built on a request's path.
    stitch: TiledDecode,
    width: u32,
    /// `value · std + mean` per latent channel, a product then a sum: the
    /// reference restores the latents in FP32 before anything rounds them.
    restore: ChannelAffine,
    post_quant: Linear,
    /// `post_quant_conv`'s bias, as a product with the identity: the reference's convolution rounds
    /// its product before it adds the bias, and this is that second rounding.
    post_quant_bias: Linear,
    input: Linear,
    registers: Weight,
    register_rows: Weight,
    zero_row: Weight,
    zero_rows: Weight,
    layers: Vec<Layer>,
    /// The final normalization's gain and bias, which fold the last layer's scaled update.
    norm_out: (Weight, Weight),
    output: Linear,
    rotary: Rotary,
    /// The offsets of the clip's sequences: `[0, rows, 2 · rows, ...]`, one per tile of the largest
    /// canvas this decoder serves. A tile's rows are one sequence; a clip of fewer tiles reads the
    /// table's prefix.
    offsets: Weight,
    config: VideoVaeConfig,
    /// Elements per head.
    head: u64,
    /// Query heads per token.
    heads: u64,
    attention: DenseAttention,
}

/// The two tables a layout's decode reads, keyed by its latent extents.
///
/// The model's arithmetic, keyed by the **layout**: the rotary positions of the largest tile of the
/// latent raster, which every other tile's positions are a slice of because the tiles are uniform, and
/// the stitch's two axis tables, rows then columns, which say where each pixel of the raster comes
/// from (`decoding::stitch_table`).
pub(crate) fn table_specs(
    config: &VideoVaeConfig,
    latent_height: u32,
    latent_width: u32,
) -> Result<Vec<StateTable>> {
    let positions = tile(config, &Canvas::clip(config, latent_height, latent_width))?.rows();
    let ratio = u64::from(config.spatial_compression());
    let pixels = (u64::from(latent_height) + u64::from(latent_width)) * ratio;
    Ok(vec![
        StateTable {
            name: "video_positions",
            dtype: DType::F32,
            shape: Shape::new(&[3, positions]),
        },
        StateTable {
            name: "video_stitch",
            dtype: DType::I32,
            shape: Shape::new(&[pixels, ops::tile_stitch::TABLE_WIDTH]),
        },
    ])
}

/// Write the host values of [`table_specs`], in its order, each checked against its table.
pub(crate) fn table_values_into(
    config: &VideoVaeConfig,
    latent_height: u32,
    latent_width: u32,
    out: &mut [TableValues],
) -> Result<()> {
    let tables = table_specs(config, latent_height, latent_width)?;
    if out.len() != tables.len() {
        return Err(Error::config(format!(
            "{} tables are read and {} buffers given",
            tables.len(),
            out.len()
        )));
    }
    let clip = Canvas::clip(config, latent_height, latent_width);
    let positions = TableValues::F32(positions(&tile(config, &clip)?)?);
    let (height, width) = clip.pixels(config);
    let stitch = TableValues::I32(
        grid(config, &clip)?
            .stitch_table(height, width)
            .into_iter()
            .flatten()
            .collect(),
    );
    for ((held, table), values) in out.iter_mut().zip(&tables).zip([positions, stitch]) {
        held.write(table.name, &values)?;
    }
    Ok(())
}

/// The attention of the decoder's block: FlashAttention 4 over one packed sequence per tile, which is
/// the kernel the reference runs at this declaration.
fn block_attention(head: u32, heads: u32, rows: u32) -> Result<DenseAttention> {
    let scale = (head as f32).sqrt().recip();
    Ok(DenseAttention::new(DenseSpec {
        head_dim: head,
        heads,
        kv_heads: heads,
        max_keys: rows,
        causal: false,
        window: None,
        scale,
    })?)
}

impl VideoDecoder {
    /// Read the 36 layers and the head of the checkpoint's decoder.
    /// Load one decoder for **every raster the deployment serves**.
    ///
    /// `rasters` are the deployment's canvases in pixels, which is where a canvas comes from: the
    /// checkpoint states its tile and its compression, the deployment states which rasters it
    /// serves. The constants a decode reads are a **tile's** or a **width's**, so the tables are
    /// published once and every canvas of the set is a lookup; only the sequence offsets and the
    /// attention's row bound follow the largest canvas.
    ///
    /// The projections are read at [`ACTIVATION`] and everything else at the checkpoint's own type:
    /// the module's table, stated at the only place a weight's type is decided.
    pub fn load(
        config: &VideoVaeConfig,
        rasters: &[(u32, u32)],
        loader: &mut WeightLoader,
    ) -> Result<Self> {
        // The deployment's canvases, in its own order. A raster is whole latent cells or it is not a
        // canvas this decoder can address.
        let spatial = config.spatial_compression();
        let mut canvases: Vec<Canvas> = Vec::with_capacity(rasters.len());
        for &(pixel_width, pixel_height) in rasters {
            if pixel_width == 0 || pixel_height == 0 {
                return Err(Error::config(
                    "a deployment canvas of no pixels decodes nothing".to_owned(),
                ));
            }
            if pixel_width % spatial != 0 || pixel_height % spatial != 0 {
                return Err(Error::config(format!(
                    "the deployment's canvas is {pixel_width}x{pixel_height} pixels and this \
                     checkpoint's decoder compresses {spatial} pixels into one latent cell, so the \
                     raster is not a whole number of cells"
                )));
            }
            let canvas = Canvas::clip(config, pixel_height / spatial, pixel_width / spatial);
            if !canvases.contains(&canvas) {
                canvases.push(canvas);
            }
        }
        let canvas = *canvases.first().ok_or_else(|| {
            Error::config("a video decoder serves at least one raster".to_owned())
        })?;
        // The tables a decode reads are a **tile's**, not a canvas's: one table serves every tile of
        // every canvas only where every canvas is the same tiles. A deployment whose rasters tile
        // differently would need a table per raster, and is refused by both named.
        let tile = tile(config, &canvas)?;
        for other in &canvases[1..] {
            if self::tile(config, other)? != tile {
                return Err(Error::config(format!(
                    "the deployment serves a {}x{} raster and a {}x{} one whose largest tiles \
                     differ: the tables a decode reads are one tile's, so every canvas must be a \
                     whole number of the same tiles",
                    canvas.height, canvas.width, other.height, other.width
                )));
            }
        }
        // The most tiles any served canvas splits into, which the sequence offsets and the
        // attention's bound answer to.
        let mut most_tiles = 0_u64;
        for other in &canvases {
            most_tiles = most_tiles.max(grid(config, other)?.len() as u64);
        }
        let width = config.decoder_num_attention_heads * config.decoder_attention_head_dim;
        let head = config.decoder_attention_head_dim;
        let heads = config.decoder_num_attention_heads;
        let ffn = width * config.decoder_ffn_mult;
        let stitch = TiledDecode::new(
            ACTIVATION,
            config.out_channels,
            config.temporal_compression(),
            config.spatial_compression(),
        )?;
        let out_channels = config.out_channels
            * config.temporal_compression()
            * config.spatial_compression()
            * config.spatial_compression();
        let epsilon = config.decoder_norm_eps;
        let channels = u64::from(config.latent_channels);
        // The constants the export does not hold - the latent restore, the identity the
        // post-quantization bias rides on, the zero row the token layout appends, the two index
        // tables that address a tile's trailing rows, the rotary frequencies and the sequence
        // offsets - are the decoder's own, named under `video_vae`.
        let (restore_scale, restore_shift) = restore_tables(config)?;
        let restore = {
            let mut constants = loader.scope("video_vae.restore");
            // The latents are `[voxels, channels]`: the channel is the last axis.
            ChannelAffine::new(
                constants.constant("scale", STREAM, &[channels], &restore_scale)?,
                constants.constant("shift", STREAM, &[channels], &restore_shift)?,
                1,
            )?
        };
        // The post-quantization convolution is 1x1x1: its `[channels, channels, 1, 1, 1]` weight is
        // the matrix of a projection, whose bias the next product adds (see `post_quant_bias`).
        let mut post_quant_conv = loader.scope("post_quant_conv");
        let mut post_quant_conv = post_quant_conv.pinned(ACTIVATION);
        let post_quant = Linear::load(
            &mut post_quant_conv,
            width_of(channels)?,
            width_of(channels)?,
            Bias::Absent,
            Parallel::None,
        )?;
        let post_quant_bias = post_quant_conv.tensor("bias", &[channels])?;
        let identity: Vec<f32> = (0..channels * channels)
            .map(|index| {
                if index / channels == index % channels {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let post_quant_bias = Linear::new(
            loader.scope("video_vae").constant(
                "identity",
                ACTIVATION,
                &[channels, channels],
                &identity,
            )?,
            Some(post_quant_bias),
            WeightOrder::NxK,
        )?;
        let input = Linear::load(
            &mut loader.scope("decoder.proj_in").pinned(ACTIVATION),
            width_of(channels)?,
            width,
            Bias::Present,
            Parallel::None,
        )?;

        let registers = loader.scope("decoder").pinned(STREAM).tensor(
            "register_tokens",
            &[u64::from(canvas.registers), u64::from(width)],
        )?;
        if canvas.tail == 0 {
            return Err(Error::config(
                "the decoder canvas has the trailing zero row".to_owned(),
            ));
        }
        let mut constants = loader.scope("video_vae");
        let zero_row = constants.zeros("zero_row", STREAM, &[1, u64::from(width)])?;
        // **A tile's rows, not the clip's.** These two tables say where a tile's register tokens and
        // its trailing zero row go, and the row write launches against that tile's own slice of the
        // stream, so one tile's offsets serve every tile and every canvas of a uniform grid.
        let first_row = |rows: u64| {
            i32::try_from(rows).map_err(|_| Error::config(format!("a tile row {rows} above I32")))
        };
        let register_rows = constants.indices(
            "register_rows",
            first_row(tile.voxels())?,
            u64::from(canvas.registers),
        )?;
        let zero_rows = constants.indices(
            "zero_rows",
            first_row(tile.voxels() + u64::from(canvas.registers))?,
            u64::from(canvas.tail),
        )?;

        // The normalizations' gains and the residual scales stay at the stream's width.
        let mut norm_out = loader.scope("decoder.norm_out");
        let mut norm_out = norm_out.pinned(STREAM);
        let norm_out = (
            norm_out.tensor("weight", &[u64::from(width)])?,
            norm_out.tensor("bias", &[u64::from(width)])?,
        );
        let count = config.decoder_num_layers;
        let mut layers = Vec::with_capacity(count as usize);
        for index in 0..count {
            let base = format!("decoder.transformer_blocks.{index}");
            // The projections are read at the activation type, the gains and the layer scales at
            // the stream's, and every update is scaled per channel and folded into the next norm.
            let block = PreNormBlock::load(
                &mut loader.scope(&base).pinned(ACTIVATION),
                PreNormSpec {
                    hidden: width,
                    eps: epsilon,
                    norm1: "norm1",
                    norm2: "norm2",
                    layer_scale: Some(["scale1", "scale2"]),
                    norm_dtype: Some(STREAM),
                    residual: ResidualForm::Fused,
                    // The feed-forward's projection is packed: its rows hold the value half and
                    // then the gate half.
                    mlp: GatedMlpSpec {
                        hidden: width,
                        intermediate: ffn,
                        activation: ops::activation::Activation::Silu,
                        rounding: Rounding::ValueProduct,
                        gate_up: GateUp::ValueFirst("ff.net.0.proj"),
                        down: "ff.net.2",
                        bias: Bias::Present,
                    },
                },
                Split::WHOLE,
            )?;
            // The three projections are one product: their matrices stacked query, key, value.
            let query_key_value = MergedColumnLinear::load(
                &mut loader.scope(format!("{base}.attn")).pinned(ACTIVATION),
                &["to_q", "to_k", "to_v"],
                width,
                &[width, width, width],
                Bias::Present,
            )?;
            let output = Linear::load(
                &mut loader
                    .scope(format!("{base}.attn.to_out.0"))
                    .pinned(ACTIVATION),
                width,
                width,
                Bias::Present,
                Parallel::None,
            )?;
            layers.push(Layer {
                query_key_value,
                output,
                block,
            });
        }

        let output = Linear::load(
            &mut loader.scope("decoder.proj_out").pinned(ACTIVATION),
            width,
            out_channels,
            Bias::Present,
            Parallel::None,
        )?;
        // The checkpoint stores no per-head gain, so the fused norm before the rotation has none.
        let rotary = Rotary::load(
            &mut loader.scope("video_vae.rotary_emb"),
            spectrum(config)?,
            RotarySpec {
                head_dim: head,
                heads,
                kv_heads: heads,
                qk_norm: QkNorm::Fused { epsilon },
            },
        )?;
        let rows = tile.rows();
        // `[0, rows, ...]`, one offset per tile of the widest canvas and the end.
        let offsets = (0..=most_tiles)
            .map(|sequence| {
                i32::try_from(sequence * rows)
                    .map_err(|_| Error::config("a clip's rows exceed a sequence offset".to_owned()))
            })
            .collect::<Result<Vec<i32>>>()?;
        let offsets = loader.scope("video_vae").constant_i32(
            "sequence_offsets",
            &[most_tiles + 1],
            &offsets,
        )?;
        Ok(Self {
            canvases,
            stitch,
            width,
            restore,
            post_quant,
            post_quant_bias,
            input,
            registers,
            register_rows,
            zero_row,
            zero_rows,
            layers,
            norm_out,
            output,
            rotary,
            offsets,
            // One block for the whole set: its bound is the largest served clip's rows, and a clip of
            // a smaller canvas reads fewer.
            attention: block_attention(
                head,
                heads,
                u32::try_from(most_tiles * rows)
                    .map_err(|_| Error::config("too many tokens".to_owned()))?,
            )?,
            config: config.clone(),
            head: u64::from(head),
            heads: u64::from(heads),
        })
    }
}

/// A channel count as a projection's width.
fn width_of(channels: u64) -> Result<u32> {
    u32::try_from(channels)
        .map_err(|_| Error::config(format!("{channels} channels exceed a projection's width")))
}

/// The restore's scale and shift: the latent deviations and the means, one per channel.
fn restore_tables(config: &VideoVaeConfig) -> Result<(Vec<f32>, Vec<f32>)> {
    let channels = config.latents_std.len();
    if channels == 0 || config.latents_mean.len() != channels {
        return Err(Error::config(format!(
            "the decoder's latent statistics hold {} means and {channels} deviations",
            config.latents_mean.len()
        )));
    }
    Ok((config.latents_std.clone(), config.latents_mean.clone()))
}

/// The decoder's rotary spectrum: one axis's frequencies repeated over the three axes, in f64.
///
/// The reference builds its embedding with `head_dim · ratio / 3` and reuses those frequencies on
/// all three axes, so axis `a` takes `theta^(-2k / sections[a])` rather than a slice of one wider
/// spectrum; the decoder publishes it as `video_vae.rotary_emb.inv_freq`.
pub fn spectrum(config: &VideoVaeConfig) -> Result<Spectrum> {
    let (_, sections) = rotary(config)?;
    Ok(Spectrum::Repeated {
        theta: config.decoder_rope_theta,
        pairs: sections[0] / 2,
        axes: sections.len() as u32,
        precision: Precision::F64,
    })
}

impl VideoDecoder {
    /// The tables one layout's decode reads, in the order the table walk writes them.
    pub fn tables(&self, latent_height: u32, latent_width: u32) -> Result<Vec<StateTable>> {
        table_specs(&self.config, latent_height, latent_width)
    }

    /// Write the host values of each table of [`Self::tables`], in the same order.
    pub fn table_values(
        &self,
        latent_height: u32,
        latent_width: u32,
        out: &mut [TableValues],
    ) -> Result<()> {
        table_values_into(&self.config, latent_height, latent_width, out)
    }

    /// The rasters this decoder serves: the deployment's canvases, every one of them.
    ///
    /// The constants a decode reads are a **tile's** (its register and zero-row offsets, its sequence
    /// offsets) or a **width's** (the zero row, the rotary frequencies), so they are the same for every
    /// canvas this deployment names: serving another raster costs a lookup, not a table.
    pub fn canvases(&self) -> &[Canvas] {
        &self.canvases
    }

    /// The transformer's scratch for one clip: every tile's rows at once, because every tile runs in
    /// the same launches.
    pub fn scratch(&self, f: &Forward, canvas: &Canvas) -> Result<ClipScratch> {
        let tile = tile(&self.config, canvas)?;
        let tiles = grid(&self.config, canvas)?.len() as u64;
        let rows = tiles * tile.rows();
        let voxels = tiles * tile.voxels();
        let width = u64::from(self.width);
        let half = u64::from(rotary(&self.config)?.0 / 2);
        let channels = u64::from(self.config.latent_channels);
        let arena = f.arena();
        Ok(ClipScratch {
            stream: arena.tensor(STREAM, Shape::new(&[rows, width]))?,
            restored: arena.tensor(STREAM, Shape::new(&[voxels, channels]))?,
            rounded: arena.tensor(ACTIVATION, Shape::new(&[voxels, channels]))?,
            quantized: arena.tensor(ACTIVATION, Shape::new(&[voxels, channels]))?,
            projected: arena.tensor(ACTIVATION, Shape::new(&[voxels, width]))?,
            cosine: arena.tensor(ACTIVATION, Shape::new(&[tile.rows(), half]))?,
            sine: arena.tensor(ACTIVATION, Shape::new(&[tile.rows(), half]))?,
            normalized: arena.tensor(ACTIVATION, Shape::new(&[rows, width]))?,
            query_key_value: arena.tensor(ACTIVATION, Shape::new(&[rows, 3 * width]))?,
            attended: arena.tensor(ACTIVATION, Shape::new(&[rows, width]))?,
            update: arena.tensor(ACTIVATION, Shape::new(&[rows, width]))?,
        })
    }

    /// Decode one clip's latent rows into the pixel block of every token.
    ///
    /// `latents` is `[voxels, latent_channels]` FP32 — every tile's voxels, tile after tile, in the
    /// raster order the caller's clip rows select — `positions` is one tile's `[3, rows]` F32,
    /// axis-major, the per-layout table the table walk declares, and `out` is
    /// `[tiles · rows, out_channels · 4 · 16 · 16]` at [`ACTIVATION`]: one token's 3×4×16×16 pixel
    /// block, tile after tile, which the caller scatters into the clip's raster. A tile's rows are its
    /// voxels, then the register rows, then the trailing zero row.
    pub fn decode(
        &self,
        f: &Forward,
        canvas: &Canvas,
        latents: TensorView,
        positions: TensorView,
        out: TensorView,
        scratch: &ClipScratch,
    ) -> Result<()> {
        let tile = tile(&self.config, canvas)?;
        let tiles = grid(&self.config, canvas)?.len() as u64;
        let (tile_rows, tile_voxels) = (tile.rows(), tile.voxels());
        let (rows, voxels) = (tiles * tile_rows, tiles * tile_voxels);
        let width = u64::from(self.width);
        if latents.shape.rank() != 2 || latents.shape.dim(0) != voxels {
            return Err(Error::config(format!(
                "the decoder reads a clip's {voxels} latent rows, {tiles} tiles of {tile_voxels}, \
                 not {:?}",
                latents.shape.dims()
            )));
        }
        if out.shape.rank() != 2 || out.shape.dim(0) != rows {
            return Err(Error::config(format!(
                "the decoder writes {rows} token rows for this clip's {voxels} voxels, not {:?}",
                out.shape.dims()
            )));
        }
        // The rotation reads one leading coordinate per axis; a row-major table would be read
        // transposed and rotate every row by another row's angles, so the shape is checked here
        // rather than trusted.
        if positions.shape.rank() != 2
            || positions.shape.dim(0) != 3
            || positions.shape.dim(1) != tile_rows
        {
            return Err(Error::config(format!(
                "the decoder reads a [3, {tile_rows}] position table, not {:?}",
                positions.shape.dims()
            )));
        }

        // The latents restore in FP32 and round once on their way into the FP16 projections.
        self.restore.forward(f, latents, scratch.restored)?;
        Cast.forward(f, scratch.restored, scratch.rounded)?;
        self.post_quant
            .forward(f, scratch.rounded, scratch.quantized)?;
        self.post_quant_bias
            .forward(f, scratch.quantized, scratch.rounded)?;
        self.input.forward(f, scratch.rounded, scratch.projected)?;

        // A tile's sequence is its voxels in the order they arrive, then the register rows, then the
        // zero row (`cat((hidden, registers, zeros))`), so a
        // voxel's token row is its own index within the tile. The voxels widen into the stream here:
        // that `cat` is what promotes the stream to FP32.
        let stream = scratch.stream;
        for index in 0..tiles {
            let tile_stream = TensorView::new(
                stream.offset_elements(index * tile_rows * width),
                stream.dtype,
                Shape::new(&[tile_rows, width]),
            );
            let voxel_rows = TensorView::new(
                stream.offset_elements(index * tile_rows * width),
                stream.dtype,
                Shape::new(&[tile_voxels, width]),
            );
            let projected = TensorView::new(
                scratch
                    .projected
                    .offset_elements(index * tile_voxels * width),
                scratch.projected.dtype,
                Shape::new(&[tile_voxels, width]),
            );
            Cast.forward(f, projected, voxel_rows)?;
            IndexRows.scatter(
                f,
                self.registers.view(),
                self.register_rows.view(),
                tile_stream,
            )?;
            IndexRows.scatter(f, self.zero_row.view(), self.zero_rows.view(), tile_stream)?;
        }

        let (cosine, sine) = (scratch.cosine, scratch.sine);
        self.rotary.factors(f, positions, cosine, sine)?;

        let (head, heads) = (self.head, self.heads);
        let normalized = scratch.normalized;
        let (attended, update) = (scratch.attended, scratch.update);
        let per_head = |view: TensorView| view.reshape(Shape::new(&[rows, heads, head]));
        // The three column ranges of the merged projection, turned and read in place through its row
        // stride, which is the reference's arrangement: the keys and the values share one stride.
        let merged = scratch.query_key_value;
        let projection = |index: u64| {
            TensorView::with_strides(
                merged.offset_elements(index * width),
                merged.dtype,
                Shape::new(&[rows, heads, head]),
                nn::Strides::new(&[3 * width, head, 1]),
            )
        };
        // One offset per tile and the end; a clip of fewer tiles than the largest reads a prefix.
        let offsets = TensorView::new(
            self.offsets.view().offset_elements(0),
            DType::I32,
            Shape::new(&[tiles + 1]),
        );
        let sequence = u32::try_from(tile_rows)
            .map_err(|_| Error::config("too many tokens for one sequence".to_owned()))?;
        let bound = u32::try_from(rows).map_err(|_| Error::config("too many tokens".to_owned()))?;

        // The first layer's first norm has no update to fold: it narrows the FP32 stream into the
        // FP16 rows the projections read. Every later norm folds the update before it.
        let mut pending = None;
        for layer in &self.layers {
            let attend = |normalized: TensorView, update: TensorView| -> nn::Result<()> {
                layer.query_key_value.forward(f, normalized, merged)?;
                self.rotary
                    .apply_strided(f, projection(0), projection(1), cosine, sine)?;
                self.attention.forward(
                    f,
                    projection(0),
                    projection(1),
                    projection(2),
                    Some(Varlen {
                        cu_q: offsets,
                        cu_k: offsets,
                        max_q: Some(sequence),
                        max_k: bound,
                    }),
                    per_head(attended),
                )?;
                layer.output.forward(f, attended, update)
            };
            pending = Some(
                layer
                    .block
                    .forward(f, stream, pending, normalized, update, attend)?,
            );
        }

        // The final norm folds the last layer's scaled update into the stream.
        let last = pending.ok_or_else(|| Error::config("a decoder of no layer".to_owned()))?;
        let scale = last
            .scale
            .ok_or_else(|| Error::config("the decoder's layers scale their updates".to_owned()))?;
        let (weight, bias) = self.norm_out;
        LayerNorm::new(
            weight,
            Some(bias),
            self.config.decoder_norm_eps,
            Epilogue::ScaledResidual { scale, bias: None },
        )?
        .forward(
            f,
            last.update,
            Some(ResidualStream::in_place(stream)),
            normalized,
        )?;
        Ok(self.output.forward(f, normalized, out)?)
    }
}

/// The transformer's scratch for one clip, reserved once and read by every layer.
#[derive(Clone, Copy)]
pub struct ClipScratch {
    stream: TensorView,
    restored: TensorView,
    rounded: TensorView,
    quantized: TensorView,
    projected: TensorView,
    cosine: TensorView,
    sine: TensorView,
    normalized: TensorView,
    query_key_value: TensorView,
    attended: TensorView,
    update: TensorView,
}

impl Decoder for VideoDecoder {
    /// The voxels of every tile, in the order `decode_clip` reads them: each tile's own latent slice,
    /// frame by frame, so a voxel in an overlap has one row per tile that holds it.
    fn latent_rows(&self, layout: &nn::capability::Layout) -> nn::Result<Option<Vec<u32>>> {
        let clip = Canvas::clip(&self.config, layout.video.height, layout.video.width);
        let tiles = grid(&self.config, &clip)?.tiles(self.config.spatial_compression());
        Ok(Some(nn::tiling::voxels(
            &tiles,
            layout.video.frames,
            layout.video.height,
            layout.video.width,
        )))
    }

    fn layout_tables(&self, layout: &nn::capability::Layout) -> nn::Result<Vec<StateTable>> {
        // A layout whose extents contradict the checkpoint is a named error and not an empty list: a
        // runner that made a mistake hears which one, where an empty answer would have reached the
        // decode as a block with no tables and read nothing.
        Ok(self.tables(layout.video.height, layout.video.width)?)
    }

    fn prepare_layout(
        &self,
        layout: &nn::capability::Layout,
        out: &mut [TableValues],
    ) -> nn::Result<()> {
        Ok(self.table_values(layout.video.height, layout.video.width, out)?)
    }

    /// The checkpoint's clip schedule for a timeline of `frames` frames ([`clips`]), which a runner
    /// that bound only a decoder reads as well as the postprocessor does.
    fn splices(&self, frames: u32) -> nn::Result<Vec<nn::capability::Splice>> {
        clips(&self.config).splices(frames)
    }

    /// Clip `k` decodes latent frames `[5k, 5k + 7)`: five that advance the timeline, and two the
    /// next clip's window also reads.
    fn clip_start(&self, clip: u32) -> u32 {
        clips(&self.config).start(clip)
    }

    /// The raster's element type, which the module's table declares: the reference casts its decoded
    /// clip to FP16 and blends it there.
    fn element_type(&self) -> nn::DType {
        ACTIVATION
    }

    /// The latents stay FP32 until they are restored, as a denoiser publishes them.
    fn latent_type(&self) -> nn::DType {
        STREAM
    }

    /// One strided copy per tile: a tile's voxels are a rectangle of every frame of the window, and a
    /// row of the block holds a voxel's channels, which lie a whole timeline apart in the trajectory.
    fn gather_latents(
        &self,
        f: &Forward,
        layout: &nn::capability::Layout,
        trajectory: TensorView,
        first: u32,
        block: TensorView,
    ) -> nn::Result<()> {
        // The trajectory a denoiser publishes is FP32 and this decoder restores it in FP32.
        if trajectory.dtype != STREAM || block.dtype != STREAM {
            return Err(Error::config(format!(
                "the decoder gathers {STREAM:?} latents, and the trajectory is {:?} and the block {:?}",
                trajectory.dtype, block.dtype
            ))
            .into());
        }
        let clip = Canvas::clip(&self.config, layout.video.height, layout.video.width);
        let tiles = grid(&self.config, &clip)?.tiles(self.config.spatial_compression());
        nn::tiling::gather(
            f,
            &tiles,
            trajectory,
            self.config.latent_channels,
            (layout.video.height, layout.video.width),
            first,
            clip.frames,
            block,
        )
    }

    /// The latent layout one clip's raster decodes from.
    ///
    /// The argument is the **decoded raster** - frames, rows and columns of the media this block
    /// writes - and the return is the **latent** layout its tables and arena are keyed by. The
    /// ratios between them are this deployment's own, read at construction from the checkpoint's
    /// config (`temporal_compression` frames per latent frame, `spatial_compression()` pixels per
    /// latent cell), which is why the map is the decoder's: a runtime that divided by either would
    /// be holding this checkpoint's compression in its own code.
    ///
    /// **The decoder is built for one canvas** (`VideoDecoder::load` takes it), because its register
    /// rows and its sequence offsets are the canvas's constants rather than a
    /// request's. So this states that canvas's layout and refuses any other clip by both extents:
    /// a runner that handed it another canvas would capture a graph whose constants belong to a
    /// different raster.
    fn layout_of(&self, clip: nn::capability::Clip) -> nn::Result<nn::capability::Layout> {
        let refuse = |problem: String| -> nn::Error { Error::config(problem).into() };
        // `pixels` answers `(height, width)` in that order, and this destructures it in the same
        // order: the two were once written the other way round, which a square canvas cannot show.
        // The pair below is compared in one order and the refusal names both rasters in one order,
        // so a transposed clip cannot read as agreement.
        // The raster includes the padding frames the splice drops, so the clip is compared against
        // the canvas's own native output rather than against a muxed frame count.
        let canvas = self
            .canvases
            .iter()
            .copied()
            .find(|canvas| {
                canvas.pixels(&self.config) == (clip.height, clip.width)
                    && canvas.frames_out(&self.config) == clip.frames
            })
            .ok_or_else(|| {
                refuse(format!(
                    "this decoder serves {} and the call states a {}x{} raster of {} frame(s)",
                    self.canvases
                        .iter()
                        .map(|canvas| {
                            let (height, width) = canvas.pixels(&self.config);
                            format!(
                                "{width}x{height} of {} frame(s)",
                                canvas.frames_out(&self.config)
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(", "),
                    clip.width,
                    clip.height,
                    clip.frames
                ))
            })?;
        Ok(nn::capability::Layout {
            video: nn::capability::LatentShape {
                channels: self.config.latent_channels,
                frames: canvas.frames,
                height: canvas.height,
                width: canvas.width,
            },
            audio: None,
            text_rows: 0,
            condition_rows: 0,
            condition_elements: 0,
        })
    }

    /// Decode one clip: every tile of its canvas, stitched into `out.video`.
    ///
    /// `latents` is the clip's `[voxels, latent_channels]` FP32 rows in tile order — the runner gathers
    /// them through [`Self::latent_rows`] — `state` holds this block's two tables, `video_positions`
    /// `[3, rows]` axis-major in radians and `video_stitch`, and `out.video` is the clip's raster. The
    /// raster is `[channels, frames, rows, columns]` in the decoder's own element type: the stitch writes
    /// and the byte step reads a planar channels-first raster, and `frames` is the clip's **native**
    /// output, `temporal` frames per latent frame, so it still holds the three padding frames the splice
    /// drops. `out.audio` must be absent.
    ///
    /// The grid is arithmetic on the raster's extent, so one block serves every canvas of this
    /// checkpoint; the clips are the runner's loop, and so is the splice.
    ///
    /// **A caller owes this call one clip at a time.** The transformer scratch is *reserved once per
    /// clip* ([`Self::scratch`]), so two decodes in flight on one decoder object
    /// write the same buffers of the same arena - the second clip corrupts the first silently rather
    /// than costing another column. A warmup of one clip beside a served clip of another is the same
    /// defect, and it looks identical from outside because the arena is the same size either way.
    fn decode(
        &self,
        f: &Forward,
        latents: TensorView,
        state: &[TensorView],
        out: DecodedMedia,
    ) -> nn::Result<()> {
        let refuse = |problem: String| -> nn::Error { Error::config(problem).into() };
        let [positions, stitch] = state else {
            return Err(refuse(format!(
                "the video decoder reads two tables, video_positions and video_stitch, and {} were \
                 given",
                state.len()
            )));
        };
        let video = out
            .video
            .ok_or_else(|| refuse("a video decode writes a raster".to_owned()))?;
        if out.audio.is_some() {
            return Err(refuse("the video decoder writes no audio".to_owned()));
        }
        let spatial = self.config.spatial_compression();
        let temporal = self.config.temporal_compression();
        let channels = u64::from(crate::output::RGB_CHANNELS);
        if video.shape.rank() != 4 || video.shape.dim(0) != channels {
            return Err(refuse(format!(
                "the clip's raster is [{channels}, frames, rows, columns], not {:?}",
                video.shape.dims()
            )));
        }
        let (frames, height, width) = (video.shape.dim(1), video.shape.dim(2), video.shape.dim(3));
        if frames % u64::from(temporal) != 0
            || height % u64::from(spatial) != 0
            || width % u64::from(spatial) != 0
        {
            return Err(refuse(format!(
                "a {frames}x{height}x{width} raster is not a whole number of {temporal}-frame, \
                 {spatial}-pixel latent cells"
            )));
        }
        let clip = Canvas::clip(
            &self.config,
            u32::try_from(height / u64::from(spatial)).unwrap_or(u32::MAX),
            u32::try_from(width / u64::from(spatial)).unwrap_or(u32::MAX),
        );
        let latent_frames = u32::try_from(frames / u64::from(temporal)).unwrap_or(u32::MAX);
        if latent_frames != clip.frames {
            return Err(refuse(format!(
                "a clip decodes {} latent frames to {frames} raster frames, and this one is {latent_frames}",
                clip.frames
            )));
        }
        // The trajectory a denoiser publishes is FP32 and this decoder restores it in FP32, so nothing
        // converts it on the way in. Reading the bytes in another type would be a silent
        // reinterpretation of the trajectory, so the mismatch is refused here by name and both types.
        if latents.dtype != STREAM {
            return Err(refuse(format!(
                "the decoder reads its latents as {STREAM:?} (`latent_type`), and this trajectory is \
                 {:?}",
                latents.dtype
            )));
        }
        // **One canvas set per decoder object**: the constants a decode reads are the served
        // canvases' tile's, so a canvas outside the set is refused by name.
        if !self.canvases.contains(&clip) {
            return Err(refuse(format!(
                "a {}x{} latent clip of {} frame(s) is not a canvas this decoder serves",
                clip.height, clip.width, clip.frames
            )));
        }
        let tile = tile(&self.config, &clip)?;
        // The transformer runs over every tile at once into the tokens the stitch reads; its scratch
        // is reserved after the tokens, inside the decode.
        self.stitch.forward(
            f,
            &grid(&self.config, &clip)?,
            tile.rows(),
            |tokens| {
                let scratch = self.scratch(f, &clip)?;
                Ok(self.decode(f, &clip, latents, *positions, tokens, &scratch)?)
            },
            *stitch,
            video,
        )
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A tile's sequence is its voxels in the order they arrive, then the trailing rows
    /// (`cat((hidden, registers, zeros))`), so a voxel's token row is its own index
    /// within the tile: the position table and the stitch read the same row. The projected voxels
    /// therefore widen straight into the stream's leading rows, tile by tile, and only the register
    /// rows and the zero row are placed. Scattering the voxels through the denoiser's patch rows sent
    /// the four voxels of a `2 x 2` patch to one token and left three quarters of the sequence unwritten,
    /// which decoded as the right scene in a mosaic.
    #[test]
    fn the_projected_voxels_fill_each_tiles_leading_rows() {
        struct Launches(Vec<&'static str>);
        impl ops::capture::Sink for Launches {
            fn launched(&mut self, operator: &'static str, _key: &str) {
                self.0.push(operator);
            }
        }
        let config = video_config();
        // Two tiles across and one down: the written rows repeat per tile.
        let canvas = Canvas::clip(&config, 16, 28);
        let tiles = 2_u64;
        let tile = tile(&config, &canvas).expect("a tile");
        let decoder = stated_decoder(&config, canvas);
        let arena = nn::Arena::new(nn::DevicePtr::new(0), 1 << 40);
        let forward = nn::Forward::recording(&arena);
        let scratch = decoder.scratch(&forward, &canvas).expect("the scratch");
        let latents = arena
            .tensor(
                STREAM,
                Shape::new(&[tiles * tile.voxels(), u64::from(config.latent_channels)]),
            )
            .expect("the latents");
        let positions = arena
            .tensor(DType::F32, Shape::new(&[3, tile.rows()]))
            .expect("the positions");
        let out = arena
            .tensor(
                ACTIVATION,
                Shape::new(&[tiles * tile.rows(), 3 * 4 * 16 * 16]),
            )
            .expect("the tokens");
        let mut launches = Launches(Vec::new());
        ops::capture::set_sink(Some(&mut launches));
        let decoded = decoder.decode(&forward, &canvas, latents, positions, out, &scratch);
        ops::capture::set_sink(None);
        decoded.expect("a recorded clip");
        assert_eq!(
            launches.0[..11],
            [
                "channel_affine", // the restore
                "scaled_add",     // the round down
                "gemm",           // post_quant_conv's product
                "gemm",           // its bias, a rounding of its own
                "gemm",           // the input projection
                "scaled_add",
                "rows",
                "rows", // the first tile's voxels, registers and zero row
                "scaled_add",
                "rows",
                "rows", // the second tile's
            ],
        );
        assert_eq!(
            launches.0.iter().filter(|name| **name == "rows").count(),
            2 * tiles as usize,
            "nothing else is placed"
        );
    }

    /// A denoiser publishes its trajectory in FP32 and this decoder restores it in FP32. Reading the
    /// bytes in another type would reinterpret the trajectory silently, so the mismatch is refused and
    /// both types are named.
    #[test]
    fn a_trajectory_of_another_type_is_refused_and_both_types_are_named() {
        let config = video_config();
        let canvas = Canvas::clip(&config, 48, 48);
        let decoder = stated_decoder(&config, canvas);
        let clip = clip_of(&config, canvas);
        let arena = nn::Arena::counting();
        let forward = nn::Forward::recording(&arena);
        let latents = arena
            .tensor(
                DType::F16,
                Shape::new(&[canvas.voxels(), u64::from(config.latent_channels)]),
            )
            .expect("a latent column");
        let video = arena
            .tensor(
                ACTIVATION,
                Shape::new(&[
                    3,
                    u64::from(clip.frames),
                    u64::from(clip.height),
                    u64::from(clip.width),
                ]),
            )
            .expect("a raster");
        let positions = arena
            .tensor(DType::F32, Shape::new(&[3, 1]))
            .expect("a table");
        // The capability's `decode`, not the inherent tile decoder of the same name.
        let error = nn::capability::Decoder::decode(
            &decoder,
            &forward,
            latents,
            &[positions, positions],
            nn::capability::DecodedMedia {
                video: Some(video),
                audio: None,
            },
        )
        .expect_err("the decoder reads its own element type");
        assert!(
            // The tree spells element types the way `DType` prints them, which is lowercase
            // (`f16`, `f32`): the assertion compares against that spelling rather than a
            // capitalized one the message never has, which fails for a reason that is not the
            // refusal's subject.
            error.to_string().contains(&format!("{:?}", DType::F16))
                && error.to_string().contains(&format!("{:?}", DType::F32)),
            "the refusal names both types: {error}"
        );
    }

    /// Write one shard into `directory` whose header states `tensors` — name, element type, extents
    /// — and whose payload is never written, and open it.
    ///
    /// The payload is a hole the file's length covers (`set_len`), so a checkpoint of a full-size
    /// decoder costs a header's bytes on disk. Only a meta loader may load from it: the header is
    /// the real format's, and the payload is zeros nobody wrote.
    pub(crate) fn header_checkpoint(
        directory: &std::path::Path,
        tensors: &[(String, DType, Vec<u64>)],
    ) -> nn::Checkpoint {
        let mut header = String::from("{");
        let mut offset = 0_u64;
        for (index, (name, dtype, shape)) in tensors.iter().enumerate() {
            if index > 0 {
                header.push(',');
            }
            let bytes = shape.iter().product::<u64>() * dtype.bytes().expect("a whole-byte type");
            let extents = shape
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(",");
            let format = match dtype {
                DType::BF16 => "BF16",
                DType::F16 => "F16",
                DType::F32 => "F32",
                other => panic!("the fixture writes float tensors, not {other}"),
            };
            header.push_str(&format!(
                r#""{name}":{{"dtype":"{format}","shape":[{extents}],"data_offsets":[{offset},{}]}}"#,
                offset + bytes
            ));
            offset += bytes;
        }
        header.push('}');
        // The payload starts eight-byte aligned, as every exporter writes it.
        let padded = header.len().div_ceil(8) * 8;
        header.push_str(&" ".repeat(padded - header.len()));

        let path = directory.join("model.safetensors");
        let mut file = std::fs::File::create(&path).expect("a shard");
        std::io::Write::write_all(&mut file, &(header.len() as u64).to_le_bytes())
            .expect("the header length");
        std::io::Write::write_all(&mut file, header.as_bytes()).expect("the header");
        file.set_len(8 + header.len() as u64 + offset)
            .expect("the payload's extent");
        nn::Checkpoint::open(directory).expect("the shard opens")
    }

    /// A decoder of one canvas, loaded by a meta loader over a checkpoint that states the decoder's
    /// tensors in BF16: the geometry is the subject, so no payload and no device are needed.
    fn stated_decoder(config: &VideoVaeConfig, canvas: Canvas) -> VideoDecoder {
        let channels = u64::from(config.latent_channels);
        let width =
            u64::from(config.decoder_num_attention_heads * config.decoder_attention_head_dim);
        let ffn = width * u64::from(config.decoder_ffn_mult);
        let out = u64::from(
            config.out_channels
                * config.temporal_compression()
                * config.spatial_compression()
                * config.spatial_compression(),
        );
        // The checkpoint's tensors, as the export names them.
        let mut tensors: Vec<(String, Vec<u64>)> = vec![
            (
                "post_quant_conv.weight".to_owned(),
                vec![channels, channels],
            ),
            ("post_quant_conv.bias".to_owned(), vec![channels]),
            ("decoder.proj_in.weight".to_owned(), vec![width, channels]),
            ("decoder.proj_in.bias".to_owned(), vec![width]),
            (
                "decoder.register_tokens".to_owned(),
                vec![u64::from(canvas.registers), width],
            ),
            ("decoder.norm_out.weight".to_owned(), vec![width]),
            ("decoder.norm_out.bias".to_owned(), vec![width]),
            ("decoder.proj_out.weight".to_owned(), vec![out, width]),
            ("decoder.proj_out.bias".to_owned(), vec![out]),
        ];
        for index in 0..config.decoder_num_layers {
            let at = |name: &str| format!("decoder.transformer_blocks.{index}.{name}");
            for (name, shape) in [
                ("norm1.weight", vec![width]),
                ("norm2.weight", vec![width]),
                ("scale1", vec![width]),
                ("scale2", vec![width]),
                ("attn.to_q.weight", vec![width, width]),
                ("attn.to_q.bias", vec![width]),
                ("attn.to_k.weight", vec![width, width]),
                ("attn.to_k.bias", vec![width]),
                ("attn.to_v.weight", vec![width, width]),
                ("attn.to_v.bias", vec![width]),
                ("attn.to_out.0.weight", vec![width, width]),
                ("attn.to_out.0.bias", vec![width]),
                ("ff.net.0.proj.weight", vec![2 * ffn, width]),
                ("ff.net.0.proj.bias", vec![2 * ffn]),
                ("ff.net.2.weight", vec![width, ffn]),
                ("ff.net.2.bias", vec![width]),
            ] {
                tensors.push((at(name), shape));
            }
        }
        let tensors: Vec<(String, DType, Vec<u64>)> = tensors
            .into_iter()
            .map(|(name, shape)| (name, DType::BF16, shape))
            .collect();
        let directory = tempfile::tempdir().expect("a temp directory");
        let checkpoint = header_checkpoint(directory.path(), &tensors);
        let mut loader = WeightLoader::meta(&checkpoint, None, None);
        let spatial = config.spatial_compression();
        let raster = (canvas.width * spatial, canvas.height * spatial);
        VideoDecoder::load(config, &[raster], &mut loader).expect("the decoder")
    }

    /// The clip a canvas decodes: the raster's pixels and its native frame count.
    fn clip_of(config: &VideoVaeConfig, canvas: Canvas) -> nn::capability::Clip {
        // `pixels` answers `(height, width)` - the order `layout_of` destructures it in - so this
        // helper must not read it as `(width, height)`. It did, and on a rectangle that made the
        // decoder's own canvas arrive transposed: `layout_of` refused it correctly and the two tests
        // built on this helper were the ones that failed. A square canvas hides a transposition here,
        // which is why `a_rectangular_canvas_refuses_its_own_transpose` exists.
        let (height, width) = canvas.pixels(config);
        nn::capability::Clip {
            frames: canvas.frames_out(config),
            height,
            width,
        }
    }

    /// The decoder is one canvas, and it refuses a clip of another by both extents.
    ///
    /// `VideoDecoder::load` binds a canvas because its registers, its row counts and its seam table
    /// belong to one raster, and `layout_of` is where a runner learns that: a decode of another clip
    /// would capture a graph whose constants belong to a different raster. Both refusals are asserted
    /// by what they name, because a runner reads them to know which side was wrong, and a refusal that
    /// fired for another reason would leave the mismatch to the first launch.
    /// A rectangular canvas is the case a transposition hides in: a square extent compares equal to
    /// itself whichever order the two halves are written in, so only a rectangle can see the bug that
    /// `pixels` answers `(height, width)` while its caller read `(width, height)`.
    #[test]
    fn a_rectangular_canvas_refuses_its_own_transpose() {
        let config = video_config();
        let ratio = config.spatial_compression();
        let canvas = Canvas::clip(&config, 2 * ratio, 3 * ratio);
        let decoder = stated_decoder(&config, canvas);

        let clip = clip_of(&config, canvas);
        let layout = decoder.layout_of(clip).expect("its own rectangular canvas");
        assert_eq!(
            (layout.video.height, layout.video.width),
            (canvas.height, canvas.width),
            "the layout is the canvas's own latent extents, height first"
        );

        // The transpose: the same area and the same frame count with the two extents exchanged. It is
        // not this decoder's canvas, and the message names the two rasters distinctly rather than
        // printing one number pair twice.
        let transposed = nn::capability::Clip {
            height: clip.width,
            width: clip.height,
            ..clip
        };
        let error = decoder
            .layout_of(transposed)
            .expect_err("the transposed raster is not this decoder's canvas");
        let message = error.to_string();
        assert!(
            message.contains(&format!("{}x{}", clip.width, clip.height)),
            "the refusal names the raster it decodes, width first: {message}"
        );
        assert!(
            message.contains(&format!("{}x{}", clip.height, clip.width)),
            "and the raster the call stated, in the same order: {message}"
        );
    }
    #[test]
    fn the_decoder_states_its_own_canvas_and_refuses_another_by_both_extents() {
        let config = video_config();
        let ratio = config.spatial_compression();
        // A latent canvas small enough to build from stated weights: the geometry is what is under
        // test, and every extent below follows from it.
        let canvas = Canvas::clip(&config, 2 * ratio, 2 * ratio);
        let decoder = stated_decoder(&config, canvas);

        let clip = clip_of(&config, canvas);
        let layout = decoder.layout_of(clip).expect("its own canvas");
        assert_eq!(
            (layout.video.height, layout.video.width, layout.video.frames),
            (canvas.height, canvas.width, canvas.frames),
            "the layout is the canvas's own latent extents"
        );

        let wider = nn::capability::Clip {
            width: clip.width + ratio,
            ..clip
        };
        let error = decoder
            .layout_of(wider)
            .expect_err("a raster of another width");
        assert!(
            error
                .to_string()
                .contains(&format!("{}x{}", clip.width, clip.height)),
            "the refusal names the raster it decodes: {error}"
        );

        let longer = nn::capability::Clip {
            frames: clip.frames + 1,
            ..clip
        };
        let error = decoder.layout_of(longer).expect_err("another frame count");
        assert!(
            error
                .to_string()
                .contains(&format!("{} frame(s)", clip.frames)),
            "the refusal names the frame count it decodes: {error}"
        );
    }

    #[test]
    fn the_decoder_table_names_one_position_per_row() {
        let config = video_config();
        let canvas = crate::config::Canvas {
            short_edge: 768,
            aspect: (7, 4),
            width: 1344,
            height: 768,
        };
        // The layout's own extents: the canvas over the spatial compression.
        let raster = crate::packing::latent_raster(&canvas).expect("raster");
        let clip = Canvas::clip(&config, raster.height, raster.width);
        // The position table is axis-major over the largest tile's rows, and the register rows and the
        // trailing zero row carry no coordinate.
        let tile = tile(&config, &clip).expect("a tile");
        let table = positions(&tile).expect("positions");
        assert_eq!(table.len() as u64, 3 * tile.rows());
        for channel in 0..3u64 {
            for row in tile.voxels()..tile.rows() {
                assert_eq!(table[(channel * tile.rows() + row) as usize], 0.0);
            }
        }
        // The one table a layout declares is the one this decoder reads, and its descriptor is the
        // extent the layout states - the adapter adds no arithmetic of its own.
        //
        // Two different quantities live in this one table, and three rounds were spent on this test
        // for want of saying so. Its **rows** are a row per row of the largest tile - the tile's voxels
        // plus the four register rows and the trailing zero row, 1,792 + 5 = 1,797 on every canvas - and
        // one clip's rows are the whole of it, because how many clips a request has is the runner's and
        // the splice's question, never this table's. Its **three** is the coordinate axis, which is a
        // dimension of its own: the values are `3 · rows` in a flat buffer and `[3, rows]` as declared,
        // so the twenty-four-radian frame row of each axis is contiguous. Neither number is a frame
        // count and neither multiplies the other.
        let layout = nn::capability::Layout {
            video: nn::capability::LatentShape {
                channels: config.latent_channels,
                frames: clip.frames,
                height: raster.height,
                width: raster.width,
            },
            audio: None,
            text_rows: 0,
            condition_rows: 0,
            condition_elements: 0,
        };
        let declared =
            table_specs(&config, layout.video.height, layout.video.width).expect("specs");
        assert_eq!(declared.len(), 2);
        assert_eq!(declared[0].name, "video_positions");
        // `[3, rows]`: the coordinate axis first, then one row per row of the largest tile.
        assert_eq!(declared[0].shape.rank(), 2);
        assert_eq!(declared[0].shape.dim(0), 3);
        assert_eq!(declared[0].shape.dim(1), tile.rows());
        assert_eq!(declared[0].dtype, DType::F32);
        // The stitch's table: one row of three words per pixel of each axis.
        assert_eq!(declared[1].name, "video_stitch");
        assert_eq!(declared[1].dtype, DType::I32);
        assert_eq!(declared[1].shape.dims(), [768 + 1344, 3]);
    }

    #[test]
    fn the_position_table_is_one_upload_for_every_canvas() {
        // The tiles are uniform, so the table is the largest tile's however large the canvas is: a
        // layout costs the worker one upload of the same size whatever its extents. The two canvases
        // here are the served ones - 768p at 768 x 1344 (48 x 84 latents) and 672p at 672 x 1536
        // (42 x 96) - and they are the same table, not two of the same shape by coincidence.
        let config = video_config();
        let zeroed = |height, width| -> Vec<TableValues> {
            table_specs(&config, height, width)
                .expect("the tables")
                .iter()
                .map(|table| TableValues::zeroed(table).expect("a host table"))
                .collect()
        };
        let mut vertical = zeroed(48, 84);
        let mut wide = zeroed(42, 96);
        table_values_into(&config, 48, 84, &mut vertical).expect("the vertical canvas");
        table_values_into(&config, 42, 96, &mut wide).expect("the wide canvas");
        assert_eq!(vertical[0], wide[0]);
    }

    #[test]
    fn a_layout_with_the_wrong_number_of_tables_is_refused() {
        // The worker sizes `out` from `table_specs`, so a count that disagrees with the block is a
        // protocol error and must surface as one rather than as a partially written table.
        let config = video_config();
        let mut none: Vec<TableValues> = Vec::new();
        assert!(table_values_into(&config, 48, 84, &mut none).is_err());
        let mut three = vec![TableValues::I32(Vec::new()); 3];
        assert!(table_values_into(&config, 48, 84, &mut three).is_err());
        // A table of the right count whose extents are not the layout's is refused by its name.
        let mut short = vec![TableValues::F32(Vec::new()), TableValues::I32(Vec::new())];
        let error = table_values_into(&config, 48, 84, &mut short).expect_err("short tables");
        assert!(error.to_string().contains("video_positions"), "{error}");
    }

    #[test]
    fn a_canvas_tile_is_its_grids_uniform_tile_with_the_token_suffix() {
        let config = video_config();
        let canvas = Canvas::clip(&config, 48, 84);
        assert_eq!(canvas.frames, 7);
        assert_eq!(canvas.rows(), 7 * 48 * 84 + 4 + 1);
        assert_eq!(grid(&config, &canvas).expect("a grid").len(), 4 * 7);
        let one = tile(&config, &canvas).expect("a tile");
        assert_eq!((one.height, one.width), (16, 16));
        assert_eq!(one.rows(), 7 * 16 * 16 + 4 + 1);
        // A canvas inside one tile is its own tile.
        let small = Canvas::clip(&config, 8, 12);
        assert_eq!(tile(&config, &small).expect("a tile"), small);
    }

    #[test]
    fn three_quarters_of_a_head_rotate_over_three_axes() {
        let config = video_config();
        let (width, sections) = rotary(&config).expect("a rotation");
        assert_eq!(width, 48);
        assert_eq!(sections, [16, 16, 16]);
        // The sections are the rotated width, and eight pairs rotate per axis.
        assert_eq!(sections.iter().sum::<u32>(), width);
        assert_eq!(width / 6, 8);
        // A head narrower than the rotation, or a fraction that is not whole, is refused.
        let mut broken = config.clone();
        broken.decoder_rope_dim_ratio = 0.7;
        assert!(rotary(&broken).is_err());
        let mut broken = config.clone();
        broken.decoder_attention_head_dim = 6;
        assert!(rotary(&broken).is_err());
        // A narrower head rotates all of itself: six elements, two per axis.
        let mut whole = config.clone();
        whole.decoder_attention_head_dim = 6;
        whole.decoder_rope_dim_ratio = 1.0;
        assert_eq!(rotary(&whole).expect("a rotation"), (6, [2, 2, 2]));
    }

    #[test]
    fn a_clip_decodes_seven_latent_frames_into_twenty_five() {
        let config = video_config();
        let clips = clips(&config);
        assert_eq!(clips.window(), 7);
        assert_eq!(config.temporal_compression(), 4);
        // The decode emits 28 frames and the splice keeps 25 of them: 25 = 17 + 3 + 5.
        assert_eq!(clips.native(), 25);
        assert_eq!(
            clips.window() * config.temporal_compression() - clips.native(),
            3
        );
        assert_eq!(clips.start(2), 10);
        let canvas = Canvas::clip(&config, 16, 16);
        assert_eq!(canvas.frames, 7);
        assert_eq!(canvas.frames_out(&config), 28);
        assert_eq!(canvas.rows(), 7 * 256 + 5);
    }

    #[test]
    fn the_position_table_is_the_cell_centre_grid_of_the_raster() {
        let config = video_config();
        let canvas = Canvas {
            frames: 2,
            height: 2,
            width: 3,
            registers: 4,
            tail: 1,
        };
        let rows = canvas.rows() as usize;
        let table = positions(&canvas).expect("positions");
        assert_eq!(table.len(), 3 * rows);
        let turn = std::f64::consts::TAU;
        // Frame axis of two: -0.5, 0.5. Row axis of two: -0.5, 0.5. Column axis of three: -2/3, 0,
        // 2/3 — times 2π, and one leading row per axis. The voxel rows go frame, row, column.
        let expected = [
            [-0.5, -0.5, -2.0 / 3.0],
            [-0.5, -0.5, 0.0],
            [-0.5, -0.5, 2.0 / 3.0],
            [-0.5, 0.5, -2.0 / 3.0],
            [-0.5, 0.5, 0.0],
            [-0.5, 0.5, 2.0 / 3.0],
            [0.5, -0.5, -2.0 / 3.0],
            [0.5, -0.5, 0.0],
            [0.5, -0.5, 2.0 / 3.0],
            [0.5, 0.5, -2.0 / 3.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 2.0 / 3.0],
        ];
        for (row, values) in expected.iter().enumerate() {
            for (channel, value) in values.iter().enumerate() {
                // The value the table must carry: the cell centre times one turn. The arithmetic is
                // float32, so the assertion is the semantic value within a float32 step rather than a
                // bit pattern, which is what makes it survive an equivalent implementation.
                let want = turn * value;
                let got = f64::from(table[channel * rows + row]);
                assert!(
                    (got - want).abs() < 1e-5,
                    "row {row} axis {channel}: {got} vs {want}"
                );
            }
        }
        // The register tokens and the trailing zero token are not placed: their rows are zeros.
        let trailing = 5;
        for channel in 0..3 {
            for row in rows - trailing..rows {
                assert_eq!(table[channel * rows + row], 0.0, "row {row} axis {channel}");
            }
        }
        // The checkpoint's clip canvas: 7 latent frames of 16x16 voxels, four registers, one tail.
        let clip = Canvas::clip(&config, 16, 16);
        let table = positions(&clip).expect("positions");
        assert_eq!(clip.rows(), 7 * 256 + 5);
        assert_eq!(table.len() as u64, clip.rows() * 3);
    }

    /// The dense checkpoint's video VAE config: the sidecar's numbers (`vae/config.json`).
    pub(crate) fn video_config() -> VideoVaeConfig {
        let sidecar = serde_json::json!({
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 24,
            "block_out_channels": [128, 256, 256, 512, 512, 1024],
            "layers_per_block": 2,
            "spatial_downsample_factors": [2, 2, 2, 2, 1, 1],
            "temporal_downsample_factors": [1, 2, 2, 1, 1, 1],
            "norm_num_groups": 32,
            "norm_eps": 1e-6,
            "spatial_padding_mode": "reflect",
            "decoder_num_layers": 36,
            "decoder_num_attention_heads": 32,
            "decoder_attention_head_dim": 64,
            "decoder_num_register_tokens": 4,
            "decoder_ffn_mult": 4,
            "decoder_rope_theta": 100.0,
            "decoder_rope_dim_ratio": 0.75,
            "decoder_norm_eps": 1e-5,
            "clip_length": 17,
            "token_drop": 3,
            "latents_mean": vec![0.0; 24],
            "latents_std": vec![1.0; 24],
        });
        VideoVaeConfig::parse(&sidecar).expect("the checkpoint's video VAE sidecar")
    }
}
