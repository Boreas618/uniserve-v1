//! The H3 video VAE decoder: its raster and tile arithmetic, and the numerical model that decodes one
//! clip of latents into RGB frames.
//!
//! `uniserve/models/minimax_h3/video_vae.py` is the reference. The decoder is a non-causal ViT: one
//! latent voxel becomes one token, four learned register tokens and one zero token trail the voxels,
//! and each token expands into a patch of RGB. `post_quant_conv` projects the latent channels, 36
//! transformer layers attend over the whole token sequence, and the output projection writes
//! `out_channels · temporal_compression · spatial_compression²` elements per token, which the
//! unpacking step lays out as a planar clip of frames.
//!
//! ## What this module owns
//!
//! * the arithmetic that turns a raster into token counts ([`Canvas`]) and a canvas into its
//!   overlapping decode tiles ([`Tiles`], [`Seams`]) — the same `split_tiles` the reference runs, so a
//!   tile's extent and every seam's overlap are host values a captured bucket can be planned from;
//! * the per-layer composition and the element type each layer declares (specs/worker.md 9:
//!   `torch.autocast` is replaced by a declaration, not by an implicit conversion).
//!
//! ## Precision
//!
//! The reference runs this decoder under `torch.autocast(dtype=FP16)` with the checkpoint's weights
//! converted per `precision.py`, and what autocast does to each operation is the table below. Every
//! row is stated in code here, and no conversion happens that a row does not name. `dtype = "auto"`
//! still decides everything this table does not: the parameters it lists as stored stay in the
//! checkpoint's own representation, which is FP32. This is the model's own declaration, so it takes
//! no preset and no switch.
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

use crate::output::VideoPostprocessor;
use crate::{Error, Result, VideoVaeConfig};

/// The latent frames one clip decodes: seven for the checkpoint.
///
/// A clip's native output is the 17-frame body, the three frames the final clip drops and the
/// five-frame tail — 25 frames — and the decoder emits `temporal_compression` frames per latent frame,
/// so it reads `ceil(25 / 4) = 7` latent frames and trims the three it does not keep. The reference
/// states the seven directly (`decoding.py`, `clip_input`); it is derived here so a checkpoint whose
/// clip, drop or compression differs cannot silently decode the wrong span.
pub fn decode_latents(config: &VideoVaeConfig) -> u32 {
    let native = config.clip_length + config.token_drop + crate::decoding::OVERLAP_FRAMES;
    native.div_ceil(config.temporal_compression())
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
    /// The canvas of one clip of `config`: [`decode_latents`] latent frames at the given latent extents.
    pub fn clip(config: &VideoVaeConfig, height: u32, width: u32) -> Self {
        Self {
            frames: decode_latents(config),
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

/// One axis of a canvas split into overlapping decode tiles.
///
/// `starts` and `lengths` are output pixels, `overlaps[i]` is the overlap between tile `i` and tile
/// `i + 1`. Every value is a multiple of the decoder's spatial compression, and a tile's latent slice
/// is its pixel extent divided by that ratio. The arithmetic is the reference `split_tiles`: the
/// fewest tiles whose minimum-overlap coverage reaches the extent, with the slack beyond the minimum
/// spread over the seams in tile order.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Seams {
    /// Where each tile starts, in output pixels.
    pub starts: Vec<u32>,
    /// How long each tile is, in output pixels.
    pub lengths: Vec<u32>,
    /// How much tile `i` overlaps tile `i + 1`.
    pub overlaps: Vec<u32>,
}

impl Seams {
    /// Split `length` output pixels into tiles of `tile` with at least `overlap` shared, on a grid of
    /// `alignment` pixels.
    ///
    /// The extent, the tile and the overlap must be multiples of the alignment, which is the
    /// decoder's spatial compression: a tile reads whole latent columns.
    pub fn split(length: u32, tile: u32, overlap: u32, alignment: u32) -> Result<Self> {
        if length == 0 || tile == 0 || alignment == 0 {
            return Err(Error::config(
                "a tile split needs a positive extent, tile and alignment".to_owned(),
            ));
        }
        if overlap >= tile {
            return Err(Error::config(format!(
                "a {overlap}-pixel overlap does not fit a {tile}-pixel tile"
            )));
        }
        for (name, value) in [("extent", length), ("tile", tile), ("overlap", overlap)] {
            if !value.is_multiple_of(alignment) {
                return Err(Error::config(format!(
                    "the {name} {value} is not a multiple of the {alignment}-pixel alignment"
                )));
            }
        }
        if tile >= length {
            return Ok(Self {
                starts: vec![0],
                lengths: vec![length],
                overlaps: Vec::new(),
            });
        }

        // The fewest tiles whose minimum-overlap coverage reaches the extent.
        let mut count = length.div_ceil(tile);
        while u64::from(tile) * u64::from(count) - u64::from(overlap) * u64::from(count - 1)
            < u64::from(length)
        {
            count += 1;
        }

        // Spread the slack beyond the minimum overlap over the seams, in tile order, in whole
        // alignment steps: the first seams of the grid take one step each, cyclically.
        let seams = count - 1;
        let mut overlaps = vec![overlap; seams as usize];
        let covered: u64 =
            u64::from(tile) * u64::from(count) - u64::from(overlap) * u64::from(seams);
        let slack = covered - u64::from(length);
        for index in 0..(slack / u64::from(alignment)) as usize {
            let seam = index % overlaps.len();
            overlaps[seam] += alignment;
        }

        let mut starts = Vec::with_capacity(count as usize);
        starts.push(0);
        for overlap in &overlaps {
            let next = starts[starts.len() - 1] + tile - overlap;
            starts.push(next);
        }
        Ok(Self {
            starts,
            lengths: vec![tile; count as usize],
            overlaps,
        })
    }

    /// How many tiles this axis has.
    pub fn len(&self) -> usize {
        self.starts.len()
    }


    /// The latent span of tile `index`: `(first, end)` in latent cells, end exclusive, over a raster
    /// whose extent is `raster` **output pixels** (the same unit `split` was given).
    pub fn latent(&self, index: usize, raster: u32, ratio: u32) -> (u32, u32) {
        let start = self.starts[index];
        let end = (start + self.lengths[index]).min(raster);
        (start / ratio, end.div_ceil(ratio))
    }

    /// The overlap between tile `index` and tile `index + 1`, in output pixels.
    pub fn overlap(&self, index: usize) -> Option<u32> {
        self.overlaps.get(index).copied()
    }
}

/// The tile grid of one clip's canvas: the same split applied to the height and to the width.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Tiles {
    /// Rows of tiles.
    pub rows: Seams,
    /// Columns of tiles.
    pub columns: Seams,
}

/// One tile of a canvas, in the order the decoder decodes and blends it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Tile {
    /// Row of the tile grid.
    pub row: usize,
    /// Column of the tile grid.
    pub column: usize,
    /// The tile's latent rows: `(start, end)`.
    pub latent_rows: (u32, u32),
    /// The tile's latent columns: `(start, end)`.
    pub latent_columns: (u32, u32),
    /// The overlap this tile shares with the tile above it, in output pixels.
    pub above: u32,
    /// The overlap this tile shares with the tile to its left, in output pixels.
    pub left: u32,
    /// The overlap the tile below this one shares with it, in output pixels.
    pub below: u32,
    /// The overlap the tile to the right shares with it, in output pixels.
    pub right: u32,
    /// The tile's own token canvas.
    pub canvas: Canvas,
}

impl Tiles {
    /// The tile grid of `canvas`, with the checkpoint's 256-pixel tiles and 64-pixel minimum overlap.
    pub fn of(config: &VideoVaeConfig, canvas: &Canvas) -> Result<Self> {
        let ratio = config.spatial_compression();
        let (height, width) = canvas.pixels(config);
        Ok(Self {
            rows: Seams::split(height, TILE_PIXELS, MINIMUM_OVERLAP, ratio)?,
            columns: Seams::split(width, TILE_PIXELS, MINIMUM_OVERLAP, ratio)?,
        })
    }

    /// Every tile of the grid, row-major: the order in which the reference decodes and blends them.
    ///
    /// A tile's canvas is its own latent slice, so no tile reads a neighbour's latents (the reference
    /// has no halo). The overlaps are stated on the tile that is blended *into* its predecessor: `above`
    /// and `left` are the ones this tile cross-fades, `below` and `right` are the ones its successor
    /// will, and a tile trims its trailing rows and columns exactly when those are nonzero.
    pub fn tiles(&self, config: &VideoVaeConfig, canvas: &Canvas) -> Vec<Tile> {
        let ratio = config.spatial_compression();
        // The seams are split on the canvas's output pixels, so the tile's span is clipped there and
        // then divided into latent cells; using the latent extent here would clip pixel coordinates
        // to a latent count and give a tile that ends before it starts.
        let (height, width) = canvas.pixels(config);
        let mut tiles = Vec::with_capacity(self.rows.len() * self.columns.len());
        for row in 0..self.rows.len() {
            for column in 0..self.columns.len() {
                let (row_start, row_end) = self.rows.latent(row, height, ratio);
                let (column_start, column_end) = self.columns.latent(column, width, ratio);
                let above = row
                    .checked_sub(1)
                    .map_or(0, |previous| self.rows.overlap(previous).unwrap_or(0));
                let left = column
                    .checked_sub(1)
                    .map_or(0, |previous| self.columns.overlap(previous).unwrap_or(0));
                tiles.push(Tile {
                    row,
                    column,
                    latent_rows: (row_start, row_end),
                    latent_columns: (column_start, column_end),
                    above,
                    left,
                    below: self.rows.overlap(row).unwrap_or(0),
                    right: self.columns.overlap(column).unwrap_or(0),
                    canvas: Canvas {
                        frames: canvas.frames,
                        height: row_end - row_start,
                        width: column_end - column_start,
                        registers: canvas.registers,
                        tail: canvas.tail,
                    },
                });
            }
        }
        tiles
    }
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
/// cell-centre grid (`video_vae.py`, `Transformer.forward`). The rows are the canvas's latent voxels in
/// raster order — frame, then row, then column — followed by the register tokens and the trailing zero
/// token, whose coordinates are all zero because they are not placed in space.
///
/// The table is `[3, rows]` F32, axis-major, because the rotation reads one leading coordinate per
/// axis (`Sections::new(&[16, 16, 16])`): a row-major `[rows, 3]` table would be read transposed and
/// silently rotate every row by another row's angles. Coordinates are in radians — the reference's
/// cell-centre grid times `2π` — so the raw inverse frequencies give the reference's angle.
///
/// The table is a per-layout constant: the runner uploads it once per captured graph
/// (`notes/h3/runner-interface.md` 2.1, `video_positions`).
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
    // The reference's own arithmetic: the cell centre and the `2π` turn are both float32
    // (`video_vae.py:583-600`, whose meshgrid is float32 and whose caller multiplies by `2π` in the
    // same type), so one rounding stands between the reference and this table rather than two.
    let centre = (index as f32 + 0.5) / size as f32;
    std::f32::consts::TAU * (2.0 * centre - 1.0)
}

/// The tile extent of the checkpoint's decoder, in output pixels.
pub const TILE_PIXELS: u32 = 256;

/// The minimum overlap between two adjacent tiles, in output pixels.
pub const MINIMUM_OVERLAP: u32 = 64;

/// Every parameter of the decoder, with its checkpoint key.
///
/// The decoder is the post-quantization projection, the input projection, the register table, 36
/// transformer layers of twelve parameters each, and the final LayerNorm and output projection read
/// from the checkpoint's name-translated keys (`decoder.proj_in`, `decoder.transformer_blocks.<i>`,
/// `decoder.norm_out`, `decoder.proj_out`, `post_quant_conv`). The feed-forward projection is one
/// packed tensor, value rows first, which is what the value-first SwiGLU reads: the reference splits
/// it into two views only for its quantized fused path, and the dense decoder never does.
///
/// The vocabulary is [`crate::audio_vae::Source`]'s: a logical parameter named once, and the
/// checkpoint tensor that fills it.
pub fn sources(config: &VideoVaeConfig) -> crate::Result<Vec<crate::audio_vae::Source>> {
    use crate::audio_vae::{Source, Tensor};

    let mut sources = Vec::new();
    let region = |parameter: &str, tensor: String| Source {
        parameter: parameter.to_owned(),
        tensor: Tensor::Region(tensor),
    };

    for (parameter, tensor) in [
        ("post_quant.weight", "post_quant_conv.weight".to_owned()),
        ("post_quant.bias", "post_quant_conv.bias".to_owned()),
        ("input.weight", "decoder.proj_in.weight".to_owned()),
        ("input.bias", "decoder.proj_in.bias".to_owned()),
        ("registers", "decoder.register_tokens".to_owned()),
    ] {
        sources.push(region(parameter, tensor));
    }
    for index in 0..config.decoder_num_layers {
        let at = |name: &str| format!("decoder.transformer_blocks.{index}.{name}");
        for (parameter, tensor) in [
            ("norm1.weight", at("norm1.weight")),
            ("norm2.weight", at("norm2.weight")),
            ("scale1", at("scale1")),
            ("scale2", at("scale2")),
            ("query.weight", at("attn.to_q.weight")),
            ("query.bias", at("attn.to_q.bias")),
            ("key.weight", at("attn.to_k.weight")),
            ("key.bias", at("attn.to_k.bias")),
            ("value.weight", at("attn.to_v.weight")),
            ("value.bias", at("attn.to_v.bias")),
            ("output.weight", at("attn.to_out.0.weight")),
            ("output.bias", at("attn.to_out.0.bias")),
            ("gate_up.weight", at("ff.net.0.proj.weight")),
            ("gate_up.bias", at("ff.net.0.proj.bias")),
            ("down.weight", at("ff.net.2.weight")),
            ("down.bias", at("ff.net.2.bias")),
        ] {
            sources.push(region(&format!("layers.{index}.{parameter}"), tensor));
        }
    }
    for (parameter, tensor) in [
        ("norm.weight", "decoder.norm_out.weight".to_owned()),
        ("norm.bias", "decoder.norm_out.bias".to_owned()),
        ("output.weight", "decoder.proj_out.weight".to_owned()),
        ("output.bias", "decoder.proj_out.bias".to_owned()),
    ] {
        sources.push(region(parameter, tensor));
    }
    Ok(sources)
}

// ---------------------------------------------------------------------------------------------
// The video decoder's transformer.
// ---------------------------------------------------------------------------------------------

use nn::Forward;
use nn::layers::{
    Attention, Linear, PackedSwiGlu, Rotary, Rows, ScaledAdd, ScaledResidualLayerNorm,
    ScaledResidualRmsNorm, Sections, WeightedRmsNorm,
};
use nn::weights::{Weight, WeightLayout, WeightLoader, WeightRequest};
use nn::{DType, DecodedMedia, Decoder, Shape, StateTable, TensorView};
use ops::gemm::WeightOrder;

/// The element type of the decoder's projections and of the rows they read and write, and the type
/// its rasters are written in: the model's own declaration of the reference's `quality` row (see the
/// module's table).
pub const ACTIVATION: DType = DType::F16;

/// The element type of the latents a decode reads and of the residual stream, which the reference
/// keeps wider than the rows its projections read.
pub const STREAM: DType = DType::F32;

/// One transformer layer of the decoder: the attention half and the gated feed-forward half.
///
/// The residual updates are folded into the normalization that reads them, as the reference's
/// `_residual_norm` does: the attention's output joins the stream and normalizes for the feed-forward
/// in one launch, and the feed-forward's output joins it and normalizes for the next layer.
struct Layer {
    /// The query, key and value projections as one `[3 · width, width]` product: one read of the
    /// normalized rows, and the three outputs are column ranges of one buffer.
    query_key_value: Linear,
    output: Linear,
    /// `scale1`, then the second normalization.
    attended: ScaledResidualRmsNorm,
    gate_up: Linear,
    gated: PackedSwiGlu,
    down: Linear,
    /// `scale2`, then the next layer's first normalization; the last layer folds into `last`.
    fed: Option<ScaledResidualRmsNorm>,
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
    /// The tile stitch, built at load so that no layer is built on a request's path.
    stitch: crate::decoding::ClipDecoder,
    width: u32,
    /// `value · std + mean` per latent channel, as a product with the diagonal of the deviations: the
    /// reference restores the latents in FP32 before anything rounds them.
    restore: Linear,
    post_quant: Linear,
    /// `post_quant_conv`'s bias, as a product with the identity: the reference's convolution rounds
    /// its product before it adds the bias, and this is that second rounding.
    post_quant_bias: Linear,
    input: Linear,
    registers: Weight,
    register_rows: Weight,
    zero_row: Weight,
    zero_rows: Weight,
    /// The first layer's first normalization, which has no update to fold.
    first: WeightedRmsNorm,
    layers: Vec<Layer>,
    /// The last layer's feed-forward update and the final normalization.
    last: ScaledResidualLayerNorm,
    output: Linear,
    rotary: Rotary,
    place: Rows,
    /// The offsets of the clip's sequences: `[0, rows, 2 · rows, ...]`, one per tile of the largest
    /// canvas this decoder serves. A tile's rows are one sequence; a clip of fewer tiles reads the
    /// table's prefix.
    offsets: Weight,
    config: VideoVaeConfig,
    ffn_mult: u32,
    /// Elements per head.
    head: u64,
    /// Query heads per token.
    heads: u64,
    attention: Attention,
}

/// The canvas of the grid's largest tile, whose position table every tile's is a slice of.
///
/// The grid's tiles are uniform where the canvas is at least one tile wide, so this is the checkpoint's
/// tile; a canvas narrower than one is its own largest tile.
fn largest_tile(config: &VideoVaeConfig, latent_height: u32, latent_width: u32) -> Result<Canvas> {
    let clip = Canvas::clip(config, latent_height, latent_width);
    let tiles = Tiles::of(config, &clip)?.tiles(config, &clip);
    tiles
        .iter()
        .max_by_key(|tile| tile.canvas.rows())
        .map(|tile| tile.canvas)
        .ok_or_else(|| Error::config("a canvas with no tile has no position table".to_owned()))
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
    let positions = u64::from(positions_table(config, latent_height, latent_width)?);
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

/// Write the host bytes of [`table_specs`], in its order, checking every buffer's length.
pub(crate) fn table_bytes_into(
    config: &VideoVaeConfig,
    latent_height: u32,
    latent_width: u32,
    out: &mut [Vec<u8>],
) -> Result<()> {
    let tables = table_specs(config, latent_height, latent_width)?;
    if out.len() != tables.len() {
        return Err(Error::config(format!(
            "{} tables are read and {} buffers given",
            tables.len(),
            out.len()
        )));
    }
    let values = positions(&largest_tile(config, latent_height, latent_width)?)?;
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    out[0] = bytes;
    let clip = Canvas::clip(config, latent_height, latent_width);
    out[1] = crate::decoding::stitch_table(config, &clip)?
        .iter()
        .flatten()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    for (buffer, table) in out.iter().zip(&tables) {
        let expected = table.shape.numel() as usize * (table.dtype.bits() as usize / 8);
        if buffer.len() != expected {
            return Err(Error::config(format!(
                "{} is {} bytes and the table needs {expected}",
                table.name,
                buffer.len()
            )));
        }
    }
    Ok(())
}

/// Rows of the largest tile's position table, the extent every tile's positions are a slice of.
fn positions_table(config: &VideoVaeConfig, latent_height: u32, latent_width: u32) -> Result<u32> {
    u32::try_from(largest_tile(config, latent_height, latent_width)?.rows())
        .map_err(|_| Error::config("a tile taller than a table bound".to_owned()))
}

/// The attention of the decoder's block: FlashAttention 4 over one packed sequence per tile, which is
/// the kernel the reference runs at this declaration.
fn block_attention(head: u32, heads: u32, rows: u32) -> Result<Attention> {
    let scale = (head as f32).sqrt().recip();
    Ok(Attention::dense(
        head, heads, heads, rows, false, None, scale,
    )?)
}

/// The tiles of the grid `canvas` splits into: how many sequences a clip of it decodes in one launch.
fn tile_count(config: &VideoVaeConfig, canvas: &Canvas) -> Result<u64> {
    let tiles = Tiles::of(config, canvas)?;
    Ok(tiles.rows.len() as u64 * tiles.columns.len() as u64)
}

impl VideoDecoder {
    /// Every checkpoint tensor the video decoder reads, by name.
    ///
    /// The decoder is a component whose names were already data ([`sources`] maps each parameter to
    /// the checkpoint key that fills it), so this list is that mapping's other column rather than a
    /// second spelling of it: the 585 names are the export's 583 `decoder.*` and the two
    /// `post_quant_conv.*`. A checkpoint that lacks one is refused by block and name before a byte is
    /// read.
    pub fn keys(config: &VideoVaeConfig) -> crate::Result<Vec<String>> {
        Ok(crate::audio_vae::tensor_keys(&sources(config)?))
    }

    /// The programs one clip's decode of `config` launches.
    ///
    /// Stated from the configuration alone, because a deployment compiles its store before it opens
    /// the checkpoint, and each row's element types are this module's declaration. **No extent enters a
    /// key**: a clip's token count is a launch value the attention and the normalizations read
    /// position by position, so one program serves every raster — which is why this takes no bucket
    /// set while the post-processor, whose blend is keyed by its pixel count, does.
    pub fn programs(config: &VideoVaeConfig) -> Result<Vec<(String, String)>> {
        use crate::programs::Programs;
        use ops::activation::ValueFirstSwiglu as SwigluOp;
        use ops::attention::dense::DenseAttention;
        use ops::gemm::Gemm;
        use ops::residual::{
            ScaledAdd as ScaledAddOp, ScaledResidualLayerNorm as LayerNormOp,
            ScaledResidualRmsNorm as RmsNormOp, WeightedRmsNorm as FirstNormOp,
        };
        use ops::rope::{QkNormRope, RotaryFactors, Rotation};
        use ops::rows::{self, Rows as RowsOp};
        use ops::tile_stitch::TileStitch as TileStitchOp;

        let (head, heads) = (
            config.decoder_attention_head_dim,
            config.decoder_num_attention_heads,
        );
        let width = heads * head;
        let ffn = width * config.decoder_ffn_mult;
        let (rotated, sections) = rotary(config)?;
        let sections = nn::layers::Sections::new(&sections)?;

        let mut programs = Programs::default();

        // The restore in FP32, then every projection: all of them carry a bias in their epilogue, as
        // the reference's dense layers do, so the contraction length is the only thing that differs
        // and it is no part of the key.
        programs.launch::<Gemm>(&Linear::descriptor_of(
            config.latent_channels,
            STREAM,
            STREAM,
            WeightOrder::NxK,
            true,
            STREAM,
        ));
        for k in [config.latent_channels, width, ffn] {
            programs.launch::<Gemm>(&Linear::descriptor_of(
                k,
                ACTIVATION,
                ACTIVATION,
                WeightOrder::NxK,
                true,
                ACTIVATION,
            ));
        }
        // `post_quant_conv`'s product, which rounds before its bias joins it.
        programs.launch::<Gemm>(&Linear::descriptor_of(
            config.latent_channels,
            ACTIVATION,
            ACTIVATION,
            WeightOrder::NxK,
            false,
            ACTIVATION,
        ));

        // The gather of a clip's rows out of the trajectory, one strided copy per tile.
        programs
            .launch::<ops::copy_block::CopyBlock>(&nn::layers::CopyBlock::descriptor_of(STREAM));
        // The tail a clip leaves its successor, out of the clip's raster.
        programs.launch::<ops::copy_block::CopyBlock>(&nn::layers::CopyBlock::wide_descriptor_of(
            ACTIVATION,
        ));

        // The two crossings between the stream's type and the activation's: the restored latents
        // round down, and the projected voxels widen into the stream.
        programs.launch::<ScaledAddOp>(&ScaledAdd.descriptor(ACTIVATION, STREAM));
        programs.launch::<ScaledAddOp>(&ScaledAdd.descriptor(STREAM, ACTIVATION));

        // The two row writes into the token stream, the register rows and the zero row: every
        // row-major gather is one program.
        programs.launch::<RowsOp>(&Rows::descriptor_of(STREAM, rows::Mode::Replace, width));

        // The rotation's table once, then its application in every layer.
        programs.launch::<RotaryFactors>(&Rotary::factors_descriptor_at(
            rotated,
            1.0,
            Some(sections),
            DType::F32,
            ACTIVATION,
        ));
        programs.launch::<QkNormRope>(&Rotary::strided_descriptor_of(
            head,
            heads,
            Rotation::Half,
            config.decoder_norm_eps,
            rotated,
            ACTIVATION,
            ACTIVATION,
        ));

        // The stream's three normalizations and the gated feed-forward's packed activation.
        programs.launch::<FirstNormOp>(&WeightedRmsNorm::descriptor_of(width, STREAM, ACTIVATION));
        programs.launch::<RmsNormOp>(&ScaledResidualRmsNorm::descriptor_of(
            width, STREAM, ACTIVATION, false,
        ));
        programs.launch::<LayerNormOp>(&ScaledResidualLayerNorm::descriptor_of(
            width, STREAM, ACTIVATION, false,
        ));
        programs.launch::<SwigluOp>(&PackedSwiGlu::descriptor_of(
            ACTIVATION,
            u64::from(ffn),
            2 * u64::from(ffn),
            false,
            false,
        )?);

        // The output path: the one stitch.
        programs.launch::<TileStitchOp>(
            &crate::decoding::ClipDecoder::new(config)?
                .stitch()
                .descriptor(),
        );

        // One attention for the whole decoder: the sequences and their rows are launch values.
        let attention = block_attention(head, heads, width)?;
        programs.launch::<DenseAttention>(
            &attention.dense_descriptor(ACTIVATION, ACTIVATION, ACTIVATION, width, true),
        );

        Ok(programs.finish())
    }

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
        let tile = largest_tile(config, canvas.height, canvas.width)?;
        for other in &canvases[1..] {
            if largest_tile(config, other.height, other.width)? != tile {
                return Err(Error::config(format!(
                    "the deployment serves a {}x{} raster and a {}x{} one whose largest tiles \
                     differ: the tables a decode reads are one tile's, so every canvas must be a \
                     whole number of the same tiles",
                    canvas.height, canvas.width, other.height, other.width
                )));
            }
        }
        // The canvas of the most tiles, which the sequence offsets and the attention's bound answer to.
        let mut widest = canvas;
        for other in &canvases {
            if tile_count(config, other)? > tile_count(config, &widest)? {
                widest = *other;
            }
        }
        let most_tiles = tile_count(config, &widest)?;
        let width = config.decoder_num_attention_heads * config.decoder_attention_head_dim;
        let head = config.decoder_attention_head_dim;
        let heads = config.decoder_num_attention_heads;
        let ffn = width * config.decoder_ffn_mult;
        let (rotated, sections) = rotary(config)?;
        let stitch = crate::decoding::ClipDecoder::new(config)?;
        let out_channels = config.out_channels
            * config.temporal_compression()
            * config.spatial_compression()
            * config.spatial_compression();
        let epsilon = config.decoder_norm_eps;
        let channels = u64::from(config.latent_channels);
        // The constants this canvas publishes, declared once: a caller that reserves device storage
        // before loading sizes its pool by this list, and every publish below is checked against it.
        let declared_sizes = constants(config, widest, u64::from(width))?;

        let (restore_weight, restore_bias) = restore_tables(config)?;
        let restore = Linear::new(
            constant(
                loader,
                &declared_sizes,
                "video_vae.restore.weight",
                Shape::new(&[channels, channels]),
                &restore_weight,
            )?,
            Some(constant(
                loader,
                &declared_sizes,
                "video_vae.restore.bias",
                Shape::new(&[channels]),
                &restore_bias,
            )?),
            WeightOrder::NxK,
        )?;
        let post_quant = Linear::new(
            loader.load(
                &WeightRequest::region(
                    "post_quant_conv.weight",
                    0,
                    Shape::new(&[channels, channels]),
                )
                .pinned(ACTIVATION),
            )?,
            None,
            WeightOrder::NxK,
        )?;
        let identity: Vec<u8> = (0..channels * channels)
            .flat_map(|index| {
                let one = index / channels == index % channels;
                nn::weights::f32_to_half(if one { 1.0 } else { 0.0 }).to_le_bytes()
            })
            .collect();
        declared(&declared_sizes, "video_vae.identity", identity.len())?;
        let post_quant_bias = Linear::new(
            loader.constant(
                "video_vae.identity",
                ACTIVATION,
                Shape::new(&[channels, channels]),
                &identity,
                "minimax-h3",
            )?,
            Some(
                loader.load(
                    &WeightRequest::region("post_quant_conv.bias", 0, Shape::new(&[channels]))
                        .pinned(ACTIVATION),
                )?,
            ),
            WeightOrder::NxK,
        )?;
        let input = linear(loader, "decoder.proj_in", channels, u64::from(width))?;

        let registers = read(
            loader,
            "decoder.register_tokens",
            &[u64::from(canvas.registers), u64::from(width)],
        )?;
        let zero_row = zeroes(
            loader,
            &declared_sizes,
            "video_vae.zero_row",
            u64::from(width),
        )?;
        if canvas.tail == 0 {
            return Err(Error::config(
                "the decoder canvas has the trailing zero row".to_owned(),
            ));
        }
        // **A tile's rows, not the clip's.** These two tables say where a tile's register tokens and
        // its trailing zero row go, and the row write launches against that tile's own slice of the
        // stream, so one tile's offsets serve every tile and every canvas of a uniform grid.
        let register_rows = indices(
            loader,
            &declared_sizes,
            "video_vae.register_rows",
            tile.voxels() as u32,
            canvas.registers,
        )?;
        let zero_rows = indices(
            loader,
            &declared_sizes,
            "video_vae.zero_rows",
            (tile.voxels() + u64::from(canvas.registers)) as u32,
            canvas.tail,
        )?;

        let vector =
            |loader: &mut WeightLoader, name: String| read(loader, &name, &[u64::from(width)]);
        let first = WeightedRmsNorm::new(
            vector(
                loader,
                "decoder.transformer_blocks.0.norm1.weight".to_owned(),
            )?,
            epsilon,
        )?;
        let weight = vector(loader, "decoder.norm_out.weight".to_owned())?;
        let bias = vector(loader, "decoder.norm_out.bias".to_owned())?;
        let count = config.decoder_num_layers;
        let last_scale = vector(
            loader,
            format!("decoder.transformer_blocks.{}.scale2", count - 1),
        )?;
        let last = ScaledResidualLayerNorm::new(last_scale, None, weight, bias, epsilon)?;

        let mut layers = Vec::with_capacity(count as usize);
        for index in 0..count {
            let base = format!("decoder.transformer_blocks.{index}");
            let scale1 = vector(loader, format!("{base}.scale1"))?;
            let norm2 = vector(loader, format!("{base}.norm2.weight"))?;
            let fed = if index + 1 < count {
                let scale2 = vector(loader, format!("{base}.scale2"))?;
                let next = format!("decoder.transformer_blocks.{}.norm1.weight", index + 1);
                let norm1 = vector(loader, next)?;
                Some(ScaledResidualRmsNorm::new(scale2, None, norm1, epsilon)?)
            } else {
                None
            };
            let parts = |name: &str| {
                ["to_q", "to_k", "to_v"]
                    .map(|projection| format!("{base}.attn.{projection}.{name}"))
                    .to_vec()
            };
            let query_key_value = Linear::new(
                loader.load(
                    &WeightRequest::parts(
                        format!("{base}.attn.qkv.weight"),
                        parts("weight"),
                        Shape::new(&[3 * u64::from(width), u64::from(width)]),
                        WeightLayout::ROW_MAJOR,
                    )
                    .pinned(ACTIVATION),
                )?,
                Some(
                    loader.load(
                        &WeightRequest::parts(
                            format!("{base}.attn.qkv.bias"),
                            parts("bias"),
                            Shape::new(&[3 * u64::from(width)]),
                            WeightLayout::ROW_MAJOR,
                        )
                        .pinned(ACTIVATION),
                    )?,
                ),
                WeightOrder::NxK,
            )?;
            layers.push(Layer {
                query_key_value,
                output: linear(
                    loader,
                    &format!("{base}.attn.to_out.0"),
                    u64::from(width),
                    u64::from(width),
                )?,
                attended: ScaledResidualRmsNorm::new(scale1, None, norm2, epsilon)?,
                // The feed-forward's projection is packed: its rows hold the value half and then the
                // gate half, so it is twice the feed-forward's width.
                gate_up: linear(
                    loader,
                    &format!("{base}.ff.net.0.proj"),
                    u64::from(width),
                    2 * u64::from(ffn),
                )?,
                gated: PackedSwiGlu::new(None)?,
                down: linear(
                    loader,
                    &format!("{base}.ff.net.2"),
                    u64::from(ffn),
                    u64::from(width),
                )?,
                fed,
            });
        }

        let output = linear(
            loader,
            "decoder.proj_out",
            u64::from(width),
            u64::from(out_channels),
        )?;
        let frequencies = frequencies(loader, &declared_sizes, config)?;
        let rotary = Rotary::new(
            frequencies,
            head,
            heads,
            heads,
            epsilon,
            crate::SPLIT_HALF,
            1.0,
        )?
        .partial(rotated)?
        .sectioned(Sections::new(&sections)?)?;
        let rows = tile.rows();
        let offsets = {
            let mut bytes = Vec::with_capacity((most_tiles as usize + 1) * 4);
            for sequence in 0..=most_tiles {
                let offset = i32::try_from(sequence * rows).map_err(|_| {
                    Error::config("a clip's rows exceed a sequence offset".to_owned())
                })?;
                bytes.extend_from_slice(&offset.to_le_bytes());
            }
            declared(&declared_sizes, "video_vae.sequence_offsets", bytes.len())?;
            loader.constant(
                "video_vae.sequence_offsets",
                DType::I32,
                Shape::new(&[most_tiles + 1]),
                &bytes,
                "minimax-h3",
            )?
        };
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
            first,
            layers,
            last,
            output,
            rotary,
            place: Rows::new(STREAM, u64::from(width))?,
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
            ffn_mult: config.decoder_ffn_mult,
            head: u64::from(head),
            heads: u64::from(heads),
        })
    }
}

/// Read a projection of the checkpoint at [`ACTIVATION`]: `[out, in]` row-major, with its bias.
fn linear(loader: &mut WeightLoader, name: &str, from: u64, to: u64) -> Result<Linear> {
    let weight = loader.load(
        &WeightRequest::region(format!("{name}.weight"), 0, Shape::new(&[to, from]))
            .pinned(ACTIVATION),
    )?;
    let bias = loader.load(
        &WeightRequest::region(format!("{name}.bias"), 0, Shape::new(&[to])).pinned(ACTIVATION),
    )?;
    Ok(Linear::new(weight, Some(bias), WeightOrder::NxK)?)
}

/// The restore's weight and bias: the diagonal of the latent deviations, and the means.
fn restore_tables(config: &VideoVaeConfig) -> Result<(Vec<f32>, Vec<f32>)> {
    let channels = config.latents_std.len();
    if channels == 0 || config.latents_mean.len() != channels {
        return Err(Error::config(format!(
            "the decoder's latent statistics hold {} means and {channels} deviations",
            config.latents_mean.len()
        )));
    }
    let mut weight = vec![0.0_f32; channels * channels];
    for (channel, deviation) in config.latents_std.iter().enumerate() {
        weight[channel * channels + channel] = *deviation;
    }
    Ok((weight, config.latents_mean.clone()))
}

/// The constants a decoder of this canvas publishes, as `(name, bytes)`.
///
/// `load` builds tensors the export does not hold - the latent restore, the zero row the token layout
/// appends, the two index tables that address the trailing rows, the rotary frequencies and the
/// sequence offsets - and a caller that reserves device storage before loading has to know what they
/// cost: the pool charges a block per publish, so a reservation computed from the checkpoint's names
/// alone is short by every one of these. The list is one statement of them, and `load` checks what it
/// publishes against it, so the two cannot drift. `canvas` is the served canvas of the most tiles.
pub fn constants(
    config: &VideoVaeConfig,
    canvas: Canvas,
    width: u64,
) -> Result<Vec<(String, u64)>> {
    let (rotated, _) = rotary(config)?;
    let element = STREAM.bytes().unwrap_or(4);
    let channels = u64::from(config.latent_channels);
    Ok(vec![
        (
            "video_vae.restore.weight".to_owned(),
            channels * channels * element,
        ),
        ("video_vae.restore.bias".to_owned(), channels * element),
        (
            "video_vae.identity".to_owned(),
            channels * channels * ACTIVATION.bytes().unwrap_or(2),
        ),
        ("video_vae.zero_row".to_owned(), width * element),
        (
            "video_vae.register_rows".to_owned(),
            u64::from(canvas.registers) * 4,
        ),
        ("video_vae.zero_rows".to_owned(), u64::from(canvas.tail) * 4),
        (
            format!("video_vae.rotary.{rotated}.{}", config.decoder_rope_theta),
            u64::from(rotated / 2) * 4,
        ),
        // `[0, rows, ...]`, one offset per tile and the end.
        (
            "video_vae.sequence_offsets".to_owned(),
            (tile_count(config, &canvas)? + 1) * 4,
        ),
    ])
}

/// Read one of the parameters the module's table keeps at the stream's width.
fn read(loader: &mut WeightLoader, name: &str, shape: &[u64]) -> Result<Weight> {
    Ok(
        loader
            .load(&WeightRequest::region(name.to_owned(), 0, Shape::new(shape)).pinned(STREAM))?,
    )
}

/// Check one published constant against [`constants`]' list, so the declaration a caller reserves by
/// and the tensors the loader publishes are one statement rather than two that can drift.
fn declared(declared: &[(String, u64)], name: &str, bytes: usize) -> Result<()> {
    match declared.iter().find(|(entry, _)| entry == name) {
        Some((_, expected)) if *expected == bytes as u64 => Ok(()),
        Some((_, expected)) => Err(Error::config(format!(
            "the constant `{name}` publishes {bytes} bytes where its declared size is {expected}"
        ))),
        None => Err(Error::config(format!(
            "the constant `{name}` is published but not declared"
        ))),
    }
}

/// A constant row of `width` zeros, which the checkpoint's token layout appends to the stream.
fn zeroes(
    loader: &mut WeightLoader,
    declared_sizes: &[(String, u64)],
    name: &str,
    width: u64,
) -> Result<Weight> {
    let bytes = vec![0u8; (width as usize) * STREAM.bytes().unwrap_or(4) as usize];
    declared(declared_sizes, name, bytes.len())?;
    Ok(loader.constant(name, STREAM, Shape::new(&[1, width]), &bytes, "minimax-h3")?)
}

/// A constant of FP32 `values`.
fn constant(
    loader: &mut WeightLoader,
    declared_sizes: &[(String, u64)],
    name: &str,
    shape: Shape,
    values: &[f32],
) -> Result<Weight> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    declared(declared_sizes, name, bytes.len())?;
    Ok(loader.constant(name, STREAM, shape, &bytes, "minimax-h3")?)
}

/// A constant index table: `count` rows starting at `start`.
fn indices(
    loader: &mut WeightLoader,
    declared_sizes: &[(String, u64)],
    name: &str,
    start: u32,
    count: u32,
) -> Result<Weight> {
    declared(declared_sizes, name, count as usize * 4)?;
    let mut bytes = Vec::with_capacity(count as usize * 4);
    for index in 0..count {
        bytes.extend_from_slice(&(start + index).to_le_bytes());
    }
    Ok(loader.constant(
        name,
        DType::I32,
        Shape::new(&[u64::from(count)]),
        &bytes,
        "minimax-h3",
    )?)
}

/// The rotary frequency table of the decoder: `theta^(-2i/rotated)` over the rotated width.
/// The decoder's inverse frequencies, in the order its factors are laid out.
///
/// One block per axis, each over that axis's own width: the reference builds its embedding with
/// `head_dim · ratio / 3` and reuses those frequencies on all three axes (`video_vae.py:547-549`,
/// `uniserve/nn/rope.py:355-380`), so axis `a` takes `theta^(-2k / sections[a])` rather than a slice
/// of one wider spectrum.
///
/// This is the table the decoder publishes as `video_vae.rotary.{rotated}.{theta}` and the one a
/// factors comparison reads: a wrong spectrum or a wrong axis order is a phase drift that no shape,
/// element type or key check can see, so the numbers have to be comparable without a device
/// (`examples/vae_rope.rs`, `tools/compare_rope_factors.py --vae`).
pub fn inverse_frequencies(config: &VideoVaeConfig) -> Result<Vec<f32>> {
    let (rotated, sections) = rotary(config)?;
    let mut table = Vec::new();
    for width in &sections {
        for pair in 0..width / 2 {
            let exponent = 2.0 * f64::from(pair) / f64::from(*width);
            table.push(f64::from(config.decoder_rope_theta).powf(-exponent) as f32);
        }
    }
    if table.len() as u32 != rotated / 2 {
        return Err(Error::config(format!(
            "the rotary table of {rotated} channels holds {} frequencies, {} expected",
            table.len(),
            rotated / 2
        )));
    }
    Ok(table)
}

fn frequencies(
    loader: &mut WeightLoader,
    declared_sizes: &[(String, u64)],
    config: &VideoVaeConfig,
) -> Result<Weight> {
    let (rotated, _) = rotary(config)?;
    let table = inverse_frequencies(config)?;
    let mut bytes = Vec::with_capacity(table.len() * 4);
    for value in &table {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let name = format!("video_vae.rotary.{rotated}.{}", config.decoder_rope_theta);
    declared(declared_sizes, &name, bytes.len())?;
    Ok(loader.constant(
        &name,
        DType::F32,
        Shape::new(&[table.len() as u64]),
        &bytes,
        "minimax-h3",
    )?)
}

impl VideoDecoder {
    /// The tables one layout's decode reads, in the order the table walk writes them.
    pub fn tables(&self, latent_height: u32, latent_width: u32) -> Result<Vec<StateTable>> {
        table_specs(&self.config, latent_height, latent_width)
    }

    /// Write the host bytes of each table of [`Self::tables`], in the same order.
    pub fn table_bytes(
        &self,
        latent_height: u32,
        latent_width: u32,
        out: &mut [Vec<u8>],
    ) -> Result<()> {
        table_bytes_into(&self.config, latent_height, latent_width, out)
    }



    /// The rasters this decoder serves: the deployment's canvases, every one of them.
    ///
    /// The constants a decode reads are a **tile's** (its register and zero-row offsets, its sequence
    /// offsets) or a **width's** (the zero row, the rotary frequencies), so they are the same for every
    /// canvas this deployment names: serving another raster costs a lookup, not a table.
    pub fn canvases(&self) -> &[Canvas] {
        &self.canvases
    }


    /// The largest of this clip's decode tiles: every tile of a served grid has its rows.
    pub fn largest_tile(&self, canvas: &Canvas) -> Result<Canvas> {
        let grid = Tiles::of(&self.config, canvas)?.tiles(&self.config, canvas);
        grid.into_iter()
            .map(|tile| tile.canvas)
            .max_by_key(|tile| (tile.voxels(), tile.height, tile.width))
            .ok_or_else(|| {
                Error::config(format!(
                    "a canvas of {} x {} latent cells has no tile to decode",
                    canvas.height, canvas.width
                ))
            })
    }

    /// The transformer's scratch for one clip: every tile's rows at once, because every tile runs in
    /// the same launches. [`crate::decoding::video_arena`] states the same buffers.
    pub fn scratch(&self, f: &Forward, canvas: &Canvas) -> Result<ClipScratch> {
        let tile = self.largest_tile(canvas)?;
        let tiles = tile_count(&self.config, canvas)?;
        let rows = tiles * tile.rows();
        let voxels = tiles * tile.voxels();
        let width = u64::from(self.width);
        let ffn = width * u64::from(self.ffn_mult);
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
            gate_up: arena.tensor(ACTIVATION, Shape::new(&[rows, 2 * ffn]))?,
            gated: arena.tensor(ACTIVATION, Shape::new(&[rows, ffn]))?,
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
        let tile = self.largest_tile(canvas)?;
        let tiles = tile_count(&self.config, canvas)?;
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
        ScaledAdd.forward(f, scratch.rounded, scratch.restored, 1.0, 0.0)?;
        self.post_quant
            .forward(f, scratch.rounded, scratch.quantized)?;
        self.post_quant_bias
            .forward(f, scratch.quantized, scratch.rounded)?;
        self.input.forward(f, scratch.rounded, scratch.projected)?;

        // A tile's sequence is its voxels in the order they arrive, then the register rows, then the
        // zero row (`video_vae.py`, `Transformer.forward`: `cat((hidden, registers, zeros))`), so a
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
            ScaledAdd.forward(f, voxel_rows, projected, 1.0, 0.0)?;
            self.place.replace(
                f,
                tile_stream,
                self.registers.view(),
                self.register_rows.view(),
            )?;
            self.place
                .replace(f, tile_stream, self.zero_row.view(), self.zero_rows.view())?;
        }

        let (cosine, sine) = (scratch.cosine, scratch.sine);
        self.rotary.factors(f, positions, cosine, sine)?;

        let (head, heads) = (self.head, self.heads);
        let normalized = scratch.normalized;
        let (attended, update) = (scratch.attended, scratch.update);
        let (gate_up, gated) = (scratch.gate_up, scratch.gated);
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

        self.first.forward(f, stream, normalized)?;
        for layer in &self.layers {
            layer.query_key_value.forward(f, normalized, merged)?;
            self.rotary
                .apply_strided(f, projection(0), projection(1), cosine, sine)?;
            self.attention.attend_packed(
                f,
                projection(0),
                projection(1),
                projection(2),
                offsets,
                bound,
                sequence,
                per_head(attended),
            )?;
            layer.output.forward(f, attended, update)?;
            layer
                .attended
                .forward(f, stream, update, normalized, None)?;
            layer.gate_up.forward(f, normalized, gate_up)?;
            layer.gated.forward(f, gate_up, gated, None)?;
            layer.down.forward(f, gated, update)?;
            if let Some(fed) = &layer.fed {
                fed.forward(f, stream, update, normalized, None)?;
            }
        }

        self.last.forward(f, stream, update, normalized, None)?;
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
    gate_up: TensorView,
    gated: TensorView,
}

impl Decoder for VideoDecoder {
    /// The voxels of every tile, in the order `decode_clip` reads them: each tile's own latent slice,
    /// frame by frame, so a voxel in an overlap has one row per tile that holds it.
    fn latent_rows(&self, layout: &nn::capability::Layout) -> nn::Result<Option<Vec<u32>>> {
        let refuse = |problem: String| nn::Error::Shape {
            owner: "VideoDecoder",
            problem,
        };
        let clip = Canvas::clip(&self.config, layout.video.height, layout.video.width);
        let tiles = Tiles::of(&self.config, &clip)
            .map_err(|error| refuse(format!("the clip's tile grid: {error}")))?
            .tiles(&self.config, &clip);
        let (height, width) = (layout.video.height, layout.video.width);
        let mut rows = Vec::new();
        for tile in &tiles {
            for frame in 0..layout.video.frames {
                for row in tile.latent_rows.0..tile.latent_rows.1 {
                    for column in tile.latent_columns.0..tile.latent_columns.1 {
                        rows.push((frame * height + row) * width + column);
                    }
                }
            }
        }
        Ok(Some(rows))
    }

    fn layout_tables(&self, layout: &nn::capability::Layout) -> nn::Result<Vec<StateTable>> {
        // A layout whose extents contradict the checkpoint is a named error and not an empty list: a
        // runner that made a mistake hears which one, where an empty answer would have reached the
        // decode as a block with no tables and read nothing.
        self.tables(layout.video.height, layout.video.width)
            .map_err(|error| nn::Error::Shape {
                owner: "VideoDecoder",
                problem: error.to_string(),
            })
    }

    fn prepare_layout(
        &self,
        layout: &nn::capability::Layout,
        out: &mut [Vec<u8>],
    ) -> nn::Result<()> {
        self.table_bytes(layout.video.height, layout.video.width, out)
            .map_err(|error| nn::Error::Shape {
                owner: "VideoDecoder",
                problem: error.to_string(),
            })
    }

    /// The checkpoint's clip schedule for a timeline of `frames` frames.
    ///
    /// The schedule is `17 * n + 5` whatever block is asked: it depends on the checkpoint's clip body
    /// and overlap frame counts, which the decoder holds as well as the postprocessor does. So both
    /// answer through [`VideoPostprocessor::splices_of`], and a runner that bound only a decoder - a
    /// checkpoint with no postprocessor publishes the clip's own raster - still gets the timeline's
    /// unit list rather than the trait's refusal.
    fn splices(&self, frames: u32) -> nn::Result<Vec<nn::capability::Splice>> {
        VideoPostprocessor::splices_of(frames)
    }

    /// Clip `k` decodes latent frames `[5k, 5k + 7)`: five that advance the timeline, and two the
    /// next clip's window also reads.
    fn clip_start(&self, clip: u32) -> u32 {
        clip * crate::CLIP_LATENTS
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
        let refuse = |problem: String| nn::Error::Shape {
            owner: "VideoDecoder",
            problem,
        };
        let channels = u64::from(self.config.latent_channels);
        let (height, width) = (
            u64::from(layout.video.height),
            u64::from(layout.video.width),
        );
        let plane = height * width;
        let voxels = trajectory.numel() / channels;
        let clip = Canvas::clip(&self.config, layout.video.height, layout.video.width);
        let frames = u64::from(clip.frames);
        if trajectory.numel() % channels != 0
            || (u64::from(first) + frames) * plane > voxels
            || trajectory.dtype != STREAM
            || block.dtype != STREAM
        {
            return Err(refuse(format!(
                "a clip of {frames} latent frames from frame {first} of a {plane}-voxel raster is \
                 past a trajectory of {} {} elements in {channels} channels",
                trajectory.numel(),
                trajectory.dtype
            )));
        }
        let tiles = Tiles::of(&self.config, &clip)
            .map_err(|error| refuse(format!("the clip's tile grid: {error}")))?
            .tiles(&self.config, &clip);
        let copy = nn::layers::CopyBlock::new(STREAM)?;
        let mut row = 0_u64;
        for tile in &tiles {
            let (top, bottom) = tile.latent_rows;
            let (left, right) = tile.latent_columns;
            let (rows, columns) = (u64::from(bottom - top), u64::from(right - left));
            let shape = Shape::new(&[channels, frames, rows, columns]);
            let start = u64::from(first) * plane + u64::from(top) * width + u64::from(left);
            let source = TensorView::with_strides(
                trajectory.offset_elements(start),
                STREAM,
                shape,
                nn::Strides::new(&[voxels, plane, width, 1]),
            );
            let strides = [1, rows * columns * channels, columns * channels, channels];
            let destination = TensorView::with_strides(
                block.offset_elements(row * channels),
                STREAM,
                shape,
                nn::Strides::new(&strides),
            );
            copy.forward(f, source, destination, &strides.map(|stride| stride as i64))?;
            row += frames * rows * columns;
        }
        Ok(())
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
        let refuse = |problem: String| nn::Error::Shape {
            owner: "VideoDecoder",
            problem,
        };
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

    /// The bytes one decode of `layout` needs, from this block's own plan.
    ///
    /// The layout states the canvas's latent extents, and the block decodes one canvas, so a layout
    /// naming another raster is refused by name - the same mismatch `layout_of` refuses from its side,
    /// reached here by a caller that built the layout itself. The plan is `video_arena`'s for that
    /// canvas at this decoder's element type, so the number and the plan a runner allocates from are
    /// one statement rather than two.
    fn arena_bytes(&self, layout: &nn::capability::Layout) -> nn::Result<u64> {
        let canvas = Canvas::clip(&self.config, layout.video.height, layout.video.width);
        if !self.canvases.contains(&canvas) {
            return Err(nn::Error::Shape {
                owner: "VideoDecoder",
                problem: format!(
                    "this decoder serves {} and the layout states {}x{} latent cells",
                    self.canvases
                        .iter()
                        .map(|canvas| format!("{}x{}", canvas.width, canvas.height))
                        .collect::<Vec<_>>()
                        .join(", "),
                    layout.video.width,
                    layout.video.height,
                ),
            });
        }
        let plan = crate::decoding::video_arena(&self.config, &canvas).map_err(|error| {
            nn::Error::Shape {
                owner: "VideoDecoder",
                problem: format!("this decoder's plan for its canvas: {error}"),
            }
        })?;
        Ok(crate::decoding::arena_reservation(&plan))
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
    /// clip* ([`Self::scratch`], `decoding::video_arena`), so two decodes in flight on one decoder object
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
        let refuse = |problem: String| nn::Error::Shape {
            owner: "VideoDecoder",
            problem,
        };
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
        let channels = u64::from(crate::output::Rgb::CHANNELS);
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
        let tiles = Tiles::of(&self.config, &clip)
            .map_err(|error| refuse(format!("the clip's tile grid: {error}")))?
            .tiles(&self.config, &clip);
        self.stitch
            .decode_clip(f, self, &tiles, latents, *positions, *stitch, video)
            .map_err(|error| refuse(format!("{error}")))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A tile's sequence is its voxels in the order they arrive, then the trailing rows
    /// (`video_vae.py`: `cat((hidden, registers, zeros))`), so a voxel's token row is its own index
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
        let tile = largest_tile(&config, canvas.height, canvas.width).expect("a tile");
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
                "gemm",       // the restore
                "scaled_add", // the round down
                "gemm",       // post_quant_conv's product
                "gemm",       // its bias, a rounding of its own
                "gemm",       // the input projection
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

    /// A clip's transformer scratch holds every tile's rows, and the plan states the same: the two must
    /// agree, or a device run stops inside the arena naming a buffer while the plan is the thing that
    /// is wrong.
    #[test]
    fn a_clips_scratch_fits_the_plan_its_canvas_states() {
        let config = video_config();
        // A canvas one tile wide and two tall, so a tile's row count and the clip's differ: on a
        // single-tile canvas the two are the same number and this test would see nothing.
        let canvas = Canvas::clip(&config, 3 * config.spatial_compression(), 17);
        let grid = Tiles::of(&config, &canvas)
            .expect("a grid")
            .tiles(&config, &canvas);
        assert!(
            grid.len() > 1,
            "the canvas must need more than one tile, or this test proves nothing: {} tile(s)",
            grid.len()
        );
        assert!(
            grid[0].canvas.voxels() < canvas.voxels(),
            "and a tile must be smaller than the clip: {} against {}",
            grid[0].canvas.voxels(),
            canvas.voxels()
        );

        let decoder = stated_decoder(&config, canvas);
        let plan = crate::decoding::video_arena(&config, &canvas).expect("a plan");
        let need = crate::decoding::arena_reservation(&plan);
        // Exactly the plan and not one byte more: the scratch a clip reserves has to fit inside the
        // number its decoder states to the runner.
        let arena = nn::Arena::new(nn::DevicePtr::new(0), need);
        let forward = nn::Forward::recording(&arena);
        decoder
            .scratch(&forward, &canvas)
            .expect("one clip's transformer scratch fits the plan its decoder states");
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
        let plan = crate::decoding::video_arena(&config, &canvas).expect("a plan");
        let arena = nn::Arena::new(
            nn::DevicePtr::new(0),
            crate::decoding::arena_reservation(&plan),
        );
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
            matches!(&error, nn::Error::Shape { problem, .. }
                if problem.contains(&format!("{:?}", DType::F16))
                    && problem.contains(&format!("{:?}", DType::F32))),
            "the refusal names both types: {error}"
        );
    }

    /// A decoder of one canvas, built from stated weights: the geometry is the subject, so neither a
    /// checkpoint nor a device is needed.
    fn stated_decoder(config: &VideoVaeConfig, canvas: Canvas) -> VideoDecoder {
        let mut loader = WeightLoader::stated(DType::F32, None);
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

        // The arena is the canvas's own plan: a wider raster needs more bytes, and this decoder
        // refuses the other's layout by name rather than answering with a number for its own raster.
        let bytes = decoder
            .arena_bytes(&layout)
            .expect("the plan of its own canvas");
        let wider_canvas = Canvas::clip(&config, 2 * ratio, 3 * ratio);
        let wider_decoder = stated_decoder(&config, wider_canvas);
        let wider_layout = wider_decoder
            .layout_of(clip_of(&config, wider_canvas))
            .expect("the wider decoder's own canvas");
        let wider_bytes = wider_decoder
            .arena_bytes(&wider_layout)
            .expect("the wider canvas's plan");
        assert!(
            bytes > 0 && wider_bytes > bytes,
            "the wider canvas states {wider_bytes} B, not more than the {bytes} B of the narrower one"
        );
        let error = decoder
            .arena_bytes(&wider_layout)
            .expect_err("the layout of another canvas");
        assert!(
            error.to_string().contains("latent cells"),
            "the refusal names the canvas mismatch: {error}"
        );
        assert!(
            matches!(error, nn::Error::Shape { .. }),
            "a layout this decoder cannot serve is a shape refusal and not the `Unsupported` a \
             block with no plan at all states: {error:?}"
        );
    }

    fn seams(starts: &[u32], overlaps: &[u32], length: u32) -> Seams {
        Seams {
            starts: starts.to_vec(),
            lengths: vec![TILE_PIXELS; starts.len()],
            overlaps: overlaps.to_vec(),
        }
        .with_length(length)
    }

    impl Seams {
        fn with_length(mut self, length: u32) -> Self {
            if self.starts.len() == 1 {
                self.lengths = vec![length];
            }
            self
        }
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
        let tile = largest_tile(&config, raster.height, raster.width).expect("a tile");
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
        let mut vertical = vec![Vec::new(), Vec::new()];
        let mut wide = vec![Vec::new(), Vec::new()];
        table_bytes_into(&config, 48, 84, &mut vertical).expect("the vertical canvas");
        table_bytes_into(&config, 42, 96, &mut wide).expect("the wide canvas");
        assert_eq!(vertical[0].len(), wide[0].len());
        let tables = table_specs(&config, 48, 84).expect("the one table");
        // The buffer is the descriptor's own expression: elements times the element's size. Written as
        // a number it was `5,391` elements against `21,564` bytes - the same table in two units.
        let expected = tables[0].shape.numel() * u64::from(tables[0].dtype.bits()) / 8;
        assert_eq!(vertical[0].len() as u64, expected);
    }

    #[test]
    fn a_layout_with_the_wrong_number_of_tables_is_refused() {
        // The worker sizes `out` from `table_specs`, so a count that disagrees with the block is a
        // protocol error and must surface as one rather than as a partially written table.
        let config = video_config();
        let mut none: Vec<Vec<u8>> = Vec::new();
        assert!(table_bytes_into(&config, 48, 84, &mut none).is_err());
        let mut three = vec![Vec::new(), Vec::new(), Vec::new()];
        assert!(table_bytes_into(&config, 48, 84, &mut three).is_err());
    }

    #[test]
    fn a_raster_that_fits_one_tile_is_not_split() {
        let split = Seams::split(256, TILE_PIXELS, MINIMUM_OVERLAP, 16).expect("one tile");
        assert_eq!(split.starts, vec![0]);
        assert_eq!(split.lengths, vec![256]);
        assert!(split.overlaps.is_empty());
    }

    #[test]
    fn the_768p_16_by_9_canvas_splits_as_the_reference_does() {
        // 1344 × 768 pixels: seven columns and four rows, the slack spread over the first seams.
        let width = Seams::split(1344, TILE_PIXELS, MINIMUM_OVERLAP, 16).expect("columns");
        assert_eq!(
            width,
            seams(
                &[0, 176, 352, 528, 704, 896, 1088],
                &[80, 80, 80, 80, 64, 64],
                1344
            )
        );
        let height = Seams::split(768, TILE_PIXELS, MINIMUM_OVERLAP, 16).expect("rows");
        assert_eq!(height, seams(&[0, 160, 336, 512], &[96, 80, 80], 768));
    }

    #[test]
    fn the_last_tile_ends_on_the_raster() {
        for canvas in [480u32, 640, 672, 768, 832, 1024, 1344, 1536] {
            let split = Seams::split(canvas, TILE_PIXELS, MINIMUM_OVERLAP, 16).expect("split");
            let last = split.starts[split.len() - 1];
            assert!(
                last < canvas && last + split.lengths[split.len() - 1] >= canvas,
                "the last of {canvas} starts at {last}"
            );
            // The whole raster is covered: each start is the previous end less its overlap.
            for index in 1..split.len() {
                let previous_end = split.starts[index - 1] + split.lengths[index - 1];
                assert_eq!(
                    split.starts[index] + split.overlaps[index - 1],
                    previous_end
                );
            }
        }
    }

    #[test]
    fn an_unaligned_raster_is_refused() {
        assert!(Seams::split(1000, TILE_PIXELS, MINIMUM_OVERLAP, 16).is_err());
        assert!(Seams::split(1024, TILE_PIXELS, TILE_PIXELS, 16).is_err());
        assert!(Seams::split(0, TILE_PIXELS, MINIMUM_OVERLAP, 16).is_err());
    }

    #[test]
    fn a_tiles_latent_span_is_its_pixels_over_the_compression() {
        let config = video_config();
        let canvas = Canvas::clip(&config, 48, 84);
        assert_eq!(canvas.frames, 7);
        assert_eq!(canvas.voxels(), 7 * 48 * 84);
        assert_eq!(canvas.rows(), 7 * 48 * 84 + 4 + 1);
        let tiles = Tiles::of(&config, &canvas).expect("tiles");
        let planned = tiles.tiles(&config, &canvas);
        assert_eq!(planned.len(), 4 * 7);
        let first = planned[0];
        assert_eq!(first.latent_rows, (0, 16));
        assert_eq!(first.latent_columns, (0, 16));
        assert_eq!((first.above, first.left), (0, 0));
        assert_eq!((first.below, first.right), (96, 80));
        // The second column of the first row cross-fades 80 pixels of its predecessor: the width's
        // slack of 64 pixels is four alignment steps over six seams, so the first four seams carry
        // 80 pixels and the last two the minimum 64, and the second tile starts 176 pixels in.
        assert_eq!(planned[1].left, 80);
        assert_eq!(planned[1].latent_columns, (176 / 16, (176 + 256) / 16));
        assert_eq!(planned[5].left, 64);
        // A tile that has a successor below trims its trailing rows; the last row does not.
        let below = planned[7];
        assert_eq!(below.row, 1);
        assert_eq!(below.above, 96);
        assert_eq!(planned[planned.len() - 1].below, 0);
    }

    #[test]
    fn every_tile_of_a_canvas_has_a_canvas_of_its_own() {
        let config = video_config();
        let canvas = Canvas::clip(&config, 48, 52);
        let tiles = Tiles::of(&config, &canvas).expect("tiles");
        for tile in tiles.tiles(&config, &canvas) {
            assert!(tile.latent_rows.1 <= canvas.height);
            assert!(tile.latent_columns.1 <= canvas.width);
            assert!(tile.latent_rows.0 < tile.latent_rows.1);
            assert!(tile.latent_columns.0 < tile.latent_columns.1);
            assert_eq!(
                tile.canvas.rows(),
                u64::from(tile.canvas.frames)
                    * u64::from(tile.latent_rows.1 - tile.latent_rows.0)
                    * u64::from(tile.latent_columns.1 - tile.latent_columns.0)
                    + u64::from(tile.canvas.registers)
                    + u64::from(tile.canvas.tail)
            );
        }
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
        assert_eq!(decode_latents(&config), 7);
        assert_eq!(config.temporal_compression(), 4);
        // The decode emits 28 frames and the splice keeps 25 of them: 25 = 17 + 3 + 5.
        let native = config.clip_length + config.token_drop + crate::decoding::OVERLAP_FRAMES;
        assert_eq!(native, 25);
        assert_eq!(
            decode_latents(&config) * config.temporal_compression() - native,
            3
        );
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

    #[test]
    fn the_video_decoders_keys_ask_the_export_and_nothing_else() {
        let config = video_config();
        let keys = VideoDecoder::keys(&config).expect("the decoder's keys");
        // The export's own number: `vae/` holds 703 names, the decoder's 585 are the 583 under
        // `decoder.` plus the two `post_quant_conv.` tensors, and the encoder reads the remaining 118.
        // A checkpoint that lacks one is refused by block and name.
        assert_eq!(keys.len(), 585);
        let mut unique = keys.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), keys.len(), "no tensor is named twice");
        assert!(
            keys.iter()
                .any(|key| key == "decoder.transformer_blocks.35.ff.net.0.proj.weight"),
            "the last layer's projection is asked for"
        );
        assert!(
            keys.iter().any(|key| key == "post_quant_conv.weight"),
            "the post-quantization convolution's tensor is asked for"
        );
        let encoder = crate::encoding::VideoEncoder::keys(&config).expect("the encoder's keys");
        assert!(
            encoder.iter().all(|key| !keys.contains(key)),
            "the two halves read disjoint tensors, or one block would be binding another's weight"
        );
        assert_eq!(
            keys.len() + encoder.len(),
            703,
            "together they are every name in `vae/`, with none unread"
        );
    }

    #[test]
    fn the_weight_table_names_every_parameter_of_the_checkpoint() {
        let table = sources(&video_config()).expect("the decoder's parameters");
        // 36 layers of sixteen parameters, plus the post-quantization pair, the input pair, the
        // register table, and the head's four.
        assert_eq!(table.len(), 36 * 16 + 5 + 4);
        let names: Vec<&str> = table
            .iter()
            .map(|source| source.parameter.as_str())
            .collect();
        assert!(names.contains(&"post_quant.weight"));
        assert!(names.contains(&"registers"));
        assert!(names.contains(&"layers.0.norm1.weight"));
        assert!(names.contains(&"layers.35.gate_up.bias"));
        assert!(names.contains(&"norm.weight"));
        assert!(names.contains(&"output.weight"));
        let checkpoint: Vec<String> = table
            .iter()
            .map(|source| match &source.tensor {
                crate::audio_vae::Tensor::Region(tensor) => tensor.clone(),
                crate::audio_vae::Tensor::Normalized { .. } => {
                    panic!("the video decoder folds no weight norm")
                }
            })
            .collect();
        assert!(
            checkpoint.contains(&"decoder.transformer_blocks.35.ff.net.0.proj.weight".to_owned())
        );
        assert!(checkpoint.contains(&"decoder.proj_out.bias".to_owned()));
        // No parameter and no checkpoint tensor is named twice.
        for list in [names, checkpoint.iter().map(String::as_str).collect()] {
            let mut unique = list.clone();
            unique.sort_unstable();
            unique.dedup();
            assert_eq!(unique.len(), list.len());
        }
    }

    /// The dense checkpoint's video VAE config: the sidecar's numbers (specs/worker.md 9, `vae/config.json`).
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

    /// The decoder's programs follow from its configuration: one key per operator, because nothing
    /// that varies with a launch is in a key.
    #[test]
    fn the_decoder_states_its_programs_from_its_own_config() {
        let config = video_config();
        let programs = VideoDecoder::programs(&config).expect("the decoder's programs");
        let count = |operator: &str| programs.iter().filter(|(name, _)| name == operator).count();

        // The contraction length is a descriptor field of one program, so the projections are three
        // keys: the FP32 restore, every biased FP16 product, and `post_quant_conv`'s bias-free one.
        assert_eq!(count("gemm"), 3);
        assert_eq!(count("rows"), 1, "two row writes, one program");
        assert_eq!(count("rotary_factors"), 1);
        assert_eq!(count("qk_norm_rope"), 1);
        assert_eq!(count("weighted_rms_norm"), 1);
        assert_eq!(count("scaled_residual_rms_norm"), 1);
        assert_eq!(count("scaled_residual_layer_norm"), 1);
        assert_eq!(count("value_first_swiglu"), 1);
        assert_eq!(
            count("scaled_add"),
            2,
            "the two crossings of the stream's type"
        );
        assert_eq!(count("tile_stitch"), 1, "the whole output path");
        assert_eq!(
            count("copy_block"),
            2,
            "the gather out of the trajectory and the tail a clip leaves"
        );

        // One attention whatever the clip: the sequences and their rows are launch values, so the
        // decoder's programs do not depend on the raster a request asks for.
        assert_eq!(count("dense_attention"), 1, "{programs:?}");
    }

    /// `specs/worker.md`'s table replaces `torch.autocast` with a declaration per layer, and these rows
    /// are that declaration: the stream and everything that reads or writes it wide says so in its key,
    /// and everything else is the activation's type. A row in a type nobody declared would be a layer
    /// silently computing in it, which is the behaviour the design removed.
    #[test]
    fn the_rows_state_the_precision_the_module_declares() {
        let config = video_config();
        let rows = VideoDecoder::programs(&config).expect("the decoder's programs");
        let key = |operator: &str, part: &str| {
            rows.iter()
                .find(|(name, key)| name == operator && key.contains(part))
                .map(|(_, key)| key.clone())
                .unwrap_or_else(|| panic!("no {operator} row holds {part}: {rows:?}"))
        };
        // The normalizations read the wide stream and write the narrow rows.
        for operator in [
            "weighted_rms_norm",
            "scaled_residual_rms_norm",
            "scaled_residual_layer_norm",
        ] {
            let row = key(operator, "dtype_f32");
            assert!(row.ends_with("activation_f16"), "{row}");
        }
        // The projections are FP16 with FP32 accumulation, and only the restore is wide.
        key("gemm", "input_f16_weight_f16_accumulate_f32_output_f16");
        key("gemm", "input_f32_weight_f32_accumulate_f32_output_f32");
        // The two crossings, and the rotary factors in the rows' type.
        key("scaled_add", "target_f16_source_f32");
        key("scaled_add", "target_f32_source_f16");
        key("rotary_factors", "dtype_f16_positions_f32");
        let rotation = key("qk_norm_rope", "dtype_f16_output_f16");
        assert!(rotation.contains("factors_f16"), "{rotation}");
        assert!(rotation.ends_with("strided_true"), "{rotation}");
        // The attention is the packed FlashAttention form, which names no offsets in its key.
        let attention = key("dense_attention", "dtype_f16_kv_f16_out_f16");
        assert!(
            attention.ends_with("mask_dense_window_false"),
            "{attention}"
        );
        key("value_first_swiglu", "dtype_f16");
        key("tile_stitch", "dtype_f16");
    }
}
