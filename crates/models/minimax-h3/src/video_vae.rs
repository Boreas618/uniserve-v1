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
//! * the per-layer composition and the element type it declares (specs/worker.md 9: `torch.autocast`
//!   is replaced by a declaration, not by an implicit conversion).
//!
//! ## Precision
//!
//! The decoder declares one element type, read from the checkpoint's own projection rather than stated
//! here (`dtype = "auto"` converts nothing): every parameter it reads, the register table, the zero row
//! and every activation and scratch buffer hold it. The reference instead keeps the normalization
//! parameters and the channel scales in FP32 and promotes the residual stream to FP32, which `torch`'s
//! autocast does not reach; the reference measures that family of difference at a **relative L2 of
//! 1.007e-3 against its own float32 decode** (one rounding in the gating) and 1.015e-3 with one
//! rounding per operation, on a seeded seven-latent-frame clip at 768x1344 tiled as the checkpoint
//! specifies (`uniserve/models/minimax_h3/video_vae.py:438-445`). That is the number this declaration
//! is measured against, and quality is confirmed by viewing (specs/worker.md 10), not by bit equality.
//!
//! Three of the reference's operators before the transformer - the latent denormalisation, the
//! one-by-one post-quantization convolution and the input projection - are all affine over the latent
//! channel axis with no nonlinearity between them, so they are composed into **one** affine at the
//! loading boundary ([`folded_input`]). It is the only place in this module where two or more reference
//! operators become one, and the block-diagonal numeric comparison with today's implementation records
//! it as a declared composition rather than a difference (design 10: the comparison is a diagnosis,
//! never an acceptance gate).

use crate::config::AttentionKind;
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

    /// Whether the axis has no tile, which only an empty extent would produce.
    pub fn is_empty(&self) -> bool {
        self.starts.is_empty()
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
    Attention, LayerNorm, Linear, PackedSwiGlu, RmsNorm, Rotary, Rows, ScaledResidual, Sections,
};
use nn::weights::{Weight, WeightLoader, WeightRequest};
use nn::{AttentionBackend, DType, DecodedMedia, Decoder, Shape, StateTable, TensorView};
use ops::gemm::WeightOrder;

/// One transformer layer of the decoder: the attention half and the gated feed-forward half, each
/// with its own normalization and its own per-channel residual scale.
struct Layer {
    norm1: RmsNorm,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    /// The scaled anchor of the attention residual, carrying the projection's bias.
    residual1: ScaledResidual,
    norm2: RmsNorm,
    gate_up: Linear,
    down: Linear,
    /// The scaled anchor of the feed-forward residual, carrying the projection's bias.
    residual2: ScaledResidual,
    gated: PackedSwiGlu,
}

/// The video decoder's latent stage: one clip's latent voxels become the clip's pixels.
///
/// The decoder reads a clip's latents `[frames, height, width]` of [`Canvas`] (seven latent frames of
/// 24 channels for the checkpoint), runs the 36-layer transformer over the raster's voxel rows plus the
/// learned register rows and one zero row, and writes one 3×4×16×16 pixel block per voxel. The
/// per-head normalization before the rotation has no learned weight — the checkpoint stores none,
/// which is what `Rotary::apply`'s absent gains express.
pub struct VideoDecoder {
    canvas: Canvas,
    /// The tile stitch: the unpatchify and one compiled blend per seam width, built at load so that no
    /// layer is built on a request's path.
    stitch: crate::decoding::ClipDecoder,
    width: u32,
    input: Linear,
    registers: Weight,
    register_rows: Weight,
    zero_row: Weight,
    zero_rows: Weight,
    layers: Vec<Layer>,
    norm: LayerNorm,
    output: Linear,
    rotary: Rotary,
    place: Rows,
    /// The index table that places each voxel row of the stream: for every latent voxel of one clip, the
    /// packed row of the patch it lies in, under **this deployment's** packing. A sparse deployment orders
    /// its rows by spatiotemporal tile, so the table differs from the dense one and is the model's own
    /// constant, published at load by [`pack_voxel_rows`].
    voxel_rows: Weight,
    /// The offsets of the decoder's **one** sequence: `[0, rows]`, a constant of the canvas rather than
    /// a checkpoint tensor. The dense attention's flash backend serves the packed form only — it hands
    /// the vendor's `cu_seqlens` path its offsets and refuses a null table — so one clip is the packed
    /// form of one sequence and states its offsets.
    offsets: Weight,
    config: VideoVaeConfig,
    ffn_mult: u32,
    /// Elements per head.
    head: u64,
    /// Query heads per token.
    heads: u64,
    attention: Attention,
    /// The checkpoint's own element type for this decoder: the type its weights are stored in and the
    /// type every activation and scratch buffer holds. The design's `dtype = "auto"` converts nothing,
    /// so this is read from the checkpoint rather than stated.
    dtype: DType,
}

/// The packed row of every latent voxel one clip's decode reads, in clip-then-raster order.
///
/// One clip's voxel rows: for every voxel of the clip's latent raster, the packed row it belongs to.
///
/// The decoder has one token per **latent voxel** - the VAE's patch is `1 x 2 x 2` latent cells and the
/// transformer reads one token per cell - so this table is four times longer than the denoiser's row
/// table: each voxel names the packed row of the `2 x 2` patch it lies in. It describes **one clip**,
/// because that is what the checkpoint's `video_vae.voxel_rows` holds and all a decode ever reads; how
/// many clips a request has is the runner's question and the splice's, not this table's.
///
/// The checkpoint holds no such table — its `video_vae` tensors are the weights and the state tables —
/// so the model publishes this one as a constant of the canvas at load ([`pack_voxel_rows`]), and it is
/// the arithmetic below rather than a stored tensor that decides what the table says.
fn voxel_rows(
    config: &VideoVaeConfig,
    attention: AttentionKind,
    latent_height: u32,
    latent_width: u32,
) -> Result<Vec<u32>> {
    let raster = crate::packing::LatentRaster {
        height: latent_height,
        width: latent_width,
    };
    // A clip's own window of latent frames, which is the timeline of exactly one clip.
    let clip = Canvas::clip(config, latent_height, latent_width);
    let order = crate::packing::video_order_of(attention, clip.frames, raster)?;
    let mut packed_of_raster = vec![0_u32; order.len()];
    for (packed, raster_row) in order.iter().enumerate() {
        packed_of_raster[*raster_row as usize] = u32::try_from(packed)
            .map_err(|_| Error::config("a packed row wider than a table entry".to_owned()))?;
    }
    let tokens = u64::from(raster.patch_rows());
    let side = u64::from(crate::packing::PATCH_SIDE);
    let columns = u64::from(latent_width) / side;
    let mut rows = Vec::with_capacity(clip.voxels() as usize);
    for frame in 0..u64::from(clip.frames) {
        for height in 0..u64::from(latent_height) {
            for column in 0..u64::from(latent_width) {
                let patch = frame * tokens + (height / side) * columns + column / side;
                rows.push(packed_of_raster[patch as usize]);
            }
        }
    }
    Ok(rows)
}

/// Publish the one voxel-row table a deployment's decode reads, encoded as the gather's `i32` indices.
///
/// The packed order is the deployment's: dense packing reads the raster order, a sparse deployment reads
/// spatiotemporal tiles, so a decoder handed a sparsely packed latent cannot be read through the dense
/// table — the entries would name another voxel's row. One table per decoder, built from the packing this
/// decoder serves; [`voxel_rows`] is the arithmetic and this is only its encoding.
fn pack_voxel_rows(
    loader: &mut WeightLoader,
    config: &VideoVaeConfig,
    attention: AttentionKind,
    canvas: Canvas,
) -> Result<Weight> {
    let rows = voxel_rows(config, attention, canvas.height, canvas.width)?;
    if rows.len() as u64 != canvas.voxels() {
        return Err(Error::config(format!(
            "the decoder's voxel table names {} rows for a clip of {} voxels",
            rows.len(),
            canvas.voxels()
        )));
    }
    let mut bytes = Vec::with_capacity(rows.len() * 4);
    for row in &rows {
        let row = i32::try_from(*row)
            .map_err(|_| Error::config("a packed row wider than an index".to_owned()))?;
        bytes.extend_from_slice(&row.to_le_bytes());
    }
    Ok(loader.constant(
        "video_vae.voxel_rows",
        DType::I32,
        Shape::new(&[rows.len() as u64]),
        &bytes,
        "minimax-h3",
    )?)
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

/// The one table a layout's decode reads, keyed by its latent extents.
///
/// The model's arithmetic, keyed by the **layout**: the rotary positions of the largest tile of the latent
/// raster, which every other tile's positions are a slice of because the tiles are uniform. There is one
/// table and not two: the clip's voxel rows are published once at load by [`pack_voxel_rows`] from the
/// deployment's own packing, so a per-layout copy of them would be a second source of truth for a value
/// that never varies with the layout.
pub(crate) fn table_specs(
    config: &VideoVaeConfig,
    latent_height: u32,
    latent_width: u32,
) -> Result<Vec<StateTable>> {
    let positions = u64::from(positions_table(config, latent_height, latent_width)?);
    Ok(vec![StateTable {
        name: "video_positions",
        dtype: DType::F32,
        shape: Shape::new(&[3, positions]),
    }])
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

/// The decoder's first affine, read from the checkpoint and composed at load.
///
/// Reads the three reference steps - the latent statistics, the one-by-one post-quantization
/// convolution and the projection onto the transformer's width - and answers the composed
/// `[width, latent_channels]` weight and `[width]` bias, encoded in `dtype`. See
/// [`fold_input_affine`] for the algebra; this only reads the checkpoint's own bytes, so nothing rounds
/// on the way in, and rounds once on the way out.
///
/// The answer is `(width, weight bytes, bias bytes)`. A checkpoint whose extents disagree with the
/// sidecar's latent channel count is refused here rather than read past.
pub fn folded_input(
    checkpoint: &nn::Checkpoint,
    config: &VideoVaeConfig,
) -> Result<(u64, DType, Vec<u8>, Vec<u8>)> {
    let latent = u64::from(config.latent_channels);
    let read = |name: &str| -> Result<nn::checkpoint::HostTensor> {
        checkpoint.read(name).map_err(|error| {
            Error::checkpoint(format!("the decoder's first affine reads {name}: {error}"))
        })
    };
    let projection = read("decoder.proj_in.weight")?;
    // The design's `dtype = "auto"` converts nothing, so the block computes in the type the checkpoint
    // stores its decoder in, which is the projection's: the reference reads `input.weight.dtype` for
    // exactly this.
    let dtype = projection.dtype;
    let projection_bias = read("decoder.proj_in.bias")?;
    let post_quant = read("post_quant_conv.weight")?;
    let post_quant_bias = read("post_quant_conv.bias")?;
    let width = projection.shape.dim(0);
    if projection.shape.rank() != 2 || projection.shape.dim(1) != latent {
        return Err(Error::checkpoint(format!(
            "the decoder projects {:?} onto {width} channels and the sidecar states {latent} latent \
             channels",
            projection.shape.dims()
        )));
    }
    for (name, tensor, expected) in [
        ("decoder.proj_in.bias", &projection_bias, width),
        ("post_quant_conv.weight", &post_quant, latent * latent),
        ("post_quant_conv.bias", &post_quant_bias, latent),
    ] {
        if tensor.shape.numel() != expected {
            return Err(Error::checkpoint(format!(
                "{name} holds {} elements, {expected} expected",
                tensor.shape.numel()
            )));
        }
    }
    let values = |tensor: &nn::checkpoint::HostTensor, name: &str| {
        crate::transformer::host_f32(tensor.dtype, &tensor.bytes, name)
    };
    let (weight, bias) = fold_input_affine(
        &values(&projection, "decoder.proj_in.weight")?,
        &values(&projection_bias, "decoder.proj_in.bias")?,
        &values(&post_quant, "post_quant_conv.weight")?,
        &values(&post_quant_bias, "post_quant_conv.bias")?,
        &config.latents_mean,
        &config.latents_std,
    )?;
    Ok((
        width,
        dtype,
        crate::transformer::host_bytes(dtype, &weight)?,
        crate::transformer::host_bytes(dtype, &bias)?,
    ))
}

/// The attention of the decoder's block, for the element type the clip is decoded at.
///
/// FlashAttention-4's program is a half-precision kernel and serves the packed form of one
/// sequence, the form that carries row offsets and refuses a null table; the FP32 program is a GEMM
/// and a softmax, serves one sequence without offsets, and its package declares exactly that cell.
/// The block is one sequence either way, so the element type alone decides which program is asked
/// for - and a clip decoded at f32 must not ask for a half-precision kernel.
fn block_attention(head: u32, heads: u32, rows: u32, dtype: DType) -> Result<Attention> {
    let scale = (head as f32).sqrt().recip();
    match dtype {
        DType::F32 => Ok(Attention::fp32(head, heads, heads, rows, false, scale)?),
        _ => Ok(Attention::dense(
            head, heads, heads, rows, false, None, scale,
        )?),
    }
}

/// Whether this attention serves the packed form: the one the offsets table selects.
///
/// The descriptor prints `cu_seq_lens` only for the form *without* offsets, so a launch and the walk
/// that states it must agree on this one predicate or the store builds a cell no launcher binds.
fn packed(attention: &Attention) -> bool {
    matches!(
        attention.backend(),
        AttentionBackend::Dense | AttentionBackend::Sparse
    )
}

impl VideoDecoder {
    /// Every checkpoint tensor the video decoder reads, by name.
    ///
    /// The decoder is a component whose names were already data ([`sources`] maps each parameter to
    /// the checkpoint key that fills it), so this list is that mapping's other column rather than a
    /// second spelling of it: the 585 names are the export's 583 `decoder.*` and the two
    /// `post_quant_conv.*` the input affine is folded from. A checkpoint that lacks one is refused by
    /// block and name before a byte is read.
    pub fn keys(config: &VideoVaeConfig) -> crate::Result<Vec<String>> {
        Ok(crate::audio_vae::tensor_keys(&sources(config)?))
    }

    /// The programs one clip's decode of `config` launches.
    ///
    /// Stated from the configuration alone, because a deployment compiles its store before it opens
    /// the checkpoint: the geometry is the config's and the element type is the checkpoint's (the
    /// design's `dtype = "auto"` converts nothing). **No extent enters a key**: a clip's token count is
    /// a launch value the dense attention and the normalizations read position by position, so one
    /// program serves every raster — which is why this takes no bucket set while the post-processor,
    /// whose blend is keyed by its pixel count, does. The bias each projection carries is its own (see
    /// the projections below) and every residual its projection's, as the checkpoint's mapping states.
    ///
    /// **The element type is the deployment's, and every layer declares it.** `specs/worker.md`'s table
    /// replaces `torch.autocast` with a per-layer declaration, and this is that mechanism in the
    /// decoder: `dtype` is the type the deployment publishes its weights at (`dtype = "auto"` converts
    /// nothing, so it is the checkpoint's own), and each `descriptor_of` below is handed the type its
    /// own layer computes in. The reference wraps the whole decode in
    /// `torch.autocast(dtype=decoder.decoder.input.weight.dtype)` (`video_vae.py`), with
    /// `precision.py:148` putting that type on the projections and on every layer's `qkv`, `output` and
    /// `mlp`. Two kinds of layer it does *not* move down - the normalizations and the attention's
    /// softmax - and both are handled where they run: the softmax is the dense-attention kernel's own
    /// FP32 accumulation, and the normalizations are declared here at the stream's type, which is the
    /// one place this declaration is coarser than autocast was, because an operator of ours reads and
    /// writes one element type. It is stated rather than hidden: a converted deployment keys the
    /// decoder's rows in its own type, and the test beside this one is what says so.
    pub fn programs(config: &VideoVaeConfig, dtype: DType) -> Result<Vec<(String, String)>> {
        use crate::programs::Programs;
        use nn::layers::UnpatchifyVideo;
        use ops::activation::ValueFirstSwiglu as SwigluOp;
        use ops::attention::dense::DenseAttention;
        use ops::cross_fade::CrossFade as CrossFadeOp;
        use ops::gemm::Gemm;
        use ops::norm::LayerNorm as LayerNormOp;
        use ops::norm::RmsNorm as RmsNormOp;
        use ops::residual::ScaledResidual as ResidualOp;
        use ops::rope::{QkNormRope, RotaryFactors, Rotation};
        use ops::rows::{self, Rows as RowsOp};
        use ops::unpatchify::Unpatchify;

        let (head, heads) = (
            config.decoder_attention_head_dim,
            config.decoder_num_attention_heads,
        );
        let width = heads * head;
        let ffn = width * config.decoder_ffn_mult;
        let spatial = config.spatial_compression().max(1);
        let patch = config.out_channels * config.temporal_compression() * spatial * spatial;
        let (rotated, sections) = rotary(config)?;
        let sections = nn::layers::Sections::new(&sections)?;
        let bias = true;

        let mut programs = Programs::default();

        // The clip's latents in, the clip's pixels out. **The bias flag is the projection's own, not a
        // policy of the block's**: `proj_in`, the attention's three projections and `proj_out` carry one,
        // while `to_out`, the packed feed-forward projection and `ff.net.2` do not - their biases are
        // folded into the residuals and into the packed activation. The epilogue is the only part of the
        // key that differs, so a statement that assumed one flag for all five launches would name a cell
        // the decode never asks for and leave the one it does ask for unbuilt.
        for (k, bias) in [
            (config.latent_channels, true),
            (width, true),
            (width, false),
            (ffn, false),
            (patch, true),
        ] {
            programs.launch::<Gemm>(&Linear::descriptor_of(
                k,
                dtype,
                dtype,
                WeightOrder::NxK,
                bias,
                dtype,
            ));
        }

        // The three placements into the token stream: every row-major gather is one program.
        programs.launch::<RowsOp>(&Rows::descriptor_of(dtype, rows::Mode::Replace, width));

        // The rotation's table once, then its application in every layer.
        programs.launch::<RotaryFactors>(&Rotary::factors_descriptor_of(
            rotated,
            1.0,
            Some(sections),
            DType::F32,
        ));
        programs.launch::<QkNormRope>(&Rotary::apply_descriptor_of(
            head,
            heads,
            heads,
            Rotation::Half,
            config.decoder_norm_eps,
            rotated,
            dtype,
            dtype,
            // The checkpoint stores no per-head normalization weight, which is what its absence means.
            false,
        ));

        // The layer's two halves: their normalization, the residual each projection folds into, and
        // the gated feed-forward's packed activation.
        programs.launch::<RmsNormOp>(&RmsNorm::descriptor_of(width, dtype));
        programs.launch::<ResidualOp>(&ScaledResidual::descriptor_of(
            dtype,
            u64::from(width),
            bias,
        )?);
        programs.launch::<SwigluOp>(&PackedSwiGlu::descriptor_of(
            dtype,
            u64::from(ffn),
            2 * u64::from(ffn),
            bias,
            false,
        )?);

        // The output path: the final normalization, the projection to pixels, and the stitch's
        // unpatchify and its blends.
        programs.launch::<LayerNormOp>(&LayerNorm::descriptor_of(dtype, width, bias));
        programs.launch::<Unpatchify>(
            &UnpatchifyVideo::new(
                dtype,
                u64::from(config.out_channels),
                config.temporal_compression(),
                spatial,
            )?
            .descriptor(),
        );
        for (_, _, blend) in crate::decoding::ClipDecoder::new(config, dtype)?.blends() {
            programs.launch::<CrossFadeOp>(&blend.descriptor());
        }
        // The stitch copies each seam's neighbour into one contiguous block before the blend reads it:
        // `cross_fade` addresses its sources flat, and a neighbour in a raster wider than the tile is
        // not one contiguous block. One copy serves every seam and every canvas, whatever the tile's
        // extents, because its key is the pair of element types and the extents are a launch value.
        use ops::copy_block::CopyBlock as CopyBlockOp;
        programs.launch::<CopyBlockOp>(&nn::layers::CopyBlock::descriptor_of(dtype));

        // One dense attention for the whole decoder: the clip's row count is a launch value, not a
        // key field, so the largest sequence the decoder was accepted for states the program too.
        let attention = block_attention(head, heads, width, dtype)?;
        // The form the backend serves: the flash kernel's packed one with offsets, or the FP32
        // program's single sequence without them.
        programs.launch::<DenseAttention>(&attention.dense_descriptor(
            dtype,
            dtype,
            dtype,
            width,
            packed(&attention),
        ));

        Ok(programs.finish())
    }

    /// Read the 36 layers and the head of the checkpoint's decoder.
    pub fn load(
        config: &VideoVaeConfig,
        canvas: Canvas,
        attention: AttentionKind,
        dtype: DType,
        input: Linear,
        loader: &mut WeightLoader,
    ) -> Result<Self> {
        let width = config.decoder_num_attention_heads * config.decoder_attention_head_dim;
        let head = config.decoder_attention_head_dim;
        let heads = config.decoder_num_attention_heads;
        let ffn = width * config.decoder_ffn_mult;
        let (rotated, sections) = rotary(config)?;
        let stitch = crate::decoding::ClipDecoder::new(config, dtype)?;
        let out_channels = config.out_channels
            * config.temporal_compression()
            * config.spatial_compression()
            * config.spatial_compression();
        let epsilon = config.decoder_norm_eps;
        // The constants this canvas publishes, declared once: a caller that reserves device storage
        // before loading sizes its pool by this list, and every publish below is checked against it.
        let declared_sizes = constants(config, canvas, u64::from(width), dtype)?;

        let registers = read(
            loader,
            "decoder.register_tokens",
            &[u64::from(canvas.registers), u64::from(width)],
        )?;
        let zero_row = zeroes(
            loader,
            &declared_sizes,
            "video_vae.zero_row",
            dtype,
            u64::from(width),
        )?;
        if canvas.tail == 0 {
            return Err(Error::config(
                "the decoder canvas has the trailing zero row".to_owned(),
            ));
        }
        let register_rows = indices(
            loader,
            &declared_sizes,
            "video_vae.register_rows",
            canvas.voxels() as u32,
            canvas.registers,
        )?;
        let zero_rows = indices(
            loader,
            &declared_sizes,
            "video_vae.zero_rows",
            (canvas.voxels() + u64::from(canvas.registers)) as u32,
            canvas.tail,
        )?;

        let mut layers = Vec::with_capacity(config.decoder_num_layers as usize);
        for index in 0..config.decoder_num_layers {
            let base = format!("decoder.transformer_blocks.{index}");
            let weight = read(loader, &format!("{base}.norm1.weight"), &[u64::from(width)])?;
            let norm1 = RmsNorm::new(weight, epsilon)?;
            let weight = read(loader, &format!("{base}.norm2.weight"), &[u64::from(width)])?;
            let norm2 = RmsNorm::new(weight, epsilon)?;
            let scale1 = read(loader, &format!("{base}.scale1"), &[u64::from(width)])?;
            let scale2 = read(loader, &format!("{base}.scale2"), &[u64::from(width)])?;
            // The attention and feed-forward projections' biases belong to their residuals, as the
            // reference groups them; the projections themselves are bias-free.
            let out_bias = read(
                loader,
                &format!("{base}.attn.to_out.0.bias"),
                &[u64::from(width)],
            )?;
            let down_bias = read(
                loader,
                &format!("{base}.ff.net.2.bias"),
                &[u64::from(width)],
            )?;
            layers.push(Layer {
                norm1,
                query: linear(
                    loader,
                    &format!("{base}.attn.to_q"),
                    u64::from(width),
                    u64::from(width),
                    true,
                )?,
                key: linear(
                    loader,
                    &format!("{base}.attn.to_k"),
                    u64::from(width),
                    u64::from(width),
                    true,
                )?,
                value: linear(
                    loader,
                    &format!("{base}.attn.to_v"),
                    u64::from(width),
                    u64::from(width),
                    true,
                )?,
                output: linear(
                    loader,
                    &format!("{base}.attn.to_out.0"),
                    u64::from(width),
                    u64::from(width),
                    false,
                )?,
                residual1: ScaledResidual::new(scale1, Some(out_bias))?,
                norm2,
                // The feed-forward's projection is packed: its rows hold the value half and then the
                // gate half, so it is twice the feed-forward's width and its bias is the packed pair
                // that the activation adds to each half. Reading it at the feed-forward's own width
                // loads the value half alone and leaves the gate half at zero.
                gate_up: linear(
                    loader,
                    &format!("{base}.ff.net.0.proj"),
                    u64::from(width),
                    2 * u64::from(ffn),
                    false,
                )?,
                down: linear(
                    loader,
                    &format!("{base}.ff.net.2"),
                    u64::from(ffn),
                    u64::from(width),
                    false,
                )?,
                residual2: ScaledResidual::new(scale2, Some(down_bias))?,
                gated: PackedSwiGlu::new(Some(read(
                    loader,
                    &format!("{base}.ff.net.0.proj.bias"),
                    &[2 * u64::from(ffn)],
                )?))?,
            });
        }

        let weight = read(loader, "decoder.norm_out.weight", &[u64::from(width)])?;
        let bias = read(loader, "decoder.norm_out.bias", &[u64::from(width)])?;
        let norm = LayerNorm::new(weight, Some(bias), epsilon)?;
        let output = linear(
            loader,
            "decoder.proj_out",
            u64::from(width),
            u64::from(out_channels),
            true,
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
        Ok(Self {
            canvas,
            dtype,
            stitch,
            width,
            input,
            registers,
            register_rows,
            zero_row,
            zero_rows,
            layers,
            norm,
            output,
            rotary,
            place: Rows::new(dtype, u64::from(width))?,
            voxel_rows: pack_voxel_rows(loader, config, attention, canvas)?,
            offsets: {
                // `[0, rows]`: one sequence starting at row zero. A constant of this canvas, so it is
                // created once with the weights and read by every clip's attention launch.
                let rows = i32::try_from(canvas.rows()).map_err(|_| {
                    Error::config("a clip's rows exceed a sequence offset".to_owned())
                })?;
                let mut bytes = Vec::with_capacity(8);
                for offset in [0i32, rows] {
                    bytes.extend_from_slice(&offset.to_le_bytes());
                }
                loader.constant(
                    "video_vae.sequence_offsets",
                    DType::I32,
                    Shape::new(&[2]),
                    &bytes,
                    "minimax-h3",
                )?
            },
            attention: block_attention(
                head,
                heads,
                u32::try_from(canvas.rows())
                    .map_err(|_| Error::config("too many tokens".to_owned()))?,
                dtype,
            )?,
            config: config.clone(),
            ffn_mult: config.decoder_ffn_mult,
            head: u64::from(head),
            heads: u64::from(heads),
        })
    }
}

/// Read a projection of the checkpoint: `[out, in]` row-major, with its bias when it has one.
fn linear(loader: &mut WeightLoader, name: &str, from: u64, to: u64, bias: bool) -> Result<Linear> {
    let weight = read(loader, &format!("{name}.weight"), &[to, from])?;
    let bias = if bias {
        Some(read(loader, &format!("{name}.bias"), &[to])?)
    } else {
        None
    };
    Ok(Linear::new(weight, bias, WeightOrder::NxK)?)
}

/// The constants a decoder of this canvas publishes, as `(name, bytes)`.
///
/// `load` builds five tensors the export does not hold - the zero row the token layout appends, the
/// two index tables that address them, the rotary frequencies and the voxel rows - and a caller that
/// reserves device storage before loading has to know what they cost: the pool charges a block per
/// publish, so a reservation computed from the checkpoint's names alone is short by every one of
/// these. The list is one statement of them, and `load` checks what it publishes against it, so the
/// two cannot drift.
pub fn constants(
    config: &VideoVaeConfig,
    canvas: Canvas,
    width: u64,
    dtype: DType,
) -> Result<Vec<(String, u64)>> {
    let (rotated, _) = rotary(config)?;
    let element = dtype.bytes().unwrap_or(4);
    Ok(vec![
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
        ("video_vae.voxel_rows".to_owned(), canvas.voxels() * 4),
    ])
}

/// Read one weight.
fn read(loader: &mut WeightLoader, name: &str, shape: &[u64]) -> Result<Weight> {
    Ok(loader.load(&WeightRequest::region(
        name.to_owned(),
        0,
        Shape::new(shape),
    ))?)
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

/// A constant row of `width` zeros, which the checkpoint's token layout appends.
fn zeroes(
    loader: &mut WeightLoader,
    declared_sizes: &[(String, u64)],
    name: &str,
    dtype: DType,
    width: u64,
) -> Result<Weight> {
    let bytes = vec![0u8; (width as usize) * dtype.bytes().unwrap_or(4) as usize];
    declared(declared_sizes, name, bytes.len())?;
    Ok(loader.constant(name, dtype, Shape::new(&[1, width]), &bytes, "minimax-h3")?)
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
    /// The tables one layout's decode reads, in [`table_bytes_into`]' order.
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

    /// The layout's tables, keyed by the extents the layout states.
    ///
    /// This is the shape the capability protocol asks for; a request's own state is none of it.
    pub fn layout_tables(&self, layout: &nn::capability::Layout) -> Result<Vec<StateTable>> {
        let video = layout.video;
        self.tables(video.height, video.width)
    }

    /// Write the layout's table bytes, in [`Self::layout_tables`]' order.
    pub fn prepare_layout(
        &self,
        layout: &nn::capability::Layout,
        out: &mut [Vec<u8>],
    ) -> Result<()> {
        let video = layout.video;
        self.table_bytes(video.height, video.width, out)
    }

    /// The transformer's scratch for one tile, reserved once per clip and reused by every tile.
    ///
    /// Every buffer here is sized by the block's own canvas - one tile's rows and voxels - so a clip's
    /// tiles share one set: they are used sequentially, and the arena never rewinds, so a per-tile
    /// reservation would grow the clip's footprint by the tile count. The plan states these once
    /// ([`crate::decoding::video_arena`]) and this is what makes that true.
    pub fn scratch(&self, f: &Forward) -> Result<TileScratch> {
        let rows = self.canvas.rows();
        let voxels = self.canvas.voxels();
        let width = u64::from(self.width);
        let ffn = width * u64::from(self.ffn_mult);
        let half = u64::from(rotary(&self.config)?.0 / 2);
        Ok(TileScratch {
            stream: f.arena().tensor(self.dtype, Shape::new(&[rows, width]))?,
            projected: f.arena().tensor(self.dtype, Shape::new(&[voxels, width]))?,
            cosine: f.arena().tensor(DType::F32, Shape::new(&[rows, half]))?,
            sine: f.arena().tensor(DType::F32, Shape::new(&[rows, half]))?,
            normalized: f.arena().tensor(self.dtype, Shape::new(&[rows, width]))?,
            query: f.arena().tensor(self.dtype, Shape::new(&[rows, width]))?,
            key: f.arena().tensor(self.dtype, Shape::new(&[rows, width]))?,
            value: f.arena().tensor(self.dtype, Shape::new(&[rows, width]))?,
            rotated_query: f.arena().tensor(self.dtype, Shape::new(&[rows, width]))?,
            rotated_key: f.arena().tensor(self.dtype, Shape::new(&[rows, width]))?,
            attended: f.arena().tensor(self.dtype, Shape::new(&[rows, width]))?,
            update: f.arena().tensor(self.dtype, Shape::new(&[rows, width]))?,
            gate_up: f.arena().tensor(self.dtype, Shape::new(&[rows, 2 * ffn]))?,
            gated: f.arena().tensor(self.dtype, Shape::new(&[rows, ffn]))?,
        })
    }

    /// Decode one clip's latent rows into the pixel block of every token.
    ///
    /// `latents` is `[voxels, latent_channels]` — one row per latent voxel, in the raster order the
    /// caller's clip rows select — `positions` is `[3, rows]` F32, axis-major, the per-layout table
    /// [`table_specs`] declares, and `out` is `[rows, out_channels · 4 · 16 · 16]`: one token's
    /// 3×4×16×16 pixel block,
    /// which the caller scatters into the clip's raster. The rows are the voxels, then the register
    /// rows, then the trailing zero row.
    pub fn decode(
        &self,
        f: &Forward,
        latents: TensorView,
        positions: TensorView,
        out: TensorView,
        scratch: &TileScratch,
    ) -> Result<()> {
        let rows = self.canvas.rows();
        let voxels = self.canvas.voxels();
        if latents.shape.rank() != 2 || latents.shape.dim(0) != voxels {
            return Err(Error::config(format!(
                "the decoder reads {voxels} latent rows, not {:?}",
                latents.shape.dims()
            )));
        }
        if out.shape.rank() != 2 || out.shape.dim(0) != rows {
            return Err(Error::config(format!(
                "the decoder writes {rows} token rows, not {:?}",
                out.shape.dims()
            )));
        }
        // The rotation reads one leading coordinate per axis; a row-major table would be read
        // transposed and rotate every row by another row's angles, so the shape is checked here
        // rather than trusted.
        if positions.shape.rank() != 2
            || positions.shape.dim(0) != 3
            || positions.shape.dim(1) != rows
        {
            return Err(Error::config(format!(
                "the decoder reads a [3, {rows}] position table, not {:?}",
                positions.shape.dims()
            )));
        }

        let stream = scratch.stream;
        let projected = scratch.projected;
        self.input.forward(f, latents, projected)?;
        self.place
            .replace(f, stream, projected, self.voxel_rows.view())?;
        self.place
            .replace(f, stream, self.registers.view(), self.register_rows.view())?;
        self.place
            .replace(f, stream, self.zero_row.view(), self.zero_rows.view())?;

        let (cosine, sine) = (scratch.cosine, scratch.sine);
        self.rotary.factors(f, positions, cosine, sine)?;

        let (head, heads) = (self.head, self.heads);
        let normalized = scratch.normalized;
        let (query, key, value) = (scratch.query, scratch.key, scratch.value);
        let (rotated_query, rotated_key) = (scratch.rotated_query, scratch.rotated_key);
        let (attended, update) = (scratch.attended, scratch.update);
        let (gate_up, gated) = (scratch.gate_up, scratch.gated);
        let hidden = stream;
        let per_head = |view: TensorView| view.reshape(Shape::new(&[rows, heads, head]));

        for layer in &self.layers {
            layer.norm1.forward(f, hidden, normalized)?;
            layer.query.forward(f, normalized, query)?;
            layer.key.forward(f, normalized, key)?;
            layer.value.forward(f, normalized, value)?;
            self.rotary.apply(
                f,
                per_head(query),
                per_head(key),
                None,
                None,
                cosine,
                sine,
                per_head(rotated_query),
                per_head(rotated_key),
            )?;
            // One sequence: the clip's rows, its registers and its tail attended together.
            //
            // The row offsets are the flash backend's packed form and nothing else: it serves
            // sequences packed by offsets and no other shape, while the FP32 program - a GEMM and a
            // softmax - serves one sequence without them, and its key says which it is
            // (`cu_seq_lens` is printed only for the form without offsets). One sequence makes the
            // two arithmetically identical, so the launch states the form its own backend has
            // rather than asking for a cell the other backend's package declares.
            let offsets = if packed(&self.attention) {
                Some(self.offsets.view())
            } else {
                None
            };
            self.attention.attend_dense(
                f,
                per_head(rotated_query),
                per_head(rotated_key),
                per_head(value),
                offsets,
                offsets,
                u32::try_from(rows)
                    .map_err(|_| Error::config("too many tokens for one sequence".to_owned()))?,
                // The attention writes one row per head, so its output is the same three axes its
                // inputs are: the scratch buffer is a flat `[rows, width]` block and the heads are
                // the middle axis of the same bytes. Passing the flat view was `dense_attention:
                // expected 3 axes, found 2` at the first device run - the only call site of the six
                // in this model that did not reshape its output, and the shapes of the other five
                // were already `[rows, heads, head]` at the arena.
                per_head(attended),
            )?;
            layer.output.forward(f, attended, update)?;
            layer.residual1.forward(f, hidden, update)?;

            layer.norm2.forward(f, hidden, normalized)?;
            layer.gate_up.forward(f, normalized, gate_up)?;
            layer.gated.forward(f, gate_up, gated, None)?;
            layer.down.forward(f, gated, update)?;
            layer.residual2.forward(f, hidden, update)?;
        }

        self.norm.forward(f, hidden, normalized)?;
        Ok(self.output.forward(f, normalized, out)?)
    }
}

/// The transformer's scratch for one tile: fourteen buffers, reserved once per clip.
#[derive(Clone, Copy)]
pub struct TileScratch {
    stream: TensorView,
    projected: TensorView,
    cosine: TensorView,
    sine: TensorView,
    normalized: TensorView,
    query: TensorView,
    key: TensorView,
    value: TensorView,
    rotated_query: TensorView,
    rotated_key: TensorView,
    attended: TensorView,
    update: TensorView,
    gate_up: TensorView,
    gated: TensorView,
}

impl Decoder for VideoDecoder {
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

    /// Decode one clip: every tile of its canvas, stitched into `out.video`.
    ///
    /// `latents` is the clip's `[voxels, latent_channels]` rows in tile order — the runner gathers them
    /// through the checkpoint's own `video_vae.voxel_rows` — `state` holds this block's one table, `video_positions`
    /// `[3, rows]` axis-major in radians, and `out.video` is the clip's raster. The raster is
    /// `[channels, frames, rows, columns]` in the decoder's own element type: the layers on this path —
    /// the unpatchify, the seam blends and the byte step — all read a planar channels-first raster, and
    /// `frames` is the clip's **native** output, `temporal` frames per latent frame, so it still holds
    /// the three padding frames the splice drops. `out.audio` must be absent.
    ///
    /// The grid is arithmetic on the raster's extent, so one block serves every canvas of this
    /// checkpoint; the clips are the runner's loop, and so is the splice.
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
        let [positions] = state else {
            return Err(refuse(format!(
                "the video decoder reads one table, video_positions, and {} were given",
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
        let tiles = Tiles::of(&self.config, &clip)
            .map_err(|error| refuse(format!("the clip's tile grid: {error}")))?
            .tiles(&self.config, &clip);
        self.stitch
            .decode_clip(f, self, &tiles, latents, *positions, video)
            .map_err(|error| refuse(format!("{error}")))
    }
}

/// The decoder's whole first affine, composed at the loading boundary.
///
/// The reference decodes a clip in three steps before its transformer: it restores each latent channel
/// (`s ⊙ x + m`), runs the checkpoint's one-by-one post-quantization convolution, and projects the
/// latent channels onto the transformer's width. There is no nonlinearity between them, so the three
/// are one affine:
///
/// ```text
/// z = Wpi · (Wpq · (s ⊙ x + m) + bpq) + bpi
///   = (Wpi · Wpq · diag(s)) · x + (Wpi · (Wpq · m + bpq) + bpi)
/// ```
///
/// Composing them here is what the crate does for every other affine the checkpoint states as two
/// operators (`adaln_proj`, the student heads): one element type, one operator, one rounding, and the
/// host product paid once at load. The product is `[width, latent_channels]` from two matrices no wider
/// than the latent's channels.
///
/// `wpi` is `[width, latent]` row-major, `wpq` is `[latent, latent]`, `bpi` is `[width]`, `bpq`, `mean`
/// and `std` are `[latent]`. The result is the folded `(weight, bias)`; a checkpoint whose extents
/// disagree is refused rather than read past.
pub fn fold_input_affine(
    wpi: &[f32],
    bpi: &[f32],
    wpq: &[f32],
    bpq: &[f32],
    mean: &[f32],
    std: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    let latent = mean.len();
    let width = bpi.len();
    if latent == 0 || width == 0 {
        return Err(Error::config(
            "the decoder's first affine needs a width and a latent channel count".to_owned(),
        ));
    }
    if std.len() != latent || bpq.len() != latent {
        return Err(Error::config(format!(
            "the decoder's first affine has {latent} latent channels, {} deviations and {} post-quant \
             biases",
            std.len(),
            bpq.len()
        )));
    }
    if wpi.len() != width * latent || wpq.len() != latent * latent {
        return Err(Error::config(format!(
            "the decoder's first affine reads a {}-element projection and a {}-element post-quant \
             convolution for {width} outputs of {latent} inputs",
            wpi.len(),
            wpq.len()
        )));
    }
    for (name, values) in [("mean", mean), ("deviation", std), ("post-quant bias", bpq)] {
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Error::config(format!(
                "the decoder's latent {name} is not finite"
            )));
        }
    }
    // `post-quant` scaled by the deviations, so the composition needs no separate diagonal: the
    // statistics enter as the post-quantization matrix's input columns.
    let mut scaled = wpq.to_vec();
    for row in 0..latent {
        for column in 0..latent {
            scaled[row * latent + column] *= std[column];
        }
    }
    // The projection of the scaled matrix: one pass per output row over the latent channels.
    let mut weight = vec![0.0_f32; width * latent];
    for row in 0..width {
        let source = &wpi[row * latent..][..latent];
        for column in 0..latent {
            let mut sum = 0.0_f32;
            for inner in 0..latent {
                sum += source[inner] * scaled[inner * latent + column];
            }
            weight[row * latent + column] = sum;
        }
    }
    // The bias: the post-quantization convolution's shift of the restored mean, projected.
    let mut shifted = vec![0.0_f32; latent];
    for row in 0..latent {
        let mut sum = bpq[row];
        for column in 0..latent {
            sum += wpq[row * latent + column] * mean[column];
        }
        shifted[row] = sum;
    }
    let mut bias = vec![0.0_f32; width];
    for row in 0..width {
        let source = &wpi[row * latent..][..latent];
        let mut sum = bpi[row];
        for inner in 0..latent {
            sum += source[inner] * shifted[inner];
        }
        bias[row] = sum;
    }
    Ok((weight, bias))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

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
    fn the_voxel_table_names_packed_rows_and_follows_the_deployment_packing() {
        let config = video_config();
        let canvas = crate::config::Canvas {
            short_edge: 768,
            aspect: (7, 4),
            width: 1344,
            height: 768,
        };
        let raster = crate::packing::latent_raster(&canvas).expect("raster");
        let clip = Canvas::clip(&config, raster.height, raster.width);
        let side = crate::packing::PATCH_SIDE;
        let sparse = AttentionKind::sparse(64, 0.8, None).expect("a sparse kind");

        // The gather writes `destination[indices[voxel]] = source[voxel]`, so every entry must name a
        // packed row of the clip. A table copied from the voxel indices runs to four times that bound
        // and scatters the clip's tokens past the end of the stream - which is why this is asserted
        // rather than assumed.
        for attention in [AttentionKind::Dense, sparse] {
            let rows =
                voxel_rows(&config, attention, raster.height, raster.width).expect("voxel rows");
            let packed = clip.rows();
            assert_eq!(rows.len() as u64, clip.voxels());
            assert!(
                rows.iter().all(|row| u64::from(*row) < packed),
                "a table entry names a row outside the clip's {packed} packed rows"
            );
            // Each video row is named once per voxel of its own patch, so the counts are the patch's
            // area and nothing else: no row is skipped and none is named twice as often. The register
            // and tail rows of the clip are not voxels and are named by no entry.
            let mut counts = vec![0_u32; clip.rows() as usize];
            for row in &rows {
                counts[*row as usize] += 1;
            }
            let video = (clip.frames * raster.patch_rows()) as usize;
            assert!(
                counts[..video].iter().all(|count| *count == side * side),
                "{attention:?} names the clip's video rows unevenly"
            );
            assert!(
                counts[video..].iter().all(|count| *count == 0),
                "{attention:?} names a register or tail row with a voxel"
            );
        }

        // The deployments do not read the same table: a sparse packing orders the rows by tile, so a
        // decoder built for one packing cannot be read through the other's.
        let dense = voxel_rows(&config, AttentionKind::Dense, raster.height, raster.width)
            .expect("the dense table");
        let tiled =
            voxel_rows(&config, sparse, raster.height, raster.width).expect("the tiled one");
        assert_ne!(dense, tiled, "{dense:?}");
    }

    #[test]
    fn the_decoder_tables_name_one_voxel_and_one_position_per_row() {
        let config = video_config();
        let canvas = crate::config::Canvas {
            short_edge: 768,
            aspect: (7, 4),
            width: 1344,
            height: 768,
        };
        // The layout's own extents: the canvas over the spatial compression.
        let raster = crate::packing::latent_raster(&canvas).expect("raster");
        // One entry per latent voxel of **one clip** - the checkpoint's own `video_vae.voxel_rows` - and
        // the values are the packed rows of the patches those voxels lie in, not the voxel indices,
        // which is the factor of the patch area apart. How many clips a request has is the runner's
        // question, so no clip count appears here.
        let rows =
            voxel_rows(&config, AttentionKind::Dense, raster.height, raster.width).expect("rows");
        let clip = Canvas::clip(&config, raster.height, raster.width);
        assert_eq!(rows.len() as u64, clip.voxels());
        let side = u64::from(crate::packing::PATCH_SIDE);
        let columns = u64::from(raster.width) / side;
        let per_frame = u64::from(raster.patch_rows());
        // Dense packing is the identity, so the first clip's first frame names its own patch rows in
        // order, each one repeated for the `2 x 2` voxels that share it.
        for (index, row) in rows[..(raster.height * raster.width) as usize]
            .iter()
            .enumerate()
        {
            let (height, column) = (
                index as u64 / u64::from(raster.width),
                index as u64 % u64::from(raster.width),
            );
            assert_eq!(
                u64::from(*row),
                (height / side) * columns + column / side,
                "voxel ({height}, {column}) names its patch's packed row"
            );
        }
        // Every value is a row of the packed sequence the denoiser writes.
        let packed = crate::packing::video_order_of(AttentionKind::Dense, clip.frames, raster)
            .expect("order")
            .len() as u64;
        assert!(rows.iter().all(|row| u64::from(*row) < packed));
        assert!(per_frame < packed);
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
        assert_eq!(declared.len(), 1);
        assert_eq!(declared[0].name, "video_positions");
        // `[3, rows]`: the coordinate axis first, then one row per row of the largest tile.
        assert_eq!(declared[0].shape.rank(), 2);
        assert_eq!(declared[0].shape.dim(0), 3);
        assert_eq!(declared[0].shape.dim(1), tile.rows());
        assert_eq!(declared[0].dtype, DType::F32);
    }

    #[test]
    fn the_position_table_is_one_upload_for_every_canvas() {
        // The tiles are uniform, so the table is the largest tile's however large the canvas is: a
        // layout costs the executor one upload of the same size whatever its extents. The two canvases
        // here are the served ones - 768p at 768 x 1344 (48 x 84 latents) and 672p at 672 x 1536
        // (42 x 96) - and they are the same table, not two of the same shape by coincidence.
        let config = video_config();
        let mut vertical = vec![Vec::new()];
        let mut wide = vec![Vec::new()];
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
        // The executor sizes `out` from `table_specs`, so a count that disagrees with the block is a
        // protocol error and must surface as one rather than as a partially written table.
        let config = video_config();
        let mut none: Vec<Vec<u8>> = Vec::new();
        assert!(table_bytes_into(&config, 48, 84, &mut none).is_err());
        let mut two = vec![Vec::new(), Vec::new()];
        assert!(table_bytes_into(&config, 48, 84, &mut two).is_err());
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
    fn the_three_first_steps_are_one_affine() {
        // Two outputs of two latent channels: the projection is the identity, the post-quantization
        // convolution scales each channel, and the statistics are the two-step restore. The composed
        // affine must agree with the three reference steps applied one after another.
        let wpi = vec![1.0, 0.0, 0.0, 1.0];
        let bpi = vec![0.0, 0.0];
        let wpq = vec![2.0, 0.0, 0.0, 3.0];
        let bpq = vec![0.5, 0.5];
        let std = vec![2.0, 4.0];
        let mean = vec![1.0, -1.0];
        let (weight, bias) =
            fold_input_affine(&wpi, &bpi, &wpq, &bpq, &mean, &std).expect("the fold");
        // `Wpq · diag(std)`: the statistics enter as the post-quantization matrix's input columns.
        assert_eq!(weight, vec![4.0, 0.0, 0.0, 12.0]);
        // `bpi + Wpi · (Wpq · mean + bpq)`.
        assert_eq!(bias, vec![2.5, -2.5]);
        // And it is the same function: the three steps on one latent.
        for (x0, x1) in [(0.0, 0.0), (1.0, -2.0), (0.25, 3.5)] {
            let restored = [x0 * std[0] + mean[0], x1 * std[1] + mean[1]];
            let posted = [wpq[0] * restored[0] + bpq[0], wpq[3] * restored[1] + bpq[1]];
            let (z0, z1) = (wpi[0] * posted[0] + bpi[0], wpi[3] * posted[1] + bpi[1]);
            let (f0, f1) = (
                weight[0] * x0 + weight[1] * x1 + bias[0],
                weight[2] * x0 + weight[3] * x1 + bias[1],
            );
            assert!((z0 - f0).abs() < 1e-6 && (z1 - f1).abs() < 1e-6);
        }
        // Extents that disagree are refused rather than read past.
        assert!(fold_input_affine(&wpi, &bpi, &wpq, &bpq, &[0.0], &std).is_err());
        assert!(fold_input_affine(&wpi, &bpi, &wpq, &bpq, &mean, &[1.0]).is_err());
        assert!(fold_input_affine(&wpi[..3], &bpi, &wpq, &bpq, &mean, &std).is_err());
        assert!(fold_input_affine(&wpi, &bpi, &wpq, &bpq, &mean, &[f32::NAN, 1.0]).is_err());
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
        // `decoder.` plus the two `post_quant_conv.` tensors its input affine is folded from, and the
        // encoder reads the remaining 118. A checkpoint that lacks one is refused by block and name.
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
            "the folded input affine's tensor is asked for"
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

    /// The decoder's programs follow from its configuration and the rasters the deployment serves:
    /// one key per operator whose key does not vary with the launch, a blend per seam width the stitch
    /// compiles, and one dense attention per distinct token count.
    #[test]
    fn the_decoder_states_its_programs_from_its_own_config() {
        let config = video_config();
        let programs =
            VideoDecoder::programs(&config, DType::BF16).expect("the decoder's programs");
        let count = |operator: &str| programs.iter().filter(|(name, _)| name == operator).count();

        // Every projection this decoder has, whatever its contraction: the key carries the element
        // types and the order, and the contraction length is a descriptor field of one program. Two
        // programs, because the epilogue is the one field that differs: `proj_in`, the attention's
        // three projections and `proj_out` fold a bias, while `to_out`, the packed feed-forward
        // projection and `ff.net.2` carry none.
        assert_eq!(count("gemm"), 2);
        assert_eq!(count("rows"), 1, "three placements, one program");
        assert_eq!(count("rotary_factors"), 1);
        assert_eq!(count("qk_norm_rope"), 1);
        assert_eq!(count("rms_norm"), 1);
        assert_eq!(count("layer_norm"), 1);
        assert_eq!(count("scaled_residual"), 1);
        assert_eq!(count("value_first_swiglu"), 1);
        assert_eq!(count("unpatchify_video"), 1);
        // The stitch's third family: each seam's neighbour is copied into one contiguous block before
        // the blend reads it, because a blend addresses its sources flat and a neighbour in a raster
        // wider than the tile is not one contiguous block. One copy serves every seam of every canvas.
        assert_eq!(count("copy_block"), 1);
        assert!(
            programs
                .iter()
                .any(|(name, key)| name == "copy_block" && key.contains("source_f16_out_f16")),
            "the copy states the deployment's element type twice, as its source and its out"
        );

        // The stitch compiles one blend per seam width on each of the two axes.
        let blends = crate::decoding::ClipDecoder::new(&config, DType::BF16)
            .expect("the stitch")
            .blends()
            .len();
        assert!(blends > 0);
        assert_eq!(count("cross_fade"), blends);

        // One dense attention whatever the clip: the row count is a launch value, so the decoder's
        // programs do not depend on the raster a request asks for.
        assert_eq!(count("dense_attention"), 1, "{programs:?}");
    }

    /// A converted deployment keys the decoder's rows in its own element type, and nothing is implicit.
    ///
    /// `specs/worker.md`'s table replaces `torch.autocast` with a declaration per layer. The observable
    /// consequence is this: the same decoder at two deployments is two row sets, and every row that
    /// names an element type names the deployment's - except the rotation's factor table, whose
    /// *positions* are FP32 whatever the decode is (they are coordinates, not activations). A row that
    /// stayed FP32 under a half deployment would be a layer silently computing in a type nobody chose,
    /// which is the behaviour the design removed.
    #[test]
    fn a_converted_deployment_keys_the_decoders_rows_in_its_own_element_type() {
        let config = video_config();
        let rows =
            |dtype: DType| VideoDecoder::programs(&config, dtype).expect("the decoder's programs");
        let whole = rows(DType::F32);
        let half = rows(DType::F16);
        assert_eq!(
            half.len(),
            whole.len(),
            "the same layers, in the same order"
        );
        let mut moved = 0;
        for (converted, declared) in half.iter().zip(&whole) {
            assert_eq!(converted.0, declared.0, "the same layer");
            if converted.1 == declared.1 {
                continue;
            }
            moved += 1;
            // The one exception the declaration states: the factor table's positions are coordinates,
            // FP32 whatever the decode is. Everything else in the row is the deployment's type.
            // The row's own element type is the deployment's. An operand that stays FP32 - the
            // rotation's positions, the attention's kv - is a layer's declaration, not the row's.
            assert!(
                converted.1.contains("dtype_f16"),
                "{} did not follow the deployment: {} at f16",
                converted.0,
                converted.1
            );
        }
        assert!(
            moved > 0,
            "a converted deployment moves the rows it must: {whole:?}"
        );
    }

    /// One dense attention, and its key is the form the element type it decodes at has.
    ///
    /// The FP32 program is a GEMM and a softmax over one sequence: it reads no offsets, and its
    /// key says so (`_cu_seq_lens_false` is printed exactly for the form without them), which is
    /// the only cell `dense_attention/f32` declares. The half-precision kernel serves the packed
    /// form, which carries offsets and prints no such field. A clip decoded at f32 that asked the
    /// half-precision kernel for its attention would name a program the f32 package cannot build.
    #[test]
    fn the_decoders_attention_names_the_form_its_element_type_has() {
        let config = video_config();
        let key = |dtype: DType| {
            VideoDecoder::programs(&config, dtype)
                .expect("the decoder's programs")
                .into_iter()
                .find(|(name, _)| name == "dense_attention")
                .map(|(_, key)| key)
                .expect("one dense attention")
        };

        let fp32 = key(DType::F32);
        assert!(
            fp32.ends_with("mask_dense_window_false_cu_seq_lens_false"),
            "{fp32}"
        );
        assert!(fp32.contains("dtype_f32_kv_f32_out_f32"), "{fp32}");

        for half in [DType::F16, DType::BF16] {
            let packed = key(half);
            assert!(packed.ends_with("mask_dense_window_false"), "{packed}");
            assert!(
                !packed.contains("cu_seq_lens"),
                "the packed form names no offsets: {packed}"
            );
        }
    }
}
