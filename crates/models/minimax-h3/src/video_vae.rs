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
//! The decoder declares one element type — the checkpoint's own representation, FP16 for the dense
//! `quality` checkpoint (specs/worker.md 1). Every parameter it reads is converted to that type once
//! at the loading boundary, including the RMSNorm weights and the channel scales, because the fused
//! residual and normalization operators carry one element type for the rows, the parameters and the
//! residual together. The reference instead keeps those parameters FP32 and promotes the residual
//! stream to FP32 (torch's autocast does not reach them); the reference measures that family of
//! difference at 1.0e-3 relative L2 against an FP32 decode, inside the tolerance口径 of specs/worker.md 1,
//! and quality is confirmed by viewing (specs/worker.md 10), not by bit equality.

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

    /// The latent span of tile `index`: `(start, length)` in latent columns, and the tile's extent
    /// clipped to the raster.
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
        let mut tiles = Vec::with_capacity(self.rows.len() * self.columns.len());
        for row in 0..self.rows.len() {
            for column in 0..self.columns.len() {
                let (row_start, row_end) = self.rows.latent(row, canvas.height, ratio);
                let (column_start, column_end) = self.columns.latent(column, canvas.width, ratio);
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

/// The cell-centre rotary coordinates of the transformer's token rows, as `[rows, 3]` FP32.
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
    let centre = (f64::from(index) + 0.5) / f64::from(size);
    (std::f64::consts::TAU * (2.0 * centre - 1.0)) as f32
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
use nn::layers::Rotation;
use nn::layers::{
    Attention, LayerNorm, Linear, PackedSwiGlu, RmsNorm, Rotary, Rows, ScaledResidual, Sections,
};
use nn::weights::{Weight, WeightLoader, WeightRequest};
use nn::{DType, Shape, TensorView};
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
    /// The index table that places each voxel row of the stream.
    voxel_rows: Weight,
    config: VideoVaeConfig,
    ffn_mult: u32,
    /// Elements per head.
    head: u64,
    /// Query heads per token.
    heads: u64,
    attention: Attention,
}

impl VideoDecoder {
    /// Read the 36 layers and the head of the checkpoint's decoder.
    pub fn load(
        config: &VideoVaeConfig,
        canvas: Canvas,
        loader: &mut WeightLoader,
    ) -> Result<Self> {
        let width = config.decoder_num_attention_heads * config.decoder_attention_head_dim;
        let head = config.decoder_attention_head_dim;
        let heads = config.decoder_num_attention_heads;
        let ffn = width * config.decoder_ffn_mult;
        let (rotated, sections) = rotary(config)?;
        let out_channels = config.out_channels
            * config.temporal_compression()
            * config.spatial_compression()
            * config.spatial_compression();
        let epsilon = config.decoder_norm_eps;

        let input = linear(
            loader,
            "decoder.proj_in",
            u64::from(config.latent_channels),
            u64::from(width),
            true,
        )?;
        let registers = read(
            loader,
            "decoder.register_tokens",
            &[u64::from(canvas.registers), u64::from(width)],
        )?;
        let zero_row = zeroes(loader, "video_vae.zero_row", DType::F16, u64::from(width))?;
        if canvas.tail == 0 {
            return Err(Error::config(
                "the decoder canvas has the trailing zero row".to_owned(),
            ));
        }
        let register_rows = indices(
            loader,
            "video_vae.register_rows",
            canvas.voxels() as u32,
            canvas.registers as u32,
        )?;
        let zero_rows = indices(
            loader,
            "video_vae.zero_rows",
            (canvas.voxels() + u64::from(canvas.registers)) as u32,
            canvas.tail as u32,
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
                gate_up: linear(
                    loader,
                    &format!("{base}.ff.net.0.proj"),
                    u64::from(width),
                    u64::from(ffn),
                    true,
                )?,
                down: linear(
                    loader,
                    &format!("{base}.ff.net.2"),
                    u64::from(ffn),
                    u64::from(width),
                    false,
                )?,
                residual2: ScaledResidual::new(scale2, Some(down_bias))?,
                gated: PackedSwiGlu::new(None)?,
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
        let frequencies = frequencies(loader, config)?;
        let rotary = Rotary::new(
            frequencies,
            head,
            heads,
            heads,
            epsilon,
            Rotation::Half,
            1.0,
        )?
        .partial(rotated)?
        .sectioned(Sections::new(&sections)?)?;
        Ok(Self {
            canvas,
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
            place: Rows::new(DType::F16, u64::from(width))?,
            voxel_rows: indices(loader, "video_vae.voxel_rows", 0, canvas.voxels() as u32)?,
            attention: Attention::dense(
                head,
                heads,
                heads,
                u32::try_from(canvas.rows())
                    .map_err(|_| Error::config("too many tokens".to_owned()))?,
                false,
                None,
                1.0 / (head as f32).sqrt(),
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

/// Read one weight.
fn read(loader: &mut WeightLoader, name: &str, shape: &[u64]) -> Result<Weight> {
    Ok(loader.load(&WeightRequest::region(
        name.to_owned(),
        0,
        Shape::new(shape),
    ))?)
}

/// A constant row of `width` zeros, which the checkpoint's token layout appends.
fn zeroes(loader: &mut WeightLoader, name: &str, dtype: DType, width: u64) -> Result<Weight> {
    let bytes = vec![0u8; (width as usize) * dtype.bytes().unwrap_or(4) as usize];
    Ok(loader.constant(name, dtype, Shape::new(&[1, width]), &bytes)?)
}

/// A constant index table: `count` rows starting at `start`.
fn indices(loader: &mut WeightLoader, name: &str, start: u32, count: u32) -> Result<Weight> {
    let mut bytes = Vec::with_capacity(count as usize * 4);
    for index in 0..count {
        bytes.extend_from_slice(&(start + index).to_le_bytes());
    }
    Ok(loader.constant(name, DType::I32, Shape::new(&[u64::from(count)]), &bytes)?)
}

/// The rotary frequency table of the decoder: `theta^(-2i/rotated)` over the rotated width.
fn frequencies(loader: &mut WeightLoader, config: &VideoVaeConfig) -> Result<Weight> {
    let (rotated, sections) = rotary(config)?;
    // One block per axis, each over that axis's own width: the reference builds its embedding with
    // `head_dim · ratio / 3` and reuses those frequencies on all three axes, so axis `a` takes
    // `theta^(-2k / sections[a])` rather than a slice of one wider spectrum.
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
    let mut bytes = Vec::with_capacity(table.len() * 4);
    for value in &table {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let name = format!("video_vae.rotary.{rotated}.{}", config.decoder_rope_theta);
    Ok(loader.constant(&name, DType::F32, Shape::new(&[table.len() as u64]), &bytes)?)
}

impl VideoDecoder {
    /// Decode one clip's latent rows into the pixel block of every token.
    ///
    /// `latents` is `[voxels, latent_channels]` — one row per latent voxel, in the raster order the
    /// caller's clip rows select — `positions` is `[rows, 3]` F32, the per-layout table [`positions`]
    /// computes, and `out` is `[rows, out_channels · 4 · 16 · 16]`: one token's 3×4×16×16 pixel block,
    /// which the caller scatters into the clip's raster. The rows are the voxels, then the register
    /// rows, then the trailing zero row.
    pub fn decode(
        &self,
        f: &Forward,
        latents: TensorView,
        positions: TensorView,
        out: TensorView,
    ) -> Result<()> {
        let rows = self.canvas.rows();
        let voxels = self.canvas.voxels();
        let width = u64::from(self.width);
        let ffn = width * u64::from(self.ffn_mult);
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

        let stream = f.arena().tensor(DType::F16, Shape::new(&[rows, width]))?;
        let projected = f.arena().tensor(DType::F16, Shape::new(&[voxels, width]))?;
        self.input.forward(f, latents, projected)?;
        self.place
            .replace(f, stream, projected, self.voxel_rows.view())?;
        self.place
            .replace(f, stream, self.registers.view(), self.register_rows.view())?;
        self.place
            .replace(f, stream, self.zero_row.view(), self.zero_rows.view())?;

        let (rotated, _) = rotary(&self.config)?;
        let half = u64::from(rotated / 2);
        let cosine = f.arena().tensor(DType::F32, Shape::new(&[rows, half]))?;
        let sine = f.arena().tensor(DType::F32, Shape::new(&[rows, half]))?;
        self.rotary.factors(f, positions, cosine, sine)?;

        let (head, heads) = (self.head, self.heads);
        let normalized = f.arena().tensor(DType::F16, Shape::new(&[rows, width]))?;
        let query = f.arena().tensor(DType::F16, Shape::new(&[rows, width]))?;
        let key = f.arena().tensor(DType::F16, Shape::new(&[rows, width]))?;
        let value = f.arena().tensor(DType::F16, Shape::new(&[rows, width]))?;
        let rotated_query = f.arena().tensor(DType::F16, Shape::new(&[rows, width]))?;
        let rotated_key = f.arena().tensor(DType::F16, Shape::new(&[rows, width]))?;
        let attended = f.arena().tensor(DType::F16, Shape::new(&[rows, width]))?;
        let update = f.arena().tensor(DType::F16, Shape::new(&[rows, width]))?;
        let gate_up = f.arena().tensor(DType::F16, Shape::new(&[rows, 2 * ffn]))?;
        let gated = f.arena().tensor(DType::F16, Shape::new(&[rows, ffn]))?;
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
            self.attention.attend_dense(
                f,
                per_head(rotated_query),
                per_head(rotated_key),
                per_head(value),
                None,
                None,
                u32::try_from(rows)
                    .map_err(|_| Error::config("too many tokens for one sequence".to_owned()))?,
                attended,
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
        // The second column of the first row cross-fades 80 pixels of its predecessor.
        assert_eq!(planned[1].left, 80);
        assert_eq!(planned[1].latent_columns, (192 / 16, (192 + 256) / 16));
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
    fn the_latent_statistics_fold_into_the_projection_that_follows() {
        // Two outputs of two inputs: the identity, with the statistics of channels 1 and 2.
        let weight = [1.0, 0.0, 0.0, 1.0];
        let bias = [0.5, -0.5];
        let std = [2.0, 3.0];
        let mean = [10.0, 20.0];
        let (folded, shift) = fold_statistics(&weight, &bias, &std, &mean).expect("the fold");
        assert_eq!(folded, vec![2.0, 0.0, 0.0, 3.0]);
        assert_eq!(shift, vec![10.5, 19.5]);
        // The fold restores what the two steps compute separately: `W (std x + mean) + b`.
        let latent = [0.25f32, -0.5];
        let restored = [latent[0] * std[0] + mean[0], latent[1] * std[1] + mean[1]];
        for output in 0..2 {
            let separate = weight[output * 2] * restored[0]
                + weight[output * 2 + 1] * restored[1]
                + bias[output];
            let together =
                folded[output * 2] * latent[0] + folded[output * 2 + 1] * latent[1] + shift[output];
            assert!(
                (separate - together).abs() < 1e-5,
                "{separate} vs {together}"
            );
        }
        // A projection of the wrong extent, or non-finite statistics, is refused.
        assert!(fold_statistics(&weight, &bias, &[2.0], &mean).is_err());
        assert!(fold_statistics(&weight[..3], &bias, &std, &mean).is_err());
        assert!(fold_statistics(&weight, &bias, &[f32::NAN, 3.0], &mean).is_err());
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
                let want = (turn * value) as f32;
                let got = table[channel * rows + row];
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
                crate::audio_vae::Tensor::Normalized { .. }
                | crate::audio_vae::Tensor::Parts(_) => {
                    panic!("the video decoder folds no weight norm and fuses no tensor")
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
}

/// Fold the latent statistics into the projection that follows them.
///
/// The decoder's first step restores a latent channel (`value · std + mean`) and then applies the
/// checkpoint's one-by-one post-quantization convolution. The two compose into one affine:
/// `W · (std ⊙ x + mean) + b`, i.e. a weight whose input column `i` is scaled by `std[i]` and a bias
/// shifted by `W · mean`. Folding them at the loading boundary keeps one element type and one
/// operator, with the checkpoint's own rounding of the stored result.
///
/// `weight` is `[out, in]` row-major, `bias` is `[out]`, and `std`/`mean` are `[in]`. The result is the
/// folded `(weight, bias)`; a checkpoint whose extents disagree is refused rather than read past.
pub fn fold_statistics(
    weight: &[f32],
    bias: &[f32],
    std: &[f32],
    mean: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    let out = bias.len();
    if out == 0 || mean.len() != std.len() || mean.is_empty() {
        return Err(Error::config(
            "the latent statistics and the projection's channels disagree".to_owned(),
        ));
    }
    let channels = mean.len();
    if weight.len() != out * channels {
        return Err(Error::config(format!(
            "a {}-element projection for {out} outputs of {channels} inputs",
            weight.len()
        )));
    }
    if std.iter().any(|value| !value.is_finite()) || mean.iter().any(|value| !value.is_finite()) {
        return Err(Error::config(
            "the latent statistics are not finite".to_owned(),
        ));
    }
    let mut folded = weight.to_vec();
    let mut shift = bias.to_vec();
    for channel in 0..channels {
        for output in 0..out {
            let at = output * channels + channel;
            folded[at] = weight[at] * std[channel];
            shift[output] += weight[at] * mean[channel];
        }
    }
    Ok((folded, shift))
}
