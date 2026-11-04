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
    /// The canvas of one clip of `config`: 7 latent frames at the given latent extents.
    pub fn clip(config: &VideoVaeConfig, height: u32, width: u32) -> Self {
        Self {
            frames: config.clip_latents() + config.token_drop.div_ceil(config.clip_latents()),
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
    pub fn split(
        length: u32,
        tile: u32,
        overlap: u32,
        alignment: u32,
    ) -> Result<Self> {
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
        while u64::from(tile) * u64::from(count)
            - u64::from(overlap) * u64::from(count - 1)
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
                let (column_start, column_end) =
                    self.columns.latent(column, canvas.width, ratio);
                let above = row.checked_sub(1).map_or(0, |previous| {
                    self.rows.overlap(previous).unwrap_or(0)
                });
                let left = column.checked_sub(1).map_or(0, |previous| {
                    self.columns.overlap(previous).unwrap_or(0)
                });
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

#[cfg(test)]
mod tests {
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
        assert_eq!(width, seams(&[0, 176, 352, 528, 704, 896, 1088], &[80, 80, 80, 80, 64, 64], 1344));
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
                assert_eq!(split.starts[index] + split.overlaps[index - 1], previous_end);
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
        assert!(checkpoint.contains(&"decoder.transformer_blocks.35.ff.net.0.proj.weight".to_owned()));
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
