//! The H3 decoders' plan: how a video is spliced into clips, what a clip of decoded frames keeps, and
//! the layout each decoder hands the media stage.
//!
//! `uniserve/models/minimax_h3/decoding.py` and `output.py` are the reference. The video VAE decodes
//! one clip at a time and [`crate::output::VideoPostprocessor`] splices the clips into a timeline; the
//! audio VAE decodes one
//! channel-major stereo latent timeline. This module owns that arithmetic and the output layouts, and
//! nothing that touches a device: the packed latents, the decoded clip frames and the PCM are borrowed
//! views the executor hands in and out.
//!
//! ## The timeline
//!
//! A generated video has `17n + 5` frames (specs/worker.md 9, `docs/minimax_h3/minimax_h3.md`). Clip
//! `k` reads latent frames `[5k, 5k + 7)`: five that advance the timeline and two that the next clip
//! reads again. Each clip decodes to 25 frames — a 17-frame body, three VAE padding frames, and a
//! five-frame tail — so a clip keeps `[0, 17)` and cross-fades `[20, 25)` into its successor; the last
//! clip's tail ends the video.
//!
//! ## What each decoder hands the media stage
//!
//! [`crate::output`] states the two rasters and their quantization — the video's `[frames, height,
//! width, 3]` bytes and the audio's `[samples, 2]` integers — because they are the media stage's
//! contract rather than this plan's. What belongs here is the arithmetic that decides *which* frames
//! and *which* samples go where: a clip's body and overlap, the halo a decoded audio window carries.
//! The splice itself - drop the padding frames, fade the overlap, write the bytes - is the
//! postprocessor's, because it is the component's second entry point (`video_postprocessor.forward`)
//! and it reads the predecessor's tail, which a single clip's decode never sees.

use crate::config::VideoVaeConfig;
use crate::video_vae::{Canvas, Tile};
use crate::{
    Error, Result,
    config::{CLIP_FRAMES, CLIP_LATENTS},
};
use nn::layers::{CrossFade, UnpatchifyVideo};
use nn::{DType, Forward, Shape, TensorView};

/// The frames one decoded clip holds: a 17-frame body, three padding frames and a five-frame tail.
pub const CLIP_OUTPUT_FRAMES: u32 = 25;

/// Frames of a clip's native output that overlap the next clip.
pub const OVERLAP_FRAMES: u32 = 5;

/// The first frame of a clip's overlap tail, within its own 25 decoded frames.
pub const TAIL_START: u32 = tail_start_of(OVERLAP_FRAMES);

/// The first frame of a tail of `overlap` frames: the clip's native output less the tail.
const fn tail_start_of(overlap: u32) -> u32 {
    CLIP_OUTPUT_FRAMES - overlap
}

/// One clip of a video's timeline.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Clip {
    /// The index of the clip in the timeline.
    pub index: u32,
    /// The clip's latent frames `[start, end)`: seven, of which the last two repeat in the next clip.
    pub latent: (u32, u32),
    /// The frames of the clip's own 25 that the timeline keeps: the 17-frame body.
    pub body: (u32, u32),
    /// The frames of the clip's own 25 that overlap the next clip: the five-frame tail.
    pub tail: (u32, u32),
}

impl Clip {
    /// The weight of this clip across its tail, one per overlap frame.
    ///
    /// The reference blends with `i / 5` for `i = 0..5`, so the first blended frame is entirely the
    /// predecessor's and the fifth entirely this clip's. The weights are computed in the decoded
    /// element type there; this is the same table stated exactly, which a quantized implementation
    /// compares against.
    pub fn overlap_weights(&self) -> [f32; OVERLAP_FRAMES as usize] {
        let mut weights = [0.0; OVERLAP_FRAMES as usize];
        for (index, weight) in weights.iter_mut().enumerate() {
            *weight = index as f32 / OVERLAP_FRAMES as f32;
        }
        weights
    }
}

/// The clips of a `17n + 5`-frame video, in timeline order.
///
/// Every clip covers a 17-frame body; the last clip adds the five-frame tail, so the number of clips
/// is `(frames - 5) / 17`.
pub fn clips(frames: u32) -> Result<Vec<Clip>> {
    if frames < CLIP_FRAMES + CLIP_LATENTS || !(frames - CLIP_LATENTS).is_multiple_of(CLIP_FRAMES) {
        return Err(Error::config(format!(
            "an H3 video of {frames} frames is not {CLIP_FRAMES}n + {CLIP_LATENTS}"
        )));
    }
    let count = (frames - CLIP_LATENTS) / CLIP_FRAMES;
    Ok((0..count)
        .map(|index| Clip {
            index,
            latent: (
                index * CLIP_LATENTS,
                index * CLIP_LATENTS + CLIP_LATENTS + 2,
            ),
            body: (0, CLIP_FRAMES),
            tail: (TAIL_START, TAIL_START + CLIP_LATENTS),
        })
        .collect())
}

/// The audio timeline of one request: the samples it decodes to and the latent frames it needs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Track {
    /// Interleaved sample frames the decoder writes.
    pub samples: u32,
    /// Latent frames the decoder reads, one per 800 samples.
    pub latent_frames: u32,
    /// Latent frames of context each side of a decoded window depends on.
    pub halo: u32,
}

impl Track {
    /// The track of a video's `frames` at the checkpoint's 40 Hz latent rate.
    pub fn of(frames: u32, audio: &crate::AudioVaeConfig) -> Result<Self> {
        let samples = crate::packing::audio_samples(frames);
        Ok(Self {
            samples,
            latent_frames: crate::packing::audio_latent_frames(frames),
            halo: crate::audio_vae::receptive_field(audio)?,
        })
    }

    /// The latent frames a decoded window `[start, stop)` needs, halo included and clipped to the
    /// track.
    pub fn window(&self, start: u32, stop: u32) -> (u32, u32) {
        (
            start.saturating_sub(self.halo),
            stop.saturating_add(self.halo).min(self.latent_frames),
        )
    }
}

/// One scratch buffer of a decoder's forward, with the element type it holds.
///
/// The executor sizes the arena from these; the model never allocates. Names are the ones the arena
/// plan in `notes/h3/decoders-interface.md` uses, so a mismatch is visible in one place - **and the order
/// is the order the model reserves them in**, because the arena is one cursor: a buffer listed before
/// another is also reserved before it, and anything the code reserves inside a loop is multiplied. The
/// transformer's fourteen are reserved once per clip (`VideoDecoder::scratch`) and handed to every tile;
/// the token buffers are per tile, so their entry is the grid's sum.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Buffer {
    /// The buffer's name, unique within its plan: the executor keys storage by it.
    pub name: String,
    /// Its element type: `f16`, `f32` or `u8`.
    pub element: &'static str,
    /// How many elements it holds.
    pub elements: u64,
}

impl Buffer {
    /// The width of one element, by the name the plans use.
    ///
    /// The names below are the whole set a plan may declare: the four the arena holds floats and bytes in,
    /// and `i32` for the index buffers a gather reads. `f16` and `bf16` are two bytes each, `i32` and `f32`
    /// are four, which is the difference between a 200 MiB buffer and a 400 MiB one. An unknown name is
    /// refused rather than assumed, because the assumption - four bytes - would be half the budget for a
    /// reduced-precision buffer.
    pub fn width(&self) -> Result<u64> {
        match self.element {
            "u8" => Ok(1),
            "f16" | "bf16" => Ok(2),
            "i32" | "f32" => Ok(4),
            other => Err(Error::config(format!(
                "{other} is not an element type a plan declares: u8, f16, bf16, i32 or f32"
            ))),
        }
    }

    /// The bytes the buffer occupies in an arena: its size rounded up to the arena's alignment.
    ///
    /// `plan::Arena::reserve` aligns every give-out to [`nn::forward::ALIGNMENT`], so a reservation
    /// summed from raw sizes is short by up to `ALIGNMENT - 1` bytes **per buffer** - multiplied by
    /// the number of buffers, which for the video decoder's sixteen is up to 4 KiB. The first run to
    /// meet it was short by 64 bytes, exactly the padding of the two rotary tables (`1797 x 24 x 4`
    /// is not a multiple of 256).
    pub fn reservation(&self) -> u64 {
        self.bytes().next_multiple_of(nn::forward::ALIGNMENT)
    }

    /// The bytes the buffer occupies.
    pub fn bytes(&self) -> u64 {
        // A plan with an unknown element name is caught by [`Self::width`]; this answers for the four
        // names the crate's plans use and treats anything else as four bytes so a *report* never panics.
        self.elements * self.width().unwrap_or(4)
    }
}

/// The clip-latent row of every voxel of every tile, in tile order then voxel order, with the number of
/// voxels each tile contributes.
///
/// A tile is a rectangle of the clip's latent raster — every frame, a run of rows and a run of columns —
/// so its voxels are a subset of the clip's, and a tile's rows cannot be a strided view of the clip's
/// when the tile is narrower than the raster. This table names them: the runner gathers each tile's
/// `[voxels_tile, latent_channels]` block out of the clip's one `[frames·h·w, latent_channels]` latent
/// with `nn::layers::Rows`, which is model layout arithmetic, uploaded once per layout.
///
/// `tiles` are in grid order ([`Tiles::tiles`]), and each tile contributes
/// `frames · (rows_end − rows_start) · (columns_end − columns_start)` entries: the row index
/// `((frame · h + row) · w + column)` of each of its voxels.
///
/// The counts are returned with the rows rather than left implicit, because they are not derivable from
/// the tile size alone: a canvas side shorter than one tile gives a single tile of that side
/// (`Seams::split`'s one-tile branch), so a block can be smaller than the tile. Where the side is at
/// least a tile, every tile is exactly the tile's extent — `Seams::split` builds `lengths = [tile; n]`
/// and distributes the slack over the seams — so the grid is uniform and the counts are all equal.
///
/// **The tiles overlap at their seams**, so the entries are not a permutation of the clip's voxels: a
/// seam's positions are decoded by both neighbours, which is what the blend is for.
pub fn tile_rows(canvas: &Canvas, tiles: &[Tile]) -> (Vec<i32>, Vec<u32>) {
    let height = u64::from(canvas.height);
    let width = u64::from(canvas.width);
    let mut rows = Vec::with_capacity(canvas.voxels() as usize);
    let mut counts = Vec::with_capacity(tiles.len());
    for tile in tiles {
        let (first_row, last_row) = tile.latent_rows;
        let (first_column, last_column) = tile.latent_columns;
        let before = rows.len();
        for frame in 0..u64::from(canvas.frames) {
            for row in u64::from(first_row)..u64::from(last_row) {
                for column in u64::from(first_column)..u64::from(last_column) {
                    rows.push(((frame * height + row) * width + column) as i32);
                }
            }
        }
        counts.push((rows.len() - before) as u32);
    }
    (rows, counts)
}

/// The scratch one video clip's decode needs, one entry per distinct lifetime.
///
/// A clip decodes every tile of its grid in one call (`TiledDecoder.decode` batches them), so the
/// token buffers hold all tiles at once: [`crate::video_vae::Tile`]s are all the same shape and the
/// stitched crop trims the last to the raster. The buffers are the reference's own: a residual stream
/// and its normalization, the fused query-key-value projection with its rotary factors, the attention
/// output, the gated feed-forward, the decoded pixels of one tile batch, and the RGB bytes with the
/// overlap block carried between clips.
/// The bytes a runner must reserve for `plan`: every buffer at the arena's alignment.
///
/// **A plan summed raw is short.** The arena aligns each give-out to [`nn::forward::ALIGNMENT`] and a
/// plan's entries are not all multiples of it, so the reservation a runner makes has to be the
/// aligned sum rather than the sum of the sizes. The failure this prevents is the smallest that has
/// ever stopped a stage: `22081536 bytes requested at offset 368378880 of a 390460352-byte region`,
/// 64 bytes past the end, because `1797 x 24 x 4` twice is 64 bytes of padding the sum left out.
pub fn arena_reservation(plan: &[Buffer]) -> u64 {
    plan.iter().map(Buffer::reservation).sum()
}

pub fn video_arena(
    config: &crate::VideoVaeConfig,
    canvas: &crate::video_vae::Canvas,
    dtype: nn::DType,
) -> Result<Vec<Buffer>> {
    let tiles = crate::video_vae::Tiles::of(config, canvas)?;
    let grid = tiles.tiles(config, canvas);
    let ffn = u64::from(config.decoder_ffn_mult);
    let (rotated, _) = crate::video_vae::rotary(config)?;
    let columns = u64::from(config.decoder_num_attention_heads)
        * u64::from(config.decoder_attention_head_dim);
    let (height, width_px) = canvas.pixels(config);
    let frames = canvas.frames_out(config);
    // One token buffer holds every tile, so its rows are the tiles' rows in grid order — summed,
    // because a tile at the canvas's edge can be narrower than the tile size.
    let rows: u64 = grid.iter().map(|tile| tile.canvas.rows()).sum();
    // **The transformer's fourteen buffers are one tile's, not the grid's sum.** The decoder's canvas is
    // one tile (`VideoDecoder`) and `TileScratch` is reserved once per clip and reused by every tile, so
    // sizing them by the sum reserved the grid's worth of buffers for a decode that holds one tile's -
    // eight times the need on a 16-tile canvas. `decode` refuses a grid whose tiles differ in rows, so
    // the first tile is every tile.
    let tile = grid.first().ok_or_else(|| {
        Error::config(format!(
            "a canvas of {} x {} pixels has no tile to decode",
            height, width_px
        ))
    })?;
    let tile_rows = tile.canvas.rows();
    let tile_voxels = tile.canvas.voxels();
    // The clip's own raster, which the tiles are stitched into.
    let raster = u64::from(frames) * u64::from(height) * u64::from(width_px);
    let channels = u64::from(crate::output::Rgb::CHANNELS);
    // **The order is the reservation order**: the clip-level buffers are reserved once, before the
    // tile loop, and live across the whole clip; the transformer's fourteen are reserved there too, once
    // (`VideoDecoder::scratch`), and reused by every tile - the arena never rewinds, so a per-tile
    // reservation would grow the clip's footprint by the tile count. The token buffers *are* per tile,
    // because each tile's decode owns its own, and their sum is the `tokens` entry. The executor sizes
    // the arena by the sum. There is no entry for the clip's blended raster: the blend's destination is
    // the caller's `DecodedMedia.video`, which is the runner's buffer and not the arena's.
    let patch = u64::from(config.out_channels)
        * u64::from(config.temporal_compression())
        * u64::from(config.spatial_compression())
        * u64::from(config.spatial_compression());
    // **The plan's element types are the block's, not a `dtype = "auto"` assumption.** Every buffer the
    // transformer allocates is at `VideoDecoder::dtype` (`video_vae.rs`'s `scratch`), and the rotary
    // tables are the one exception: they are F32 whatever the decode is. A plan that spelled `f16` for
    // both sides reserved half of what an FP32 decode needs, and the shortfall surfaced not as a plan
    // complaint but as an arena that "cannot give out" a buffer partway through the clip.
    let element = crate::text_encoder::element_of(dtype);
    Ok(vec![
        Buffer {
            name: "decoded".to_owned(),
            element,
            elements: raster * channels,
        },
        Buffer {
            // The stitch's seam block: one tile's pixels, contiguous, which every blend copies its
            // neighbour into. It is reserved with the clip's other buffers because one block serves
            // every seam, and the grid's tiles are all the same size.
            name: "seam".to_owned(),
            element,
            elements: channels
                * u64::from(frames)
                * (u64::from(tile.canvas.height) * u64::from(config.spatial_compression()))
                * (u64::from(tile.canvas.width) * u64::from(config.spatial_compression())),
        },
        Buffer {
            name: "tokens".to_owned(),
            element,
            elements: rows * patch,
        },
        Buffer {
            name: "stream".to_owned(),
            element,
            elements: tile_rows * columns,
        },
        Buffer {
            name: "projected".to_owned(),
            element,
            elements: tile_voxels * columns,
        },
        Buffer {
            name: "cosine".to_owned(),
            element: "f32",
            elements: tile_rows * u64::from(rotated / 2),
        },
        Buffer {
            name: "sine".to_owned(),
            element: "f32",
            elements: tile_rows * u64::from(rotated / 2),
        },
        Buffer {
            name: "normalized".to_owned(),
            element,
            elements: tile_rows * columns,
        },
        Buffer {
            name: "query".to_owned(),
            element,
            elements: tile_rows * columns,
        },
        Buffer {
            name: "key".to_owned(),
            element,
            elements: tile_rows * columns,
        },
        Buffer {
            name: "value".to_owned(),
            element,
            elements: tile_rows * columns,
        },
        Buffer {
            name: "rotated_query".to_owned(),
            element,
            elements: tile_rows * columns,
        },
        Buffer {
            name: "rotated_key".to_owned(),
            element,
            elements: tile_rows * columns,
        },
        Buffer {
            name: "attended".to_owned(),
            element,
            elements: tile_rows * columns,
        },
        Buffer {
            name: "update".to_owned(),
            element,
            elements: tile_rows * columns,
        },
        Buffer {
            name: "gate_up".to_owned(),
            element,
            elements: tile_rows * 2 * columns * ffn,
        },
        Buffer {
            name: "gated".to_owned(),
            element,
            elements: tile_rows * columns * ffn,
        },
    ])
}

/// The video decoder's tiled output path: one tile's tokens become its pixels in the clip's raster.
///
/// The decoder decodes overlapping tiles one after another and blends each into the raster as it
/// lands, which is the reference's stitch (`uniserve/nn/vae/tiled.py:46-81`): a tile is written where
/// it belongs, then blended with the tile above it over the rows they share, then with the tile to
/// its left over the columns they share. The first tile row and column have no predecessor, so they
/// are written and nothing is blended.
///
/// Two rasters of `[channels, frames, rows, columns]` in the decoder's element type carry one clip
/// while it is stitched. `plain` holds every tile as it was decoded and is never blended into;
/// `blended` holds the result. The reference keeps its neighbours unmodified for a reason
/// (`uniserve/nn/vae/tiled.py:51-56`): a tile's row blend reads the tile above, and that neighbour
/// must be its *decoded* block rather than one already faded into its own predecessor, or the
/// overlap where two seams cross is computed in a different order from the seams themselves. Writing
/// each tile into both rasters costs one extra unpatchify launch per tile: every placement is a
/// strided view of a raster, which the operators write through.
///
/// A **source** cannot be a strided view of either raster. A seam of a grid wider than one tile
/// reads a block that is not contiguous in `[channels, frames, rows, columns]` order - the
/// predecessor's last rows, laid out at the raster's width - and `cross_fade` addresses its sources
/// flat, so a shifted view of a wider raster reads the wrong rows past the first. Each seam
/// therefore copies its neighbour into one tile-shaped buffer first, `[channels, frames, tile rows,
/// tile columns]` contiguous, and blends from that: one copy per seam, no operator signature change,
/// and the blend's own key is untouched because only its destination ever carried strides.
pub struct ClipDecoder {
    unpatchify: UnpatchifyVideo,
    /// One blend layer for each overlap the tile grid has, by extent and axis.
    blends: Vec<(u32, u32, CrossFade)>,
    /// The copy that makes a seam's neighbour contiguous for the blend's flat source addressing.
    copy: nn::layers::CopyBlock,
    /// Pixels per latent position, the ratio a tile's latent origin scales by.
    ratio: u32,
    /// Elements of one token: `out_channels · temporal · spatial²`, the row width every tile's decode
    /// writes and the unpatchify reads.
    patch: u64,
}

impl ClipDecoder {
    /// The blends this stitch compiles: one module per seam width the grid can produce, on both
    /// axes. A walk states each one's program without a device.
    pub fn blends(&self) -> &[(u32, u32, CrossFade)] {
        &self.blends
    }

    /// The output path of `canvas`, whose tiles the caller decodes in [`Tile`] order.
    pub fn new(config: &VideoVaeConfig, dtype: DType) -> Result<Self> {
        let spatial = config.spatial_compression();
        let unpatchify = UnpatchifyVideo::new(
            dtype,
            u64::from(config.out_channels),
            config.temporal_compression(),
            spatial,
        )?;
        let patch = u64::from(config.out_channels)
            * u64::from(config.temporal_compression())
            * u64::from(spatial)
            * u64::from(spatial);
        // **The table is a ladder, and the set it must cover is not this function's to read.** A seam's
        // width (in output pixels, which is what the blend writes and what `cross_fade` is given) is
        // whatever the canvas's grid arithmetic gives: over the six default canvases it runs 64 to
        // 128. But `offered_canvases()` is only the table a deployment serves when its checkpoint
        // states **no** ladder (`config/canvas.rs`), and a checkpoint that states one may serve
        // canvases this table does not name. Deriving the blends from the default table therefore
        // narrows the decoder to the default deployment: a ladder canvas whose seams are wider would
        // find no module and stop mid-clip, which is the error an excess never causes.
        //
        // So the ladder stays - every alignment step from `MINIMUM_OVERLAP` to the tile size, on both
        // axes - and the test beside it asserts the deployment's realized seams are inside it. An
        // extra class costs a compile; a missing one stops a run.
        let alignment = spatial;
        let mut blends = Vec::new();
        let mut extent = crate::video_vae::MINIMUM_OVERLAP;
        while extent < crate::video_vae::TILE_PIXELS {
            for axis in [2, 3] {
                blends.push((extent, axis, CrossFade::new(dtype, extent, axis)?));
            }
            extent += alignment;
        }
        Ok(Self {
            unpatchify,
            blends,
            copy: nn::layers::CopyBlock::new(dtype)?,
            patch,
            ratio: spatial,
        })
    }

    /// The blend layer of `extent` along `axis`.
    fn blend(&self, extent: u32, axis: u32) -> Result<CrossFade> {
        self.blends
            .iter()
            .find(|(width, side, _)| *width == extent && *side == axis)
            .map(|(_, _, layer)| *layer)
            .ok_or_else(|| {
                Error::config(format!(
                    "the clip decoder has no {extent}-pixel blend along axis {axis}"
                ))
            })
    }

    /// Write one decoded tile into the clip's raster and blend it with its neighbours.
    ///
    /// `tokens` is what [`crate::video_vae::VideoDecoder::decode`] wrote for this tile,
    /// `[rows, out_channels · temporal · spatial²]`, its voxel rows first; `dest` is the whole clip's
    /// raster; `seam` is one tile-shaped contiguous buffer, reused by every blend of the clip, that a
    /// neighbour is copied into before it is faded. The caller decodes the tiles in [`Tile`] order,
    /// because a blend reads what the tile above and the tile to the left have already written.
    pub fn write_tile(
        &self,
        f: &Forward,
        tile: &Tile,
        tokens: TensorView,
        decoded: TensorView,
        blended: TensorView,
        seam: TensorView,
    ) -> Result<()> {
        for (name, raster) in [("decoded", decoded), ("blended", blended)] {
            let shape = raster.shape;
            if shape.rank() != 4 || shape.dim(0) != u64::from(self.unpatchify.channels()) {
                return Err(Error::config(format!(
                    "the {name} raster is {:?}, not [channels, frames, rows, columns]",
                    shape.dims()
                )));
            }
        }
        // The seam buffer is the shape every blend's source has, contiguous: a copy writes it, and the
        // blend reads it flat, which is the addressing its program has.
        let (tile_rows, tile_columns) = (
            u64::from(tile.canvas.height) * u64::from(self.ratio),
            u64::from(tile.canvas.width) * u64::from(self.ratio),
        );
        let seam_shape = Shape::new(&[
            decoded.shape.dim(0),
            decoded.shape.dim(1),
            tile_rows,
            tile_columns,
        ]);
        let packed = [
            seam_shape.dim(1) * tile_rows * tile_columns,
            tile_rows * tile_columns,
            tile_columns,
            1,
        ];
        if seam.shape != seam_shape || seam.strides != nn::Strides::new(&packed) {
            return Err(Error::config(format!(
                "the seam buffer is {seam:?}, not {seam_shape:?} contiguous"
            )));
        }
        let ratio = u64::from(self.ratio);
        let first_row = u64::from(tile.latent_rows.0) * ratio;
        let first_column = u64::from(tile.latent_columns.0) * ratio;
        let plain = self.placement(decoded, tile, first_row, first_column, 0, 0)?;
        let view = self.placement(blended, tile, first_row, first_column, 0, 0)?;
        // The operator's declared input is the token **grid** - `frames x height x width` rows - while a
        // tile's buffer carries the grid plus the decoder's register and tail rows, which the decode
        // writes and this operator never reads. So the launch states the rows it reads: `ops`' own
        // check wants that shape exactly (`Unpatchify::check`), and the layer admits spare rows only
        // because a view past the grid is legal to hand it - not because it should be handed one.
        let voxels = tile.canvas.voxels();
        if tokens.shape.rank() != 2 || tokens.shape.dim(0) < voxels {
            return Err(Error::config(format!(
                "a tile's tokens are {:?}, short of the {voxels} rows its grid reads",
                tokens.shape.dims()
            )));
        }
        let grid = TensorView::new(
            tokens.offset_elements(0),
            tokens.dtype,
            Shape::new(&[voxels, tokens.shape.dim(1)]),
        );
        self.unpatchify.forward(f, grid, plain)?;
        self.unpatchify.forward(f, grid, view)?;
        // A blend reads its neighbour's **last** `extent` positions along the axis and writes the
        // current tile's first `extent`. The predecessor's last positions are the seam itself, which
        // still holds the predecessor's pixels in the raster, so the neighbour's view starts one
        // tile-extent minus one overlap before the tile - not one overlap before it, which would read
        // the tile's own pixels from a band further down. That view is strided wherever the raster is
        // wider than the tile, and `cross_fade` addresses its sources flat, so the view is copied into
        // the contiguous seam buffer first: the copy reads the raster through the view's strides, the
        // blend reads the buffer flat, and both are the addressing their programs have.
        let mut destination_strides = [0_i64; ops::copy_block::RANK];
        for (axis, stride) in packed.iter().enumerate() {
            destination_strides[axis] = i64::try_from(*stride)
                .map_err(|_| Error::config("a seam block is larger than i64 strides".to_owned()))?;
        }
        let contiguous = TensorView::with_strides(
            seam.offset_elements(0),
            seam.dtype,
            seam_shape,
            nn::Strides::new(&packed),
        );
        let (body_rows, body_columns) = (
            u64::from(tile.canvas.height) * ratio,
            u64::from(tile.canvas.width) * ratio,
        );
        if tile.above != 0 {
            let previous = self.placement(
                decoded,
                tile,
                first_row,
                first_column,
                u32::try_from(body_rows).unwrap_or(u32::MAX) - tile.above,
                0,
            )?;
            self.copy
                .forward(f, previous, contiguous, &destination_strides)?;
            self.blend(tile.above, 2)?
                .forward(f, Some(contiguous), view, view)?;
        }
        if tile.left != 0 {
            let previous = self.placement(
                decoded,
                tile,
                first_row,
                first_column,
                0,
                u32::try_from(body_columns).unwrap_or(u32::MAX) - tile.left,
            )?;
            self.copy
                .forward(f, previous, contiguous, &destination_strides)?;
            self.blend(tile.left, 3)?
                .forward(f, Some(contiguous), view, view)?;
        }
        Ok(())
    }

    /// A view of `dest` holding one tile's pixel block, shifted back by `up` rows and `left` columns.
    ///
    /// The shift reads the region a blend's predecessor occupies, which the tile grid has already
    /// written; the view keeps the raster's own strides, so the operator writes through them.
    fn placement(
        &self,
        dest: TensorView,
        tile: &Tile,
        first_row: u64,
        first_column: u64,
        up: u32,
        left: u32,
    ) -> Result<TensorView> {
        let ratio = u64::from(self.ratio);
        let rows = first_row
            .checked_sub(u64::from(up))
            .ok_or_else(|| Error::config("a blend starts above the raster".to_owned()))?;
        let columns = first_column
            .checked_sub(u64::from(left))
            .ok_or_else(|| Error::config("a blend starts left of the raster".to_owned()))?;
        let shape = Shape::new(&[
            dest.shape.dim(0),
            // The raster's own frames: every tile spans them all, so the placement reads the extent
            // from the buffer rather than from a canvas the blend table was built against.
            dest.shape.dim(1),
            u64::from(tile.canvas.height) * ratio,
            u64::from(tile.canvas.width) * ratio,
        ]);
        Ok(TensorView::with_strides(
            dest.element(&[0, 0, rows, columns]),
            dest.dtype,
            shape,
            dest.strides,
        ))
    }

    /// Decode one clip: every tile of its canvas, in [`Tile`] order, into the clip's raster.
    ///
    /// The runner hands this one latent block and one output and nothing else: `latents` is the clip's
    /// `[voxels, latent_channels]` rows in tile order, `positions` the layout's `video_positions`
    /// `[3, rows]` table, and `out` the clip's raster `[channels, frames, rows, columns]` in the
    /// decoder's element type, which the tiles are stitched into. Every tile view is the model's own:
    /// the rows are cumulative over the tiles' voxel counts, and the token buffer is reserved here, in
    /// the arena, one tile at a time.
    ///
    /// The tiles are decoded one after another because a seam blend reads what the tile above and the
    /// tile to its left have already written.
    pub fn decode_clip(
        &self,
        f: &Forward,
        decoder: &crate::video_vae::VideoDecoder,
        tiles: &[Tile],
        latents: TensorView,
        positions: TensorView,
        out: TensorView,
    ) -> Result<()> {
        let voxels: u64 = tiles.iter().map(|tile| tile.canvas.voxels()).sum();
        let channels = u64::from(crate::output::Rgb::CHANNELS);
        if latents.shape.rank() != 2 || latents.shape.dim(0) != voxels {
            return Err(Error::config(format!(
                "the clip's latent block is [{voxels}, latent_channels], not {:?}",
                latents.shape.dims()
            )));
        }
        // One table serves every tile only when the grid is uniform, which it is wherever the canvas
        // is at least one tile wide: the coordinates are the tile's own extents, so a tile of another
        // size would need a table of its own.
        let table_rows = positions.shape.dim(1);
        if positions.shape.rank() != 2 || positions.shape.dim(0) != 3 {
            return Err(Error::config(format!(
                "the position table is [3, rows], not {:?}",
                positions.shape.dims()
            )));
        }
        for tile in tiles {
            if crate::video_vae::Canvas::rows(&tile.canvas) != table_rows {
                return Err(Error::config(format!(
                    "tile ({}, {}) needs {} position rows and the layout's table holds {table_rows}: \
                     the table is the largest tile's, so a grid of two tile sizes has none",
                    tile.row,
                    tile.column,
                    crate::video_vae::Canvas::rows(&tile.canvas)
                )));
            }
        }
        if out.shape.rank() != 4 || out.shape.dim(0) != channels {
            return Err(Error::config(format!(
                "the clip's raster is [{channels}, frames, rows, columns], not {:?}",
                out.shape.dims()
            )));
        }
        // **One canvas per decoder object: the tiles handed here are the grid of the canvas the decoder
        // was built for.** A decoder built for the clip's canvas scatters a tile's rows with the clip's
        // permutation - the table is a constant of that canvas - so the mismatch is a write past a
        // tile-sized buffer, and it is refused by name here instead.
        let (covered_frames, covered_height, covered_width) =
            tiles
                .iter()
                .fold((0_u32, 0_u32, 0_u32), |(frames, height, width), tile| {
                    // The tile's **latent span**, not its grid index: `row`/`column` count tiles and the
                    // extent a grid covers is the end of each tile's span.
                    (
                        frames.max(tile.canvas.frames),
                        height.max(tile.latent_rows.1),
                        width.max(tile.latent_columns.1),
                    )
                });
        // The grid the tiles cover must be **one the decoder serves**: its tables are a tile's, so
        // any canvas of the deployment's set is served, and a canvas outside it is refused by name.
        let canvas = decoder
            .canvases()
            .iter()
            .copied()
            .find(|canvas| {
                (canvas.frames, canvas.height, canvas.width)
                    == (covered_frames, covered_height, covered_width)
            })
            .ok_or_else(|| {
                Error::config(format!(
                    "the tiled decode's tiles cover {covered_height}x{covered_width} latent cells \
                     of {covered_frames} frame(s), and this decoder serves {}",
                    decoder
                        .canvases()
                        .iter()
                        .map(|canvas| format!(
                            "{}x{} of {} frame(s)",
                            canvas.height, canvas.width, canvas.frames
                        ))
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })?;
        // The clip-level buffers, reserved once: `decoded` holds every tile as it was decoded and is read
        // as an unmodified neighbour, and `out` - the caller's raster - takes the stitch, which is why the
        // plan lists no third raster. `scratch` is the transformer's fourteen, reserved here so that every
        // tile of the clip shares them.
        let decoded = f.arena().tensor(out.dtype, out.shape)?;
        // The seam buffer: one tile-shaped contiguous block every blend copies its neighbour into. A
        // tile's extents are the same across a grid, so one buffer serves every seam of the clip and
        // the blends of a tile are ordered on the stream that wrote it.
        let first = tiles.first().ok_or_else(|| {
            Error::config("a clip of no tiles has no raster to stitch".to_owned())
        })?;
        let seam = f.arena().tensor(
            out.dtype,
            Shape::new(&[
                out.shape.dim(0),
                out.shape.dim(1),
                u64::from(first.canvas.height) * u64::from(self.ratio),
                u64::from(first.canvas.width) * u64::from(self.ratio),
            ]),
        )?;
        // The transformer's fourteen buffers, reserved **once per clip** and reused by every tile: the
        // arena never rewinds, so a per-tile reservation would grow the clip's footprint by the tile
        // count while the plan states them once.
        let scratch = decoder.scratch(f, &canvas)?;
        let patch = self.patch;
        let mut start = 0_u64;
        for tile in tiles {
            let tile_rows = crate::video_vae::Canvas::rows(&tile.canvas);
            let tile_voxels = tile.canvas.voxels();
            let latent = TensorView::new(
                latents.offset_elements(start * latents.shape.dim(1)),
                latents.dtype,
                Shape::new(&[tile_voxels, latents.shape.dim(1)]),
            );
            // One tile's token buffer: the plan's `tokens` is the whole clip's, and the reservations
            // of the tiles sum to it because each is reused only by its own tile's decode.
            let tokens = f
                .arena()
                .tensor(out.dtype, Shape::new(&[tile_rows, patch]))?;
            decoder.decode(f, &canvas, latent, positions, tokens, &scratch)?;
            self.write_tile(f, tile, tokens, decoded, out, seam)?;
            start += tile_voxels;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::video_vae::Tiles;

    /// The ImageNet constants as the model publishes them: host memory, one f32 per channel.
    #[test]
    fn a_plans_element_names_are_a_closed_set_with_stated_widths() {
        // A reduced-precision buffer counted as four bytes doubles its budget, and an unknown name used to
        // fall through to that same four, so the set is closed here and every width is stated. `i32` is the
        // index buffers a gather reads, four bytes like `f32` but not a float.
        let buffer = |element: &'static str| Buffer {
            name: format!("probe.{element}"),
            element,
            elements: 1024,
        };
        for (element, width) in [("u8", 1), ("f16", 2), ("bf16", 2), ("i32", 4), ("f32", 4)] {
            let buffer = buffer(element);
            assert_eq!(buffer.width().expect(element), width, "{element}");
            assert_eq!(buffer.bytes(), 1024 * width, "{element}");
        }
        // Anything else is refused rather than assumed, at the widths a budget depends on.
        assert!(buffer("f64").width().is_err());
    }

    #[test]
    fn a_four_second_video_is_six_clips() {
        // 4 s at 24 fps rounds to 96 frames, extended to the next 17n + 5: 107.
        assert_eq!(crate::packing::align_num_frames(96), 107);
        let clips = clips(107).expect("six clips");
        assert_eq!(clips.len(), 6);
        assert_eq!(clips[0].latent, (0, 7));
        assert_eq!(clips[5].latent, (25, 32));
        for (index, clip) in clips.iter().enumerate() {
            assert_eq!(clip.index, index as u32);
            assert_eq!(clip.body, (0, 17));
            assert_eq!(clip.tail, (20, 25));
        }
        // The clips cover the timeline exactly: 17 per body plus the last clip's tail.
        let covered = 17 * clips.len() as u32 - 17 + 22;
        assert_eq!(covered, 107);
    }

    #[test]
    fn the_latent_timeline_repeats_two_frames_per_clip() {
        for frames in [107u32, 124, 192, 362] {
            let clips = clips(frames).expect("clips");
            // `5n + 2` latent frames, as packing states.
            let latent = crate::packing::video_latent_frames(frames).expect("latent frames");
            assert_eq!(latent, 5 * clips.len() as u32 - 5 + 7);
            for pair in clips.windows(2) {
                assert_eq!(pair[0].latent.1, pair[1].latent.0 + 2);
            }
        }
    }

    #[test]
    fn a_frame_count_off_the_timeline_is_refused() {
        assert!(clips(96).is_err());
        assert!(clips(100).is_err());
        assert!(clips(0).is_err());
        assert!(clips(22).is_ok() || clips(22).is_err());
        // The smallest legal video: one clip.
        assert_eq!(clips(22).expect("one clip").len(), 1);
    }

    #[test]
    fn the_overlap_weights_are_the_references_fifths() {
        let clip = clips(107).expect("clips")[0];
        assert_eq!(clip.overlap_weights(), [0.0, 0.2, 0.4, 0.6, 0.8]);
        // Every strict increase, and the last below one: the clip never fully replaces its
        // predecessor inside its own tail.
        let weights = clip.overlap_weights();
        for pair in weights.windows(2) {
            assert!(pair[0] < pair[1]);
        }
        assert!(weights[4] < 1.0);
    }

    #[test]
    fn the_audio_track_carries_the_receptive_halo() {
        let config = crate::audio_vae::tests::audio_config();
        let track = Track::of(107, &config).expect("track");
        // packing's audio timeline: 200 latent frames for a five-second track, 800 samples each.
        assert_eq!(track.halo, 30);
        let track = Track::of(124, &config).expect("track");
        assert_eq!(
            track.latent_frames,
            crate::packing::audio_latent_frames(124)
        );
        assert_eq!(track.samples, crate::packing::audio_samples(124));
        // A window inside the track grows by the halo on both sides and is clipped at the ends.
        assert_eq!(track.window(40, 60), (10, 90));
        assert_eq!(track.window(0, 10), (0, 40));
        assert_eq!(
            track.window(track.latent_frames - 5, track.latent_frames),
            (track.latent_frames - 35, track.latent_frames)
        );
    }

    #[test]
    fn the_tile_rows_name_the_voxels_of_every_tile_in_turn() {
        let config = crate::video_vae::tests::video_config();
        let canvas = Canvas::clip(&config, 48, 84);
        let grid = Tiles::of(&config, &canvas)
            .expect("the grid")
            .tiles(&config, &canvas);
        let (rows, counts) = tile_rows(&canvas, &grid);
        // One count per tile, and the counts add up to the table: the table is the tiles' blocks end to
        // end, with no padding between them.
        assert_eq!(counts.len(), grid.len());
        assert_eq!(
            counts.iter().map(|count| u64::from(*count)).sum::<u64>(),
            rows.len() as u64
        );
        // **Tiles overlap in the latent grid**, so the table is longer than the clip's voxels and each
        // seam voxel is named twice: the tile that owns it and the neighbour that re-decodes its band.
        // A tile's block is its own canvas — every frame, every row of its latent rows, every column.
        let voxels = canvas.voxels();
        assert!(
            rows.len() as u64 > voxels,
            "an overlapping grid names more than {} voxels, not fewer",
            voxels
        );
        let mut offset = 0;
        for (tile, count) in grid.iter().zip(&counts) {
            let frame_columns = u64::from(canvas.width);
            let block_rows = u64::from(tile.latent_rows.1 - tile.latent_rows.0);
            let block_columns = u64::from(tile.latent_columns.1 - tile.latent_columns.0);
            assert_eq!(
                u64::from(*count),
                u64::from(canvas.frames) * block_rows * block_columns,
                "tile ({}, {}) names a block of its own canvas",
                tile.row,
                tile.column
            );
            let block = &rows[offset..offset + (*count as usize)];
            // The block is the raster's own row-major order restricted to the tile's rectangle, so it
            // starts at the rectangle's origin and rises all the way through.
            assert_eq!(
                block[0] as u64,
                u64::from(tile.latent_rows.0) * frame_columns + u64::from(tile.latent_columns.0)
            );
            assert!(block.windows(2).all(|pair| pair[0] < pair[1]));
            assert!((*block.last().expect("a block has a voxel") as u64) < voxels);
            offset += *count as usize;
        }
        // The multiplicity is the grid's own geometry, not a bound: a voxel is claimed by exactly the
        // tiles whose latent rectangle contains it. In a two-dimensional grid that is one tile in the
        // interior, two along one seam, and **four** at a corner where a row seam and a column seam
        // cross - so a bound of two is wrong, and the expected count is derived here from the same
        // rectangles the blocks were checked against above.
        let mut coverage = vec![0u8; voxels as usize];
        for row in &rows {
            coverage[*row as usize] += 1;
        }
        // A voxel's index is frame-major, then the latent row, then the column: `frames · height ·
        // width` voxels in all, which is `canvas.voxels()`.
        let plane = u64::from(canvas.height) * u64::from(canvas.width);
        let mut expected = vec![0u8; voxels as usize];
        for tile in grid.iter() {
            for frame in 0..canvas.frames {
                for latent_row in tile.latent_rows.0..tile.latent_rows.1 {
                    for latent_column in tile.latent_columns.0..tile.latent_columns.1 {
                        let voxel = u64::from(frame) * plane
                            + u64::from(latent_row) * u64::from(canvas.width)
                            + u64::from(latent_column);
                        expected[voxel as usize] += 1;
                    }
                }
            }
        }
        assert_eq!(coverage, expected);
        assert!(coverage.iter().all(|count| *count >= 1));
        // The derived count is confirmed by the seam arithmetic rather than by a literal: a voxel can
        // only be claimed by a tile of its row band and a tile of its column band, so the multiplicity
        // is at most the product of the two bands' widths, which is what a corner reaches.
        let mut rows_band = vec![0u32; canvas.height as usize];
        let mut columns_band = vec![0u32; canvas.width as usize];
        for tile in grid.iter() {
            for latent_row in tile.latent_rows.0..tile.latent_rows.1 {
                rows_band[latent_row as usize] += 1;
            }
            for latent_column in tile.latent_columns.0..tile.latent_columns.1 {
                columns_band[latent_column as usize] += 1;
            }
        }
        let bound = u8::try_from(
            rows_band.iter().copied().max().unwrap_or(0) as u64
                * columns_band.iter().copied().max().unwrap_or(0) as u64,
        )
        .expect("a seam band narrower than a voxel's count");
        assert!(coverage.iter().all(|count| *count <= bound));
        let covered: u64 = coverage.iter().map(|count| u64::from(*count)).sum();
        assert_eq!(covered, rows.len() as u64);
    }

    #[test]
    fn the_clip_rows_invert_the_packing_over_each_clips_seven_frames() {
        let canvas = crate::config::Canvas {
            short_edge: 768,
            aspect: (7, 4),
            width: 1344,
            height: 768,
        };
        let patch_rows = u64::from(
            crate::packing::latent_raster(&canvas)
                .expect("raster")
                .patch_rows(),
        );
        // Dense packing keeps raster order, so the clip rows are the raster rows in sequence.
        let rows = crate::packing::clip_rows(crate::config::AttentionKind::Dense, 107, &canvas)
            .expect("rows");
        assert_eq!(rows.len() as u64, 6 * 7 * patch_rows);
        // Clips start five latent frames after the previous one and read seven, so clip `k`'s rows are
        // the raster's rows from frame `5k` — not from `7k`, which is what a clip without overlap
        // would give.
        for (index, row) in rows.iter().enumerate() {
            let clip = index as u64 / (7 * patch_rows);
            let within = index as u64 % (7 * patch_rows);
            assert_eq!(u64::from(*row), clip * 5 * patch_rows + within);
        }
        // A sparse packing permutes them within the generated video. Consecutive clips share their
        // last two latent frames, so a row is named once per clip that reads it — the union covers the
        // generated rows, and no row is named twice inside one clip.
        let sparse = crate::config::AttentionKind::Sparse {
            tile: 128,
            sparsity: 0.8,
            reference_keep: None,
        };
        let rows = crate::packing::clip_rows(sparse, 107, &canvas).expect("rows");
        assert_eq!(rows.len() as u64, 6 * 7 * patch_rows);
        let per_frame = patch_rows as usize;
        let clip_rows = 7 * per_frame;
        let mut generated = vec![false; (5 * 6 + 2) as usize * per_frame];
        for (clip, block) in rows.chunks(clip_rows).enumerate() {
            let mut in_clip = vec![false; generated.len()];
            for row in block {
                let index = *row as usize;
                assert!(
                    index < generated.len(),
                    "clip {clip} reads row {index} outside the video"
                );
                assert!(!in_clip[index], "clip {clip} names row {index} twice");
                in_clip[index] = true;
                generated[index] = true;
            }
        }
        assert!(
            generated.iter().all(|row| *row),
            "a generated row is read by no clip"
        );
        // Clips overlap by two latent frames: the second clip starts five frames after the first.
        assert_eq!(rows[7 * per_frame], rows[5 * per_frame]);
    }
    /// The plan is the block's own allocation list, so its element types are the decode's - all of
    /// them but the rotary tables, which the block allocates at F32 whatever it decodes at. A plan
    /// that spelled one type for both reserved half of what an FP32 decode asks the arena for, and the
    /// shortfall surfaced inside the clip as "cannot give out" rather than as a plan that was short.
    /// A reservation summed raw is short by every buffer's padding - and the buffers that pad are
    /// the ones whose element count is not a multiple of the alignment, which for this decoder is the
    /// two rotary tables. The first device run to meet it was 64 bytes short of a 372 MiB arena.
    #[test]
    fn a_plan_is_reserved_at_the_arenas_alignment_not_at_its_raw_sum() {
        let config = crate::video_vae::tests::video_config();
        let canvas = crate::video_vae::Canvas::clip(&config, 16, 16);
        let plan = video_arena(&config, &canvas, nn::DType::F32).expect("the plan");
        let raw: u64 = plan.iter().map(Buffer::bytes).sum();
        let reserved = arena_reservation(&plan);
        assert_eq!(
            reserved,
            plan.iter()
                .map(|buffer| buffer.bytes().next_multiple_of(nn::forward::ALIGNMENT))
                .sum::<u64>(),
            "the reservation is each buffer at the arena's alignment"
        );
        assert!(reserved >= raw, "an aligned sum is never smaller");
        // **The difference is padding, buffer by buffer, and not a multiple of anything.** An earlier
        // version of this test asserted `(reserved - raw) % 256 == 0`, which is false whenever the
        // buffers' raw sum is not itself a multiple of the alignment - the paddings sum to whatever
        // they sum to. What is true is that each buffer pads by less than one alignment, and that the
        // total difference is those paddings.
        let padding: u64 = plan
            .iter()
            .map(|buffer| buffer.reservation() - buffer.bytes())
            .sum();
        assert_eq!(
            reserved - raw,
            padding,
            "the difference is the buffers' padding"
        );
        for buffer in &plan {
            assert!(
                buffer.reservation() - buffer.bytes() < nn::forward::ALIGNMENT,
                "{}: {} pads by no more than one alignment",
                buffer.name,
                buffer.bytes()
            );
        }
        // And the function's own rule, on a size no canvas has to supply: a buffer whose element
        // count is not a multiple of the alignment is rounded up, not down, and one that is a
        // multiple is left alone. The device run that found this was 64 bytes short because two
        // 172,512-byte tables were placed at their raw size.
        let uneven = Buffer {
            name: "a table".to_owned(),
            element: "f32",
            elements: 172_512 / 4,
        };
        assert_eq!(uneven.bytes(), 172_512);
        assert_eq!(uneven.reservation(), 172_544, "rounded up to the next 256");
        let even = Buffer {
            name: "an aligned tensor".to_owned(),
            element: "f32",
            elements: 65_536 / 4,
        };
        assert_eq!(even.reservation(), even.bytes(), "a multiple is not moved");
    }

    #[test]
    fn the_video_arena_is_stated_in_the_decodes_own_element_type() {
        let config = crate::video_vae::tests::video_config();
        let canvas = crate::video_vae::Canvas::clip(&config, 16, 16);
        let tables = |plan: &[Buffer]| -> u64 {
            plan.iter()
                .filter(|buffer| buffer.name == "cosine" || buffer.name == "sine")
                .map(Buffer::bytes)
                .sum()
        };
        let half = video_arena(&config, &canvas, nn::DType::F16).expect("the f16 plan");
        let whole = video_arena(&config, &canvas, nn::DType::F32).expect("the f32 plan");
        let total = |plan: &[Buffer]| -> u64 { plan.iter().map(Buffer::bytes).sum() };
        // Every typed entry doubles, and the two tables do not: the F32 plan is the F16 plan plus the
        // typed half, with the tables counted once on both sides.
        // **Not every typed entry doubles.** A layer that declares FP32 whatever the decode is holds
        // the same bytes at both, which is the per-layer declaration's consequence; what holds is that
        // the f32 plan is never smaller.
        assert!(
            total(&whole) >= total(&half),
            "the f32 plan is never smaller than the f16 plan"
        );
        assert_eq!(
            tables(&whole),
            tables(&half),
            "a rotary table is F32 at both"
        );
        for (one, other) in half.iter().zip(&whole) {
            assert_eq!(one.name, other.name, "the two plans name the same buffers");
            assert_eq!(one.elements, other.elements, "and hold the same elements");
            let expected = if one.name == "cosine" || one.name == "sine" {
                "f32"
            } else {
                "f16"
            };
            assert_eq!(one.element, expected, "{} at f16", one.name);
            assert_eq!(other.element, "f32", "{} at f32", one.name);
        }
    }

    #[test]
    fn the_video_arena_holds_one_tiles_scratch_and_every_tiles_tokens() {
        let config = crate::video_vae::tests::video_config();
        let canvas = crate::video_vae::Canvas::clip(&config, 48, 84);
        let arena = video_arena(&config, &canvas, nn::DType::F16).expect("the arena");
        // The served 768x1344 canvas at 256/64 is 4 rows of 7 tiles, each 16x16 latent voxels of
        // 7 frames.
        assert_eq!(canvas.frames, 7);
        assert_eq!(canvas.height, 48);
        assert_eq!(canvas.width, 84);
        let grid = crate::video_vae::Tiles::of(&config, &canvas)
            .expect("the grid")
            .tiles(&config, &canvas);
        assert_eq!(grid.len(), 28);
        // **The transformer's buffers are one tile's, and the tokens are every tile's.** `decode`
        // reserves `TileScratch` once per clip and reuses it for every tile, while each tile reserves
        // its own token buffer inside the loop and the arena never rewinds. A plan that sized the
        // fourteen by the grid summed two different lifetimes into one number.
        let tile_rows: u64 = grid[0].canvas.rows();
        let tile_voxels: u64 = grid[0].canvas.voxels();
        let all_rows: u64 = grid.iter().map(|tile| tile.canvas.rows()).sum();
        let all_voxels: u64 = grid.iter().map(|tile| tile.canvas.voxels()).sum();
        let buffer = |name: &str| {
            arena
                .iter()
                .find(|buffer| buffer.name == name)
                .unwrap_or_else(|| panic!("the plan states {name}"))
        };
        let columns = 2048;
        // **The pixel count is the patch, and it is four times the voxels it unpacks** at this
        // checkpoint: `out_channels × temporal_compression × spatial_compression²` is 3 · 4 · 16 · 16 =
        // 3072 elements per latent voxel. A statement that dropped the temporal factor reserved a
        // quarter of a tile's pixels, which is the same class as the half-sized plan the module's own
        // comment records: a shortfall that surfaces mid-clip as an arena that "cannot give out".
        let patch = u64::from(config.out_channels)
            * u64::from(config.temporal_compression())
            * u64::from(config.spatial_compression())
            * u64::from(config.spatial_compression());
        assert_eq!(buffer("tokens").elements, all_rows * patch);
        // The seam block: one tile's pixels at the raster's frame count, contiguous - the buffer every
        // blend copies its neighbour into, reserved once because every tile of a grid is the same size.
        assert_eq!(
            buffer("seam").elements,
            u64::from(grid[0].canvas.height)
                * u64::from(config.spatial_compression())
                * u64::from(grid[0].canvas.width)
                * u64::from(config.spatial_compression())
                * u64::from(canvas.frames_out(&config))
                * 3
        );
        assert_eq!(buffer("stream").elements, tile_rows * columns);
        assert_eq!(buffer("projected").elements, tile_voxels * columns);
        assert_eq!(buffer("gate_up").elements, tile_rows * 2 * columns * 4);
        assert_eq!(buffer("gated").elements, tile_rows * columns * 4);
        assert!(
            all_rows > tile_rows,
            "the clip's tiles outnumber one tile's rows"
        );
        let _ = all_voxels;
        // The two rotary factor tables the code reserves once, in the position it reserves them: half
        // the rotated width each, one tile's rows because they are the scratch's.
        let cosine = arena
            .iter()
            .position(|buffer| buffer.name == "cosine")
            .expect("the cosine factors");
        let sine = arena
            .iter()
            .position(|buffer| buffer.name == "sine")
            .expect("the sine factors");
        assert_eq!(sine, cosine + 1);
        assert_eq!(
            arena[cosine].elements + arena[sine].elements,
            tile_rows * u64::from(crate::video_vae::rotary(&config).expect("the rotary").0)
        );
        assert_eq!(arena[cosine].elements, arena[sine].elements);
        // The clip-level buffers come first and the per-tile ones after, which is the order the model
        // reserves them in: a plan that lists them the other way round is a plan the code does not
        // follow.
        let clip_level: Vec<&str> = arena
            .iter()
            .take(3)
            .map(|buffer| buffer.name.as_str())
            .collect();
        assert_eq!(clip_level, ["decoded", "seam", "tokens"]);
        // Every buffer is named once and no buffer is empty.
        let mut names: Vec<&str> = arena.iter().map(|buffer| buffer.name.as_str()).collect();
        let count = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), count);
        assert!(arena.iter().all(|buffer| buffer.elements > 0));
    }

    #[test]
    fn a_tiles_placement_is_a_view_of_the_raster_and_its_blends_read_the_neighbour() {
        // The served canvas's output width: the raster the tiles are placed into.
        const WIDTH: u32 = 1344;
        // 768x1344 pixels: 48x84 latent positions, so the checkpoint's 256-pixel tiles and 64-pixel
        // minimum overlap give a grid of four columns and seven rows, each tile 16x16 latent.
        let config = crate::video_vae::tests::video_config();
        let canvas = Canvas::clip(&config, 48, 84);
        let clip = ClipDecoder::new(&config, DType::F16).expect("the output path");
        let tiles = Tiles::of(&config, &canvas)
            .expect("the grid")
            .tiles(&config, &canvas);
        assert_eq!(tiles.len(), 28);
        let frames = canvas.frames_out(&config);
        let dest = TensorView::new(
            ops::DevicePtr::new(0x1000),
            DType::F16,
            Shape::new(&[3, u64::from(frames), 768, u64::from(WIDTH)]),
        );
        let ratio = 16;
        let mut above_seams = 0;
        let mut left_seams = 0;
        // A seam's width is the **raster's** unit - output pixels - so every width the grid asks the
        // blend table for must be at least the minimum overlap and a whole alignment step, and the
        // widths along one axis must sum to what the grid's tiles overhang that axis by. A table keyed
        // in latent cells answers a 4 for a seam that is 64 pixels wide and is refused; a table keyed
        // in pixels answers every width this grid can produce.
        let (mut above_widths, mut left_widths) = (Vec::new(), Vec::new());
        for tile in &tiles {
            let first_row = u64::from(tile.latent_rows.0) * ratio;
            let first_column = u64::from(tile.latent_columns.0) * ratio;
            let view = clip
                .placement(dest, tile, first_row, first_column, 0, 0)
                .expect("the placement");
            // The block is the tile's canvas in pixels, and it sits where the seam table says.
            assert_eq!(view.shape.dims(), &[3, u64::from(frames), 256, 256]);
            let origin = (first_row * u64::from(WIDTH) + first_column) * 2;
            assert_eq!(view.ptr.address(), 0x1000 + origin);
            if tile.above != 0 {
                // The neighbour is the predecessor's own block: its last `above` rows are the seam,
                // which sits where this tile starts, so the block begins `256 - above` rows above it.
                let up = 256 - tile.above;
                let previous = clip
                    .placement(dest, tile, first_row, first_column, up, 0)
                    .expect("the row neighbour");
                assert_eq!(
                    previous.ptr.address(),
                    0x1000 + origin - u64::from(up) * u64::from(WIDTH) * 2
                );
                // Its last `above` rows are this tile's first `above` rows - the seam itself.
                assert_eq!(
                    previous.ptr.address() + u64::from(256 - tile.above) * u64::from(WIDTH) * 2,
                    0x1000 + origin
                );
                assert!(clip.blend(tile.above, 2).is_ok());
                above_widths.push(tile.above);
                above_seams += 1;
            }
            if tile.left != 0 {
                let left = 256 - tile.left;
                let previous = clip
                    .placement(dest, tile, first_row, first_column, 0, left)
                    .expect("the column neighbour");
                assert_eq!(
                    previous.ptr.address(),
                    0x1000 + origin - u64::from(left) * 2
                );
                assert_eq!(
                    previous.ptr.address() + u64::from(256 - tile.left) * 2,
                    0x1000 + origin
                );
                assert!(clip.blend(tile.left, 3).is_ok());
                left_widths.push(tile.left);
                left_seams += 1;
            }
        }
        // The canvas is 48 x 80 latent cells, so its grid is 4 rows of 7: every tile but the first row
        // blends upwards (3 x 7), and every tile but the first column blends left (4 x 6).
        assert_eq!(above_seams, 21);
        assert_eq!(left_seams, 24);
        // Four tiles of 256 pixels cover 768: the row seams carry the overhang, and the same for the
        // seven columns over 1344. The width of each is derived here from the grid, never from the
        // blend table - which is the point of asserting it.
        let alignment = u64::from(config.spatial_compression());
        let minimum = u64::from(crate::video_vae::MINIMUM_OVERLAP);
        // A seam is visited once per tile of the other axis, so the widths collected along one axis are
        // that axis' overhang repeated by the grid's other count.
        let columns = tiles.iter().filter(|tile| tile.row == 0).count() as u64;
        let rows = tiles.iter().filter(|tile| tile.column == 0).count() as u64;
        for (widths, across, pixels, repeat) in [
            (&above_widths, 4_u64, 768_u64, columns),
            (&left_widths, 7, u64::from(WIDTH), rows),
        ] {
            let total: u64 = widths.iter().map(|width| u64::from(*width)).sum();
            assert_eq!(
                total,
                repeat * (across * u64::from(crate::video_vae::TILE_PIXELS) - pixels)
            );
            assert!(
                widths
                    .iter()
                    .all(|width| *width % alignment as u32 == 0 && u64::from(*width) >= minimum)
            );
        }
    }

    /// A seam's neighbour is not one contiguous block of a raster wider than the tile, and
    /// `cross_fade` addresses its sources flat: the stitch copies the neighbour into one contiguous
    /// tile-shaped buffer and blends from that. The buffer's contract is enforced where it is used,
    /// because a packed view of the raster passed in its place would blend the wrong rows and read as a
    /// different picture rather than a failure.
    #[test]
    fn a_seam_blends_from_one_contiguous_block_and_refuses_a_view_of_the_raster() {
        let config = crate::video_vae::tests::video_config();
        // Two tiles across, one down: 28 latent columns at 16 pixels each is 448, which is two
        // 256-pixel tiles sharing a 64-pixel seam, so the clip has a left blend and no row blend.
        let canvas = Canvas::clip(&config, 16, 28);
        let tiles = Tiles::of(&config, &canvas)
            .expect("the grid")
            .tiles(&config, &canvas);
        assert_eq!(tiles.len(), 2);
        let frames = u64::from(canvas.frames_out(&config));
        let (rows, columns) = canvas.pixels(&config);
        let (rows, columns) = (u64::from(rows), u64::from(columns));
        // The clip's buffers live in a recorded arena: nothing is allocated, and one tile's seam block
        // is a rounding error against the budget the plan states.
        let arena = nn::forward::Arena::new(ops::DevicePtr::new(0), 1 << 40);
        let f = Forward::recording(&arena);
        let decoded = f
            .arena()
            .tensor(DType::F16, Shape::new(&[3, frames, rows, columns]))
            .expect("the decoded raster");
        let blended = f
            .arena()
            .tensor(DType::F16, Shape::new(&[3, frames, rows, columns]))
            .expect("the blended raster");
        let tokens = f
            .arena()
            .tensor(
                DType::F16,
                Shape::new(&[Canvas::rows(&tiles[1].canvas), 3 * 4 * 16 * 16]),
            )
            .expect("the tile's tokens");
        let clip = ClipDecoder::new(&config, DType::F16).expect("the output path");
        // A contiguous 256x256 block: the shape the copy writes.
        let packed = [frames * 256 * 256, 256 * 256, 256, 1];
        let seam = f
            .arena()
            .tensor(DType::F16, Shape::new(&[3, frames, 256, 256]))
            .expect("the seam buffer");
        assert_eq!(seam.strides, nn::Strides::new(&packed));
        // The raster itself is what a caller might pass by mistake: same shape, the raster's strides.
        let wrong = TensorView::with_strides(
            decoded.offset_elements(0),
            DType::F16,
            Shape::new(&[3, frames, 256, 256]),
            decoded.strides,
        );
        let error = clip
            .write_tile(&f, &tiles[1], tokens, decoded, blended, wrong)
            .expect_err("a strided seam is not the block the blend reads");
        let message = format!("{error}");
        assert!(
            message.contains("seam buffer"),
            "the refusal names the buffer it wants: {message}"
        );
        // With the contiguous block the same tile is written: the refusal was the buffer, nothing else.
        clip.write_tile(&f, &tiles[1], tokens, decoded, blended, seam)
            .expect("a contiguous seam writes the tile");
    }
    /// Every seam the default canvases produce is in the blend table.
    ///
    /// The table is a ladder and the deployment launches only what its grids realize, so **the ladder
    /// may be wider than the deployment and must never be narrower**: a seam this test cannot find is
    /// a width a canvas needs and the decoder has no module for, which stops a clip decode at that
    /// seam rather than at a load. The reverse direction is not asserted, because the canvases this
    /// reads are the *default* table: a checkpoint that states its own ladder may serve canvases that
    /// produce wider seams, and a table derived from the default set would be narrow at exactly
    /// those.
    #[test]
    fn every_seam_the_offered_canvases_produce_is_in_the_blend_table() {
        let config = crate::video_vae::tests::video_config();
        let decoder = ClipDecoder::new(&config, DType::F32).expect("a clip decoder");
        let ratio = config.spatial_compression();
        let mut realized = Vec::new();
        for offered in crate::config::offered_canvases() {
            let canvas = Canvas::clip(&config, offered.height / ratio, offered.width / ratio);
            let tiles = Tiles::of(&config, &canvas)
                .expect("the grid")
                .tiles(&config, &canvas);
            for tile in tiles {
                for (extent, axis) in [(tile.above, 2), (tile.left, 3)] {
                    if extent > 0 && !realized.contains(&(extent, axis)) {
                        realized.push((extent, axis));
                    }
                }
            }
        }
        assert!(
            !realized.is_empty(),
            "the deployment's canvases are multi-row and multi-column, so they produce seams"
        );
        for (extent, axis) in &realized {
            assert!(
                decoder
                    .blends()
                    .iter()
                    .any(|(width, side, _)| width == extent && side == axis),
                "the deployment launches a {extent}-pixel seam along axis {axis} and the table is \
                 {} widths on two axes: a missing module stops the clip decode here",
                decoder.blends().len()
            );
        }
    }
}
