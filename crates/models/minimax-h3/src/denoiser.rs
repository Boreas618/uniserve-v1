//! The denoiser's layout: what one request reserves, and the tables that describe it.
//!
//! A request is snapped to a legal frame count and a legal canvas, packed into the row order the
//! transformer reads (`[video | audio | prefix | padding]`, [`crate::packing`]), and evaluated once
//! per schedule point. This module owns what surrounds the evaluation: the layout the worker
//! captures against, and the tables that make a captured graph independent of the prompt.
//!
//! ## Layouts, not requests
//!
//! A graph is captured per **layout** - the frame count, canvas, text capacity and condition
//! capacity - and replayed for every request that fits it. So the layout snaps: frames to `17n + 5`
//! ([`crate::packing::align_num_frames`]), sides to the checkpoint's 32-pixel grid, and both prefix
//! capacities up to whole capacity units (64 rows under dense attention, the sparse tile
//! otherwise). Two requests of one layout share one set of tables, and the worker keys its
//! storage by `(layout, name)`.
//!
//! ## The prefix source is one buffer
//!
//! The packed rows are assembled by gathering the prompt's prefix: every text row reads its own
//! row of the source, every condition row reads its row in packed order, and every generated or
//! padding row reads the source's **zero row**. That source is one buffer of `text_rows +
//! condition_rows + 1` rows - the prompt's rows, then one row per packed condition row, then the
//! zero row last - and [`Layouts::table_specs`]'s `prefix_index` addresses it directly
//! ([`crate::packing::DensePacking::zero_row`]). The runner builds it, because the encoders produce
//! the pieces and the packing decides the order; the DiT reads no other conditioning view, so a
//! condition the source does not carry cannot reach the model.
//!
//! Two model calls fill it, in either order, because they write disjoint rows and neither reads the
//! other's: the text encoder's own features and the zero row are the encoder stage's, and then
//! [`Denoiser::encode_conditions`] projects each condition's own rows into the condition rows, and
//! `Denoiser::prepare_conditioning` refines the prompt's rows in place. The condition projection
//! needs the request's declared condition shapes and therefore the stage that holds the request;
//! the refinement needs only the buffer, which is why it is the denoiser's own per-request hook.
//!
//! ## Host bytes
//!
//! Everything a request changes arrives as device tables the worker allocates: the rotary
//! coordinates, the modulation tags, the timestep groups, the prefix index of every packed row, the
//! AdaLN row each row reads, and the step index. `Layouts::prepare_state` writes their host bytes
//! in the order `Layouts::state_tables` names them; nothing of a request reaches a capture as a
//! host value, and no table is uploaded from inside a forward.

use nn::capability::{ConditionRole, LatentShape, Layout, StateTable, VideoRequest};
use nn::{DType, Forward, Shape, TensorView, Weight, WeightLoader};

use crate::attention::{Factors, Visible};
use crate::config::{Canvas, DenoiserConfig, MAX_FRAMES};
use crate::modulation::{AdaLnCache, Groups, Projections};
use crate::packing::{self, DensePacking};
use crate::partition::Partition;
use crate::schedule::{Modality, Schedule};
use crate::solver::Solver;
use crate::transformer::Transformer;
use crate::{Error, Result};

/// The modulation columns one AdaLN row holds: six affine vectors of three tags each.
const ADALN_COLUMNS: u64 = 18;

/// The least rung of a deployment's condition ladder: the tiles of one keyframe of the released
/// canvases, which is the smallest condition a request brings.
const LEAST_CONDITION_RUNG: u32 = 1024;

/// The row capacities and the sequence degree a deployment serves.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Arrangement {
    /// Text rows every layout of this deployment holds, a whole number of capacity units.
    pub text_rows: u32,
    /// Condition rows it holds, in capacity units.
    pub condition_rows: u32,
    /// Rows of one sequence shard: the Ulysses degree.
    pub shard: u32,
    /// This rank's place in the shard: which interval of the packed rows it owns.
    pub rank: u32,
}

impl Arrangement {
    /// A deployment's arrangement, checked against the attention kind's capacity unit.
    pub fn new(
        config: &DenoiserConfig,
        text_rows: u32,
        condition_rows: u32,
        shard: u32,
    ) -> Result<Self> {
        let unit = config.attention.capacity_rows();
        if shard == 0 || !shard.is_power_of_two() {
            return Err(Error::config(format!(
                "a sequence shard of {shard} rows is not a power of two"
            )));
        }
        for (name, rows) in [("text", text_rows), ("condition", condition_rows)] {
            if rows == 0 || !rows.is_multiple_of(unit) {
                return Err(Error::config(format!(
                    "a {name} capacity of {rows} rows is not a multiple of the {unit}-row unit"
                )));
            }
        }
        Ok(Self {
            text_rows,
            condition_rows,
            shard,
            rank: 0,
        })
    }

    /// The same arrangement, for the rank at `rank` of the shard.
    pub const fn at_rank(mut self, rank: u32) -> Self {
        self.rank = rank;
        self
    }
}

/// The deployment's layout rule: how a request snaps, and what its tables are.
/// The element type of the sample the trajectory accumulates, and of its noise.
///
/// The reference keeps its trajectory in FP32 and refuses any other type at its own entry
/// (`denoiser.py:1206-1216`, and `precision.py` pins the video and audio latents' input and output
/// there), so the draw, the patch and the unpatchify all read and write this one type - **not the
/// stack's**, which is the checkpoint's. Both sites read this constant: the forward refuses a latent
/// of another type by name ([`Denoiser::denoise_traced`]), and [`Denoiser::programs`] states the rows
/// the launch will ask for. Keying those rows from the stack's type instead was a real store gap -
/// a BF16 `unpatchify_video` row for a launch that asks for `dtype_f32`.
pub const SAMPLE_DTYPE: DType = DType::F32;

pub struct Layouts {
    config: DenoiserConfig,
    arrangement: Arrangement,
}

impl Layouts {
    /// The rule of a checkpoint and a deployment.
    pub fn new(config: DenoiserConfig, arrangement: Arrangement) -> Result<Self> {
        if config.transformer.head_dim == 0 || config.transformer.num_attention_heads == 0 {
            return Err(Error::config(
                "a denoiser's transformer has attention heads".to_owned(),
            ));
        }
        Ok(Self {
            config,
            arrangement,
        })
    }

    /// The canvas a request renders: its pixels rounded to the checkpoint's grid, and the latent
    /// raster they imply.
    ///
    /// The runner allocates rasters from this and never builds a [`Canvas`] itself - a canvas
    /// reconstructed from a layout is a second statement of a rounding, and the two would agree
    /// only on the sizes where the rounding is the identity.
    pub fn canvas(&self, request: &VideoRequest) -> Result<Canvas> {
        canvas_of(request)
    }

    /// The element offset and length of each condition's draw in the request's stream, in packed
    /// order: `(offset, elements)`, one entry per condition, so a runner slices one drawn buffer
    /// without computing a prefix sum.
    ///
    /// An audio reference draws nothing (`condition_elements`) and reads as a zero-length entry at
    /// the cursor its neighbours left - which is why this is a function and not a runner's prefix
    /// sum over their counts: a sum over counts is exactly where the audio-only case gets dropped.
    /// The entries' offsets are a partition of `[0, condition_elements(all))` in request order.
    pub fn condition_offsets(&self, request: &VideoRequest) -> Result<Vec<(u64, u64)>> {
        let canvas = canvas_of(request)?;
        let declared = conditions_of(request);
        let mut cursor = 0_u64;
        Ok(declared
            .iter()
            .map(|condition| {
                let elements =
                    u64::from(self.condition_elements(std::slice::from_ref(condition), &canvas));
                let entry = (cursor, elements);
                cursor += elements;
                entry
            })
            .collect())
    }

    /// The elements the request's condition draws occupy, in request order.
    ///
    /// A visual condition's noise is drawn at latent resolution - patched afterwards, so its packed
    /// rows overcount by four. **An audio reference draws none**: the reference conditions it at its
    /// VAE posterior's mean, and its stream position is not advanced
    /// (`uniserve/models/minimax_h3/denoiser.py:452-472`), so counting its rows too would shift
    /// every later draw of the request - the video's and the audio's - by a few thousand elements
    /// and change the video silently. The count is computed from the conditions themselves, not
    /// from a packing, so it is total: every condition whose raster and frame count are ones the
    /// packing would accept contributes exactly its elements, and one it would refuse contributes
    /// none. A refused condition never reaches here - the runner lays the request out, and that
    /// packing is the same function of the same values - so a served request's offsets are exact.
    pub fn condition_elements(&self, conditions: &[packing::Condition], _canvas: &Canvas) -> u32 {
        let channels = u64::from(self.config.transformer.video_channels);
        let elements: u64 = conditions
            .iter()
            .map(|condition| {
                condition.frame.map_or(0, |frame| {
                    // A condition is cut into clips, so its latent count is the encoder's rule
                    // ([`crate::encoding::Clip::latent_frames`]), not the generated timeline's:
                    // a 30-frame reference is 7 latent frames, and counting 0 - which the
                    // generated rule does, since 30 is not `17n + 5` - would shift every later
                    // draw of the request by the condition's whole volume.
                    let frames = self
                        .config
                        .clip
                        .latent_frames(frame.num_frames)
                        .unwrap_or(0);
                    match packing::latent_raster(&frame.canvas) {
                        Ok(raster) => {
                            u64::from(frames)
                                * u64::from(raster.height)
                                * u64::from(raster.width)
                                * channels
                        }
                        Err(_) => 0,
                    }
                })
            })
            .sum();
        elements as u32
    }

    /// The largest packed row count any layout of this deployment reaches.
    ///
    /// The attention allocates its scratch per packed row at load, so `rows` must bound every layout
    /// the deployment can be asked for. Only the packing the checkpoint's attention kind reads is
    /// such a bound: a dense deployment is sized by the dense packing, and taking the sparse one as
    /// well would over-size its attention by a quarter for requests it cannot receive. The two
    /// differ because the sparse packing pads to whole tiles (46848 against 38080 at the released
    /// canvases).
    pub fn rows_bound(&self) -> Result<u32> {
        let mut bound = 0;
        let sparse = matches!(
            self.config.attention,
            crate::config::AttentionKind::Sparse { .. }
        );
        for canvas in self.config.offered_canvases() {
            let mut frames = crate::CLIP_FRAMES + crate::CLIP_LATENTS;
            while frames <= MAX_FRAMES {
                if sparse {
                    let tile = self.config.attention.capacity_rows();
                    if let Ok(segment) = packing::segment_packing(
                        frames,
                        canvas,
                        self.arrangement.text_rows,
                        self.arrangement.condition_rows,
                        tile,
                        tile,
                    ) {
                        bound = bound.max(segment.extent.padded_tokens);
                    }
                } else if let Ok(dense) = packing::dense_packing(
                    frames,
                    canvas,
                    self.arrangement.text_rows,
                    self.arrangement.condition_rows,
                    self.arrangement.shard,
                ) {
                    bound = bound.max(dense.padded_tokens);
                }
                frames += crate::CLIP_FRAMES;
            }
        }
        if bound == 0 {
            return Err(Error::config(
                "a deployment offers no canvas a layout can be built for".to_owned(),
            ));
        }
        Ok(bound)
    }

    /// The largest number of video tiles any layout of this deployment reaches.
    ///
    /// The compression's score matrix is `[heads, query tiles, tiles]`, so its tile axis must cover
    /// every video tile of every layout the deployment can be asked for. It counts **all** video
    /// tiles and not the kept ones: a tile the sparsity drops keeps its column and says so through
    /// its size being zero, which is what masks it out of the softmax, so a bound of the kept tiles
    /// would under-size the matrix by `1 / (1 - sparsity)` - a fifth of it at `.8`.
    ///
    /// Only a sparse deployment has tiles. A dense one refuses by name rather than answering zero,
    /// because a zero bound would size nothing and the refusal would then arrive at a launch.
    pub fn tiles_bound(&self) -> Result<u32> {
        let crate::config::AttentionKind::Sparse { .. } = self.config.attention else {
            return Err(Error::config(
                "a dense deployment has no tiles to bound".to_owned(),
            ));
        };
        let tile = self.config.attention.capacity_rows();
        let mut bound = 0;
        for canvas in self.config.offered_canvases() {
            let mut frames = crate::CLIP_FRAMES + crate::CLIP_LATENTS;
            while frames <= MAX_FRAMES {
                if let Ok(segment) = packing::segment_packing(
                    frames,
                    canvas,
                    self.arrangement.text_rows,
                    self.arrangement.condition_rows,
                    tile,
                    tile,
                ) {
                    bound = bound.max(segment.extent.video_tiles);
                }
                frames += crate::CLIP_FRAMES;
            }
        }
        if bound == 0 {
            return Err(Error::config(
                "a deployment offers no canvas a layout can be built for".to_owned(),
            ));
        }
        Ok(bound)
    }

    pub fn segment_of(&self, layout: &Layout) -> Result<packing::SegmentPacking> {
        let tile = match self.config.attention {
            crate::config::AttentionKind::Sparse { tile, .. } => tile,
            crate::config::AttentionKind::Dense => {
                return Err(Error::config(
                    "a dense deployment has no segment packing".to_owned(),
                ));
            }
        };
        let frames = frame_count(layout.video.frames)?;
        let canvas = canvas_of_latents(layout)?;
        let packed = packing::segment_packing(
            frames,
            &canvas,
            layout.text_rows,
            layout.condition_rows,
            tile,
            tile,
        )?;
        let extent = &packed.extent;
        let raster = packing::latent_raster(&canvas)?;
        let video_rows = extent.video_grid.0 * extent.video_grid.1 * extent.video_grid.2;
        // The layout's capacities are rows of whole tokens; a segment packing rounds each prefix
        // capacity up to its tile, and its audio and video rows are the layout's own.
        let tiles = |rows: u32| rows.div_ceil(tile) * tile;
        if u64::from(video_rows) != u64::from(layout.video.frames) * raster.patch_rows() as u64
            || extent.audio_rows != packing::audio_rows(frames)
            || extent.text_rows != tiles(layout.text_rows)
            || extent.condition_rows != tiles(layout.condition_rows)
        {
            return Err(Error::config(format!(
                "the layout's {:?} extents are not a segment packing this deployment builds",
                layout.video
            )));
        }
        Ok(packed)
    }

    /// The layout a request is laid out in, without packing it.
    ///
    /// The capability asks for a layout and cannot refuse one, so this snaps the request the way
    /// [`Layouts::layout`] does and takes the deployment's text capacity and the request's rung. A
    /// request whose packing is then refused is one the runner never asks for: the packing is the
    /// same function of the same values, and the runner must lay the request out before it captures
    /// anything.
    pub fn cover(&self, request: &VideoRequest<'_>) -> Layout {
        let frames = frames_of(request);
        let canvas = canvas_of(request).unwrap_or(Canvas {
            short_edge: 768,
            aspect: (16, 9),
            width: 1344,
            height: 768,
        });
        let raster = packing::latent_raster(&canvas).unwrap_or(packing::LatentRaster {
            // A snapped canvas is a whole number of 16-pixel latent cells, so this is the raster the
            // packing states; the fallback only keeps the capability's infallible call free of a
            // panic when a caller passes a raster the checkpoint never served.
            height: canvas.height / 16,
            width: canvas.width / 16,
        });
        Layout {
            video: LatentShape {
                channels: self.config.transformer.video_channels,
                frames: packing::video_latent_frames(frames).unwrap_or(2),
                height: raster.height,
                width: raster.width,
            },
            audio: Some(self.audio_latent(packing::audio_latent_frames(frames))),
            text_rows: self.arrangement.text_rows,
            condition_rows: self.condition_rows_of(request),
            condition_elements: self.condition_elements(&conditions_of(request), &canvas),
        }
    }

    /// The condition rows `request`'s layout holds: the rung it states, the deployment's capacity
    /// when it states none, and none for a request without conditions.
    ///
    /// A deployment's capacity is sized for the widest request it serves, and a layout's rows are
    /// real rows of every projection, every attention and every feed-forward of every step. A
    /// `t2va` request at a capacity sized for `fl2va` carried 2048 empty rows through all of them,
    /// and an image reference at a capacity sized for a reference video carried 40000.
    fn condition_rows_of(&self, request: &VideoRequest) -> u32 {
        if request.task == nn::capability::VideoTask::TextToVideoAudio {
            0
        } else {
            request
                .condition_rows
                .unwrap_or(self.arrangement.condition_rows)
        }
    }

    /// The condition rows a layout of this deployment may hold, ascending and ending at its
    /// capacity: one captured graph per rung and timeline, and a request rides the smallest rung
    /// that holds its conditions.
    ///
    /// The rungs start at one keyframe and double, with a rung halfway between two doublings from
    /// four keyframes up, where a request's empty rows start to cost more than their graphs. Each
    /// is a whole number of capacity units, so a layout of any rung packs.
    pub fn condition_ladder(&self) -> Vec<u32> {
        let capacity = self.arrangement.condition_rows;
        let unit = self.config.attention.capacity_rows();
        let mut rungs = Vec::new();
        let mut doubling = LEAST_CONDITION_RUNG;
        while doubling < capacity {
            rungs.push(round_up(doubling, unit));
            if doubling >= 4 * LEAST_CONDITION_RUNG && doubling + doubling / 2 < capacity {
                rungs.push(round_up(doubling + doubling / 2, unit));
            }
            doubling *= 2;
        }
        rungs.push(capacity);
        rungs
    }

    /// The arrangement this rule serves.
    pub const fn arrangement(&self) -> Arrangement {
        self.arrangement
    }

    /// The layout a request is served in, with the packing of its row extents.
    ///
    /// The canvas comes from the request's raster and the frames are snapped; the text capacity is
    /// the deployment's, because a layout holds every prompt that fits it, and the condition rows
    /// are the rung the request states.
    pub fn layout(&self, request: &VideoRequest) -> Result<(Layout, DensePacking)> {
        let frames = packing::align_num_frames(request.frames);
        let canvas = canvas_of(request)?;
        let packing = packing::dense_packing(
            frames,
            &canvas,
            self.arrangement.text_rows,
            self.condition_rows_of(request),
            self.arrangement.shard,
        )?;
        let elements = self.condition_elements(&conditions_of(request), &canvas);
        let layout = self.layout_of(&packing, &canvas, elements)?;
        Ok((layout, packing))
    }

    /// The layout of a packing, the canvas it packs, and the condition elements it holds.
    pub fn layout_of(
        &self,
        packing: &DensePacking,
        canvas: &Canvas,
        condition_elements: u32,
    ) -> Result<Layout> {
        let raster = packing::latent_raster(canvas)?;
        // The packing's video rows are latent frames of raster patch rows, so the division recovers
        // the latent frame count directly: it is not a generated count and must not be snapped.
        let frames = packing.video_rows / raster.patch_rows();
        Ok(Layout {
            video: LatentShape {
                channels: self.config.transformer.video_channels,
                frames,
                height: raster.height,
                width: raster.width,
            },
            audio: Some(self.audio_latent(audio_steps(packing.audio_rows)?)),
            text_rows: packing.text_rows,
            condition_rows: packing.condition_rows,
            condition_elements,
        })
    }

    /// The audio latent a generated frame count holds.
    ///
    /// The layout's `channels * frames` is the latent's element count and the worker sizes the
    /// buffer and its noise draw from that product; the denoiser reads those elements as
    /// `AUDIO_CHANNELS * steps` rows of `audio_channels` features, channel-major. So the stereo pair
    /// of channel planes is the `channels`, and a `steps` count is the `frames` - stating
    /// `audio_channels` as the channels declares half the elements the branch reads, which a single
    /// rank cannot see and every rank after it pays for in the draw's offset.
    ///
    /// **Every constructor of a layout builds its audio stream here.** The half-buffer got in twice
    /// because two constructors stated the decomposition separately.
    fn audio_latent(&self, steps: u32) -> LatentShape {
        LatentShape {
            channels: crate::config::AUDIO_CHANNELS * self.config.transformer.audio_channels,
            frames: steps,
            height: 1,
            width: 1,
        }
    }

    /// The packing a layout's extents came from.
    ///
    /// A layout states the latent extents and the two capacities; the packing is
    /// [`packing::dense_packing`]'s for the frame count and canvas those extents imply, and the two
    /// must agree - a disagreement is a layout this model did not build.
    pub fn packing_of(&self, layout: &Layout) -> Result<DensePacking> {
        if matches!(
            self.config.attention,
            crate::config::AttentionKind::Sparse { .. }
        ) {
            return Err(Error::config(
                "a sparse deployment packs its layout in tiles, not densely".to_owned(),
            ));
        }
        let canvas = canvas_of_latents(layout)?;
        let frames = frame_count(layout.video.frames)?;
        let packing = packing::dense_packing(
            frames,
            &canvas,
            layout.text_rows,
            layout.condition_rows,
            self.arrangement.shard,
        )?;
        let rebuilt = self.layout_of(&packing, &canvas, layout.condition_elements)?;
        if rebuilt.video.frames != layout.video.frames
            || rebuilt.video.height != layout.video.height
            || rebuilt.video.width != layout.video.width
            || rebuilt.text_rows != layout.text_rows
            || rebuilt.condition_rows != layout.condition_rows
            || rebuilt.condition_elements != layout.condition_elements
        {
            return Err(Error::config(format!(
                "the layout's {:?} latent extents are not a frame count and canvas this model packs",
                layout.video
            )));
        }
        Ok(packing)
    }

    /// The tables one request of `layout` needs, in the order `Layouts::prepare_state` fills them.
    ///
    /// Sorted by name, so the order is a property of the layout alone and two requests of one shape
    /// share one set. Every table is one the forward reads; none is a host value the worker would
    /// otherwise have to pass through a launch.
    pub fn table_specs(&self, layout: &Layout) -> Result<Vec<StateTable>> {
        if matches!(
            self.config.attention,
            crate::config::AttentionKind::Sparse { .. }
        ) {
            return self.segment_table_specs(layout);
        }
        let packing = self.packing_of(layout)?;
        let rows = u64::from(packing.padded_tokens);
        let video = u64::from(packing.video_rows);
        let audio = u64::from(packing.audio_rows);
        let (video_rows, audio_rows) = dense_rows(&packing);
        let mut tables = owned_tables(
            self.owned_rows(rows, &video_rows).0.len() as u64,
            self.owned_rows(rows, &audio_rows).0.len() as u64,
        );
        tables.extend([
            // The row each modality's projected latents are scattered into: the layout places the
            // video rows first and the audio rows after them.
            StateTable {
                name: "audio_indices",
                dtype: DType::I32,
                shape: Shape::new(&[audio]),
            },
            StateTable {
                name: "video_indices",
                dtype: DType::I32,
                shape: Shape::new(&[video]),
            },
            // One sequence over the whole shard: every query row sees the used key rows and no
            // query sees a padding row. `cu_seq_lens_k` is the dense visibility, `_q` the rows, and
            // a single run is named by its two ends - a start and a stop - not by every row.
            StateTable {
                name: "cu_seq_lens_k",
                dtype: DType::I32,
                shape: Shape::new(&[2]),
            },
            StateTable {
                name: "cu_seq_lens_q",
                dtype: DType::I32,
                shape: Shape::new(&[2]),
            },
            StateTable {
                name: "groups",
                dtype: DType::I32,
                shape: Shape::new(&[rows]),
            },
            StateTable {
                name: "modulation_indices",
                dtype: DType::I32,
                shape: Shape::new(&[rows]),
            },
            // Axis-major, `[axes, rows]`: `nn::Rotary`'s sectioned rotation reads one coordinate
            // per axis per token and refuses a leading extent that is not the axis count. **The
            // coordinates are real numbers**: the reference's `position_ids` is FP64, and the clock
            // and the raster are rescaled (`ROPE_FRAME_RESCALE`, a column is `4/3`), so an integer
            // table would truncate every generated row's place.
            StateTable {
                name: "position_ids",
                dtype: DType::F32,
                shape: Shape::new(&[3, rows]),
            },
            StateTable {
                name: "prefix_index",
                dtype: DType::I32,
                shape: Shape::new(&[rows]),
            },
            StateTable {
                name: "token_tags",
                dtype: DType::I32,
                shape: Shape::new(&[rows]),
            },
            // The text refiner's own boundaries: its queries are every padded row and its keys stop
            // at the prompt's last row, so a padding row is never a key of any document.
            StateTable {
                name: "refiner_cu_seq_lens_k",
                dtype: DType::I32,
                shape: Shape::new(&[2]),
            },
            StateTable {
                name: "refiner_cu_seq_lens_q",
                dtype: DType::I32,
                shape: Shape::new(&[2]),
            },
            // The step the schedule evaluates: written by the worker before each replay, one
            // value per (request, capture) rather than per row.
            StateTable {
                name: "step",
                dtype: DType::I32,
                shape: Shape::new(&[1]),
            },
        ]);
        tables.sort_by_key(|table| table.name);
        Ok(tables)
    }

    /// The rows of one modality this rank owns: `(sources, slots)`.
    ///
    /// `packed` holds each of the modality's rows' place in the packed sequence, in the modality's
    /// own order. A rank owns one contiguous `rows / shard` interval of the packed sequence, so
    /// `sources[i]` is the modality row a rank projects and `slots[i]` the row of its own interval
    /// that row lands in. At one rank every row is owned and the two are the identity and `packed`.
    fn owned_rows(&self, rows: u64, packed: &[i32]) -> (Vec<i32>, Vec<i32>) {
        let shard = u64::from(self.arrangement.shard.max(1));
        let local = rows / shard;
        let first = u64::from(self.arrangement.rank) * local;
        let mut sources = Vec::new();
        let mut slots = Vec::new();
        for (index, row) in packed.iter().enumerate() {
            let row = *row as u64;
            if row >= first && row < first + local {
                sources.push(index as i32);
                slots.push((row - first) as i32);
            }
        }
        (sources, slots)
    }

    /// The tables a sparse layout needs, in the order [`Layouts::write_state`] fills them.
    ///
    /// A sparse layout holds four tables a dense one does not - the tile sizes, each tile's video
    /// segment, each segment's first tile, and the key tiles a query keeps - and neither of the dense
    /// attention's two sequence-length tables: the sparse attention reads its tiles, not one run.
    fn segment_table_specs(&self, layout: &Layout) -> Result<Vec<StateTable>> {
        let packed = self.segment_of(layout)?;
        let extent = &packed.extent;
        let rows = u64::from(extent.padded_tokens);
        let tiles = u64::from(extent.padded_tokens / extent.tile);
        let video = u64::from(extent.video_grid.0 * extent.video_grid.1 * extent.video_grid.2);
        let audio = u64::from(extent.audio_rows);
        let (video_rows, audio_rows) = segment_rows(&packed);
        let mut tables = owned_tables(
            self.owned_rows(rows, &video_rows).0.len() as u64,
            self.owned_rows(rows, &audio_rows).0.len() as u64,
        );
        tables.extend([
            StateTable {
                name: "audio_indices",
                dtype: DType::I32,
                shape: Shape::new(&[audio]),
            },
            StateTable {
                name: "video_indices",
                dtype: DType::I32,
                shape: Shape::new(&[video]),
            },
            StateTable {
                name: "groups",
                dtype: DType::I32,
                shape: Shape::new(&[rows]),
            },
            StateTable {
                name: "modulation_indices",
                dtype: DType::I32,
                shape: Shape::new(&[rows]),
            },
            StateTable {
                name: "position_ids",
                dtype: DType::F32,
                shape: Shape::new(&[3, rows]),
            },
            StateTable {
                name: "prefix_index",
                dtype: DType::I32,
                shape: Shape::new(&[rows]),
            },
            // The text refiner's own boundaries: its queries are every padded row and its keys stop
            // at the prompt's last row, so a padding row is never a key of any document.
            StateTable {
                name: "refiner_cu_seq_lens_k",
                dtype: DType::I32,
                shape: Shape::new(&[2]),
            },
            StateTable {
                name: "refiner_cu_seq_lens_q",
                dtype: DType::I32,
                shape: Shape::new(&[2]),
            },
            StateTable {
                // Over-allocated to the tile count and filled by segment index, exactly as the
                // reference builds it (`packing.py:1228-1239`); `select` reads only its first
                // `segments` entries, which is the prefix view the layer hands it.
                name: "segment_keep",
                dtype: DType::I32,
                shape: Shape::new(&[tiles]),
            },
            StateTable {
                name: "segment_starts",
                dtype: DType::I32,
                shape: Shape::new(&[tiles]),
            },
            StateTable {
                name: "step",
                dtype: DType::I32,
                shape: Shape::new(&[1]),
            },
            StateTable {
                name: "tile_segments",
                dtype: DType::I32,
                shape: Shape::new(&[tiles]),
            },
            StateTable {
                name: "token_tags",
                dtype: DType::I32,
                shape: Shape::new(&[rows]),
            },
            StateTable {
                name: "variable_block_sizes",
                dtype: DType::I32,
                shape: Shape::new(&[tiles]),
            },
        ]);
        tables.sort_by_key(|table| table.name);
        Ok(tables)
    }

    /// Fill a sparse layout's tables: the tile packing's own indices and the segment tables.
    fn write_segment_state(
        &self,
        request: &VideoRequest<'_>,
        layout: &Layout,
        out: &mut [Vec<u8>],
    ) -> Result<()> {
        let (sparsity, reference_keep) = match self.config.attention {
            crate::config::AttentionKind::Sparse {
                sparsity,
                reference_keep,
                ..
            } => (sparsity, reference_keep),
            crate::config::AttentionKind::Dense => {
                return Err(Error::config(
                    "a dense deployment holds no segment tables".to_owned(),
                ));
            }
        };
        let packing = self.segment_of(layout)?;
        let frames = frame_count(layout.video.frames)?;
        let canvas = canvas_of_latents(layout)?;
        if request.text_tokens < 1 || request.text_tokens > packing.extent.text_rows {
            return Err(Error::config(format!(
                "a prompt of {} rows does not fit a text capacity of {}",
                request.text_tokens, packing.extent.text_rows
            )));
        }
        let conditions = conditions_of(request);
        let segments = packing::condition_segments(&conditions, &canvas, self.config.clip)?;
        let tables = packing::segment_tables(
            &packing,
            frames,
            &canvas,
            request.text_tokens,
            &segments,
            &vision_spans(request),
            packing::Sparsity {
                sparsity,
                reference_keep,
            },
        )?;
        let rows = tables.groups.len();
        if rows != packing.extent.padded_tokens as usize {
            return Err(Error::config(format!(
                "{rows} table rows for {} packed rows",
                packing.extent.padded_tokens
            )));
        }
        let tiles = tables.variable_block_sizes.len();
        let specs = self.segment_table_specs(layout)?;
        let sizes: Vec<i32> = tables
            .variable_block_sizes
            .iter()
            .map(|size| *size as i32)
            .collect();
        let (video, audio) = segment_rows(&packing);
        let packed_rows = u64::from(packing.extent.padded_tokens);
        let (video_sources, video_slots) = self.owned_rows(packed_rows, &video);
        let (audio_sources, audio_slots) = self.owned_rows(packed_rows, &audio);
        // The refiner's document: one prompt, so its queries are every padded row and its keys stop
        // at the prompt's last row. The prompt's row count is the media clock's own start.
        let refiner_q = integers(&[0, layout.text_rows as i32]);
        let refiner_k = integers(&[0, request.text_tokens as i32]);
        let filled: [(&str, Vec<u8>); 16] = [
            ("audio_indices", integers(&audio)),
            ("audio_slots", integers(&audio_slots)),
            ("audio_sources", integers(&audio_sources)),
            ("video_indices", integers(&video)),
            ("video_slots", integers(&video_slots)),
            ("video_sources", integers(&video_sources)),
            ("groups", integers(&tables.groups)),
            ("refiner_cu_seq_lens_q", refiner_q.clone()),
            ("refiner_cu_seq_lens_k", refiner_k.clone()),
            (
                "modulation_indices",
                modulation_indices(&tables.groups, &tables.token_tags),
            ),
            ("position_ids", coordinates(&tables.position_ids)),
            ("prefix_index", integers(&tables.prefix_index)),
            ("segment_keep", integers(&tables.segment_keep)),
            ("segment_starts", integers(&tables.segment_starts)),
            ("tile_segments", integers(&tables.tile_segments)),
            ("token_tags", integers(&tables.token_tags)),
        ];
        for (name, bytes) in filled {
            let position = specs
                .iter()
                .position(|table| table.name == name)
                .ok_or_else(|| Error::config(format!("{name} is not a table of this layout")))?;
            let buffer = out
                .get_mut(position)
                .ok_or_else(|| Error::config(format!("the worker holds no buffer for {name}")))?;
            if buffer.len() != bytes.len() {
                return Err(Error::config(format!(
                    "{name} is {} bytes, its buffer {}",
                    bytes.len(),
                    buffer.len()
                )));
            }
            buffer.copy_from_slice(&bytes);
        }
        let position = specs
            .iter()
            .position(|table| table.name == "variable_block_sizes")
            .ok_or_else(|| Error::config("the block-size table is missing".to_owned()))?;
        let buffer = out
            .get_mut(position)
            .ok_or_else(|| Error::config("the worker holds no block-size buffer".to_owned()))?;
        let bytes = integers(&sizes);
        if buffer.len() != bytes.len() || tiles != sizes.len() {
            return Err(Error::config(format!(
                "variable_block_sizes is {} bytes for {tiles} tiles, its buffer {}",
                bytes.len(),
                buffer.len()
            )));
        }
        buffer.copy_from_slice(&bytes);
        Ok(())
    }

    /// Write one request's tables, in `Layouts::state_tables`' order.
    ///
    /// The bytes are host bytes in each table's own element type: the worker uploads them once
    /// per request, before the capture it will replay. `text_tokens` is the prompt's row count,
    /// which the media clock starts at; a prompt that does not fit the layout's capacity is
    /// refused here rather than captured.
    pub fn write_state(
        &self,
        request: &VideoRequest<'_>,
        layout: &Layout,
        out: &mut [Vec<u8>],
    ) -> Result<()> {
        if matches!(
            self.config.attention,
            crate::config::AttentionKind::Sparse { .. }
        ) {
            return self.write_segment_state(request, layout, out);
        }
        let packing = self.packing_of(layout)?;
        let text_tokens = request.text_tokens;
        if text_tokens > layout.text_rows {
            return Err(Error::config(format!(
                "a prompt of {text_tokens} rows does not fit a text capacity of {}",
                layout.text_rows
            )));
        }
        let frames = frame_count(layout.video.frames)?;
        let canvas = canvas_of_latents(layout)?;
        let conditions = conditions_of(request);
        let segments = packing::condition_segments(&conditions, &canvas, self.config.clip)?;
        let tables = packing::dense_tables(
            &packing,
            frames,
            &canvas,
            text_tokens,
            &segments,
            &vision_spans(request),
        )?;
        let rows = tables.groups.len();
        let (video_indices, audio_indices) = dense_rows(&packing);
        let packed_rows = u64::from(packing.padded_tokens);
        let (video_sources, video_slots) = self.owned_rows(packed_rows, &video_indices);
        let (audio_sources, audio_slots) = self.owned_rows(packed_rows, &audio_indices);
        // One sequence: every query row sees the used key rows and no query sees a padding row.
        let queries = [0, rows as i32];
        let keys = [0, tables.used as i32];
        let specs = self.table_specs(layout)?;
        // The text refiner's document: one prompt, so its queries are every padded row and its
        // keys stop at the prompt's last row - a padding row is never a key of any document.
        let refiner_k = integers(&[0, text_tokens as i32]);
        let refiner_q = integers(&[0, layout.text_rows as i32]);
        for (name, bytes) in [
            ("audio_indices", integers(&audio_indices)),
            ("audio_slots", integers(&audio_slots)),
            ("audio_sources", integers(&audio_sources)),
            ("video_indices", integers(&video_indices)),
            ("video_slots", integers(&video_slots)),
            ("video_sources", integers(&video_sources)),
            ("cu_seq_lens_k", integers(&keys)),
            ("cu_seq_lens_q", integers(&queries)),
            ("refiner_cu_seq_lens_k", refiner_k),
            ("refiner_cu_seq_lens_q", refiner_q),
            ("groups", integers(&tables.groups)),
            (
                "modulation_indices",
                modulation_indices(&tables.groups, &tables.token_tags),
            ),
            ("position_ids", coordinates(&tables.position_ids)),
            ("prefix_index", integers(&tables.prefix_index)),
            ("token_tags", integers(&tables.token_tags)),
        ] {
            let position = specs
                .iter()
                .position(|table| table.name == name)
                .ok_or_else(|| Error::config(format!("{name} is not a table of this layout")))?;
            let Some(buffer) = out.get_mut(position) else {
                return Err(Error::config(format!(
                    "the worker holds no buffer for {name}"
                )));
            };
            if buffer.len() != bytes.len() {
                return Err(Error::config(format!(
                    "{name} is {} bytes, its buffer {}",
                    bytes.len(),
                    buffer.len()
                )));
            }
            buffer.copy_from_slice(&bytes);
        }
        // The step is the worker's to write through `prepare_evaluation`: one i32 a replay
        // changes.
        let position = specs
            .iter()
            .position(|table| table.name == "step")
            .ok_or_else(|| Error::config("the step table is missing".to_owned()))?;
        if let Some(buffer) = out.get_mut(position)
            && !buffer.is_empty()
            && buffer.len() != 4
        {
            return Err(Error::config(format!(
                "the step table is {} bytes, not one i32",
                buffer.len()
            )));
        }
        Ok(())
    }
}

/// The four tables that say which modality rows a rank owns (`Layouts::owned_rows`).
fn owned_tables(video: u64, audio: u64) -> Vec<StateTable> {
    [
        ("audio_slots", audio),
        ("audio_sources", audio),
        ("video_slots", video),
        ("video_sources", video),
    ]
    .into_iter()
    .map(|(name, rows)| StateTable {
        name,
        dtype: DType::I32,
        shape: Shape::new(&[rows]),
    })
    .collect()
}

/// Each modality's rows' places in a dense packing: `(video, audio)`.
///
/// The layout places the video rows first and the audio rows after them, so each modality's scatter
/// indices are its own contiguous range.
fn dense_rows(packing: &DensePacking) -> (Vec<i32>, Vec<i32>) {
    let video = packing.video_rows as i32;
    let audio = packing.audio_rows as i32;
    ((0..video).collect(), (video..video + audio).collect())
}

/// Each modality's rows' places in a segment packing: `(video, audio)`.
///
/// The video table is read in raster order: the patch writes raster rows, the scatter puts row `r`
/// at `table[r]`, and the gather before the head reads them back. So it holds the packed row of each
/// raster row, which is what orders a sparse layout's video rows tile-major.
fn segment_rows(packing: &packing::SegmentPacking) -> (Vec<i32>, Vec<i32>) {
    let mut video = vec![0_i32; packing.video_indices.len()];
    for (packed, raster) in packing
        .video_indices
        .iter()
        .zip(&packing.video_raster_indices)
    {
        video[*raster as usize] = *packed as i32;
    }
    (
        video,
        packing.audio_indices().map(|row| row as i32).collect(),
    )
}

/// Where each modality's initial noise begins in the request's one Philox stream.
///
/// The reference draws one stream per request, in the order conditions, video, audio, and offsets
/// the draws by element rather than by tensor: a condition's noise is its video rows times the
/// video channels plus its audio rows times the audio channels, so the count cannot be read off a
/// row capacity alone. Every value here is a property of the layout, which is why a captured graph
/// can replay for another request of the same layout.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NoiseOffsets {
    /// Elements the conditions' noise occupies, which is where the video's begins.
    pub condition: u64,
    /// Where the video latent's noise begins.
    pub video: u64,
    /// Where the audio latent's noise begins.
    pub audio: u64,
}

impl Layouts {
    /// The noise stream's offsets for a layout and the conditions it holds.
    ///
    /// The conditions are the request's, in request order; a layout's `condition_rows` is their
    /// packed row capacity and cannot replace them, because rows and elements differ by modality
    /// (`video_channels` against `audio_channels`).
    pub fn noise_offsets(
        &self,
        layout: &Layout,
        conditions: &[packing::Condition],
    ) -> Result<NoiseOffsets> {
        let canvas = canvas_of_latents(layout)?;
        let segments = packing::condition_segments(conditions, &canvas, self.config.clip)?;
        let video_channels = u64::from(self.config.transformer.video_channels);
        // The noise is drawn at latent resolution and patchified afterwards, so a condition's
        // elements are its latent volume times the channels - not its packed rows, which is one row
        // per 2x2 patch. Audio is not patched: its rows are already its elements, channel-major.
        let mut condition = 0_u64;
        for segment in &segments {
            if let Some(visual) = segment.visual {
                condition += u64::from(visual.frames)
                    * u64::from(visual.raster.height)
                    * u64::from(visual.raster.width)
                    * video_channels;
            }
            // Audio rows are already channel-major elements: two stereo channels per latent step.
            condition += u64::from(segment.audio_rows());
        }
        let video = condition;
        let audio = video
            + u64::from(layout.video.frames)
                * u64::from(layout.video.height)
                * u64::from(layout.video.width)
                * video_channels;
        Ok(NoiseOffsets {
            condition,
            video,
            audio,
        })
    }
}

/// The prompt rows a request's vision blocks occupy, as the packing reads them: `[start, stop)`.
///
/// `VideoRequest` states a span by its first row and its rows, which is what a run of the prompt's
/// tags is; the packing marks every row of the half-open range.
fn vision_spans(request: &VideoRequest<'_>) -> Vec<(u32, u32)> {
    request
        .vision_spans
        .iter()
        .map(|(first, rows)| (*first, first + rows))
        .collect()
}

/// The frame count a request is snapped to.
pub fn frames_of(request: &VideoRequest) -> u32 {
    packing::align_num_frames(request.frames)
}

/// The steps a packing's audio rows hold: one row per channel plane per step.
///
/// The packing states the stream in rows, `AUDIO_CHANNELS * steps`, which is the one place the
/// decomposition is written down; a row count that does not divide is a packing this model did not
/// build, not a stream to round.
fn audio_steps(rows: u32) -> Result<u32> {
    rows.checked_div(crate::config::AUDIO_CHANNELS)
        .filter(|_| rows.is_multiple_of(crate::config::AUDIO_CHANNELS))
        .ok_or_else(|| {
            Error::config(format!(
                "the packing's {rows} audio rows do not divide into {} channel planes",
                crate::config::AUDIO_CHANNELS
            ))
        })
}

/// The stream offsets a request's video and audio draws start at.
///
/// A request's noise is **one stream addressed by element** (`denoiser.py:452-500`): the visual
/// conditions' draws first, then the video's rows, then the audio's. Every rank of a Ulysses group
/// holds the whole sample and draws the whole stream, so these are one pair for the group and every
/// rank's trajectory starts from the same numbers a single rank's does.
fn draw_offsets(layout: Layout) -> (u64, u64) {
    let video = u64::from(layout.condition_elements);
    // One packed row is a 2x2 patch of one latent frame, so it covers four voxels' channels.
    let elements = u64::from(layout.video.frames)
        * u64::from(layout.video.height / 2)
        * u64::from(layout.video.width / 2)
        * u64::from(layout.video.channels)
        * 4;
    (video, video + elements)
}

/// The canvas a request is laid out at, rounded up to the model's 32-pixel grid.
///
/// Public for a caller that must size a stage before loading it: the canvas is what a case's rows,
/// elements and tiles all follow from, and a second rounding is a second answer to it.
pub fn canvas_of(request: &VideoRequest) -> Result<Canvas> {
    let width = round_up(request.width, crate::CANVAS_MULTIPLE);
    let height = round_up(request.height, crate::CANVAS_MULTIPLE);
    let divisor = gcd(width, height);
    Ok(Canvas {
        short_edge: width.min(height),
        aspect: (width / divisor, height / divisor),
        width,
        height,
    })
}

/// The canvas a layout's latent raster came from.
fn canvas_of_latents(layout: &Layout) -> Result<Canvas> {
    let canvas = Canvas {
        short_edge: (layout.video.height * 16).min(layout.video.width * 16),
        aspect: (layout.video.width, layout.video.height),
        width: layout.video.width * 16,
        height: layout.video.height * 16,
    };
    packing::latent_raster(&canvas)?;
    Ok(canvas)
}

/// The generated frame count a latent frame count came from: `5n + 2` latent frames are `17n + 5`
/// generated frames ([`crate::packing::video_latent_frames`], inverted).
fn frame_count(latent_frames: u32) -> Result<u32> {
    if latent_frames < 2 || !(latent_frames - 2).is_multiple_of(crate::CLIP_LATENTS) {
        return Err(Error::config(format!(
            "{latent_frames} latent frames are not 5n + 2"
        )));
    }
    Ok((latent_frames - 2) / crate::CLIP_LATENTS * crate::CLIP_FRAMES + crate::CLIP_LATENTS)
}

/// The request's conditions in the geometry the packing places them by.
///
/// Public because a caller that must **size** a stage before loading it - a harness reserving its
/// condition capacity from the case it will run - needs the same mapping the packing is given; a
/// second construction of it is a second answer to "what rows does this case ask for".
pub fn conditions_of(request: &VideoRequest) -> Vec<packing::Condition> {
    // The request states each condition's role and the encoded raster it arrives at; the packing
    // wants that raster as a canvas, which is the same reduction `canvas_of` makes of the request's
    // own width and height. A condition without pixels is an audio reference and carries none.
    request
        .conditions
        .iter()
        .map(|shape| packing::Condition {
            role: match shape.role {
                ConditionRole::FirstFrame => packing::Role::FirstFrame,
                ConditionRole::LastFrame => packing::Role::LastFrame,
                ConditionRole::Reference => packing::Role::Reference,
            },
            frame: canvas_of_raster(shape.width, shape.height).map(|canvas| packing::Frame {
                canvas,
                num_frames: shape.frames,
            }),
            audio_samples: shape.audio_samples,
        })
        .collect()
}

/// The canvas a raster of `width` × `height` pixels is, or `None` without pixels.
///
/// The reduction is `canvas_of`'s: the short edge and the aspect, both exact. A raster no canvas
/// can hold is refused by the packing that reads it, not here.
fn canvas_of_raster(width: u32, height: u32) -> Option<Canvas> {
    if width == 0 || height == 0 {
        return None;
    }
    let divisor = gcd(width, height);
    Some(Canvas {
        short_edge: width.min(height),
        aspect: (width / divisor, height / divisor),
        width,
        height,
    })
}

/// The segments a request's conditions pack into, in packed order.
///
/// A runner hands one raster, one soundtrack and one draw per segment and reads back the row counts
/// the packing reserved - it does not re-derive them, because those rows are the same arithmetic the
/// packer's own ranges compute and a second statement of them is what makes a dropped condition
/// invisible. Each segment carries its kind, its request index, its anchor (a keyframe's), its
/// pixels' raster when it has any, and its audio frames.
///
/// The rows a condition occupies are `video_rows()` and `audio_rows()`, and a visual condition's
/// draw elements are [`Layouts::condition_offsets`]'s entry for it.
impl Denoiser {
    pub fn segments(&self, request: &VideoRequest) -> Result<Vec<packing::Segment>> {
        let canvas = canvas_of(request)?;
        packing::condition_segments(&conditions_of(request), &canvas, self.config.clip)
    }
}

/// The pieces of `request`'s conditions, in the order the packing holds them.
///
/// The draws are the reference's: one per visual condition, in packed order, each at the condition's
/// own latent shape, leading the request's one stream (`denoiser.py`, `condition_noise_shapes`). An
/// audio reference draws nothing. `hop` is the samples of one audio latent frame, which a track is
/// padded to; a deployment that holds no audio encoder states none, and a track is then refused.
fn pieces_of(
    config: &DenoiserConfig,
    hop: Option<u64>,
    request: &VideoRequest,
) -> Result<Vec<nn::capability::ConditionPiece>> {
    let canvas = canvas_of(request)?;
    let segments = packing::condition_segments(&conditions_of(request), &canvas, config.clip)?;
    let channels = config.transformer.video_channels;
    let mut cursor = 0_u64;
    let mut pieces = Vec::with_capacity(segments.len());
    for segment in segments {
        let shape = request
            .conditions
            .get(segment.condition as usize)
            .ok_or_else(|| {
                Error::config(format!(
                    "a segment of condition {} in a request of {}",
                    segment.condition,
                    request.conditions.len()
                ))
            })?;
        let draw = segment.visual.map(|visual| LatentShape {
            channels,
            frames: visual.frames,
            height: visual.raster.height,
            width: visual.raster.width,
        });
        let track = if shape.audio_samples > 0 {
            let hop = hop.ok_or_else(|| {
                Error::config(
                    "a request carries a soundtrack and this deployment holds no condition \
                     encoder"
                        .to_owned(),
                )
            })?;
            Some([
                u64::from(crate::config::AUDIO_CHANNELS),
                1,
                crate::encoding::padded_samples(u64::from(shape.audio_samples), hop)?,
            ])
        } else {
            None
        };
        pieces.push(nn::capability::ConditionPiece {
            condition: segment.condition,
            role: match shape.role {
                ConditionRole::FirstFrame => nn::capability::MediaRole::FirstFrame,
                ConditionRole::LastFrame => nn::capability::MediaRole::LastFrame,
                ConditionRole::Reference => nn::capability::MediaRole::Reference,
            },
            rows: segment.video_rows(),
            audio_rows: segment.audio_rows(),
            pixels: segment.visual.map(|_| {
                [
                    1,
                    u64::from(crate::encoding::PIXEL_CHANNELS),
                    u64::from(shape.frames),
                    u64::from(shape.height),
                    u64::from(shape.width),
                ]
            }),
            track,
            draw,
            draw_offset: cursor,
        });
        cursor += draw.map_or(0, |draw| {
            u64::from(draw.channels)
                * u64::from(draw.frames)
                * u64::from(draw.height)
                * u64::from(draw.width)
        });
    }
    Ok(pieces)
}

/// The rows a request's conditions encode to, and their noise draws.
///
/// Both are the runner's: the encoders produce the rows and the request's Philox stream produces the
/// draws, so the model reads views it allocated nothing for and writes only the prefix source it was
/// given. **Every view here is FP32**, rows and draws alike: the reference widens both and anchors
/// and projects in FP32, rounding once into the source (`denoiser.py:605-620`). A condition with pixels contributes one **normalized patch-row** view - `[rows, channels *
/// 4]`, what the video encoder emits, not `[rows, channels]` - and, if it carries a soundtrack, one
/// channel-major audio view after it (`[2 * frames, audio_channels]`); an audio-only reference
/// contributes its audio view alone. The reference's tuple has the same shape and the same order
/// (`uniserve/models/minimax_h3/denoiser.py:512-516, 553-561`).
pub struct ConditionLatents<'a> {
    /// Each condition's encoded rows, in request order and in the order above.
    pub latents: &'a [TensorView],
    /// One draw per visual condition, in packed order: `[channels, frames, height, width]`, **FP32**.
    ///
    /// The conditioned path works in FP32 from the draw to the projection, which is the reference's
    /// precision (`uniserve/models/minimax_h3/denoiser.py:605-620`: the rows and the draw are
    /// widened, the anchoring is FP32, and the projection's result is rounded once into the prefix
    /// source's dtype). The model refuses a draw, a patch-row view or an audio view of another
    /// dtype by name rather than rounding it silently.
    pub noise: &'a [TensorView],
}

/// The conditioned path's own diagnostic views, which a caller fills by offering them.
///
/// The design's numeric diagnosis compares one evaluation layer by layer, and on a conditioned
/// request that comparison cannot localise a fault: a wrong conditioning input makes **every** layer
/// output self-consistently wrong, so the stack agrees with itself and the video merely differs. The
/// tensors below are the ones that decide it, and they are dumped before the stack reads them.
///
/// Nothing is written unless a view is offered: a forward with no trace is the served path, launch
/// for launch. The design is explicit that this is a diagnosis and never a gate
/// (`specs/worker.md` 10) - a difference here is evidence about where two implementations part
/// company, not a verdict on either.
pub struct ConditionTrace<'a> {
    /// `[condition rows, hidden]`: the prefix source's condition rows as the projection left them,
    /// in packed order - audio before video within a segment (`notes/h3/conditioned-diagnosis.md`).
    pub source: Option<TensorView>,
    /// `[prompt tokens, hidden]`: the refiner's rows the pack gathers. The source's rows above the
    /// prompt are never read (`prefix_index` sends every padding row to the zero row), so they are
    /// not part of the trace and a report must not compare them.
    pub refined: Option<TensorView>,
    /// One `[rows, channels * 4]` view per visual condition, in packed order: the patched draw,
    /// before the anchoring mixes the condition's own rows in.
    pub draws: &'a [TensorView],
    /// One `[rows, channels * 4]` view per visual condition, in packed order: the condition's latents
    /// as they arrived, so a report can tell a wrong draw from a wrong condition.
    pub clean: &'a [TensorView],
}

/// The level a visual condition is anchored at: `t z + (1 - t) noise` with `t` this
/// (`uniserve/models/minimax_h3/denoiser.py:102`), and the level its modulation group sits at
/// until the generated video is cleaner than it ([`crate::modulation::entries`]).
pub const VISUAL_CONDITION_TIME: f32 = 0.999;

/// The level an audio reference sits at: its latent unchanged, so nothing is mixed in
/// (`uniserve/models/minimax_h3/denoiser.py:104`).
///
/// Its packed rows are the **channel-major** audio latent - `2 * frames` rows of `audio_channels`
/// elements, 74 rows for a 22-frame reference - which is the layout the encoder writes and the
/// audio input projection reads; a `[frames, channels]` reshape would give 37 rows and shift every
/// row after the reference.
pub const AUDIO_CONDITION_TIME: f32 = 1.0;

/// The seed the encoder samples every visual condition's posterior under
/// (`uniserve/models/minimax_h3/encoding.py:29-32`: `POSTERIOR_SEED = 42`).
///
/// It is not the request's: the reference draws this noise from a fresh generator of its own, so a
/// condition always encodes to the same latent, while the request's draw is the **anchor's** alone
/// (`denoiser.py:524`, `encode_conditions(..., noise=...)`). One buffer for both would put an
/// independent draw under every condition latent - `mean + std * eps` differs by
/// `std * (eps_request - eps_42)` - and the rows that disagreement lands in feed every evaluation.
pub const POSTERIOR_SEED: u64 = 42;

/// A request's condition rows, each condition's visual view and its audio view.
struct ConditionViews {
    pairs: Vec<(Option<TensorView>, Option<TensorView>)>,
}

impl ConditionViews {
    /// Pair the request's flat view list with its conditions: each condition contributes its
    /// visual patch rows when it has pixels and its audio rows when it has samples, in that order,
    /// which is the order the reference's `latents` tuple holds them
    /// (`uniserve/models/minimax_h3/denoiser.py:553-561`).
    fn new(conditions: &[packing::Condition], views: &[TensorView]) -> Result<Self> {
        let mut offered = views.iter();
        let mut pairs = Vec::with_capacity(conditions.len());
        let missing = || Error::config("a condition's latents were not offered".to_owned());
        for condition in conditions {
            let visual = if condition.frame.is_some() {
                Some(*offered.next().ok_or_else(missing)?)
            } else {
                None
            };
            let audio = if condition.audio_samples > 0 {
                Some(*offered.next().ok_or_else(missing)?)
            } else {
                None
            };
            pairs.push((visual, audio));
        }
        if offered.next().is_some() {
            return Err(Error::config(format!(
                "{} condition latents were offered for {} conditions",
                views.len(),
                conditions.len()
            )));
        }
        Ok(Self { pairs })
    }

    /// One condition's visual and audio latents, in request order.
    fn of(&self, condition: u32) -> (Option<TensorView>, Option<TensorView>) {
        self.pairs[condition as usize]
    }
}

/// Refuse a diagnostic view whose shape is not the tensor it is meant to hold.
fn check_trace_view(view: TensorView, rows: u64, columns: u64, name: &str) -> Result<()> {
    if view.shape.dims() != [rows, columns] {
        return Err(Error::config(format!(
            "the condition trace's {name} view is [{rows}, {columns}], got {:?}",
            view.shape.dims()
        )));
    }
    Ok(())
}

/// `rows` rows of `source` starting at `start`, addressed by the same layout.
///
/// The prefix source is one contiguous `[rows, columns]` buffer, so a condition's rows are a view
/// of it rather than a copy into it.
/// Where each of a request's condition rows sits in the prefix source.
///
/// `total` is the source's row count and `condition_rows` the **layout's** capacity for them, so the
/// text capacity is what the layout left: `total - condition_rows - 1`. Subtracting the request's own
/// piece rows instead would be a second convention for the same split, and it disagrees with the
/// packing wherever the request does not fill the layout's capacity - the reader (`prefix_index`)
/// places a condition row at `text_rows + (the rows before it)` from the layout's counts, so a writer
/// that placed them after a shorter text segment would write rows nothing gathers. That is a silent
/// drop of the whole conditioning: the page would generate unconditioned media and nothing would
/// refuse.
///
/// A request whose pieces do not fit the layout's capacity is refused here rather than written past
/// the rows the packing reserved for it.
fn condition_ranges(
    total: u64,
    condition_rows: u32,
    pieces: &[(u32, u32)],
) -> Result<(u64, Vec<(u64, u64)>)> {
    let planned = pieces
        .iter()
        .map(|(visual, audio)| u64::from(*visual) + u64::from(*audio))
        .sum::<u64>();
    let reserved = u64::from(condition_rows);
    if planned > reserved {
        return Err(Error::config(format!(
            "the request's conditions carry {planned} rows and the layout reserves {reserved}"
        )));
    }
    // The text capacity, the condition rows and the zero row are the whole source.
    let text_rows = total.checked_sub(reserved + 1).ok_or_else(|| {
        Error::config(format!(
            "a {total}-row prefix source cannot hold {reserved} condition rows and the zero row \
             the layout reserves"
        ))
    })?;
    let mut start = text_rows;
    let mut ranges = Vec::with_capacity(pieces.len());
    for (visual, audio) in pieces {
        ranges.push((start, start + u64::from(*audio) + u64::from(*visual)));
        start += u64::from(*visual) + u64::from(*audio);
    }
    Ok((text_rows, ranges))
}

fn row_range(source: TensorView, start: u32, rows: u64) -> Result<TensorView> {
    let columns = source.shape.dim(1);
    if u64::from(start) + rows > source.shape.dim(0) {
        return Err(Error::config(format!(
            "rows {start}..{} past the prefix source's {}",
            u64::from(start) + rows,
            source.shape.dim(0)
        )));
    }
    Ok(TensorView::with_strides(
        source.offset_elements(u64::from(start) * columns),
        source.dtype,
        Shape::new(&[rows, columns]),
        nn::Strides::new(&[columns, 1]),
    ))
}

/// Round `value` up to the next multiple of `unit`.
fn round_up(value: u32, unit: u32) -> u32 {
    value.div_ceil(unit) * unit
}

/// The greatest common divisor, for the aspect ratio a raster states.
fn gcd(left: u32, right: u32) -> u32 {
    if right == 0 {
        left
    } else {
        gcd(right, left % right)
    }
}

/// Coordinates as little-endian FP32, axis-major: every row's first axis, then every row's second.
///
/// The packing states one coordinate triple per row; `nn::Rotary`'s sectioned rotation indexes a
/// table by `[axis, token]` and a row-major table would rotate each row by another row's angles.
fn coordinates(points: &[[f64; 3]]) -> Vec<u8> {
    // Axis-major FP32, which is the table's declared type: a coordinate is a place on a rescaled
    // clock or raster, and the rotary reads one real number per axis per token.
    let mut bytes = Vec::with_capacity(points.len() * 3 * 4);
    for axis in 0..3 {
        for point in points {
            bytes.extend_from_slice(&(point[axis] as f32).to_le_bytes());
        }
    }
    bytes
}

/// Integers as little-endian int32.
fn integers(values: &[i32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// The AdaLN row of every packed row: `3 * group + tag`.
fn modulation_indices(groups: &[i32], tags: &[i32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(groups.len() * 4);
    for (group, tag) in groups.iter().zip(tags) {
        bytes.extend_from_slice(&(3 * group + tag).to_le_bytes());
    }
    bytes
}

/// One checkpoint's denoiser: the DiT, the schedule it is evaluated on, its solver and the AdaLN
/// cache the runner fills once after loading.
///
/// ## The AdaLN cache is model memory
///
/// The reference derives the modulation tables at load (`weights._prepare_adaln_cache`), so they are
/// weights rather than per-request execution state: 50 tables of `[entries, 18 * hidden]` plus
/// `[entries, 2 * hidden]`, about 1.9 GiB at FP32 for a 50-step checkpoint and 0.33 GiB for the DMD
/// ladder. The cache is allocated through the weight loader, at the schedule's own entry count, and
/// filled by one post-load launch.
///
/// That is a declared deviation from the capability's constant-table mechanism: [`Denoiser::constant_tables`]
/// is empty and [`Denoiser::prepare_constants`] writes the model's own region, so the worker
/// allocates nothing and no table is duplicated. It is one model's load-time fold, sized by that
/// model from its config alone and read only by it; a capability field carrying those views would
/// make every worker size a table it never reads. The launch stays capture-safe, writing a region
/// allocated once.
pub struct Denoiser {
    config: DenoiserConfig,
    rules: Layouts,
    schedule: Schedule,
    solver: Solver,
    transformer: Transformer,
    projections: Projections,
    cache: AdaLnCache,
    /// The layer tables `cache` gathers rows from, retained so the post-load launch can write them.
    adaln_layers: Vec<Weight>,
    /// The output table of the same cache.
    adaln_output: Weight,
    /// The schedule's entry times as their sinusoidal embedding, `[entries, frequency_dim]` FP32.
    frequencies: Weight,
    evaluations: u32,
    /// The text refiner: the prompt features the DiT conditions on, refined once per request.
    refiner: crate::conditioning::TokenRefiner,
    /// Whether this deployment places the condition encoders at all.
    ///
    /// The partition's, not the checkpoint's: a base release serves all three tasks, and a deployment
    /// that places only `t2va` holds no `latent_encoder` component and must not attach one.
    places_encoders: bool,
    /// The two condition encoders, when this deployment places them.
    ///
    /// A task with conditions (`fl2va`, `ref2va`) encodes the request's rasters and samples before its
    /// first evaluation; `t2va` has no conditions and constructs neither. They live inside this block
    /// rather than in one of their own because `nn` has no condition-encoder capability -
    /// `EncoderInputs` takes a token batch, and these take pixels and samples.
    encoders: Option<ConditionEncoders>,
    /// The packed row bound this deployment was loaded for.
    ///
    /// The stack's dense attention is compiled with it as its key bound (`nn`'s `DenseDescriptor`
    /// carries `max_kv_len`), so a request wider than this has no module to launch: the bound is
    /// consumed here, at load, and a layout or a request beyond it is refused by name rather than
    /// discovered at the first layer.
    rows: u32,
    /// This rank's position in the sequence shard and the shard's degree.
    rank: u32,
    shard: u32,
}

/// The two condition encoders a deployment with conditioned tasks holds.
///
/// `fl2va` and `ref2va` read a keyframe's or a reference's pixels and samples, encode them, and write
/// the resulting rows into the prefix source before the request's first evaluation. A `t2va`
/// deployment places no encoder and attaches none.
pub struct ConditionEncoders {
    /// A condition's pixels to its normalised latent ([`crate::encoding::VideoEncoder`]).
    pub video: crate::encoding::VideoEncoder,
    /// A condition's samples to its latent rows ([`crate::encoding::AudioEncoder`]).
    pub audio: crate::encoding::AudioEncoder,
}

impl ConditionEncoders {
    /// Load both encoders from the checkpoint their two components hold.
    ///
    /// A deployment's components are built in the checkpoint's order, and `transformer` comes before
    /// `latent_encoder`; a block is boxed as it is built, so the encoders have to be in hand before
    /// the denoiser block is constructed. The loader calls this in a pass of its own and hands the
    /// result to [`Denoiser::attach_encoders`].
    pub fn load(
        config: &crate::Config,
        checkpoint: &nn::Checkpoint,
        loader: &mut WeightLoader<'_>,
    ) -> Result<Self> {
        let video = crate::encoding::VideoEncoder::load(
            &config.video_vae,
            &mut loader.reborrow(checkpoint.component("vae")?)?,
        )?;
        // The audio VAE directory holds both halves, as the video one does.
        let audio = crate::encoding::AudioEncoder::load(
            &config.audio_vae,
            &mut loader.reborrow(checkpoint.component("audio_vae")?)?,
        )?;
        Ok(Self { video, audio })
    }
}

/// A model error as the capability's.
fn flattened(error: Error) -> nn::Error {
    nn::Error::Weight {
        tensor: "minimax-h3".to_owned(),
        problem: error.to_string(),
    }
}

/// The capability surface the worker binds: layouts, tables, and the step.
///
/// Every method delegates to the inherent one, so the model's own API stays the one the tests and
/// the runner's documentation read.
/// The denoising steps a denoiser of `config` runs for `task`, from the checkpoint's own schedule.
///
/// The task is a key: a checkpoint states the tasks it serves, and one it does not state is refused
/// by name rather than answered with another task's ladder. Every H3 task shares one grid today,
/// which is exactly why a wrong key would go unnoticed - the numbers would be right for a task the
/// deployment's own contract denies.
///
/// The video and audio ladders advance on one step counter ([`Schedule::num_steps`] refuses a grid
/// whose two disagree), so the video ladder is the request's ladder rather than one of two.
///
/// Public because it is arithmetic over a config and a schedule and nothing else: a host test cannot
/// build a [`Denoiser`], and this is the half of the trait's answer that can be wrong while every
/// number is right.
pub fn schedule_for(
    config: &DenoiserConfig,
    schedule: &Schedule,
    task: nn::capability::VideoTask,
) -> Result<Vec<nn::Step>> {
    if !config.tasks.contains(&task) {
        return Err(Error::config(format!(
            "this checkpoint serves {:?} and not {task:?}",
            config.tasks
        )));
    }
    schedule.steps(Modality::Video)
}

impl nn::capability::Denoiser for Denoiser {
    fn layout(&self, request: &VideoRequest<'_>) -> Layout {
        self.rules.cover(request)
    }

    /// The bytes one evaluation of `request` reserves, from this deployment's own plan.
    ///
    /// The layout is the trait's own and the plan is [`Denoiser::denoise_arena`], so a caller sizes
    /// storage from the number the forward fills rather than from a literal beside it. Nothing here
    /// allocates or touches a device: the model states what its forward needs and the runtime owner
    /// allocates to it.
    /// The stage's floor: **the larger of one evaluation and the prompt's fold**, because the two
    /// reuse one region rather than following one another in it.
    ///
    /// The captured graph's nodes carry the offsets the planning walk recorded, and that walk starts
    /// its arena at the region's base and runs the evaluation alone. The fold runs only on the live
    /// path (`first_step`, before the first replay) and writes its result into the conditioning
    /// column, so its scratch and the graph's buffers occupy the same offsets by construction: the
    /// graph clobbers the fold's scratch, which is dead, and never the fold's output, which is a
    /// column. Adding the two would refuse a deployment that serves - and the plan's entries are
    /// storage the worker allocates by name, so an added entry is memory rather than a bound.
    ///
    /// The larger, then, and not the sum: at every deployment here the evaluation is the larger (the
    /// fold is 532 MiB against the evaluation's 6.1 GiB at a 1024-row capacity), and the `max` is
    /// stated rather than assumed because a deployment whose evaluation were smaller would need it.
    fn arena_bytes(&self, request: &VideoRequest<'_>) -> nn::Result<u64> {
        let layout = self.layout(request);
        let plan = self.denoise_arena(&layout).map_err(flattened)?;
        let evaluation = crate::decoding::arena_reservation(&plan);
        let rows = layout.text_rows.saturating_add(layout.condition_rows);
        let fold = crate::conditioning::fold_bytes(
            &self.config.transformer,
            rows,
            self.transformer.dtype(),
        );
        Ok(evaluation.max(fold))
    }

    /// The denoising steps this checkpoint runs for `task`.
    ///
    /// Both modalities advance on one step counter - [`Schedule::num_steps`] refuses a grid whose two
    /// ladders disagree, and the solver refuses a step a modality does not have - so the video ladder
    /// **is** the request's ladder, not one of two.
    ///
    /// `task` is a key and not a position. Every H3 task shares this checkpoint's grid, which is
    /// exactly why a wrong key would go unnoticed: a runner that took the first schedule would run
    /// the right numbers for a task the checkpoint does not serve, and the deployment would serve a
    /// task its own contract denies.
    fn schedule(&self, task: nn::capability::VideoTask) -> nn::Result<Vec<nn::Step>> {
        schedule_for(&self.config, &self.schedule, task).map_err(flattened)
    }

    /// The stack's width: the refiner projects the encoder's `text_dim` columns into it.
    ///
    /// H3's text encoder writes 5120-column features (`transformer.text_dim`) and every packed row
    /// the denoiser reads is `hidden_size` wide - the `context_embedder` the refiner's input layer
    /// is, `conditioning.py:195`. So `Conditioning::text` and `Conditioning::source` are two widths
    /// and the runner sizes each from its owner: the encoder's features stay 5120 wide, and the rows
    /// the denoiser reads are 5376.
    fn text_width(&self) -> Option<u32> {
        Some(self.config.transformer.hidden_size)
    }

    fn draw_noise(
        &self,
        f: &Forward,
        seed: nn::capability::Seed,
        layout: Layout,
        out: nn::capability::Latents,
    ) -> nn::Result<()> {
        // One stream per request, addressed by element: the conditions' own draws first, then the
        // video latents, then the audio latents.
        let (video_offset, audio_offset) = draw_offsets(layout);
        // A layout with an audio stream and no audio latents would draw the video's noise and leave
        // the audio's undrawn, which is a silent disagreement rather than an absent stream.
        match (layout.audio, out.audio) {
            (Some(_), None) => {
                return Err(flattened(Error::config(
                    "this layout draws audio noise and no audio latents were offered".to_owned(),
                )));
            }
            (None, Some(_)) => {
                return Err(flattened(Error::config(
                    "audio latents were offered to a layout with no audio stream".to_owned(),
                )));
            }
            _ => {}
        }
        // The draw writes the trajectory, so it reads the sample's type: a latent of another type
        // would launch a noise row the plan does not state, and `denoise` would refuse it one call
        // later with the launch already made.
        if out.video.dtype != SAMPLE_DTYPE {
            return Err(flattened(Error::config(format!(
                "the video latents are {} and the draw writes {SAMPLE_DTYPE}",
                out.video.dtype
            ))));
        }
        if let Some(audio) = out.audio
            && audio.dtype != SAMPLE_DTYPE
        {
            return Err(flattened(Error::config(format!(
                "the audio latents are {} and the draw writes {SAMPLE_DTYPE}",
                audio.dtype
            ))));
        }
        let noise = nn::Noise::new();
        // The video's stream is drawn at the latent's **native** shape, `[channels, frames, height,
        // width]`, which is the buffer `denoise` reads and the solver folds a velocity into - its
        // own patch is `Patch::forward(f, video, patched, frames)`, so the sample a step updates is
        // the raster, not the rows. The reference draws the same shape (`noise_shape("video")`)
        // and patchifies it on the way into its state (`prepare_latents`, `denoiser.py:1056-1110`).
        // Drawing the packed rows here instead would be the same element count in another
        // arrangement: every value the seed named, in the patch's order rather than the latent's,
        // which no shape refuses and no trajectory shows.
        let video = layout.video;
        let native = [
            u64::from(video.channels),
            u64::from(video.frames),
            u64::from(video.height),
            u64::from(video.width),
        ];
        if out.video.shape.dims() != &native[..] {
            return Err(flattened(Error::config(format!(
                "the video latents are {:?} and this layout's draw is {native:?}: a sample is the \
                 native raster a step patchifies, not the rows it packs into",
                out.video.shape.dims()
            ))));
        }
        // The draw is a flat stream - `(seed, element)` - so the operator reads a two-axis view and
        // the raster keeps its own shape: `noise_view` merges the leading three axes into rows and
        // leaves the last as the width, one reinterpretation of the same bytes, and the element
        // order is the raster's flat order either way. Handing the four-axis view over is what the
        // operator refused (`expected 2 axes, found 4`), and it refused it after the load, one
        // launch after the first stage's.
        let flat = out.video;
        // Both draws carry the **same** `stage` and `draw`, and only the offset differs: the
        // operator's address is `counter = [pair_low, pair_high, stage, draw]`, so either of those
        // words selects a different stream rather than a later position in this one. The reference
        // draws a request's conditions, its video latent and its audio latent through one Philox
        // stream, each buffer consuming the elements the one before it ended at
        // (`runner/denoiser.py:fill_noise`, "the seed's stream draws each condition's noise, then
        // each modality's"), so the audio's noise is that stream continued - not an independent
        // stream drawn under the same seed. A `draw` of one per modality would give every audio
        // latent noise the reference never produces, which a comparison of `video` would pass and
        // which no shape or type refuses.
        noise.draw(f, flat, nn::NoiseKey::Scalar(seed.0), 0, 0, video_offset)?;
        if let Some(audio) = out.audio {
            noise.draw(f, audio, nn::NoiseKey::Scalar(seed.0), 0, 0, audio_offset)?;
        }
        Ok(())
    }

    fn condition_pieces(
        &self,
        request: &VideoRequest<'_>,
    ) -> nn::Result<Vec<nn::capability::ConditionPiece>> {
        let hop = self.encoders.as_ref().map(|encoders| encoders.audio.hop());
        pieces_of(&self.config, hop, request).map_err(flattened)
    }

    fn draw_condition_noise(
        &self,
        f: &Forward,
        seed: nn::capability::Seed,
        piece: &nn::capability::ConditionPiece,
        out: TensorView,
    ) -> nn::Result<()> {
        nn::Noise::new().draw(
            f,
            out,
            nn::NoiseKey::Scalar(seed.0),
            0,
            0,
            piece.draw_offset,
        )
    }

    fn prepare_latents<'a>(
        &self,
        f: &Forward,
        cond: nn::capability::Conditioning<'a>,
        out: nn::capability::Latents,
    ) -> nn::Result<()> {
        // The schedule's start: unit noise for the released and distilled grids, a scaled draw for
        // a parallel-decoding student.
        self.solver
            .start(f, Modality::Video, out.video)
            .map_err(flattened)?;
        if let Some(audio) = out.audio {
            self.solver
                .start(f, Modality::Audio, audio)
                .map_err(flattened)?;
        }
        let _ = cond;
        Ok(())
    }

    fn prepare_conditioning(
        &self,
        f: &Forward,
        cond: nn::capability::Conditioning<'_>,
    ) -> nn::Result<()> {
        self.prepare_conditioning_traced(f, cond, None)
            .map_err(flattened)
    }

    fn denoise<'a>(
        &self,
        f: &Forward,
        step: nn::Step,
        latents: nn::capability::Latents,
        cond: nn::capability::Conditioning<'a>,
    ) -> nn::Result<()> {
        Denoiser::denoise(self, f, &step, latents, cond).map_err(flattened)
    }

    fn evaluations(&self) -> u32 {
        self.evaluations
    }

    fn constant_tables(&self) -> Vec<StateTable> {
        Denoiser::constant_tables(self)
    }

    fn prepare_constants(&self, f: &Forward, out: &[TensorView]) -> nn::Result<()> {
        Denoiser::prepare_constants(self, f, out).map_err(flattened)
    }

    fn layout_tables(&self, _layout: &Layout) -> nn::Result<Vec<StateTable>> {
        // Every table this denoiser reads is a request's: the layout's are its packing's, which the
        // model derives rather than stores.
        Ok(Vec::new())
    }

    fn prepare_layout(&self, _layout: &Layout, out: &mut [Vec<u8>]) -> nn::Result<()> {
        if out.is_empty() {
            Ok(())
        } else {
            Err(flattened(Error::config(format!(
                "the H3 denoiser has no layout tables; {} buffers were offered",
                out.len()
            ))))
        }
    }

    fn state_tables(&self, layout: &Layout) -> Vec<StateTable> {
        self.rules.table_specs(layout).unwrap_or_default()
    }

    fn prepare_state(
        &self,
        request: &VideoRequest<'_>,
        layout: &Layout,
        out: &mut [Vec<u8>],
    ) -> nn::Result<()> {
        self.rules
            .write_state(request, layout, out)
            .map_err(flattened)
    }

    fn prepare_evaluation(&self, index: u32, out: &mut [Vec<u8>]) -> nn::Result<()> {
        if index >= self.evaluations {
            return Err(flattened(Error::config(format!(
                "evaluation {index} is outside the {} this schedule has",
                self.evaluations
            ))));
        }
        // An evaluation has no host bytes of its own: every coefficient a step reads is a constant
        // the loader wrote once (`constant_tables` is empty), and the step index is device data the
        // worker writes before each replay. A buffer offered here means the caller and
        // `constant_tables` disagree about what an evaluation owns.
        if !out.is_empty() {
            return Err(flattened(Error::config(format!(
                "{} coefficient buffers offered, and this deployment declares none",
                out.len()
            ))));
        }
        Ok(())
    }
}

/// The state tables of a dense layout, in `table_specs`' order (sorted by name).
const DENSE_TABLES: [&str; 16] = [
    "audio_indices",
    "audio_slots",
    "audio_sources",
    "cu_seq_lens_k",
    "cu_seq_lens_q",
    "groups",
    "modulation_indices",
    "position_ids",
    "prefix_index",
    "refiner_cu_seq_lens_k",
    "refiner_cu_seq_lens_q",
    "step",
    "token_tags",
    "video_indices",
    "video_slots",
    "video_sources",
];

/// The state tables of a segment layout, in `segment_table_specs`' order (sorted by name).
///
/// A segment layout has no `cu_seq_lens` pair and carries four tables a dense one does not, so the
/// two sets do not share positions: a name resolved against the wrong set silently reads another
/// tensor. [`Tables`] picks the set from the deployment's attention kind for exactly that reason.
const SEGMENT_TABLES: [&str; 18] = [
    "audio_indices",
    "audio_slots",
    "audio_sources",
    "groups",
    "modulation_indices",
    "position_ids",
    "prefix_index",
    "refiner_cu_seq_lens_k",
    "refiner_cu_seq_lens_q",
    "segment_keep",
    "segment_starts",
    "step",
    "tile_segments",
    "token_tags",
    "variable_block_sizes",
    "video_indices",
    "video_slots",
    "video_sources",
];

/// A rank-one row table as the rows of width one a row gather moves.
///
/// The per-row tables (`groups`, `prefix_index`, `modulation_indices`, `step`) are declared
/// one-dimensional, because a layer reads one integer per row. `nn::layers::gather_rows` states its
/// source and its output as `[rows, columns]`, so a table enters it as rows of width one: the same
/// buffer with row-major strides for the new shape, no copy and no change of value.
fn one_column(view: TensorView) -> TensorView {
    view.reshape(Shape::new(&[view.numel(), 1]))
}

/// One request's state views, resolved by name against its kind's table set.
struct Tables<'a> {
    names: &'static [&'static str],
    views: &'a [TensorView],
}

impl<'a> Tables<'a> {
    fn new(kind: crate::config::AttentionKind, views: &'a [TensorView]) -> Self {
        let names: &'static [&'static str] = match kind {
            crate::config::AttentionKind::Dense => &DENSE_TABLES,
            crate::config::AttentionKind::Sparse { .. } => &SEGMENT_TABLES,
        };
        Self { names, views }
    }

    fn get(&self, name: &str) -> Result<TensorView> {
        let position = self
            .names
            .iter()
            .position(|table| *table == name)
            .ok_or_else(|| {
                Error::config(format!(
                    "{name} is not a state table of this layout, which declares {}",
                    if self.names.is_empty() {
                        "none".to_owned()
                    } else {
                        self.names.join(", ")
                    }
                ))
            })?;
        self.views
            .get(position)
            .copied()
            .ok_or_else(|| Error::config(format!("the worker passed no {name} table")))
    }
}

/// This layout's packed row count, as the forward's own packing computes it.
///
/// A dense deployment's rows are the packing's padded token count and a sparse deployment's are the
/// segment packing's, which rounds to whole tiles. Both come from the same functions the loader builds
/// layouts from, so a plan and a forward cannot disagree about a row. The capacities are the
/// layout's own: a layout of a low rung is fewer rows than the deployment's bound.
pub fn packed_rows(
    config: &DenoiserConfig,
    arrangement: &Arrangement,
    layout: &Layout,
) -> Result<u32> {
    let frames = frame_count(layout.video.frames)?;
    let canvas = canvas_of_latents(layout)?;
    match config.attention {
        crate::config::AttentionKind::Sparse { tile, .. } => Ok(packing::segment_packing(
            frames,
            &canvas,
            layout.text_rows,
            layout.condition_rows,
            tile,
            tile,
        )?
        .extent
        .padded_tokens),
        crate::config::AttentionKind::Dense => Ok(packing::dense_packing(
            frames,
            &canvas,
            layout.text_rows,
            layout.condition_rows,
            arrangement.shard,
        )?
        .padded_tokens),
    }
}

/// Every buffer one evaluation reserves at `layout`, in the order the forward reserves them.
///
/// The worker sizes the denoiser's arena by the sum of this plan and keys its storage by the names,
/// as it does for the decoder's two plans and the splice's. The list is the forward's own reservation
/// order - `Denoiser::denoise_traced`, then the transformer's forward, then each modality's head in
/// the order the forward reaches them - because the arena is one cursor: a buffer listed before
/// another is reserved before it.
///
/// Nothing here is multiplied by the layer count. The stack reserves its scratch once and its
/// attention's once, which is what makes a 52-layer stack affordable; a per-layer reservation would
/// multiply every entry below by the number of layers. A parallel-decoding student's fused head is
/// the one repeated entry, gathered once per modality.
///
/// The rows are this layout's own packed count, so a plan is exact for a canvas rather than a bound,
/// and the worker takes the largest layout it will serve.
///
/// This is the plan itself, with every value the deployment supplies passed in, so a host tool that
/// has a configuration and no checkpoint can size a stage from it.
pub fn denoise_arena(
    config: &DenoiserConfig,
    arrangement: &Arrangement,
    groups: u32,
    dtype: DType,
    layout: &Layout,
) -> Result<Vec<crate::decoding::Buffer>> {
    let element = match dtype {
        DType::BF16 => "bf16",
        DType::F16 => "f16",
        DType::F32 => "f32",
        other => {
            return Err(Error::config(format!(
                "the denoiser's activations are {other}, which has no arena element"
            )));
        }
    };
    let transformer = &config.transformer;
    let hidden = u64::from(transformer.hidden_size);
    let ffn = u64::from(transformer.intermediate_size);
    let heads = u64::from(transformer.num_attention_heads);
    let head_dim = u64::from(transformer.head_dim);
    // One rank's share of the heads; the exchange's buffers are per rank, as the layers are.
    let local_heads = heads / u64::from(arrangement.shard);
    let groups = u64::from(groups);
    let total = u64::from(packed_rows(config, arrangement, layout)?);
    // The rows a rank owns: everything before the attention's exchange, and everything after it.
    let tokens = total / u64::from(arrangement.shard.max(1));
    let channels = u64::from(layout.video.channels);
    let frames = u64::from(layout.video.frames);
    let height = u64::from(layout.video.height);
    let columns = u64::from(layout.video.width);
    // The `1x2x2` patch: one row per latent frame and each pair of rows and columns.
    let video_rows = frames * (height / 2) * (columns / 2);
    // A rank's share of them, at most its whole interval.
    let own_video = video_rows.min(tokens);
    let patches = channels * 4;
    let rotated = u64::from(transformer.rope_frequency_dim) * 3;
    let audio = layout.audio.map(|audio| {
        let rows = u64::from(crate::config::AUDIO_CHANNELS) * u64::from(audio.frames);
        (rows, u64::from(transformer.audio_channels))
    });

    /// One line of the plan.
    fn push(
        plan: &mut Vec<crate::decoding::Buffer>,
        name: &str,
        element: &'static str,
        elements: u64,
    ) {
        plan.push(crate::decoding::Buffer {
            name: name.to_owned(),
            element,
            elements,
        });
    }

    // One modality's head: the norm every deployment runs, then the weight and bias a student
    // gathers out of its fused tables.
    fn head(
        plan: &mut Vec<crate::decoding::Buffer>,
        element: &'static str,
        hidden: u64,
        student: bool,
        name: &str,
        rows: u64,
        out: u64,
    ) {
        push(plan, &format!("{name}.normed"), element, rows * hidden);
        // The floor of the head is FP32 (`transformer.py:404-421`), so the normalised rows are
        // widened before the projection reads them. A stack whose own element type is already FP32
        // reads them in place and leaves this buffer untouched; the plan sizes for the widening.
        push(plan, &format!("{name}.wide"), "f32", rows * hidden);
        // The fused heads are FP32 tables, as the projection they replace is, so the row a step
        // gathers out of them is FP32 whatever the stack's type is.
        if student {
            push(plan, &format!("{name}.head"), "f32", out * hidden);
            push(plan, &format!("{name}.head_bias"), "f32", out);
        }
    }

    let mut plan = Vec::new();

    // The packed rows, the two projections that fill them, and the rotary tables of the one
    // sequence.
    push(&mut plan, "packed", element, tokens * hidden);
    // The sample's own type is FP32 - the draw writes it, the solver updates it, and the reference
    // refuses anything else - so the patch's input and output are FP32 and so is nothing else here
    // that the stack reads. The input projections are FP32 parameters at every preset
    // (`precision.py:106-117`, pinned at load) and read the widened FP32 rows, and the one rounding
    // into the packed rows is the narrowing that follows them.
    push(&mut plan, "patched", "f32", video_rows * patches);
    push(&mut plan, "video.owned", "f32", own_video * patches);
    push(&mut plan, "video.projected", "f32", own_video * hidden);
    push(
        &mut plan,
        "video.projected.stack",
        element,
        own_video * hidden,
    );
    if let Some((audio_rows, audio_features)) = audio {
        let audio_rows = audio_rows.min(tokens);
        push(&mut plan, "audio.owned", "f32", audio_rows * audio_features);
        push(&mut plan, "audio.projected", "f32", audio_rows * hidden);
        push(
            &mut plan,
            "audio.projected.stack",
            element,
            audio_rows * hidden,
        );
    }
    push(&mut plan, "cosine", "f32", total * rotated);
    push(&mut plan, "sine", "f32", total * rotated);

    // The stack: the step's gathered modulation, one alternate residual buffer, and the scratch
    // every layer shares.
    push(&mut plan, "step.entries", "i32", groups);
    push(
        &mut plan,
        "step.modulation",
        element,
        groups * 3 * 6 * hidden,
    );
    push(
        &mut plan,
        "step.following",
        element,
        groups * 3 * 6 * hidden,
    );
    push(&mut plan, "alternate", element, tokens * hidden);
    push(&mut plan, "normalized", element, tokens * hidden);
    push(&mut plan, "update", element, tokens * hidden);
    push(&mut plan, "mlp.packed", element, tokens * 2 * ffn);
    push(&mut plan, "mlp", element, tokens * ffn);
    let local = local_heads * head_dim;
    match config.attention {
        crate::config::AttentionKind::Dense => {
            for name in [
                "query",
                "key",
                "value",
                "rotated_query",
                "rotated_key",
                "attended",
            ] {
                push(
                    &mut plan,
                    &format!("attention.{name}"),
                    element,
                    total * local,
                );
            }
        }
        crate::config::AttentionKind::Sparse { .. } => {
            // The segment packing rounds the rows to whole tiles, so this divides exactly.
            let tiles = total / u64::from(config.attention.capacity_rows());
            for name in ["query", "key", "value", "gate", "fine"] {
                push(
                    &mut plan,
                    &format!("attention.{name}"),
                    element,
                    total * local,
                );
            }
            for name in ["pooled_query", "pooled_key", "pooled_value"] {
                push(
                    &mut plan,
                    &format!("attention.{name}"),
                    "f32",
                    tiles * local_heads * head_dim,
                );
            }
            push(
                &mut plan,
                "attention.scores",
                "f32",
                local_heads * tiles * tiles,
            );
            push(
                &mut plan,
                "attention.indices",
                "i32",
                local_heads * tiles * tiles,
            );
            push(&mut plan, "attention.counts", "i32", local_heads * tiles);
            push(
                &mut plan,
                "attention.compressed",
                element,
                tiles * local_heads * head_dim,
            );
        }
    }
    // A layer that does not communicate reserves nothing for it: the exchange's staging and the
    // merged rows are views of buffers a rank of one already holds. The gather mode adds the whole
    // sequence's normalized rows.
    if arrangement.shard > 1 {
        let inner = heads * head_dim;
        push(&mut plan, "attention.staging", element, tokens * inner);
        push(&mut plan, "attention.returned", element, tokens * inner);
        push(&mut plan, "attention.merged", element, tokens * inner);
        push(&mut plan, "attention.gathered", element, total * hidden);
    }

    // Each modality's head, in the order the forward reaches them: the norm, then a student's
    // gathered weight and bias.
    let student = u64::from(transformer.output_heads) > 1;
    head(
        &mut plan, element, hidden, student, "video", own_video, patches,
    );

    // The output's modulation and the solver's one staging word.
    push(&mut plan, "output.entries", "i32", groups);
    push(&mut plan, "output.modulation", element, groups * 2 * hidden);
    push(&mut plan, "coefficient", "f32", 1);

    // The video step: the rows gathered out of the packed sequence, their groups, the velocity
    // at the patch width, and the velocity the solver writes into the latents from.
    push(&mut plan, "video.hidden", element, own_video * hidden);
    push(&mut plan, "video.groups", "i32", own_video);
    // The prediction is FP32 from the head (`transformer.py:404-421`) and the solver reads one type
    // for the sample and the velocity, so neither of these is the stack's element type.
    push(&mut plan, "video.velocity_rows", "f32", own_video * patches);
    if arrangement.shard > 1 {
        // The rows every member predicted, collected: a rank's in its own interval, then the whole
        // sequence's, then the modality's in raster order.
        push(&mut plan, "video.velocity_owned", "f32", tokens * patches);
        push(&mut plan, "video.velocity_packed", "f32", total * patches);
        push(&mut plan, "video.velocity_all", "f32", video_rows * patches);
    }
    push(
        &mut plan,
        "video.velocity",
        "f32",
        channels * frames * height * columns,
    );
    if let Some((audio_rows, audio_features)) = audio {
        let own_audio = audio_rows.min(tokens);
        head(
            &mut plan,
            element,
            hidden,
            student,
            "audio",
            own_audio,
            audio_features,
        );
        push(&mut plan, "audio.hidden", element, own_audio * hidden);
        push(&mut plan, "audio.groups", "i32", own_audio);
        push(
            &mut plan,
            "audio.velocity",
            "f32",
            own_audio * audio_features,
        );
        if arrangement.shard > 1 {
            push(
                &mut plan,
                "audio.velocity_owned",
                "f32",
                tokens * audio_features,
            );
            push(
                &mut plan,
                "audio.velocity_packed",
                "f32",
                total * audio_features,
            );
            push(
                &mut plan,
                "audio.velocity_all",
                "f32",
                audio_rows * audio_features,
            );
        }
    }
    Ok(plan)
}

impl Denoiser {
    /// The programs one denoiser launches, from its configuration, its partition and the deployment's
    /// largest canvas.
    ///
    /// The two stacks are their own functions ([`Transformer::programs`],
    /// [`crate::conditioning::TokenRefiner::programs`]); this adds the orchestration around them — the
    /// pack and its inverse, the draw, the strided copies, the coefficient gather, and the blends the
    /// solver and the conditioner issue. `elements` is the element count of the largest trajectory a
    /// modality steps: the solver's residual is one operator over the whole sample, so its width is a
    /// deployment statement and not a constant.
    #[allow(clippy::too_many_arguments)]
    pub fn programs(
        config: &crate::config::TransformerConfig,
        attention: crate::config::AttentionKind,
        dtype: nn::DType,
        boundary: nn::DType,
        rows: u32,
        groups: Groups,
        partition: &Partition,
        elements: &[u64],
        tiles: u32,
    ) -> Result<Vec<(String, String)>> {
        use crate::programs::Programs;
        use nn::layers::gather_rows_descriptor;
        use ops::noise::NoiseKind;
        use ops::noise::PhiloxNoise as NoiseOp;
        use ops::patch::Patch as PatchOp;
        use ops::residual::{ScaledAdd as ScaledAddOp, ScaledResidual as ResidualOp};
        use ops::unpatchify::Unpatchify;

        let mut programs = Programs::default();
        programs.extend(Transformer::programs(
            config, attention, dtype, boundary, rows, groups, partition, tiles,
        )?);
        // The refiner's stream is the boundary's type, not the stack's: it is handed the prompt's
        // features where they are widened into the network, and every row it launches is keyed from
        // the tensors it reads. Keyed from the stack it states four rows the forward never makes -
        // its per-head norms, its two modulation gathers and its attention - and lacks the four it
        // does.
        programs.extend(crate::conditioning::TokenRefiner::programs(
            config, boundary,
        )?);

        let video = config.video_channels;

        // The pack and its inverse, the draw, and the strided copy the conditioner reads through.
        // The first three read or write the **sample**, not the stack: the trajectory is FP32
        // (`SAMPLE_DTYPE`, and the forward refuses any other), so a row keyed from `dtype` would name
        // a program the launch never asks for.
        programs.launch::<PatchOp>(&nn::layers::Patch::descriptor_of(SAMPLE_DTYPE));
        programs.launch::<Unpatchify>(
            &nn::layers::UnpatchifyVideo::new(SAMPLE_DTYPE, u64::from(video), 1, 2)?.descriptor(),
        );
        programs.launch::<NoiseOp>(&nn::layers::Noise::descriptor_of(
            SAMPLE_DTYPE,
            NoiseKind::Normal,
            ops::noise::KeySource::Scalar,
        ));
        programs.launch::<ops::copy_block::CopyBlock>(&nn::layers::CopyBlock::descriptor_of(
            nn::DType::F32,
        ));

        // The three gathers outside the modulation: the prefix rows of the packed stream, the solver's
        // step coefficient, and the trace's row copy the diagnosis takes. The wide one gathers the
        // stack's packed rows; the one-column one gathers the **solver's coefficient**, which is an
        // FP32 constant (`Solver::load` publishes the two tables as F32), so its row is the sample's
        // type and not the stack's.
        programs.launch::<ops::logits::LogitsGather>(&gather_rows_descriptor(
            dtype,
            u64::from(config.hidden_size),
        )?);
        programs.launch::<ops::logits::LogitsGather>(&gather_rows_descriptor(SAMPLE_DTYPE, 1)?);
        // The rows of each modality's sample this rank owns, gathered at the sample's own widths;
        // under Ulysses the predicted rows are placed in the rank's interval and gathered back the
        // same way once every member's have arrived.
        let widths = [4 * u64::from(video), u64::from(config.audio_channels)];
        for width in widths {
            programs
                .launch::<ops::logits::LogitsGather>(&gather_rows_descriptor(SAMPLE_DTYPE, width)?);
            if partition.ulysses_degree > 1 {
                programs.launch::<ops::rows::Rows>(&nn::layers::Rows::descriptor_of(
                    SAMPLE_DTYPE,
                    ops::rows::Mode::Replace,
                    width as u32,
                ));
            }
        }

        // The conditioner's anchor mix over a patch's own columns, and the refine-in-place copy over a
        // text row's.
        // One descriptor: the anchors' widths are the operands', and neither reaches the key.
        for _columns in [4 * u64::from(video), u64::from(config.hidden_size)] {
            programs.launch::<ScaledAddOp>(
                &nn::layers::ScaledAdd.descriptor(nn::DType::F32, nn::DType::F32),
            );
        }

        // The solver's step: one gated residual over the whole trajectory of each modality. The
        // trajectory is the sample (`Solver::step` reads the sample and the velocity in one type,
        // and the forward refuses a sample that is not `SAMPLE_DTYPE`), so these rows are the
        // sample's type - keyed from the stack they would name a row no step ever launches.
        programs.launch::<ResidualOp>(&nn::layers::ScaledResidual::descriptor_of(
            SAMPLE_DTYPE,
            u64::from(video) * 4,
            false,
        )?);
        for count in elements {
            programs.launch::<ResidualOp>(&nn::layers::ScaledResidual::descriptor_of(
                SAMPLE_DTYPE,
                *count,
                false,
            )?);
        }
        Ok(programs.finish())
    }

    /// Attach the condition encoders this deployment places.
    ///
    /// `load.rs` builds them from their own checkpoint directories (`vae` and `audio_vae`) and calls
    /// this before the block is bound - `Block::Denoiser` is a trait object, so this is the last
    /// moment at which the concrete block is reachable. A deployment whose tasks carry no conditions
    /// places no encoder, and attaching one is refused here rather than left to sit unread: the
    /// checkpoint's weights and the block's behaviour would then disagree.
    pub fn attach_encoders(&mut self, encoders: ConditionEncoders) -> Result<()> {
        if self.encoders.is_some() {
            return Err(Error::config(
                "this H3 denoiser already holds its condition encoders".to_owned(),
            ));
        }
        if !self.places_encoders {
            return Err(Error::config(format!(
                "this deployment places {:?}, which carries no conditions, so it holds no condition \
                 encoder",
                self.rules.config.tasks
            )));
        }
        self.encoders = Some(encoders);
        Ok(())
    }

    /// The condition encoders, when this deployment holds them.
    pub fn encoders(&self) -> Option<&ConditionEncoders> {
        self.encoders.as_ref()
    }

    /// Whether this deployment places the condition encoders.
    ///
    /// The runner asks before it offers a conditioned request: a task that carries conditions on a
    /// deployment that places none is a configuration the runner refuses, and it is better refused
    /// there than discovered as a video conditioned on nothing.
    pub fn places_encoders(&self) -> bool {
        self.places_encoders
    }

    /// Every buffer one evaluation at `layout` reserves, in the forward's own reservation order.
    ///
    /// The worker sizes the denoiser's arena by the sum of this plan and keys its storage by the
    /// names, as it does for the decoder's two plans and the splice's. It is [`denoise_arena`] with
    /// this deployment's own numbers - the checkpoint's element type and the schedule's group count -
    /// so the plan and the forward read one configuration.
    pub fn denoise_arena(&self, layout: &Layout) -> Result<Vec<crate::decoding::Buffer>> {
        let rows = self.bound(layout)?;
        let _ = rows;
        denoise_arena(
            &self.config,
            &self.rules.arrangement,
            self.cache.groups().count(),
            self.transformer.dtype(),
            layout,
        )
    }

    /// Refuse a layout whose packed rows exceed the bound this deployment was loaded for.
    ///
    /// The stack's dense attention is compiled with that bound as `nn`'s `DenseDescriptor.max_kv_len`
    /// and refuses a longer key run at its own launch, so a wider layout has no module at all: the
    /// refusal belongs where a layout first meets the deployment, not at the first layer of a
    /// capture that then has to be thrown away.
    fn bound(&self, layout: &Layout) -> Result<u32> {
        let tokens = packed_rows(&self.config, &self.rules.arrangement, layout)?;
        if tokens > self.rows {
            return Err(Error::config(format!(
                "this denoiser was loaded for {} packed rows and {} is a layout beyond it; the \
                 attention is compiled for the bound, so no module would launch",
                self.rows, tokens
            )));
        }
        Ok(tokens)
    }

    /// Encode the request's raw media and write its condition rows into the prefix source.
    ///
    /// The rows are the layout's - `MediaInput::rows` and `audio_rows` carry the counts the packing
    /// stated - so this only decides where each block starts ([`condition_ranges`]) and checks
    /// the encoders against it. Within a piece the audio rows come first, which is the reference's
    /// order (`uniserve/models/minimax_h3/denoiser.py:585-620`).
    ///
    /// The visual half reads the encoder's 48-channel moments, not the rows a sampler consumes: a
    /// condition's rows are the 24 normalised channels the posterior produces (`encoder-audit.md`
    /// H1), so this projects the stitched raster once, takes the posterior's sample, patches it to
    /// rows and anchors it against a draw at [`VISUAL_CONDITION_TIME`]. Conditioning on the moments
    /// instead would be a different image, quietly.
    fn encode_media(
        &self,
        f: &Forward,
        media: &[nn::capability::MediaInput],
        text: TensorView,
        trace: Option<&ConditionTrace<'_>>,
    ) -> Result<()> {
        let encoders = self.encoders.as_ref().ok_or_else(|| {
            Error::config(
                "a request carries conditions and this deployment holds no condition encoder"
                    .to_owned(),
            )
        })?;
        let pieces: Vec<(u32, u32)> = media
            .iter()
            .map(|piece| (piece.rows, piece.audio_rows))
            .collect();
        // The layout's condition capacity is what the source holds past the text capacity and the
        // zero row: a rung is the request's, and the source is sized by it, so the writer and the
        // packing's `prefix_index` read one number.
        let total = text.shape.dim(0);
        let text_rows = u64::from(self.rules.arrangement().text_rows);
        let reserved = total.checked_sub(text_rows + 1).ok_or_else(|| {
            Error::config(format!(
                "a {total}-row prefix source cannot hold {text_rows} text rows and the zero row"
            ))
        })?;
        let reserved = u32::try_from(reserved).map_err(|_| {
            Error::config(format!("{reserved} condition rows do not fit a row count"))
        })?;
        let (_, ranges) = condition_ranges(total, reserved, &pieces)?;
        let audio_channels = u64::from(self.config.transformer.audio_channels);
        // The trace's `clean` views are one per **visual** condition, in packed order, so the
        // counter advances on the visual pieces alone - an audio-only condition contributes none.
        let mut visual_index = 0_usize;
        for (piece, (start, _)) in media.iter().zip(ranges) {
            if piece.audio_rows > 0 {
                let samples = piece.audio.ok_or_else(|| {
                    Error::config(format!(
                        "condition {:?} declares {} audio rows and no samples were offered",
                        piece.role, piece.audio_rows
                    ))
                })?;
                let rows = u64::from(piece.audio_rows);
                // The encoder writes channel-major latent rows in FP32, which is what the input
                // projection reads.
                let encoded = f
                    .arena()
                    .tensor(DType::F32, Shape::new(&[rows, audio_channels]))?;
                encoders.audio.encode(f, samples, encoded)?;
                let start = u32::try_from(start).map_err(|_| {
                    Error::config("a condition row start above u32::MAX".to_owned())
                })?;
                // The projection's output is FP32, and the one rounding is the copy into the rows
                // (`denoiser.py`: `audio_input(rows, output_dtype=torch.float32).to(out.dtype)`).
                let hidden = u64::from(self.config.transformer.hidden_size);
                let projected = f.arena().tensor(DType::F32, Shape::new(&[rows, hidden]))?;
                self.transformer.project_audio(f, encoded, projected)?;
                nn::layers::ScaledAdd
                    .forward(f, row_range(text, start, rows)?, projected, 1.0, 0.0)
                    .map_err(Error::from)?;
            }
            if piece.rows > 0 {
                let visual = u32::try_from(start).map_err(|_| {
                    Error::config("a condition row start above u32::MAX".to_owned())
                })? + piece.audio_rows;
                let clean = trace.and_then(|trace| trace.clean.get(visual_index).copied());
                self.encode_visual(f, piece, encoders, text, visual, clean)?;
                visual_index += 1;
            }
        }
        Ok(())
    }

    /// Encode one visual condition and write its patch rows into the prefix source.
    ///
    /// The chain is the reference's, and each step is its own program:
    ///
    /// 1. the tiled video encoder over the piece's pixels ([`crate::tiled_encode::ClipEncoder`]),
    ///    stitching its tiles into the run's raster;
    /// 2. `quant_conv` once, on the stitched moments ([`crate::encoding::VideoEncoder::project`]);
    /// 3. the posterior epilogue, which samples the condition's latent from the moments and the
    ///    piece's draw and applies the per-channel affine
    ///    ([`crate::encoding::VideoEncoder::posterior`]);
    /// 4. the patchify into `video_channels * 4`-wide rows;
    /// 5. the anchoring, `t · clean + (1 - t) · draw`, in FP32, and the projection into the source.
    ///
    /// The draw is patchified like the latent: the reference's condition noise is drawn at latent
    /// resolution, so the two operands of the anchor are rows only after the same mapping.
    fn encode_visual(
        &self,
        f: &Forward,
        piece: &nn::capability::MediaInput,
        encoders: &ConditionEncoders,
        text: TensorView,
        start: u32,
        trace: Option<TensorView>,
    ) -> Result<()> {
        let pixels = piece.pixels.ok_or_else(|| {
            Error::config(format!(
                "condition {:?} declares {} visual rows and no pixels were offered",
                piece.role, piece.rows
            ))
        })?;
        let draw = piece.noise.ok_or_else(|| {
            Error::config(format!(
                "condition {:?} is visual and the runner offered no draw for it",
                piece.role
            ))
        })?;
        let compression = u64::from(crate::encoding::SPATIAL_RATIO);
        if pixels.shape.rank() != 5 || pixels.shape.dim(0) != 1 {
            return Err(Error::config(format!(
                "condition {:?} reads [1, channels, frames, height, width] pixels, not {:?}",
                piece.role,
                pixels.shape.dims()
            )));
        }
        if pixels.dtype != DType::F32 || draw.dtype != DType::F32 {
            return Err(Error::config(format!(
                "condition {:?} is encoded in FP32: pixels are {} and the draw {}",
                piece.role, pixels.dtype, draw.dtype
            )));
        }
        let (in_channels, frames, height, width) = (
            pixels.shape.dim(1),
            pixels.shape.dim(2),
            pixels.shape.dim(3),
            pixels.shape.dim(4),
        );
        let (taller, wider) = (
            u32::try_from(height).unwrap_or(0),
            u32::try_from(width).unwrap_or(0),
        );
        let clip = crate::tiled_encode::ClipEncoder::new(taller, wider)?;
        let plan = clip.plan();
        // One packed copy per tile, taken from the piece's own raster. `FramePad` - which pads a
        // tile's edges - reads each frame's rows in place, so it wants the *tile's* row stride and a
        // tile cut from a wider raster has the raster's: the kernel refuses it by name ("a frame's
        // pixels of one channel are not contiguous"). `CopyBlock` is the strided-to-packed copy, and
        // the tile's own pixels are a few megabytes against the encoder's set.
        let raster = {
            let strides = pixels.strides.as_slice();
            TensorView::with_strides(
                pixels.offset_elements(0),
                pixels.dtype,
                Shape::new(&[in_channels, frames, height, width]),
                nn::Strides::new(&strides[1..]),
            )
        };
        // A condition is encoded a clip at a time, each clip alone: a still is one clip of one
        // frame, and a reference video its whole clips and a shorter last one.
        let frame_count = u32::try_from(frames)
            .map_err(|_| Error::config("a condition of more frames than u32 holds".to_owned()))?;
        let clips = encoders.video.clips(frame_count)?;
        let longest = clips.iter().map(|clip| clip.frames).max().unwrap_or(1);
        // A run's tiles are uniform (the seam split keeps every tile at the tile size), so one
        // buffer holds any tile of this run and its extents are the plan's largest.
        // The run's raster is the whole encoding's: the plan is the identity crop, so the raster's
        // own latent extent is what the tiles stitch into. It carries the **moments'** channels and
        // not the latent's - the encoder's last level writes `2 * latent_channels`, and `quant_conv`
        // projects those to the two fields the posterior samples from, so the stitch, the projection
        // and the posterior's first operand share one width
        // (`crate::encoding::VideoEncoder::moments`).
        let video_channels = u64::from(self.config.transformer.video_channels);
        let (origin, end) = plan.extent();
        let run_rows = u64::from(end - origin);
        let run_columns = u64::from(wider / crate::encoding::SPATIAL_RATIO);
        let latent_frames = u64::from(encoders.video.latent_frames(frame_count)?);
        let (tile_height, tile_width) = plan.tiles.iter().fold((0_u64, 0_u64), |held, tile| {
            (
                held.0.max(u64::from(tile.rows.1) * compression),
                held.1.max(u64::from(tile.columns.1) * compression),
            )
        });
        let buffer = f.arena().tensor(
            DType::F32,
            Shape::new(&[in_channels, u64::from(longest), tile_height, tile_width]),
        )?;
        // Each tile's block of a clip of the piece's raster, which the encoding packs into `buffer`
        // just before it reads it: one buffer serves every tile only because a tile is encoded
        // before the next is copied, and copying them all first would encode the last tile's pixels
        // every time.
        let tiles_of = |first: u32, length: u32| {
            plan.tiles
                .iter()
                .map(|tile| {
                    TensorView::with_strides(
                        raster.offset_elements(
                            u64::from(first) * height * width
                                + u64::from(tile.rows.0) * compression * width
                                + u64::from(tile.columns.0) * compression,
                        ),
                        raster.dtype,
                        Shape::new(&[
                            in_channels,
                            u64::from(length),
                            u64::from(tile.rows.1) * compression,
                            u64::from(tile.columns.1) * compression,
                        ]),
                        raster.strides,
                    )
                })
                .collect::<Vec<_>>()
        };

        let moment_channels = 2 * video_channels;
        let run = f.arena().tensor(
            DType::F32,
            Shape::new(&[moment_channels, latent_frames, run_rows, run_columns]),
        )?;
        // One reservation for every clip, at the widest clip's blocks, then each clip into its own
        // latent frames of the run.
        let lengths: Vec<u32> = clips.iter().map(|clip| clip.frames).collect();
        let widest = clips.iter().map(|clip| clip.latents).max().unwrap_or(1);
        let widest = TensorView::with_strides(
            run.element(&[0, 0, 0, 0]),
            DType::F32,
            Shape::new(&[moment_channels, u64::from(widest), run_rows, run_columns]),
            run.strides,
        );
        let reserved = clip.reserve(f, &encoders.video, &plan, &lengths, widest)?;
        for span in &clips {
            let blended = TensorView::with_strides(
                run.element(&[0, u64::from(span.latent), 0, 0]),
                DType::F32,
                Shape::new(&[
                    moment_channels,
                    u64::from(span.latents),
                    run_rows,
                    run_columns,
                ]),
                run.strides,
            );
            clip.encode_in(
                f,
                &encoders.video,
                &plan,
                crate::tiled_encode::Scene {
                    pixels: &tiles_of(span.first, span.frames),
                    packed: Some(buffer),
                    blended,
                },
                &reserved,
            )?;
        }

        // `quant_conv` once over the stitched raster, then the posterior's sample: the moments in,
        // the normalised latent out at half their width.
        let moments = f.arena().tensor(DType::F32, run.shape)?;
        encoders.video.project(f, run, moments)?;
        let latent = f.arena().tensor(
            DType::F32,
            Shape::new(&[video_channels, latent_frames, run_rows, run_columns]),
        )?;
        // The posterior's own draw, at the module's seed and the latent's shape: a fresh generator
        // per condition, element zero each time (`encoding.py:174-185`). The piece's draw is used
        // below, for the anchor, and only there.
        let posterior = f.arena().tensor(DType::F32, latent.shape)?;
        let flat = posterior.reshape(Shape::new(&[
            video_channels * latent_frames * run_rows,
            run_columns,
        ]));
        nn::Noise::new().draw(f, flat, nn::NoiseKey::Scalar(POSTERIOR_SEED), 0, 0, 0)?;
        encoders.video.posterior(f, moments, posterior, latent)?;

        // The latent becomes rows, the draw becomes rows - the draw is drawn at latent resolution,
        // so it takes the same mapping - and the anchor mixes them in place over the latent's rows.
        // `ScaledAdd` is `target ← β · target + α · source`, so the reference's `t · clean + (1 - t) ·
        // draw` is β = t on the latent's rows and α = 1 - t on the draw's.
        let patch = nn::layers::Patch::new(video_channels, run_rows, run_columns)?;
        let rows = u64::from(piece.rows);
        let columns = video_channels * 4;
        let clean = f.arena().tensor(DType::F32, Shape::new(&[rows, columns]))?;
        patch.forward(f, latent, clean, latent_frames)?;
        // The diagnosis's view of the condition, before the anchor mixes the draw in: the rows the
        // video encoder emits, which is what a reference's own encode probe returns. `ScaledAdd`
        // with β = 0 is the packed copy, and the served path offers no view and launches nothing.
        if let Some(view) = trace {
            check_trace_view(view, rows, columns, "clean")?;
            nn::layers::ScaledAdd
                .forward(f, view, clean, 1.0, 0.0)
                .map_err(Error::from)?;
        }
        let anchored = f.arena().tensor(DType::F32, Shape::new(&[rows, columns]))?;
        patch.forward(f, draw, anchored, latent_frames)?;
        // `target ← β · target + α · source` with the target the latent's rows and the source the
        // draw's: α weights the draw, β the latent, so the result is `t · clean + (1 - t) · draw`.
        nn::layers::ScaledAdd
            .forward(
                f,
                clean,
                anchored,
                1.0 - VISUAL_CONDITION_TIME,
                VISUAL_CONDITION_TIME,
            )
            .map_err(Error::from)?;
        // The projection's output is FP32, and the one rounding is the copy into the rows
        // (`denoiser.py`: `video_input(anchored, output_dtype=torch.float32).to(out.dtype)`).
        let hidden = u64::from(self.config.transformer.hidden_size);
        let projected = f.arena().tensor(DType::F32, Shape::new(&[rows, hidden]))?;
        self.transformer.project(f, clean, projected)?;
        nn::layers::ScaledAdd
            .forward(f, row_range(text, start, rows)?, projected, 1.0, 0.0)
            .map_err(Error::from)?;
        Ok(())
    }

    /// Refine one request's conditioning: the text encoder's features into the DiT's width.
    ///
    /// `source` is the encoder's `[rows, text_dim]` output and `text` the request's `[rows, hidden]`
    /// conditioning buffer; both are the worker's, and neither is retained. The refiner's
    /// document boundaries are the `refiner_cu_seq_lens_*` tables of `state`, so its queries are
    /// every padded row and its keys stop at the prompt's last row.
    ///
    /// It runs once per request, before the first evaluation
    /// (`uniserve/models/minimax_h3/generation.py:145-171`): the conditioning never changes across a
    /// trajectory, so refining inside the step loop would pay twice per layer per evaluation for a
    /// value that is already right.
    pub fn refine_conditioning(
        &self,
        f: &Forward,
        source: TensorView,
        text: TensorView,
        state: &[TensorView],
        trace: Option<TensorView>,
    ) -> Result<()> {
        let tables = Tables::new(self.config.attention, state);
        let documents = crate::conditioning::Documents {
            cu_seq_lens_q: tables.get("refiner_cu_seq_lens_q")?,
            cu_seq_lens_k: tables.get("refiner_cu_seq_lens_k")?,
            key_rows: self.refiner.rows(),
        };
        // The conditioning column holds the layout's condition rows below the prompt's, and they are
        // the encoders' to write: the refiner's bound is the text capacity, and it reads and writes
        // that many rows.
        let rows = text.shape.dim(0).min(u64::from(self.refiner.rows()));
        let source = row_range(source, 0, rows)?;
        let text = row_range(text, 0, rows)?;
        self.refiner.forward(f, source, &documents, text)?;
        if let Some(trace) = trace {
            // The prompt's rows only: whatever the refiner wrote above them is unreachable and a
            // report that compared it would differ with nothing wrong.
            let tokens = trace.shape.dim(0);
            if tokens > text.shape.dim(0) {
                return Err(Error::config(format!(
                    "the refiner's trace is {tokens} rows of a {}-row source",
                    text.shape.dim(0)
                )));
            }
            if trace.shape.dim(1) != text.shape.dim(1) {
                return Err(Error::config(format!(
                    "the refiner's trace is {} wide and the source {}",
                    trace.shape.dim(1),
                    text.shape.dim(1)
                )));
            }
            nn::layers::ScaledAdd
                .forward(f, trace, row_range(text, 0, tokens)?, 1.0, 0.0)
                .map_err(Error::from)?;
        }
        Ok(())
    }

    /// Project a request's conditions into its prefix source.
    ///
    /// The source is the compact prefix buffer - the prompt's rows, then one row per packed
    /// condition row, then the zero row - which `Layouts::table_specs`' `prefix_index` addresses
    /// directly ([`crate::packing::DensePacking::zero_row`]). The conditions' rows start at the
    /// layout's text capacity, in packed order, audio before video within a segment
    /// ([`crate::packing::condition_segments`]).
    ///
    /// A visual condition is anchored as `t z + (1 - t) noise` at `t = VISUAL_CONDITION_TIME` and
    /// projected by the layer that projects the generated rows of its modality; an audio reference
    /// is projected unchanged, which is [`AUDIO_CONDITION_TIME`]'s `1.0`
    /// (`uniserve/models/minimax_h3/denoiser.py:510-532`).
    ///
    /// **Precision, declared.** The reference patches, mixes and projects a visual condition in
    /// FP32 and rounds once, at the copy into the prefix source. This build patches the draw and
    /// stores the mix in the prefix source's dtype, so a visual condition's rows round twice: the
    /// draw at the patch (its relative error is BF16's half-ulp) and the sum at the mix. The audio
    /// reference rounds once, at its projection. The residual is a fraction of the conditioning
    /// level's `1 - t` term and is far below the noise it is mixed with; an exact path needs an F32
    /// column of the input projection (2 MB of widened weights at load) because the crate has no
    /// F32→BF16 convert and `Patch` writes one dtype.
    ///
    /// The model allocates nothing: `out`, the latents and the draws are the runner's views, and
    /// the one scratch row buffer a visual condition needs comes from the forward pass's arena.
    /// `Denoiser::prepare_conditioning` with the conditioned path's diagnostic views.
    ///
    /// **The served arm is this one with no trace**, so a page a harness writes through it is the
    /// arithmetic a request runs, launch for launch. A trace is the design's numeric diagnosis
    /// (`specs/worker.md` 10) and never a gate: `clean` receives each visual condition's rows as the
    /// video encoder emits them (`[rows, channels * 4]` FP32), before the anchor mixes the draw in,
    /// which is the value a reference's encoder probe returns; `refined` receives the refiner's rows
    /// where the pack gathers them. Nothing is written unless a view is offered.
    pub fn prepare_conditioning_traced(
        &self,
        f: &Forward,
        cond: nn::capability::Conditioning<'_>,
        trace: Option<&ConditionTrace<'_>>,
    ) -> Result<()> {
        // The refiner's rows are the denoiser's conditioning (`denoiser.py:510-532`): every task's
        // text passes through it once, in place over the runner's `text` buffer, and the encoder's
        // own features arrive in `source`.
        //
        // The two widths are what makes the refiner the column's only producer: the encoder's
        // features are `text_dim` (5120) and the conditioning column is `hidden_size` (5376), so
        // nothing on the encoder's side writes a row of `text`. A request with no source therefore
        // asks this model to denoise a column nothing projected - the shape a model whose encoder
        // width *is* its column width has, and never this one's served path.
        //
        // A conditioned request's condition rows, below the prompt's: the encoders write them from
        // `media`, at the offsets the layout's own counts place them, and the two writes are
        // disjoint. A request with no media has no condition rows at all, which is every `t2va`.
        let Some(source) = cond.source else {
            return Ok(());
        };
        if !cond.media.is_empty() {
            self.encode_media(f, cond.media, cond.text, trace)?;
        }
        self.refine_conditioning(
            f,
            source,
            cond.text,
            cond.state,
            trace.and_then(|t| t.refined),
        )
    }

    pub fn encode_conditions(
        &self,
        f: &Forward,
        request: &VideoRequest<'_>,
        layout: Layout,
        conditions: ConditionLatents<'_>,
        out: TensorView,
        trace: Option<&ConditionTrace<'_>>,
    ) -> Result<()> {
        let canvas = canvas_of_latents(&layout)?;
        let declared = conditions_of(request);
        let segments = packing::condition_segments(&declared, &canvas, self.config.clip)?;
        // The source is sized by the layout the runner laid the request out under, so a buffer of
        // another size is refused here rather than read past its rows below.
        let packing = self.rules.packing_of(&layout)?;
        let hidden = u64::from(self.config.transformer.hidden_size);
        let rows = u64::from(packing.text_rows + packing.condition_rows + 1);
        if out.shape.rank() != 2 || out.shape.dim(0) != rows || out.shape.dim(1) != hidden {
            return Err(Error::config(format!(
                "the H3 prefix source is [{rows}, {hidden}] rows for this layout, got {:?}",
                out.shape.dims()
            )));
        }
        if segments.is_empty() {
            return Ok(());
        }
        let video_channels = u64::from(self.config.transformer.video_channels);
        let audio_channels = u64::from(self.config.transformer.audio_channels);
        let width = video_channels * 4;
        let views = ConditionViews::new(&declared, conditions.latents)?;
        let visual = segments
            .iter()
            .filter(|segment| segment.visual.is_some())
            .count();
        if let Some(trace) = trace
            && (trace.draws.len() != visual || trace.clean.len() != visual)
        {
            return Err(Error::config(format!(
                "the condition trace offers {} draws and {} latents for {visual} visual conditions",
                trace.draws.len(),
                trace.clean.len()
            )));
        }
        let mut traced = 0usize;
        let mut draws = conditions.noise.iter();
        let mut row = packing.text_rows;
        for segment in &segments {
            let (visual_view, audio_view) = views.of(segment.condition);
            // Audio rows lead their segment in the packed order.
            if segment.audio_rows() > 0 {
                let audio = audio_view.ok_or_else(|| {
                    Error::config(format!(
                        "condition {} carries {} audio rows and no latents were offered",
                        segment.condition,
                        segment.audio_rows()
                    ))
                })?;
                let count = u64::from(segment.audio_rows());
                if audio.dtype != DType::F32 {
                    return Err(Error::config(format!(
                        "condition {}'s audio rows are {}, and the reference projects them in \
                         FP32 (`denoiser.py:613-618`)",
                        segment.condition, audio.dtype
                    )));
                }
                if audio.shape.dims() != [count, audio_channels] {
                    return Err(Error::config(format!(
                        "condition {}'s audio rows are [{count}, {audio_channels}], got {:?}",
                        segment.condition,
                        audio.shape.dims()
                    )));
                }
                self.transformer
                    .project_audio(f, audio, row_range(out, row, count)?)?;
                row += segment.audio_rows();
            }
            let Some(visual) = segment.visual else {
                continue;
            };
            let clean = visual_view.ok_or_else(|| {
                Error::config(format!(
                    "condition {} carries {} visual rows and no latents were offered",
                    segment.condition,
                    visual.rows()
                ))
            })?;
            let count = u64::from(visual.rows());
            if clean.dtype != DType::F32 {
                return Err(Error::config(format!(
                    "condition {}'s patch rows are {}, and the encoder's normalised rows are \
                     FP32 (`denoiser.py:605`)",
                    segment.condition, clean.dtype
                )));
            }
            if clean.shape.dims() != [count, width] {
                return Err(Error::config(format!(
                    "condition {}'s visual patch rows are [{count}, {width}], got {:?}",
                    segment.condition,
                    clean.shape.dims()
                )));
            }
            let (frames, height, columns) = (
                u64::from(visual.frames),
                u64::from(visual.raster.height),
                u64::from(visual.raster.width),
            );
            let draw = draws.next().ok_or_else(|| {
                Error::config(format!(
                    "condition {} is visual and the runner offered no draw for it",
                    segment.condition
                ))
            })?;
            if draw.shape.dims() != [video_channels, frames, height, columns] {
                return Err(Error::config(format!(
                    "condition {}'s draw is [{video_channels}, {frames}, {height}, {columns}], \
                     got {:?}",
                    segment.condition,
                    draw.shape.dims()
                )));
            }
            if draw.dtype != DType::F32 {
                return Err(Error::config(format!(
                    "condition {}'s draw is {}, and the reference anchors in FP32 \
                     (`denoiser.py:609-615`)",
                    segment.condition, draw.dtype
                )));
            }
            // The draw becomes packed rows, then the anchoring mixes the condition's latents in at
            // the conditioning level: `target = (1 - t) target + t clean`. The canvas is FP32
            // because the reference's is: both operands are FP32 there, and the projection reads
            // the anchored rows in FP32 and rounds once, into `out` (`denoiser.py:609-620`).
            let anchor = f.arena().tensor(DType::F32, Shape::new(&[count, width]))?;
            nn::layers::Patch::new(video_channels, height, columns)?
                .forward(f, *draw, anchor, frames)
                .map_err(Error::from)?;
            if let Some(trace) = trace {
                // The two operands before the mix: `target = 0 * target + 1 * source` writes the
                // view without reading what it held.
                check_trace_view(trace.draws[traced], count, width, "draw")?;
                check_trace_view(trace.clean[traced], count, width, "clean")?;
                nn::layers::ScaledAdd
                    .forward(f, trace.draws[traced], anchor, 1.0, 0.0)
                    .map_err(Error::from)?;
                nn::layers::ScaledAdd
                    .forward(f, trace.clean[traced], clean, 1.0, 0.0)
                    .map_err(Error::from)?;
                traced += 1;
            }
            nn::layers::ScaledAdd
                .forward(
                    f,
                    anchor,
                    clean,
                    VISUAL_CONDITION_TIME,
                    1.0 - VISUAL_CONDITION_TIME,
                )
                .map_err(Error::from)?;
            self.transformer
                .project(f, anchor, row_range(out, row, count)?)?;
            row += visual.rows();
        }
        if let Some(source) = trace.and_then(|trace| trace.source) {
            let rows = u64::from(packing.condition_rows);
            check_trace_view(source, rows, hidden, "source")?;
            nn::layers::ScaledAdd
                .forward(
                    f,
                    source,
                    row_range(out, packing.text_rows, rows)?,
                    1.0,
                    0.0,
                )
                .map_err(Error::from)?;
        }
        Ok(())
    }

    /// One denoising step: the latents in, the model's velocity, and the schedule's update.
    ///
    /// The packed rows are assembled from the layout's tables: every row gathers its prefix source
    /// (the refined text row it reads, or the zero row), then each modality's patchified latents are
    /// projected and scattered into their own rows. The stack runs once, the step's modulation is
    /// gathered from the AdaLN cache, and each modality's velocity is projected back, unpatched and
    /// folded into its latents by the solver.
    pub fn denoise(
        &self,
        f: &Forward,
        step: &nn::Step,
        latents: nn::Latents,
        cond: nn::Conditioning<'_>,
    ) -> Result<()> {
        self.denoise_traced(f, step, latents, cond, None)
    }

    /// [`Denoiser::denoise`] with the design's numeric diagnosis attached.
    ///
    /// `trace` names layers and borrows one view each, which this forward fills with the packed
    /// rows that layer produced. With `None` the forward is the served one: the only difference a
    /// trace makes is one gather per named layer (`specs/worker.md` 10 - a diagnosis, never a gate).
    pub fn denoise_traced(
        &self,
        f: &Forward,
        step: &nn::Step,
        latents: nn::Latents,
        cond: nn::Conditioning<'_>,
        trace: Option<&crate::transformer::Trace<'_>>,
    ) -> Result<()> {
        let tables = Tables::new(self.config.attention, cond.state);
        let video = latents.video;
        let channels = video.shape.dim(0);
        let frames = video.shape.dim(1);
        let height = video.shape.dim(2);
        let width = video.shape.dim(3);
        let tokens = u64::from(tables.get("groups")?.shape.dim(0) as u32);
        // The same bound `denoise_arena` checks, at the one other place a request reaches this block:
        // a request wider than the deployment was loaded for has no attention module to run.
        if tokens > u64::from(self.rows) {
            return Err(Error::config(format!(
                "this denoiser was loaded for {} packed rows and the request has {tokens}; the \
                 attention is compiled for the bound",
                self.rows
            )));
        }
        let hidden_size = u64::from(self.config.transformer.hidden_size);
        let dtype = cond.text.dtype;
        // The trajectory is FP32 and the reference says so with its own refusal
        // (`denoiser.py:1206-1216`: a latent input of any other type is a `ValueError`). A sample in
        // the stack's type would round the trajectory once per evaluation - a drift no shape refuses
        // and no page localises - so this refuses instead of running the cheaper arithmetic.
        if video.dtype != SAMPLE_DTYPE {
            return Err(Error::config(format!(
                "the video latents are {} and a trajectory is {SAMPLE_DTYPE}: the sample is what \
                 the solver accumulates into, and the stack's own element type is {dtype}",
                video.dtype
            )));
        }
        // The trajectory is FP32 and the stack is the deployment's, and the two are crossed twice
        // per evaluation: the input projection reads the FP32 sample, the head reads the stack's own
        // rows. Both crossings are legal whatever the stack's type is, because **the four end
        // projections are FP32 parameters at every preset** (`precision.py:106-117`) and the
        // activation is widened to meet them (`Transformer::wide`) - so the products are FP32 on
        // both sides and the stack's type never touches a weight of another type.
        //
        // What remains the stack's alone is the packed rows in between: `scaled_add` narrows the
        // projection's FP32 output into them and widens them back at the head, one launch each, and
        // both cells exist for a BF16 and an FP32 target (`target_bf16_source_f32` and
        // `target_f32_source_bf16`, 5376 wide).
        let arena = f.arena();

        // This rank owns one contiguous interval of the packed rows, and the sample and the tables
        // are the whole sequence's on every rank: a table a row reads is cut to the interval here,
        // and each modality's owned rows are named by the rank's own `*_sources` and `*_slots`.
        let shard = u64::from(self.shard.max(1));
        if !tokens.is_multiple_of(shard) {
            return Err(Error::config(format!(
                "{tokens} packed rows do not divide over {shard} ranks"
            )));
        }
        let local = tokens / shard;
        let first = u64::from(self.rank) * local;
        let own = |table: TensorView| {
            table
                .at(table.offset_elements(first))
                .reshape(Shape::new(&[local]))
        };

        // The packed rows: every row gathers its prefix source, so a generated row reads the zero
        // row and a text row reads itself.
        let hidden = arena.tensor(dtype, Shape::new(&[local, hidden_size]))?;
        nn::layers::gather_rows(f, cond.text, own(tables.get("prefix_index")?), hidden)?;

        // The video rows: patchify, project, scatter.
        let video_indices = tables.get("video_indices")?;
        let video_rows = u64::from(video_indices.shape.dim(0) as u32);
        let video_sources = tables.get("video_sources")?;
        let video_slots = tables.get("video_slots")?;
        let own_video = video_sources.shape.dim(0);
        // **The sample is the latent's own type, and the stack is the checkpoint's.** The reference
        // keeps its trajectory in FP32 (`denoiser.py:1206-1216` refuses anything else) and rounds
        // only inside a launch, so the boundary is crossed twice per evaluation and each crossing is
        // one launch: the patch and the input projection read the sample's type, the projection's
        // output is narrowed into the packed rows, and the prediction is widened back before the
        // solver - whose step reads one type for the sample and the velocity (`solver.rs`).
        // Allocating these in `cond.text.dtype` instead would round the trajectory itself at every
        // evaluation, which no shape refuses and no page shows.
        let sample_dtype = SAMPLE_DTYPE;
        let patch = nn::layers::Patch::new(channels, height, width)?;
        // The one inverse of the pack above: the DiT's `1x2x2` tiles back into the latent raster.
        // It reads the token grid from `out`, so a clip of another length replays the same launch.
        let unpatch = nn::layers::UnpatchifyVideo::new(sample_dtype, channels, 1, 2)?;
        let patched = arena.tensor(sample_dtype, Shape::new(&[video_rows, channels * 4]))?;
        patch.forward(f, video, patched, frames)?;
        // The reference projects with `output_dtype=torch.float32` and narrows afterwards
        // (`denoiser.py:1216-1221`): the projection's weights are the checkpoint's, its activation
        // and its output are FP32, and the one rounding into the packed rows is the narrowing.
        if own_video > 0 {
            let owned = arena.tensor(sample_dtype, Shape::new(&[own_video, channels * 4]))?;
            nn::layers::gather_rows(f, patched, video_sources, owned)?;
            let projected = arena.tensor(DType::F32, Shape::new(&[own_video, hidden_size]))?;
            self.transformer.project(f, owned, projected)?;
            let narrowed = arena.tensor(dtype, Shape::new(&[own_video, hidden_size]))?;
            nn::layers::ScaledAdd.forward(f, narrowed, projected, 1.0, 0.0)?;
            nn::layers::Rows::new(dtype, hidden_size)?.replace(f, hidden, narrowed, video_slots)?;
        }

        // One audio stream or none. A layout with audio rows and no audio latents would otherwise
        // denoise the video and return it without sound, and a missing table where the layout has
        // one is a disagreement about the layout rather than an absent stream.
        let audio = match (latents.audio, tables.get("audio_indices")) {
            (Some(audio), Ok(indices)) => Some((audio, indices)),
            (None, Ok(indices)) if indices.shape.dim(0) > 0 => {
                return Err(Error::config(format!(
                    "this layout has {} audio rows and no audio latents",
                    indices.shape.dim(0)
                )));
            }
            (Some(_), Err(error)) => return Err(error),
            _ => None,
        };

        // The audio rows: one element per channel-step pair, projected like any other row.
        if let Some((audio, audio_indices)) = audio {
            // The audio trajectory is the video's type for the same reason, and the refusal belongs
            // here rather than in the solver: a sample in another type would first widen the
            // projection's 32-wide rows (`scaled_add target_f32_source_bf16_columns_32`) and only
            // then fail against the prediction, which reads as a velocity defect instead of a dtype
            // one.
            if audio.dtype != DType::F32 {
                return Err(Error::config(format!(
                    "the audio latents are {} and a trajectory is FP32",
                    audio.dtype
                )));
            }
            let audio_rows = u64::from(audio_indices.shape.dim(0) as u32);
            let features = u64::from(self.config.transformer.audio_channels);
            let flat = TensorView::with_strides(
                audio.ptr,
                audio.dtype,
                Shape::new(&[audio_rows, features]),
                nn::Strides::new(&[features, 1]),
            );
            let audio_sources = tables.get("audio_sources")?;
            let own_audio = audio_sources.shape.dim(0);
            if own_audio > 0 {
                let owned = arena.tensor(sample_dtype, Shape::new(&[own_audio, features]))?;
                nn::layers::gather_rows(f, flat, audio_sources, owned)?;
                // The audio's own two crossings, for the reason the video's: the latent is the
                // sample's type and the projection's FP32 output is narrowed into the packed rows.
                let projected = arena.tensor(DType::F32, Shape::new(&[own_audio, hidden_size]))?;
                self.transformer.project_audio(f, owned, projected)?;
                let narrowed = arena.tensor(dtype, Shape::new(&[own_audio, hidden_size]))?;
                nn::layers::ScaledAdd.forward(f, narrowed, projected, 1.0, 0.0)?;
                nn::layers::Rows::new(dtype, hidden_size)?.replace(
                    f,
                    hidden,
                    narrowed,
                    tables.get("audio_slots")?,
                )?;
            }
        }

        // The rotary tables and the visibility of the one packed sequence. Each of the three axes
        // rotates `rope_frequency_dim` channel pairs, and a factor table holds one column per pair.
        // One column per rotated **pair**, and `rope_freq_dim` is already the pairs of one axis:
        // the operator's `rotated` counts channels (three axes of `2 * rope_freq_dim` = 96) and a
        // factor table is half that (48), so this product is the table's width and not the
        // operator's field. The two numbers differ by two and are named alike.
        let pairs = u64::from(self.config.transformer.rope_frequency_dim) * 3;
        let cosine = arena.tensor(DType::F32, Shape::new(&[tokens, pairs]))?;
        let sine = arena.tensor(DType::F32, Shape::new(&[tokens, pairs]))?;
        self.transformer
            .factors(f, tables.get("position_ids")?, cosine, sine)?;
        // The dense kind reads the whole packed sequence in one go; the sparse kind reads the
        // segment tables instead and has no visibility pair at all.
        let visible = self.config.attention.is_dense().then(|| {
            Ok::<_, Error>(Visible {
                cu_seq_lens_q: tables.get("cu_seq_lens_q")?,
                cu_seq_lens_k: tables.get("cu_seq_lens_k")?,
                key_rows: tokens as u32,
            })
        });
        let visible = match visible {
            None => None,
            Some(visible) => Some(visible?),
        };
        // A sparse deployment's layers read the valid rows of each tile, the segment of each tile,
        // and the keep table; a dense one reads the whole packed sequence through `cu_seq_lens`
        // and declares none of them. The two are exclusive, and the condition here was the wrong
        // way round: a dense deployment asked its layout for `variable_block_sizes` - a table that
        // layout never declares - and the refusal named the table rather than the inversion
        // (`configuration: variable_block_sizes is not a state table of this kind`, launch 18).
        let segments = if self.config.attention.is_dense() {
            None
        } else {
            Some(crate::attention_sparse::Segments {
                sizes: tables.get("variable_block_sizes")?,
                segments: tables.get("tile_segments")?,
                keep: tables.get("segment_keep")?,
            })
        };

        // The stack, conditioned by the step's gathered modulation.
        let step_index = tables.get("step")?;
        self.transformer.forward(
            f,
            &self.cache,
            step_index,
            hidden,
            own(tables.get("modulation_indices")?),
            Factors { cosine, sine },
            visible.as_ref(),
            segments.as_ref(),
            tokens as u32,
            trace,
        )?;

        // The output: the step's final modulation, then each modality's head.
        let groups = self.cache.groups().count() as u64;
        let entries = arena.tensor(DType::I32, Shape::new(&[1, groups]))?;
        self.cache.entries_of(f, step_index, entries)?;
        // The plan states this buffer's element as the stack's (`element`, `output.modulation`), and
        // so does this line: one number, two guards. The cache then refuses a table of another type
        // (`AdaLnCache::output` compares the destination with its own weights), so a cache whose
        // projections drifted away from the stack's type is a named refusal and not a mixed gather.
        let modulation = arena.tensor(hidden.dtype, Shape::new(&[groups, 2 * hidden_size]))?;
        self.cache.output(f, entries, modulation)?;

        // The solver's scale: one FP32 word the `step` table indexes into the coefficient table.
        let coefficient = arena.tensor(DType::F32, Shape::new(&[1, 1]))?;
        // A rank predicts the rows it owns, and every rank then holds every row's prediction: the
        // sample is whole on every rank, so the solver's update is the same arithmetic everywhere
        // and the group's trajectories cannot drift apart.
        let groups_table = own(tables.get("groups")?);
        let predicted = if own_video > 0 {
            let video_hidden = arena.tensor(dtype, Shape::new(&[own_video, hidden_size]))?;
            nn::layers::gather_rows(f, hidden, video_slots, video_hidden)?;
            let video_groups = arena.tensor(DType::I32, Shape::new(&[own_video]))?;
            nn::layers::gather_rows(
                f,
                one_column(groups_table),
                video_slots,
                one_column(video_groups),
            )?;
            // The head is FP32 in the reference (`transformer.py:404-421`), so the prediction is
            // written wide and never narrowed: the solver's sample is the latent's type and the
            // velocity must be the same one.
            let rows = arena.tensor(sample_dtype, Shape::new(&[own_video, channels * 4]))?;
            self.transformer.velocity(
                f,
                video_hidden,
                Modality::Video,
                modulation,
                video_groups,
                step.index,
                step_index,
                rows,
            )?;
            Some(rows)
        } else {
            None
        };
        let velocity_rows = self.collected(
            f,
            predicted,
            video_slots,
            video_indices,
            channels * 4,
            local,
            tokens,
        )?;
        let velocity = arena.tensor(sample_dtype, video.shape)?;
        unpatch.forward(f, velocity_rows, velocity)?;
        // The solver's coefficient is gathered on the device from the worker's `step` table, so
        // this launch addresses one staging word whose contents change per evaluation rather than
        // one buffer per evaluation; the same word serves the audio step's gather.
        self.solver.step(
            f,
            Modality::Video,
            video,
            velocity,
            step.index,
            tables.get("step")?,
            coefficient,
        )?;

        if let Some((audio, audio_indices)) = audio {
            let audio_rows = u64::from(audio_indices.shape.dim(0) as u32);
            let features = u64::from(self.config.transformer.audio_channels);
            let audio_slots = tables.get("audio_slots")?;
            let own_audio = audio_slots.shape.dim(0);
            let predicted = if own_audio > 0 {
                let audio_hidden = arena.tensor(dtype, Shape::new(&[own_audio, hidden_size]))?;
                nn::layers::gather_rows(f, hidden, audio_slots, audio_hidden)?;
                let audio_groups = arena.tensor(DType::I32, Shape::new(&[own_audio]))?;
                nn::layers::gather_rows(
                    f,
                    one_column(groups_table),
                    audio_slots,
                    one_column(audio_groups),
                )?;
                let rows = arena.tensor(sample_dtype, Shape::new(&[own_audio, features]))?;
                self.transformer.velocity(
                    f,
                    audio_hidden,
                    Modality::Audio,
                    modulation,
                    audio_groups,
                    step.index,
                    step_index,
                    rows,
                )?;
                Some(rows)
            } else {
                None
            };
            let velocity = self.collected(
                f,
                predicted,
                audio_slots,
                audio_indices,
                features,
                local,
                tokens,
            )?;
            let flat = TensorView::with_strides(
                audio.ptr,
                audio.dtype,
                Shape::new(&[audio_rows, features]),
                nn::Strides::new(&[features, 1]),
            );
            self.solver.step(
                f,
                Modality::Audio,
                flat,
                velocity,
                step.index,
                tables.get("step")?,
                coefficient,
            )?;
        }
        Ok(())
    }

    /// Every row's prediction of one modality, in the modality's order, from the rows this rank
    /// predicted.
    ///
    /// At one rank the predicted rows are all of them. A rank of a group places its rows in its own
    /// interval of the packed sequence, every member's intervals are gathered in member order, and
    /// the modality's rows are read out of the whole sequence by `indices`. The rows of an interval
    /// no modality row lands in are never read, so they are left as the arena holds them.
    #[allow(clippy::too_many_arguments)]
    fn collected(
        &self,
        f: &Forward,
        predicted: Option<TensorView>,
        slots: TensorView,
        indices: TensorView,
        width: u64,
        local: u64,
        tokens: u64,
    ) -> Result<TensorView> {
        let arena = f.arena();
        if self.shard <= 1 {
            return predicted.ok_or_else(|| {
                Error::config("one rank owns every row, and it predicted none".to_owned())
            });
        }
        let interval = arena.tensor(SAMPLE_DTYPE, Shape::new(&[local, width]))?;
        if let Some(predicted) = predicted {
            nn::layers::Rows::new(SAMPLE_DTYPE, width)?.replace(f, interval, predicted, slots)?;
        }
        let sequence = arena.tensor(SAMPLE_DTYPE, Shape::new(&[tokens, width]))?;
        let group = f.group.ok_or_else(|| {
            Error::Unsupported(
                "a sequence shard collects its predictions over a group, and this forward \
                 carries none"
                    .to_owned(),
            )
        })?;
        group.all_gather(interval, sequence, f.stream()?)?;
        let rows = arena.tensor(SAMPLE_DTYPE, Shape::new(&[indices.shape.dim(0), width]))?;
        nn::layers::gather_rows(f, sequence, indices, rows)?;
        Ok(rows)
    }

    /// Read one DiT partition and everything around it.
    ///
    /// `rows` bounds the packed row count this deployment serves, which sizes the attention's
    /// scratch. The cache is sized from the checkpoint's schedule alone - the distinct noise levels
    /// of its grid - so no request grows it and two requests of one shape share it.
    pub fn load(
        config: DenoiserConfig,
        arrangement: Arrangement,
        rows: u32,
        partition: &crate::Partition,
        loader: &mut WeightLoader<'_>,
        heads: Option<crate::transformer::StudentHeads>,
    ) -> Result<Self> {
        let schedule = Schedule::new(&config.schedule)?;
        let evaluations = schedule.num_steps()?;
        let conditioned = config
            .tasks
            .iter()
            .any(|task| !matches!(task, nn::capability::VideoTask::TextToVideoAudio));
        let groups = Groups::of(conditioned);
        let (times, table) = crate::modulation::entries(
            &schedule.timesteps(Modality::Video),
            &schedule.timesteps(Modality::Audio),
            groups,
        )?;
        let entries = times.len() as u64;
        let hidden = u64::from(config.transformer.hidden_size);
        let columns = ADALN_COLUMNS * hidden;
        let frequency = u64::from(config.transformer.frequency_dim);
        let projections = Projections::load(&config.transformer, loader)?;

        let frequencies = loader.constant(
            "minimax_h3.adaln.frequencies",
            DType::F32,
            Shape::new(&[entries, frequency]),
            &projections.frequencies(&times)?,
            "minimax-h3",
        )?;
        let mut entry_bytes = Vec::with_capacity(table.len() * 4);
        for value in &table {
            entry_bytes.extend_from_slice(&value.to_le_bytes());
        }
        // One row per timestep of the ladder, and the row count comes from the table itself: the
        // grid holds a level for the clean end as well, so a 49-evaluation schedule states 50 rows
        // (one more than the steps a request takes), while `entries` counts the *distinct* levels
        // those rows point at - which dedup can make fewer than the rows. Declaring `evaluations` or
        // `entries` rows made the bytes and the shape disagree and the load refused.
        let entry_rows = u64::try_from(table.len() / groups.count() as usize).map_err(|_| {
            Error::checkpoint("an AdaLN entry table wider than u64::MAX rows".to_owned())
        })?;
        if entry_rows < u64::from(evaluations) {
            return Err(Error::checkpoint(format!(
                "the AdaLN entry table has {entry_rows} rows for {evaluations} evaluations"
            )));
        }
        let entry_table = loader.constant(
            "minimax_h3.adaln.entries",
            DType::I32,
            Shape::new(&[entry_rows, u64::from(groups.count())]),
            &entry_bytes,
            "minimax-h3",
        )?;

        // The tables are the modulation projections' own type, because the reference allocates them
        // from the product (`adaln_cache.py:141-145`) - so at a BF16 stack they are BF16 and at an
        // FP32 deployment FP32. The timestep pair in front of them stays FP32, and the one narrowing
        // between the two halves lives in `Projections::compute`.
        let table_type = projections.dtype();
        let unit = table_type.bytes().unwrap_or(4);
        let mut adaln_layers = Vec::with_capacity(config.transformer.num_hidden_layers as usize);
        let zeros = vec![0_u8; (entries * columns * unit) as usize];
        for index in 0..config.transformer.num_hidden_layers {
            adaln_layers.push(loader.writable_constant(
                &format!("minimax_h3.adaln.layers.{index}"),
                table_type,
                Shape::new(&[entries, columns]),
                &zeros,
                "minimax-h3",
            )?);
        }
        let adaln_output = loader.writable_constant(
            "minimax_h3.adaln.output",
            table_type,
            Shape::new(&[entries, 2 * hidden]),
            &vec![0_u8; (entries * 2 * hidden * unit) as usize],
            "minimax-h3",
        )?;
        let cache = AdaLnCache::new(
            adaln_layers.clone(),
            adaln_output,
            entry_table,
            groups,
            config.transformer.hidden_size,
            config.transformer.time_dim,
        )?;
        // A sparse deployment states the compression's tile bound once, from its own layouts: the
        // score matrix is `[heads, query tiles, tiles]` and must cover every video tile any layout
        // it serves can hold. A dense one has no tiles, and the stack's attention kind never reads
        // the zero.
        let tile_bound = match config.attention {
            crate::config::AttentionKind::Sparse { .. } => {
                // The value is this deployment's own bound - the longest layout it can be asked
                // for, a cap over canvases and frame counts rather than any one request's count -
                // and the key names its *class*: `CompressPooledDescriptor`'s key is
                // `max_tiles.next_power_of_two()`, because the package compiles the reduction to
                // the bound's next power of two and two bounds inside one class are one module.
                // The descriptor keeps the live bound, since the tolerance is written against the
                // reduction length itself.
                Layouts::new(config.clone(), arrangement)?.tiles_bound()?
            }
            crate::config::AttentionKind::Dense => 0,
        };
        let transformer = Transformer::load(
            &config.transformer,
            config.attention,
            rows,
            tile_bound,
            partition,
            loader,
            heads,
        )?;
        let solver = Solver::load(&schedule, loader)?;
        // The refiner reads the text encoder's features and writes the DiT's conditioning; its rows
        // are bounded by the same packed-row bound the stack's attention is sized for.
        // The refiner reads the encoder's rows, whose count is the text capacity - not the packed
        // row bound: its scratch is sized per row it can see, and a packed bound would size it for
        // every video row of the widest layout.
        let refiner = crate::conditioning::TokenRefiner::load(
            &config.transformer,
            arrangement.text_rows,
            loader,
        )?;
        let rules = Layouts::new(config.clone(), arrangement)?;
        Ok(Self {
            config,
            rules,
            schedule,
            solver,
            transformer,
            projections,
            cache,
            adaln_layers,
            adaln_output,
            frequencies,
            evaluations,
            refiner,
            places_encoders: partition
                .blocks
                .contains(crate::partition::Component::LatentEncoder),
            encoders: None,
            rows,
            // The denoiser's own rank, not the text encoder's: the two shards are independent, and
            // with one tensor shard every rank's `tensor_rank` is 0 (`partition.rs`).
            rank: arrangement.rank,
            shard: arrangement.shard,
        })
    }

    /// Every checkpoint tensor this block's partition reads, as complete names.
    ///
    /// The refiner's, the stack's and the AdaLN cache's, in the order `load` reads them. The
    /// condition encoders are **not** here: they are the `latent_encoder` component's, loaded from
    /// their own checkpoint directories (`vae`, `audio_vae`) and answered by their own `keys`, even
    /// though this block holds them once a deployment places them.
    pub fn keys(config: &DenoiserConfig) -> Vec<String> {
        let transformer = &config.transformer;
        let mut keys = crate::conditioning::TokenRefiner::keys(transformer);
        keys.extend(Transformer::keys(transformer, config.attention));
        keys.extend(crate::modulation::AdaLnCache::keys(transformer));
        keys
    }

    /// The checkpoint this denoiser evaluates.
    pub const fn config(&self) -> &DenoiserConfig {
        &self.config
    }

    /// The layout rule a request is laid out by.
    pub const fn rules(&self) -> &Layouts {
        &self.rules
    }

    /// The schedule this denoiser steps along.
    pub const fn schedule(&self) -> &Schedule {
        &self.schedule
    }

    /// The solver that advances a sample.
    pub const fn solver(&self) -> &Solver {
        &self.solver
    }

    /// The element type the block computes in: the checkpoint's own, read from a block's gain.
    ///
    /// Not from an end projection: those four are pinned FP32 at every preset, so they would report
    /// the pin and a harness at `auto` would allocate FP32 activations against BF16 blocks.
    ///
    /// A runner allocates the prefix source, the latents and the conditioning's views in it. The
    /// design's default `dtype = "auto"` is the checkpoint's representation converted nowhere, so the
    /// model states the type rather than letting a caller assume one.
    pub fn dtype(&self) -> DType {
        self.transformer.dtype()
    }

    /// How many solver evaluations one request makes.
    pub const fn evaluations(&self) -> u32 {
        self.evaluations
    }

    /// The tables the runner fills after loading.
    ///
    /// Empty by construction: the AdaLN cache is model memory (see [`Denoiser`]).
    pub fn constant_tables(&self) -> Vec<StateTable> {
        Vec::new()
    }

    /// Fill the AdaLN cache from the schedule's times: the one post-load launch.
    pub fn prepare_constants(&self, f: &Forward, out: &[TensorView]) -> Result<()> {
        if !out.is_empty() {
            return Err(Error::config(format!(
                "the H3 AdaLN cache is model memory; {} worker views were offered",
                out.len()
            )));
        }
        let views: Vec<TensorView> = self.adaln_layers.iter().map(|layer| layer.view()).collect();
        self.projections
            .compute(f, self.frequencies.view(), &views, self.adaln_output.view())
    }
}

#[cfg(test)]
mod tests {

    /// The denoiser states the two stacks and the orchestration around them: the pack and its
    /// inverse, the draw, the strided copy, the gathers outside the modulation, and the blends the
    /// conditioner and the solver issue. The trajectory's width is a deployment fact and the only one.
    #[test]
    fn the_denoiser_states_its_orchestration_beside_its_stacks() {
        let config = crate::config::tests_support::released_config();
        let (_, denoiser) = config
            .denoisers
            .iter()
            .next()
            .expect("a released DiT partition");
        let programs = Denoiser::programs(
            &denoiser.transformer,
            denoiser.attention,
            nn::DType::BF16,
            nn::DType::F32,
            32_000,
            Groups::of(true),
            &crate::partition::Partition::single(),
            &[1_000_000],
            0,
        )
        .expect("the denoiser's programs");
        let has = |operator: &str| programs.iter().any(|(name, _)| name == operator);

        for operator in [
            "patch",
            "philox_noise",
            "copy_block",
            "unpatchify_video",
            "scaled_add",
            "scaled_residual",
            "logits_gather",
        ] {
            assert!(has(operator), "{operator} is missing: {programs:?}");
        }
        // The solver's residual over the whole trajectory is a second program from the patch-column
        // one, because the width is in the key.
        let residuals = programs
            .iter()
            .filter(|(name, _)| name == "scaled_residual")
            .map(|(_, key)| key.clone())
            .collect::<Vec<_>>();
        assert!(
            residuals.iter().any(|key| key.contains("columns_1000000")),
            "{residuals:?}"
        );
    }
    use super::*;
    use crate::config::{AttentionKind, DenoiserConfig, ScheduleSpec, Shifts, TransformerConfig};

    fn transformer() -> TransformerConfig {
        TransformerConfig {
            hidden_size: 5376,
            num_attention_heads: 56,
            head_dim: 128,
            num_hidden_layers: 50,
            num_refiner_layers: 2,
            intermediate_size: 14336,
            video_channels: 24,
            audio_channels: 32,
            text_dim: 5120,
            frequency_dim: 256,
            time_hidden_dim: 5376,
            time_dim: 2688,
            rope_frequency_dim: 16,
            rope_theta: 10_000.0,
            norm_eps: 1e-5,
            qk_norm_eps: 1e-5,
            output_heads: 1,
        }
    }

    fn layouts() -> Layouts {
        let config = DenoiserConfig {
            transformer: transformer(),
            schedule: ScheduleSpec::uniform(
                50,
                Shifts {
                    video: 12.0,
                    audio: 3.0,
                },
            )
            .expect("the base checkpoint's schedule"),
            attention: AttentionKind::DENSE,
            tasks: vec![nn::capability::VideoTask::TextToVideoAudio],
            canvases: None,
            clip: crate::encoding::Clip {
                frames: 17,
                latents: 5,
                token_drop: 3,
            },
        };
        Layouts::new(
            config,
            Arrangement {
                text_rows: 128,
                condition_rows: 64,
                shard: 1,
                rank: 0,
            },
        )
        .unwrap()
    }

    #[test]
    fn the_traits_schedule_is_the_checkpoints_ladder_keyed_by_task() {
        let config = DenoiserConfig {
            transformer: transformer(),
            schedule: ScheduleSpec::uniform(
                4,
                Shifts {
                    video: 12.0,
                    audio: 3.0,
                },
            )
            .expect("a uniform grid"),
            attention: AttentionKind::DENSE,
            tasks: vec![nn::capability::VideoTask::TextToVideoAudio],
            canvases: None,
            clip: crate::encoding::Clip {
                frames: 17,
                latents: 5,
                token_drop: 3,
            },
        };
        let schedule = Schedule::new(&config.schedule).expect("the checkpoint's schedule");
        let count = schedule.num_steps().expect("an aligned grid");
        let steps = schedule_for(
            &config,
            &schedule,
            nn::capability::VideoTask::TextToVideoAudio,
        )
        .expect("the task this checkpoint serves");

        // The ladders are the schedule's own, walked once: one step per pair of sigma points, indexed
        // from zero, ending at the clean endpoint. A runner that rebuilt them from the spec would be
        // a second statement of a grid only the checkpoint owns.
        assert_eq!(steps.len() as u32, count);
        assert_eq!(steps[0].index, 0);
        assert_eq!(steps[0].count, count);
        assert_eq!(steps.last().expect("steps").sigma_next, 0.0);
        assert!(
            steps
                .windows(2)
                .all(|pair| pair[0].sigma_next == pair[1].sigma
                    && pair[1].index == pair[0].index + 1),
            "{steps:?}"
        );
        assert_eq!(steps, schedule.steps(Modality::Video).expect("a ladder"));

        // A task the checkpoint does not state is refused by name, and the refusal lists what it
        // does serve: every H3 task shares one grid, so the numbers alone cannot tell a wrong key.
        let error = schedule_for(
            &config,
            &schedule,
            nn::capability::VideoTask::ReferenceToVideoAudio,
        )
        .expect_err("a task this checkpoint does not serve");
        let message = error.to_string();
        assert!(message.contains("ReferenceToVideoAudio"), "{message}");
        assert!(message.contains("TextToVideoAudio"), "{message}");
    }

    fn request(frames: u32, width: u32, height: u32) -> VideoRequest<'static> {
        VideoRequest {
            task: nn::capability::VideoTask::TextToVideoAudio,
            width,
            height,
            frames,
            fps: 24,
            first_frames: 0,
            last_frames: 0,
            references: 0,
            text_tokens: 0,
            conditions: &[],
            vision_spans: &[],
            condition_rows: None,
        }
    }

    const TILE: u32 = 64;

    fn sparse_layouts() -> Layouts {
        let mut config = layouts().config;
        config.attention = AttentionKind::sparse(TILE, 0.8, None).unwrap();
        Layouts::new(config, layouts().arrangement).unwrap()
    }

    /// The ladder starts at one keyframe, doubles, gains a halfway rung from four keyframes up, and
    /// ends at the capacity whatever it is; a capacity below the least rung is the only rung.
    #[test]
    fn the_condition_ladder_climbs_to_the_capacity() {
        let ladder = |capacity| {
            let mut rules = sparse_layouts();
            rules.arrangement.condition_rows = capacity;
            rules.condition_ladder()
        };
        assert_eq!(
            ladder(49_152),
            [
                1024, 2048, 4096, 6144, 8192, 12_288, 16_384, 24_576, 32_768, 49_152
            ]
        );
        assert_eq!(
            ladder(40_960),
            [
                1024, 2048, 4096, 6144, 8192, 12_288, 16_384, 24_576, 32_768, 40_960
            ]
        );
        assert_eq!(ladder(2048), [1024, 2048]);
        assert_eq!(ladder(2112), [1024, 2048, 2112]);
        assert_eq!(ladder(512), [512]);
    }

    /// A layout holds the rung its request states, its rows are that rung's, and a request that
    /// states none holds the capacity. A request without conditions holds none at any rung.
    #[test]
    fn a_layout_holds_the_rung_of_its_request() {
        let mut rules = sparse_layouts();
        rules.arrangement.condition_rows = 2048;
        let widest = rules.cover(&conditioned());
        assert_eq!(widest.condition_rows, 2048);
        let rung = rules.cover(&VideoRequest {
            condition_rows: Some(1024),
            ..conditioned()
        });
        assert_eq!(rung.condition_rows, 1024);
        rules.segment_of(&rung).unwrap();
        assert_eq!(
            packed_rows(&rules.config, &rules.arrangement, &widest).unwrap()
                - packed_rows(&rules.config, &rules.arrangement, &rung).unwrap(),
            1024
        );
        let plain = rules.cover(&VideoRequest {
            task: nn::capability::VideoTask::TextToVideoAudio,
            condition_rows: Some(1024),
            ..conditioned()
        });
        assert_eq!(plain.condition_rows, 0);
    }

    #[test]
    fn a_conditions_draws_partition_the_condition_block() {
        // A runner sizes one buffer by `condition_elements` and slices each condition out of it by
        // `condition_offsets`, so the two have to agree on the same request and the entries have to
        // be contiguous in packed order. An audio-only reference is the case a prefix sum over
        // counts gets wrong: it draws nothing (`denoiser.py:452-472`), so its entry is zero-length
        // at the cursor its neighbours left, and a runner that advanced the cursor by its *rows*
        // would shift every later draw of the request and change the video silently.
        let shapes = [
            nn::capability::ConditionShape {
                role: nn::capability::ConditionRole::FirstFrame,
                frames: 1,
                height: 256,
                width: 256,
                audio_samples: 0,
            },
            nn::capability::ConditionShape {
                role: nn::capability::ConditionRole::Reference,
                frames: 25,
                height: 256,
                width: 256,
                audio_samples: 0,
            },
            nn::capability::ConditionShape {
                role: nn::capability::ConditionRole::Reference,
                frames: 0,
                height: 0,
                width: 0,
                audio_samples: 48_000,
            },
        ];
        let base = request(121, 256, 256);
        let request = VideoRequest {
            conditions: &shapes,
            references: 2,
            ..base
        };
        let rule = layouts();
        let offsets = rule
            .condition_offsets(&request)
            .expect("a request's offsets");
        assert_eq!(offsets.len(), shapes.len(), "one entry per condition");

        // Contiguous, in packed order, and starting at zero.
        let mut cursor = 0_u64;
        for (offset, elements) in &offsets {
            assert_eq!(*offset, cursor, "each piece starts where the last ended");
            cursor += elements;
        }
        assert_eq!(
            offsets[2].1, 0,
            "an audio reference draws no noise and takes no elements"
        );
        assert!(offsets[0].1 > 0 && offsets[1].1 > 0, "a visual piece draws");
        assert!(
            offsets[1].1 > offsets[0].1,
            "25 frames draw more than one still's frame"
        );

        // And the entries' total is the block the runner sizes: the two public answers agree.
        let conditions: Vec<packing::Condition> = shapes
            .iter()
            .map(|shape| packing::Condition {
                role: match shape.role {
                    nn::capability::ConditionRole::FirstFrame => packing::Role::FirstFrame,
                    nn::capability::ConditionRole::LastFrame => packing::Role::LastFrame,
                    nn::capability::ConditionRole::Reference => packing::Role::Reference,
                },
                frame: (shape.width > 0).then(|| packing::Frame {
                    canvas: Canvas {
                        short_edge: shape.width.min(shape.height),
                        aspect: (1, 1),
                        width: shape.width,
                        height: shape.height,
                    },
                    num_frames: shape.frames,
                }),
                audio_samples: shape.audio_samples,
            })
            .collect();
        let canvas = rule.canvas(&request).expect("a canvas");
        assert_eq!(
            cursor,
            u64::from(rule.condition_elements(&conditions, &canvas)),
            "the offsets partition the block `condition_elements` sizes"
        );
    }

    #[test]
    fn an_arena_plan_names_every_buffer_once_and_declares_a_known_element() {
        // The worker keys the arena's storage by these names and sizes it by the sum of the element
        // counts, so a repeated name aliases two buffers and a letter outside the plan vocabulary is
        // charged four bytes an element (`Buffer::width`'s fallback) - both silent at a canvas, both
        // wrong for the whole deployment. The group count is an input: the plan scales two entries by
        // it and nothing else.
        // One rank has no exchange to reserve for, two do: the plan follows the layer, which reserves
        // its exchange destinations only when it communicates.
        let two_ranks = Layouts::new(
            layouts().config,
            Arrangement {
                text_rows: 128,
                condition_rows: 64,
                shard: 2,
                rank: 0,
            },
        )
        .unwrap();
        for rule in [layouts(), sparse_layouts(), two_ranks] {
            let layout = rule.cover(&request(121, 1344, 768));
            let plan = denoise_arena(&rule.config, &rule.arrangement, 22, DType::BF16, &layout)
                .expect("a plan for the released canvas");
            let mut names = std::collections::HashSet::new();
            for buffer in &plan {
                assert!(
                    names.insert(buffer.name.as_str()),
                    "{} is listed twice",
                    buffer.name
                );
                assert!(buffer.elements > 0, "{} holds no elements", buffer.name);
                assert!(
                    ["bf16", "f16", "f32", "i32"].contains(&buffer.element),
                    "{} declares {}, which no plan element names",
                    buffer.name,
                    buffer.element
                );
            }
            // Every H3 TODO produces a soundtrack, so the stream's own buffers are in every plan: the
            // rows gathered from the packed sequence, and the head they are projected and solved with.
            for name in [
                "audio.hidden",
                "audio.groups",
                "audio.normed",
                "audio.velocity",
            ] {
                assert!(names.contains(name), "{name} is not in the plan");
            }
            // A layer that does not communicate reserves nothing for the exchange: at one rank these
            // entries must be absent, because they cost a row of every head each and are never read.
            for name in [
                "attention.staging",
                "attention.returned",
                "attention.merged",
                "attention.gathered",
            ] {
                assert!(
                    names.contains(name) == (rule.arrangement.shard > 1),
                    "{name} at shard {}",
                    rule.arrangement.shard
                );
            }
        }
    }

    #[test]
    fn a_layout_declares_the_refiners_boundaries_and_fills_them() {
        // The refiner's document is the prompt: its queries are every padded row and its keys stop
        // at the prompt's last row, so a padding row is never a key. Both kinds declare the pair -
        // a sparse deployment's attention reads tiles, but its refiner reads a document.
        let rule = layouts();
        let mut request = request(49, 640, 480);
        request.text_tokens = 12;
        let (layout, _) = rule.layout(&request).unwrap();
        let specs = rule.table_specs(&layout).unwrap();
        let names: Vec<&str> = specs.iter().map(|table| table.name).collect();
        for name in ["refiner_cu_seq_lens_q", "refiner_cu_seq_lens_k"] {
            assert!(names.contains(&name), "{name}");
            let table = specs.iter().find(|table| table.name == name).unwrap();
            assert_eq!(table.shape.dims(), &[2], "{name}");
        }
        let mut out: Vec<Vec<u8>> = specs
            .iter()
            .map(|table| {
                vec![0_u8; (table.shape.numel() as usize) * table.dtype.bytes().unwrap() as usize]
            })
            .collect();
        rule.write_state(&request, &layout, &mut out).unwrap();
        let values = |name: &str| -> Vec<i32> {
            let index = names.iter().position(|table| *table == name).unwrap();
            out[index]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| i32::from_le_bytes(*word))
                .collect()
        };
        assert_eq!(
            values("refiner_cu_seq_lens_q"),
            vec![0, layout.text_rows as i32]
        );
        assert_eq!(values("refiner_cu_seq_lens_k"), vec![0, 12]);
    }

    #[test]
    fn a_sparse_layout_declares_the_refiners_boundaries_too() {
        let rule = sparse_layouts();
        let mut request = request(121, 1344, 768);
        request.text_tokens = 12;
        let (layout, _) = rule.layout(&request).unwrap();
        let specs = rule.table_specs(&layout).unwrap();
        let names: Vec<&str> = specs.iter().map(|table| table.name).collect();
        // The four tile tables, and neither of the dense attention's sequence-length tables.
        for name in [
            "variable_block_sizes",
            "tile_segments",
            "segment_starts",
            "segment_keep",
        ] {
            assert!(names.contains(&name), "{name}");
        }
        assert!(!names.contains(&"cu_seq_lens_k"));
        assert!(!names.contains(&"cu_seq_lens_q"));
        let tile = u64::from(TILE);
        let rows = specs
            .iter()
            .find(|table| table.name == "groups")
            .map(|table| table.shape.dim(0))
            .expect("the group table");
        assert_eq!(rows % tile, 0);
        for name in ["variable_block_sizes", "segment_keep", "tile_segments"] {
            let table = specs.iter().find(|table| table.name == name).unwrap();
            assert_eq!(table.shape.dim(0), rows / tile, "{name}");
        }
        // The tables fill, and every block size is at most one tile's rows.
        let mut out: Vec<Vec<u8>> = specs
            .iter()
            .map(|table| {
                vec![0_u8; (table.shape.numel() as usize) * table.dtype.bytes().unwrap() as usize]
            })
            .collect();
        rule.write_state(&request, &layout, &mut out).unwrap();
        let sizes = specs
            .iter()
            .position(|table| table.name == "variable_block_sizes")
            .unwrap();
        let block_sizes = decode_i32(&out[sizes]);
        assert_eq!(block_sizes.len() as u64, rows / tile);
        assert!(
            block_sizes
                .iter()
                .all(|size| 0 <= *size && *size as u64 <= tile)
        );
        assert!(block_sizes.iter().any(|size| *size > 0));
    }

    #[test]
    fn a_sparse_layout_rebuilds_its_segment_packing() {
        let rule = sparse_layouts();
        // The dense layout carries the capacities a request names; the segment packing rounds each
        // prefix capacity up to its tile and pads the rows to whole tiles.
        let (layout, _) = rule.layout(&request(121, 1344, 768)).unwrap();
        let packed = rule.segment_of(&layout).unwrap();
        assert_eq!(packed.extent.tile, 64);
        assert_eq!(packed.extent.text_rows, 128);
        assert_eq!(packed.extent.condition_rows, 0);
        assert_eq!(packed.extent.padded_tokens % 64, 0);
        assert_eq!(
            packed.video_variable_block_sizes.len() as u32,
            packed.extent.video_tiles
        );
        // A dense deployment has no segment packing to give, and says so rather than guessing.
        assert!(layouts().segment_of(&layout).is_err());
    }

    #[test]
    fn a_layout_built_densely_is_not_a_segment_packing_a_sparse_deployment_builds() {
        // The same request read by the two deployments: only the sparse one pads to whole tiles, and
        // a layout whose raster does not match its frame count is refused by both.
        let (layout, dense) = layouts().layout(&request(121, 1344, 768)).unwrap();
        let packed = sparse_layouts().segment_of(&layout).unwrap();
        assert!(packed.extent.padded_tokens > dense.padded_tokens);
        let mut broken = layout;
        broken.video.frames = 24;
        assert!(sparse_layouts().segment_of(&broken).is_err());
    }

    #[test]
    fn the_tile_bound_covers_every_layout_the_deployment_serves() {
        // The compression's score matrix has one column per video tile, so the bound has to cover
        // every layout the deployment can be asked for - not only the longest frame count, because a
        // canvas with a taller grid can hold more tiles at a shorter one.
        let rules = sparse_layouts();
        let bound = rules
            .tiles_bound()
            .expect("a sparse deployment bounds its tiles");
        let tile = rules.config.attention.capacity_rows();
        let mut largest = 0;
        for canvas in rules.config.offered_canvases() {
            let mut frames = crate::CLIP_FRAMES + crate::CLIP_LATENTS;
            while frames <= MAX_FRAMES {
                if let Ok(segment) = packing::segment_packing(
                    frames,
                    canvas,
                    rules.arrangement.text_rows,
                    rules.arrangement.condition_rows,
                    tile,
                    tile,
                ) {
                    let tiles = segment.extent.video_tiles;
                    assert!(
                        tiles <= bound,
                        "{tiles} tiles at {}x{} and {frames} frames exceed the bound {bound}",
                        canvas.width,
                        canvas.height
                    );
                    largest = largest.max(tiles);
                }
                frames += crate::CLIP_FRAMES;
            }
        }
        // And the bound is reached rather than merely safe: a number larger than any layout's spends
        // memory on nothing, and one smaller refuses on the device instead of counting.
        assert_eq!(largest, bound);
    }

    #[test]
    fn a_dense_deployment_has_no_tile_bound_to_ask_for() {
        let error = layouts()
            .tiles_bound()
            .expect_err("a dense deployment has no tiles");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("dense")),
            "{error:?}"
        );
    }

    #[test]
    fn a_dense_deployment_bounds_its_rows_by_its_own_packing() {
        let dense = layouts().rows_bound().unwrap();
        // The same deployment read as sparse pads every layout to whole tiles, which is larger for
        // every canvas; a dense checkpoint never captures one of those, so its bound must be below.
        let mut sparse = layouts();
        sparse.config.attention = AttentionKind::sparse(64, 0.8, None).unwrap();
        assert!(dense < sparse.rows_bound().unwrap(), "{dense}");
        // And it still bounds the largest dense layout the deployment offers.
        let (_, packing) = layouts().layout(&request(107, 1344, 768)).unwrap();
        assert!(dense >= packing.padded_tokens);
    }

    #[test]
    fn a_conditions_rows_start_after_the_text_the_layout_reserved() {
        // A `ref2va`-shaped request at the released layout: 128 text rows, 192 condition rows and
        // the zero row. An image takes one 64-row unit, an audio reference 37 rows, and the request
        // does not fill the capacity - which is the case the subtraction convention got wrong, so
        // the ranges are asserted against the layout's numbers and not against the pieces' sum.
        let text_rows = 128u64;
        let condition_rows = 192u32;
        let pieces = [(64u32, 0u32), (0, 37)];
        let total = text_rows + u64::from(condition_rows) + 1;
        let (capacity, ranges) = super::condition_ranges(total, condition_rows, &pieces).unwrap();
        assert_eq!(capacity, text_rows);
        // Each block starts where the previous one ended, and inside a block the audio rows lead.
        let mut expected = text_rows;
        for ((visual, audio), (start, end)) in pieces.iter().zip(&ranges) {
            assert_eq!(*start, expected);
            assert_eq!(*end - *start, u64::from(*visual) + u64::from(*audio));
            expected = *end;
        }
        assert!(
            expected < text_rows + u64::from(condition_rows),
            "the request leaves {} of the layout's {} condition rows unused, and the zero row is \
             still the source's last",
            text_rows + u64::from(condition_rows) - expected,
            condition_rows
        );

        // A request whose pieces do not fit the layout's capacity is refused rather than written
        // past the rows the packing reserved.
        assert!(super::condition_ranges(total, condition_rows, &[(128, 128)]).is_err());
        // And a source that cannot hold the capacity is refused too.
        assert!(super::condition_ranges(text_rows, condition_rows, &pieces).is_err());
    }

    #[test]
    fn a_still_condition_draws_its_latent_elements_and_nothing_more() {
        let conditions = [packing::Condition {
            role: packing::Role::FirstFrame,
            frame: Some(packing::Frame {
                canvas: crate::config::Canvas {
                    short_edge: 768,
                    aspect: (16, 9),
                    width: 1344,
                    height: 768,
                },
                num_frames: 1,
            }),
            audio_samples: 0,
        }];
        let canvas = crate::config::Canvas {
            short_edge: 768,
            aspect: (16, 9),
            width: 1344,
            height: 768,
        };
        // One latent frame of the canvas' raster, 24 channels wide: the video's stream starts here.
        assert_eq!(
            layouts().condition_elements(&conditions, &canvas),
            48 * 84 * 24
        );
    }

    #[test]
    fn a_reference_clip_draws_the_latent_frames_the_encoder_cuts_it_into() {
        // A reference clip that is not a whole number of the generated timeline's clips is still a
        // condition: the encoder cuts it into 17-frame clips and drops the last one's tail, so 30
        // frames are 7 latent frames. Counting it with the generated rule would give 0 - 30 is not
        // `17n + 5` - and every later draw of the request would move by the whole condition.
        let canvas = crate::config::Canvas {
            short_edge: 768,
            aspect: (16, 9),
            width: 1344,
            height: 768,
        };
        let conditions = [packing::Condition {
            role: packing::Role::Reference,
            frame: Some(packing::Frame {
                canvas,
                num_frames: 30,
            }),
            audio_samples: 0,
        }];
        let rule = layouts();
        let raster = packing::latent_raster(&canvas).expect("the canvas' raster");
        let latents = rule
            .config
            .clip
            .latent_frames(30)
            .expect("a clip of 30 frames");
        assert!(latents > 1, "the condition is more than a still");
        assert_eq!(
            rule.condition_elements(&conditions, &canvas),
            latents * raster.height * raster.width * rule.config.transformer.video_channels
        );
    }

    #[test]
    fn a_request_snaps_to_a_legal_frame_count_and_raster() {
        let rule = layouts();
        let (layout, packing) = rule.layout(&request(100, 1344, 768)).unwrap();
        // 100 frames snap to 107 (6 clips of 17 plus 5), which is 32 latent frames and 32 * 42 * 24
        // video rows.
        assert_eq!(packing.video_rows, 32 * 42 * 24);
        assert_eq!(layout.video.frames, 32);
        assert_eq!((layout.video.height, layout.video.width), (48, 84));
        // Two rows of 32 channels per audio latent step, and 107 frames at 24 fps are 178 steps.
        assert_eq!(packing.audio_rows, 2 * 178);
        assert_eq!(layout.audio.unwrap().frames, 178);
        assert_eq!(layout.text_rows, 128);
        // A request without conditions carries no condition rows through the stack, and a request
        // with conditions takes the deployment's capacity.
        assert_eq!(layout.condition_rows, 0);
        let (conditioned_layout, _) = rule.layout(&conditioned()).unwrap();
        assert_eq!(conditioned_layout.condition_rows, 64);
    }

    /// A request carrying one keyframe, one clip reference with a soundtrack, and one audio
    /// reference: the three ways a condition can arrive.
    fn conditioned() -> VideoRequest<'static> {
        const SHAPES: [nn::capability::ConditionShape; 3] = [
            nn::capability::ConditionShape {
                role: ConditionRole::FirstFrame,
                frames: 1,
                height: 480,
                width: 832,
                audio_samples: 0,
            },
            nn::capability::ConditionShape {
                role: ConditionRole::Reference,
                frames: 22,
                height: 480,
                width: 832,
                audio_samples: 24_000,
            },
            nn::capability::ConditionShape {
                role: ConditionRole::Reference,
                frames: 0,
                height: 0,
                width: 0,
                audio_samples: 24_000,
            },
        ];
        VideoRequest {
            task: nn::capability::VideoTask::ReferenceToVideoAudio,
            first_frames: 1,
            references: 2,
            text_tokens: 15,
            conditions: &SHAPES,
            ..request(121, 832, 480)
        }
    }

    /// A condition's pieces follow the packing, not the request: a keyframe leads, and each draw
    /// starts where the visual condition packed before it ended. An audio reference takes a place in
    /// the list and none in the stream, which is what keeps every later draw where the reference's
    /// is.
    #[test]
    fn the_pieces_are_in_packed_order_and_their_draws_partition_the_stream() {
        const SHAPES: [nn::capability::ConditionShape; 3] = [
            nn::capability::ConditionShape {
                role: ConditionRole::Reference,
                frames: 0,
                height: 0,
                width: 0,
                audio_samples: 1_000,
            },
            nn::capability::ConditionShape {
                role: ConditionRole::Reference,
                frames: 1,
                height: 64,
                width: 128,
                audio_samples: 0,
            },
            nn::capability::ConditionShape {
                role: ConditionRole::FirstFrame,
                frames: 1,
                height: 480,
                width: 832,
                audio_samples: 0,
            },
        ];
        let request = VideoRequest {
            task: nn::capability::VideoTask::ReferenceToVideoAudio,
            first_frames: 1,
            references: 2,
            text_tokens: 15,
            conditions: &SHAPES,
            ..request(121, 832, 480)
        };
        let rule = layouts();
        let pieces = pieces_of(&rule.config, Some(800), &request).expect("the conditions pack");
        assert_eq!(
            pieces
                .iter()
                .map(|piece| piece.condition)
                .collect::<Vec<_>>(),
            [2, 0, 1],
            "the keyframe leads, and the references keep their request order"
        );
        let keyframe = u64::from(rule.config.transformer.video_channels) * 30 * 52;
        assert_eq!(pieces[0].draw_offset, 0);
        assert_eq!(pieces[0].pixels, Some([1, 3, 1, 480, 832]));
        assert!(pieces[1].draw.is_none(), "an audio reference draws nothing");
        assert_eq!(pieces[1].draw_offset, keyframe);
        assert_eq!(
            pieces[1].track,
            Some([2, 1, 1_600]),
            "a track is padded to whole latent frames"
        );
        assert_eq!(pieces[1].audio_rows, 4);
        assert_eq!(pieces[2].draw_offset, keyframe);
        assert_eq!(pieces[2].rows, 2 * 4);

        // A track with no encoder to read it is refused, not conditioned on nothing.
        assert!(pieces_of(&rule.config, None, &request).is_err());
    }

    /// A request names a vision block by its first row and its rows, as a run of the prompt's tags
    /// is; the packing marks `[start, stop)`. A span that reached the packing unchanged would read a
    /// row count as a stop, and the block would take the text tag past its first rows.
    #[test]
    fn a_vision_span_reaches_the_packing_as_a_half_open_range() {
        let spans = [(3, 4), (10, 1)];
        let request = VideoRequest {
            vision_spans: &spans,
            ..request(121, 832, 480)
        };
        assert_eq!(vision_spans(&request), [(3, 7), (10, 11)]);
    }

    #[test]
    fn the_layout_declares_every_element_the_audio_branch_reads() {
        // The worker sizes the audio latent and its draw from `channels * frames`, so that product
        // is what the denoiser must read as rows. The latent is the stereo pair of channel planes:
        // `AUDIO_CHANNELS * audio_channels * steps` elements, read as `AUDIO_CHANNELS * steps` rows
        // of `audio_channels`. Declaring `audio_channels` alone would state half of them.
        let rule = layouts();
        let (layout, _) = rule.layout(&request(121, 1344, 768)).unwrap();
        let audio = layout
            .audio
            .expect("a request that produces audio carries its latent");
        let channels = rule.config.transformer.audio_channels;
        let steps = audio.frames;
        assert_eq!(
            u64::from(audio.channels) * u64::from(audio.frames),
            u64::from(crate::config::AUDIO_CHANNELS) * u64::from(channels) * u64::from(steps)
        );
    }

    #[test]
    fn both_layout_constructors_declare_one_audio_latent() {
        // Two constructors build a layout - the capability's infallible `cover` and the packing's
        // `layout` - and a runner reads whichever its stage was given. They must state one audio
        // stream: the half-buffer got in because they stated the decomposition separately, and the
        // product is what sizes the latent and its draw.
        let rule = layouts();
        let request = conditioned();
        let covered = rule.cover(&request);
        let (laid, _) = rule
            .layout(&request)
            .expect("the conditioned request packs");
        assert_eq!(covered.audio, laid.audio);
        let audio = laid
            .audio
            .expect("a request that produces audio carries its latent");
        let steps = audio.frames;
        assert_eq!(
            u64::from(audio.channels) * u64::from(steps),
            u64::from(crate::config::AUDIO_CHANNELS)
                * u64::from(rule.config.transformer.audio_channels)
                * u64::from(steps),
            "the stereo pair of channel planes is the latent's element count"
        );
    }

    #[test]
    fn every_rank_of_a_shard_draws_the_one_stream_a_single_rank_draws() {
        // One stream per request, addressed by element: the conditions first, then the video's
        // rows, then the audio's. Every rank holds the whole sample, so one pair of offsets serves
        // the group and a rank's trajectory starts from the numbers one rank's does.
        let rule = layouts();
        let (layout, _) = rule.layout(&request(121, 1344, 768)).unwrap();
        let row_elements = u64::from(layout.video.channels) * 4;
        let video_rows = u64::from(layout.video.frames)
            * u64::from(layout.video.height / 2)
            * u64::from(layout.video.width / 2);
        let condition = u64::from(layout.condition_elements);
        let (video, audio) = draw_offsets(layout);
        assert_eq!(video, condition);
        assert_eq!(audio, condition + video_rows * row_elements);
    }

    #[test]
    fn a_ranks_owned_rows_partition_each_modality() {
        // Every row of a modality lands in exactly one rank's interval, at its place in that
        // interval, and one rank owns everything with the identity for sources.
        let mut rule = layouts();
        let packed: Vec<i32> = (40..100).collect();
        let rows = 128;
        rule.arrangement.shard = 1;
        rule.arrangement.rank = 0;
        let (sources, slots) = rule.owned_rows(rows, &packed);
        assert_eq!(sources, (0..60).collect::<Vec<i32>>());
        assert_eq!(slots, packed);
        rule.arrangement.shard = 4;
        let mut seen = Vec::new();
        for rank in 0..4 {
            rule.arrangement.rank = rank;
            let (sources, slots) = rule.owned_rows(rows, &packed);
            assert_eq!(sources.len(), slots.len());
            for (source, slot) in sources.iter().zip(&slots) {
                assert_eq!(packed[*source as usize], rank as i32 * 32 + slot);
                seen.push(*source);
            }
        }
        seen.sort_unstable();
        assert_eq!(seen, (0..60).collect::<Vec<i32>>());
    }

    #[test]
    fn a_condition_arrives_as_the_packing_reads_it() {
        // The request states a role and an encoded raster; the packing wants a canvas and a frame
        // count. A reference without pixels is an audio reference and has no frame.
        let conditions = conditions_of(&conditioned());
        assert_eq!(conditions.len(), 3);
        assert_eq!(conditions[0].role, packing::Role::FirstFrame);
        let frame = conditions[0].frame.expect("a keyframe carries its pixels");
        assert_eq!((frame.canvas.width, frame.canvas.height), (832, 480));
        // A canvas is the short edge and the exact aspect, as `canvas_of` reduces one.
        assert_eq!(frame.canvas.short_edge, 480);
        assert_eq!(frame.canvas.aspect, (26, 15));
        assert_eq!(frame.num_frames, 1);
        assert_eq!(conditions[1].role, packing::Role::Reference);
        // A reference clip's frame count is a *generated* count, `17n + 5`: the fixture asks for 22,
        // the smallest one, and the packing reads it as the clip's five latent frames. 12 is not a
        // count H3 generates, and a fixture stating one would test a stream the model cannot serve.
        let clip = conditions[1].frame.expect("a clip has pixels").num_frames;
        assert_eq!(clip, 22);
        assert_eq!(clip, packing::align_num_frames(clip));
        assert_eq!(conditions[1].audio_samples, 24_000);
        assert!(
            conditions[2].frame.is_none(),
            "an audio reference has no pixels"
        );
        assert_eq!(conditions[2].audio_samples, 24_000);
    }

    #[test]
    fn an_audio_reference_draws_no_noise() {
        // The request's stream holds one draw per *visual* condition: the reference conditions an
        // audio reference at its posterior's mean, and counting its rows would shift the video's
        // draw and change the video without failing anything (`denoiser.py:452-461`).
        let rule = layouts();
        let conditions = conditions_of(&conditioned());
        let channels = u64::from(rule.config.transformer.video_channels);
        let visual: u64 = conditions
            .iter()
            .filter_map(|condition| condition.frame)
            .map(|frame| {
                let raster = packing::latent_raster(&frame.canvas).unwrap();
                let frames = if frame.num_frames <= 1 {
                    1
                } else {
                    packing::video_latent_frames(frame.num_frames).unwrap()
                };
                u64::from(frames) * u64::from(raster.height) * u64::from(raster.width) * channels
            })
            .sum();
        assert_eq!(
            u64::from(rule.condition_elements(&conditions, &canvas_of_raster(832, 480).unwrap())),
            visual
        );
    }

    #[test]
    fn an_audio_reference_is_channel_major_rows() {
        // The packed audio latent is [2 * frames, audio_channels] - channel-major - because that is
        // what the encoder writes and what the input projection reads. A [frames, channels] reshape
        // would halve the row count and shift every row after the reference, with no shape
        // complaint downstream: the projection reads whatever row count it is given.
        // A 22-frame reference carries a real track: `audio_samples` is the number of samples the
        // audio encoder produced, and the packed rows follow *that*, `samples / AUDIO_HOP` rounded
        // up - not the video's frame rate. Passing 22 latent steps' worth of samples (`22 * 800`)
        // would give 44 rows and still look consistent, which is the trap this pins.
        let shapes = [nn::capability::ConditionShape {
            role: ConditionRole::Reference,
            frames: 0,
            height: 0,
            width: 0,
            audio_samples: crate::packing::audio_samples(22),
        }];
        let shapes = &shapes;
        let request = VideoRequest {
            task: nn::capability::VideoTask::ReferenceToVideoAudio,
            references: 1,
            conditions: shapes,
            ..request(121, 832, 480)
        };
        let conditions = conditions_of(&request);
        let canvas = canvas_of_raster(832, 480).unwrap();
        let segments =
            packing::condition_segments(&conditions, &canvas, layouts().config.clip).unwrap();
        let audio: Vec<_> = segments
            .iter()
            .filter(|segment| segment.audio_rows() > 0)
            .collect();
        assert_eq!(audio.len(), 1);
        let frames = crate::packing::audio_latent_frames(22);
        // One row per channel per latent step: `AUDIO_CHANNELS * frames`, the encoder's layout.
        assert_eq!(
            audio[0].audio_rows(),
            crate::config::AUDIO_CHANNELS * frames
        );
    }

    #[test]
    fn each_condition_s_latents_are_paired_with_it_in_request_order() {
        // A condition contributes its visual latents when it has pixels and its audio latents when
        // it has samples, in that order; the packed order is another thing, so a view handed to the
        // wrong condition would project one condition's media into another's rows.
        let conditions = conditions_of(&conditioned());
        let views: Vec<TensorView> = (0..5)
            .map(|index| {
                TensorView::with_strides(
                    nn::DevicePtr::new(index * 4096),
                    DType::BF16,
                    Shape::new(&[1, 1]),
                    nn::Strides::new(&[1, 1]),
                )
            })
            .collect();
        // Four latents: the keyframe's rows, the clip's rows and its track, the audio reference's
        // track - `sum((video is not None) + (audio_samples > 0))`, the reference's own count.
        let paired = ConditionViews::new(&conditions, &views[..4]).unwrap();
        assert!(paired.of(0).0.is_some() && paired.of(0).1.is_none());
        assert!(paired.of(1).0.is_some() && paired.of(1).1.is_some());
        assert!(paired.of(2).0.is_none() && paired.of(2).1.is_some());
        // Every expected view is required, and one too many is refused rather than dropped.
        assert!(ConditionViews::new(&conditions, &views[..3]).is_err());
        assert!(ConditionViews::new(&conditions, &views).is_err());
    }

    #[test]
    fn the_tables_are_the_layout_s_and_the_step_is_the_worker_s_to_write() {
        let rule = layouts();
        // A prompt of twelve rows, the configuration `write_state` is exercised with below: every
        // write index then points into the text rows rather than at the zero row, so a table the
        // model fills is filled with something other than the sentinel this test offers.
        let mut request = request(121, 1344, 768);
        request.text_tokens = 12;
        let (layout, packing) = rule.layout(&request).unwrap();
        let tables = rule.table_specs(&layout).unwrap();
        let names: Vec<_> = tables.iter().map(|table| table.name).collect();
        // The worker addresses a table by name, so a name may appear once and no position is
        // promised: the tables are sorted by name, and any positional claim about them goes stale
        // the moment a table is added. The set itself is the code's own list, which
        // `each_kind_resolves_its_state_tables_by_its_own_names` pins against `table_specs`.
        let unique: std::collections::BTreeSet<_> = names.iter().collect();
        assert_eq!(unique.len(), names.len(), "a name repeats: {names:?}");
        // The step is the one table the model never fills: the worker writes its single index
        // before each replay, so `write_state` must leave its bytes as the caller offered them.
        const OFFERED: u8 = 0xAB;
        let mut state: Vec<Vec<u8>> = tables
            .iter()
            .map(|table| {
                vec![
                    OFFERED;
                    (table.shape.numel() as usize) * table.dtype.bytes().unwrap() as usize
                ]
            })
            .collect();
        rule.write_state(&request, &layout, &mut state)
            .expect("a dense layout's state is writable");
        for (table, bytes) in tables.iter().zip(&state) {
            if table.name == "step" {
                assert!(
                    bytes.iter().all(|byte| *byte == OFFERED),
                    "the worker writes the step index; the model must not"
                );
            } else {
                assert!(
                    bytes.iter().any(|byte| *byte != OFFERED),
                    "{} was offered to the model and left unwritten",
                    table.name
                );
            }
        }
        // Every table but the step covers one row per packed row.
        let rows = u64::from(packing.padded_tokens);
        for table in &tables {
            match table.name {
                "step" => {
                    assert_eq!(table.shape.dims(), &[1]);
                }
                // The audio table is addressed by audio row and names the packed row it occupies,
                // so its length is the audio rows rather than every packed row.
                "audio_indices" | "audio_slots" | "audio_sources" => {
                    assert_eq!(table.shape.dim(0), u64::from(packing.audio_rows));
                }
                // A run's table names both of its ends, so it carries one more entry than the run
                // has row counts: the pack's used rows, or the prompt's tokens for the refiner's.
                "cu_seq_lens_k"
                | "cu_seq_lens_q"
                | "refiner_cu_seq_lens_k"
                | "refiner_cu_seq_lens_q" => assert_eq!(table.shape.dims(), &[2]),
                // The rotation reads the coordinates axis-major, so the axes lead.
                "position_ids" => assert_eq!(table.shape.dims(), &[3, rows]),
                // The video indices name the packed row of every video row, and video rows are
                // patched: four voxels to a row.
                "video_indices" | "video_slots" | "video_sources" => {
                    assert_eq!(table.shape.dim(0), u64::from(packing.video_rows));
                }
                _ => assert_eq!(table.shape.dim(0), rows, "{}", table.name),
            }
        }
        // Axis-major: the rotation reads `[axis, token]`, so the axes are the leading extent.
        let positions = tables
            .iter()
            .find(|table| table.name == "position_ids")
            .expect("the rotation's coordinates are a table");
        assert_eq!(positions.shape.dims(), &[3, rows]);
    }

    #[test]
    fn the_state_bytes_fill_the_tables_the_layout_names() {
        let rule = layouts();
        let mut request = request(121, 1344, 768);
        request.text_tokens = 12;
        let (layout, _) = rule.layout(&request).unwrap();
        let tables = rule.table_specs(&layout).unwrap();
        let mut out: Vec<Vec<u8>> = tables
            .iter()
            .map(|table| {
                vec![0_u8; (table.shape.numel() as usize) * table.dtype.bytes().unwrap() as usize]
            })
            .collect();
        rule.write_state(&request, &layout, &mut out).unwrap();
        // A prompt of 12 rows is text, so every write index points into the text rows and every
        // generated row takes the prefix's zero row.
        let at = |name: &str| {
            tables
                .iter()
                .position(|table| table.name == name)
                .expect("the layout names the table")
        };
        let prefix: Vec<i32> = decode_i32(&out[at("prefix_index")]);
        assert_eq!(
            prefix.len(),
            tables[at("prefix_index")].shape.dim(0) as usize
        );
        // The prefix segment is packed after the generated rows, so the prompt's twelve rows sit at
        // `prefix_start()` and each points at its own source row.
        let first = rule.packing_of(&layout).unwrap().prefix_start() as usize;
        assert!(
            prefix[first..first + 12]
                .iter()
                .enumerate()
                .all(|(row, column)| *column == row as i32)
        );
        // The first packed row is a generated video row, so it takes the video timestep.
        let indices = decode_i32(&out[at("modulation_indices")]);
        assert_eq!(indices[0], packing::Group::Video as i32);
    }

    #[test]
    fn a_raster_off_the_grid_snaps_up_to_it() {
        let rule = layouts();
        // 1340x770 is not a multiple of 32: the request renders the next legal raster up, so two
        // requests either side of a multiple share one layout rather than each carrying its own.
        let (layout, _) = rule.layout(&request(121, 1340, 770)).unwrap();
        assert_eq!((layout.video.height, layout.video.width), (50, 84));
        assert_eq!(layout.video.height * 16, 800);
        assert_eq!(layout.video.width * 16, 1344);
    }

    #[test]
    fn a_layout_the_model_did_not_build_is_refused() {
        let rule = layouts();
        let (mut layout, _) = rule.layout(&request(121, 1344, 768)).unwrap();
        // 24 latent frames is not a frame count a request snaps to: no rounding produces 5n + 2 = 24.
        layout.video.frames = 24;
        assert!(rule.packing_of(&layout).is_err());
        // A prefix that holds no text row is not a layout this deployment packs either.
        let (mut layout, _) = rule.layout(&request(121, 1344, 768)).unwrap();
        layout.text_rows = 0;
        assert!(rule.packing_of(&layout).is_err());
    }

    #[test]
    fn the_row_bound_covers_both_packings() {
        let rule = layouts();
        let bound = rule.rows_bound().unwrap();
        // The bound is at least any layout the deployment can be asked for, and the sparse packing's
        // whole-tile padding makes it strictly larger than the dense one at the same request.
        let (dense, _) = rule.layout(&request(121, 1344, 768)).unwrap();
        let dense_packing = rule.packing_of(&dense).unwrap();
        assert!(bound >= dense_packing.padded_tokens);
        let segment = packing::segment_packing(
            124,
            &Canvas {
                short_edge: 768,
                aspect: (7, 4),
                width: 1344,
                height: 768,
            },
            rule.arrangement().text_rows,
            rule.arrangement().condition_rows,
            64,
            64,
        )
        .unwrap();
        assert!(segment.extent.padded_tokens > dense_packing.padded_tokens);
    }

    #[test]
    fn the_layout_s_condition_count_and_the_stream_offsets_agree() {
        // Two paths compute the same number: the layout's declared `condition_elements`, which the
        // draw addresses from, and `noise_offsets`, which counts the conditions' own draws. They
        // must never drift, because a drift is a stream that starts the video too early or too late
        // with nothing to raise.
        let rule = layouts();
        let request = request(121, 1344, 768);
        let (layout, packing) = rule.layout(&request).unwrap();
        let canvas = canvas_of(&request).unwrap();
        let conditions = conditions_of(&request);
        let declared = rule.condition_elements(&conditions, &canvas);
        assert_eq!(layout.condition_elements, declared);
        let segments = packing::condition_segments(&conditions, &canvas, rule.config.clip).unwrap();
        assert_eq!(segments.len(), conditions.len());
        assert!(packing.padded_tokens > packing.video_rows + packing.audio_rows);
        // With no conditions the video's stream starts at zero.
        assert_eq!(declared, 0);
    }

    #[test]
    fn the_noise_stream_counts_elements_not_rows() {
        let rule = layouts();
        let request = request(121, 1344, 768);
        let (layout, _) = rule.layout(&request).unwrap();
        let bare = rule.noise_offsets(&layout, &[]).unwrap();
        assert_eq!(bare.condition, 0);
        assert_eq!(bare.video, 0);
        // 121 frames snap to 124, which is 37 latent frames of 48x84 at 24 channels.
        assert_eq!(layout.video.frames, 37);
        assert_eq!(bare.audio, 37 * 48 * 84 * 24);
        // A 22-frame reference is one clip, which the VAE encodes to seven latent frames of the
        // 48x84 raster; its rows are patched, and its element count weights each modality by its own
        // channel count.
        let conditions = [packing::Condition {
            role: packing::Role::Reference,
            frame: Some(packing::Frame {
                canvas: Canvas {
                    short_edge: 768,
                    aspect: (7, 4),
                    width: 1344,
                    height: 768,
                },
                num_frames: 22,
            }),
            audio_samples: 0,
        }];
        let one = rule.noise_offsets(&layout, &conditions).unwrap();
        // A condition's noise is its latent volume times the channels, not its packed rows: seven
        // latent frames of 48x84 at 24 channels, and no audio.
        let expected = 7 * 48 * 84 * 24;
        assert_eq!(one.condition, expected);
        assert_eq!(one.video, one.condition);
        assert_eq!(one.audio, one.video + bare.audio);
    }

    #[test]
    fn a_prompt_wider_than_the_capacity_is_refused() {
        let rule = layouts();
        let request = request(121, 1344, 768);
        let (layout, _) = rule.layout(&request).unwrap();
        let tables = rule.table_specs(&layout).unwrap();
        let mut out: Vec<Vec<u8>> = tables
            .iter()
            .map(|table| {
                vec![0_u8; table.shape.numel() as usize * table.dtype.bytes().unwrap() as usize]
            })
            .collect();
        let refused = rule.write_state(&request, &layout, &mut out);
        assert!(refused.is_err());
    }

    fn decode_i32(bytes: &[u8]) -> Vec<i32> {
        let (chunks, _) = bytes.as_chunks::<4>();
        chunks
            .iter()
            .map(|chunk| i32::from_le_bytes(*chunk))
            .collect()
    }
    #[test]
    fn each_kind_resolves_its_state_tables_by_its_own_names() {
        // The names are positional: the worker passes its views in `table_specs`' order, so a set
        // that is not exactly that order reads another tensor without any shape complaining.
        let dense = layouts();
        let layout = dense.cover(&request(22, 1344, 768));
        let names: Vec<&str> = dense
            .table_specs(&layout)
            .unwrap()
            .iter()
            .map(|table| table.name)
            .collect();
        assert_eq!(names, DENSE_TABLES);

        let sparse = sparse_layouts();
        let layout = sparse.cover(&request(22, 1344, 768));
        let names: Vec<&str> = sparse
            .table_specs(&layout)
            .unwrap()
            .iter()
            .map(|table| table.name)
            .collect();
        assert_eq!(names, SEGMENT_TABLES);
    }
}
