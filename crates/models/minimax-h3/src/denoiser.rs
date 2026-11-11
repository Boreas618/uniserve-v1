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
//! ([`crate::config::FRAMES`]), sides to the checkpoint's 32-pixel grid, and both prefix
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
//! zero row last - and the packing's `prefix_index` addresses it directly
//! ([`nn::capability::PrefixSource`]). The runner builds it, because the encoders produce
//! the pieces and the packing decides the order; the DiT reads no other conditioning view, so a
//! condition the source does not carry cannot reach the model.
//!
//! Its rows are filled in two places that write disjoint rows: `Denoiser::prepare_conditioning`
//! projects each condition's own rows into the condition rows once per request, and every step
//! refines the prompt's features into the text rows ([`crate::conditioning::TokenRefiner`]), where
//! the prompt's length is known.
//!
//! ## Host values
//!
//! Everything a request changes - the rotary coordinates, the timestep groups, the prefix index of
//! every packed row, the AdaLN row each row reads, the rows each rank owns and the step index - is
//! computed on the host inside the step and placed on the device with `Forward::upload`
//! ([`Layouts::packed`]). Their extents follow from the layout alone, so a graph captured for a
//! layout replays every request of it, its uploads rewritten before each replay.

use nn::blocks::{ModulationTable, RowModulation};
use nn::capability::{ConditionRole, VideoRequest};
use nn::diffusion::{TimestepTable, TimestepTableSpec};
use nn::family::{SequenceSplit, Split};
use nn::{DType, Forward, Operand, Shape, WeightLoader};
use proto::video::{LatentShape, Layout};

use crate::attention::Visibility;
use crate::config::{CANVAS_RULE, DenoiserConfig, FRAMES, MAX_FRAMES};
use crate::modulation::Projections;
use crate::packing::{self, DensePacking};
use crate::schedule::{Modality, Schedule};
use crate::solver::Solver;
use crate::transformer::Transformer;
use crate::{Error, Result};
use proto::video::Canvas;

/// The modulation columns one AdaLN row holds: six affine vectors of three tags each.
const ADALN_COLUMNS: u64 = 18;

/// The least rung of a deployment's condition ladder: the tiles of one keyframe of the released
/// canvases, which is the smallest condition a request brings.
const LEAST_CONDITION_RUNG: u32 = 1024;

/// The element type of the sample the trajectory accumulates, and of its noise.
///
/// The trajectory is kept in FP32 and any other type is refused at the denoiser's entry (the video
/// and audio latents' input and output are pinned there), so the draw, the patch and the unpatchify
/// all read and write this one type - **not the stack's**, which is the checkpoint's. The forward
/// refuses a latent of another type by name ([`Denoiser::denoise`]).
///
/// ## The two crossings
///
/// The reference keeps its trajectory in FP32 and rounds only inside a launch, so the trajectory
/// crosses to the stack's element type at exactly two points per evaluation, one launch each:
///
/// 1. each modality's input projection reads the FP32 patch rows against its FP32-pinned weights
///    into an FP32 output, which one `Cast` narrows into the packed rows;
/// 2. each modality's head reads the stack's rows after the final norm, widened by one `Cast`
///    (`Transformer::wide`) into an FP32 product against its FP32-pinned weights, and the
///    prediction stays FP32 through the unpatchify and the solver, whose step reads one type for
///    the sample and the velocity.
///
/// The four end projections are FP32 parameters at every preset, so both products are FP32 on both
/// sides whatever the stack's type is; at an FP32 stack both casts are skipped. Allocating any of
/// the sample's buffers in the stack's type instead would round the trajectory itself at every
/// evaluation, which no shape refuses.
pub const SAMPLE_DTYPE: DType = DType::F32;

/// The deployment's layout rule: how a request snaps to a layout, and what the layout packs.
pub struct Layouts {
    config: DenoiserConfig,
    /// The text rows every layout holds: the deployment's longest prompt.
    text_rows: u32,
    /// The condition rows a layout holds at most; a request rides the rung of the condition
    /// ladder that holds its own.
    condition_rows: u32,
    /// The denoiser's sequence split: the Ulysses degree its packing aligns to, and this rank's
    /// interval of the packed rows.
    sequence: Split,
}

impl Layouts {
    /// The rule of a checkpoint and a deployment holding `(text rows, condition rows)`, at
    /// `sequence`.
    ///
    /// # Errors
    ///
    /// A transformer without heads, a sequence degree that is not a power of two, or a capacity that
    /// is not a whole number of the attention kind's units.
    pub fn new(
        config: DenoiserConfig,
        (text_rows, condition_rows): (u32, u32),
        sequence: Split,
    ) -> Result<Self> {
        if config.transformer.head_dim == 0 || config.transformer.num_attention_heads == 0 {
            return Err(Error::config(
                "a denoiser's transformer has attention heads".to_owned(),
            ));
        }
        if sequence.size == 0 || !sequence.size.is_power_of_two() || sequence.rank >= sequence.size
        {
            return Err(Error::config(format!(
                "rank {} of a sequence split of {} is not a rank of a power-of-two degree",
                sequence.rank, sequence.size
            )));
        }
        let unit = config.attention.capacity_rows();
        for (name, rows) in [("text", text_rows), ("condition", condition_rows)] {
            if rows == 0 || !rows.is_multiple_of(unit) {
                return Err(Error::config(format!(
                    "a {name} capacity of {rows} rows is not a multiple of the {unit}-row unit"
                )));
            }
        }
        Ok(Self {
            config,
            text_rows,
            condition_rows,
            sequence,
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
        let mut cursor = 0_u64;
        Ok(request
            .conditions
            .iter()
            .map(|condition| {
                let elements = u64::from(self.condition_elements(std::slice::from_ref(condition)));
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
    /// VAE posterior's mean, and its stream position is not advanced,
    /// so counting its rows too would shift
    /// every later draw of the request - the video's and the audio's - by a few thousand elements
    /// and change the video silently. The count is computed from the conditions themselves, not
    /// from a packing, so it is total: every condition whose raster and frame count are ones the
    /// packing would accept contributes exactly its elements, and one it would refuse contributes
    /// none. A refused condition never reaches here - the runner lays the request out, and that
    /// packing is the same function of the same values - so a served request's offsets are exact.
    pub fn condition_elements(&self, conditions: &[nn::ConditionShape]) -> u32 {
        let channels = u64::from(self.config.transformer.video_channels);
        let elements: u64 = conditions
            .iter()
            .filter(|condition| condition.width > 0 && condition.height > 0)
            .map(|condition| {
                {
                    // A condition is cut into clips, so its latent count is the frame rule's for
                    // any count ([`processing::FrameClips::latent_frames`]): a 30-frame reference is
                    // 7 latent frames, and counting none because 30 is not a generated count would
                    // shift every later draw of the request by the condition's whole volume.
                    let frames = FRAMES.latent_frames(condition.frames);
                    let canvas = Canvas {
                        width: condition.width,
                        height: condition.height,
                    };
                    match packing::latent_raster(&canvas) {
                        Ok(raster) => {
                            u64::from(frames)
                                * u64::from(raster.height)
                                * u64::from(raster.width)
                                * channels
                        }
                        Err(_) => 0,
                    }
                }
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
        match self.config.attention {
            crate::config::AttentionKind::Sparse { tile, .. } => self.bound(|frames, canvas| {
                packing::segment_extent(
                    frames,
                    canvas,
                    self.text_rows,
                    self.condition_rows,
                    tile,
                    tile,
                )
                .map(|extent| extent.padded_tokens)
            }),
            crate::config::AttentionKind::Dense => self.bound(|frames, canvas| {
                packing::dense_packing(
                    frames,
                    canvas,
                    self.text_rows,
                    self.condition_rows,
                    self.sequence.size,
                )
                .map(|dense| dense.padded_tokens)
            }),
        }
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
        self.bound(|frames, canvas| {
            packing::segment_extent(
                frames,
                canvas,
                self.text_rows,
                self.condition_rows,
                tile,
                tile,
            )
            .map(|extent| extent.video_tiles)
        })
    }

    /// The largest `measure` of any layout this deployment's bounds are taken over: every canvas the
    /// checkpoint generates ([`crate::config::named_canvases`]) at every frame count of the frame
    /// rule up to [`MAX_FRAMES`], at the deployment's capacities. A layout the packing refuses is
    /// not one the deployment serves and measures nothing.
    ///
    /// The bounds are taken over the checkpoint's canvases and frame counts rather than the
    /// deployment's sizes and longest output, so that a deployment's attention scratch and the
    /// programs keyed by its bounds are one set for every deployment of a checkpoint and capacity.
    fn bound(&self, measure: impl Fn(u32, &Canvas) -> Result<u32>) -> Result<u32> {
        let mut bound = 0;
        for (_, canvas) in crate::config::named_canvases()? {
            let mut frames = FRAMES.min_frames;
            while frames <= MAX_FRAMES {
                if let Ok(measured) = measure(frames, &canvas) {
                    bound = bound.max(measured);
                }
                frames += FRAMES.clip_frames;
            }
        }
        if bound == 0 {
            return Err(Error::config(
                "a deployment offers no canvas a layout can be built for".to_owned(),
            ));
        }
        Ok(bound)
    }

    /// The sparse segment layout of `layout`, checked against the layout's own extents.
    pub fn segment_of(&self, layout: &Layout) -> Result<packing::SegmentExtent> {
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
        let extent = packing::segment_extent(
            frames,
            &canvas,
            layout.text_rows,
            layout.condition_rows,
            tile,
            tile,
        )?;
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
        Ok(extent)
    }

    /// The layout a request is laid out in, without packing it.
    ///
    /// This snaps the request the way [`Layouts::layout`] does and takes the deployment's text
    /// capacity and the request's rung; a canvas or a frame count the packing refuses is refused
    /// here by the same rule.
    pub fn cover(&self, request: &VideoRequest<'_>) -> Result<Layout> {
        let frames = frames_of(request);
        let canvas = canvas_of(request)?;
        let raster = packing::latent_raster(&canvas)?;
        Ok(Layout {
            video: LatentShape {
                channels: self.config.transformer.video_channels,
                frames: FRAMES.latent_frames(frames),
                height: raster.height,
                width: raster.width,
            },
            audio: Some(self.audio_latent(packing::audio_latent_frames(frames))),
            text_rows: self.text_rows,
            condition_rows: self.condition_rows_of(request),
            condition_elements: self.condition_elements(request.conditions),
        })
    }

    /// The condition rows `request`'s layout holds: the rung it states, the deployment's capacity
    /// when it states none, and none for a request without conditions.
    ///
    /// A deployment's capacity is sized for the widest request it serves, and a layout's rows are
    /// real rows of every projection, every attention and every feed-forward of every step. A
    /// `t2va` request at a capacity sized for `fl2va` carried 2048 empty rows through all of them,
    /// and an image reference at a capacity sized for a reference video carried 40000.
    fn condition_rows_of(&self, request: &VideoRequest) -> u32 {
        if task_of(request) == crate::VideoTask::TextToVideoAudio {
            0
        } else {
            request.condition_rows.unwrap_or(self.condition_rows)
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
        let capacity = self.condition_rows;
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

    /// The text rows every layout of this rule holds.
    pub const fn text_rows(&self) -> u32 {
        self.text_rows
    }

    /// The condition rows a layout of this rule holds at most.
    pub const fn condition_capacity(&self) -> u32 {
        self.condition_rows
    }

    /// The layout a request is served in, with the packing of its row extents.
    ///
    /// The canvas comes from the request's raster and the frames are snapped; the text capacity is
    /// the deployment's, because a layout holds every prompt that fits it, and the condition rows
    /// are the rung the request states.
    pub fn layout(&self, request: &VideoRequest) -> Result<(Layout, DensePacking)> {
        let frames = frames_of(request);
        let canvas = canvas_of(request)?;
        let packing = packing::dense_packing(
            frames,
            &canvas,
            self.text_rows,
            self.condition_rows_of(request),
            self.sequence.size,
        )?;
        let elements = self.condition_elements(request.conditions);
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
            self.sequence.size,
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

    /// The rows of one modality this rank owns: `(sources, slots)`.
    ///
    /// `packed` holds each of the modality's rows' place in the packed sequence, in the modality's
    /// own order. A rank owns one contiguous `rows / shard` interval of the packed sequence, so
    /// `sources[i]` is the modality row a rank projects and `slots[i]` the row of its own interval
    /// that row lands in. At one rank every row is owned and the two are the identity and `packed`.
    fn owned_rows(&self, rows: u64, packed: &[i32]) -> (Vec<i32>, Vec<i32>) {
        let shard = u64::from(self.sequence.size.max(1));
        let local = rows / shard;
        let first = u64::from(self.sequence.rank) * local;
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
fn segment_rows(
    extent: &packing::SegmentExtent,
    tables: &packing::RowTables,
) -> (Vec<i32>, Vec<i32>) {
    let mut video = vec![0_i32; tables.video_indices.len()];
    for (packed, raster) in tables
        .video_indices
        .iter()
        .zip(&tables.video_raster_indices)
    {
        video[*raster as usize] = *packed as i32;
    }
    (
        video,
        extent.audio_indices().map(|row| row as i32).collect(),
    )
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

/// The frame count a request is snapped to: the fewest the frame rule generates of at least its
/// own.
pub fn frames_of(request: &VideoRequest) -> u32 {
    FRAMES.align_up(request.frames)
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
/// A request's noise is **one stream addressed by element**: the visual
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
    if request.width == 0 || request.height == 0 {
        return Err(Error::config(format!(
            "a {}x{} request has no canvas",
            request.width, request.height
        )));
    }
    let width = round_up(request.width, CANVAS_RULE.multiple);
    let height = round_up(request.height, CANVAS_RULE.multiple);
    Ok(Canvas { width, height })
}

/// The canvas a layout's latent raster came from.
fn canvas_of_latents(layout: &Layout) -> Result<Canvas> {
    let canvas = Canvas {
        width: layout.video.width * packing::LATENT_COMPRESSION,
        height: layout.video.height * packing::LATENT_COMPRESSION,
    };
    packing::latent_raster(&canvas)?;
    Ok(canvas)
}

/// The generated frame count a latent frame count came from: `5n + 2` latent frames are `17n + 5`
/// generated frames ([`processing::FrameClips::frames_of_latents`]).
fn frame_count(latent_frames: u32) -> Result<u32> {
    FRAMES.frames_of_latents(latent_frames).ok_or_else(|| {
        Error::config(format!(
            "{latent_frames} latent frames are no timeline of the frame rule {FRAMES:?}"
        ))
    })
}

/// The pieces of `request`'s conditions, in the order the packing holds them.
///
/// The draws are the reference's: one per visual condition, in packed order, each at the condition's
/// own latent shape, leading the request's one stream. An
/// audio reference draws nothing. `hop` is the samples of one audio latent frame, which a track is
/// padded to; a deployment that holds no audio encoder states none, and a track is then refused.
fn pieces_of(
    config: &DenoiserConfig,
    hop: Option<u64>,
    request: &VideoRequest,
) -> Result<Vec<nn::capability::ConditionPiece>> {
    let canvas = canvas_of(request)?;
    let segments = packing::condition_segments(request.conditions, &canvas)?;
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
        let draw = (segment.frames > 0).then_some(LatentShape {
            channels,
            frames: segment.frames,
            height: segment.raster.height,
            width: segment.raster.width,
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
            pixels: (segment.frames > 0).then(|| {
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

/// The level a visual condition is anchored at: `t z + (1 - t) noise` with `t` this,
/// and the level its modulation group sits at
/// until the generated video is cleaner than it ([`crate::modulation::levels`]).
pub const VISUAL_CONDITION_TIME: f32 = 0.999;

/// The level an audio reference sits at: its latent unchanged, so nothing is mixed in.
///
/// Its packed rows are the **channel-major** audio latent - `2 * frames` rows of `audio_channels`
/// elements, 74 rows for a 22-frame reference - which is the layout the encoder writes and the
/// audio input projection reads; a `[frames, channels]` reshape would give 37 rows and shift every
/// row after the reference.
pub const AUDIO_CONDITION_TIME: f32 = 1.0;

/// The seed the encoder samples every visual condition's posterior under.
///
/// It is not the request's: this noise is drawn from a fresh generator of its own, so a condition
/// always encodes to the same latent, while the request's draw is the **anchor's** alone. One
/// buffer for both would put an independent draw under every condition latent - `mean + std * eps`
/// differs by `std * (eps_request - eps_42)` - and the rows that disagreement lands in feed every
/// evaluation.
pub const POSTERIOR_SEED: u64 = 42;

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

/// Round `value` up to the next multiple of `unit`.
fn round_up(value: u32, unit: u32) -> u32 {
    value.div_ceil(unit) * unit
}

/// One checkpoint's denoiser: the DiT, the schedule it is evaluated on, its solver and the AdaLN
/// cache.
///
/// ## The AdaLN cache
///
/// The reference derives the modulation tables at load (`weights._prepare_adaln_cache`), so they are
/// constants rather than per-request execution state: 50 tables of `[entries, 18 * hidden]` plus
/// `[entries, 2 * hidden]`, about 1.9 GiB at FP32 for a 50-step checkpoint and 0.33 GiB for the DMD
/// ladder. The [`TimestepTable`] reserves them as writable constants at load, and
/// [`nn::capability::Denoiser::initialize`] fills them once before any capture.
pub struct Denoiser {
    config: DenoiserConfig,
    rules: Layouts,
    schedule: Schedule,
    solver: Solver,
    transformer: Transformer,
    projections: Projections,
    /// The modulation at each distinct level of the schedule, which `projections` fill once after
    /// loading.
    cache: TimestepTable,
    /// The text refiner: the prompt features the DiT conditions on, refined once per request.
    refiner: crate::conditioning::TokenRefiner,
    /// The two condition encoders, when this deployment places them.
    ///
    /// A task with conditions (`fl2va`, `ref2va`) encodes the request's rasters and samples before its
    /// first evaluation; `t2va` has no conditions and constructs neither. They live inside this block
    /// rather than in one of their own because `nn` has no condition-encoder capability -
    /// `EncoderInputs` takes a token batch, and these take pixels and samples.
    encoders: Option<(crate::encoding::VideoEncoder, crate::encoding::AudioEncoder)>,
    /// The packed row bound this deployment was loaded for.
    ///
    /// The stack's dense attention is compiled with it as its key bound (`nn`'s `DenseDescriptor`
    /// carries `max_kv_len`), so a request wider than this has no module to launch: the bound is
    /// consumed here, at load, and a layout or a request beyond it is refused by name rather than
    /// discovered at the first layer.
    rows: u32,
    /// This rank's position in the sequence shard and the shard's degree.
    sequence: Split,
}

/// Load the two condition encoders a deployment with conditioned tasks holds: a condition's
/// pixels to its normalised latent, and its samples to its latent rows.
///
/// `fl2va` and `ref2va` read a keyframe's or a reference's pixels and samples, encode them, and
/// write the resulting rows into the prefix source before the request's first evaluation; a `t2va`
/// deployment places neither. A deployment's stages are built in the checkpoint's order and
/// `transformer` comes before `latent_encoder`, so the loader reads the encoders in a pass of its
/// own and hands them to [`Denoiser::load`].
pub fn load_encoders(
    config: &crate::Config,
    checkpoint: &nn::Checkpoint,
    loader: &mut WeightLoader<'_>,
) -> Result<(crate::encoding::VideoEncoder, crate::encoding::AudioEncoder)> {
    let video = crate::encoding::VideoEncoder::load(
        &config.video_vae,
        &mut loader.reborrow(checkpoint.stage("vae")?)?,
    )?;
    // The audio VAE directory holds both halves, as the video one does.
    let audio = crate::encoding::AudioEncoder::load(
        &config.audio_vae,
        &mut loader.reborrow(checkpoint.stage("audio_vae")?)?,
    )?;
    Ok((video, audio))
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
    task: crate::VideoTask,
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
    fn layout(&self, request: &VideoRequest<'_>) -> nn::Result<Layout> {
        Ok(self.rules.cover(request)?)
    }

    /// The denoising steps this checkpoint runs for `request`'s task ([`crate::VideoTask::of`]).
    ///
    /// Both modalities advance on one step counter - [`Schedule::num_steps`] refuses a grid whose two
    /// ladders disagree, and the solver refuses a step a modality does not have - so the video ladder
    /// **is** the request's ladder, not one of two.
    ///
    /// `task` is a key and not a position. Every H3 task shares this checkpoint's grid, which is
    /// exactly why a wrong key would go unnoticed: a runner that took the first schedule would run
    /// the right numbers for a task the checkpoint does not serve, and the deployment would serve a
    /// task its own contract denies.
    fn schedule(&self, request: &VideoRequest<'_>) -> nn::Result<Vec<nn::Step>> {
        let task = task_of(request);
        schedule_for(&self.config, &self.schedule, task).map_err(nn::Error::from)
    }

    /// The stack's width: the refiner projects the encoder's `text_dim` columns into it.
    ///
    /// H3's text encoder writes 5120-column features (`transformer.text_dim`) and every packed row
    /// the denoiser reads is `hidden_size` wide - the width of the `context_embedder` that is the
    /// refiner's input layer. So `Conditioning::text` and `Conditioning::source` are two widths
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
                return Err(nn::Error::from(Error::config(
                    "this layout draws audio noise and no audio latents were offered".to_owned(),
                )));
            }
            (None, Some(_)) => {
                return Err(nn::Error::from(Error::config(
                    "audio latents were offered to a layout with no audio stream".to_owned(),
                )));
            }
            _ => {}
        }
        // The draw writes the trajectory, so it reads the sample's type: a latent of another type
        // would launch a noise row the plan does not state, and `denoise` would refuse it one call
        // later with the launch already made.
        if out.video.dtype != SAMPLE_DTYPE {
            return Err(nn::Error::from(Error::config(format!(
                "the video latents are {} and the draw writes {SAMPLE_DTYPE}",
                out.video.dtype
            ))));
        }
        if let Some(audio) = out.audio
            && audio.dtype != SAMPLE_DTYPE
        {
            return Err(nn::Error::from(Error::config(format!(
                "the audio latents are {} and the draw writes {SAMPLE_DTYPE}",
                audio.dtype
            ))));
        }
        let noise = nn::Noise::new();
        // The video's stream is drawn at the latent's **native** shape, `[channels, frames, height,
        // width]`, which is the buffer `denoise` reads and the solver folds a velocity into - its
        // own patch is `Patch::forward(f, video, patched, frames)`, so the sample a step updates is
        // the raster, not the rows. The reference draws the same shape (`noise_shape("video")`)
        // and patchifies it on the way into its state (`prepare_latents`).
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
            return Err(nn::Error::from(Error::config(format!(
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
        // stream, each buffer consuming the elements the one before it ended at (the seed's stream
        // draws each condition's noise, then each modality's), so the audio's noise is that stream
        // continued - not an independent stream drawn under the same seed. A `draw` of one per
        // modality would give every audio latent noise the reference never produces, which a
        // comparison of `video` would pass and which no shape or type refuses.
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
        let hop = self.encoders.as_ref().map(|(_, audio)| audio.hop());
        pieces_of(&self.config, hop, request).map_err(nn::Error::from)
    }

    fn draw_condition_noise(
        &self,
        f: &Forward,
        seed: nn::capability::Seed,
        piece: &nn::capability::ConditionPiece,
        out: Operand,
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

    fn prepare_latents(
        &self,
        f: &Forward,
        cond: nn::capability::Conditioning<'_>,
        out: nn::capability::Latents,
    ) -> nn::Result<()> {
        // The schedule's start: unit noise for the released and distilled grids, a scaled draw for
        // a parallel-decoding student.
        let _ = cond;
        self.solver
            .start(f, Modality::Video, out.video)
            .map_err(nn::Error::from)?;
        if let Some(audio) = out.audio {
            self.solver
                .start(f, Modality::Audio, audio)
                .map_err(nn::Error::from)?;
        }
        Ok(())
    }

    /// Refine the prompt's features into the prefix source's text rows and project the request's
    /// conditions into the rows below them, once per request: both are functions of the request
    /// alone, and the step's graph reads the rows they write.
    ///
    /// A request with no media has no condition rows at all, which is every `t2va`.
    fn prepare_conditioning(
        &self,
        f: &Forward,
        request: &VideoRequest<'_>,
        cond: nn::capability::Conditioning<'_>,
    ) -> nn::Result<()> {
        self.refine(f, request, cond).map_err(nn::Error::from)?;
        if cond.media.is_empty() {
            return Ok(());
        }
        self.encode_media(f, cond.media, cond.text)
            .map_err(nn::Error::from)
    }

    fn forward(
        &self,
        f: &Forward,
        step: nn::Step,
        request: &VideoRequest<'_>,
        latents: nn::capability::Latents,
        cond: nn::capability::Conditioning<'_>,
    ) -> nn::Result<()> {
        Denoiser::step(self, f, step, request, latents, cond).map_err(nn::Error::from)
    }

    /// Fill the AdaLN cache from the schedule's levels: the one post-load launch.
    fn initialize(&self, f: &Forward) -> nn::Result<()> {
        self.projections
            .compute(f, &self.cache)
            .map_err(nn::Error::from)
    }
}

/// Store the contiguous `source` into the contiguous `destination` of the same shape, rounding
/// once into the destination's element type (`destination ← 1 · source`).
fn store(f: &Forward, source: Operand, destination: Operand) -> Result<()> {
    Ok(nn::layers::Axpby.forward(f, destination, source, 1.0, 0.0)?)
}

/// The task a request runs, from the conditions it states.
fn task_of(request: &VideoRequest<'_>) -> crate::VideoTask {
    crate::VideoTask::of(
        request.first_frames,
        request.last_frames,
        request.references,
    )
}

/// A rank-one row table as the rows of width one a row gather moves.
fn one_column(view: Operand) -> Operand {
    view.reshape(Shape::new(&[view.numel(), 1]))
}

impl Denoiser {
    /// Whether this deployment places the condition encoders.
    ///
    /// The runner asks before it offers a conditioned request: a task that carries conditions on a
    /// deployment that places none is a configuration the runner refuses, and it is better refused
    /// there than discovered as a video conditioned on nothing.
    pub fn places_encoders(&self) -> bool {
        self.encoders.is_some()
    }

    /// Encode the request's raw media and write its condition rows into the prefix source.
    ///
    /// The rows are the layout's - `MediaInput::rows` and `audio_rows` carry the counts the packing
    /// stated - so this only decides where each block starts ([`condition_ranges`]) and checks
    /// the encoders against it. Within a piece the audio rows come first, which is the reference's
    /// order.
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
        text: Operand,
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
        let text_rows = u64::from(self.rules.text_rows());
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
                let encoded = encoders.1.encode(f, samples)?;
                if encoded.shape.dims() != [rows, audio_channels] {
                    return Err(Error::config(format!(
                        "condition {:?} declares {rows} audio rows and its track encodes to {:?}",
                        piece.role,
                        encoded.shape.dims()
                    )));
                }
                // The projection's output is FP32, and the one rounding is the store into the
                // rows (`audio_input(rows, output_dtype=torch.float32).to(out.dtype)`).
                let projected = self.transformer.project_audio(f, encoded)?;
                store(f, projected, text.narrow(0, start, rows)?)?;
            }
            if piece.rows > 0 {
                let visual = start + u64::from(piece.audio_rows);
                let rows = text.narrow(0, visual, u64::from(piece.rows))?;
                self.encode_visual(f, piece, encoders, rows)?;
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
        encoders: &(crate::encoding::VideoEncoder, crate::encoding::AudioEncoder),
        rows: Operand,
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
        let dims = pixels.shape.dims();
        let (frames, height, width) = (dims[2], dims[3], dims[4]);
        let raster = pixels.reshape(Shape::new(&[dims[1], frames, height, width]));
        let frame_count = u32::try_from(frames)
            .ok()
            .filter(|frames| *frames > 0)
            .ok_or_else(|| {
                Error::config(format!(
                    "a video condition of {frames} frames encodes nothing"
                ))
            })?;
        let clip = crate::tiled_encode::ClipEncoder::new(
            u32::try_from(height).unwrap_or(0),
            u32::try_from(width).unwrap_or(0),
        )?;

        // A condition is encoded a clip at a time, each clip alone, into its latent frames of the
        // run's stitched moments: a still is one clip of one frame, a reference video its whole
        // clips and a shorter last one. The moments are `2 · latent_channels` wide.
        let video_channels = u64::from(self.config.transformer.video_channels);
        let moment_channels = 2 * video_channels;
        let (run_rows, run_columns) = clip.latent_extent();
        let latent_frames = u64::from(FRAMES.latent_frames(frame_count));
        let run = f.empty(
            DType::F32,
            Shape::new(&[
                moment_channels,
                latent_frames,
                u64::from(run_rows),
                u64::from(run_columns),
            ]),
        )?;
        for span in FRAMES.clips(frame_count) {
            f.scoped(|f| {
                let clip_pixels =
                    raster.narrow(1, u64::from(span.first), u64::from(span.frames))?;
                let moments = clip.encode(
                    f,
                    &encoders.0,
                    clip_pixels,
                    (moment_channels, u64::from(span.latents)),
                )?;
                let destination = run.narrow(1, u64::from(span.latent), u64::from(span.latents))?;
                let mut strides = [0_i64; ops::copy_block::RANK];
                for (axis, stride) in destination.strides.as_slice().iter().enumerate() {
                    strides[axis] = i64::try_from(*stride).map_err(|_| {
                        Error::config("a condition's moments exceed i64 strides".to_owned())
                    })?;
                }
                nn::layers::Copy::new(DType::F32, nn::layers::Access::Element)?.forward(
                    f,
                    moments,
                    destination,
                    &strides,
                )?;
                Ok::<_, Error>(())
            })?;
        }

        // `quant_conv` once over the stitched raster, then the posterior's sample with its own
        // draw at the module's seed: a fresh generator per condition, element zero each time. The
        // piece's draw is the anchor's, and only there.
        let moments = encoders.0.project(f, run)?;
        let latent_shape = Shape::new(&[
            video_channels,
            latent_frames,
            u64::from(run_rows),
            u64::from(run_columns),
        ]);
        let posterior = f.empty(DType::F32, latent_shape)?;
        nn::Noise::new().draw(
            f,
            posterior.reshape(Shape::new(&[
                video_channels * latent_frames * u64::from(run_rows),
                u64::from(run_columns),
            ])),
            nn::NoiseKey::Scalar(POSTERIOR_SEED),
            0,
            0,
            0,
        )?;
        let latent = encoders.0.posterior(f, moments, posterior)?;

        // The latent and the draw become rows by the same mapping (the draw is drawn at latent
        // resolution), the anchor `t · clean + (1 - t) · draw` mixes them in place in FP32, and
        // the FP32 projection is stored into the rows with the one rounding.
        let clean = self.transformer.patchify(f, latent)?;
        let anchor = self.transformer.patchify(f, draw)?;
        if clean.rows() != u64::from(piece.rows) {
            return Err(Error::config(format!(
                "condition {:?} declares {} visual rows and encodes to {}",
                piece.role,
                piece.rows,
                clean.rows()
            )));
        }
        nn::diffusion::add_noise(f, clean, anchor, VISUAL_CONDITION_TIME)?;
        let projected = self.transformer.project_video(f, clean)?;
        store(f, projected, rows)
    }

    /// Refine the prompt's features, `cond.source`, into the text rows of `cond.text`.
    ///
    /// The refiner reads one document: its queries are every padded row of the text capacity and
    /// its keys stop at the prompt's last row. A model with no conditioner has no `source`.
    fn refine(
        &self,
        f: &Forward,
        request: &VideoRequest<'_>,
        cond: nn::Conditioning<'_>,
    ) -> Result<()> {
        let Some(source) = cond.source else {
            return Ok(());
        };
        let layout = self.rules.cover(request)?;
        let rows = layout.text_rows;
        let capacity = u64::from(rows);
        let int = |values: &[i32]| f.upload(values, Shape::new(&[values.len() as u64]));
        let document = nn::layers::Varlen {
            cu_q: int(&[0, rows as i32])?,
            cu_k: int(&[0, request.text_tokens as i32])?,
            max_q: None,
            max_k: rows,
        };
        let refined = self
            .refiner
            .forward(f, source.narrow(0, 0, capacity)?, document)?;
        store(f, refined, cond.text.narrow(0, 0, capacity)?)
    }

    /// One denoising step of `request`: the latents in, the model's velocity, and the schedule's
    /// update in place.
    ///
    /// The packed rows are assembled from the step's uploaded layout ([`Layouts::packed`]): every
    /// row gathers its prefix source (the refined text row it reads, a condition row or the zero
    /// row), then each modality's patchified latents are projected and scattered into this rank's
    /// own rows. The stack runs once, conditioned by the step's AdaLN rows, and each modality's
    /// velocity is projected back, gathered from every rank, unpatched and folded into its latents
    /// by the solver.
    pub fn step(
        &self,
        f: &Forward,
        step: nn::Step,
        request: &VideoRequest<'_>,
        latents: nn::Latents,
        cond: nn::Conditioning<'_>,
    ) -> Result<()> {
        let video = latents.video;
        // The trajectory is FP32 and a latent of any other type is refused: a sample in the
        // stack's type would round the trajectory once per evaluation.
        if video.dtype != SAMPLE_DTYPE
            || latents
                .audio
                .is_some_and(|audio| audio.dtype != SAMPLE_DTYPE)
        {
            return Err(Error::config(format!(
                "the latents are {} and a trajectory is {SAMPLE_DTYPE}",
                video.dtype
            )));
        }
        let layout = self.rules.cover(request)?;
        // What the step reads of its layout, computed on the host and placed on the device: the
        // packing's indices, this rank's rows of each modality, the rotary coordinates and the
        // attention's visibility. Every extent follows from `layout` alone (its packing, this
        // rank's interval of it), so a capture and every replay of the layout upload at the same
        // offsets; only the values follow from the request's prompt length, conditions and vision
        // spans.
        if request.text_tokens > layout.text_rows {
            return Err(Error::config(format!(
                "a prompt of {} rows does not fit a text capacity of {}",
                request.text_tokens, layout.text_rows
            )));
        }
        let frames = frame_count(layout.video.frames)?;
        let canvas = canvas_of_latents(&layout)?;
        let segments = packing::condition_segments(request.conditions, &canvas)?;
        let spans = vision_spans(request);
        let int = |values: &[i32]| f.upload(values, Shape::new(&[values.len() as u64]));

        let (rows, video_places, audio_places, tables, visibility) =
            match self.rules.config.attention {
                crate::config::AttentionKind::Dense => {
                    let dense = self.rules.packing_of(&layout)?;
                    let tables = packing::dense_tables(
                        &dense,
                        frames,
                        &canvas,
                        request.text_tokens,
                        &segments,
                        &spans,
                    )?;
                    let rows = tables.groups.len() as u64;
                    let (video_places, audio_places) = dense_rows(&dense);
                    // One sequence: every query row sees the used key rows, no query a padding row.
                    let visible = nn::layers::Varlen {
                        cu_q: int(&[0, rows as i32])?,
                        cu_k: int(&[0, tables.used as i32])?,
                        max_q: None,
                        max_k: rows as u32,
                    };
                    (
                        rows,
                        video_places,
                        audio_places,
                        tables,
                        Visibility::Dense(visible),
                    )
                }
                crate::config::AttentionKind::Sparse {
                    sparsity,
                    reference_keep,
                    ..
                } => {
                    let extent = self.rules.segment_of(&layout)?;
                    let tables = packing::segment_tables(
                        &extent,
                        frames,
                        &canvas,
                        (request.text_tokens, &spans),
                        &segments,
                        (sparsity, reference_keep),
                    )?;
                    let rows = tables.groups.len() as u64;
                    if rows != u64::from(extent.padded_tokens) {
                        return Err(Error::config(format!(
                            "{rows} table rows for {} packed rows",
                            extent.padded_tokens
                        )));
                    }
                    let sizes: Vec<i32> = tables
                        .variable_block_sizes
                        .iter()
                        .map(|size| *size as i32)
                        .collect();
                    let (video_places, audio_places) = segment_rows(&extent, &tables);
                    // The keep table is over-allocated to the tile count and filled by segment index,
                    // exactly as the reference builds it.
                    let sparse = nn::layers::VsaTables {
                        sizes: int(&sizes)?,
                        segments: int(&tables.tile_segments)?,
                        keep: int(&tables.segment_keep)?,
                    };
                    (
                        rows,
                        video_places,
                        audio_places,
                        tables,
                        Visibility::Sparse(sparse),
                    )
                }
            };
        let (video_sources, video_slots) = self.rules.owned_rows(rows, &video_places);
        let (audio_sources, audio_slots) = self.rules.owned_rows(rows, &audio_places);
        // The AdaLN row of every packed row: `3 · group + tag` ([`crate::modulation::row`]).
        let modulation: Vec<i32> = tables
            .groups
            .iter()
            .zip(&tables.token_tags)
            .map(|(group, tag)| crate::modulation::TAGS as i32 * group + tag)
            .collect();
        // Axis-major `[3, rows]` coordinates: the sectioned rotation reads one real coordinate per
        // axis per token, and the clock and raster are rescaled, so they stay real numbers.
        let coordinates: Vec<f32> = (0..3)
            .flat_map(|axis| {
                tables
                    .position_ids
                    .iter()
                    .map(move |point| point[axis] as f32)
            })
            .collect();
        let prefix_index = int(&tables.prefix_index)?;
        let groups = int(&tables.groups)?;
        let modulation = int(&modulation)?;
        let positions = f.upload(&coordinates, Shape::new(&[3, rows]))?;
        let video_rows = (
            int(&video_places)?,
            int(&video_sources)?,
            int(&video_slots)?,
        );
        let audio_rows = (
            int(&audio_places)?,
            int(&audio_sources)?,
            int(&audio_slots)?,
        );
        let tokens = rows;
        // The stack's attention is compiled for this deployment's packed-row bound.
        if tokens > u64::from(self.rows) {
            return Err(Error::config(format!(
                "this denoiser was loaded for {} packed rows and the layout has {tokens}",
                self.rows
            )));
        }
        let shard = u64::from(self.sequence.size.max(1));
        if !tokens.is_multiple_of(shard) {
            return Err(Error::config(format!(
                "{tokens} packed rows do not divide over {shard} ranks"
            )));
        }
        // This rank owns one contiguous interval of the packed rows; a per-row table is cut to it.
        let local = tokens / shard;
        let first = u64::from(self.sequence.rank) * local;
        let own = |table: Operand| table.narrow(0, first, local);
        let index = i32::try_from(step.index)
            .map_err(|_| Error::config(format!("evaluation {} exceeds i32", step.index)))?;
        let evaluation = f.upload(&[index], Shape::new(&[1]))?;
        let hidden_size = u64::from(self.config.transformer.hidden_size);

        // The prefix source: the refined prompt rows and the condition rows below them were
        // written once per request by `prepare_conditioning`, and the zero row is the encoder
        // stage's.
        let text = cond.text;
        let dtype = text.dtype;

        // The packed rows: every row gathers its prefix source.
        let hidden = nn::layers::IndexRows.gather(f, text, own(prefix_index)?)?;

        // The video rows: patchify, project in FP32, narrow, scatter into this rank's rows.
        let channels = video.shape.dim(0);
        let (video_indices, video_sources, video_slots) = video_rows;
        let patched = self.transformer.patchify(f, video)?;
        let own_video = video_sources.numel();
        if own_video > 0 {
            let owned = nn::layers::IndexRows.gather(f, patched, video_sources)?;
            let projected = self.transformer.project_video(f, owned)?;
            let narrowed = nn::layers::Cast.forward(f, projected, dtype)?;
            nn::layers::IndexRows.scatter(f, narrowed, video_slots, hidden)?;
        }

        // The audio rows: channel-major latent rows, projected like the video's.
        let features = u64::from(self.config.transformer.audio_channels);
        let (audio_indices, audio_sources, audio_slots) = audio_rows;
        let audio = match latents.audio {
            Some(audio) => Some(audio.reshape(Shape::new(&[audio_indices.numel(), features]))),
            None if audio_indices.numel() > 0 => {
                return Err(Error::config(format!(
                    "this layout has {} audio rows and no audio latents",
                    audio_indices.numel()
                )));
            }
            None => None,
        };
        if let Some(audio) = audio
            && audio_sources.numel() > 0
        {
            let owned = nn::layers::IndexRows.gather(f, audio, audio_sources)?;
            let projected = self.transformer.project_audio(f, owned)?;
            let narrowed = nn::layers::Cast.forward(f, projected, dtype)?;
            nn::layers::IndexRows.scatter(f, narrowed, audio_slots, hidden)?;
        }

        // The stack, conditioned by the step's gathered modulation, over the whole sequence's
        // rotary factors.
        let factors = self.transformer.rotary().factors(f, positions)?;
        let entries = self.cache.entries_of(f, evaluation)?;
        let modulation_rows = own(modulation)?;
        self.transformer.forward(
            f,
            (&self.cache, entries),
            hidden,
            modulation_rows,
            (factors, &visibility),
        )?;

        // The output: the step's final modulation, then each modality's head over its own rows.
        let output_table = ModulationTable::output(self.cache.output(f, entries)?, hidden_size)?;
        let groups = one_column(own(groups)?);
        let velocity = |rows: (Operand, Operand), modality: Modality, width: u64| {
            let (indices, slots) = rows;
            let predicted = if slots.numel() > 0 {
                let selected = nn::layers::IndexRows.gather(f, hidden, slots)?;
                let selected_groups = nn::layers::IndexRows.gather(f, groups, slots)?;
                Some(self.transformer.velocity(
                    f,
                    selected,
                    modality,
                    RowModulation {
                        table: &output_table,
                        rows: selected_groups.reshape(Shape::new(&[slots.numel()])),
                    },
                    (step.index, evaluation),
                )?)
            } else {
                None
            };
            self.collected(f, predicted, (slots, indices), (width, local, tokens))
        };
        let video_velocity = velocity((video_indices, video_slots), Modality::Video, channels * 4)?;
        let video_velocity = self
            .transformer
            .unpatchify(f, video_velocity, video.shape)?;
        self.solver.step(
            f,
            Modality::Video,
            (video, video_velocity),
            (step.index, evaluation),
        )?;
        if let Some(audio) = audio {
            let audio_velocity = velocity((audio_indices, audio_slots), Modality::Audio, features)?;
            self.solver.step(
                f,
                Modality::Audio,
                (audio, audio_velocity),
                (step.index, evaluation),
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
    fn collected(
        &self,
        f: &Forward,
        predicted: Option<Operand>,
        (slots, indices): (Operand, Operand),
        (width, local, tokens): (u64, u64, u64),
    ) -> Result<Operand> {
        if self.sequence.size <= 1 {
            return predicted.ok_or_else(|| {
                Error::config("one rank owns every row, and it predicted none".to_owned())
            });
        }
        let interval = f.empty(SAMPLE_DTYPE, Shape::new(&[local, width]))?;
        if let Some(predicted) = predicted {
            nn::layers::IndexRows.scatter(f, predicted, slots, interval)?;
        }
        let sequence = f.empty(SAMPLE_DTYPE, Shape::new(&[tokens, width]))?;
        if f.split().is_none() {
            return Err(Error::Unsupported(
                "a sequence shard collects its predictions over a group, and this forward \
                 carries none"
                    .to_owned(),
            ));
        }
        f.all_gather(interval, sequence)?;
        Ok(nn::layers::IndexRows.gather(f, sequence, indices)?)
    }

    /// Read one DiT partition and everything around it.
    ///
    /// `capacity` is the deployment's `(text rows, condition rows)`, from which the packed-row
    /// bound every attention is compiled for follows; `sequence` the denoiser's sequence split.
    /// `encoders` are the condition encoders when the deployment places them, and `heads` a
    /// parallel-decoding student's fused heads. The AdaLN cache is sized from the checkpoint's
    /// schedule alone - the distinct noise levels of its grid - so no request grows it.
    pub fn load(
        loader: &mut WeightLoader<'_>,
        config: DenoiserConfig,
        capacity: (u32, u32),
        sequence: SequenceSplit,
        encoders: Option<(crate::encoding::VideoEncoder, crate::encoding::AudioEncoder)>,
        heads: Option<(nn::layers::StepIndexed, nn::layers::StepIndexed)>,
    ) -> Result<Self> {
        let rules = Layouts::new(config.clone(), capacity, sequence.split)?;
        let rows = rules.rows_bound()?;
        // A sparse deployment's compression reduces over every video tile any layout holds.
        let tiles = match config.attention {
            crate::config::AttentionKind::Sparse { .. } => rules.tiles_bound()?,
            crate::config::AttentionKind::Dense => 0,
        };
        let schedule = Schedule::new(&config.schedule)?;
        let evaluations = schedule.num_steps()?;
        let conditioned = config
            .tasks
            .iter()
            .any(|task| !matches!(task, crate::VideoTask::TextToVideoAudio));
        let levels = crate::modulation::levels(
            &schedule.timesteps(Modality::Video),
            &schedule.timesteps(Modality::Audio),
            conditioned,
        )?;
        let hidden = u64::from(config.transformer.hidden_size);
        let projections = Projections::load(&config.transformer, loader)?;
        // The tables take the modulation projections' own type, as the reference allocates them
        // from the product.
        let cache = TimestepTable::load(
            &mut loader.scope("adaln"),
            &levels,
            crate::modulation::groups(conditioned),
            TimestepTableSpec {
                frequency_dim: config.transformer.frequency_dim,
                layers: config.transformer.num_hidden_layers,
                layer_columns: ADALN_COLUMNS * hidden,
                output_columns: 2 * hidden,
                dtype: projections.dtype(),
            },
        )?;
        if cache.steps() < u64::from(evaluations) {
            return Err(Error::checkpoint(format!(
                "the AdaLN entry table has {} rows for {evaluations} evaluations",
                cache.steps()
            )));
        }
        let transformer = Transformer::load(
            loader,
            (&config.transformer, config.attention),
            (rows, tiles),
            sequence,
            heads,
        )?;
        let solver = Solver::load(&schedule, loader)?;
        // The refiner reads the prompt's rows, whose count is the text capacity.
        let refiner = crate::conditioning::TokenRefiner::load(
            loader,
            &config.transformer,
            capacity.0,
            transformer.dtype(),
        )?;
        Ok(Self {
            config,
            rules,
            schedule,
            solver,
            transformer,
            projections,
            cache,
            refiner,
            encoders,
            rows,
            sequence: sequence.split,
        })
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
}

#[cfg(test)]
mod tests {

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
            tasks: vec![crate::VideoTask::TextToVideoAudio],
        };
        Layouts::new(config, (128, 64), Split::WHOLE).unwrap()
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
            tasks: vec![crate::VideoTask::TextToVideoAudio],
        };
        let schedule = Schedule::new(&config.schedule).expect("the checkpoint's schedule");
        let count = schedule.num_steps().expect("an aligned grid");
        let steps = schedule_for(&config, &schedule, crate::VideoTask::TextToVideoAudio)
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
        let error = schedule_for(&config, &schedule, crate::VideoTask::ReferenceToVideoAudio)
            .expect_err("a task this checkpoint does not serve");
        let message = error.to_string();
        assert!(message.contains("ReferenceToVideoAudio"), "{message}");
        assert!(message.contains("TextToVideoAudio"), "{message}");
    }

    fn request(frames: u32, width: u32, height: u32) -> VideoRequest<'static> {
        VideoRequest {
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
        Layouts::new(config, (128, 64), Split::WHOLE).unwrap()
    }

    /// The ladder starts at one keyframe, doubles, gains a halfway rung from four keyframes up, and
    /// ends at the capacity whatever it is; a capacity below the least rung is the only rung.
    #[test]
    fn the_condition_ladder_climbs_to_the_capacity() {
        let ladder = |capacity| {
            let mut rules = sparse_layouts();
            rules.condition_rows = capacity;
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

    /// A request the packing cannot lay out is refused, not covered by another canvas.
    #[test]
    fn a_request_without_a_canvas_has_no_layout() {
        let rules = layouts();
        assert!(rules.cover(&request(22, 0, 768)).is_err());
        assert!(rules.cover(&request(22, 1344, 0)).is_err());
        let covered = rules.cover(&request(22, 1344, 768)).unwrap();
        assert_eq!((covered.video.width, covered.video.height), (84, 48));
    }

    /// A layout holds the rung its request states, its rows are that rung's, and a request that
    /// states none holds the capacity. A request without conditions holds none at any rung.
    #[test]
    fn a_layout_holds_the_rung_of_its_request() {
        let mut rules = sparse_layouts();
        rules.condition_rows = 2048;
        let widest = rules.cover(&conditioned()).unwrap();
        assert_eq!(widest.condition_rows, 2048);
        let rung = rules
            .cover(&VideoRequest {
                condition_rows: Some(1024),
                ..conditioned()
            })
            .unwrap();
        assert_eq!(rung.condition_rows, 1024);
        let rows = |layout| rules.segment_of(layout).unwrap().padded_tokens;
        assert_eq!(rows(&widest) - rows(&rung), 1024);
        // A request of no anchor and no reference is `t2va`, whatever rung it states.
        let plain = rules
            .cover(&VideoRequest {
                condition_rows: Some(1024),
                first_frames: 0,
                last_frames: 0,
                references: 0,
                conditions: &[],
                ..conditioned()
            })
            .unwrap();
        assert_eq!(plain.condition_rows, 0);
    }

    #[test]
    fn a_conditions_draws_partition_the_condition_block() {
        // A runner sizes one buffer by `condition_elements` and slices each condition out of it by
        // `condition_offsets`, so the two have to agree on the same request and the entries have to
        // be contiguous in packed order. An audio-only reference is the case a prefix sum over
        // counts gets wrong: it draws nothing, so its entry is zero-length
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
        assert_eq!(
            cursor,
            u64::from(rule.condition_elements(&shapes)),
            "the offsets partition the block `condition_elements` sizes"
        );
    }

    #[test]
    fn a_sparse_layout_rebuilds_its_segment_packing() {
        let rule = sparse_layouts();
        // The dense layout carries the capacities a request names; the segment packing rounds each
        // prefix capacity up to its tile and pads the rows to whole tiles.
        let (layout, _) = rule.layout(&request(121, 1344, 768)).unwrap();
        let packed = rule.segment_of(&layout).unwrap();
        assert_eq!(packed.tile, 64);
        assert_eq!(packed.text_rows, 128);
        assert_eq!(packed.condition_rows, 0);
        assert_eq!(packed.padded_tokens % 64, 0);
        assert!(packed.video_start() + packed.video_tiles * 64 <= packed.padded_tokens);
        // A dense deployment has no segment packing to give, and says so rather than guessing.
        assert!(layouts().segment_of(&layout).is_err());
    }

    #[test]
    fn a_layout_built_densely_is_not_a_segment_packing_a_sparse_deployment_builds() {
        // The same request read by the two deployments: only the sparse one pads to whole tiles, and
        // a layout whose raster does not match its frame count is refused by both.
        let (layout, dense) = layouts().layout(&request(121, 1344, 768)).unwrap();
        let packed = sparse_layouts().segment_of(&layout).unwrap();
        assert!(packed.padded_tokens > dense.padded_tokens);
        let mut broken = layout;
        broken.video.frames = 24;
        assert!(sparse_layouts().segment_of(&broken).is_err());
    }

    #[test]
    fn the_tile_bound_covers_the_longest_layout_of_every_canvas() {
        // The compression's score matrix has one column per video tile, so the bound has to cover
        // the longest layout of every canvas the checkpoint generates, and is reached by one of
        // them: a number larger than any layout's spends memory on nothing, and one smaller refuses
        // on the device instead of counting.
        let rules = sparse_layouts();
        let bound = rules
            .tiles_bound()
            .expect("a sparse deployment bounds its tiles");
        let longest = FRAMES.align_down(MAX_FRAMES).expect("a timeline");
        let tiles: Vec<u32> = crate::config::named_canvases()
            .expect("the named canvases")
            .into_iter()
            .map(|(_, canvas)| {
                let request = request(longest, canvas.width, canvas.height);
                rules
                    .segment_of(&rules.cover(&request).expect("a layout"))
                    .expect("a segment packing")
                    .video_tiles
            })
            .collect();
        assert!(tiles.iter().all(|tiles| *tiles <= bound), "{tiles:?}");
        assert_eq!(tiles.iter().max(), Some(&bound));
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
        let conditions = [nn::ConditionShape {
            role: nn::ConditionRole::FirstFrame,
            frames: 1,
            height: 768,
            width: 1344,
            audio_samples: 0,
        }];
        // One latent frame of the canvas' raster, 24 channels wide: the video's stream starts here.
        assert_eq!(layouts().condition_elements(&conditions), 48 * 84 * 24);
    }

    #[test]
    fn a_reference_clip_draws_the_latent_frames_the_encoder_cuts_it_into() {
        // A reference clip that is not a whole number of the generated timeline's clips is still a
        // condition: the encoder cuts it into 17-frame clips and drops the last one's tail, so 30
        // frames are 7 latent frames. Counting it with the generated rule would give 0 - 30 is not
        // `17n + 5` - and every later draw of the request would move by the whole condition.
        let canvas = proto::video::Canvas {
            width: 1344,
            height: 768,
        };
        let conditions = [nn::ConditionShape {
            role: nn::ConditionRole::Reference,
            frames: 30,
            height: canvas.height,
            width: canvas.width,
            audio_samples: 0,
        }];
        let rule = layouts();
        let raster = packing::latent_raster(&canvas).expect("the canvas' raster");
        let latents = FRAMES.latent_frames(30);
        assert!(latents > 1, "the condition is more than a still");
        assert_eq!(
            rule.condition_elements(&conditions),
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
        let covered = rule.cover(&request).unwrap();
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
        rule.sequence.size = 1;
        rule.sequence.rank = 0;
        let (sources, slots) = rule.owned_rows(rows, &packed);
        assert_eq!(sources, (0..60).collect::<Vec<i32>>());
        assert_eq!(slots, packed);
        rule.sequence.size = 4;
        let mut seen = Vec::new();
        for rank in 0..4 {
            rule.sequence.rank = rank;
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
    fn an_audio_reference_draws_no_noise() {
        // The request's stream holds one draw per *visual* condition: the reference conditions an
        // audio reference at its posterior's mean, and counting its rows would shift the video's
        // draw and change the video without failing anything.
        let rule = layouts();
        let request = conditioned();
        let conditions = request.conditions;
        let channels = u64::from(rule.config.transformer.video_channels);
        let visual: u64 = conditions
            .iter()
            .filter(|condition| condition.width > 0)
            .map(|condition| {
                let raster = packing::latent_raster(&Canvas {
                    width: condition.width,
                    height: condition.height,
                })
                .unwrap();
                let frames = if condition.frames <= 1 {
                    1
                } else {
                    FRAMES.latent_frames(condition.frames)
                };
                u64::from(frames) * u64::from(raster.height) * u64::from(raster.width) * channels
            })
            .sum();
        assert_eq!(u64::from(rule.condition_elements(conditions)), visual);
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
            references: 1,
            conditions: shapes,
            ..request(121, 832, 480)
        };
        let canvas = Canvas {
            width: 832,
            height: 480,
        };
        let segments = packing::condition_segments(request.conditions, &canvas).unwrap();
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
        let segment = packing::segment_extent(
            124,
            &Canvas {
                width: 1344,
                height: 768,
            },
            rule.text_rows(),
            rule.condition_capacity(),
            64,
            64,
        )
        .unwrap();
        assert!(segment.padded_tokens > dense_packing.padded_tokens);
    }

    #[test]
    fn the_layout_states_the_condition_elements_it_draws() {
        // The layout's declared `condition_elements` is where the video's draw begins in the
        // request's stream; a request with no conditions starts the video at zero.
        let rule = layouts();
        let request = request(121, 1344, 768);
        let (layout, packing) = rule.layout(&request).unwrap();
        let canvas = canvas_of(&request).unwrap();
        let declared = rule.condition_elements(request.conditions);
        assert_eq!(layout.condition_elements, declared);
        let segments = packing::condition_segments(request.conditions, &canvas).unwrap();
        assert_eq!(segments.len(), request.conditions.len());
        assert!(packing.padded_tokens > packing.video_rows + packing.audio_rows);
        // With no conditions the video's stream starts at zero.
        assert_eq!(declared, 0);
    }

    #[test]
    fn a_prompt_wider_than_the_capacity_is_refused() {
        let rule = layouts();
        let request = request(121, 1344, 768);
        let (layout, packing) = rule.layout(&request).unwrap();
        // The step's row tables are built from the request: a prompt longer than the layout's text
        // capacity is refused there.
        let frames = frame_count(layout.video.frames).unwrap();
        let canvas = canvas_of_latents(&layout).unwrap();
        let tables =
            |text_tokens| packing::dense_tables(&packing, frames, &canvas, text_tokens, &[], &[]);
        assert!(tables(layout.text_rows).is_ok());
        assert!(tables(layout.text_rows + 1).is_err());
    }
}
