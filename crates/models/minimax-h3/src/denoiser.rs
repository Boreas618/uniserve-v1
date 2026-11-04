//! The denoiser's layout: what one request reserves, and the tables that describe it.
//!
//! A request is snapped to a legal frame count and a legal canvas, packed into the row order the
//! transformer reads (`[video | audio | prefix | padding]`, [`crate::packing`]), and evaluated once
//! per schedule point. This module owns what surrounds the evaluation: the layout the executor
//! captures against, and the tables that make a captured graph independent of the prompt.
//!
//! ## Layouts, not requests
//!
//! A graph is captured per **layout** - the frame count, canvas, text capacity and condition
//! capacity - and replayed for every request that fits it. So the layout snaps: frames to `17n + 5`
//! ([`crate::packing::align_num_frames`]), sides to the checkpoint's 32-pixel grid, and both prefix
//! capacities up to whole capacity units (64 rows under dense attention, the sparse tile
//! otherwise). Two requests of one layout share one set of tables, and the executor keys its
//! storage by `(layout, name)`.
//!
//! ## Host bytes
//!
//! Everything a request changes arrives as device tables the executor allocates: the rotary
//! coordinates, the modulation tags, the timestep groups, the prefix index of every packed row, the
//! AdaLN row each row reads, and the step index. [`Layouts::prepare_state`] writes their host bytes
//! in the order [`Layouts::state_tables`] names them; nothing of a request reaches a capture as a
//! host value, and no table is uploaded from inside a forward.

use nn::capability::{LatentShape, Layout, StateTable, VideoRequest};
use nn::{DType, Forward, Shape, TensorView, Weight, WeightLoader};

use crate::config::{Canvas, DenoiserConfig};
use crate::modulation::{AdaLnCache, Groups, Projections};
use crate::packing::{self, DensePacking};
use crate::schedule::{Modality, Schedule};
use crate::solver::Solver;
use crate::transformer::Transformer;
use crate::{Error, Result};

/// The modulation columns one AdaLN row holds: six affine vectors of three tags each.
const ADALN_COLUMNS: u64 = 18;

/// The row capacities and the sequence degree a deployment serves.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Arrangement {
    /// Text rows every layout of this deployment holds, a whole number of capacity units.
    pub text_rows: u32,
    /// Condition rows it holds, in capacity units.
    pub condition_rows: u32,
    /// Rows of one sequence shard: the Ulysses degree.
    pub shard: u32,
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
        })
    }
}

/// The deployment's layout rule: how a request snaps, and what its tables are.
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

    /// The arrangement this rule serves.
    pub const fn arrangement(&self) -> Arrangement {
        self.arrangement
    }

    /// The layout a request is served in, with the packing of its row extents.
    ///
    /// The canvas comes from the request's raster and the frames are snapped; the prefix capacities
    /// are the deployment's, because a layout holds every prompt that fits it.
    pub fn layout(&self, request: &VideoRequest) -> Result<(Layout, DensePacking)> {
        let frames = packing::align_num_frames(request.frames);
        let canvas = canvas_of(request)?;
        let packing = packing::dense_packing(
            frames,
            &canvas,
            self.arrangement.text_rows,
            self.arrangement.condition_rows,
            self.arrangement.shard,
        )?;
        let layout = self.layout_of(&packing, &canvas)?;
        Ok((layout, packing))
    }

    /// The layout of a packing and the canvas it packs.
    pub fn layout_of(&self, packing: &DensePacking, canvas: &Canvas) -> Result<Layout> {
        let raster = packing::latent_raster(canvas)?;
        let frames = packing::video_latent_frames(packing.video_rows / raster.patch_rows())?;
        Ok(Layout {
            video: LatentShape {
                channels: self.config.transformer.video_channels,
                frames,
                height: raster.height,
                width: raster.width,
            },
            audio: Some(LatentShape {
                channels: self.config.transformer.audio_channels,
                frames: packing.audio_rows / crate::config::AUDIO_CHANNELS,
                height: 1,
                width: 1,
            }),
            text_rows: packing.text_rows,
            condition_rows: packing.condition_rows,
        })
    }

    /// The packing a layout's extents came from.
    ///
    /// A layout states the latent extents and the two capacities; the packing is
    /// [`packing::dense_packing`]'s for the frame count and canvas those extents imply, and the two
    /// must agree - a disagreement is a layout this model did not build.
    pub fn packing_of(&self, layout: &Layout) -> Result<DensePacking> {
        let canvas = canvas_of_latents(layout)?;
        let frames = frame_count(layout.video.frames)?;
        let packing = packing::dense_packing(
            frames,
            &canvas,
            layout.text_rows,
            layout.condition_rows,
            self.arrangement.shard,
        )?;
        let rebuilt = self.layout_of(&packing, &canvas)?;
        if rebuilt.video.frames != layout.video.frames
            || rebuilt.video.height != layout.video.height
            || rebuilt.video.width != layout.video.width
            || rebuilt.text_rows != layout.text_rows
            || rebuilt.condition_rows != layout.condition_rows
        {
            return Err(Error::config(format!(
                "the layout's {:?} latent extents are not a frame count and canvas this model packs",
                layout.video
            )));
        }
        Ok(packing)
    }

    /// The tables one request of `layout` needs, in the order [`Layouts::prepare_state`] fills them.
    ///
    /// Sorted by name, so the order is a property of the layout alone and two requests of one shape
    /// share one set. Every table is one the forward reads; none is a host value the executor would
    /// otherwise have to pass through a launch.
    pub fn state_tables(&self, layout: &Layout) -> Result<Vec<StateTable>> {
        let packing = self.packing_of(layout)?;
        let rows = u64::from(packing.padded_tokens);
        let video = u64::from(packing.video_rows);
        let audio = u64::from(packing.audio_rows);
        let mut tables = vec![
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
            // query sees a padding row. `cu_seq_lens_k` is the dense visibility, `_q` the rows.
            StateTable {
                name: "cu_seq_lens_k",
                dtype: DType::I32,
                shape: Shape::new(&[rows + 1]),
            },
            StateTable {
                name: "cu_seq_lens_q",
                dtype: DType::I32,
                shape: Shape::new(&[rows + 1]),
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
            // per axis per token and refuses a leading extent that is not the axis count.
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
            // The step the schedule evaluates: written by the executor before each replay, one
            // value per (request, capture) rather than per row.
            StateTable {
                name: "step",
                dtype: DType::I32,
                shape: Shape::new(&[1]),
            },
        ];
        tables.sort_by_key(|table| table.name);
        Ok(tables)
    }

    /// Write one request's tables, in [`Layouts::state_tables`]' order.
    ///
    /// The bytes are host bytes in each table's own element type: the executor uploads them once
    /// per request, before the capture it will replay. `text_tokens` is the prompt's row count,
    /// which the media clock starts at; a prompt that does not fit the layout's capacity is
    /// refused here rather than captured.
    pub fn prepare_state(
        &self,
        request: &VideoRequest,
        text_tokens: u32,
        layout: &Layout,
        out: &mut [Vec<u8>],
    ) -> Result<()> {
        let packing = self.packing_of(layout)?;
        if text_tokens > layout.text_rows {
            return Err(Error::config(format!(
                "a prompt of {text_tokens} rows does not fit a text capacity of {}",
                layout.text_rows
            )));
        }
        let frames = frame_count(layout.video.frames)?;
        let canvas = canvas_of_latents(layout)?;
        let conditions = conditions_of(request);
        let segments = packing::condition_segments(&conditions, &canvas)?;
        let tables = packing::dense_tables(
            &packing,
            frames,
            &canvas,
            text_tokens,
            &segments,
            // The prompt's vision rows arrive with `VideoRequest`'s condition shapes; until then a
            // prompt is all text and every row takes the text clock.
            &[],
        )?;
        let rows = tables.groups.len();
        let video_rows = packing.video_rows as usize;
        let audio_rows = packing.audio_rows as usize;
        // The layout places the video rows first and the audio rows after them, so each modality's
        // scatter indices are its own contiguous range.
        let mut video_indices = Vec::with_capacity(video_rows);
        let mut audio_indices = Vec::with_capacity(audio_rows);
        for index in 0..video_rows {
            video_indices.push(index as i32);
        }
        for index in 0..audio_rows {
            audio_indices.push((video_rows + index) as i32);
        }
        // One sequence: every query row sees the used key rows and no query sees a padding row.
        let queries = [0, rows as i32];
        let keys = [0, tables.used as i32];
        let specs = self.state_tables(layout)?;
        for (name, bytes) in [
            ("audio_indices", integers(&audio_indices)),
            ("video_indices", integers(&video_indices)),
            ("cu_seq_lens_k", integers(&keys)),
            ("cu_seq_lens_q", integers(&queries)),
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
                    "the executor holds no buffer for {name}"
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
        // The step is the executor's to write through `prepare_evaluation`: one i32 a replay
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
        let segments = packing::condition_segments(conditions, &canvas)?;
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

/// The frame count a request is snapped to.
pub fn frames_of(request: &VideoRequest) -> u32 {
    packing::align_num_frames(request.frames)
}

/// The canvas a request renders, on the checkpoint's 32-pixel grid.
fn canvas_of(request: &VideoRequest) -> Result<Canvas> {
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
fn conditions_of(request: &VideoRequest) -> Vec<packing::Condition> {
    let _ = request;
    // The conditions' roles and rasters arrive with `VideoRequest`'s condition shapes
    // (nn::capability::ConditionShape); until they do, an unconditioned layout is the one this
    // build can pack, which is `t2va`.
    Vec::new()
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
/// is empty and [`Denoiser::prepare_constants`] writes the model's own region, so the executor
/// allocates nothing and no table is duplicated. It is one model's load-time fold, sized by that
/// model from its config alone and read only by it; a capability field carrying those views would
/// make every executor size a table it never reads. The launch stays capture-safe, writing a region
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
}

impl Denoiser {
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
        )?;
        let mut entry_bytes = Vec::with_capacity(table.len() * 4);
        for value in &table {
            entry_bytes.extend_from_slice(&value.to_le_bytes());
        }
        let entry_table = loader.constant(
            "minimax_h3.adaln.entries",
            DType::I32,
            Shape::new(&[u64::from(evaluations), u64::from(groups.count())]),
            &entry_bytes,
        )?;

        let mut adaln_layers = Vec::with_capacity(config.transformer.num_hidden_layers as usize);
        let zeros = vec![0_u8; (entries * columns * 4) as usize];
        for index in 0..config.transformer.num_hidden_layers {
            adaln_layers.push(loader.constant(
                &format!("minimax_h3.adaln.layers.{index}"),
                DType::F32,
                Shape::new(&[entries, columns]),
                &zeros,
            )?);
        }
        let adaln_output = loader.constant(
            "minimax_h3.adaln.output",
            DType::F32,
            Shape::new(&[entries, 2 * hidden]),
            &vec![0_u8; (entries * 2 * hidden * 4) as usize],
        )?;
        let cache = AdaLnCache::new(
            adaln_layers.clone(),
            adaln_output,
            entry_table,
            groups,
            config.transformer.hidden_size,
        )?;
        let transformer = Transformer::load(
            &config.transformer,
            config.attention,
            rows,
            partition,
            loader,
        )?;
        let solver = Solver::load(&schedule, loader)?;
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

    /// The solver that advances a sample.
    pub const fn solver(&self) -> &Solver {
        &self.solver
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
                "the H3 AdaLN cache is model memory; {} executor views were offered",
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
        };
        Layouts::new(
            config,
            Arrangement {
                text_rows: 128,
                condition_rows: 64,
                shard: 1,
            },
        )
        .unwrap()
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
        }
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
        // Four stereo rows per audio latent step.
        assert_eq!(packing.audio_rows, 4 * 179);
        assert_eq!(layout.audio.unwrap().frames, 179);
        assert_eq!(layout.text_rows, 128);
        assert_eq!(layout.condition_rows, 64);
    }

    #[test]
    fn the_tables_are_the_layout_s_and_the_step_is_last() {
        let rule = layouts();
        let (layout, packing) = rule.layout(&request(121, 1344, 768)).unwrap();
        let tables = rule.state_tables(&layout).unwrap();
        let names: Vec<_> = tables.iter().map(|table| table.name).collect();
        assert_eq!(
            names,
            vec![
                "audio_indices",
                "cu_seq_lens_k",
                "cu_seq_lens_q",
                "groups",
                "modulation_indices",
                "position_ids",
                "prefix_index",
                "step",
                "token_tags",
                "video_indices"
            ]
        );
        // Every table but the step covers one row per packed row.
        let rows = u64::from(packing.padded_tokens);
        for table in &tables {
            if table.name == "step" {
                assert_eq!(table.shape.dims(), &[1]);
                continue;
            }
            assert_eq!(table.shape.dim(0), rows, "{}", table.name);
        }
        // Axis-major: the rotation reads `[axis, token]`, so the axes are the leading extent.
        assert_eq!(tables[5].shape.dims(), &[3, rows]);
    }

    #[test]
    fn the_state_bytes_fill_the_tables_the_layout_names() {
        let rule = layouts();
        let request = request(121, 1344, 768);
        let (layout, _) = rule.layout(&request).unwrap();
        let tables = rule.state_tables(&layout).unwrap();
        let mut out: Vec<Vec<u8>> = tables
            .iter()
            .map(|table| {
                vec![0_u8; (table.shape.numel() as usize) * table.dtype.bytes().unwrap() as usize]
            })
            .collect();
        rule.prepare_state(&request, 12, &layout, &mut out).unwrap();
        // A prompt of 12 rows is text, so every write index points into the text rows and every
        // generated row takes the prefix's zero row.
        let prefix: Vec<i32> = decode_i32(&out[6]);
        assert_eq!(prefix.len(), tables[3].shape.dim(0) as usize);
        assert!(
            prefix[..12]
                .iter()
                .enumerate()
                .all(|(row, column)| *column == row as i32)
        );
        // The modulation index of the first text row is `3 * video_group + text`.
        let indices = decode_i32(&out[4]);
        assert_eq!(indices[0], 1);
    }

    #[test]
    fn a_raster_off_the_grid_snaps_up_to_it() {
        let rule = layouts();
        // 1340x770 is not a multiple of 32: the request renders the next legal raster, so two
        // requests either side of a multiple share one layout rather than each carrying its own.
        let (layout, _) = rule.layout(&request(121, 1340, 770)).unwrap();
        assert_eq!((layout.video.height, layout.video.width), (48, 84));
        assert_eq!(layout.video.height * 16, 768);
        assert_eq!(layout.video.width * 16, 1344);
    }

    #[test]
    fn a_layout_the_model_did_not_build_is_refused() {
        let rule = layouts();
        let (mut layout, _) = rule.layout(&request(121, 1344, 768)).unwrap();
        // 24 latent frames is not a frame count a request snaps to: no rounding produces 5n + 2 = 24.
        layout.video.frames = 24;
        assert!(rule.packing_of(&layout).is_err());
        // A prefix capacity that is not a whole attention unit is not this deployment's either.
        let (mut layout, _) = rule.layout(&request(121, 1344, 768)).unwrap();
        layout.text_rows = 100;
        assert!(rule.packing_of(&layout).is_err());
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
        // One condition of two latent frames at a 48x84 raster carries two stereo channels per
        // latent step: its rows are patched, and its element count weights each modality by its own
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
                num_frames: 2,
            }),
            audio_samples: 0,
        }];
        let one = rule.noise_offsets(&layout, &conditions).unwrap();
        // A condition's noise is its latent volume times the channels, not its packed rows: two
        // latent frames of 48x84 at 24 channels, and no audio.
        let expected = 2 * 48 * 84 * 24;
        assert_eq!(one.condition, expected);
        assert_eq!(one.video, one.condition);
        assert_eq!(one.audio, one.video + bare.audio);
    }

    #[test]
    fn a_prompt_wider_than_the_capacity_is_refused() {
        let rule = layouts();
        let request = request(121, 1344, 768);
        let (layout, _) = rule.layout(&request).unwrap();
        let tables = rule.state_tables(&layout).unwrap();
        let mut out: Vec<Vec<u8>> = tables
            .iter()
            .map(|table| {
                vec![0_u8; table.shape.numel() as usize * table.dtype.bytes().unwrap() as usize]
            })
            .collect();
        let refused = rule.prepare_state(&request, 129, &layout, &mut out);
        assert!(refused.is_err());
    }

    fn decode_i32(bytes: &[u8]) -> Vec<i32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }
}
