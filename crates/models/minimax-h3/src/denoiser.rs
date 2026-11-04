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
//! ## Host bytes
//!
//! Everything a request changes arrives as device tables the executor allocates: the rotary
//! coordinates, the modulation tags, the timestep groups, the prefix index of every packed row, the
//! AdaLN row each row reads, and the step index. [`Layouts::prepare_state`] writes their host bytes
//! in the order [`Layouts::state_tables`] names them; nothing of a request reaches a capture as a
//! host value, and no table is uploaded from inside a forward.

use nn::capability::{LatentShape, Layout, StateTable, VideoRequest};
use nn::{DType, Forward, Shape, TensorView, Weight, WeightLoader};

use crate::attention::{Factors, Visible};
use crate::config::{Canvas, DenoiserConfig};
use crate::modulation::{AdaLnCache, Groups, Projections};
use crate::packing::{self, DensePacking};
use crate::schedule::{Modality, Schedule};
use crate::solver::Solver;
use crate::transformer::Transformer;
use crate::{Error, Result};

/// The longest generated clip a deployment's bounds are taken over, in frames.
const MAX_FRAMES: u32 = 1024;

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

    /// The elements the request's condition draws occupy, in request order.
    ///
    /// A condition's noise is drawn at latent resolution - patched afterwards, so its packed rows
    /// overcount by four - and its audio rows are already channel-major elements. The count is
    /// computed from the conditions themselves, not from a packing, so it is total: every condition
    /// whose raster and frame count are ones the packing would accept contributes exactly its
    /// elements, and one it would refuse contributes none. A refused condition never reaches here -
    /// the runner lays the request out, and that packing is the same function of the same values -
    /// so a served request's offsets are exact.
    pub fn condition_elements(&self, conditions: &[packing::Condition], _canvas: &Canvas) -> u32 {
        let channels = u64::from(self.config.transformer.video_channels);
        let elements: u64 = conditions
            .iter()
            .map(|condition| {
                let visual = condition.frame.map_or(0, |frame| {
                    let frames = if frame.num_frames <= 1 {
                        1
                    } else {
                        packing::video_latent_frames(frame.num_frames).unwrap_or(0)
                    };
                    match packing::latent_raster(&frame.canvas) {
                        Ok(raster) => {
                            u64::from(frames)
                                * u64::from(raster.height)
                                * u64::from(raster.width)
                                * channels
                        }
                        Err(_) => 0,
                    }
                });
                let audio = u64::from(crate::config::AUDIO_CHANNELS)
                    * u64::from(condition.audio_samples.div_ceil(crate::config::AUDIO_HOP));
                visual + audio
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

    /// The segment packing of a sparse layout, rebuilt from the layout's own extents.
    ///
    /// The tile capacities a sparse layout holds are tile-rounded, so a rebuild from the layout's
    /// numbers must return the same extent: a mismatch is a layout this deployment did not pack,
    /// and the tables built from it would address rows that are not there.
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
    /// [`Layouts::layout`] does and takes the deployment's prefix capacities. A request whose
    /// packing is then refused is one the runner never asks for: the packing is the same function
    /// of the same values, and the runner must lay the request out before it captures anything.
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
            audio: Some(LatentShape {
                channels: self.config.transformer.audio_channels,
                frames: packing::audio_latent_frames(frames),
                height: 1,
                width: 1,
            }),
            text_rows: self.arrangement.text_rows,
            condition_rows: self.arrangement.condition_rows,
            condition_elements: self.condition_elements(&conditions_of(request), &canvas),
        }
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
            audio: Some(LatentShape {
                channels: self.config.transformer.audio_channels,
                frames: packing.audio_rows / crate::config::AUDIO_CHANNELS,
                height: 1,
                width: 1,
            }),
            text_rows: packing.text_rows,
            condition_rows: packing.condition_rows,
            condition_elements,
        })
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

    /// The tables one request of `layout` needs, in the order [`Layouts::prepare_state`] fills them.
    ///
    /// Sorted by name, so the order is a property of the layout alone and two requests of one shape
    /// share one set. Every table is one the forward reads; none is a host value the executor would
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
        let mut tables = vec![
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
        ];
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
        let segments = packing::condition_segments(&conditions, &canvas)?;
        let tables = packing::segment_tables(
            &packing,
            frames,
            &canvas,
            request.text_tokens,
            &segments,
            // The prompt's vision rows arrive with `VideoRequest`'s condition shapes.
            &[],
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
        let audio: Vec<i32> = packing.audio_indices().map(|row| row as i32).collect();
        let video: Vec<i32> = packing
            .video_indices
            .iter()
            .map(|row| *row as i32)
            .collect();
        let filled: [(&str, Vec<u8>); 10] = [
            ("audio_indices", integers(&audio)),
            ("video_indices", integers(&video)),
            ("groups", integers(&tables.groups)),
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
                .ok_or_else(|| Error::config(format!("the executor holds no buffer for {name}")))?;
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
            .ok_or_else(|| Error::config("the executor holds no block-size buffer".to_owned()))?;
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

    /// Write one request's tables, in [`Layouts::state_tables`]' order.
    ///
    /// The bytes are host bytes in each table's own element type: the executor uploads them once
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
        let specs = self.table_specs(layout)?;
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

/// The audio rows a layout holds: one element per channel and latent step.
fn audio_rows(layout: Layout) -> Result<u64> {
    let audio = layout
        .audio
        .ok_or_else(|| Error::config("a layout without audio has no audio stream".to_owned()))?;
    Ok(u64::from(audio.channels) * u64::from(audio.frames))
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
    /// This rank's position in the sequence shard and the shard's degree.
    rank: u32,
    shard: u32,
}

/// A model error as the capability's.
fn flattened(error: Error) -> nn::Error {
    nn::Error::Weight {
        tensor: "minimax-h3".to_owned(),
        problem: error.to_string(),
    }
}

/// The capability surface the executor binds: layouts, tables, and the step.
///
/// Every method delegates to the inherent one, so the model's own API stays the one the tests and
/// the runner's documentation read.
impl nn::capability::Denoiser for Denoiser {
    fn layout(&self, request: &VideoRequest<'_>) -> Layout {
        self.rules.cover(request)
    }

    fn draw_noise(
        &self,
        f: &Forward,
        seed: nn::capability::Seed,
        layout: Layout,
        out: nn::capability::Latents,
    ) -> nn::Result<()> {
        // One stream per request, addressed by element: the conditions' own draws first, then the
        // video latents, then the audio latents. A shard's first row is not element zero, so each
        // modality's offset carries the rank's global first row.
        let channels = u64::from(layout.video.channels);
        // One packed row is a 2x2 patch of one latent frame, so it covers four voxels' channels.
        let row_elements = channels * 4;
        let video_rows = u64::from(layout.video.frames)
            * u64::from(layout.video.height / 2)
            * u64::from(layout.video.width / 2);
        let condition = u64::from(layout.condition_elements);
        let video_offset =
            condition + self.first_row(video_rows).map_err(flattened)? * row_elements;
        let audio_offset = condition
            + video_rows * row_elements
            + self
                .first_row(audio_rows(layout).map_err(flattened)?)
                .map_err(flattened)?;
        let noise = nn::Noise::new();
        noise.draw(
            f,
            out.video,
            nn::NoiseKey::Scalar(seed.0),
            0,
            0,
            video_offset,
        )?;
        if let Some(audio) = out.audio {
            noise.draw(f, audio, nn::NoiseKey::Scalar(seed.0), 0, 1, audio_offset)?;
        }
        Ok(())
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

    fn layout_tables(&self, _layout: &Layout) -> Vec<StateTable> {
        // Every table this denoiser reads is a request's: the layout's are its packing's, which the
        // model derives rather than stores.
        Vec::new()
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
        // executor writes before each replay. A buffer offered here means the caller and
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
const DENSE_TABLES: [&str; 10] = [
    "audio_indices",
    "cu_seq_lens_k",
    "cu_seq_lens_q",
    "groups",
    "modulation_indices",
    "position_ids",
    "prefix_index",
    "step",
    "token_tags",
    "video_indices",
];

/// The state tables of a segment layout, in `segment_table_specs`' order (sorted by name).
///
/// A segment layout has no `cu_seq_lens` pair and carries four tables a dense one does not, so the
/// two sets do not share positions: a name resolved against the wrong set silently reads another
/// tensor. [`Tables`] picks the set from the deployment's attention kind for exactly that reason.
const SEGMENT_TABLES: [&str; 12] = [
    "audio_indices",
    "groups",
    "modulation_indices",
    "position_ids",
    "prefix_index",
    "segment_keep",
    "segment_starts",
    "step",
    "tile_segments",
    "token_tags",
    "variable_block_sizes",
    "video_indices",
];

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
            .ok_or_else(|| Error::config(format!("{name} is not a state table of this kind")))?;
        self.views
            .get(position)
            .copied()
            .ok_or_else(|| Error::config(format!("the executor passed no {name} table")))
    }
}

impl Denoiser {
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
        let tables = Tables::new(self.config.attention, cond.state);
        let video = latents.video;
        let channels = video.shape.dim(0);
        let frames = video.shape.dim(1);
        let height = video.shape.dim(2);
        let width = video.shape.dim(3);
        let tokens = u64::from(tables.get("groups")?.shape.dim(0) as u32);
        let hidden_size = u64::from(self.config.transformer.hidden_size);
        let dtype = cond.text.dtype;
        let arena = f.arena();

        // The packed rows: every row gathers its prefix source, so a generated row reads the zero
        // row and a text row reads itself.
        let hidden = arena.tensor(dtype, Shape::new(&[tokens, hidden_size]))?;
        nn::layers::gather_rows(f, cond.text, tables.get("prefix_index")?, hidden)?;

        // The video rows: patchify, project, scatter.
        let video_indices = tables.get("video_indices")?;
        let video_rows = u64::from(video_indices.shape.dim(0) as u32);
        let patch = nn::layers::Patch::new(channels, height, width)?;
        // The one inverse of the pack above: the DiT's `1x2x2` tiles back into the latent raster.
        // It reads the token grid from `out`, so a clip of another length replays the same launch.
        let unpatch = nn::layers::UnpatchifyVideo::new(dtype, channels, 1, 2)?;
        let patched = arena.tensor(dtype, Shape::new(&[video_rows, channels * 4]))?;
        patch.forward(f, video, patched, frames)?;
        let projected = arena.tensor(dtype, Shape::new(&[video_rows, hidden_size]))?;
        self.transformer.project(f, patched, projected)?;
        nn::layers::Rows::new(dtype, hidden_size)?.replace(f, hidden, projected, video_indices)?;

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
            let audio_rows = u64::from(audio_indices.shape.dim(0) as u32);
            let features = u64::from(self.config.transformer.audio_channels);
            let flat = TensorView::with_strides(
                audio.ptr,
                audio.dtype,
                Shape::new(&[audio_rows, features]),
                nn::Strides::new(&[features, 1]),
            );
            let projected = arena.tensor(dtype, Shape::new(&[audio_rows, hidden_size]))?;
            self.transformer.project_audio(f, flat, projected)?;
            nn::layers::Rows::new(dtype, hidden_size)?.replace(
                f,
                hidden,
                projected,
                audio_indices,
            )?;
        }

        // The rotary tables and the visibility of the one packed sequence. Each of the three axes
        // rotates `rope_frequency_dim` channel pairs, and a factor table holds one column per pair.
        let rotated = u64::from(self.config.transformer.rope_frequency_dim) * 3;
        let cosine = arena.tensor(DType::F32, Shape::new(&[tokens, rotated]))?;
        let sine = arena.tensor(DType::F32, Shape::new(&[tokens, rotated]))?;
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
        // and the keep table.
        let segments = self
            .config
            .attention
            .is_dense()
            .then_some(())
            .map_or(Ok(None), |()| {
                Ok::<_, Error>(Some(crate::attention_sparse::Segments {
                    sizes: tables.get("variable_block_sizes")?,
                    segments: tables.get("tile_segments")?,
                    keep: tables.get("segment_keep")?,
                }))
            })?;

        // The stack, conditioned by the step's gathered modulation.
        let step_index = tables.get("step")?;
        self.transformer.forward(
            f,
            &self.cache,
            step_index,
            hidden,
            tables.get("modulation_indices")?,
            Factors { cosine, sine },
            visible.as_ref(),
            segments.as_ref(),
            tokens as u32,
        )?;

        // The output: the step's final modulation, then each modality's head.
        let groups = self.cache.groups().count() as u64;
        let entries = arena.tensor(DType::I32, Shape::new(&[1, groups]))?;
        self.cache.entries_of(f, step_index, entries)?;
        let modulation = arena.tensor(DType::F32, Shape::new(&[groups, 2 * hidden_size]))?;
        self.cache.output(f, entries, modulation)?;

        let video_hidden = arena.tensor(dtype, Shape::new(&[video_rows, hidden_size]))?;
        nn::layers::gather_rows(f, hidden, video_indices, video_hidden)?;
        let video_groups = arena.tensor(DType::I32, Shape::new(&[video_rows]))?;
        nn::layers::gather_rows(f, tables.get("groups")?, video_indices, video_groups)?;
        let velocity_rows = arena.tensor(dtype, Shape::new(&[video_rows, channels * 4]))?;
        self.transformer.velocity(
            f,
            video_hidden,
            Modality::Video,
            modulation,
            video_groups,
            velocity_rows,
        )?;
        let velocity = arena.tensor(dtype, video.shape)?;
        unpatch.forward(f, velocity_rows, velocity)?;
        self.solver
            .step(f, Modality::Video, video, velocity, step.index)?;

        if let Some((audio, audio_indices)) = audio {
            let audio_rows = u64::from(audio_indices.shape.dim(0) as u32);
            let audio_hidden = arena.tensor(dtype, Shape::new(&[audio_rows, hidden_size]))?;
            nn::layers::gather_rows(f, hidden, audio_indices, audio_hidden)?;
            let audio_groups = arena.tensor(DType::I32, Shape::new(&[audio_rows]))?;
            nn::layers::gather_rows(f, tables.get("groups")?, audio_indices, audio_groups)?;
            let features = u64::from(self.config.transformer.audio_channels);
            let velocity = arena.tensor(dtype, Shape::new(&[audio_rows, features]))?;
            self.transformer.velocity(
                f,
                audio_hidden,
                Modality::Audio,
                modulation,
                audio_groups,
                velocity,
            )?;
            let flat = TensorView::with_strides(
                audio.ptr,
                audio.dtype,
                Shape::new(&[audio_rows, features]),
                nn::Strides::new(&[features, 1]),
            );
            self.solver
                .step(f, Modality::Audio, flat, velocity, step.index)?;
        }
        Ok(())
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
            rank: partition.tensor_rank as u32,
            shard: partition.ulysses_degree as u32,
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

    /// The first row of this rank's shard, in the layout's global row order.
    ///
    /// A shard is a contiguous block of rows, so a rank's elements begin at its first row's.
    fn first_row(&self, rows: u64) -> Result<u64> {
        let shard = u64::from(self.shard.max(1));
        let rank = u64::from(self.rank);
        if rank >= shard {
            return Err(Error::config(format!(
                "rank {rank} is outside a shard of {shard}"
            )));
        }
        Ok(rank * rows / shard)
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

    const TILE: u32 = 64;

    fn sparse_layouts() -> Layouts {
        let mut config = layouts().config;
        config.attention = AttentionKind::sparse(TILE, 0.8, None).unwrap();
        Layouts::new(config, layouts().arrangement).unwrap()
    }

    #[test]
    fn a_sparse_layout_declares_its_tile_tables_and_fills_them() {
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
        assert_eq!(packed.extent.condition_rows, 64);
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
        assert_eq!(layout.condition_rows, 64);
    }

    #[test]
    fn the_tables_are_the_layout_s_and_the_step_is_last() {
        let rule = layouts();
        let (layout, packing) = rule.layout(&request(121, 1344, 768)).unwrap();
        let tables = rule.table_specs(&layout).unwrap();
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
            match table.name {
                "step" => {
                    assert_eq!(table.shape.dims(), &[1]);
                }
                // The audio table is addressed by audio row and names the packed row it occupies,
                // so its length is the audio rows rather than every packed row.
                "audio_indices" => assert_eq!(table.shape.dim(0), u64::from(packing.audio_rows)),
                // The dense visibility table names both ends of the used row run, so it carries one
                // more entry than there are packed rows.
                "cu_seq_lens_k" | "cu_seq_lens_q" => assert_eq!(table.shape.dims(), &[2]),
                // The rotation reads the coordinates axis-major, so the axes lead.
                "position_ids" => assert_eq!(table.shape.dims(), &[3, rows]),
                // The video indices name the packed row of every video row, and video rows are
                // patched: four voxels to a row.
                "video_indices" => {
                    assert_eq!(table.shape.dim(0), u64::from(packing.video_rows));
                }
                _ => assert_eq!(table.shape.dim(0), rows, "{}", table.name),
            }
        }
        // Axis-major: the rotation reads `[axis, token]`, so the axes are the leading extent.
        assert_eq!(tables[5].shape.dims(), &[3, rows]);
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
        assert_eq!(prefix.len(), tables[3].shape.dim(0) as usize);
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
        let segments = packing::condition_segments(&conditions, &canvas).unwrap();
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
        // The names are positional: the executor passes its views in `table_specs`' order, so a set
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
