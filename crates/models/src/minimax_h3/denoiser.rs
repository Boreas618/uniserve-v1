//! The denoiser: one network of the checkpoint with its schedule, its conditioning and the
//! solver step, as execution serves it ([`nn::Denoiser`]).
//!
//! A request is laid out ([`crate::minimax_h3::packing`]), its conditioning is prepared once, and each
//! schedule point evaluates the network once and advances both latents:
//!
//! ```text
//! prepare_conditioning   cond.text[text rows]      = refiner(cond.source)
//!                        cond.text[condition rows] = proj_in(anchor(encode(media)))
//! forward                rows     = [cond.text | audio_proj_in(audio) | proj_in(patchify(video))]
//!                        hidden   = rows[the layout's row of each packed row]
//!                        hidden   = transformer(hidden)
//!                        video   += unpatchify(velocity rows) · (sigma − sigma_next)
//!                        audio   += velocity rows · (sigma − sigma_next)
//! ```
//!
//! ## Layouts, not requests
//!
//! A graph is captured per layout (the frame count, the resolution, the text capacity and the
//! condition rows) and replayed for every request of it. Everything a request changes inside a
//! layout (the rotary coordinates, each row's modulation row, the conditioning row each packed
//! row gathers, the rows a query attends) is computed in the forward as CPU tensors and moved
//! to the device; their shapes follow from the layout alone.
//!
//! ## The trajectory is FP32
//!
//! The sample a trajectory accumulates and its noise are FP32 ([`SAMPLE_DTYPE`]), not the
//! stack's type: the reference keeps its trajectory in FP32 and rounds only inside the
//! network, so a sample crosses to the stack's type once after its input projection and back
//! once before the output heads ([`crate::minimax_h3::transformer`]). A latent of another type is refused.
//!
//! ## Sequence parallelism
//!
//! A rank of a sequence split computes one contiguous interval of the packed rows. Every rank
//! holds the whole sample and draws the whole noise stream, projects the modality rows and
//! gathers its own interval of them; after the stack the ranks' velocity rows are gathered in
//! rank order, so every rank advances the same latents.

use nn::diffusion::{add_noise, euler_step, scale_noise};
use nn::family::SequenceSplit;
use nn::modules::{TimestepTable, TimestepTableConfig};
use nn::{
    ConditionPiece, Conditioning, DType, Device, MediaInput, Shape, Split, Step, Tensor,
    VarBuilder, VideoRequest, functional,
};
use proto::video::Layout;

use crate::minimax_h3::audio_vae::{AudioEncoder, padded_samples};
use crate::minimax_h3::config::{
    AUDIO_CHANNELS, DenoiserConfig, FPS, FRAMES, MAX_FRAMES, VsaConfig,
};
use crate::minimax_h3::modulation::{self, Modulation, VISUAL_CONDITION_TIME};
use crate::minimax_h3::refiner::TokenRefiner;
use crate::minimax_h3::transformer::Transformer;
use crate::minimax_h3::video_vae::VideoEncoder;
use crate::minimax_h3::{Error, Result, VideoTask, packing};

/// The element type of the sample a trajectory accumulates, and of its noise.
pub const SAMPLE_DTYPE: DType = DType::F32;

/// The channels of a condition's pixels: RGB.
const PIXEL_CHANNELS: usize = 3;

/// The least rung of a deployment's condition ladder: the rows of one keyframe of the released
/// resolutions, which is the smallest condition a request brings.
const LEAST_CONDITION_RUNG: u32 = 1024;

/// The task a request runs, from the conditions it states.
fn task_of(request: &VideoRequest) -> VideoTask {
    VideoTask::of(
        request.first_frames,
        request.last_frames,
        request.references,
    )
}

/// A host index table on `like`'s device, uploaded as its own elements from the first byte of
/// its block.
///
/// A captured step's kernels read each table at the address its capture gave it, and a replay
/// writes there the bytes the step's forward computes for the request it serves
/// (`ops::input`). A view moved as it is carries its whole storage and keeps its first element's
/// offset in that address, and a view whose offset follows from the request (a modality's rows
/// of the document, after the prompt's) would be read at the offset of the request the bucket
/// was captured from.
fn on_device(table: &Tensor, like: &Tensor) -> Result<Tensor> {
    Ok(table.force_contiguous()?.to_device(like.device())?)
}

/// One checkpoint's denoiser: the network, the refiner of its prompt, its timestep modulation
/// at the schedule's levels and the condition encoders a conditioned deployment places.
pub struct Denoiser {
    config: DenoiserConfig,
    /// The text rows every layout holds: the deployment's longest prompt.
    text_rows: u32,
    /// The condition rows a layout holds at most; a request rides the rung of the condition
    /// ladder that holds its own.
    condition_rows: u32,
    /// This rank's place in the sequence split.
    sequence: Split,
    /// The deployment's bound on the packed rows of a layout: the key bound the attention was
    /// loaded for.
    max_rows: u32,
    transformer: Transformer,
    refiner: TokenRefiner,
    modulation: Modulation,
    /// The modulation at each distinct level of the schedule, which [`Modulation::fill`] writes
    /// once after the load.
    table: TimestepTable,
    /// The video and the audio condition encoder, when the deployment places them.
    encoders: Option<(VideoEncoder, AudioEncoder)>,
}

impl Denoiser {
    /// The denoiser of `config` at `vb` (the network's checkpoint root), for layouts of
    /// `capacity`'s `(text rows, condition rows)` at `sequence`'s rank, with the condition
    /// encoders when the deployment places them.
    ///
    /// # Errors
    ///
    /// A network without heads, a sequence split that is not a rank of a power of two, a
    /// capacity that is not whole capacity units, or a weight the loader refuses.
    pub fn load(
        config: &DenoiserConfig,
        capacity: (u32, u32),
        sequence: SequenceSplit,
        encoders: Option<(VideoEncoder, AudioEncoder)>,
        vb: VarBuilder<'_, '_>,
    ) -> Result<Self> {
        let transformer = &config.transformer;
        let split = sequence.split;
        let (text_rows, condition_rows) = capacity;

        if transformer.head_dim == 0 || transformer.num_attention_heads == 0 {
            return Err(Error::config(
                "a denoiser's transformer has attention heads".to_owned(),
            ));
        }
        if split.size == 0 || !split.size.is_power_of_two() || split.rank >= split.size {
            return Err(Error::config(format!(
                "rank {} of a sequence split of {} is not a rank of a power-of-two split",
                split.rank, split.size
            )));
        }
        let unit = config.capacity_unit();
        for (name, rows) in [("text", text_rows), ("condition", condition_rows)] {
            if rows == 0 || !rows.is_multiple_of(unit) {
                return Err(Error::config(format!(
                    "a {name} capacity of {rows} rows is not a multiple of the {unit}-row unit"
                )));
            }
        }

        let max_rows = max_rows(config, capacity, split)?;
        let modulation = Modulation::load(transformer, vb.clone())?;

        // The tables take the modulation projections' own type, as the reference allocates
        // them from the product.
        let hidden = transformer.hidden_size;
        let levels = modulation::levels(&config.schedule, config.conditioned())?;
        let table = TimestepTable::load(
            &TimestepTableConfig {
                frequency_dim: transformer.frequency_dim,
                layers: transformer.num_hidden_layers,
                layer_columns: modulation::BLOCK_COLUMNS * hidden,
                output_columns: modulation::OUTPUT_COLUMNS * hidden,
                dtype: modulation.dtype(),
            },
            &levels,
            vb.pp("adaln"),
        )?;

        Ok(Self {
            transformer: Transformer::load(config, sequence, max_rows, vb.clone())?,
            refiner: TokenRefiner::load(transformer, text_rows, vb)?,
            modulation,
            table,
            encoders,
            config: config.clone(),
            text_rows,
            condition_rows,
            sequence: split,
            max_rows,
        })
    }

    /// The configuration this denoiser was loaded with.
    pub const fn config(&self) -> &DenoiserConfig {
        &self.config
    }

    /// The text rows every layout of this deployment holds.
    pub const fn text_rows(&self) -> u32 {
        self.text_rows
    }

    /// The condition rows a layout of this deployment holds at most.
    pub const fn condition_capacity(&self) -> u32 {
        self.condition_rows
    }

    /// Whether this deployment places the condition encoders: a conditioned request on a
    /// deployment that places none is refused where its conditions are prepared.
    pub const fn places_encoders(&self) -> bool {
        self.encoders.is_some()
    }

    /// The condition rows a layout of this deployment may hold, ascending and ending at its
    /// capacity: one captured graph per rung, and a request rides the smallest rung that holds
    /// its conditions.
    ///
    /// The rungs start at one keyframe and double, with a rung halfway between two doublings
    /// from four keyframes up, where a request's empty rows start to cost more than their
    /// graphs. Each is whole capacity units, so a layout of any rung packs.
    pub fn condition_ladder(&self) -> Vec<u32> {
        let capacity = self.condition_rows;
        let unit = self.config.capacity_unit();

        let mut rungs = Vec::new();
        let mut doubling = LEAST_CONDITION_RUNG;
        while doubling < capacity {
            rungs.push(doubling.next_multiple_of(unit));
            if doubling >= 4 * LEAST_CONDITION_RUNG && doubling + doubling / 2 < capacity {
                rungs.push((doubling + doubling / 2).next_multiple_of(unit));
            }
            doubling *= 2;
        }
        rungs.push(capacity);
        rungs
    }

    /// The rows of `cond.text` the condition rows of a request start at, checked against the
    /// rows its pieces take: the layout's text capacity, with the condition rows and the row
    /// of zeros after it.
    fn condition_start(&self, text: &Tensor, media: &[MediaInput]) -> Result<usize> {
        let total = text.dims().first().copied().unwrap_or(0);
        let text_rows = self.text_rows as usize;
        let taken: usize = media
            .iter()
            .map(|piece| (piece.rows + piece.audio_rows) as usize)
            .sum();

        // The text capacity, the condition rows and the row of zeros are the whole tensor.
        if total < text_rows + taken + 1 {
            return Err(Error::request(format!(
                "the request's conditions take {taken} rows and a conditioning of {total} rows \
                 holds {text_rows} text rows and the row of zeros"
            )));
        }
        Ok(text_rows)
    }

    /// Encodes the request's raw media and writes its condition rows into `text`, in place.
    ///
    /// The rows are the layout's: a piece states the rows the packing counted for it, so this
    /// places each block after the text capacity in packed order, the audio rows of a piece
    /// before its visual rows, and checks the encoders against the counts.
    fn encode_media(&self, media: &[MediaInput], text: &Tensor) -> Result<()> {
        let Some((video_encoder, audio_encoder)) = &self.encoders else {
            return Err(Error::request(
                "a request carries conditions and this deployment places no condition encoder"
                    .to_owned(),
            ));
        };
        let transformer = &self.config.transformer;
        let mut start = self.condition_start(text, media)?;

        for piece in media {
            if piece.audio_rows > 0 {
                let Some(samples) = &piece.audio else {
                    return Err(Error::request(format!(
                        "condition {:?} states {} audio rows and no samples",
                        piece.role, piece.audio_rows
                    )));
                };
                let rows = piece.audio_rows as usize;

                // The encoder writes channel-major latent rows in FP32, which the input
                // projection reads; the one rounding is the store into the conditioning.
                let encoded = audio_encoder.encode(samples)?;
                if encoded.dims() != [rows, transformer.audio_channels as usize] {
                    return Err(Error::request(format!(
                        "condition {:?} states {rows} audio rows and its track encodes to {:?}",
                        piece.role,
                        encoded.dims()
                    )));
                }
                let projected = self.transformer.project_audio(&encoded)?;

                text.slice_set(&projected.to_dtype(text.dtype())?, 0, start)?;
                start += rows;
            }

            if piece.rows > 0 {
                let rows = self.encode_visual(piece, video_encoder)?;

                text.slice_set(&rows.to_dtype(text.dtype())?, 0, start)?;
                start += piece.rows as usize;
            }
        }
        Ok(())
    }

    /// One visual condition's projected rows, `[rows, hidden]` in FP32.
    ///
    /// The encoder returns the condition's normalized latent; the latent and the piece's draw
    /// become rows by the same patch mapping (the draw is drawn at latent resolution), the
    /// anchor `t · clean + (1 − t) · draw` mixes them in FP32 at [`VISUAL_CONDITION_TIME`]
    /// (`refs/vllm-omni/…/minimax_h3/pipeline_minimax_h3.py:2238-2243`), and the input
    /// projection reads the anchored rows.
    fn encode_visual(&self, piece: &MediaInput, encoder: &VideoEncoder) -> Result<Tensor> {
        let (Some(pixels), Some(draw)) = (&piece.pixels, &piece.noise) else {
            return Err(Error::request(format!(
                "condition {:?} states {} visual rows without its pixels and its draw",
                piece.role, piece.rows
            )));
        };
        if pixels.dtype() != DType::F32 || draw.dtype() != SAMPLE_DTYPE {
            return Err(Error::request(format!(
                "condition {:?} is encoded in FP32: its pixels are {:?} and its draw {:?}",
                piece.role,
                pixels.dtype(),
                draw.dtype()
            )));
        }

        let latent = encoder.encode(pixels)?;
        let clean = Transformer::patchify(&latent)?;
        let anchor = Transformer::patchify(draw)?;
        if clean.dims()[0] != piece.rows as usize || anchor.dims() != clean.dims() {
            return Err(Error::request(format!(
                "condition {:?} states {} visual rows, encodes to {:?} and draws {:?}",
                piece.role,
                piece.rows,
                clean.dims(),
                anchor.dims()
            )));
        }

        let anchored = add_noise(&clean, &anchor, VISUAL_CONDITION_TIME)?;
        self.transformer.project_video(&anchored)
    }

    /// The step of the audio trajectory that `step` of the video trajectory runs beside.
    fn audio_step(&self, step: Step) -> Result<Step> {
        let sigmas = &self.config.schedule.audio_sigmas;
        let index = step.index as usize;

        match (sigmas.get(index), sigmas.get(index + 1)) {
            (Some(sigma), Some(sigma_next)) => Ok(Step {
                sigma: *sigma,
                sigma_next: *sigma_next,
                ..step
            }),
            _ => Err(Error::request(format!(
                "evaluation {index} is outside the {} the schedule has",
                self.config.schedule.num_steps()
            ))),
        }
    }

    /// The layout's attention tables on `like`'s device ([`crate::minimax_h3::transformer::Attention`]):
    /// the query and key offsets of a dense layout, or the tile tables of a tiled one.
    fn attention_tables(
        &self,
        request: &VideoRequest,
        layout: &Layout,
        rows: &Tensor,
        like: &Tensor,
    ) -> Result<Vec<Tensor>> {
        let Some(vsa) = &self.config.vsa else {
            // One sequence: every row is a query and the used rows are the keys, so no query
            // attends a padding row.
            let offsets = |end: usize| Tensor::new(&[0_u32, end as u32], &Device::Cpu);

            return Ok(vec![
                on_device(&offsets(rows.elem_count())?, like)?,
                on_device(&offsets(packing::dense_used(request, layout))?, like)?,
            ]);
        };

        let table = packing::tile_table(rows, request, layout, vsa)?;
        Ok(vec![
            on_device(&table.get(0)?.to_dtype(DType::U32)?, like)?,
            on_device(&table.get(1)?.to_dtype(DType::I32)?, like)?,
            on_device(&table.get(2)?.to_dtype(DType::U32)?, like)?,
        ])
    }
}

/// The layout row count the deployment's attention is loaded for: the largest of any layout
/// of `capacity` over every resolution the checkpoint generates and every frame count of the
/// frame rule up to [`MAX_FRAMES`].
///
/// The bound is taken over the checkpoint's resolutions and frame counts and not the
/// deployment's sizes and longest output, so the kernels keyed by it are one set for every
/// deployment of a checkpoint and capacity.
fn max_rows(config: &DenoiserConfig, capacity: (u32, u32), sequence: Split) -> Result<u32> {
    let resolutions = processing::minimax_h3::named_canvases()
        .map_err(|error| Error::config(format!("the resolution rule: {error}")))?;

    let mut bound = 0;
    for resolution in &resolutions {
        for frames in (FRAMES.min_frames..=MAX_FRAMES).step_by(FRAMES.clip_frames as usize) {
            let request = VideoRequest {
                width: resolution.canvas.width,
                height: resolution.canvas.height,
                frames,
                fps: FPS,
                first_frames: 0,
                last_frames: 0,
                references: 0,
                text_tokens: 1,
                conditions: Vec::new(),
                vision_spans: Vec::new(),
                condition_rows: None,
            };
            let layout = packing::layout(&request, &config.transformer, capacity.0, capacity.1)?;

            bound = bound.max(layout_rows(&layout, config.vsa.as_ref(), sequence));
        }
    }
    Ok(bound as u32)
}

/// The rows of `layout`'s packed sequence: the tiled layout's under sparse attention, the
/// dense layout's aligned to the sequence split otherwise.
fn layout_rows(layout: &Layout, vsa: Option<&VsaConfig>, sequence: Split) -> usize {
    match vsa {
        Some(vsa) => packing::tiled_len(layout, vsa),
        None => packing::dense_len(layout, sequence.size),
    }
}

impl nn::Denoiser for Denoiser {
    /// The layout a request is served in: its frames and sides snapped, the deployment's text
    /// capacity, and the condition rows of the rung the request states (the deployment's
    /// capacity when it states none, and none for a request without conditions, whose rows
    /// would be empty rows of every projection and attention of every step).
    fn layout(&self, request: &VideoRequest) -> nn::Result<Layout> {
        let condition_rows = match task_of(request) {
            VideoTask::TextToVideoAudio => 0,
            _ => request.condition_rows.unwrap_or(self.condition_rows),
        };

        Ok(packing::layout(
            request,
            &self.config.transformer,
            self.text_rows,
            condition_rows,
        )?)
    }

    /// The steps of the checkpoint's schedule, for a task the network serves.
    ///
    /// The task is a key: every H3 task shares one grid, so the numbers alone cannot tell a
    /// task the checkpoint does not serve, and it is refused by name.
    fn schedule(&self, request: &VideoRequest) -> nn::Result<Vec<Step>> {
        let task = task_of(request);
        if !self.config.tasks.contains(&task) {
            return Err(Error::request(format!(
                "this checkpoint serves {:?} and not {task:?}",
                self.config.tasks
            ))
            .into());
        }
        Ok(self.config.schedule.steps())
    }

    /// The stack's width: the refiner projects the encoder's features into it, so the rows the
    /// denoiser reads are wider than the rows the text encoder writes.
    fn text_width(&self) -> Option<u32> {
        Some(self.config.transformer.hidden_size)
    }

    /// One stream per request, addressed by element: the conditions' draws lead it, then the
    /// video latent in its own `[channels, frames, height, width]` order, then the audio
    /// latent. Every rank of a sequence split draws the whole stream.
    fn draw_noise(
        &self,
        seed: u64,
        layout: &Layout,
        video: &Tensor,
        audio: Option<&Tensor>,
    ) -> nn::Result<()> {
        let shape = layout.video;
        let dims = [shape.channels, shape.frames, shape.height, shape.width].map(|d| d as usize);
        if video.dims() != dims || video.dtype() != SAMPLE_DTYPE {
            return Err(Error::request(format!(
                "the video latents are {:?} {:?} and this layout draws FP32 {dims:?}",
                video.dtype(),
                video.dims()
            ))
            .into());
        }
        if layout.audio.is_some() != audio.is_some()
            || audio.is_some_and(|audio| audio.dtype() != SAMPLE_DTYPE)
        {
            return Err(Error::request(
                "a layout's audio stream is drawn into FP32 audio latents, and only into them"
                    .to_owned(),
            )
            .into());
        }

        let video_offset = u64::from(layout.condition_elements);
        functional::philox_normal(video, seed, 0, 0, video_offset)?;

        if let Some(audio) = audio {
            let audio_offset = video_offset + video.elem_count() as u64;
            functional::philox_normal(audio, seed, 0, 0, audio_offset)?;
        }
        Ok(())
    }

    /// The pieces of a request's conditions in packed order: keyframes, then references. A
    /// visual piece draws its anchoring noise at its own latent shape, the draws leading the
    /// request's stream in packed order; an audio reference draws none.
    fn condition_pieces(&self, request: &VideoRequest) -> nn::Result<Vec<ConditionPiece>> {
        let layout = self.layout(request)?;
        packing::check(request, &layout).map_err(nn::Error::from)?;
        let channels = self.config.transformer.video_channels;

        let mut offset = 0_u64;
        let mut pieces = Vec::with_capacity(request.conditions.len());
        for index in packing::condition_order(request) {
            let condition = &request.conditions[index];
            let (audio_rows, rows) = packing::condition_rows(condition);
            let draw = packing::condition_latent(condition, channels);

            let track = if condition.audio_samples > 0 {
                let Some((_, encoder)) = &self.encoders else {
                    return Err(Error::request(
                        "a request carries a soundtrack and this deployment places no condition \
                         encoder"
                            .to_owned(),
                    )
                    .into());
                };
                let padded = padded_samples(u64::from(condition.audio_samples), encoder.hop())
                    .map_err(nn::Error::from)?;

                Some(Shape::from((AUDIO_CHANNELS as usize, 1, padded as usize)))
            } else {
                None
            };

            pieces.push(ConditionPiece {
                condition: index as u32,
                role: condition.role,
                rows: rows as u32,
                audio_rows: audio_rows as u32,
                pixels: draw.map(|_| {
                    Shape::from((
                        PIXEL_CHANNELS,
                        condition.frames as usize,
                        condition.height as usize,
                        condition.width as usize,
                    ))
                }),
                track,
                draw,
                draw_offset: offset,
            });
            offset += draw.map_or(0, |draw| {
                u64::from(draw.channels)
                    * u64::from(draw.frames)
                    * u64::from(draw.height)
                    * u64::from(draw.width)
            });
        }
        Ok(pieces)
    }

    fn draw_condition_noise(
        &self,
        seed: u64,
        piece: &ConditionPiece,
        out: &Tensor,
    ) -> nn::Result<()> {
        functional::philox_normal(out, seed, 0, 0, piece.draw_offset)
    }

    /// The schedule's start: unit noise for the released and the distilled grids, the draw
    /// scaled to the first node's level for a parallel-decoding student.
    fn prepare_latents(
        &self,
        _cond: &Conditioning,
        video: &Tensor,
        audio: Option<&Tensor>,
    ) -> nn::Result<()> {
        let Some((video_sigma, audio_sigma)) = self.config.schedule.start_sigmas() else {
            return Ok(());
        };

        for (latents, sigma) in [(Some(video), video_sigma), (audio, audio_sigma)] {
            if let Some(latents) = latents
                && sigma != 1.0
            {
                latents.slice_set(&scale_noise(latents, sigma)?, 0, 0)?;
            }
        }
        Ok(())
    }

    /// Refines the prompt's features into the text rows of `cond.text` and projects the
    /// request's conditions into the rows after them, once per request: both are functions of
    /// the request alone, and every step's graph reads the rows they write.
    fn prepare_conditioning(&self, request: &VideoRequest, cond: &Conditioning) -> nn::Result<()> {
        if let Some(source) = &cond.source {
            let rows = self.text_rows as usize;
            let features = source.narrow(0, 0, rows)?.to_dtype(self.refiner.dtype())?;
            let refined = self
                .refiner
                .forward(&features, request.text_tokens)
                .map_err(nn::Error::from)?;

            cond.text
                .slice_set(&refined.to_dtype(cond.text.dtype())?, 0, 0)?;
        }

        if cond.media.is_empty() {
            return Ok(());
        }
        Ok(self.encode_media(&cond.media, &cond.text)?)
    }

    fn forward(
        &self,
        step: Step,
        request: &VideoRequest,
        video: &Tensor,
        audio: Option<&Tensor>,
        cond: &Conditioning,
    ) -> nn::Result<()> {
        Ok(self.step(step, request, video, audio, cond)?)
    }

    /// Fills the modulation tables from the schedule's levels, and a student's fused heads.
    fn initialize(&self) -> nn::Result<()> {
        self.modulation.fill(&self.table)?;
        Ok(self.transformer.initialize()?)
    }
}

impl Denoiser {
    /// One denoising step of `request`: the network's velocity of both latents, and the
    /// schedule's update of each in place.
    fn step(
        &self,
        step: Step,
        request: &VideoRequest,
        video: &Tensor,
        audio: Option<&Tensor>,
        cond: &Conditioning,
    ) -> Result<()> {
        let layout = nn::Denoiser::layout(self, request)?;
        packing::check(request, &layout)?;
        let transformer = &self.config.transformer;

        let Some(audio) = audio else {
            return Err(Error::request(
                "an H3 layout has an audio stream and no audio latents were lent".to_owned(),
            ));
        };
        if video.dtype() != SAMPLE_DTYPE || audio.dtype() != SAMPLE_DTYPE {
            return Err(Error::request(format!(
                "the latents are {:?} and {:?}, and a trajectory is FP32",
                video.dtype(),
                audio.dtype()
            )));
        }
        let text = &cond.text;
        if text.dims()[0] != nn::capability::prefix_rows(&layout) as usize {
            return Err(Error::request(format!(
                "a conditioning of {} rows for a layout that gathers {}",
                text.dims()[0],
                nn::capability::prefix_rows(&layout)
            )));
        }

        // The layout: the document row each packed row holds, and this rank's interval of it.
        let rows = match &self.config.vsa {
            Some(vsa) => {
                let tile = vsa.tile_rows;
                if !layout.text_rows.is_multiple_of(tile)
                    || !layout.condition_rows.is_multiple_of(tile)
                {
                    return Err(Error::request(format!(
                        "a tiled layout's capacities are whole {tile}-row tiles, not {} text \
                         rows and {} condition rows",
                        layout.text_rows, layout.condition_rows
                    )));
                }
                packing::tiled_rows(request, &layout, vsa)?
            }
            None => packing::dense_rows(request, &layout, self.sequence.size)?,
        };
        let tokens = rows.elem_count();
        let ranks = self.sequence.size as usize;
        if tokens > self.max_rows as usize || !tokens.is_multiple_of(ranks) {
            return Err(Error::request(format!(
                "a layout of {tokens} packed rows on a denoiser loaded for {} rows over {ranks} \
                 rank(s)",
                self.max_rows
            )));
        }
        let own = tokens / ranks;
        let first = self.sequence.rank as usize * own;
        let interval =
            |table: &Tensor| -> Result<Tensor> { on_device(&table.narrow(0, first, own)?, video) };

        // The document's tensors in the layout's row order. A padding row sits at the origin,
        // reads the generated video's modulation and gathers the row of zeros.
        let document = packing::document_rows(request, &layout);
        let video_rows = video.elem_count() / transformer.patch_width() as usize;
        let audio_rows = audio.elem_count() / transformer.audio_channels as usize;
        let generated = video_rows + audio_rows;
        let zero_row = nn::capability::prefix_zero_row(&layout) as usize;

        let positions = packing::packed(&packing::positions(request, &layout)?, &rows, 0.0)?
            .to_dtype(DType::F32)?
            .t()?;
        let tags = packing::tags(request, &layout)?;
        let groups = packing::groups(request, &layout)?;
        let modulation_rows = (groups.affine(3.0, 0.0)? + tags)?;

        // The row each document row is embedded from: the conditioning's row of a prompt or
        // condition row, and for a generated row its own projected row, which follow the
        // conditioning's rows, audio first.
        let embedded = Tensor::cat(
            &[
                packing::prefix_index(request, &layout)?.narrow(0, 0, document - generated)?,
                Tensor::arange(
                    (zero_row + 1) as u32,
                    (zero_row + 1 + generated) as u32,
                    &Device::Cpu,
                )?,
            ],
            0,
        )?;

        let groups = interval(&packing::packed(&groups, &rows, 0.0)?)?;
        let modulation_rows = interval(&packing::packed(&modulation_rows, &rows, 0.0)?)?;
        let embedded = interval(&packing::packed(&embedded, &rows, zero_row as f64)?)?;
        let positions = on_device(&positions, video)?;
        let tables = self.attention_tables(request, &layout, &rows, video)?;

        // Where each modality's rows sit in the layout, in the modality's own order.
        let audio_places = packing::layout_rows_of(&rows, document - generated, audio_rows)?;
        let video_places = packing::layout_rows_of(&rows, document - video_rows, video_rows)?;

        // The packed rows: each modality's latent rows projected in FP32 and narrowed once into
        // the stack's type, after the conditioning's rows, and this rank's rows gathered.
        let stack = text.dtype();
        let video_hidden = self
            .transformer
            .project_video(&Transformer::patchify(video)?)?
            .to_dtype(stack)?;
        let audio_latent = audio.reshape((audio_rows, transformer.audio_channels as usize))?;
        let audio_hidden = self
            .transformer
            .project_audio(&audio_latent)?
            .to_dtype(stack)?;
        let hidden =
            Tensor::cat(&[text, &audio_hidden, &video_hidden], 0)?.index_select(&embedded, 0)?;

        // The stack, conditioned by the step's gathered modulation, over the whole sequence's
        // rotary factors.
        let (cos, sin) = self.transformer.rotary_factors(&positions)?;
        let entries = self.table.entries_of(step.index)?;
        let hidden = self.transformer.forward(
            &self.table,
            &entries,
            &hidden,
            &modulation_rows,
            (&cos, &sin),
            &tables,
        )?;
        let (video_velocity, audio_velocity) =
            self.transformer
                .velocities(&self.table, &entries, &hidden, &groups, step.index)?;

        // Each modality's velocity rows, in its own order: every rank's interval gathered in
        // rank order, and the modality's rows read out of the whole sequence.
        let collected = |velocity: Tensor, places: &Tensor| -> Result<Tensor> {
            let sequence = match ranks {
                1 => velocity,
                _ => nn::collective::gathered(&velocity)?,
            };
            Ok(sequence.index_select(&on_device(places, video)?, 0)?)
        };

        let velocity =
            Transformer::unpatchify(&collected(video_velocity, &video_places)?, video.dims())?;
        video.slice_set(&euler_step(video, &velocity, step)?, 0, 0)?;

        let velocity = collected(audio_velocity, &audio_places)?.reshape(audio.shape())?;
        audio.slice_set(&euler_step(audio, &velocity, self.audio_step(step)?)?, 0, 0)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A table narrowed at any row reaches the device at the address the uploads before it
    /// leave, which is what a replay of a captured step writes it to.
    #[test]
    fn a_narrowed_table_is_uploaded_from_its_first_element() {
        let context = ops::Context::meta(None);
        let _guard = context.enter();
        let like = ops::retained(1 << 40, 4, DType::F32, &ops::meta()).unwrap();
        let host = Tensor::arange(0_u32, 16, &Device::Cpu).unwrap();

        let address = |first: usize| {
            context.reset();
            let table = on_device(&host.narrow(0, first, 4).unwrap(), &like).unwrap();
            assert_eq!(table.dims(), [4]);
            ops::operand(&table).unwrap().ptr.address()
        };
        assert_eq!(address(2), address(7));
    }
}
