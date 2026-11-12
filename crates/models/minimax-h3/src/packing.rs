//! The packed sequence of one request: the rows its prompt, its conditions and its generated
//! audio and video occupy, as CPU tensor expressions.
//!
//! ## The document
//!
//! The reference builds one request's sequence in one function, from slices, `arange`, `cat`,
//! `meshgrid` and indexed assignment (`refs/vllm-omni/…/minimax_h3/packed_sequence.py:149-288`
//! for `t2va` and `fl2va`, `:341-738` for the multi-block `ref2va` form). Its row order is the
//! **document**:
//!
//! ```text
//! [ text | keyframes | references (each: audio rows, then visual rows) | audio | video ]
//! ```
//!
//! Each tensor of the document is one function here, in the reference's order and arithmetic:
//! [`positions`] (the rotary coordinates), [`tags`] (the modality a row's modulation reads),
//! [`groups`] (the timestep a row is conditioned on) and [`prefix_index`] (the conditioning row
//! a row gathers). A request's conditions are its [`ConditionShape`]s, as request processing
//! prepared them; a row's tag and group are the integer constants below.
//!
//! ## The layouts
//!
//! A forward does not compute over the document's order. It computes over a **layout** whose
//! row count follows from the request's [`Layout`] alone, so one captured graph serves every
//! request of the layout, whatever its prompt's length:
//!
//! - the dense layout ([`dense_rows`]) is `[ video | audio | text | conditions | padding ]`.
//!   The generated rows lead, so the rows a sequence-parallel rank owns of each modality do not
//!   move with the prompt; the text and the conditions follow, and the rest is padding that no
//!   query attends;
//! - the tiled layout of video sparse attention ([`tiled_rows`]) is
//!   `[ text | conditions | audio | video | padding ]` in whole tiles: the text and the
//!   conditions are capacities filled from their start, every dense segment fills tiles of its
//!   own, and a video's rows are tile-major over its token grid
//!   (`refs/vllm-omni/…/attention/ops/video_tiles.py:11-62`), so every tile the attention keeps
//!   is a whole tile of rows. [`tile_table`] states each tile's valid rows, its video segment
//!   and each segment's kept tiles.
//!
//! A layout is a `[rows]` `U32` tensor naming the document row each of its rows holds, with the
//! document's row count for a padding row; a document tensor becomes the layout's by one gather
//! ([`packed`]).

use std::ops::Range;

use nn::{ConditionRole, ConditionShape, DType, Device, LatentShape, Layout, Tensor, VideoRequest};

use crate::config::{
    AUDIO_CHANNELS, AUDIO_HOP, AUDIO_LATENTS_PER_SECOND, FPS, FRAMES, LATENT_COMPRESSION,
    PATCH_SIDE, TransformerConfig, VsaConfig,
};
use crate::{Error, Result};

/// The tag of a generated or condition video row, and of a padding row.
pub const TAG_VIDEO: u32 = 0;
/// The tag of a prompt row.
pub const TAG_TEXT: u32 = 1;
/// The tag of a generated or reference audio row.
pub const TAG_AUDIO: u32 = 2;

/// The timestep group of the generated video, which prompt and padding rows share
/// (`refs/vllm-omni/…/minimax_h3/denoise_loop.py:194-198`).
pub const GROUP_VIDEO: u32 = 0;
/// The timestep group of the generated audio.
pub const GROUP_AUDIO: u32 = 1;
/// The timestep group of a visual condition, anchored near the clean end.
pub const GROUP_VISUAL_CONDITION: u32 = 2;
/// The timestep group of an audio reference, which is clean.
pub const GROUP_AUDIO_CONDITION: u32 = 3;

/// The scale of the spatial rotary axes (`packed_sequence.py:41`).
const INTERP: f64 = 32.0;

/// The rotary time of one output frame (`packed_sequence.py:44`).
const FRAME_RESCALE: f64 = 5.0 / 3.0;

/// Output frames each of the five latent frames of one clip covers (`packed_sequence.py:43`).
const FRAME_PER_TOKEN: [f64; 5] = [1.0, 4.0, 4.0, 4.0, 4.0];

/// The pixels one patch spans along each side: the VAE's compression times the patch side.
const PATCH_PIXELS: u32 = LATENT_COMPRESSION * PATCH_SIDE;

/// The CPU device every tensor of a packed sequence is built on.
const CPU: &Device = &Device::Cpu;

/// A `[len]` `U32` tensor holding `value`.
fn full(value: usize, len: usize) -> Result<Tensor> {
    Ok(Tensor::full(value as u32, len, CPU)?)
}

/// The `[end − start]` `U32` tensor `start..end`.
fn range(start: usize, end: usize) -> Result<Tensor> {
    Ok(Tensor::arange(start as u32, end as u32, CPU)?)
}

/// Whether `condition` anchors a generated frame.
fn is_keyframe(condition: &ConditionShape) -> bool {
    condition.role != ConditionRole::Reference
}

/// The `[frames, rows, columns]` token grid of a condition's pixels: its latent frames by the
/// frame rule and one token per patch; zeros without pixels.
fn token_grid(condition: &ConditionShape) -> [usize; 3] {
    if condition.width == 0 || condition.height == 0 {
        return [0; 3];
    }

    [
        FRAMES.latent_frames(condition.frames) as usize,
        (condition.height / PATCH_PIXELS) as usize,
        (condition.width / PATCH_PIXELS) as usize,
    ]
}

/// The latent steps of a condition's soundtrack: one per hop, the last one padded.
fn audio_steps(condition: &ConditionShape) -> usize {
    condition.audio_samples.div_ceil(AUDIO_HOP) as usize
}

/// The `(audio, visual)` rows `condition` takes in the document: one row per channel per latent
/// step, channel-major, then one row per patch of every latent frame.
pub fn condition_rows(condition: &ConditionShape) -> (usize, usize) {
    let [frames, rows, columns] = token_grid(condition);

    (
        AUDIO_CHANNELS as usize * audio_steps(condition),
        frames * rows * columns,
    )
}

/// The places of `request`'s conditions in the document, as indices into
/// [`VideoRequest::conditions`]: the keyframes in request order, then the references in request
/// order.
///
/// The released `fl2va` layout holds keyframes alone, and a `ref2va` request's keyframes lead
/// its references as the reference's guide blocks do (`packed_sequence.py:370-377`).
pub fn condition_order(request: &VideoRequest) -> Vec<usize> {
    let conditions = &request.conditions;
    let keyframes = (0..conditions.len()).filter(|index| is_keyframe(&conditions[*index]));
    let references = (0..conditions.len()).filter(|index| !is_keyframe(&conditions[*index]));

    keyframes.chain(references).collect()
}

/// Checks `request` against `layout`: a prompt that fits the text capacity, conditions the
/// document can hold and that fit the condition capacity, and vision spans inside the prompt.
///
/// Every function of this module assumes a checked request; the denoiser checks once, where it
/// lays the request out.
///
/// # Errors
///
/// A prompt of no row or beyond the capacity, a condition whose sides are not whole patches, a
/// keyframe without its frame or at another resolution than the generated video, an image
/// reference with a soundtrack, conditions beyond the capacity, or a span outside the prompt.
pub fn check(request: &VideoRequest, layout: &Layout) -> Result<()> {
    if request.text_tokens < 1 || request.text_tokens > layout.text_rows {
        return Err(Error::request(format!(
            "a prompt of {} row(s) does not fit the {} text rows of the layout",
            request.text_tokens, layout.text_rows
        )));
    }

    for condition in &request.conditions {
        let visual = condition.width > 0 && condition.height > 0;
        if visual
            && (!condition.width.is_multiple_of(PATCH_PIXELS)
                || !condition.height.is_multiple_of(PATCH_PIXELS)
                || condition.frames == 0)
        {
            return Err(Error::request(format!(
                "a condition of {} frame(s) at {}x{} is not whole {PATCH_PIXELS}-pixel patches \
                 of at least one frame",
                condition.frames, condition.width, condition.height
            )));
        }

        let [frames, rows, columns] = token_grid(condition);
        if is_keyframe(condition)
            && (rows, columns)
                != (
                    (layout.video.height / PATCH_SIDE) as usize,
                    (layout.video.width / PATCH_SIDE) as usize,
                )
        {
            return Err(Error::request(
                "an H3 keyframe carries one frame at the generated video's resolution".to_owned(),
            ));
        }
        if !is_keyframe(condition) && frames == 1 && condition.audio_samples > 0 {
            return Err(Error::request(
                "an H3 image reference carries no audio".to_owned(),
            ));
        }
    }

    let taken: usize = request
        .conditions
        .iter()
        .map(|condition| {
            let (audio, visual) = condition_rows(condition);
            audio + visual
        })
        .sum();
    if taken > layout.condition_rows as usize {
        return Err(Error::request(format!(
            "the request's conditions take {taken} row(s) and the layout reserves {}",
            layout.condition_rows
        )));
    }

    let prompt = request.text_tokens;
    if request
        .vision_spans
        .iter()
        .any(|(first, rows)| *rows == 0 || first + rows > prompt)
    {
        return Err(Error::request(
            "vision spans must lie within the prompt".to_owned(),
        ));
    }
    Ok(())
}

/// The layout `request` is laid out in under `transformer`'s latent widths, with `text_rows`
/// text rows and `condition_rows` condition rows.
///
/// The frames snap up to a count the frame rule generates and the sides up to whole patches;
/// the audio runs at forty latent steps a second, rounded half to even as the reference's
/// `round` does. The audio latent is the stereo pair of channel planes by its steps, so its
/// `channels · frames` is the element count the denoiser reads as channel-major rows.
///
/// # Errors
///
/// A request of no pixel.
pub fn layout(
    request: &VideoRequest,
    transformer: &TransformerConfig,
    text_rows: u32,
    condition_rows: u32,
) -> Result<Layout> {
    if request.width == 0 || request.height == 0 {
        return Err(Error::request(format!(
            "a {}x{} request has no resolution",
            request.width, request.height
        )));
    }
    let frames = FRAMES.align_up(request.frames);
    let steps = (f64::from(frames) * f64::from(AUDIO_LATENTS_PER_SECOND) / f64::from(FPS))
        .round_ties_even() as u32;

    Ok(Layout {
        video: LatentShape {
            channels: transformer.video_channels,
            frames: FRAMES.latent_frames(frames),
            height: request.height.next_multiple_of(PATCH_PIXELS) / LATENT_COMPRESSION,
            width: request.width.next_multiple_of(PATCH_PIXELS) / LATENT_COMPRESSION,
        },
        audio: Some(LatentShape {
            channels: AUDIO_CHANNELS * transformer.audio_channels,
            frames: steps,
            height: 1,
            width: 1,
        }),
        text_rows,
        condition_rows,
        condition_elements: condition_elements(request, transformer) as u32,
    })
}

/// The elements `request`'s condition draws occupy at the head of its noise stream.
///
/// A visual condition's anchoring noise is drawn at its own latent shape, four voxels' channels
/// a patch row; an audio reference is conditioned at its posterior's mean and draws none, so
/// it leaves the stream where its neighbours left it.
pub fn condition_elements(request: &VideoRequest, transformer: &TransformerConfig) -> usize {
    request
        .conditions
        .iter()
        .map(|condition| condition_rows(condition).1 * transformer.patch_width() as usize)
        .sum()
}

/// The generated video's `[frames, rows, columns]` token grid: one token per `2 x 2` patch of
/// every latent frame.
fn target_grid(layout: &Layout) -> [usize; 3] {
    [
        layout.video.frames as usize,
        (layout.video.height / PATCH_SIDE) as usize,
        (layout.video.width / PATCH_SIDE) as usize,
    ]
}

/// The generated audio's latent steps.
fn target_audio_steps(layout: &Layout) -> usize {
    layout.audio.map_or(0, |audio| audio.frames as usize)
}

/// The rows of the document's segments, in document order: the prompt, each condition's audio
/// and visual rows in packed order, the generated audio, the generated video.
fn segment_rows(request: &VideoRequest, layout: &Layout) -> Vec<usize> {
    let mut rows = vec![request.text_tokens as usize];

    for index in condition_order(request) {
        let (audio, visual) = condition_rows(&request.conditions[index]);
        rows.extend([audio, visual]);
    }

    let [frames, height, width] = target_grid(layout);
    rows.extend([
        AUDIO_CHANNELS as usize * target_audio_steps(layout),
        frames * height * width,
    ]);
    rows
}

/// The rows of `request`'s document: its prompt, its conditions and its generated rows.
pub fn document_rows(request: &VideoRequest, layout: &Layout) -> usize {
    segment_rows(request, layout).iter().sum()
}

/// The patch coordinates along one spatial axis of `extent` latent cells: `extent / 2` evenly
/// spaced points of an interval of length `extent / sqrt_area` centred on one half, the right
/// endpoint excluded, scaled by [`INTERP`]. `[extent / 2]` FP64.
///
/// This is `np.linspace(left, right, extent // 2, endpoint=False) * 32`
/// (`packed_sequence.py:307-312`): the step is the quotient of the interval's own difference,
/// and each point is `i · step + left`.
fn axis_grid(extent: usize, sqrt_area: f64) -> Result<Tensor> {
    let ratio = extent as f64 / sqrt_area;
    let left = (1.0 - ratio) * 1.0 / 2.0;
    let right = left + ratio * 1.0;

    let count = extent / PATCH_SIDE as usize;
    let step = (right - left) / count as f64;

    Ok(Tensor::arange(0.0_f64, count as f64, CPU)?
        .affine(step, left)?
        .affine(INTERP, 0.0)?)
}

/// The two spatial axes of a latent raster of `height` by `width` cells: `(rows, columns)`,
/// both on the scale of the raster's geometric-mean side (`packed_sequence.py:223-225`).
fn axis_grids(height: usize, width: usize) -> Result<(Tensor, Tensor)> {
    let sqrt_area = ((height * width) as f64).sqrt();

    Ok((axis_grid(height, sqrt_area)?, axis_grid(width, sqrt_area)?))
}

/// The first and the last coordinate of an axis: where the two channels of a soundtrack sit.
fn axis_ends(axis: &Tensor) -> Result<(f64, f64)> {
    let points = axis.to_vec1::<f64>()?;

    match (points.first(), points.last()) {
        (Some(first), Some(last)) => Ok((*first, *last)),
        _ => Err(Error::request(
            "a raster of no patch has no spatial axis".to_owned(),
        )),
    }
}

/// The rotary time one latent frame spans: the output frames it covers, rescaled.
fn frame_span(frame: usize) -> f64 {
    FRAME_RESCALE * FRAME_PER_TOKEN[frame % FRAME_PER_TOKEN.len()]
}

/// The rotary time `frames` latent frames span: the running sum of their spans, in FP64
/// (`packed_sequence.py:325-328`).
fn video_span(frames: usize) -> f64 {
    (0..frames).map(frame_span).sum()
}

/// The rotary times of `frames` latent frames from `origin`: `[frames]` FP64.
///
/// `origin + cat([0, cumsum(spans[:-1])])` (`packed_sequence.py:315-322`). The running sum is
/// taken frame by frame in FP64, as `torch.cumsum` takes it: the tensor library's cumulative
/// sum is a matrix product, whose summation order is its kernel's.
fn video_times(frames: usize, origin: f64) -> Result<Tensor> {
    let elapsed: Vec<f64> = (0..frames)
        .scan(0.0_f64, |elapsed, frame| {
            let before = *elapsed;
            *elapsed += frame_span(frame);

            Some(before)
        })
        .collect();

    Ok(Tensor::from_vec(elapsed, frames, CPU)?.affine(1.0, origin)?)
}

/// The coordinates of the prompt's `rows` rows: row `i` at `(i, 0, 0)`. `[rows, 3]` FP64.
fn text_positions(rows: usize) -> Result<Tensor> {
    let time = Tensor::arange(0.0_f64, rows as f64, CPU)?;
    let zeros = time.zeros_like()?;

    Ok(Tensor::stack(&[&time, &zeros, &zeros], 1)?)
}

/// The coordinates of video rows: every frame of `times` (`[frames]`) over one frame's patch
/// coordinates in raster order. `[frames · patches, 3]` FP64.
fn video_positions(times: &Tensor, rows: &Tensor, columns: &Tensor) -> Result<Tensor> {
    let frames = times.elem_count();
    let grids = Tensor::meshgrid(&[rows, columns], false)?;
    let patches = grids[0].elem_count();

    let time = times
        .reshape((frames, 1))?
        .broadcast_as((frames, patches))?;
    let spatial = |grid: &Tensor| -> Result<Tensor> {
        Ok(grid
            .reshape((1, patches))?
            .broadcast_as((frames, patches))?)
    };

    let stacked = Tensor::stack(&[time, spatial(&grids[0])?, spatial(&grids[1])?], 2)?;
    Ok(stacked.reshape((frames * patches, 3))?)
}

/// The coordinates of one stereo track of `steps` latent steps from `origin`, channel-major:
/// a step advances the clock by one, there is no row coordinate, and the two channels sit at
/// the two ends `(first, last)` of a column axis. `[2 · steps, 3]` FP64
/// (`packed_sequence.py:247-254`).
fn audio_positions(steps: usize, origin: f64, (first, last): (f64, f64)) -> Result<Tensor> {
    let clock = Tensor::arange(0.0_f64, steps as f64, CPU)?.affine(1.0, origin)?;
    let time = Tensor::cat(&[&clock, &clock], 0)?;

    let row = time.zeros_like()?;
    let column = Tensor::cat(
        &[
            Tensor::full(first, steps, CPU)?,
            Tensor::full(last, steps, CPU)?,
        ],
        0,
    )?;

    Ok(Tensor::stack(&[time, row, column], 1)?)
}

/// The rotary coordinates `(time, row, column)` of every row of `request`'s document:
/// `[rows, 3]` FP64, which the rotary embedding reads rounded to FP32
/// (`refs/vllm-omni/…/minimax_h3/minimax_h3_transformer.py:346-355`).
///
/// Prompt row `i` sits at `(i, 0, 0)` and the media clock starts at the prompt's length. The
/// references advance it in packed order: an image by one unit, at its own spatial grid; an
/// audio track by its latent steps, its channels at the ends of the generated video's column
/// axis; a video by the longer of its soundtrack and its frames, its soundtrack at the ends of
/// its own column axis. The generated audio and video start where the references leave the
/// clock, and a keyframe sits on the generated video's grid at the time of the frame it
/// anchors.
///
/// # Errors
///
/// A tensor operation's refusal.
pub fn positions(request: &VideoRequest, layout: &Layout) -> Result<Tensor> {
    let text_rows = request.text_tokens as usize;
    let [frames, _, _] = target_grid(layout);
    let (target_rows, target_columns) =
        axis_grids(layout.video.height as usize, layout.video.width as usize)?;
    let target_ends = axis_ends(&target_columns)?;

    // The clock the references leave: each advances it by its own span, in packed order
    // (`packed_sequence.py:475-488`).
    let order = condition_order(request);
    let advance = |condition: &ConditionShape| -> f64 {
        let [frames, _, _] = token_grid(condition);

        match (frames, audio_steps(condition)) {
            (0, steps) => steps as f64,
            (1, _) => 1.0,
            (frames, steps) => (steps as f64).max(video_span(frames)),
        }
    };
    let target_origin = order
        .iter()
        .map(|index| &request.conditions[*index])
        .filter(|condition| !is_keyframe(condition))
        .fold(text_rows as f64, |origin, condition| {
            origin + advance(condition)
        });

    let mut pieces = vec![text_positions(text_rows)?];
    let mut origin = text_rows as f64;

    for index in order {
        let condition = &request.conditions[index];
        let [condition_frames, height, width] = token_grid(condition);
        let steps = audio_steps(condition);

        // A keyframe: one frame of the generated video's grid, at the first frame's time or at
        // the last one's (`packed_sequence.py:231-245`).
        if is_keyframe(condition) {
            let time = match condition.role {
                ConditionRole::FirstFrame => target_origin,
                _ => target_origin + video_span(frames) - FRAME_RESCALE,
            };
            let times = Tensor::full(time, 1, CPU)?;

            pieces.push(video_positions(&times, &target_rows, &target_columns)?);
            continue;
        }

        // A reference's soundtrack sits at the ends of its own video's column axis, and a
        // track alone at the ends of the generated video's (`packed_sequence.py:605-658`).
        let (rows, columns) = if condition_frames > 0 {
            axis_grids(height * PATCH_SIDE as usize, width * PATCH_SIDE as usize)?
        } else {
            (target_rows.clone(), target_columns.clone())
        };

        if steps > 0 {
            let ends = match condition_frames {
                0 => target_ends,
                _ => axis_ends(&columns)?,
            };
            pieces.push(audio_positions(steps, origin, ends)?);
        }
        if condition_frames > 0 {
            let times = video_times(condition_frames, origin)?;
            pieces.push(video_positions(&times, &rows, &columns)?);
        }

        origin += advance(condition);
    }

    let steps = target_audio_steps(layout);
    if steps > 0 {
        pieces.push(audio_positions(steps, target_origin, target_ends)?);
    }
    pieces.push(video_positions(
        &video_times(frames, target_origin)?,
        &target_rows,
        &target_columns,
    )?);

    Ok(Tensor::cat(&pieces, 0)?)
}

/// The tag of every row of `request`'s document: `[rows]` `U32` of [`TAG_VIDEO`], [`TAG_TEXT`]
/// and [`TAG_AUDIO`] (`packed_sequence.py:709-714`).
///
/// A prompt row is text, except the rows of its vision spans, which present an image and read
/// the video modulation (`refs/vllm-omni/…/minimax_h3/pipeline_minimax_h3.py:2160-2161`).
///
/// # Errors
///
/// A tensor operation's refusal.
pub fn tags(request: &VideoRequest, layout: &Layout) -> Result<Tensor> {
    let mut text = full(TAG_TEXT as usize, request.text_tokens as usize)?;
    for (first, rows) in &request.vision_spans {
        let span = *first as usize..(first + rows) as usize;
        let video = full(TAG_VIDEO as usize, span.len())?;

        text = text.slice_assign(std::slice::from_ref(&span), &video)?;
    }

    // After the prompt the segments alternate: each condition's audio and visual rows, then
    // the generated audio and video.
    let mut pieces = vec![text];
    for pair in segment_rows(request, layout)[1..].chunks(2) {
        pieces.push(full(TAG_AUDIO as usize, pair[0])?);
        pieces.push(full(TAG_VIDEO as usize, pair[1])?);
    }

    Ok(Tensor::cat(&pieces, 0)?)
}

/// The timestep group of every row of `request`'s document: `[rows]` `U32` of the `GROUP_*`
/// constants.
///
/// A generated video row and a generated audio row read their own level; a condition's visual
/// rows and its audio rows read the two condition levels; a prompt row reads the generated
/// video's (`refs/vllm-omni/…/minimax_h3/denoise_loop.py:190-229`).
///
/// # Errors
///
/// A tensor operation's refusal.
pub fn groups(request: &VideoRequest, layout: &Layout) -> Result<Tensor> {
    let rows = segment_rows(request, layout);
    let generated = rows.len() - 2;

    let mut pieces = vec![full(GROUP_VIDEO as usize, rows[0])?];
    for pair in rows[1..generated].chunks(2) {
        pieces.push(full(GROUP_AUDIO_CONDITION as usize, pair[0])?);
        pieces.push(full(GROUP_VISUAL_CONDITION as usize, pair[1])?);
    }
    pieces.push(full(GROUP_AUDIO as usize, rows[generated])?);
    pieces.push(full(GROUP_VIDEO as usize, rows[generated + 1])?);

    Ok(Tensor::cat(&pieces, 0)?)
}

/// The row of the request's conditioning each row of its document gathers: `[rows]` `U32`.
///
/// The conditioning holds the layout's text capacity, then its condition capacity, then one
/// row of zeros ([`nn::capability::prefix_rows`]): prompt row `i` gathers row `i`, the
/// condition rows gather the rows after the text capacity in packed order, and every generated
/// row gathers the row of zeros, which its own projection replaces.
///
/// # Errors
///
/// A tensor operation's refusal.
pub fn prefix_index(request: &VideoRequest, layout: &Layout) -> Result<Tensor> {
    let rows = segment_rows(request, layout);
    let generated = rows.len() - 2;

    let text = rows[0];
    let conditions: usize = rows[1..generated].iter().sum();
    let capacity = layout.text_rows as usize;
    let zero_row = nn::capability::prefix_zero_row(layout) as usize;

    Ok(Tensor::cat(
        &[
            range(0, text)?,
            range(capacity, capacity + conditions)?,
            full(zero_row, rows[generated] + rows[generated + 1])?,
        ],
        0,
    )?)
}

/// The rows of the dense layout of `layout`, a multiple of `align`: the generated rows and
/// both capacities.
pub fn dense_len(layout: &Layout, align: u32) -> usize {
    let [frames, height, width] = target_grid(layout);
    let rows = frames * height * width
        + AUDIO_CHANNELS as usize * target_audio_steps(layout)
        + layout.text_rows as usize
        + layout.condition_rows as usize;

    rows.next_multiple_of(align.max(1) as usize)
}

/// The rows of the dense layout a query attends: the generated rows, the prompt and the
/// conditions, which lead the layout. The rows after them are padding.
pub fn dense_used(request: &VideoRequest, layout: &Layout) -> usize {
    document_rows(request, layout)
}

/// The dense layout of `request`: `[dense_len]` `U32`, the document row each row holds and the
/// document's row count for a padding row.
///
/// `[ video | audio | text | conditions | padding ]`: the generated video in raster order, the
/// generated audio channel-major, then the document's prefix as it is.
///
/// # Errors
///
/// A tensor operation's refusal.
pub fn dense_rows(request: &VideoRequest, layout: &Layout, align: u32) -> Result<Tensor> {
    let rows = segment_rows(request, layout);
    let document: usize = rows.iter().sum();

    let video = rows[rows.len() - 1];
    let audio = rows[rows.len() - 2];
    let prefix = document - audio - video;

    Ok(Tensor::cat(
        &[
            range(prefix + audio, document)?,
            range(prefix, prefix + audio)?,
            range(0, prefix)?,
            full(document, dense_len(layout, align) - document)?,
        ],
        0,
    )?)
}

/// The tiles a token grid of `grid` holds in tiles of `shape` tokens, partial tiles included.
fn grid_tiles(grid: [usize; 3], shape: [u32; 3]) -> usize {
    grid.iter()
        .zip(shape)
        .map(|(extent, tile)| extent.div_ceil(tile as usize))
        .product()
}

/// The rows of the tiled layout of `layout` under `vsa`: the text capacity, the condition
/// capacity, the generated audio's tiles and the generated video's.
pub fn tiled_len(layout: &Layout, vsa: &VsaConfig) -> usize {
    let tile = vsa.tile_rows as usize;
    let audio = AUDIO_CHANNELS as usize * target_audio_steps(layout);

    layout.text_rows as usize
        + layout.condition_rows as usize
        + audio.next_multiple_of(tile)
        + grid_tiles(target_grid(layout), vsa.tile_shape()) * tile
}

/// The tiles of the generated video in the tiled layout of `layout`.
pub fn video_tiles(layout: &Layout, vsa: &VsaConfig) -> usize {
    grid_tiles(target_grid(layout), vsa.tile_shape())
}

/// Calls `run(first_tile, rows, grid)` for each segment of `request`'s document that holds
/// rows, in tile order: the tile the segment starts at, its document rows, and its token grid
/// when it is tiled as a video.
///
/// The prompt starts at the first tile and the conditions at the text capacity's end, each
/// audio and each visual segment in tiles of its own; the generated audio starts at the
/// condition capacity's end and the generated video in the tile after it. A reference video
/// and the generated video are tiled over their token grids; every other segment is dense.
fn tile_runs(
    request: &VideoRequest,
    layout: &Layout,
    vsa: &VsaConfig,
    mut run: impl FnMut(usize, Range<usize>, Option<[usize; 3]>) -> Result<()>,
) -> Result<()> {
    let tile = vsa.tile_rows as usize;
    let shape = vsa.tile_shape();
    let dense_tiles = |rows: usize| rows.div_ceil(tile);

    let text = request.text_tokens as usize;
    run(0, 0..text, None)?;

    let mut tiles = layout.text_rows as usize / tile;
    let mut row = text;
    for index in condition_order(request) {
        let condition = &request.conditions[index];
        let (audio, visual) = condition_rows(condition);
        let grid = token_grid(condition);

        if audio > 0 {
            run(tiles, row..row + audio, None)?;
            tiles += dense_tiles(audio);
            row += audio;
        }
        if visual > 0 {
            // More than one latent frame is a video, tiled over its grid; a keyframe and an
            // image stay dense (`attention/fastvideo_h3.py:86-90`).
            let video = (!is_keyframe(condition) && grid[0] > 1).then_some(grid);
            run(tiles, row..row + visual, video)?;

            tiles += video.map_or(dense_tiles(visual), |grid| grid_tiles(grid, shape));
            row += visual;
        }
    }

    let audio = AUDIO_CHANNELS as usize * target_audio_steps(layout);
    let tiles = (layout.text_rows + layout.condition_rows) as usize / tile;
    if audio > 0 {
        run(tiles, row..row + audio, None)?;
    }

    let grid = target_grid(layout);
    run(
        tiles + dense_tiles(audio),
        row + audio..row + audio + grid.iter().product::<usize>(),
        Some(grid),
    )
}

/// The rows of a video's tiles: `[tiles · tile_rows]` `U32`, tile-major over the token grid
/// `grid`, each tile's cells inside the grid in raster order at its front and `pad` after
/// them. Cell `c` of the grid is document row `first + c`.
///
/// This is the reference's tile partition laid out in whole tiles
/// (`refs/vllm-omni/…/attention/ops/video_tiles.py:11-62`): the grid padded to whole tiles,
/// cut into tiles, and each tile's cells ordered with the padding last.
fn video_tile_rows(first: usize, grid: [usize; 3], shape: [u32; 3], pad: usize) -> Result<Tensor> {
    let [frames, height, width] = grid;
    let [tile_frames, tile_height, tile_width] = shape.map(|extent| extent as usize);
    let across = [
        frames.div_ceil(tile_frames),
        height.div_ceil(tile_height),
        width.div_ceil(tile_width),
    ];
    let cells = frames * height * width;

    // A raster of the grid's cells, padded on each axis to whole tiles and cut into tiles.
    let tiled = |raster: Tensor| -> Result<Tensor> {
        Ok(raster
            .reshape((frames, height, width))?
            .pad_with_zeros(0, 0, across[0] * tile_frames - frames)?
            .pad_with_zeros(1, 0, across[1] * tile_height - height)?
            .pad_with_zeros(2, 0, across[2] * tile_width - width)?
            .reshape((
                across[0],
                tile_frames,
                across[1],
                tile_height,
                across[2],
                tile_width,
            ))?
            .permute((0, 2, 4, 1, 3, 5))?
            .reshape((
                across[0] * across[1] * across[2],
                tile_frames * tile_height * tile_width,
            ))?)
    };
    let rows = tiled(range(first, first + cells)?)?;
    let inside = tiled(Tensor::ones(cells, DType::U32, CPU)?)?;

    // Every document row is below `pad`, so an ascending sort leaves each tile's cells in
    // raster order at its front.
    let padding = Tensor::full(pad as u32, rows.shape(), CPU)?;
    let (sorted, _) = inside
        .where_cond(&rows, &padding)?
        .contiguous()?
        .sort_last_dim(true)?;

    Ok(sorted.flatten_all()?)
}

/// The tiled layout of `request` under `vsa`: `[tiled_len]` `U32`, the document row each row
/// holds and the document's row count for a padding row.
///
/// # Errors
///
/// A tensor operation's refusal.
pub fn tiled_rows(request: &VideoRequest, layout: &Layout, vsa: &VsaConfig) -> Result<Tensor> {
    let tile = vsa.tile_rows as usize;
    let shape = vsa.tile_shape();
    let document = document_rows(request, layout);

    let mut pieces: Vec<Tensor> = Vec::new();
    let mut filled = 0;
    tile_runs(request, layout, vsa, |first_tile, rows, grid| {
        // The tiles a capacity leaves between two segments hold padding.
        pieces.push(full(document, first_tile * tile - filled)?);

        let piece = match grid {
            Some(grid) => video_tile_rows(rows.start, grid, shape, document)?,
            None => Tensor::cat(
                &[
                    range(rows.start, rows.end)?,
                    full(document, rows.len().next_multiple_of(tile) - rows.len())?,
                ],
                0,
            )?,
        };

        filled = first_tile * tile + piece.elem_count();
        pieces.push(piece);
        Ok(())
    })?;

    Ok(Tensor::cat(&pieces, 0)?)
}

/// The tile tables of the tiled layout `rows` ([`tiled_rows`]) of `request`: `[3, tiles]`
/// `I64`, one column per tile.
///
/// ```text
/// row 0   the tile's valid rows; zero for a tile the request leaves empty
/// row 1   the tile's video segment, −1 for a dense or an empty tile
/// row 2   read by segment index: the key tiles a query of that segment keeps
/// ```
///
/// A segment is one video's tiles: each reference video in packed order, then the generated
/// video. A reference segment keeps its reference rate of its own tiles and the generated
/// video its own rate ([`VsaConfig::keep_rates`]); a tile outside every segment is read by
/// every query (`refs/vllm-omni/…/minimax_h3/attention/fastvideo_h3.py:132-148`).
///
/// # Errors
///
/// A tensor operation's refusal.
pub fn tile_table(
    rows: &Tensor,
    request: &VideoRequest,
    layout: &Layout,
    vsa: &VsaConfig,
) -> Result<Tensor> {
    let tile = vsa.tile_rows as usize;
    let tiles = rows.elem_count() / tile;
    let document = document_rows(request, layout);

    // A tile's valid rows are the rows of it that hold a document row.
    let sizes = rows
        .reshape((tiles, tile))?
        .lt(document as u32)?
        .to_dtype(DType::I64)?
        .sum(1)?;

    // The tiles of each video segment, in tile order.
    let mut spans: Vec<Range<usize>> = Vec::new();
    tile_runs(request, layout, vsa, |first_tile, _, grid| {
        if let Some(grid) = grid {
            spans.push(first_tile..first_tile + grid_tiles(grid, vsa.tile_shape()));
        }
        Ok(())
    })?;

    let mut segments = Tensor::full(-1_i64, tiles, CPU)?;
    for (segment, span) in spans.iter().enumerate() {
        let index = Tensor::full(segment as i64, span.len(), CPU)?;

        segments = segments.slice_assign(std::slice::from_ref(span), &index)?;
    }

    // The last segment is the generated video's; the ones before it are references.
    let (reference_rate, target_rate) = vsa.keep_rates();
    let kept = spans.iter().enumerate().map(|(segment, span)| {
        let rate = if segment + 1 == spans.len() {
            target_rate
        } else {
            reference_rate
        };

        i64::from(VsaConfig::kept_tiles(rate, span.len() as u32))
    });
    let keep = Tensor::from_iter(kept, CPU)?.pad_with_zeros(0, 0, tiles - spans.len())?;

    Ok(Tensor::stack(&[sizes, segments, keep], 0)?)
}

/// A document tensor in a layout's row order: row `r` of the result is row `rows[r]` of
/// `document`, and a padding row holds `fill`.
///
/// `document` is `[document rows, …]` and `rows` a layout ([`dense_rows`], [`tiled_rows`]); the
/// result is `[layout rows, …]` in `document`'s element type.
///
/// # Errors
///
/// A tensor operation's refusal.
pub fn packed(document: &Tensor, rows: &Tensor, fill: f64) -> Result<Tensor> {
    let mut row = document.dims().to_vec();
    row[0] = 1;
    let padding = Tensor::zeros(row, document.dtype(), CPU)?.affine(1.0, fill)?;

    Ok(Tensor::cat(&[document, &padding], 0)?.index_select(rows, 0)?)
}

/// The layout row of each of the document rows `first..first + count`: `[count]` `U32`, the
/// inverse of the layout `rows` over one segment.
///
/// # Errors
///
/// A tensor operation's refusal.
pub fn layout_rows_of(rows: &Tensor, first: usize, count: usize) -> Result<Tensor> {
    // The document rows are distinct and below the padding's value, so the ascending order of
    // a layout lists the layout row of every document row, in document order.
    Ok(rows
        .contiguous()?
        .arg_sort_last_dim(true)?
        .narrow(0, first, count)?)
}
