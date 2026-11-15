//! What a video deployment captures and warms: the requests its denoiser's graphs are captured
//! for, the condition sets each captured layout's profile opens with, and the prompts its text
//! encoder's eager vision encodes are profiled over.
//!
//! The model states the mathematics (`nn::family::VideoModel`): the canvases it generates, the
//! timelines a duration admits, the kinds of request its network serves with their conditions'
//! roles and extents, the rows a condition set takes, the largest references a layout's rows
//! hold, the unit of its capacities and how a prompt presents a vision block. The deployment
//! states what it serves ([`VideoServing`]). This module is the engine's policy over both, beside
//! the bucket choice that serves a request from what it captures (`super::buckets`):
//!
//! - the denoiser's layouts are sized for the deployment's longest prompt, its condition rows and
//!   a key bound over timelines of up to [`KEY_BOUND_FRAMES`] ([`VideoServing::capacity`]);
//! - a conditioned request rides a rung of the condition ladder, which starts at
//!   [`LEAST_CONDITION_RUNG`] and doubles ([`condition_ladder`]);
//! - one request is captured per served canvas, kind of request, timeline, prompt capacity and
//!   rung ([`candidates`]), each opened, outside its graph, with the condition sets a request of
//!   its layout may encode: every keyframe set at the canvas, and the largest references, each
//!   on the smallest rung that holds it;
//! - the text encoder warms the prompts that present the most vision rows a request's blocks
//!   bring ([`warm_prompts`]).
//!
//! Everything here is host arithmetic over the model's declarations, run once when a rank sizes
//! its runners.

use nn::family::{ConditionMedia, LayoutCapacity, VideoModel, VisionTokens};
use nn::{ConditionRole, ConditionShape, VideoRequest};

use crate::worker::{Error, Result};

/// The longest timeline, in frames, a denoiser's attention key bound is taken over: 1024 frames,
/// beyond the longest output a deployment serves (fifteen seconds is 362 frames at 24 frames per
/// second).
///
/// The bound is taken over this many frames at every canvas the checkpoint generates and not over
/// the deployment's sizes and longest output, so the kernels keyed by it are one set for every
/// deployment of a checkpoint and capacity.
pub(crate) const KEY_BOUND_FRAMES: u32 = 1024;

/// The least rung of a deployment's condition ladder, in denoiser rows: one keyframe of a 768p
/// canvas (1008 rows), the smallest condition a request brings, rounded up to a power of two.
pub(crate) const LEAST_CONDITION_RUNG: u32 = 1024;

/// A request a deployment captures its denoiser's graphs for, with the condition sets, each in
/// request order, a request of its layout may open with.
pub(crate) type Candidate = (VideoRequest, Vec<Vec<ConditionShape>>);

/// A prompt the text encoder's eager encodes are profiled over: the `[time, rows, columns]` patch
/// grids of its vision blocks in presentation order, and its token ids.
pub(crate) type WarmPrompt = (Vec<[u32; 3]>, Vec<u32>);

/// What a video deployment serves: the bounds its requests are held to and its graphs are
/// captured for.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct VideoServing {
    /// The output resolutions it serves, `(width, height)` in pixels, in the order it names them.
    pub(crate) sizes: Vec<(u32, u32)>,
    /// The longest output it serves, in seconds.
    pub(crate) max_seconds: f64,
    /// The prompt capacities it captures, strictly increasing, the largest the longest prompt it
    /// admits: a request rides the smallest that holds its prompt.
    pub(crate) text_capacities: Vec<u32>,
    /// The condition rows a layout holds at most: the deployment's statement, or the rows of the
    /// conditions its network's tasks are sized for (`nn::family::VideoCheckpoint::condition_rows`).
    pub(crate) condition_rows: u32,
}

impl VideoServing {
    /// The rows the deployment's denoiser layouts are sized for: its longest prompt, its
    /// condition rows, and a key bound over timelines of up to [`KEY_BOUND_FRAMES`].
    ///
    /// # Errors
    ///
    /// A deployment that states no prompt capacity.
    pub(crate) fn capacity(&self) -> Result<LayoutCapacity> {
        let text_rows = *self.text_capacities.last().ok_or_else(|| {
            Error::Invalid(
                "this deployment states no prompt capacity, so no layout could hold a prompt"
                    .to_owned(),
            )
        })?;

        Ok(LayoutCapacity {
            text_rows,
            condition_rows: self.condition_rows,
            frames: KEY_BOUND_FRAMES,
        })
    }
}

/// The condition rows a layout of a deployment of `capacity` condition rows may hold, ascending
/// and ending at the capacity: one captured graph per rung, and a request rides the smallest rung
/// that holds its conditions.
///
/// The rungs start at [`LEAST_CONDITION_RUNG`] and double, with a rung halfway between two
/// doublings from four times the least rung up, where a request's empty rows start to cost more
/// than another graph. Each is whole `unit`s (the model's capacity unit), so a layout of any rung
/// packs.
pub(crate) fn condition_ladder(capacity: u32, unit: u32) -> Vec<u32> {
    let unit = unit.max(1);

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

/// Refuses `sizes` unless the deployment serves at least one and the model generates each.
///
/// # Errors
///
/// No size, a size the model does not generate, or the model's refusal to state its canvases.
pub(crate) fn check_sizes(model: &dyn VideoModel, sizes: &[(u32, u32)]) -> Result<()> {
    if sizes.is_empty() {
        return Err(Error::Invalid(
            "this deployment serves no size, so no raster could be decoded".to_owned(),
        ));
    }

    let generated = model.canvases()?;
    if let Some((width, height)) = sizes.iter().find(|size| !generated.contains(size)) {
        let named: Vec<String> = generated
            .iter()
            .map(|(width, height)| format!("{width}x{height}"))
            .collect();
        return Err(Error::Invalid(format!(
            "this deployment serves {width}x{height}, which the checkpoint does not generate: it \
             generates {}",
            named.join(", ")
        )));
    }
    Ok(())
}

/// The requests a deployment serving `serving` captures its denoiser's graphs for, each with the
/// condition sets a request of its layout may open with ([`Candidate`]).
///
/// One request per served canvas, kind of request the model's network serves, timeline the
/// longest output admits, prompt capacity and, for a kind that carries conditions, rung of the
/// condition ladder; a kind without conditions reserves no condition row. A request states its
/// kind's roles and none of its conditions: a captured graph is keyed by its layout, which the
/// model derives from the request ([`nn::Denoiser::layout`]).
///
/// A request's opening encodes its own conditions eagerly, outside any captured graph, so the
/// profile run opens each layout once per set to measure that encode and state its kernels:
///
/// - for keyframes at the canvas, every non-empty subset of the kind's roles, on the smallest
///   rung that holds the rows the set takes, and none a rung of the ladder cannot hold;
/// - for references of the client's media, per rung the largest of each form the rung holds
///   ([`VideoModel::largest_conditions`]), each on the smallest rung that holds it, once.
///
/// # Errors
///
/// A rank that places no denoiser, a size the model does not generate, a duration it admits no
/// timeline of, or a condition set it cannot lay out.
pub(crate) fn candidates(model: &dyn VideoModel, serving: &VideoServing) -> Result<Vec<Candidate>> {
    let ladder = condition_ladder(serving.condition_rows, model.capacity_unit()?);
    let timelines = model.timelines(serving.max_seconds)?;
    let longest = model.timeline(serving.max_seconds)?;
    check_sizes(model, &serving.sizes)?;

    let fps = model.frame_rate();
    let kinds = model.requests();

    let mut candidates = Vec::new();
    for &canvas in &serving.sizes {
        for kind in &kinds {
            let rungs: Vec<Option<u32>> = if kind.roles.is_empty() {
                vec![None]
            } else {
                ladder.iter().copied().map(Some).collect()
            };
            let openings = match kind.media {
                ConditionMedia::None => Vec::new(),
                ConditionMedia::Canvas => keyframe_openings(model, &kind.roles, canvas, &ladder)?,
                ConditionMedia::Client => reference_openings(model, canvas, longest, &ladder)?,
            };

            for &frames in &timelines {
                for &text_tokens in &serving.text_capacities {
                    for &rung in &rungs {
                        let request = VideoRequest {
                            width: canvas.0,
                            height: canvas.1,
                            frames,
                            fps,
                            roles: kind.roles.clone(),
                            text_tokens,
                            conditions: Vec::new(),
                            vision_spans: Vec::new(),
                            condition_rows: rung,
                        };
                        let opened = openings
                            .iter()
                            .filter(|(served, _)| Some(*served) == rung)
                            .map(|(_, set)| set.clone())
                            .collect();

                        candidates.push((request, opened));
                    }
                }
            }
        }
    }
    Ok(candidates)
}

/// The keyframe sets a request at `canvas` anchoring any of `roles` may present, each with the
/// rung of `ladder` that serves it: every non-empty subset of the roles, each keyframe one frame
/// at the canvas, on the smallest rung holding the rows the model packs it into.
fn keyframe_openings(
    model: &dyn VideoModel,
    roles: &[ConditionRole],
    (width, height): (u32, u32),
    ladder: &[u32],
) -> Result<Vec<(u32, Vec<ConditionShape>)>> {
    let mut openings = Vec::new();
    for mask in 1_u32..(1 << roles.len()) {
        let set: Vec<ConditionShape> = roles
            .iter()
            .enumerate()
            .filter(|(bit, _)| mask & (1 << bit) != 0)
            .map(|(_, role)| ConditionShape {
                role: *role,
                frames: 1,
                height,
                width,
                audio_samples: 0,
            })
            .collect();

        let rows = model.condition_rows(&set, (width, height))?;
        let Some(rung) = ladder.iter().copied().find(|rung| *rung >= rows) else {
            continue;
        };
        openings.push((rung, set));
    }
    Ok(openings)
}

/// The reference sets a request at `canvas` lasting at most `longest` frames opens its layout's
/// profile with, each with the rung of `ladder` that serves it: per rung, each of the largest
/// references the rung's rows hold, alone, on the smallest rung that holds it, as a request of
/// it rides; a set two rungs bound alike is opened once.
fn reference_openings(
    model: &dyn VideoModel,
    canvas: (u32, u32),
    longest: u32,
    ladder: &[u32],
) -> Result<Vec<(u32, Vec<ConditionShape>)>> {
    let mut openings: Vec<(u32, Vec<ConditionShape>)> = Vec::new();
    for &rung in ladder {
        for reference in model.largest_conditions(canvas, longest, rung)? {
            let set = vec![reference];
            let taken = model.condition_rows(&set, canvas)?;

            if let Some(served) = ladder.iter().copied().find(|served| *served >= taken) {
                let opening = (served, set);
                if !openings.contains(&opening) {
                    openings.push(opening);
                }
            }
        }
    }
    Ok(openings)
}

/// The prompts with vision blocks the text encoder's eager encodes are profiled over, for a
/// deployment serving `serving` ([`WarmPrompt`]), each once, in the order of the model's kinds of
/// request.
///
/// - Keyframes at the canvas fix their grids: per served size, a prompt of each count of
///   keyframes up to the kind's most, each at the model's grid of the size.
/// - References present grids that follow the client's media, so their prompts bound every
///   presentation the deployment admits ([`reference_prompts`]).
/// - A kind whose prompts present no vision block warms none.
///
/// A prompt's ids present each block as the text encoder reads one: the model's placeholder
/// token per merged row, between two text tokens that stand for the block's delimiting markers.
/// An encode's work follows from the placeholder runs and the prompt's length alone, so any text
/// token stands for a marker. A prompt presenting vision blocks is encoded eagerly over its own
/// layout, in the stream region the captured encodes take their memory from, so the profile run
/// encodes each beside the captured encodes and the region's peak holds the eager encode too.
///
/// # Errors
///
/// A size the model does not generate, or a model that states no rule to present a vision block
/// while one of its kinds presents one.
pub(crate) fn warm_prompts(
    model: &dyn VideoModel,
    serving: &VideoServing,
) -> Result<Vec<WarmPrompt>> {
    let kinds: Vec<_> = model
        .requests()
        .into_iter()
        .filter(|kind| kind.vision_blocks > 0 && kind.media != ConditionMedia::None)
        .collect();
    if kinds.is_empty() {
        return Ok(Vec::new());
    }
    let vision = model.vision_tokens()?;

    let mut prompts: Vec<WarmPrompt> = Vec::new();
    let mut add = |prompt: WarmPrompt| {
        if !prompts.contains(&prompt) {
            prompts.push(prompt);
        }
    };
    for kind in &kinds {
        match kind.media {
            ConditionMedia::Canvas => {
                check_sizes(model, &serving.sizes)?;
                for &(width, height) in &serving.sizes {
                    let grid = model.vision_grid(width, height)?;
                    let tokens = vision.tokens(grid);
                    for count in 1..=kind.vision_blocks {
                        add(presented(
                            &vec![grid; count as usize],
                            &[tokens],
                            vision.placeholder,
                        ));
                    }
                }
            }
            ConditionMedia::Client => {
                for prompt in
                    reference_prompts(&serving.text_capacities, kind.vision_blocks, vision)
                {
                    add(prompt);
                }
            }
            ConditionMedia::None => {}
        }
    }
    Ok(prompts)
}

/// The prompts that bound every presentation of references a deployment admits: per prompt
/// capacity of `capacities`, the capacity's tokens presenting vision rows in one block, and in
/// `blocks` (the most a request presents), every token but the blocks' markers a vision row;
/// each block an image of one time step and `vision.merge` patch rows, its merged rows sharing
/// the blocks' rows as evenly as they divide.
///
/// An admitted presentation rides a capacity of at least its tokens (the deployment refuses a
/// prompt longer than its largest), and each reference presents one block, so it presents at
/// most that capacity less two markers per block as vision rows, in at most `blocks` blocks. The
/// text encoder's host values and the region its eager encode computes in grow with the vision
/// rows and the grids that present them, and an image of one time step costs the most of any
/// grid per merged row: the position table is sampled for every frame, and a video's time steps
/// share their frame's. The capacity's presentation in one grid holds the largest grid any
/// admitted prompt encodes, and in the most grids the most host copies.
fn reference_prompts(capacities: &[u32], blocks: u32, vision: VisionTokens) -> Vec<WarmPrompt> {
    // The markers around each block's run of placeholders.
    const MARKERS: u32 = 2;
    let merge = vision.merge.max(1);

    let mut prompts = Vec::new();
    for &capacity in capacities {
        for count in [1, blocks] {
            // Each block holds at least one vision row beside its markers.
            if capacity < count * (MARKERS + 1) {
                continue;
            }
            let rows = capacity - count * MARKERS;
            let shares: Vec<u32> = (0..count)
                .map(|block| rows / count + u32::from(block < rows % count))
                .collect();

            // A block of `share` merged rows is `merge` patch rows by `merge · share` columns.
            let grids: Vec<[u32; 3]> = shares
                .iter()
                .map(|share| [1, merge, merge * share])
                .collect();
            prompts.push(presented(&grids, &shares, vision.placeholder));
        }
    }
    prompts
}

/// The prompt that presents `grids`, block `i` of `tokens[i]` merged rows (the last entry
/// standing for every later block): each block's placeholders between two marker tokens.
fn presented(grids: &[[u32; 3]], tokens: &[u32], placeholder: u32) -> WarmPrompt {
    /// The text token that stands for a block's markers.
    const MARKER: u32 = 0;

    let mut ids = Vec::new();
    for index in 0..grids.len() {
        let rows = tokens.get(index).or(tokens.last()).copied().unwrap_or(0);
        ids.push(MARKER);
        ids.extend(std::iter::repeat_n(placeholder, rows as usize));
        ids.push(MARKER);
    }
    (grids.to_vec(), ids)
}

#[cfg(test)]
mod tests;
