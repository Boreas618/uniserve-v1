//! The rows of a step's batches (`specs/scheduler-design.md` §2.3, §3.4, §7.1, §7.2): a request's
//! call joins the batch of its group and phase as one token row ([`Scheduler::push_token_rows`])
//! or one media row ([`Scheduler::push_media_row`]), and a batch leaves the step with its
//! generation and the shape that selects its bucket ([`Scheduler::seal`]).
//!
//! A row carries a slot, positions and inputs, never a unit or a run: a token's cache position
//! in its request, a recurrent-state checkpoint's position in the request's state table, an
//! encoder output's position in its encoder-output table and a media product's position in its
//! product table. The workers find every block through their device block tables.
//!
//! # A token request's positions
//!
//! A token request's context is its prompt text (a resumed request's generated tokens included)
//! with the encoder blocks of each image at the image's prompt position, then its output. Its
//! cache positions count the context's rows, one per text token and one per row of an encoder
//! block, so `Request::num_computed_tokens` and `Request::num_prompt_tokens` count rows while
//! `Request::token_ids` holds the text and the output alone. A text token rotates from its
//! logical position; every row of an image's blocks rotates from the position after the text
//! before the image, and the text after the image continues `num_positions` later. A context
//! that ends with an image block samples nothing from its prompt, and its output starts with the
//! beginning-of-sequence token, which its first decode writes.
//!
//! The recurrent-state checkpoint that holds the state after cache position `b - 1`, `b` a
//! multiple of the state group's block size, lies at position `b / block - 1` of the request's
//! state table: the logical index of the block that ends at `b`, as for every cache group.

use std::collections::BTreeMap;
use std::iter;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::batch::ValidationResult;
use crate::batch::philox::{DrawAddress, request_sampling_key};
use crate::batch::{
    Batch, Columns, DraftWork, Extent, MediaRow, NewSlot, Phase, Row, Shape, StateCopy,
    StateDirection, Variant, VisionRows,
};
use proto::module::commit::{self, COMMIT_FIELDS};
use proto::module::sampler::{PARAMETER_COLUMNS, TEMPERATURE};
use proto::request::{
    DiffusionRequest, GenerationRequest, ImageEncoder, RequestId, SamplingParams,
};

use super::group::variants;
use super::memory::{kept, restored};
use super::{Capacity, Request, Scheduler};

/// The sampler's parameter columns of one row.
const PARAMETERS: usize = PARAMETER_COLUMNS as usize;

impl Scheduler {
    /// Appends request `id`'s call of the token phase `phase` to `batch`, the batch of its group
    /// and phase: one row of `tokens` tokens, the count the step charged for the call (a prefill
    /// or refill chunk, one for a decode, one and the drafts for a verification), and the
    /// request's [`NewSlot`] when the group has not served it before.
    ///
    /// The call starts at the request's `num_computed_tokens`, a refill at its
    /// `num_draft_tokens`, which the caller advances after this returns. The caller has charged
    /// the call, found that the request holds the blocks it writes, and keeps every image block
    /// whole within one chunk.
    ///
    /// Returns false and appends nothing when the request's state does not state the row: a phase
    /// that is not a token phase, a request that is not a token request or holds no slot on the
    /// batch's group, a prefill past the prompt, a chunk that cuts an image block or reads an
    /// encoder output whose position is not known, a verification of more drafts than the request
    /// holds, a refill over image rows, a row other than a decode reading a token in flight, or a
    /// restore of the recurrent state at a position that ends no checkpoint block.
    #[must_use]
    pub(super) fn push_token_rows(
        &mut self,
        batch: &mut Batch,
        id: RequestId,
        phase: Phase,
        tokens: u32,
    ) -> bool {
        let Some(started) = self.append_token_row(batch, id, phase, tokens) else {
            return false;
        };

        // The batch starts the request's sampling state on its slot; its later rows on the group
        // continue it.
        if started && let Some(request) = self.requests.get_mut(&id) {
            for attachment in &mut request.attachments {
                if attachment.0 == batch.group {
                    attachment.3 = true;
                }
            }
        }
        true
    }

    /// Appends the row of [`Scheduler::push_token_rows`]: every value is computed before the
    /// batch changes, so a row the request cannot state leaves the batch as it was. Returns
    /// whether the row started the request on the group.
    fn append_token_row(
        &self,
        batch: &mut Batch,
        id: RequestId,
        phase: Phase,
        tokens: u32,
    ) -> Option<bool> {
        let request = self.requests.get(&id)?;
        let input = generation(request)?;
        let capacity = &self.group(batch.group)?.capacity;
        let &(_, _, slot, seen) = request
            .attachments
            .iter()
            .find(|attached| attached.0 == batch.group)?;
        if slot == 0 || tokens == 0 {
            return None;
        }

        // Where the call starts and how many of its tokens the context gives: a verification's
        // first token is its committed token and the rest its drafts. A refill evaluates tokens
        // the cache holds, from the end of the draft cache, and writes only the draft cache.
        let prompt = request.num_prompt_tokens;
        let (start, context_tokens, writes) = match phase {
            Phase::Prefill => (request.num_computed_tokens, tokens, true),
            Phase::Decode if tokens == 1 => (request.num_computed_tokens, 1, true),
            Phase::Verify => (request.num_computed_tokens, 1, true),
            Phase::Refill => (request.num_draft_tokens, tokens, false),
            _ => return None,
        };
        let end = start.checked_add(tokens)?;
        if phase == Phase::Prefill && end > prompt {
            return None;
        }

        // The row's input ids and rotary positions, its image blocks, the rotary position after
        // it, and the prompt text before its end.
        let bos = capacity.bos_token_id;
        let (mut ids, mut rotary, mut vision) = (Vec::new(), Vec::new(), Vec::new());
        let range = start..start.checked_add(context_tokens)?;
        let (mut after, text_end, cut) =
            context(request, range, bos, &mut ids, &mut rotary, &mut vision)?;
        let drafts = (tokens - context_tokens) as usize;
        for &draft in request.spec_token_ids.get(..drafts)? {
            ids.push(i32::try_from(draft).ok()?);
            rotary.push(i32::try_from(after).ok()?);
            after = after.checked_add(1)?;
        }

        // One call writes an image block whole, and a refill reads none. Only a pipelined
        // decode reads a token in flight, which the runner takes from the slot.
        if cut
            || (phase == Phase::Refill && !vision.is_empty())
            || (phase != Phase::Decode && ids.contains(&-1))
        {
            return None;
        }

        // Only the chunk that ends the prompt samples, and not when the context ends with an
        // image block; a decode and a verification sample their row.
        let image_end = ends_with_image(input);
        let samples = match phase {
            Phase::Decode | Phase::Verify => true,
            Phase::Prefill => end == prompt && !image_end,
            _ => false,
        };
        let generated = request.generated_ids();

        // The tokens the row may not sample: under the `min_tokens` floor counted with the
        // decodes in flight, the end-of-sequence tokens; and the last token of each bad word.
        let banned = if samples {
            let decode = Phase::Decode.index();
            let in_flight = request.scheduled[decode].saturating_sub(request.completed[decode]);
            banned_tokens(
                &input.sampling,
                &capacity.eos_token_ids,
                generated,
                generated.len() + in_flight as usize,
                &[],
            )
        } else {
            Vec::new()
        };

        // A prefill of a request that scores its prompt names, for each token, the prompt token
        // after it: the next input, after the last the next prompt token, after the prompt's last
        // none. A chunk with image blocks scores nothing.
        let mut targets = vec![-1; tokens as usize];
        if phase == Phase::Prefill
            && input.sampling.prompt_logprobs_requested()
            && vision.is_empty()
        {
            let next = match input.prompt_token_ids.get(text_end) {
                Some(&token) => i32::try_from(token).ok()?,
                None => -1,
            };
            targets = ids.iter().skip(1).copied().chain([next]).collect();
        }

        // What the call does for the draft cache, in a deployment whose drafter keeps one.
        let speculates = self.config.draft_tokens > 0
            && capacity
                .cache_groups
                .iter()
                .any(|(kind, _)| kind.is_draft());
        let proposals = if request.num_draft_tokens >= start {
            u32::try_from(self.config.draft_tokens).ok()?
        } else {
            0
        };
        let draft = match phase {
            // A chunk short of the prompt's end knows the token after it, unless an image row
            // follows, which carries none.
            Phase::Prefill if speculates => DraftWork {
                proposals: if end == prompt { proposals } else { 0 },
                lookahead: if end < prompt {
                    lookahead(request, end, bos)
                } else {
                    None
                },
                refill: None,
            },
            Phase::Verify => DraftWork {
                proposals,
                lookahead: None,
                refill: None,
            },
            Phase::Refill => DraftWork {
                proposals: 0,
                lookahead: Some(lookahead(request, end, bos)?),
                refill: Some(start),
            },
            _ => DraftWork::default(),
        };

        // The recurrent state: the first row on a slot that continues from a cached prefix
        // restores the checkpoint the prefix ends at; a prompt or refill chunk keeps the state at
        // the last block boundary it reaches, a decode or a verification only at a multiple of
        // the state interval.
        let (mut restore, mut keep) = (None, None);
        if let (Some(block), Some(interval)) = (state_block(capacity), state_interval(capacity)) {
            if !seen && start > 0 {
                restore = Some(restored(start, block)?);
            }
            keep = kept(phase, start, end, block, interval);
        }

        // Each row draws at the rotary position of the token it samples, in its request's
        // stream. The stream draws once per applied decode and once for the prompt's token.
        let draw = DrawAddress {
            key: request_sampling_key(input.sampling.seed, 0, id.0, 0),
            position: u64::from(after),
        };
        let drawn =
            request.completed[Phase::Decode.index()] + u32::from(start >= prompt && !image_end);

        // The commit row: how the step advances the slot's device state.
        let mode = match phase {
            Phase::Decode => commit::mode::DECODE,
            Phase::Verify => commit::mode::VERIFY,
            Phase::Prefill if samples => commit::mode::EXPLICIT,
            _ => commit::mode::CACHE,
        };
        let columns = &batch.columns;
        let row = u32::try_from(columns.rows.calls.len()).ok()?;
        let first_token = u32::try_from(columns.input_ids.len()).ok()?;
        let tasks = columns.rows.logits_rows.iter().filter(|row| **row >= 0);
        let task = if samples {
            i64::try_from(tasks.count()).ok()?
        } else {
            -1
        };

        // A sampling prefill installs its token at the host's coordinates and a verification the
        // base its accepted span advances from; a decode advances from the slot's own state.
        let (logical, sampling, visible) = match mode {
            commit::mode::EXPLICIT => (i64::from(after), i64::from(drawn) + 1, 0),
            commit::mode::VERIFY => (i64::from(*rotary.first()?), i64::from(drawn), start.into()),
            _ => (0, 0, 0),
        };
        let mut cells = [0_i64; COMMIT_FIELDS];
        cells[commit::column::SLOT] = i64::from(slot);
        cells[commit::column::MODE] = mode;
        cells[commit::column::TASK] = task;
        // A prefill states its cache extent, and a refill the target's, which it leaves.
        cells[commit::column::CACHE_LENGTH] = match phase {
            Phase::Prefill => i64::from(end),
            Phase::Refill => i64::from(request.num_computed_tokens),
            _ => 0,
        };
        cells[commit::column::LOGICAL] = logical;
        cells[commit::column::SAMPLING] = sampling;
        cells[commit::column::VISIBLE] = visible;
        cells[commit::column::PENALTIES] = i64::from(samples && uses_penalties(&input.sampling));

        // The request's first row on the group starts its sampling state there: the tokens its
        // penalties count, its bias row, and the two token ids before the row.
        let new_slot = if seen {
            None
        } else {
            // The end-of-text id where the request has no earlier token: the beginning-of-
            // sequence token of a family whose checkpoint states no other.
            let (mut before, range) = (Vec::new(), start.saturating_sub(2)..start);
            context(
                request,
                range,
                bos,
                &mut before,
                &mut Vec::new(),
                &mut Vec::new(),
            )?;
            let mut previous = [bos; 2];
            for (entry, token) in previous.iter_mut().rev().zip(before.iter().rev()) {
                *entry = u32::try_from(*token).unwrap_or(bos);
            }

            // The bias row in the canonical order: a repeated token's last bias, and the allowed
            // tokens sorted once each.
            let bias: BTreeMap<u32, f32> = input.sampling.logit_bias.iter().copied().collect();
            let mut allowed = input.sampling.allowed_token_ids.clone();
            if let Some(allowed) = &mut allowed {
                allowed.sort_unstable();
                allowed.dedup();
            }
            let counted = uses_penalties(&input.sampling);
            Some(NewSlot {
                slot,
                history: if counted {
                    generated.to_vec()
                } else {
                    Vec::new()
                },
                bias: bias.into_iter().collect(),
                allowed,
                context: previous,
            })
        };

        let cache_positions = if writes {
            (start..end)
                .map(|position| i32::try_from(position).ok())
                .collect::<Option<Vec<_>>>()?
        } else {
            vec![-1; tokens as usize]
        };
        let seq_len = i32::try_from(end).ok()?;
        let query_end = i32::try_from(first_token.checked_add(tokens)?).ok()?;
        let logits_row = if samples {
            i32::try_from(row).ok()?
        } else {
            -1
        };
        let checkpoint = keep.map_or(Some(0), |boundary| i32::try_from(boundary).ok())?;
        let slot_column = i32::try_from(slot).ok()?;

        // Everything is stated: the row joins the batch.
        let columns = &mut batch.columns;
        let calls = columns.rows.calls.len();
        columns.rows.calls.push(Row {
            slot,
            kind: phase,
            cache_group: 0,
            rows: row..row + 1,
        });
        columns.rows.positions.extend(rotary);
        columns.rows.cache_positions.extend(cache_positions);
        columns.rows.seq_lens.push(seq_len);
        if columns.rows.query_offsets.is_empty() {
            columns.rows.query_offsets.push(0);
        }
        columns.rows.query_offsets.push(query_end);
        columns.rows.logits_rows.push(logits_row);
        columns.input_ids.extend(ids);
        columns.slots.push(slot_column);
        columns.parameters.extend(parameters(&input.sampling));
        columns.draws.push(draw);
        push_commit(&mut columns.commit, calls, cells);
        columns.page_size = page_size(capacity);
        columns.max_kv_len = capacity.max_model_len;

        // The optional columns: each stays empty until a row needs it.
        put(&mut columns.checkpoints, calls, vec![checkpoint], 0);
        if let Some(position) = restore {
            // Every restore runs before the forward, so it precedes every checkpoint.
            let at = columns
                .state_copies
                .iter()
                .position(|copy| copy.direction == StateDirection::Keep)
                .unwrap_or(columns.state_copies.len());
            let copy = StateCopy {
                slot,
                position,
                direction: StateDirection::Restore,
            };
            columns.state_copies.insert(at, copy);
        }
        if let (Some(boundary), Some(block)) = (keep, state_block(capacity)) {
            columns.state_copies.push(StateCopy {
                slot,
                position: boundary / block - 1,
                direction: StateDirection::Keep,
            });
        }
        put(
            &mut columns.drafts,
            calls,
            vec![draft],
            DraftWork::default(),
        );
        columns
            .vision
            .extend(vision.into_iter().map(|rows| VisionRows {
                token: first_token + rows.token,
                ..rows
            }));
        let adjusts =
            !input.sampling.logit_bias.is_empty() || input.sampling.allowed_token_ids.is_some();
        put(&mut columns.biased, calls, vec![adjusts], false);
        put(&mut columns.banned, calls, vec![banned], Vec::new());
        put(&mut columns.targets, first_token as usize, targets, -1);

        batch.requests.push(id);
        batch.new_slots.extend(new_slot);
        Some(!seen)
    }

    /// Appends request `id`'s call of the media phase `phase` to `batch`: one media row from the
    /// request's input and counters, with the results its call reads and then those it writes
    /// as positions in the request's product table (`Request::results`), and for a text
    /// encoding the request's prompt as the batch's columns. The call is the request's next of
    /// the phase, `Request::scheduled`, which the caller advances after this returns.
    ///
    /// A token request's image encoding writes the encoder output of its next encoder block to
    /// compute, `Request::num_encoded_inputs` in context order, named by its position in the
    /// request's encoder-output table, and carries the image.
    ///
    /// Returns false and appends nothing when the request's state does not state the row: a
    /// phase the request has no call of, or no slot on the batch's group.
    #[must_use]
    pub(super) fn push_media_row(&self, batch: &mut Batch, id: RequestId, phase: Phase) -> bool {
        let Some(request) = self.requests.get(&id) else {
            return false;
        };
        let Some(slot) = request.slot_on(batch.group).filter(|slot| *slot != 0) else {
            return false;
        };

        let row = match &request.input {
            proto::request::Request::Generation(input) => image_row(request, input, phase, slot),
            proto::request::Request::Diffusion(input) => {
                self.video_row(request, input, phase, slot)
            }
        };
        let Some(row) = row else {
            return false;
        };

        // A text encoding reads its prompt as one row of token columns.
        if let (Phase::TextEncode, proto::request::Request::Diffusion(input)) =
            (phase, &request.input)
            && push_prompt(&mut batch.columns, &input.prompt_token_ids).is_none()
        {
            return false;
        }
        batch.requests.push(id);
        batch.media.push(row);
        true
    }

    /// The media row of a video request's next call of `phase` on slot `slot`.
    fn video_row(
        &self,
        request: &Request,
        input: &DiffusionRequest,
        phase: Phase,
        slot: u32,
    ) -> Option<MediaRow> {
        let index = request.scheduled[phase.index()];
        // A video request's token-phase calls have no media row.
        if matches!(
            phase,
            Phase::Prefill | Phase::Decode | Phase::Verify | Phase::Refill
        ) {
            return None;
        }

        // What the call reads and writes, and the media units it covers, are the request's
        // rounds (`video.rs`): a decode round writes the splice tail of each unit it decodes and
        // a round after the first reads the tail of the unit before its own.
        let step = request.scheduled[Phase::Denoise.index()];
        let steps = request.placement[Phase::Denoise.index()]
            .and_then(|group| self.group(group))
            .map_or(0, |group| group.capacity.denoise_steps);
        let (named, read) = request.results(phase, index, &self.groups);

        // A video call's row names no rank: every rank of the decoder group decodes the units of
        // a round dealt to it, and the encode of a round reads its units wherever they lie in the
        // process (SD §3.3).
        let unit = request.round_units(phase, index, &self.groups).start;

        // A muxing call hands over the encode round it follows; the last hands none.
        let mux_rounds =
            u32::from(phase == Phase::Mux && request.completed[Phase::VideoEncode.index()] > index);

        Some(MediaRow {
            request: request.id,
            slot,
            conditioning: input.conditioning.clone(),
            // The reader decodes the condition media; no other call reads them.
            media: if phase == Phase::MediaRead {
                input.media.clone()
            } else {
                Vec::new()
            },
            step,
            steps,
            seed: input.sampling.seed,
            width: input.sampling.width,
            height: input.sampling.height,
            frames: input.sampling.num_frames,
            rounds: mux_rounds,
            unit,
            rank: None,
            image: None,
            results: named,
            reads: read,
        })
    }

    /// Completes a batch the step formed: stamps the generation it is assembled for, orders its
    /// new slots, states the shape that selects its bucket, and checks it ([`Batch::check`]).
    /// The batch's number is the one it was formed with.
    ///
    /// The shape's rows and tokens are the batch's own. A prefill or refill batch is served by
    /// the scoring variant when a row scores its prompt, by the writing one when no row samples,
    /// greedily when every row that samples is greedy and with a draw otherwise; a decode or a
    /// verification batch greedily only when every row is greedy. A media batch is served by its
    /// phase's variant, and a video request's batch states its first request's extent, by which
    /// the workers find the bucket that holds it.
    ///
    /// # Errors
    ///
    /// The first rule of the batch [`Batch::check`] finds broken, which is a defect of the rows
    /// the step appended.
    pub(super) fn seal(&self, batch: &mut Batch) -> ValidationResult<()> {
        batch.generation = self.generation.load(Ordering::Acquire);
        batch.new_slots.sort_unstable_by_key(|new| new.slot);
        if let Some(shape) = self.shape(batch) {
            batch.shape = shape;
        }
        batch.check()
    }

    /// The shape of [`Scheduler::seal`]; `None` for a batch that names no phase.
    fn shape(&self, batch: &Batch) -> Option<Shape> {
        let columns = &batch.columns;

        // A row is greedy at temperature zero; the rows that sample are those whose logits are
        // read.
        let greedy = |row: usize| {
            columns
                .parameters
                .get(row * PARAMETERS + TEMPERATURE as usize)
                .is_none_or(|temperature| *temperature == 0.0)
        };
        let sampling: Vec<usize> = (0..columns.rows())
            .filter(|row| {
                columns
                    .rows
                    .logits_rows
                    .get(*row)
                    .is_some_and(|logits| *logits >= 0)
            })
            .collect();
        let scores = columns.targets.iter().any(|target| *target >= 0);
        let greedy_endings = sampling.iter().all(|row| greedy(*row));
        let greedy_rows = (0..columns.rows()).all(greedy);

        let variant = match batch.phase {
            Phase::Prefill | Phase::Refill if scores => Variant::PrefillScored,
            Phase::Prefill | Phase::Refill if sampling.is_empty() => Variant::PrefillWrite,
            Phase::Prefill | Phase::Refill if greedy_endings => Variant::PrefillGreedy,
            Phase::Prefill | Phase::Refill => Variant::PrefillOutput,
            Phase::Decode if greedy_rows => Variant::DecodeGreedy,
            Phase::Decode => Variant::DecodeSampled,
            Phase::Verify if greedy_rows => Variant::VerifyGreedy,
            Phase::Verify => Variant::VerifySampled,
            // A media phase is served by its one variant.
            phase => *variants(phase).first()?,
        };

        let first = batch
            .requests
            .first()
            .and_then(|id| self.requests.get(id))
            .map(|request| &request.input);
        let extent = match first {
            Some(proto::request::Request::Diffusion(input)) if variant.media() => Extent::Request {
                frames: input.sampling.num_frames,
                width: input.sampling.width,
                height: input.sampling.height,
                text_rows: u32::try_from(input.prompt_token_ids.len()).ok()?,
                condition_rows: input.sampling.condition_rows,
            },
            _ => Extent::None,
        };
        let rows = if variant.media() {
            batch.media.len()
        } else {
            columns.rows()
        };
        Some(Shape {
            rows: u32::try_from(rows).ok()?,
            tokens: u32::try_from(columns.tokens()).ok()?,
            variant,
            extent,
        })
    }
}

/// The tokens a request sampling with `sampling` may not sample after its `generated` tokens,
/// `count` of them generated in all (those in flight included), followed by `drafts`: its
/// end-of-sequence tokens `eos` while it is under its `min_tokens` floor, and the last token of
/// every bad word whose prefix the generated tokens and the drafts end with (vLLM's
/// `apply_bad_words`, over the output alone). Sorted and deduplicated.
pub(super) fn banned_tokens(
    sampling: &SamplingParams,
    eos: &[u32],
    generated: &[u32],
    count: usize,
    drafts: &[u32],
) -> Vec<u32> {
    let mut banned = Vec::new();
    if count.saturating_add(drafts.len()) < sampling.min_tokens {
        banned.extend_from_slice(eos);
    }

    for word in &sampling.bad_words_ids {
        let Some((&last, prefix)) = word.split_last() else {
            continue;
        };
        // The history is the generated tokens, then the drafts: a prefix longer than the drafts
        // ends in the generated tokens, a shorter one in the drafts alone.
        let ends = if prefix.len() <= drafts.len() {
            drafts.ends_with(prefix)
        } else {
            let (head, tail) = prefix.split_at(prefix.len() - drafts.len());
            tail == drafts && generated.ends_with(head)
        };
        if ends {
            banned.push(last);
        }
    }

    banned.sort_unstable();
    banned.dedup();
    banned
}

/// Appends the input id and rotary position of each of a token request's cache positions in
/// `range`, and each image block the range holds whole, with its first token counted from the
/// range's start. Returns the rotary position at `range.end`, the prompt text tokens before it,
/// and whether the range cuts an image block.
///
/// A text or output token embeds its id. An image block's rows embed 0 and rotate from the
/// image's start. An output token still in flight is -1, and a context that ends with an image
/// block puts `bos` first in its output.
///
/// The request's `token_ids` hold one entry per cache position of its context (`Request::new`):
/// the text between its image blocks, a 0 for each block row, a resumed request's generated
/// tokens; its output follows from `num_prompt_tokens`. A segment of text is read at its cache
/// positions, and the text index of the range's end counts the text alone.
///
/// `None` for a request that is not a token request, or a block held whole whose encoder output
/// has no known position.
fn context(
    request: &Request,
    range: Range<u32>,
    bos: u32,
    ids: &mut Vec<i32>,
    rotary: &mut Vec<i32>,
    vision: &mut Vec<VisionRows>,
) -> Option<(u32, usize, bool)> {
    let input = generation(request)?;
    let prompt = input.prompt_token_ids.len() + input.generated_token_ids.len();
    let output = request
        .token_ids
        .get(usize::try_from(request.num_prompt_tokens).ok()?..)?;

    // Where the next segment starts: its cache and rotary positions, the text and the encoder
    // blocks before it. The rotary position and the text at the range's end, once reached.
    let (mut cache, mut logical, mut written, mut block) = (0_u32, 0_u32, 0_usize, 0_usize);
    let mut end = None;
    let mut cut = false;

    let images = &input.multimodal_inputs.images;
    for image in images.iter().map(Some).chain(iter::once(None)) {
        // The text before the image, or after the last one, at its cache positions.
        let until = image.map_or(prompt, |image| image.position as usize);
        let from = usize::try_from(cache).ok()?;
        let segment = request
            .token_ids
            .get(from..from.checked_add(until.checked_sub(written)?)?)?;
        let count = u32::try_from(segment.len()).ok()?;
        let segment_end = cache.checked_add(count)?;
        if end.is_none() && range.end < segment_end {
            end = Some((
                logical + (range.end - cache),
                written + (range.end - cache) as usize,
            ));
        }
        for position in range.start.max(cache)..range.end.min(segment_end) {
            let offset = position - cache;
            ids.push(i32::try_from(*segment.get(offset as usize)?).ok()?);
            rotary.push(i32::try_from(logical + offset).ok()?);
        }
        cache = segment_end;
        logical = logical.checked_add(count)?;
        written = until;

        let Some(image) = image else {
            break;
        };
        for encoder in &image.encoders {
            let rows = encoder.num_cache_tokens.or(encoder.max_cache_tokens)?;
            let block_end = cache.checked_add(rows)?;
            if end.is_none() && range.end < block_end {
                end = Some((logical, written));
            }
            if range.start < block_end && cache < range.end {
                let (from, to) = (range.start.max(cache), range.end.min(block_end));
                if (from, to) == (cache, block_end) {
                    let feature = request.encodings.get(block)?.1?;
                    vision.push(VisionRows {
                        token: cache - range.start,
                        rows,
                        start: logical,
                        feature,
                    });
                } else {
                    cut = true;
                }
                let start = i32::try_from(logical).ok()?;
                ids.extend(iter::repeat_n(0, (to - from) as usize));
                rotary.extend(iter::repeat_n(start, (to - from) as usize));
            }
            cache = block_end;
            block += 1;
        }
        logical = logical.checked_add(image.num_positions)?;
    }

    // The output, after the beginning-of-sequence token when the context ends with an image.
    let lead = usize::from(ends_with_image(input));
    for position in range.start.max(cache)..range.end {
        let index = (position - cache) as usize;
        let id = match index.checked_sub(lead) {
            None => i32::try_from(bos).ok()?,
            Some(index) => match output.get(index) {
                Some(&token) => i32::try_from(token).ok()?,
                None => -1,
            },
        };
        ids.push(id);
        rotary.push(i32::try_from(logical.checked_add(position - cache)?).ok()?);
    }

    let (after, text_end) = end.unwrap_or((logical + range.end.saturating_sub(cache), prompt));
    Some((after, text_end, cut))
}

/// The token at a token request's cache position `end`, which a refill ending there reads as
/// the token after its last: a text or output token, not one in flight or an image row.
fn lookahead(request: &Request, end: u32, bos: u32) -> Option<u32> {
    let (mut ids, mut images, range) = (Vec::new(), Vec::new(), end..end.checked_add(1)?);
    let (_, _, cut) = context(request, range, bos, &mut ids, &mut Vec::new(), &mut images)?;
    if cut || !images.is_empty() {
        return None;
    }
    u32::try_from(*ids.first()?).ok()
}

/// The row of a token request's image encoding on slot `slot`: the image of its next encoder
/// block to compute, written to that block's encoder output. Encoder input `i`'s output lies at
/// position `i` of the request's encoder-output table, where its admission allocated it when no
/// `Pin` found it (`memory.rs`); `Request::encodings` states the position once the output is
/// held, which is what a prefill reading it waits for, not the encode that computes it.
fn image_row(
    request: &Request,
    input: &GenerationRequest,
    phase: Phase,
    slot: u32,
) -> Option<MediaRow> {
    let block = request.num_encoded_inputs as usize;
    let (image, encoder) = input
        .multimodal_inputs
        .images
        .iter()
        .flat_map(|image| image.encoders.iter().map(move |encoder| (image, encoder)))
        .nth(block)?;
    let encodes = match encoder.encoder {
        ImageEncoder::Vision => Phase::VisionEncode,
        ImageEncoder::Latent => Phase::LatentEncode,
    };
    if phase != encodes {
        return None;
    }
    let output = u32::try_from(block).ok()?;

    Some(MediaRow {
        request: request.id,
        slot,
        conditioning: Vec::new(),
        media: Vec::new(),
        step: 0,
        steps: 0,
        seed: 0,
        width: 0,
        height: 0,
        frames: 0,
        rounds: 0,
        unit: 0,
        rank: None,
        image: Some(Arc::from(image.encoded.as_slice())),
        results: vec![output],
        reads: 0,
    })
}

/// Appends a text encoding's prompt to `columns`: one row of its tokens from rotary position 0,
/// whose last token's hidden state is read. `None`, appending nothing, for a prompt the columns
/// cannot count.
fn push_prompt(columns: &mut Columns, prompt: &[u32]) -> Option<()> {
    let ids = prompt
        .iter()
        .map(|token| i32::try_from(*token).ok())
        .collect::<Option<Vec<_>>>()?;
    let length = i32::try_from(prompt.len()).ok()?;
    let end = i32::try_from(columns.input_ids.len() + prompt.len()).ok()?;

    if columns.rows.query_offsets.is_empty() {
        columns.rows.query_offsets.push(0);
    }
    columns.input_ids.extend(ids);
    columns.rows.positions.extend(0..length);
    columns.rows.seq_lens.push(length);
    columns.rows.query_offsets.push(end);
    columns.rows.logits_rows.push(end - 1);
    Some(())
}

/// The token request a request is, if it is one.
fn generation(request: &Request) -> Option<&GenerationRequest> {
    match &request.input {
        proto::request::Request::Generation(input) => Some(input),
        proto::request::Request::Diffusion(_) => None,
    }
}

/// Whether a token request's context ends with an image block: its last image lies at the end
/// of a prompt to which no generated tokens were appended.
fn ends_with_image(input: &GenerationRequest) -> bool {
    input.generated_token_ids.is_empty()
        && input
            .multimodal_inputs
            .images
            .last()
            .is_some_and(|image| image.position as usize == input.prompt_token_ids.len())
}

/// The sampler's parameter columns of one row, in the sampler's order: temperature, top-k,
/// top-p, min-p, the repetition, frequency and presence penalties, and an eighth that no
/// operator reads.
fn parameters(sampling: &SamplingParams) -> [f32; PARAMETERS] {
    [
        sampling.temperature,
        sampling.top_k as f32,
        sampling.top_p,
        sampling.min_p,
        sampling.repetition_penalty,
        sampling.frequency_penalty,
        sampling.presence_penalty,
        0.0,
    ]
}

/// Whether a request's token-count penalties shape its selection: vLLM's
/// `SamplingParams::uses_penalties`, which the commit table states per sampling row.
fn uses_penalties(sampling: &SamplingParams) -> bool {
    sampling.repetition_penalty != 1.0
        || sampling.frequency_penalty != 0.0
        || sampling.presence_penalty != 0.0
}

/// The KV page size in tokens a token batch states: the block size of the model's first cache
/// group of key and value rows, or of its first cache group; 0 for a group that holds no cache.
fn page_size(capacity: &Capacity) -> u32 {
    let groups = &capacity.cache_groups;
    groups
        .iter()
        .find(|(kind, _)| kind.required_rows())
        .or(groups.first())
        .map_or(0, |&(_, block)| block)
}

/// The block size in tokens of the model's recurrent-state cache group, when it keeps one.
fn state_block(capacity: &Capacity) -> Option<u32> {
    capacity
        .cache_groups
        .iter()
        .find(|(kind, _)| kind.is_state())
        .map(|&(_, block)| block)
        .filter(|block| *block > 0)
}

/// The decoded tokens between two recurrent-state checkpoints of a group that holds a cache: its
/// deployment's `kv_cache_config.state_interval`.
pub(super) fn state_interval(capacity: &Capacity) -> Option<u32> {
    capacity
        .kv_cache
        .as_ref()
        .map(|config| config.state_interval.get())
}

/// Appends one call's cells to a commit table laid out column-major over `calls` calls: cell
/// `c` ends column `c`, which then holds `calls + 1` cells.
fn push_commit(table: &mut Vec<i64>, calls: usize, cells: [i64; COMMIT_FIELDS]) {
    for (column, cell) in cells.into_iter().enumerate() {
        table.insert(column * (calls + 1) + calls, cell);
    }
}

/// Appends `values` to an optional column of `before` entries when it is full: the column stays
/// empty while every value is `absent`, and is filled with `absent` up to `before` once one is
/// not.
fn put<T: Clone + PartialEq>(column: &mut Vec<T>, before: usize, values: Vec<T>, absent: T) {
    if column.is_empty() && values.iter().all(|value| *value == absent) {
        return;
    }
    column.resize(before, absent);
    column.extend(values);
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use crate::batch::cache::CacheGroupKind;
    use crate::batch::philox::{DrawAddress, request_sampling_key};
    use crate::batch::{
        Batch, BatchId, Columns, DraftWork, Extent, MediaRow, NewSlot, Phase, Row, Shape,
        StateCopy, StateDirection, Variant, VisionRows,
    };
    use proto::config::{BlockConfig, BlockDistribution};
    use proto::module::commit::{self, COMMIT_FIELDS};
    use proto::request::{
        ConditionExtent, DiffusionRequest, DiffusionSamplingParams, GenerationRequest,
        ImageEncoder, ImageEncoderInput, ImageInput, RequestId, SamplingParams,
    };
    use storage::Key;
    use storage::runtime::{GroupId, OwnerId};

    use super::super::tests::bench;
    use super::super::{Request, RequestStatus, Scheduler};
    use super::banned_tokens;

    const A: GroupId = GroupId(0);
    const BOS: u32 = 151;
    const EOS: u32 = 2;
    /// Tokens per block of every cache group.
    const BLOCK: u32 = 4;

    /// A scheduler whose groups serve a model with a full-attention cache group of blocks of four
    /// tokens, then `groups`, and four denoising steps.
    fn scheduler(groups: &[(CacheGroupKind, u32)]) -> Scheduler {
        let mut scheduler = bench(1024).scheduler;
        for group in &mut scheduler.groups {
            let capacity = &mut group.capacity;
            // A group that holds a cache states its deployment's cache configuration, the
            // default's here.
            capacity.kv_cache = Some(proto::config::KvCacheConfig::default());
            capacity.cache_groups = [(CacheGroupKind::Full, BLOCK)]
                .into_iter()
                .chain(groups.iter().copied())
                .collect();
            capacity.bos_token_id = BOS;
            capacity.eos_token_ids = vec![EOS];
            capacity.denoise_steps = 4;
        }
        scheduler
    }

    /// Sampling parameters that differ per column, with penalties.
    fn sampling() -> SamplingParams {
        SamplingParams {
            temperature: 0.5,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.05,
            repetition_penalty: 1.1,
            frequency_penalty: 0.2,
            presence_penalty: 0.3,
            ..SamplingParams::default()
        }
    }

    /// The sampling of a request whose penalties count nothing.
    fn neutral() -> SamplingParams {
        SamplingParams {
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            ..sampling()
        }
    }

    /// Token request `id` with prompt `prompt` on slot `slot` of group A, which has served it,
    /// with the first `computed` positions of its context computed.
    fn request(id: u64, prompt: Vec<u32>, slot: u32, computed: u32) -> Request {
        let input = GenerationRequest {
            request_id: RequestId(id),
            prompt_token_ids: prompt.clone(),
            multimodal_inputs: Default::default(),
            sampling: sampling(),
            max_und_tokens: 16,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache: Default::default(),
            include_stop_token: false,
            generated_token_ids: Vec::new(),
            adapter: 0,
        };
        let prompt_tokens = u32::try_from(prompt.len()).unwrap();
        row(id, input.into(), prompt, prompt_tokens, slot, computed)
    }

    /// A running request of `input` on slot `slot` of group A, which has served it.
    fn row(
        id: u64,
        input: proto::request::Request,
        token_ids: Vec<u32>,
        prompt_tokens: u32,
        slot: u32,
        computed: u32,
    ) -> Request {
        let now = Instant::now();
        Request {
            id: RequestId(id),
            input,
            arrival: now,
            status: RequestStatus::Running,
            placement: [Some(A); Phase::COUNT],
            attachments: vec![(A, OwnerId(id), slot, true)],
            scheduled: [0; Phase::COUNT],
            completed: [0; Phase::COUNT],
            early: Vec::new(),
            blocks: Vec::new(),
            queued: false,
            pending: 0,
            token_ids,
            num_prompt_tokens: prompt_tokens,
            num_computed_tokens: computed,
            num_cached_tokens: 0,
            num_host_tokens: 0,
            num_peer_tokens: 0,
            listing: 0,
            keys: Vec::new(),
            spec_token_ids: Vec::new(),
            num_draft_tokens: 0,
            num_encoded_inputs: 0,
            encodings: Vec::new(),
            unpinned: Vec::new(),
            crossings: Vec::new(),
            num_acknowledged_tokens: 0,
            terminal: None,
            event_credit: 64,
            event_window: 64,
            draft_limit: 0,
            last_progress: now,
            dirty: false,
        }
    }

    /// The token request a test changes.
    fn input(request: &mut Request) -> &mut GenerationRequest {
        match &mut request.input {
            proto::request::Request::Generation(input) => input,
            proto::request::Request::Diffusion(_) => unreachable!("a token request"),
        }
    }

    /// An image at prompt position `position` whose one encoder writes `rows` rows and which
    /// takes two logical positions.
    fn image(position: u32, encoder: ImageEncoder, rows: u32) -> ImageInput {
        ImageInput {
            digest: [0; 32],
            encoded: b"image".to_vec(),
            position,
            num_positions: 2,
            encoders: vec![ImageEncoderInput {
                encoder,
                num_cache_tokens: Some(rows),
                max_cache_tokens: None,
                height: None,
                width: None,
            }],
        }
    }

    /// An empty batch of `phase` on group A.
    fn batch(phase: Phase) -> Batch {
        Batch {
            id: BatchId(1),
            group: A,
            phase,
            generation: 0,
            shape: Shape::decode(0, Variant::DecodeGreedy),
            requests: Vec::new(),
            columns: Columns::default(),
            media: Vec::new(),
            new_slots: Vec::new(),
        }
    }

    /// One cell of the commit table, which is column-major over the batch's calls.
    fn cell(batch: &Batch, column: usize, call: usize) -> i64 {
        batch.columns.commit[column * batch.columns.calls() + call]
    }

    /// A batch of one row of `phase` and `tokens` tokens for `request`.
    fn one(scheduler: &mut Scheduler, request: Request, phase: Phase, tokens: u32) -> Batch {
        let id = request.id;
        scheduler.requests.insert(id, request);
        let mut batch = batch(phase);
        assert!(scheduler.push_token_rows(&mut batch, id, phase, tokens));
        batch
    }

    #[test]
    fn two_prefill_chunks_are_two_rows_at_their_own_positions() {
        let mut scheduler = scheduler(&[]);
        // Request 1 continues its prompt of 16 tokens at 8; request 2 ends its prompt of 14.
        scheduler
            .requests
            .insert(RequestId(1), request(1, (1..=16).collect(), 1, 8));
        scheduler
            .requests
            .insert(RequestId(2), request(2, (21..=34).collect(), 2, 12));
        let mut batch = batch(Phase::Prefill);
        assert!(scheduler.push_token_rows(&mut batch, RequestId(1), Phase::Prefill, 4));
        assert!(scheduler.push_token_rows(&mut batch, RequestId(2), Phase::Prefill, 2));

        let rows = &batch.columns.rows;
        let row = |slot, rows| Row {
            slot,
            kind: Phase::Prefill,
            cache_group: 0,
            rows,
        };
        assert_eq!(rows.calls, vec![row(1, 0..1), row(2, 1..2)]);
        // A text token's cache position is its position in its request, and it rotates from it.
        assert_eq!(rows.positions, vec![8, 9, 10, 11, 12, 13]);
        assert_eq!(rows.cache_positions, vec![8, 9, 10, 11, 12, 13]);
        assert_eq!(rows.seq_lens, vec![12, 14]);
        assert_eq!(rows.query_offsets, vec![0, 4, 6]);
        // Only the chunk that ends its prompt samples, at its own row.
        assert_eq!(rows.logits_rows, vec![-1, 1]);
        assert_eq!(batch.columns.input_ids, vec![9, 10, 11, 12, 33, 34]);
        assert_eq!(batch.columns.slots, vec![1, 2]);
        assert_eq!(
            (batch.columns.page_size, batch.columns.max_kv_len),
            (BLOCK, 4096)
        );
        assert_eq!(batch.requests, vec![RequestId(1), RequestId(2)]);
        let tasks: Vec<i64> = (0..2)
            .map(|call| cell(&batch, commit::column::TASK, call))
            .collect();
        assert_eq!(tasks, vec![-1, 0]);

        scheduler.seal(&mut batch).unwrap();
        assert_eq!(batch.shape, Shape::prefill(2, 6, Variant::PrefillOutput));
    }

    #[test]
    fn a_prefill_that_samples_installs_its_token_at_the_host_coordinates() {
        let mut scheduler = scheduler(&[]);
        let batch = one(
            &mut scheduler,
            request(1, (1..=12).collect(), 1, 8),
            Phase::Prefill,
            4,
        );

        assert_eq!(
            cell(&batch, commit::column::MODE, 0),
            commit::mode::EXPLICIT
        );
        assert_eq!(cell(&batch, commit::column::CACHE_LENGTH, 0), 12);
        assert_eq!(
            cell(&batch, commit::column::LOGICAL, 0),
            12,
            "the position after the prompt"
        );
        assert_eq!(
            cell(&batch, commit::column::SAMPLING, 0),
            1,
            "the request's draws after the one this prefill makes"
        );
        assert_eq!(cell(&batch, commit::column::VISIBLE, 0), 0);
        assert_eq!(cell(&batch, commit::column::TASK, 0), 0);
        assert_eq!(cell(&batch, commit::column::PENALTIES, 0), 1);
        assert_eq!(batch.columns.draws[0].position, 12);
    }

    #[test]
    fn a_prefill_that_samples_nothing_still_states_its_cache_extent() {
        let mut scheduler = scheduler(&[]);
        let mut chunk = request(1, (1..=16).collect(), 1, 8);
        input(&mut chunk).sampling = neutral();
        let mut batch = one(&mut scheduler, chunk, Phase::Prefill, 2);

        assert_eq!(cell(&batch, commit::column::MODE, 0), commit::mode::CACHE);
        assert_eq!(cell(&batch, commit::column::CACHE_LENGTH, 0), 10);
        assert_eq!(cell(&batch, commit::column::TASK, 0), -1);
        for column in [
            commit::column::LOGICAL,
            commit::column::SAMPLING,
            commit::column::VISIBLE,
            commit::column::PENALTIES,
        ] {
            assert_eq!(cell(&batch, column, 0), 0);
        }
        assert_eq!(batch.columns.rows.logits_rows, vec![-1]);

        scheduler.seal(&mut batch).unwrap();
        assert_eq!(batch.shape.variant, Variant::PrefillWrite);
    }

    #[test]
    fn a_decode_advances_its_own_slot_and_states_only_its_task() {
        let mut scheduler = scheduler(&[]);
        let mut decoding = request(1, (1..=12).collect(), 3, 12);
        decoding.token_ids.push(42);
        let mut batch = one(&mut scheduler, decoding, Phase::Decode, 1);

        // The row embeds the token its prompt sampled, at the position after the prompt.
        let rows = &batch.columns.rows;
        assert_eq!(batch.columns.input_ids, vec![42]);
        assert_eq!(rows.positions, vec![12]);
        assert_eq!(rows.cache_positions, vec![12]);
        assert_eq!(rows.seq_lens, vec![13]);
        assert_eq!(rows.query_offsets, vec![0, 1]);
        assert_eq!(rows.logits_rows, vec![0]);
        assert_eq!(cell(&batch, commit::column::SLOT, 0), 3);
        assert_eq!(cell(&batch, commit::column::MODE, 0), commit::mode::DECODE);
        assert_eq!(cell(&batch, commit::column::TASK, 0), 0);
        for column in [
            commit::column::CACHE_LENGTH,
            commit::column::LOGICAL,
            commit::column::SAMPLING,
            commit::column::VISIBLE,
        ] {
            assert_eq!(
                cell(&batch, column, 0),
                0,
                "the device advances a decode's lengths from its own state"
            );
        }
        assert_eq!(cell(&batch, commit::column::PENALTIES, 0), 1);
        assert_eq!(batch.columns.commit.len(), COMMIT_FIELDS);
        assert_eq!(batch.columns.draws[0].position, 13);

        scheduler.seal(&mut batch).unwrap();
        assert_eq!(batch.shape, Shape::decode(1, Variant::DecodeSampled));
    }

    /// A decode behind one in flight reads the token its slot sampled last, and its floor of
    /// `min_tokens` counts the decode in flight.
    #[test]
    fn a_pipelined_decode_reads_the_token_its_slot_sampled_last() {
        let mut scheduler = scheduler(&[]);
        let mut decoding = request(1, (1..=12).collect(), 1, 12);
        decoding.token_ids.push(42);
        input(&mut decoding).sampling.min_tokens = 2;
        let first = one(&mut scheduler, decoding, Phase::Decode, 1);
        assert_eq!(first.columns.banned, vec![vec![EOS]]);

        let request = scheduler.requests.get_mut(&RequestId(1)).unwrap();
        request.num_computed_tokens = 13;
        request.scheduled[Phase::Decode.index()] = 1;
        let mut second = batch(Phase::Decode);
        assert!(scheduler.push_token_rows(&mut second, RequestId(1), Phase::Decode, 1));
        assert_eq!(second.columns.input_ids, vec![-1]);
        assert_eq!(second.columns.rows.positions, vec![13]);
        assert_eq!(second.columns.rows.cache_positions, vec![13]);
        assert!(second.columns.banned.is_empty());
        scheduler.seal(&mut second).unwrap();
    }

    #[test]
    fn a_verification_is_one_row_of_its_committed_token_and_its_drafts() {
        let mut scheduler = scheduler(&[(CacheGroupKind::Draft, BLOCK)]);
        scheduler.config.draft_tokens = 2;
        let mut verifying = request(1, (1..=8).collect(), 1, 8);
        verifying.token_ids.push(50);
        verifying.spec_token_ids = vec![61, 62, 63];
        verifying.num_draft_tokens = 8;
        let mut batch = one(&mut scheduler, verifying, Phase::Verify, 3);

        let rows = &batch.columns.rows;
        assert_eq!(
            rows.calls[0].rows,
            0..1,
            "one row holds the token and its drafts"
        );
        assert_eq!(batch.columns.input_ids, vec![50, 61, 62]);
        assert_eq!(rows.positions, vec![8, 9, 10]);
        assert_eq!(rows.cache_positions, vec![8, 9, 10]);
        assert_eq!(rows.seq_lens, vec![11]);
        assert_eq!(rows.query_offsets, vec![0, 3]);
        assert_eq!(rows.logits_rows, vec![0]);
        assert_eq!(cell(&batch, commit::column::MODE, 0), commit::mode::VERIFY);
        assert_eq!(
            cell(&batch, commit::column::VISIBLE, 0),
            8,
            "the base its accepted count is added to"
        );
        assert_eq!(cell(&batch, commit::column::LOGICAL, 0), 8);
        assert_eq!(
            cell(&batch, commit::column::SAMPLING, 0),
            1,
            "the draws so far: the prompt's token"
        );
        assert_eq!(cell(&batch, commit::column::CACHE_LENGTH, 0), 0);
        assert_eq!(cell(&batch, commit::column::TASK, 0), 0);
        assert_eq!(
            batch.columns.drafts,
            vec![DraftWork {
                proposals: 2,
                lookahead: None,
                refill: None,
            }]
        );
        assert_eq!(batch.columns.draws[0].position, 11);

        scheduler.seal(&mut batch).unwrap();
        assert_eq!(batch.shape, Shape::prefill(1, 3, Variant::VerifySampled));

        // A verification of more drafts than the request holds is not stated.
        let mut more = self::batch(Phase::Verify);
        assert!(!scheduler.push_token_rows(&mut more, RequestId(1), Phase::Verify, 5));
        assert_eq!(more, self::batch(Phase::Verify));
    }

    #[test]
    fn tasks_are_numbered_over_the_sampling_calls_only() {
        let mut scheduler = scheduler(&[]);
        // A chunk that ends no prompt, then two that end theirs.
        scheduler
            .requests
            .insert(RequestId(1), request(1, (1..=16).collect(), 1, 8));
        scheduler
            .requests
            .insert(RequestId(2), request(2, (1..=10).collect(), 2, 8));
        scheduler
            .requests
            .insert(RequestId(3), request(3, (1..=12).collect(), 3, 10));
        let mut batch = batch(Phase::Prefill);
        for id in 1..=3 {
            assert!(scheduler.push_token_rows(&mut batch, RequestId(id), Phase::Prefill, 2));
        }

        let column = |column| {
            (0..3)
                .map(|call| cell(&batch, column, call))
                .collect::<Vec<_>>()
        };
        assert_eq!(column(commit::column::TASK), vec![-1, 0, 1]);
        assert_eq!(column(commit::column::SLOT), vec![1, 2, 3]);
        assert_eq!(column(commit::column::CACHE_LENGTH), vec![10, 10, 12]);
        assert_eq!(batch.columns.commit.len(), COMMIT_FIELDS * 3);
        scheduler.seal(&mut batch).unwrap();
    }

    /// A request a group has not served starts its slot there: its penalties count the tokens it
    /// generated before (on another group, on another engine before it resumed), a request whose
    /// penalties count nothing starts at none, and the two ids before its row are its context.
    #[test]
    fn a_new_slot_counts_the_tokens_its_request_generated_before() {
        let mut scheduler = scheduler(&[]);
        let mut first = request(1, (1..=8).collect(), 4, 9);
        first.token_ids.extend([42, 7]);
        first.attachments[0].3 = false;
        let mut second = request(2, (11..=18).collect(), 3, 8);
        second.token_ids.push(43);
        second.attachments[0].3 = false;
        input(&mut second).sampling = neutral();
        scheduler.requests.insert(RequestId(1), first);
        scheduler.requests.insert(RequestId(2), second);

        let mut decode = batch(Phase::Decode);
        assert!(scheduler.push_token_rows(&mut decode, RequestId(1), Phase::Decode, 1));
        assert!(scheduler.push_token_rows(&mut decode, RequestId(2), Phase::Decode, 1));
        scheduler.seal(&mut decode).unwrap();
        assert_eq!(
            decode.new_slots,
            vec![
                NewSlot {
                    slot: 3,
                    history: Vec::new(),
                    bias: Vec::new(),
                    allowed: None,
                    context: [17, 18],
                },
                NewSlot {
                    slot: 4,
                    history: vec![42, 7],
                    bias: Vec::new(),
                    allowed: None,
                    context: [8, 42],
                },
            ],
            "slots in increasing order, each counting its own request's history"
        );

        // The group now serves request 1: its next row starts nothing.
        scheduler
            .requests
            .get_mut(&RequestId(1))
            .unwrap()
            .num_computed_tokens = 10;
        let mut next = batch(Phase::Decode);
        assert!(scheduler.push_token_rows(&mut next, RequestId(1), Phase::Decode, 1));
        assert!(next.new_slots.is_empty());

        // A resumed request's generated tokens continue its prompt and count in its penalties;
        // before its first token it has no context.
        let mut resumed = request(5, vec![1, 2, 3], 5, 0);
        input(&mut resumed).generated_token_ids = vec![5, 6];
        resumed.token_ids.extend([5, 6]);
        resumed.num_prompt_tokens = 5;
        resumed.attachments[0].3 = false;
        let batch = one(&mut scheduler, resumed, Phase::Prefill, 5);
        assert_eq!(batch.columns.input_ids, vec![1, 2, 3, 5, 6]);
        assert_eq!(batch.new_slots[0].history, vec![5, 6]);
        assert_eq!(batch.new_slots[0].context, [BOS, BOS]);
        assert_eq!(batch.columns.rows.logits_rows, vec![0]);
    }

    /// A request with a logit bias or an allowed set starts its slot's bias row in canonical
    /// order and flags each of its rows; its rows ban the end-of-sequence token under its floor
    /// and its bad words; a request with neither, beside it, states nothing, and a batch of such
    /// requests alone states no adjustment.
    #[test]
    fn a_batch_states_the_logit_adjustments_its_requests_ask_for() {
        let mut scheduler = scheduler(&[]);
        let mut restricted = request(1, (1..=12).collect(), 3, 12);
        restricted.token_ids.push(42);
        restricted.attachments[0].3 = false;
        input(&mut restricted).sampling = SamplingParams {
            // Out of order and repeated, as a client may send them.
            logit_bias: vec![(7, 1.5), (5, -2.0)],
            allowed_token_ids: Some(vec![7, 5, 7, 11]),
            min_tokens: 4,
            bad_words_ids: vec![vec![9]],
            ..sampling()
        };
        let mut plain = request(2, (1..=12).collect(), 4, 12);
        plain.token_ids.push(43);
        plain.attachments[0].3 = false;
        scheduler.requests.insert(RequestId(1), restricted);
        scheduler.requests.insert(RequestId(2), plain);

        let mut batch = batch(Phase::Decode);
        assert!(scheduler.push_token_rows(&mut batch, RequestId(1), Phase::Decode, 1));
        assert!(scheduler.push_token_rows(&mut batch, RequestId(2), Phase::Decode, 1));
        scheduler.seal(&mut batch).unwrap();
        assert_eq!(batch.new_slots[0].bias, vec![(5, -2.0), (7, 1.5)]);
        assert_eq!(batch.new_slots[0].allowed, Some(vec![5, 7, 11]));
        assert_eq!(batch.new_slots[1].bias, Vec::new());
        assert_eq!(batch.columns.biased, vec![true, false]);
        assert_eq!(batch.columns.banned, vec![vec![EOS, 9], Vec::new()]);

        // Once the slot holds the row, the request still reads it, and starts nothing.
        let mut running = self::batch(Phase::Decode);
        assert!(scheduler.push_token_rows(&mut running, RequestId(1), Phase::Decode, 1));
        assert!(scheduler.push_token_rows(&mut running, RequestId(2), Phase::Decode, 1));
        assert!(running.new_slots.is_empty());
        assert_eq!(running.columns.biased, vec![true, false]);

        // Requests that ask for none of them state nothing.
        let mut quiet = self::batch(Phase::Decode);
        assert!(scheduler.push_token_rows(&mut quiet, RequestId(2), Phase::Decode, 1));
        assert!(quiet.columns.biased.is_empty() && quiet.columns.banned.is_empty());
    }

    /// A seeded request's draw is addressed by its seed and the position of the token it samples:
    /// the same request under another identity draws the same value, and the next position
    /// another.
    #[test]
    fn a_seeded_row_draws_at_its_seed_and_sampled_position() {
        let draw = |id: u64, position: u32| -> DrawAddress {
            let mut scheduler = scheduler(&[]);
            let mut decoding = request(id, (1..=position).collect(), 3, position);
            decoding.token_ids.push(42);
            input(&mut decoding).sampling.seed = Some(7);
            one(&mut scheduler, decoding, Phase::Decode, 1)
                .columns
                .draws[0]
        };

        let at_twelve = draw(1, 12);
        // The decode of the token at position 12 samples position 13.
        assert_eq!(at_twelve.position, 13);
        assert_eq!(at_twelve.key, request_sampling_key(Some(7), 0, 0, 0));
        assert_eq!(draw(9, 12).uniform(), at_twelve.uniform());
        assert_ne!(draw(1, 13).uniform(), at_twelve.uniform());
    }

    /// An image block's rows sit in their chunk between the text around them: they embed no id,
    /// take the cache positions after the text before them, rotate from the logical position
    /// after it, and the text after them continues the image's span later.
    #[test]
    fn an_image_block_takes_cache_rows_and_its_span_of_positions() {
        let mut scheduler = scheduler(&[]);
        // Text 0..2, an image of four rows, text 2..5, an image of four rows, text 5..8: sixteen
        // cache positions, twelve logical ones.
        let mut context = request(1, (1..=8).collect(), 1, 6);
        input(&mut context).multimodal_inputs.images = vec![
            image(2, ImageEncoder::Vision, 4),
            image(5, ImageEncoder::Vision, 4),
        ];
        // One entry per cache position, each image row 0 (`Request::new`).
        context.token_ids = vec![1, 2, 0, 0, 0, 0, 3, 4, 5, 0, 0, 0, 0, 6, 7, 8];
        context.num_prompt_tokens = 16;
        context.encodings = vec![(Key([1; 32]), Some(4)), (Key([2; 32]), Some(5))];
        let batch = one(&mut scheduler, context, Phase::Prefill, 10);

        assert_eq!(batch.columns.input_ids, vec![3, 4, 5, 0, 0, 0, 0, 6, 7, 8]);
        assert_eq!(
            batch.columns.rows.positions,
            vec![4, 5, 6, 7, 7, 7, 7, 9, 10, 11]
        );
        assert_eq!(
            batch.columns.rows.cache_positions,
            (6..16).collect::<Vec<i32>>()
        );
        assert_eq!(
            batch.columns.vision,
            vec![VisionRows {
                token: 3,
                rows: 4,
                start: 7,
                feature: 5,
            }]
        );
        assert_eq!(cell(&batch, commit::column::CACHE_LENGTH, 0), 16);
        assert_eq!(
            cell(&batch, commit::column::LOGICAL, 0),
            12,
            "the sampled token follows the image's span"
        );
        assert_eq!(batch.columns.draws[0].position, 12);

        // A chunk that ends inside the block, and a block whose encoder output has no position,
        // are no row.
        let mut cut = self::batch(Phase::Prefill);
        assert!(!scheduler.push_token_rows(&mut cut, RequestId(1), Phase::Prefill, 5));
        scheduler.requests.get_mut(&RequestId(1)).unwrap().encodings[1].1 = None;
        assert!(!scheduler.push_token_rows(&mut cut, RequestId(1), Phase::Prefill, 10));
        assert_eq!(cut, self::batch(Phase::Prefill));
    }

    /// A context that ends with an image block samples nothing from its prompt, and its first
    /// decode writes the beginning-of-sequence token after the image's span.
    #[test]
    fn a_context_that_ends_with_an_image_decodes_from_its_first_token() {
        let mut scheduler = scheduler(&[]);
        let mut context = request(1, vec![1, 2, 3], 1, 0);
        input(&mut context).multimodal_inputs.images = vec![image(3, ImageEncoder::Vision, 2)];
        context.token_ids = vec![1, 2, 3, 0, 0];
        context.num_prompt_tokens = 5;
        context.encodings = vec![(Key([1; 32]), Some(0))];
        let prefill = one(&mut scheduler, context, Phase::Prefill, 5);
        assert_eq!(prefill.columns.rows.logits_rows, vec![-1]);
        assert_eq!(cell(&prefill, commit::column::MODE, 0), commit::mode::CACHE);

        scheduler
            .requests
            .get_mut(&RequestId(1))
            .unwrap()
            .num_computed_tokens = 5;
        let mut decode = batch(Phase::Decode);
        assert!(scheduler.push_token_rows(&mut decode, RequestId(1), Phase::Decode, 1));
        assert_eq!(decode.columns.input_ids, vec![i32::try_from(BOS).unwrap()]);
        assert_eq!(decode.columns.rows.positions, vec![5]);
        assert_eq!(decode.columns.rows.cache_positions, vec![5]);
        assert_eq!(decode.columns.draws[0].position, 6);

        // Its first sampled token follows the beginning-of-sequence token.
        let request = scheduler.requests.get_mut(&RequestId(1)).unwrap();
        request.token_ids.push(70);
        request.num_computed_tokens = 6;
        let mut next = batch(Phase::Decode);
        assert!(scheduler.push_token_rows(&mut next, RequestId(1), Phase::Decode, 1));
        assert_eq!(next.columns.input_ids, vec![70]);
        assert_eq!(next.columns.rows.positions, vec![6]);
    }

    /// A prefill that scores its prompt names, for each of its tokens, the prompt token after it:
    /// the next input, then the next chunk's first, and nothing after the prompt's last; its
    /// batch is served by the scoring variant. A prefill beside it that scores nothing names none,
    /// and a batch of such prefills alone states no column.
    #[test]
    fn a_scored_prefill_names_the_prompt_token_after_each_of_its_tokens() {
        let mut scheduler = scheduler(&[]);
        let scored = |id, computed| {
            let mut scored = request(
                id,
                (11..=15).collect(),
                u32::try_from(id).unwrap(),
                computed,
            );
            input(&mut scored).sampling.return_prompt_logprobs = true;
            scored
        };
        scheduler.requests.insert(RequestId(1), scored(1, 0));
        scheduler.requests.insert(RequestId(2), scored(2, 3));
        scheduler
            .requests
            .insert(RequestId(3), request(3, vec![1, 1], 3, 0));
        let mut batch = batch(Phase::Prefill);
        assert!(scheduler.push_token_rows(&mut batch, RequestId(1), Phase::Prefill, 3));
        assert!(scheduler.push_token_rows(&mut batch, RequestId(2), Phase::Prefill, 2));
        assert!(scheduler.push_token_rows(&mut batch, RequestId(3), Phase::Prefill, 2));
        assert_eq!(batch.columns.targets, vec![12, 13, 14, 15, -1, -1, -1]);
        scheduler.seal(&mut batch).unwrap();
        assert_eq!(batch.shape.variant, Variant::PrefillScored);

        let mut unscored = self::batch(Phase::Prefill);
        assert!(scheduler.push_token_rows(&mut unscored, RequestId(3), Phase::Prefill, 2));
        assert!(unscored.columns.targets.is_empty());
        scheduler.seal(&mut unscored).unwrap();
        assert_eq!(unscored.shape.variant, Variant::PrefillOutput);
    }

    #[test]
    fn a_row_states_its_sampling_parameters_in_the_samplers_order() {
        let mut scheduler = scheduler(&[]);
        let mut batch = one(
            &mut scheduler,
            request(1, (1..=4).collect(), 1, 0),
            Phase::Prefill,
            4,
        );
        assert_eq!(
            batch.columns.parameters,
            vec![0.5, 40.0, 0.9, 0.05, 1.1, 0.2, 0.3, 0.0],
            "temperature, top-k, top-p, min-p, repetition, frequency, presence"
        );
        scheduler.seal(&mut batch).unwrap();
        assert_eq!(
            (batch.shape.rows, batch.shape.tokens),
            (1, 4),
            "the shape's counts are the columns' own"
        );
    }

    /// One batch is served by one variant: greedily only when every row that samples is greedy,
    /// with a draw as soon as one draws.
    #[test]
    fn a_batch_is_served_greedily_only_when_no_row_draws() {
        let variant = |phase: Phase, temperatures: &[f32]| {
            let mut scheduler = scheduler(&[(CacheGroupKind::Draft, BLOCK)]);
            scheduler.config.draft_tokens = 2;
            let mut batch = batch(phase);
            for (id, temperature) in (1..).zip(temperatures) {
                let mut decoding = request(id, (1..=8).collect(), u32::try_from(id).unwrap(), 8);
                decoding.token_ids.push(9);
                input(&mut decoding).sampling.temperature = *temperature;
                scheduler.requests.insert(RequestId(id), decoding);
                assert!(scheduler.push_token_rows(&mut batch, RequestId(id), phase, 1));
            }
            scheduler.seal(&mut batch).unwrap();
            batch.shape.variant
        };

        assert_eq!(variant(Phase::Decode, &[0.0]), Variant::DecodeGreedy);
        assert_eq!(variant(Phase::Decode, &[0.5]), Variant::DecodeSampled);
        assert_eq!(variant(Phase::Decode, &[0.0, 0.5]), Variant::DecodeSampled);
        assert_eq!(variant(Phase::Verify, &[0.0, 0.0]), Variant::VerifyGreedy);
        assert_eq!(variant(Phase::Verify, &[0.0, 0.5]), Variant::VerifySampled);

        // A prefill whose prompt ending is greedy takes no draw, whatever its other rows state.
        let mut scheduler = scheduler(&[]);
        let mut ending = request(1, (1..=4).collect(), 1, 0);
        input(&mut ending).sampling.temperature = 0.0;
        scheduler.requests.insert(RequestId(1), ending);
        scheduler
            .requests
            .insert(RequestId(2), request(2, (1..=8).collect(), 2, 0));
        let mut batch = batch(Phase::Prefill);
        assert!(scheduler.push_token_rows(&mut batch, RequestId(1), Phase::Prefill, 4));
        assert!(scheduler.push_token_rows(&mut batch, RequestId(2), Phase::Prefill, 4));
        scheduler.seal(&mut batch).unwrap();
        assert_eq!(batch.shape.variant, Variant::PrefillGreedy);
    }

    /// A refill evaluates tokens the cache holds from the end of the draft cache: it writes no
    /// cache position and samples nothing, and its draft work names where it starts and the token
    /// after its last.
    #[test]
    fn a_refill_writes_no_cache_and_samples_nothing() {
        let mut scheduler = scheduler(&[(CacheGroupKind::Draft, BLOCK)]);
        scheduler.config.draft_tokens = 2;
        let mut refilling = request(1, (1..=8).collect(), 1, 10);
        refilling.token_ids.extend([50, 51, 52]);
        refilling.num_draft_tokens = 4;
        let mut batch = one(&mut scheduler, refilling, Phase::Refill, 4);

        assert_eq!(batch.columns.rows.calls[0].kind, Phase::Refill);
        assert_eq!(batch.columns.input_ids, vec![5, 6, 7, 8]);
        assert_eq!(batch.columns.rows.positions, vec![4, 5, 6, 7]);
        assert_eq!(batch.columns.rows.cache_positions, vec![-1; 4]);
        assert_eq!(batch.columns.rows.seq_lens, vec![8]);
        assert_eq!(batch.columns.rows.logits_rows, vec![-1]);
        assert_eq!(cell(&batch, commit::column::MODE, 0), commit::mode::CACHE);
        assert_eq!(
            cell(&batch, commit::column::CACHE_LENGTH, 0),
            10,
            "the target's cache is left as it is"
        );
        assert_eq!(cell(&batch, commit::column::TASK, 0), -1);
        assert_eq!(
            batch.columns.drafts,
            vec![DraftWork {
                proposals: 0,
                lookahead: Some(50),
                refill: Some(4),
            }]
        );
        scheduler.seal(&mut batch).unwrap();
        assert_eq!(batch.shape.variant, Variant::PrefillWrite);
    }

    /// The first row on a slot that continues from a cached prefix restores the checkpoint its
    /// prefix ends at, and a row keeps the state at the boundary it reaches: a prompt chunk at
    /// its last block boundary, a decode only at a multiple of the state interval. Each names
    /// the checkpoint by the position of the block that ends at its boundary.
    #[test]
    fn a_state_group_restores_a_resumed_prefix_and_keeps_the_boundary_it_reaches() {
        let mut scheduler = scheduler(&[(CacheGroupKind::State, BLOCK)]);
        let mut resumed = request(1, (1..=18).collect(), 1, 8);
        resumed.attachments[0].3 = false;
        let mut short = request(2, (1..=18).collect(), 2, 4);
        short.attachments[0].3 = false;
        scheduler.requests.insert(RequestId(1), resumed);
        scheduler.requests.insert(RequestId(2), short);
        let mut batch = batch(Phase::Prefill);
        assert!(scheduler.push_token_rows(&mut batch, RequestId(1), Phase::Prefill, 8));
        assert!(scheduler.push_token_rows(&mut batch, RequestId(2), Phase::Prefill, 2));
        let copy = |slot, position, direction| StateCopy {
            slot,
            position,
            direction,
        };
        assert_eq!(batch.columns.checkpoints, vec![16, 0]);
        assert_eq!(
            batch.columns.state_copies,
            vec![
                copy(1, 1, StateDirection::Restore),
                copy(2, 0, StateDirection::Restore),
                copy(1, 3, StateDirection::Keep),
            ],
            "every restore before every checkpoint"
        );
        scheduler.seal(&mut batch).unwrap();

        // A slot that holds the state restores nothing.
        let held = one(
            &mut scheduler,
            request(3, (1..=18).collect(), 3, 8),
            Phase::Prefill,
            8,
        );
        assert_eq!(
            held.columns.state_copies,
            vec![copy(3, 3, StateDirection::Keep)]
        );

        // A decode keeps the state at 256, the default interval, and not at 260.
        let decode = |scheduler: &mut Scheduler, id, at: u32| {
            let mut decoding = request(id, (1..=at).collect(), 4, at);
            decoding.token_ids.push(42);
            one(scheduler, decoding, Phase::Decode, 1)
        };
        let kept = decode(&mut scheduler, 4, 255);
        assert_eq!(kept.columns.checkpoints, vec![256]);
        assert_eq!(
            kept.columns.state_copies,
            vec![copy(4, 63, StateDirection::Keep)]
        );
        let passed = decode(&mut scheduler, 5, 259);
        assert!(passed.columns.checkpoints.is_empty() && passed.columns.state_copies.is_empty());

        // A prefix that ends inside a block has no checkpoint to restore.
        let mut inside = request(6, (1..=18).collect(), 6, 6);
        inside.attachments[0].3 = false;
        scheduler.requests.insert(RequestId(6), inside);
        let mut refused = self::batch(Phase::Prefill);
        assert!(!scheduler.push_token_rows(&mut refused, RequestId(6), Phase::Prefill, 4));
    }

    /// A call the request's state does not state appends nothing: a media phase as a token row,
    /// a prefill past its prompt, a decode of more than one token, a request with no slot on the
    /// group, and a video request.
    #[test]
    fn a_row_the_request_cannot_state_appends_nothing() {
        let mut scheduler = scheduler(&[]);
        scheduler
            .requests
            .insert(RequestId(1), request(1, (1..=8).collect(), 1, 4));
        scheduler
            .requests
            .insert(RequestId(2), request(2, (1..=8).collect(), 0, 4));
        scheduler.requests.insert(RequestId(3), video(3));

        let fresh = batch(Phase::Prefill);
        let mut batch = fresh.clone();
        for (id, phase, tokens) in [
            (1, Phase::Denoise, 1),
            (1, Phase::Prefill, 5),
            (1, Phase::Decode, 2),
            (2, Phase::Prefill, 2),
            (3, Phase::Prefill, 2),
        ] {
            assert!(!scheduler.push_token_rows(&mut batch, RequestId(id), phase, tokens));
        }
        assert!(!scheduler.push_media_row(&mut batch, RequestId(1), Phase::Denoise));
        assert_eq!(batch, fresh);
    }

    /// Video request `id` on slot 2 of group A: a video condition of two units with its audio
    /// and vision patches, three media units.
    fn video(id: u64) -> Request {
        let input = DiffusionRequest {
            request_id: RequestId(id),
            prompt_token_ids: vec![5, 6, 7],
            conditioning: proto::video::Conditions {
                prompt_tokens: 3,
                ..proto::video::Conditions::default()
            }
            .encode(),
            media: Vec::new(),
            conditions: vec![ConditionExtent {
                pixel_bytes: 300,
                audio_samples: 10,
                vision_patches: 4,
                vision_tokens: 2,
                latent_units: vec![6, 6],
                audio_rows: 3,
            }],
            priority: 0,
            sampling: DiffusionSamplingParams {
                num_frames: 9,
                video_units: 3,
                seed: 42,
                width: 64,
                height: 32,
                condition_rows: 15,
            },
            trajectory: Default::default(),
            encoder_inputs: None,
        };
        row(
            id,
            proto::request::Request::Diffusion(input),
            Vec::new(),
            0,
            2,
            0,
        )
    }

    /// A video request's media rows name the results their call reads, then those it writes, by
    /// their positions in its product table, in the order its admission declared them (SD §2.3,
    /// §3.3): the condition media, the vision pixels, the conditioning, both latents, the video
    /// units, one splice tail and one encoded unit per unit, and the audio samples. No row names a
    /// rank: a decode round covers a unit for each rank of the decoder group.
    #[test]
    fn a_video_requests_media_rows_name_what_they_read_then_write() {
        let mut scheduler = scheduler(&[]);
        let served = &scheduler.groups[0];
        let mut config = served.config.clone();
        let rank = config.ranks[0].clone();
        config.ranks.push(rank);
        let decoder = BlockConfig {
            distribution: Some(BlockDistribution::TemporalUnits),
            ..BlockConfig::parallel(vec![0, 1])
        };
        config.blocks.insert("video_decoder".to_owned(), decoder);
        // The group serves MiniMax-H3's pipeline, whose blocks it places.
        let mut capacity = served.capacity.clone();
        capacity.pipeline = Some(models::minimax_h3::family::pipeline());
        scheduler.groups[0] =
            crate::scheduler::WorkerGroup::new(config, capacity, served.layout.clone()).unwrap();
        scheduler.requests.insert(RequestId(1), video(1));

        // The row of the call `scheduled` counts of `phase`, with the calls `completed` counts.
        let media = |scheduler: &mut Scheduler, phase: Phase, scheduled: u32| -> MediaRow {
            let request = scheduler.requests.get_mut(&RequestId(1)).unwrap();
            request.scheduled[phase.index()] = scheduled;
            let mut batch = batch(phase);
            assert!(scheduler.push_media_row(&mut batch, RequestId(1), phase));
            batch.media.remove(0)
        };
        let named = |row: &MediaRow| (row.results.clone(), row.reads);

        assert_eq!(
            named(&media(&mut scheduler, Phase::MediaRead, 0)),
            (vec![0, 1, 2], 0)
        );
        // The text encoding computes the vision features from the patches it reads; the latent
        // preparation encodes the condition latents from the media it reads beside the
        // conditioning, each read named in the order of the products' classes.
        assert_eq!(
            named(&media(&mut scheduler, Phase::TextEncode, 0)),
            (vec![2, 3], 1)
        );
        assert_eq!(
            named(&media(&mut scheduler, Phase::LatentPrepare, 0)),
            (vec![0, 1, 3], 3)
        );

        // The last of the four denoising steps writes the final latents.
        let step = media(&mut scheduler, Phase::Denoise, 0);
        assert_eq!(
            (step.step, step.steps, named(&step)),
            (0, 4, (Vec::new(), 0))
        );
        let last = media(&mut scheduler, Phase::Denoise, 3);
        assert_eq!((last.step, named(&last)), (3, (vec![4, 5], 0)));

        // The decoder's two ranks take one unit each of a round: the first round decodes units 0
        // and 1 and writes both tails; the second decodes unit 2 after reading unit 1's tail.
        let first = media(&mut scheduler, Phase::VideoDecode, 0);
        assert_eq!(
            (named(&first), first.unit, first.rank),
            ((vec![4, 6, 7, 8], 1), 0, None)
        );
        let second = media(&mut scheduler, Phase::VideoDecode, 1);
        assert_eq!(
            (named(&second), second.unit, second.rank),
            ((vec![4, 8, 6, 9], 2), 2, None)
        );
        // An encode round writes the encoded unit of each unit of its round.
        let encode = media(&mut scheduler, Phase::VideoEncode, 0);
        assert_eq!(
            (named(&encode), encode.unit, encode.rank),
            ((vec![6, 10, 11], 1), 0, None)
        );
        let encode = media(&mut scheduler, Phase::VideoEncode, 1);
        assert_eq!(
            (named(&encode), encode.unit, encode.rank),
            ((vec![6, 12], 1), 2, None)
        );
        assert_eq!(
            named(&media(&mut scheduler, Phase::AudioDecode, 0)),
            (vec![5, 13], 1)
        );
        assert_eq!(
            named(&media(&mut scheduler, Phase::AudioEncode, 0)),
            (vec![13], 1)
        );

        // A muxing call hands over the encoded units of the encode round it follows; the last
        // hands none and reads nothing.
        scheduler.requests.get_mut(&RequestId(1)).unwrap().completed[Phase::VideoEncode.index()] =
            2;
        let handing = media(&mut scheduler, Phase::Mux, 1);
        assert_eq!((handing.rounds, named(&handing)), (1, (vec![12], 1)));
        let last = media(&mut scheduler, Phase::Mux, 2);
        assert_eq!((last.rounds, named(&last)), (0, (Vec::new(), 0)));

        // A text encoding reads its prompt as one row of token columns.
        let mut text = batch(Phase::TextEncode);
        assert!(scheduler.push_media_row(&mut text, RequestId(1), Phase::TextEncode));
        assert_eq!(text.columns.input_ids, vec![5, 6, 7]);
        assert_eq!(text.columns.rows.positions, vec![0, 1, 2]);
        assert_eq!(text.columns.rows.seq_lens, vec![3]);
        assert_eq!(text.columns.rows.logits_rows, vec![2]);

        // A denoising batch states its request's extent, by which its bucket is found.
        let mut denoise = batch(Phase::Denoise);
        assert!(scheduler.push_media_row(&mut denoise, RequestId(1), Phase::Denoise));
        scheduler.seal(&mut denoise).unwrap();
        assert_eq!(
            denoise.shape,
            Shape {
                rows: 1,
                tokens: 0,
                variant: Variant::Denoise,
                extent: Extent::Request {
                    frames: 9,
                    width: 64,
                    height: 32,
                    text_rows: 3,
                    condition_rows: 15,
                },
            }
        );
    }

    /// A token request's image encoding carries its image and writes the encoder output of its
    /// next encoder block, which the request does not hold yet, by its position in the request's
    /// encoder-output table: encoder input `i`'s at position `i` (SD §4.2).
    #[test]
    fn a_chat_images_encoding_writes_its_encoder_output() {
        let mut scheduler = scheduler(&[]);
        let mut chat = request(1, vec![1, 2, 3], 1, 0);
        input(&mut chat).multimodal_inputs.images = vec![
            image(0, ImageEncoder::Vision, 4),
            image(2, ImageEncoder::Latent, 4),
        ];
        chat.encodings = vec![(Key([1; 32]), None), (Key([2; 32]), None)];
        scheduler.requests.insert(RequestId(1), chat);

        let mut batch = batch(Phase::VisionEncode);
        assert!(!scheduler.push_media_row(&mut batch, RequestId(1), Phase::LatentEncode));
        assert!(scheduler.push_media_row(&mut batch, RequestId(1), Phase::VisionEncode));
        let row = &batch.media[0];
        assert_eq!(row.image.as_deref(), Some(&b"image"[..]));
        assert_eq!((row.slot, row.results.clone(), row.reads), (1, vec![0], 0));
        scheduler.seal(&mut batch).unwrap();
        assert_eq!(batch.shape.variant, Variant::ConditionEncode);
        assert_eq!(batch.shape.extent, Extent::None);

        scheduler
            .requests
            .get_mut(&RequestId(1))
            .unwrap()
            .num_encoded_inputs = 1;
        let mut latent = self::batch(Phase::LatentEncode);
        assert!(scheduler.push_media_row(&mut latent, RequestId(1), Phase::LatentEncode));
        assert_eq!(latent.media[0].results, vec![1]);
    }

    /// A bad word bans its last token exactly where the history ends with its prefix, the history
    /// being the generated tokens and then the drafts; a single-token word bans its token always,
    /// and the `min_tokens` floor bans the end-of-sequence tokens until the floor is met.
    #[test]
    fn a_bad_word_bans_its_last_token_where_its_prefix_ends_the_history() {
        let sampling = SamplingParams {
            bad_words_ids: vec![vec![4], vec![7, 8, 9]],
            min_tokens: 3,
            ..SamplingParams::default()
        };
        let eos = [EOS];
        // Under the floor: the end-of-sequence token and the single-token word.
        assert_eq!(banned_tokens(&sampling, &eos, &[1], 1, &[]), vec![EOS, 4]);
        // At the floor, with the three-token word's prefix ending the generated tokens.
        assert_eq!(
            banned_tokens(&sampling, &eos, &[1, 7, 8], 3, &[]),
            vec![4, 9]
        );
        // The prefix across the generated tokens and the drafts, and in the drafts alone.
        assert_eq!(banned_tokens(&sampling, &eos, &[1, 7], 3, &[8]), vec![4, 9]);
        assert_eq!(banned_tokens(&sampling, &eos, &[1], 3, &[7, 8]), vec![4, 9]);
        // A prefix the history does not end with bans nothing more; the drafts count toward the
        // floor.
        assert_eq!(banned_tokens(&sampling, &eos, &[7, 8], 2, &[5]), vec![4]);
        assert_eq!(banned_tokens(&sampling, &eos, &[1], 1, &[5]), vec![EOS, 4]);
    }
}
