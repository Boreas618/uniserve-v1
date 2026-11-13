//! The batch's columns (SD §8.1): what each token phase's row writes into the host columns the
//! runner copies into its input tensors (SD §2.3, §3.4 "What the three row functions write", §7.2
//! "Sampling address"), restated on the batches `schedule` hands to the executor.
//!
//! A row's commit cells follow the commit table's meaning (`proto::module::commit`): its slot; its
//! mode (a prefill that samples states its lengths explicitly, a decode advances them by one, a
//! verification by the span it accepts, any other row leaves them as they are); the task it
//! sampled, numbered over the batch's sampling rows; the cache length a prefill or refill states;
//! the logical and sampling positions an explicit or verification row installs; a
//! verification's visible base; whether the row's tokens count for its penalties. A request's
//! sampling position counts its draws: one for the prompt's token and one per decode, none for a
//! verification (SD §7.2 "Sampling address").

use engine::batch::cache::CacheGroupKind;
use engine::batch::philox::request_sampling_key;
use engine::batch::{
    Batch, DraftWork, NewSlot, Phase, Row, StateCopy, StateDirection, Variant, VisionRows,
};
use proto::module::commit::{COMMIT_FIELDS, column, mode};
use proto::request::{GenerationRequest, SamplingParams};
use storage::runtime::{Completion, Request};

use crate::harness::{BLOCK, Deployment, Harness, chat_image, prompt, token};
use crate::speculation::with_a_gap;

/// The beginning-of-sequence token every deployment of the harness states.
const BOS: u32 = 1;

/// The draws a request has made after its prompt's token: its sampling position after it.
const PROMPT_DRAW: i64 = 1;

/// Cell `field` of call `call` of `batch`'s commit table, which is column-major over its calls.
fn cell(batch: &Batch, field: usize, call: usize) -> i64 {
    let calls = batch.requests.len();
    assert_eq!(batch.columns.commit.len(), COMMIT_FIELDS * calls);
    batch.columns.commit[field * calls + call]
}

/// The commit cells of call `call` of `batch`, in the table's field order.
fn cells(batch: &Batch, call: usize) -> [i64; COMMIT_FIELDS] {
    std::array::from_fn(|field| cell(batch, field, call))
}

/// Commit cells: slot, mode, task, cache length, logical, sampling, visible, penalties.
#[expect(
    clippy::too_many_arguments,
    reason = "one argument per field of the table"
)]
fn commit(
    slot: u32,
    mode: i64,
    task: i64,
    cache_length: i64,
    logical: i64,
    sampling: i64,
    visible: i64,
    penalties: bool,
) -> [i64; COMMIT_FIELDS] {
    let mut cells = [0; COMMIT_FIELDS];
    cells[column::SLOT] = i64::from(slot);
    cells[column::MODE] = mode;
    cells[column::TASK] = task;
    cells[column::CACHE_LENGTH] = cache_length;
    cells[column::LOGICAL] = logical;
    cells[column::SAMPLING] = sampling;
    cells[column::VISIBLE] = visible;
    cells[column::PENALTIES] = i64::from(penalties);
    cells
}

/// The slot rows the buffer pool gave the session's owners, in the order it gave them.
fn attached(harness: &Harness) -> Vec<u32> {
    let answers = harness.answers().into_iter();
    answers
        .filter_map(|answer| match answer {
            (Request::Attach { .. }, Completion::Attached { slot }) => Some(slot),
            _ => None,
        })
        .collect()
}

/// A token request whose sampler is shaped by every parameter it states.
fn tuned(id: u64, prompt_tokens: u32, max_tokens: usize) -> GenerationRequest {
    let mut request = token(id, prompt_tokens, max_tokens);
    request.sampling = SamplingParams {
        temperature: 0.5,
        top_k: 7,
        top_p: 0.9,
        min_p: 0.05,
        repetition_penalty: 1.25,
        frequency_penalty: 0.5,
        presence_penalty: 0.75,
        seed: Some(77),
        ..SamplingParams::default()
    };
    request
}

/// The sampler parameters of a row of [`tuned`], in the sampler's order: temperature, top-k,
/// top-p, min-p, the repetition, frequency and presence penalties, and a column no operator
/// reads (SD §3.4 `push_token_rows`).
const TUNED: [f32; 8] = [0.5, 7.0, 0.9, 0.05, 1.25, 0.5, 0.75, 0.0];

/// The sampler parameters of a greedy row with no penalty.
const GREEDY: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0];

/// A prefill that ends two prompts in one batch: one row per request, each of its prompt's
/// tokens at their positions, each sampling its last token at the position after its prompt
/// in its own stream, and each starting its request's sampling state on its slot (SD §2.3,
/// §3.4 `push_token_rows`, `seal`; §7.2 "Sampling address").
#[test]
fn a_prefill_row_states_its_prompt_positions_lengths_sampling_and_commit() {
    let mut harness = Deployment::tokens().build();
    let plain = harness.submit(token(1, 6, 2));
    let tuned = harness.submit(tuned(2, 10, 2));
    harness.run_until(10, |harness| !harness.batches().is_empty());
    let batch = harness.batches().remove(0);
    assert_eq!(batch.phase, Phase::Prefill);
    assert_eq!(batch.requests, [plain, tuned]);

    let columns = &batch.columns;
    let rows = &columns.rows;
    let slots: Vec<u32> = rows.calls.iter().map(|call| call.slot).collect();
    let mut given = attached(&harness);
    given.sort_unstable();
    let mut named = slots.clone();
    named.sort_unstable();
    assert_eq!(named, given, "each row names its request's slot");
    assert!(slots.iter().all(|slot| *slot >= 1));

    let row = |slot, rows| Row {
        slot,
        kind: Phase::Prefill,
        cache_group: 0,
        rows,
    };
    assert_eq!(rows.calls, [row(slots[0], 0..1), row(slots[1], 1..2)]);
    let ids: Vec<i32> = prompt(1, 6)
        .into_iter()
        .chain(prompt(2, 10))
        .map(|token| token as i32)
        .collect();
    assert_eq!(columns.input_ids, ids);
    let positions: Vec<i32> = (0..6).chain(0..10).collect();
    assert_eq!(rows.positions, positions);
    assert_eq!(rows.cache_positions, positions);
    assert_eq!(rows.seq_lens, [6, 10]);
    assert_eq!(rows.query_offsets, [0, 6, 16]);
    assert_eq!(rows.logits_rows, [0, 1]);
    let row_slots: Vec<i32> = slots.iter().map(|slot| *slot as i32).collect();
    assert_eq!(columns.slots, row_slots);
    assert_eq!(columns.parameters, [GREEDY, TUNED].concat());

    // Each row draws at the position after its prompt: an unseeded request in a stream of its
    // own, a seeded one in its seed's whatever its identity.
    assert_eq!(columns.draws[0].position, 6);
    assert_eq!(columns.draws[1].position, 10);
    assert_eq!(
        columns.draws[1].key,
        request_sampling_key(Some(77), 0, 0, 0)
    );
    assert_ne!(columns.draws[0].key, columns.draws[1].key);

    // A prefill that samples states its cache length, its logical position after the prompt and
    // its sampling position after the prompt's draw.
    assert_eq!(
        cells(&batch, 0),
        commit(slots[0], mode::EXPLICIT, 0, 6, 6, PROMPT_DRAW, 0, false)
    );
    assert_eq!(
        cells(&batch, 1),
        commit(slots[1], mode::EXPLICIT, 1, 10, 10, PROMPT_DRAW, 0, true)
    );

    // Both rows start their requests on their slots, with nothing generated before them.
    let mut new_slots: Vec<NewSlot> = slots
        .iter()
        .map(|slot| NewSlot {
            slot: *slot,
            history: Vec::new(),
            bias: Vec::new(),
            allowed: None,
            context: [BOS, BOS],
        })
        .collect();
    new_slots.sort_unstable_by_key(|new| new.slot);
    assert_eq!(batch.new_slots, new_slots);

    // A drawing row makes the batch a drawing one; the columns no row needs stay empty.
    assert_eq!(batch.shape.rows, 2);
    assert_eq!(batch.shape.tokens, 16);
    assert_eq!(batch.shape.variant, Variant::PrefillOutput);
    assert!(columns.banned.is_empty() && columns.biased.is_empty());
    assert!(columns.targets.is_empty() && columns.vision.is_empty());
    assert!(columns.checkpoints.is_empty() && columns.state_copies.is_empty());
    assert!(columns.drafts.is_empty());
    harness.settle(100);
}

/// A decode is one row of one token per request: the token its request sampled last, at the
/// position after its context, sampling the next position in its stream, advancing its slot's
/// lengths by one; its request's slot has seen it, so it starts nothing (SD §3.4
/// `push_token_rows`).
#[test]
fn a_decode_row_reads_its_last_token_and_advances_its_slot_by_one() {
    let mut harness = Deployment::tokens().build();
    let plain = harness.submit(token(1, 6, 3));
    let tuned = harness.submit(tuned(2, 10, 3));
    harness.settle(100);

    let prefill = harness.batches_of_phase(plain, Phase::Prefill).remove(0);
    let batch = harness.batches_of_phase(plain, Phase::Decode).remove(0);
    assert_eq!(batch.requests, [plain, tuned]);
    let columns = &batch.columns;
    let rows = &columns.rows;
    let slots: Vec<u32> = prefill
        .columns
        .rows
        .calls
        .iter()
        .map(|call| call.slot)
        .collect();
    let row = |slot, rows| Row {
        slot,
        kind: Phase::Decode,
        cache_group: 0,
        rows,
    };
    assert_eq!(rows.calls, [row(slots[0], 0..1), row(slots[1], 1..2)]);

    let first = [harness.tokens(plain)[0], harness.tokens(tuned)[0]];
    assert_eq!(columns.input_ids, first.map(|token| token as i32));
    assert_eq!(rows.positions, [6, 10]);
    assert_eq!(rows.cache_positions, [6, 10]);
    assert_eq!(rows.seq_lens, [7, 11]);
    assert_eq!(rows.query_offsets, [0, 1, 2]);
    assert_eq!(rows.logits_rows, [0, 1]);
    assert_eq!(columns.parameters, [GREEDY, TUNED].concat());

    // The same streams as the prompts', one position on.
    assert_eq!(columns.draws[0].key, prefill.columns.draws[0].key);
    assert_eq!(columns.draws[1].key, prefill.columns.draws[1].key);
    assert_eq!(columns.draws[0].position, 7);
    assert_eq!(columns.draws[1].position, 11);
    assert_eq!(
        cells(&batch, 0),
        commit(slots[0], mode::DECODE, 0, 0, 0, 0, 0, false)
    );
    assert_eq!(
        cells(&batch, 1),
        commit(slots[1], mode::DECODE, 1, 0, 0, 0, 0, true)
    );
    assert!(batch.new_slots.is_empty());
    assert_eq!(batch.shape.variant, Variant::DecodeSampled);
}

/// A decode whose request has a call on the device reads its token as -1: the runner takes the
/// token its slot sampled last. A decode with nothing of its request in flight reads the token
/// its request was sent last. Either way the decodes write the positions after the context in
/// order (SD §3.4 "Queue depth and pipelining", `push_token_rows`).
#[test]
fn a_decode_behind_a_call_in_flight_reads_the_token_its_slot_sampled_last() {
    let mut harness = Deployment::tokens().build();
    let id = harness.submit(token(1, 6, 12));
    harness.run_until(20, |harness| harness.tokens(id).len() >= 3);

    // From here, each batch stays on the device through the step after it.
    harness.fake.hold();
    let mut decodes: Vec<(Batch, Option<u32>)> = Vec::new();
    let mut in_flight = false;
    while harness.terminal(id).is_none() && decodes.len() < 40 {
        let before = harness.batches().len();
        harness.step_slow();
        let formed = harness.batches().split_off(before);
        // The outputs the step applied before it scheduled are the request's tokens now.
        let last = harness.tokens(id).last().copied();
        let carried = formed
            .into_iter()
            .find(|batch| batch.requests.contains(&id));
        if let Some(batch) = &carried {
            decodes.push((batch.clone(), (!in_flight).then_some(last).flatten()));
        }
        in_flight = carried.is_some();
    }
    harness.fake.resume();
    harness.settle(100);

    let mut pipelined = 0;
    for (batch, read) in &decodes {
        assert_eq!(batch.phase, Phase::Decode);
        let expected = read.map_or(-1, |token| token as i32);
        assert_eq!(batch.columns.input_ids, [expected]);
        pipelined += usize::from(read.is_none());
    }
    assert!(pipelined > 0, "a decode was formed behind another");
    let written: Vec<i32> = harness
        .batches_of(id)
        .iter()
        .filter(|batch| batch.phase == Phase::Decode)
        .map(|batch| batch.columns.rows.cache_positions[0])
        .collect();
    assert_eq!(written, (6..17).collect::<Vec<i32>>());
    assert_eq!(harness.tokens(id).len(), 12);
}

/// A verification is one row of its request's committed token and the drafts it carries, at
/// consecutive positions, which asks the drafter for the deployment's proposals and states the
/// base its accepted span advances from (SD §3.4 "Speculation rows", `push_token_rows`; §7.2
/// "Sampling address": a verification draws nothing).
#[test]
fn a_verification_row_carries_its_token_and_drafts_and_asks_for_proposals() {
    let mut harness = Deployment::tokens().speculative(2).build();
    let id = harness.submit(token(1, 6, 12));
    harness.settle(200);

    let verifications = harness.batches_of_phase(id, Phase::Verify);
    let batch = verifications
        .iter()
        .find(|batch| batch.columns.input_ids.len() == 3)
        .expect("a verification carries two drafts");
    let columns = &batch.columns;
    let rows = &columns.rows;
    let start = rows.cache_positions[0];
    let slot = rows.calls[0].slot;
    assert_eq!(
        rows.calls,
        [Row {
            slot,
            kind: Phase::Verify,
            cache_group: 0,
            rows: 0..1
        }]
    );

    // Its first token is the request's token at `start`; its drafts are the drafter's
    // proposals, which the harness's drafter numbers from 900.
    let tokens: Vec<u32> = prompt(1, 6).into_iter().chain(harness.tokens(id)).collect();
    assert_eq!(columns.input_ids, [tokens[start as usize] as i32, 900, 901]);
    assert_eq!(rows.positions, [start, start + 1, start + 2]);
    assert_eq!(rows.cache_positions, [start, start + 1, start + 2]);
    assert_eq!(rows.seq_lens, [start + 3]);
    assert_eq!(rows.query_offsets, [0, 3]);
    assert_eq!(rows.logits_rows, [0]);
    assert_eq!(columns.draws[0].position, (start + 3) as u64);
    assert_eq!(
        columns.drafts,
        [DraftWork {
            proposals: 2,
            lookahead: None,
            refill: None
        }]
    );
    let start = i64::from(start);
    assert_eq!(
        cells(batch, 0),
        commit(slot, mode::VERIFY, 0, 0, start, PROMPT_DRAW, start, false)
    );
    assert_eq!(batch.shape.variant, Variant::VerifyGreedy);
    assert_eq!(batch.shape.tokens, 3);
}

/// A refill is one row over tokens its request's cache already holds, from its draft cache's
/// valid end: it writes no cache entry and samples nothing, names its first position and the
/// token after its last for the drafter, and leaves its slot's cache length where the target's
/// is (SD §3.4 `push_token_rows`; `DraftWork::refill`).
#[test]
fn a_refill_row_reads_what_its_draft_cache_lacks_and_writes_no_cache() {
    // A request whose hit pins no draft block lacks its hit in its draft cache, which a refill
    // reads; a prefill fills the draft cache of the tokens it computes.
    let mut harness = Deployment::tokens().speculative(2).build();
    let id = with_a_gap(&mut harness);
    harness.settle(200);

    let refills = harness.batches_of_phase(id, Phase::Refill);
    let Some(batch) = refills.first() else {
        panic!("no refill ran: {:?}", harness.batches_of(id));
    };
    let columns = &batch.columns;
    let rows = &columns.rows;
    let DraftWork {
        proposals,
        lookahead,
        refill: Some(start),
    } = columns.drafts[0]
    else {
        panic!("{:?}", columns.drafts);
    };
    let tokens = columns.input_ids.len() as u32;
    let end = start + tokens;
    let context: Vec<u32> = prompt(1, 13)
        .into_iter()
        .chain(prompt(2, 7))
        .chain(harness.tokens(id))
        .collect();
    let read: Vec<i32> = context[start as usize..end as usize]
        .iter()
        .map(|token| *token as i32)
        .collect();
    assert_eq!(columns.input_ids, read);
    assert_eq!(proposals, 0);
    assert_eq!(lookahead, Some(context[end as usize]));
    assert_eq!(rows.calls[0].kind, Phase::Refill);
    assert!(rows.cache_positions.iter().all(|position| *position == -1));
    assert_eq!(rows.logits_rows, [-1]);
    assert_eq!(cell(batch, column::MODE, 0), mode::CACHE);
    assert_eq!(cell(batch, column::TASK, 0), -1);
    assert_eq!(cell(batch, column::CACHE_LENGTH, 0), i64::from(end));
    assert_eq!(batch.shape.variant, Variant::PrefillWrite);
}

/// An image block in a prefill row is named in the batch's vision rows: the token of the batch
/// its rows start at, its rows, the rotary position they all share, and its encoder output's
/// position in the request's encoder-output table; the text after it continues the rotary
/// positions after the image's (SD §2.3 `VisionRows`; §3.4 `push_token_rows`).
#[test]
fn an_image_block_is_named_in_the_vision_rows_of_its_prefill() {
    let mut harness = Deployment::tokens().images().build();
    let chat = harness.submit(chat_image(1, 8, 6, 2));
    harness.settle(100);

    let batch = harness.batches_of_phase(chat, Phase::Prefill).remove(0);
    assert_eq!(batch.requests, [chat]);
    let columns = &batch.columns;
    // Its image lies at prompt position 4 and takes 6 rows and 6 positions.
    assert_eq!(
        columns.vision,
        [VisionRows {
            token: 4,
            rows: 6,
            start: 4,
            feature: 0
        }]
    );
    let positions: Vec<i32> = (0..4).chain([4; 6]).chain(10..14).collect();
    assert_eq!(columns.rows.positions, positions);
    assert_eq!(columns.rows.cache_positions, (0..14).collect::<Vec<i32>>());
    assert_eq!(columns.rows.seq_lens, [14]);
    assert_eq!(columns.draws[0].position, 14);
}

/// Around an image block a row embeds its request's tokens: the prompt text before the block,
/// no token for the block's rows, the prompt text after it, and after the prompt the tokens the
/// request generated, which its decodes read (SD §3.4 `push_token_rows`: `input_ids` are the
/// prompt slice, a decode's last token, 0 for an image's rows).
#[test]
fn the_tokens_around_an_image_block_embed_the_prompt_and_the_output() {
    let mut harness = Deployment::tokens().images().build();
    let chat = harness.submit(chat_image(1, 8, 6, 3));
    harness.settle(100);

    let prefill = harness.batches_of_phase(chat, Phase::Prefill).remove(0);
    let text = prompt(1, 8);
    let ids: Vec<i32> = text[..4]
        .iter()
        .map(|token| *token as i32)
        .chain([0; 6])
        .chain(text[4..].iter().map(|token| *token as i32))
        .collect();
    assert_eq!(prefill.columns.input_ids, ids);

    let read: Vec<i32> = harness
        .batches_of_phase(chat, Phase::Decode)
        .iter()
        .map(|batch| batch.columns.input_ids[0])
        .collect();
    let sent: Vec<i32> = harness.tokens(chat)[..2]
        .iter()
        .map(|token| *token as i32)
        .collect();
    assert_eq!(read, sent);
}

/// A model with a recurrent state keeps it at the last block boundary a prompt chunk reaches, in
/// the state table's position of the block that ends there; a decode keeps none off the state
/// interval; and the first row on a slot that continues a cached prefix restores the checkpoint
/// its prefix ends at, before the forward (SD §3.4 "Recurrent-state checkpoints"; §7.4 "Group
/// rules and kept checkpoint boundaries").
#[test]
fn a_recurrent_state_is_kept_at_block_boundaries_and_restored_where_a_prefix_ends() {
    let mut harness = Deployment::tokens()
        .cache(&[
            (CacheGroupKind::Full, BLOCK),
            (CacheGroupKind::State, BLOCK),
        ])
        .build();
    let first = harness.submit(token(1, 10, 3));
    harness.settle(100);

    let prefill = harness.batches_of_phase(first, Phase::Prefill).remove(0);
    let slot = prefill.columns.rows.calls[0].slot;
    assert_eq!(prefill.columns.checkpoints, [8]);
    assert_eq!(
        prefill.columns.state_copies,
        [StateCopy {
            slot,
            position: 1,
            direction: StateDirection::Keep
        }]
    );
    for decode in harness.batches_of_phase(first, Phase::Decode) {
        assert!(decode.columns.checkpoints.iter().all(|kept| *kept == 0));
        assert!(decode.columns.state_copies.is_empty());
    }

    // A request of the same prompt continues from its hit at 8, which the checkpoint ending at
    // 8 restores.
    let mut again = token(2, 10, 2);
    again.prompt_token_ids = prompt(1, 10);
    let again = harness.submit(again);
    harness.settle(100);
    let prefill = harness.batches_of_phase(again, Phase::Prefill).remove(0);
    let slot = prefill.columns.rows.calls[0].slot;
    assert_eq!(prefill.columns.rows.cache_positions, [8, 9]);
    assert_eq!(
        prefill.columns.state_copies,
        [StateCopy {
            slot,
            position: 1,
            direction: StateDirection::Restore
        }]
    );
    assert!(prefill.columns.checkpoints.iter().all(|kept| *kept == 0));
}

/// A request's first row on a group starts its sampling state there: the tokens it generated
/// before, which its penalties count; its logit bias, each token once with its last bias, in
/// token order; the tokens it allows, sorted once each; the two tokens before the row. Its rows
/// add the bias row; later rows on the group start nothing (SD §2.3 `NewSlot`).
#[test]
fn a_new_slot_starts_its_requests_history_bias_and_context() {
    let mut harness = Deployment::tokens().build();
    let mut request = token(1, 6, 8);
    request.generated_token_ids = vec![1500, 1501];
    request.sampling.frequency_penalty = 0.5;
    request.sampling.logit_bias = vec![(1700, 0.5), (1300, 1.0), (1700, -0.25)];
    request.sampling.allowed_token_ids = Some(vec![1900, 1300, 1900, 1500]);
    let id = harness.submit(request);
    harness.settle(100);

    let batches = harness.batches_of(id);
    let slot = batches[0].columns.rows.calls[0].slot;
    assert_eq!(
        batches[0].new_slots,
        [NewSlot {
            slot,
            history: vec![1500, 1501],
            bias: vec![(1300, 1.0), (1700, -0.25)],
            allowed: Some(vec![1300, 1500, 1900]),
            context: [BOS, BOS],
        }]
    );
    for batch in &batches {
        assert_eq!(batch.columns.biased, [true]);
    }
    assert!(batches[1..].iter().all(|batch| batch.new_slots.is_empty()));
}
