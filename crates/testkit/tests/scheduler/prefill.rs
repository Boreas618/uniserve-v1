//! Prefill (SD §8.1): how a prompt is computed and what its calls sample (SD §3.3, §3.4, §7.1
//! "A prefill chunk", §7.2). A long prompt is computed in chunks of at most the policy's chunk,
//! between the decodes of the requests that run beside it; only the chunk that ends the prompt
//! samples, and none when the context ends with an image block; prompt scores reach the requests
//! that ask for them at every position but the first; `min_tokens` holds the end of a request
//! back; and a banned word bans its last token wherever the history ends with its prefix.

use ipc::batch::{Batch, Phase, Variant};
use proto::request::{Event, FinishReason, RequestId};

use crate::harness::{
    Deployment, EOS, Harness, chat_image, prompt, row_positions, row_span, token,
};

/// Tokens a case scripts: below the fake's sampled window, above every prompt's ids.
const FIRST: u32 = 950;
const SECOND: u32 = 951;
const THIRD: u32 = 952;
const BANNED: u32 = 960;

/// The beginning-of-sequence token every deployment of the harness states.
const BOS: i32 = 1;

/// The logits row request `id`'s row in `batch` names: `-1` when it samples nothing.
fn logits_row(batch: &Batch, id: RequestId) -> i32 {
    let row = batch.requests.iter().position(|of| *of == id).unwrap();
    batch.columns.rows.logits_rows[row]
}

/// The tokens request `id`'s sampling rows may not sample, in the order the rows were formed:
/// the last chunk of its prompt, then its decodes.
fn banned(harness: &Harness, id: RequestId) -> Vec<Vec<u32>> {
    let batches = harness.batches_of(id).into_iter();
    let sampling = batches.filter(|batch| logits_row(batch, id) >= 0);
    sampling
        .map(|batch| {
            let row = batch.requests.iter().position(|of| *of == id).unwrap();
            batch.columns.banned.get(row).cloned().unwrap_or_default()
        })
        .collect()
}

/// Steps on a device that keeps each batch through the step after it ([`Harness::step_slow`])
/// until every request of `ids` has ended, and returns the batches each step formed.
fn slowly(harness: &mut Harness, ids: &[RequestId], steps: u64) -> Vec<Vec<Batch>> {
    let mut formed = Vec::new();
    for _ in 0..steps {
        if ids.iter().all(|id| harness.terminal(*id).is_some()) {
            break;
        }
        let before = harness.batches().len();
        harness.step_slow();
        formed.push(harness.batches().split_off(before));
    }
    harness.fake.resume();
    formed
}

/// Whether request `id` had calls formed in two consecutive steps of `steps`: on a device that
/// keeps each batch through the step after it, two of its calls were then in flight at once.
fn pipelined(steps: &[Vec<Batch>], id: RequestId) -> bool {
    let carries = |formed: &Vec<Batch>| formed.iter().any(|batch| batch.requests.contains(&id));
    steps
        .windows(2)
        .any(|pair| carries(&pair[0]) && carries(&pair[1]))
}

/// A prompt longer than the policy's chunk is computed in chunks of the chunk, in order, while
/// a request that decodes beside it is given its decodes in the steps between them (SD §3.4,
/// §7.1 "A prefill chunk"; a prefill runs while nothing of its request is in flight, §3.3).
#[test]
fn a_long_prompt_is_computed_in_chunks_between_the_decodes_of_a_running_request() {
    let mut harness = Deployment::tokens().batch(8, 32).chunk(8).build();
    harness.fake.hold();
    let decoding = harness.submit(token(1, 4, 40));
    for _ in 0..20 {
        if harness.tokens(decoding).len() >= 2 {
            break;
        }
        harness.step_slow();
    }
    let long = harness.submit(token(2, 40, 2));
    let steps = slowly(&mut harness, &[decoding, long], 300);
    harness.settle(100);

    let chunks: Vec<Vec<i32>> = harness
        .batches_of_phase(long, Phase::Prefill)
        .iter()
        .map(|batch| row_positions(batch, long))
        .collect();
    let expected: Vec<Vec<i32>> = (0..5)
        .map(|chunk| (chunk * 8..chunk * 8 + 8).collect())
        .collect();
    assert_eq!(chunks, expected);

    let prefills = |formed: &Vec<Batch>| {
        formed
            .iter()
            .any(|batch| batch.phase == Phase::Prefill && batch.requests.contains(&long))
    };
    let first = steps.iter().position(prefills).unwrap();
    let last = steps.iter().rposition(prefills).unwrap();
    let beside = steps[first..=last]
        .iter()
        .flatten()
        .filter(|batch| batch.phase == Phase::Decode && batch.requests.contains(&decoding));
    assert!(
        beside.count() >= 4,
        "the running request decodes between the chunks"
    );
    assert_eq!(harness.tokens(decoding).len(), 40);
    assert_eq!(harness.tokens(long).len(), 2);
}

/// Of a prompt's chunks only the one that ends the prompt samples: the others name no logits
/// row and are served by a variant that keeps no token, and the request's first token is the
/// last chunk's (SD §7.2 "Only the chunk that ends the prompt samples"; §3.4 `seal`).
#[test]
fn only_the_chunk_that_ends_the_prompt_samples() {
    let mut harness = Deployment::tokens().chunk(8).build();
    let id = harness.submit(token(1, 20, 3));
    harness.settle(100);

    let prefills = harness.batches_of_phase(id, Phase::Prefill);
    let chunks: Vec<Vec<i32>> = prefills
        .iter()
        .map(|batch| row_positions(batch, id))
        .collect();
    assert_eq!(
        chunks,
        [
            (0..8).collect::<Vec<i32>>(),
            (8..16).collect(),
            (16..20).collect()
        ]
    );
    let logits: Vec<i32> = prefills.iter().map(|batch| logits_row(batch, id)).collect();
    assert_eq!(logits, [-1, -1, 0]);
    let variants: Vec<Variant> = prefills.iter().map(|batch| batch.shape.variant).collect();
    assert_eq!(
        variants,
        [
            Variant::PrefillWrite,
            Variant::PrefillWrite,
            Variant::PrefillGreedy
        ]
    );

    // Three tokens: the last chunk's and two decodes'.
    assert_eq!(harness.tokens(id).len(), 3);
    assert_eq!(harness.batches_of_phase(id, Phase::Decode).len(), 2);
}

/// A context that ends with an image block samples nothing from its prompt: its prefill names no
/// logits row, and its first decode reads the beginning-of-sequence token after the block, where
/// its output starts (SD §7.2 "Only the chunk that ends the prompt samples": "not when the context
/// ends with an image block, where decoding starts from BOS").
#[test]
fn a_context_that_ends_with_an_image_samples_nothing_and_decodes_from_the_first_token() {
    let mut harness = Deployment::tokens().images().build();
    let mut request = chat_image(1, 6, 4, 3);
    request.multimodal_inputs.images[0].position = 6;
    let id = harness.submit(request);
    harness.settle(100);

    let prefills = harness.batches_of_phase(id, Phase::Prefill);
    assert_eq!(prefills.len(), 1);
    assert_eq!(
        row_positions(&prefills[0], id),
        (0..10).collect::<Vec<i32>>()
    );
    assert_eq!(logits_row(&prefills[0], id), -1);

    // Every token is a decode's; the first decode reads BOS after the block, at the rotary
    // position after the image's.
    let decodes = harness.batches_of_phase(id, Phase::Decode);
    assert_eq!(decodes.len(), 3);
    let first = &decodes[0];
    let at = row_span(first, id).unwrap().start;
    assert_eq!(first.columns.input_ids[at], BOS);
    assert_eq!(first.columns.rows.cache_positions[at], 10);
    assert_eq!(first.columns.rows.positions[at], 10);
    assert_eq!(harness.tokens(id).len(), 3);
    let Some(Event::Finished {
        reason,
        completion_tokens,
        ..
    }) = harness.terminal(id)
    else {
        panic!("{:?}", harness.events(id));
    };
    assert_eq!((reason, completion_tokens), (FinishReason::MaxTokens, 3));
}

/// A request that asks for prompt scores has every prompt position but its first scored, each
/// chunk's tokens naming the prompt token after them, and is sent them before its first token;
/// a request that does not ask names none and is sent none, in the same batches (SD §7.2
/// "Prompt logprobs"; §3.4 "Prompt scoring and selection": its batch is `PrefillScored`).
#[test]
fn prompt_scores_cover_every_position_but_the_first_and_only_reach_the_requests_that_ask() {
    let mut harness = Deployment::tokens().chunk(8).build();
    let mut asking = token(1, 20, 2);
    asking.sampling.n_prompt_logprobs = 2;
    let asking = harness.submit(asking);
    let silent = harness.submit(token(2, 20, 2));
    harness.settle(100);

    let targets = |id: RequestId| -> Vec<i32> {
        let prefills = harness.batches_of_phase(id, Phase::Prefill);
        let mut targets = Vec::new();
        for batch in &prefills {
            let span = row_span(batch, id).unwrap();
            let column = &batch.columns.targets;
            match column.get(span.clone()) {
                Some(named) => targets.extend_from_slice(named),
                None => targets.extend(std::iter::repeat_n(-1, span.len())),
            }
        }
        targets
    };
    let prompt = prompt(1, 20);
    let mut expected: Vec<i32> = prompt[1..].iter().map(|token| *token as i32).collect();
    expected.push(-1);
    assert_eq!(targets(asking), expected);
    assert!(targets(silent).iter().all(|target| *target == -1));
    for batch in harness.batches_of_phase(asking, Phase::Prefill) {
        assert_eq!(batch.shape.variant, Variant::PrefillScored);
        assert!(batch.requests.contains(&silent));
    }

    // The scores reach the asking request alone, one position per prompt token after the
    // first, each led by the token it scores, all before its first token.
    let events = harness.events(asking);
    let scored: Vec<u32> = events
        .iter()
        .filter_map(|event| match event {
            Event::PromptLogprobs { positions } => Some(positions),
            _ => None,
        })
        .flatten()
        .map(|position| position.entries[0].token_id)
        .collect();
    assert_eq!(scored, prompt[1..]);
    let first_token = events
        .iter()
        .position(|event| matches!(event, Event::TextToken { .. }))
        .unwrap();
    let last_score = events
        .iter()
        .rposition(|event| matches!(event, Event::PromptLogprobs { .. }))
        .unwrap();
    assert!(last_score < first_token);
    let silent_events = harness.events(silent).into_iter();
    assert!(
        !silent_events
            .into_iter()
            .any(|event| matches!(event, Event::PromptLogprobs { .. }))
    );
}

/// Under `min_tokens`, the rows that sample a request's first `min_tokens` tokens ban its
/// end-of-sequence tokens, and an end-of-sequence token sampled under the floor is a token like
/// any other; the first above it ends the request and is not sent (SD §7.2 "Tokens banned per
/// step", "Stop conditions": `generated <= min_tokens` is under the floor).
#[test]
fn min_tokens_bans_the_end_of_sequence_until_the_floor_and_an_end_under_it_continues() {
    let mut harness = Deployment::tokens().build();
    let mut request = token(1, 6, 10);
    request.sampling.min_tokens = 3;
    let id = harness.submit(request);
    harness.sample(id, &[EOS, FIRST, EOS, EOS]);
    harness.settle(100);

    let bans: Vec<bool> = banned(&harness, id)
        .iter()
        .map(|tokens| tokens.contains(&EOS))
        .collect();
    assert_eq!(bans, [true, true, true, false]);
    assert_eq!(harness.tokens(id), [EOS, FIRST, EOS]);
    let Some(Event::Finished {
        reason,
        completion_tokens,
        ..
    }) = harness.terminal(id)
    else {
        panic!("{:?}", harness.events(id));
    };
    assert_eq!((reason, completion_tokens), (FinishReason::Eos, 4));
}

/// A banned word bans its last token at every row whose history, the request's generated
/// tokens, ends with the word's prefix: a word of one token at every row (SD §7.2 "Tokens banned
/// per step": the last tokens of banned words; vLLM's `apply_bad_words`).
#[test]
fn a_banned_word_bans_its_last_token_wherever_the_history_ends_with_its_prefix() {
    let mut harness = Deployment::tokens().build();
    let mut request = token(1, 6, 5);
    request.sampling.bad_words_ids = vec![vec![FIRST, SECOND], vec![BANNED]];
    let id = harness.submit(request);
    harness.sample(id, &[FIRST, THIRD, FIRST, FIRST, THIRD]);
    harness.settle(100);

    assert_eq!(harness.tokens(id), [FIRST, THIRD, FIRST, FIRST, THIRD]);
    assert_eq!(
        banned(&harness, id),
        [
            vec![BANNED],
            vec![SECOND, BANNED],
            vec![BANNED],
            vec![SECOND, BANNED],
            vec![SECOND, BANNED],
        ]
    );
}

/// A request that bans a word of several tokens decodes with nothing of it in flight, since its
/// next ban depends on the token its last call samples; a request beside it that bans none
/// pipelines its decodes (SD §3.3 `Decode`: "nothing in flight at all when the request bans a
/// multi-token word"; §7.2 "Pipelined decodes").
#[test]
fn a_request_banning_a_word_of_several_tokens_has_one_call_in_flight() {
    let mut harness = Deployment::tokens().build();
    harness.fake.hold();
    let mut banning = token(1, 6, 12);
    banning.sampling.bad_words_ids = vec![vec![FIRST, SECOND]];
    let banning = harness.submit(banning);
    let plain = harness.submit(token(2, 6, 12));
    let steps = slowly(&mut harness, &[banning, plain], 300);
    harness.settle(100);

    assert!(pipelined(&steps, plain));
    assert!(!pipelined(&steps, banning));
    assert_eq!(harness.tokens(banning).len(), 12);
    assert_eq!(harness.tokens(plain).len(), 12);
}
