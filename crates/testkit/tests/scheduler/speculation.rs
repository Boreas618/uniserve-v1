//! Speculation (SD §8.1): a deployment whose token groups verify drafts and refill their draft
//! caches (SD §3.3 `Verify`, `Refill`; §3.4 "Speculation rows"; §3.7 `update_decode`; §7.1 "A
//! refill chunk"; §7.2 "Speculation"). A verification commits the drafts its own samples confirm
//! and the token after them, and what it rejects is computed again; a call's proposals are the
//! next verification's drafts; a draft cache with a gap is refilled from its valid end, and only
//! when the policy names the refill, which it does not when the refill does not pay.
//!
//! The drafter of these cases ([`drafter`]) proposes the tokens the model samples next, of which
//! the first `right` are right and the others a token the model never samples. The model's token
//! at a logical position is the fake's draw there, so a request generates the same tokens
//! whatever it speculates: speculation changes how many calls it takes, never what it sends.

use std::ops::Range;
use std::sync::Arc;

use engine::batch::philox::DrawAddress;
use engine::batch::{Batch, Phase};
use engine::scheduler::{Message, policy};
use proto::request::{Event, RequestId};
use scheduling::limits::Parameters;
use storage::runtime::Request;
use testkit::executor::output_of;
use testkit::text::scored;

use crate::harness::{Deployment, Harness, prompt, row_span, token};

/// The drafts a verification carries in these cases' deployments.
const DRAFTS: u32 = 2;

/// A proposal no sample of the model confirms: below its sampled tokens, above every prompt's.
const WRONG: u32 = 999;

/// The token the model samples at logical `position` of the stream `key`.
fn sample(key: u64, position: u64) -> u32 {
    testkit::draw_token(DrawAddress { key, position })
}

/// Answers every batch as the model and a drafter whose proposals are the tokens the model
/// samples next, of which the first `right` are right and the rest [`WRONG`]. A verification's
/// row samples the model's token after each of its tokens, and commits its drafts while they
/// match those samples, then the sample after the last it accepts.
fn drafter(harness: &Harness, right: usize) {
    let answer = move |batch: &Batch| {
        let mut output = output_of(batch, None);
        if !matches!(batch.phase, Phase::Prefill | Phase::Decode | Phase::Verify) {
            return Ok(output);
        }
        let columns = &batch.columns;
        let rows = &columns.rows;
        for (index, call) in output.calls.iter_mut().enumerate() {
            let Some(row) = rows.calls.get(index) else {
                continue;
            };
            if call.tokens.is_empty() {
                continue;
            }
            let DrawAddress { key, position } = columns.draws[row.rows.start as usize];

            // The draw of a row lies at the position after its last token.
            let mut last = position;
            if batch.phase == Phase::Verify {
                let start = rows.query_offsets[row.rows.start as usize] as usize;
                let end = rows.query_offsets[row.rows.end as usize] as usize;
                let first = position - (end - start) as u64 + 1;
                let mut tokens = vec![sample(key, first)];
                for (at, draft) in columns.input_ids[start + 1..end].iter().enumerate() {
                    if *draft as u32 != tokens[at] {
                        break;
                    }
                    tokens.push(sample(key, first + at as u64 + 1));
                }
                last = first + tokens.len() as u64 - 1;
                call.token_logprobs = tokens.iter().map(|token| scored(*token)).collect();
                call.tokens = tokens;
            }
            call.draft_token_ids = (0..u64::from(DRAFTS))
                .map(|draft| {
                    if (draft as usize) < right {
                        sample(key, last + 1 + draft)
                    } else {
                        WRONG
                    }
                })
                .collect();
        }
        Ok(output)
    };
    let _answers = harness.fake.clone().with_output(Arc::new(answer));
}

/// Request `id`'s stream: the key every row of it draws in.
fn stream(harness: &Harness, id: RequestId) -> u64 {
    let first = harness.batches_of(id).remove(0);
    first.columns.draws[0].key
}

/// The tokens the model samples for a request of `prompt_tokens` tokens on `key`, `count` of them.
fn model(key: u64, prompt_tokens: u32, count: usize) -> Vec<u32> {
    let first = u64::from(prompt_tokens);
    (0..count as u64)
        .map(|at| sample(key, first + at))
        .collect()
}

/// Request `id`'s verifications in order: each one's first cache position, which holds its
/// request's last committed token, and the drafts after it.
fn verifications(harness: &Harness, id: RequestId) -> Vec<(i32, Vec<u32>)> {
    let verifications = harness.batches_of_phase(id, Phase::Verify);
    verifications
        .iter()
        .map(|batch| {
            let span = row_span(batch, id).unwrap();
            let drafts = batch.columns.input_ids[span.start + 1..span.end]
                .iter()
                .map(|draft| *draft as u32)
                .collect();
            (batch.columns.rows.cache_positions[span.start], drafts)
        })
        .collect()
}

/// Request `id`'s refills in order: the positions of its draft cache each one evaluates.
fn refills(harness: &Harness, id: RequestId) -> Vec<Range<u32>> {
    let refills = harness.batches_of_phase(id, Phase::Refill);
    refills
        .iter()
        .map(|batch| {
            let row = batch.requests.iter().position(|of| *of == id).unwrap();
            let start = batch.columns.drafts[row].refill.unwrap();
            let tokens = row_span(batch, id).unwrap().len() as u32;
            start..start + tokens
        })
        .collect()
}

/// Checks that every verification of `verifications` but the last is followed by one that starts
/// after the drafts it accepted, `right` of those it carried at most, and the token after them:
/// what it rejected was not computed, and the next call writes those positions again.
fn rolled_back(verifications: &[(i32, Vec<u32>)], right: usize) {
    for pair in verifications.windows(2) {
        let ((start, drafts), (next, _)) = (&pair[0], &pair[1]);
        let accepted = drafts.len().min(right) as i32;
        assert_eq!(*next, start + accepted + 1, "{verifications:?}");
    }
}

/// The deployment of the refill cases: chunks and steps of eight tokens, so that a draft cache's
/// gap longer than that is refilled in chunks.
fn refilling() -> Deployment {
    Deployment::tokens()
        .batch(8, 8)
        .chunk(8)
        .speculative(DRAFTS)
}

/// A request with a gap in its draft cache: request 1, of a prompt of 13 tokens, commits its
/// three full blocks and ends; whatever draft blocks it committed are evicted by another epoch,
/// so that request 2, whose 20-token prompt continues request 1's, pins the three target blocks
/// and no draft block. Its draft cache then lacks every position, those its prefill writes after
/// the hit included, which do not continue the draft cache's valid prefix (SD §7.2
/// "Speculation": drafts only while the draft cache covers the visible tokens).
pub(crate) fn with_a_gap(harness: &mut Harness) -> RequestId {
    harness.submit(token(1, 13, 1));
    harness.settle(100);
    let drafts: Vec<storage::Key> = harness
        .taken()
        .into_iter()
        .filter_map(|request| match request {
            Request::Commit { class: 1, key, .. } => Some(key),
            _ => None,
        })
        .collect();
    if !drafts.is_empty() {
        harness.elsewhere(&[Request::Evict { keys: drafts }]);
    }

    let mut continuing = token(2, 20, 8);
    continuing.prompt_token_ids = prompt(1, 13).into_iter().chain(prompt(2, 7)).collect();
    harness.submit(continuing)
}

/// A verification commits the drafts its own samples confirm and the sample after them, and its
/// request's next call starts after them: a request generates the model's tokens in fewer calls
/// (SD §3.7 `update_decode`; §7.2 "Speculation": committed is the accepted prefix plus one).
#[test]
fn a_verification_commits_the_drafts_its_samples_confirm_and_the_token_after_them() {
    let mut harness = Deployment::tokens().speculative(DRAFTS).build();
    drafter(&harness, 2);
    let id = harness.submit(token(1, 6, 20));
    harness.settle(200);

    let key = stream(&harness, id);
    assert_eq!(harness.tokens(id), model(key, 6, 20));
    let verifications = verifications(&harness, id);
    assert!(
        verifications.iter().any(|(_, drafts)| drafts.len() == 2),
        "{verifications:?}"
    );
    rolled_back(&verifications, 2);
    assert!(verifications.len() < 19, "drafts save calls");
}

/// A deployment whose token groups verify in place of decoding, as a speculating rank states
/// them, serves token requests: the workers serve understanding when they serve a prefill and
/// token progress, and a verification is a speculating deployment's token progress (SD §7.2, the
/// features of the workers). A request is admitted, verifies its way to its length and finishes.
#[test]
fn a_deployment_that_verifies_in_place_of_decoding_serves_token_requests() {
    let mut harness = Deployment::tokens().speculative(DRAFTS).verifying().build();
    drafter(&harness, 2);
    let id = harness.submit(token(1, 6, 20));
    harness.settle(200);

    assert!(
        matches!(harness.terminal(id), Some(Event::Finished { .. })),
        "{:?}",
        harness.events(id)
    );
    let key = stream(&harness, id);
    assert_eq!(harness.tokens(id), model(key, 6, 20));
    assert!(!verifications(&harness, id).is_empty());
}

/// The drafts a verification rejects were not computed: its request's computed tokens roll back
/// to the token after the drafts it accepted, and its next call writes the rejected positions
/// again (SD §3.7 `update_decode`: `num_computed_tokens -= rejected`, vLLM's rule).
#[test]
fn drafts_a_verification_rejects_are_computed_again_by_the_next_call() {
    let mut harness = Deployment::tokens().speculative(DRAFTS).build();
    drafter(&harness, 0);
    let id = harness.submit(token(1, 6, 12));
    harness.settle(200);

    let key = stream(&harness, id);
    assert_eq!(harness.tokens(id), model(key, 6, 12));
    let verifications = verifications(&harness, id);
    assert!(
        verifications.iter().any(|(_, drafts)| drafts.len() == 2),
        "{verifications:?}"
    );
    rolled_back(&verifications, 0);
}

/// The proposals a call returns are its request's next verification's drafts: each verification
/// that carries drafts carries the drafter's proposals after its request's last committed token
/// (SD §3.7 `update_decode`: `spec_token_ids` from the call's `draft_token_ids`; §7.2
/// "Speculation").
#[test]
fn the_proposals_of_a_call_are_the_drafts_of_the_next_verification() {
    let mut harness = Deployment::tokens().speculative(DRAFTS).build();
    drafter(&harness, 1);
    let id = harness.submit(token(1, 6, 16));
    harness.settle(200);

    let key = stream(&harness, id);
    assert_eq!(harness.tokens(id), model(key, 6, 16));
    let verifications = verifications(&harness, id);
    let carrying: Vec<&(i32, Vec<u32>)> = verifications
        .iter()
        .filter(|(_, drafts)| !drafts.is_empty())
        .collect();
    assert!(carrying.len() >= 2, "{verifications:?}");
    for (start, drafts) in carrying {
        let next = sample(key, *start as u64 + 1);
        assert_eq!(drafts, &[next, WRONG][..drafts.len()]);
    }
    rolled_back(&verifications, 1);
}

/// A request's prefill fills its draft cache with its prompt and asks the drafter for proposals
/// after the token it samples, so its first verification carries them, and no refill is needed
/// (SD §4.1 and §3.5: an admission allocates per class the blocks of its first chunk and the low
/// watermark; §4.4 "The blocks a call needs": a draft class as full attention over the draft
/// tokens; §7.2 "Speculation": proposals are kept from every call's output, `S/speculation.rs:314`).
#[test]
fn the_first_verification_after_a_prompt_carries_the_drafts_its_prefill_proposed() {
    let mut harness = Deployment::tokens().speculative(DRAFTS).build();
    drafter(&harness, 2);
    let id = harness.submit(token(1, 6, 12));
    harness.settle(200);

    let key = stream(&harness, id);
    assert!(
        refills(&harness, id).is_empty(),
        "{:?}",
        refills(&harness, id)
    );
    let verifications = verifications(&harness, id);
    assert_eq!(
        verifications.first(),
        Some(&(6, vec![sample(key, 7), sample(key, 8)]))
    );
    assert_eq!(harness.tokens(id), model(key, 6, 12));
}

/// A draft cache with a gap is refilled when the policy names the refill: from its valid end, in
/// chunks of the step's tokens that continue each other, up to the tokens its request computed;
/// no verification carries drafts before the refill reaches them, and later ones do (SD §3.3
/// `Refill`; §3.4: the policy's `refill`; §7.1 "A refill chunk"; §7.2 "Speculation").
#[test]
fn a_draft_cache_with_a_gap_is_refilled_from_its_valid_end_when_the_policy_names_it() {
    let mut harness = refilling().build();
    let id = with_a_gap(&mut harness);
    harness.settle(300);

    let prefill = harness.batches_of(id).remove(0);
    assert_eq!(prefill.phase, Phase::Prefill);
    assert_eq!(
        prefill.columns.rows.cache_positions[0], 12,
        "it runs from its hit"
    );
    assert_eq!(refills(&harness, id), [0..8, 8..16, 16..20]);

    let calls = harness.batches_of(id);
    let last_refill = calls
        .iter()
        .rposition(|batch| batch.phase == Phase::Refill)
        .unwrap();
    for (at, batch) in calls.iter().enumerate() {
        if batch.phase == Phase::Verify && at < last_refill {
            assert_eq!(
                batch.columns.input_ids.len(),
                1,
                "no drafts before the refill"
            );
        }
    }
    let after = verifications(&harness, id);
    assert!(
        after.iter().any(|(_, drafts)| drafts.len() == 2),
        "{after:?}"
    );
    assert_eq!(harness.tokens(id).len(), 8);
}

/// A policy that finds a refill does not pay never names it, and the scheduler runs none: the
/// request verifies no drafts to its end, and still generates every token (SD §3.4: refills are
/// the policy's `refill` alone; §8.1 "no refill when it does not pay"; the packaged policies
/// weigh `expected_speedup` and `refill_token_cost`, `G/arrival.rs` `refill_due`).
#[test]
fn a_refill_that_does_not_pay_is_never_run() {
    let mut harness = refilling().build();
    // A verification saves nothing over a decode: no refill is worth a token.
    let module = Arc::new(policy::load(&testkit::packaged_module()).unwrap());
    harness.tune(|tuning| {
        tuning.chunked_prefill_size = std::num::NonZeroUsize::new(8);
        tuning.max_batch = std::num::NonZeroUsize::new(8).unwrap();
        tuning.max_num_batched_tokens = std::num::NonZeroUsize::new(8);
    });
    let parameters = Parameters {
        expected_speedup: Some(1.0),
        refill_token_cost: Some(1.0),
        ..testkit::packaged_parameters()
    };
    let (reply, replied) = crossbeam_channel::bounded(1);
    harness.apply(Message::SetPolicy(module, parameters, reply));
    assert!(replied.try_recv().unwrap().is_ok());
    harness.apply(Message::CommitPolicy);

    let id = with_a_gap(&mut harness);
    harness.settle(300);
    assert!(refills(&harness, id).is_empty());
    let verifications = verifications(&harness, id);
    assert!(!verifications.is_empty());
    assert!(
        verifications.iter().all(|(_, drafts)| drafts.is_empty()),
        "{verifications:?}"
    );
    assert_eq!(harness.tokens(id).len(), 8);
}
