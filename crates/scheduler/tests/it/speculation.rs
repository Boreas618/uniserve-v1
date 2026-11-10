//! Draft cache refills in an epoch that speculates, weighed by what the
//! epoch states: a request that reuses a prefix whose draft cache pages end
//! before its context's verifies no drafts until its draft cache is refilled,
//! and the scheduler refills the missing entries only when the decode time
//! speculation would save over the tokens the request may still generate
//! exceeds the time of evaluating them again.
//!
//! Each epoch first runs a request that speculates and a short one. Then
//! requests reuse the first one's prompt, whose draft cache page covers fewer
//! tokens than its context pages: each lacks the draft cache entries of the
//! positions between, and verifies drafts only once they are refilled.
//!
//! The worker is `testkit::FakeBatches`, answering as a drafter that proposes
//! the model's next `accepted` tokens and then wrong ones: a verification row
//! samples the fake's token at its position, and the call accepts the drafts
//! its rows' samples confirm.

use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use ipc::execution::{Batch, BatchOutput, Outcome, RowKind};
use ipc::philox::DrawAddress;
use ipc::{RequestId, SamplingParams};
use scheduler::GenerationRequest;
use scheduler::{CommandSender, Event, SchedulerConfig};
use testkit::{FakeBatches, Tokens, draw_token};

use crate::support;

/// Tokens of a context page; a draft cache page holds `SPAN` of them.
const BLOCK: u32 = 16;
const SPAN: u32 = 16;
/// The prompt the later requests reuse: past one draft cache page and
/// within the next.
const SHARED: usize = 300;
/// The drafts the epoch verifies per step.
const DRAFTS: usize = 3;

/// A greedy text request of `prompt` that generates at most `max_tokens`.
fn request(id: u64, prompt: Vec<u32>, max_tokens: usize) -> GenerationRequest {
    GenerationRequest {
        request_id: RequestId(id),
        prompt_token_ids: prompt,
        multimodal_inputs: Default::default(),
        sampling: SamplingParams::default(),
        max_und_tokens: max_tokens,
        include_stop_token: false,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: Default::default(),
        generated_token_ids: Vec::new(),
        adapter: 0,
    }
}

/// The shared prompt followed by `suffix`'s twenty tokens.
fn reusing(suffix: u32) -> Vec<u32> {
    (0..SHARED as u32)
        .map(|index| 2_000 + index)
        .chain((0..20).map(|index| suffix + index))
        .collect()
}

/// The fake's answer to `batch`: a call samples the fake's token after its
/// last row; a verification's row reads its current token and its drafts and
/// commits the drafts the fake's samples confirm and the token after them;
/// and every call that samples proposes `DRAFTS` drafts, the first `accepted`
/// of which are the tokens the fake samples next.
fn drafting(batch: &Batch, accepted: usize) -> Outcome<BatchOutput> {
    let mut output = testkit::completed(batch, Tokens::Draw, testkit::Extent::Stated);
    let columns = &batch.columns;
    let rows = &columns.rows;
    for (call, record) in rows.calls.iter().zip(&mut output.calls) {
        if record.tokens.is_empty() {
            continue;
        }
        let row = call.rows.end as usize - 1;
        // The row's draw lies at the position after its query: a query of `q`
        // tokens covers `[end - q, end)`, and the sample after its `j`-th token
        // lies at `end - q + 1 + j`.
        let DrawAddress { key, position: end } = columns.draws[row];
        let sample = |position: u64| draw_token(DrawAddress { key, position });
        let mut last = end;
        if call.kind == RowKind::Verify {
            let offset = rows.query_offsets[row] as usize;
            let query = (rows.query_offsets[row + 1] - rows.query_offsets[row]) as u64;
            let drafts = &columns.input_ids[offset + 1..offset + query as usize];
            let first = end - query + 1;
            let mut tokens = vec![sample(first)];
            for (index, &draft) in drafts.iter().enumerate() {
                if draft as u32 != tokens[index] {
                    break;
                }
                tokens.push(sample(first + index as u64 + 1));
            }
            last = first + tokens.len() as u64 - 1;
            record.token_logprobs = vec![testkit::text::LOGPROB; tokens.len()];
            record.tokens = tokens;
        }
        // The drafter proposes after the committed tokens; a wrong draft is a
        // token the fake never samples.
        record.draft_token_ids = (0..DRAFTS as u64)
            .map(|draft| {
                if (draft as usize) < accepted {
                    sample(last + 1 + draft)
                } else {
                    testkit::SAMPLED - 1
                }
            })
            .collect();
    }
    Outcome::Success(output)
}

/// A scheduler that speculates three drafts per request over a fake drafter
/// whose drafts are right `accepted` times a step, and the batches it
/// submits.
struct Epoch {
    handle: CommandSender,
    thread: thread::JoinHandle<()>,
    fake: Arc<FakeBatches>,
}

impl Epoch {
    /// `accepted` drafts a step are right; a verification costs `verify`
    /// decode steps, which the epoch states as its expected speedup.
    fn start(accepted: usize, verify: f64) -> Self {
        let mut rank = support::text_rank();
        let cache = rank
            .prefix_cache
            .as_mut()
            .expect("a text rank has a prefix cache");
        cache.groups[0].block_size = BLOCK;
        let first = cache.groups[0].clone();
        let layer = first
            .layer_ids
            .iter()
            .copied()
            .max()
            .map_or(0, |layer| layer + 1);
        cache.groups.push(ipc::CacheGroup {
            kind: ipc::CacheGroupKind::Draft,
            block_size: BLOCK * SPAN,
            units_per_page: 1,
            layer_ids: vec![layer],
            ..first
        });
        let fake = Arc::new(
            support::text_pool(rank, 1, None)
                .answering(Arc::new(move |batch| drafting(batch, accepted))),
        );
        let scheduler = support::fake::scheduler_over(
            Arc::<FakeBatches>::clone(&fake),
            SchedulerConfig {
                draft_tokens: DRAFTS,
                ..Default::default()
            },
            // What this epoch's own steps cost, as the policy weighs them:
            // the tokens a verification commits over the decode steps it
            // takes, and a refilled token's cost in decode steps. A policy
            // reads stated parameters, so the epoch states them.
            //
            // Leaving them unstated is not the same epoch: `refill_due` has no
            // value to weigh then and refills whatever the steps cost, so the
            // two cases below measure the stated path only.
            scheduler::Parameters {
                expected_speedup: Some((accepted + 1) as f64 / verify),
                refill_token_cost: Some(0.1),
                ..Default::default()
            },
        )
        .expect("the scheduler is built over the fake");
        let (tx, rx) = crossbeam_channel::unbounded();
        let thread = thread::spawn(move || {
            let _ = scheduler.run(rx);
        });
        Self {
            handle: CommandSender::new(tx),
            thread,
            fake,
        }
    }

    /// Runs `request` to its end, alone; returns how many tokens it read and
    /// the draft counts of its verifications.
    fn run(&mut self, request: GenerationRequest) -> (usize, Vec<usize>) {
        let before = self.fake.submitted().len();
        let mut events = self.handle.submit(request).unwrap();
        let mut tokens = 0;
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            assert!(Instant::now() < deadline, "the request did not end");
            match events.try_recv() {
                Ok(Event::TextToken { .. }) => tokens += 1,
                Ok(Event::TextTokens { ids }) => tokens += ids.len(),
                Ok(event) if event.is_terminal() => break,
                Ok(_) => {}
                Err(_) => thread::sleep(Duration::from_millis(1)),
            }
        }
        // The request ran alone, so every verification since it was
        // submitted is its own.
        // A verification's row reads its current token and its drafts.
        let verifications = self.fake.submitted()[before..]
            .iter()
            .flat_map(|batch| {
                let rows = &batch.columns.rows;
                rows.calls
                    .iter()
                    .filter(|call| call.kind == RowKind::Verify)
                    .map(|call| {
                        let row = call.rows.end as usize - 1;
                        (rows.query_offsets[row + 1] - rows.query_offsets[row]) as usize - 1
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        (tokens, verifications)
    }

    /// Runs a request that speculates over the shared prompt, and a request
    /// of two tokens.
    fn measure(&mut self) {
        let (_, verifications) = self.run(request(1, reusing(4_000), 40));
        assert!(
            verifications.iter().any(|&drafts| drafts > 0),
            "the first request did not speculate"
        );
        self.run(request(2, vec![5_000, 5_001, 5_002], 2));
    }

    fn stop(self) {
        self.handle.shutdown();
        self.thread.join().unwrap();
    }
}

/// An epoch whose verification step costs more decode steps than the
/// tokens it commits (two tokens a step, at three decode steps' time)
/// refills nothing: a request of two hundred tokens left verifies no
/// drafts to its end.
#[test]
fn a_epoch_whose_speculation_does_not_pay_refills_nothing() {
    let mut epoch = Epoch::start(1, 3.0);
    epoch.measure();

    let (read, verifications) = epoch.run(request(3, reusing(6_000), 200));
    assert_eq!(read, 200);
    assert!(verifications.iter().all(|&drafts| drafts == 0));
    epoch.stop();
}

/// An epoch whose speculation pays (four tokens a step at one and a half
/// decode steps' time, a refilled token at a tenth of a decode step)
/// refills a request with two hundred tokens left, which then speculates,
/// and not one with four, whose remaining tokens save less than the
/// refill's thirty-two tokens cost.
#[test]
#[ignore = "draft cache refill: no device layout for a refill call, whose batch the door drops \
            and the request ends migratable (T5)"]
fn a_epoch_whose_speculation_pays_refills_long_requests_only() {
    let mut epoch = Epoch::start(3, 1.5);
    epoch.measure();

    let (_, verifications) = epoch.run(request(3, reusing(6_000), 4));
    assert!(verifications.iter().all(|&drafts| drafts == 0));

    let (read, verifications) = epoch.run(request(4, reusing(7_000), 200));
    assert_eq!(read, 200);
    assert!(verifications.iter().any(|&drafts| drafts > 0));
    epoch.stop();
}
