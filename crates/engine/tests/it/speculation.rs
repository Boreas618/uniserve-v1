//! Draft cache refills in a world that speculates, weighed by what the
//! world measures: a request that reuses a prefix whose draft cache pages
//! end before its context's verifies no drafts, and the scheduler refills
//! the missing entries only when the decode time speculation would save
//! over the tokens the request may still generate exceeds the time of
//! evaluating them again.
//!
//! Each world first measures its steps: a request that speculates (the
//! drafter's tokens per verification step and a step with drafts), a
//! request of two tokens (a step without drafts) and both prompts (the
//! time per prompt token). Then requests reuse the first one's prompt,
//! whose draft cache page covers fewer tokens than its context pages:
//! each lacks the draft cache entries of the positions between.

use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::Receiver;
use engine::{
    BatchEvent, CommandSender, Event, SchedulerConfig, SimEngine, SimExecutor, SimSpeculation,
};
use engine::{GenerationConstraint, GenerationRequest, ImageGenerationConfig};
use ipc::{CallKind, ForwardMode, ImageParams, RequestId, SamplingParams};

use crate::support;

/// Tokens of a context page; a draft cache page holds `SPAN` of them.
const BLOCK: u32 = 16;
const SPAN: u32 = 16;
/// The prompt the later requests reuse: past one draft cache page and
/// within the next.
const SHARED: usize = 300;

/// A greedy text request of `prompt` that generates at most `max_tokens`.
fn request(id: u64, prompt: Vec<u32>, max_tokens: usize) -> GenerationRequest {
    GenerationRequest {
        request_id: RequestId(id),
        prompt_token_ids: prompt,
        multimodal_inputs: Default::default(),
        negative_prompt_token_ids: Vec::new(),
        constraint: GenerationConstraint::UndOnly,
        sampling: SamplingParams::default(),
        image: ImageParams::default(),
        max_images: 1,
        max_und_tokens: max_tokens,
        include_stop_token: false,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: Default::default(),
        image_generation: ImageGenerationConfig::default(),
        readout: Vec::new(),
        generated_token_ids: Vec::new(),
        generated_images: Vec::new(),
        trajectory: Default::default(),
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

/// A scheduler that speculates three drafts per request over the
/// simulator, which speculates and times its steps as `speculation`
/// states, and the batches it submits.
struct World {
    handle: CommandSender,
    thread: thread::JoinHandle<()>,
    batches: Receiver<BatchEvent>,
    /// The calls submitted so far.
    calls: Vec<ipc::Call>,
}

impl World {
    fn start(speculation: SimSpeculation) -> Self {
        let mut sim = SimEngine::new();
        sim.set_text_len(10_000);
        sim.set_block_size(BLOCK);
        sim.add_draft_group(SPAN);
        sim.set_speculation(speculation);
        let mut executor = SimExecutor::new(sim);
        let batches = executor.observe();
        let scheduler = support::scheduler_with(
            Box::new(executor),
            SimEngine::special_tokens(),
            SchedulerConfig {
                draft_tokens: 3,
                ..Default::default()
            },
        )
        .unwrap();
        let (tx, rx) = crossbeam_channel::unbounded();
        let thread = thread::spawn(move || {
            let _ = scheduler.run(rx);
        });
        Self {
            handle: CommandSender::new(tx),
            thread,
            batches,
            calls: Vec::new(),
        }
    }

    /// Runs `request` to its end, alone; returns how many tokens it read.
    fn run(&mut self, request: GenerationRequest) -> usize {
        let mut events = self.handle.submit(request).unwrap();
        let mut tokens = 0;
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            assert!(Instant::now() < deadline, "the request did not end");
            match events.try_recv() {
                Ok(Event::TextToken { .. }) => tokens += 1,
                Ok(event) if event.is_terminal() => break,
                Ok(_) => {}
                Err(_) => thread::sleep(Duration::from_millis(1)),
            }
        }
        self.calls.extend(
            self.batches
                .try_iter()
                .filter_map(|event| match event {
                    BatchEvent::Submitted(batch) => {
                        Some(batch.requests.into_iter().map(|(call, _)| call))
                    }
                    BatchEvent::Resolved { .. } => None,
                })
                .flatten(),
        );
        tokens
    }

    /// Measures the world's steps: a request that speculates over the
    /// shared prompt, and a request of two tokens whose second step
    /// carries no drafts.
    fn measure(&mut self) {
        self.run(request(1, reusing(4_000), 40));
        self.run(request(2, vec![5_000, 5_001, 5_002], 2));
        assert!(
            self.verifications(1).any(|drafts| drafts > 0),
            "the first request did not speculate"
        );
    }

    /// The draft counts of `id`'s verifications.
    fn verifications(&self, id: u64) -> impl Iterator<Item = usize> + '_ {
        self.calls
            .iter()
            .filter(move |call| {
                call.request_key.request_id == RequestId(id)
                    && call.kind == CallKind::Forward(ForwardMode::Verify)
            })
            .map(|call| call.input_token_ids.len() - 1)
    }

    /// Whether `id`'s draft cache was refilled.
    fn refilled(&self, id: u64) -> bool {
        self.calls
            .iter()
            .any(|call| call.request_key.request_id == RequestId(id) && call.draft.refill.is_some())
    }

    fn stop(self) {
        self.handle.shutdown();
        self.thread.join().unwrap();
    }
}

/// A world whose verification step costs more decode steps than the
/// tokens it commits (two tokens a step, at three decode steps' time)
/// refills nothing: a request of two hundred tokens left verifies no
/// drafts to its end.
#[test]
fn a_world_whose_speculation_does_not_pay_refills_nothing() {
    let mut world = World::start(SimSpeculation {
        accepted: 1,
        decode_us: 1_000,
        verify_us: 3_000,
        prompt_token_us: 100,
    });
    world.measure();

    let read = world.run(request(3, reusing(6_000), 200));
    assert_eq!(read, 200);
    assert!(!world.refilled(3));
    assert!(world.verifications(3).all(|drafts| drafts == 0));
    world.stop();
}

/// A world whose speculation pays (four tokens a step at one and a half
/// decode steps' time, a refilled token at a tenth of a decode step)
/// refills a request with two hundred tokens left, which then speculates,
/// and not one with four, whose remaining tokens save less than the
/// refill's thirty-two tokens cost.
#[test]
fn a_world_whose_speculation_pays_refills_long_requests_only() {
    let mut world = World::start(SimSpeculation {
        accepted: 3,
        decode_us: 1_000,
        verify_us: 1_500,
        prompt_token_us: 100,
    });
    world.measure();

    world.run(request(3, reusing(6_000), 4));
    assert!(!world.refilled(3));
    assert!(world.verifications(3).all(|drafts| drafts == 0));

    let read = world.run(request(4, reusing(7_000), 200));
    assert_eq!(read, 200);
    assert!(world.refilled(4));
    assert!(world.verifications(4).any(|drafts| drafts > 0));
    world.stop();
}
