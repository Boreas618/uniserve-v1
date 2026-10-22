//! A token request that migrates: the scheduler ends it as migratable, and
//! admits it again with the tokens it generated, continuing its budget, its
//! sampling constraints and its usage.

use std::thread;
use std::time::{Duration, Instant};

use engine::{
    CommandSender, Event, EventRx, FinishReason, SchedulerConfig, SimEngine, SimExecutor,
};
use engine::{GenerationConstraint, GenerationRequest, ImageGenerationConfig};
use ipc::{ImageParams, RequestId, SamplingParams};
use proto::request::MigrationCause;

use crate::support;

/// A text request of the prompt `[1, 2, 3]` that generated `generated`
/// before it migrated, under `sampling` and at most `max_und_tokens` tokens.
fn request(
    id: u64,
    generated: Vec<u32>,
    sampling: SamplingParams,
    max_und_tokens: usize,
) -> GenerationRequest {
    GenerationRequest {
        request_id: RequestId(id),
        prompt_token_ids: vec![1, 2, 3],
        multimodal_inputs: Default::default(),
        negative_prompt_token_ids: Vec::new(),
        constraint: GenerationConstraint::UndOnly,
        sampling,
        image: ImageParams::default(),
        max_images: 1,
        max_und_tokens,
        include_stop_token: false,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: Default::default(),
        image_generation: ImageGenerationConfig::default(),
        readout: Vec::new(),
        generated_token_ids: generated,
        adapter: 0,
    }
}

/// A scheduler over the simulator, which ends text after `text_len` tokens
/// with its EOS, admitting at most `max_running_requests` requests at once.
fn start(text_len: usize, max_running_requests: usize) -> (CommandSender, thread::JoinHandle<()>) {
    let mut sim = SimEngine::new();
    sim.set_text_len(text_len);
    let scheduler = support::scheduler_with(
        Box::new(SimExecutor::new(sim)),
        SimEngine::special_tokens(),
        SchedulerConfig {
            max_running_requests,
            ..Default::default()
        },
    )
    .unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let thread = thread::spawn(move || {
        let _ = scheduler.run(rx);
    });
    (CommandSender::new(tx), thread)
}

/// The tokens `events` delivers and its terminal event, within ten seconds.
fn drain(events: &mut EventRx) -> (Vec<u32>, Option<Event>) {
    let mut tokens = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        match events.try_recv() {
            Ok(Event::TextToken { id, .. }) => tokens.push(id),
            Ok(event) if event.is_terminal() => return (tokens, Some(event)),
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    (tokens, None)
}

/// A migrated request generates what its limit leaves after the tokens it
/// generated before, and its usage charges its own prompt as prompt and
/// every token it generated, before and here, as completion.
#[test]
fn a_migrated_request_continues_its_budget_and_charges_its_whole_usage() {
    let (handle, thread) = start(1_000, 8);
    let mut events = handle
        .submit(request(1, vec![1_500, 1_501], SamplingParams::default(), 6))
        .unwrap();
    let (tokens, terminal) = drain(&mut events);
    handle.shutdown();
    thread.join().unwrap();

    assert_eq!(tokens.len(), 4, "{tokens:?}");
    match terminal {
        Some(Event::Finished {
            reason,
            prompt_tokens,
            completion_tokens,
            ..
        }) => {
            assert_eq!(reason, FinishReason::MaxTokens);
            assert_eq!(prompt_tokens, 3);
            assert_eq!(completion_tokens, 6);
        }
        other => panic!("expected a finish, got {other:?}"),
    }
}

/// The tokens a request generated before it migrated count toward its
/// `min_tokens`: the simulator prefers EOS from its second token on, and
/// the scheduler suppresses it only until the request generated five tokens
/// in all.
#[test]
fn generated_tokens_count_toward_min_tokens() {
    let (handle, thread) = start(1, 8);
    let sampling = SamplingParams {
        min_tokens: 5,
        ..Default::default()
    };
    let mut events = handle
        .submit(request(1, vec![1_500, 1_501], sampling, 50))
        .unwrap();
    let (tokens, terminal) = drain(&mut events);
    handle.shutdown();
    thread.join().unwrap();

    let eos = SimEngine::special_tokens().eos;
    let text = tokens.iter().filter(|token| !eos.contains(token)).count();
    assert_eq!(text, 3, "{tokens:?}");
    assert!(
        matches!(terminal, Some(Event::Finished { reason: FinishReason::Eos, completion_tokens, .. }) if completion_tokens >= 5),
        "{terminal:?}"
    );
}

/// A bad word whose prefix the generated tokens end with bans its last token
/// at once: the bias makes 4321 the natural choice, and the request that
/// generated a 4321 before it migrated may not repeat it.
#[test]
fn a_bad_word_continues_from_the_generated_tokens() {
    let (handle, thread) = start(1_000, 8);
    let sampling = SamplingParams {
        allowed_token_ids: Some(vec![1234, 4321]),
        logit_bias: vec![(4321, 1000.0)],
        bad_words_ids: vec![vec![4321, 4321]],
        ..Default::default()
    };
    let mut events = handle.submit(request(1, vec![4321], sampling, 4)).unwrap();
    let (tokens, _) = drain(&mut events);
    handle.shutdown();
    thread.join().unwrap();

    assert_eq!(tokens.first(), Some(&1234), "{tokens:?}");
}

/// Migrating ends a running token request after the tokens it generated, and
/// a queued one at once, each with a migratable end in place of a finish.
#[test]
fn migrating_ends_running_and_queued_requests_as_migratable() {
    let (handle, thread) = start(100_000, 1);
    let mut events = handle
        .submit_all([
            request(1, Vec::new(), SamplingParams::default(), 100_000).into(),
            request(2, Vec::new(), SamplingParams::default(), 100_000).into(),
        ])
        .unwrap()
        .into_iter();
    let (mut running, mut queued) = (events.next().unwrap(), events.next().unwrap());

    // The first request runs alone and generates before the migration.
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut before = 0;
    while before < 3 && Instant::now() < deadline {
        match running.try_recv() {
            Ok(Event::TextToken { .. }) => before += 1,
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    assert_eq!(before, 3, "the first request generates before it migrates");
    handle.migrate(MigrationCause::Drain);

    let (_, terminal) = drain(&mut running);
    assert!(
        matches!(
            terminal,
            Some(Event::Migratable {
                cause: MigrationCause::Drain
            })
        ),
        "{terminal:?}"
    );
    let (tokens, terminal) = drain(&mut queued);
    assert!(tokens.is_empty(), "{tokens:?}");
    assert!(
        matches!(
            terminal,
            Some(Event::Migratable {
                cause: MigrationCause::Drain
            })
        ),
        "{terminal:?}"
    );
    handle.shutdown();
    thread.join().unwrap();
}
