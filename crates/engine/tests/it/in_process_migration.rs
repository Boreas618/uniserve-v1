//! A request whose group is lost leaves the engine as migratable, and a
//! continuation picks up from the tokens it generated, over the simulated
//! executor.
//!
//! The launched case kills a rank under a running request and asserts two
//! events: the request ends as `Migratable` with a loss cause, and a request
//! submitted with the tokens it generated finishes with the reference's
//! sequence. Here the loss is the simulated executor's own, so the same two
//! statements hold without a process to kill.

use std::thread;
use std::time::{Duration, Instant};

use engine::{
    CommandSender, Event, FinishReason, GenerationConstraint, GenerationRequest, SimEngine,
    SimExecutor, SimLoss, SpecialTokenIds, WorkerGroupId,
};
use ipc::{ImageParams, RequestId, SamplingParams};
use proto::request::MigrationCause;

use crate::support;

/// How long a request may take to reach its terminal event.
const DEADLINE: Duration = Duration::from_secs(60);

/// How many tokens the requests generate.
const GENERATED: usize = 6;

/// The special tokens the simulated engine states.
fn special_tokens() -> SpecialTokenIds {
    SimEngine::special_tokens()
}

/// The group the simulated executor serves.
fn group() -> WorkerGroupId {
    WorkerGroupId("sim".to_owned())
}

/// A greedy text request that carries `generated` and generates up to
/// `budget` tokens in all.
fn request(id: u64, generated: Vec<u32>, budget: usize) -> GenerationRequest {
    GenerationRequest {
        request_id: RequestId(id),
        prompt_token_ids: vec![7; 8],
        negative_prompt_token_ids: Vec::new(),
        multimodal_inputs: Default::default(),
        constraint: GenerationConstraint::UndOnly,
        sampling: SamplingParams::default(),
        image: ImageParams::default(),
        max_images: 1,
        max_und_tokens: budget,
        include_stop_token: false,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: Default::default(),
        image_generation: Default::default(),
        readout: Vec::new(),
        generated_token_ids: generated,
        adapter: 0,
        generated_images: Vec::new(),
        trajectory: Default::default(),
    }
}

/// A scheduler over a fresh simulated executor, its commands, and the loss
/// the case injects.
fn serving() -> (CommandSender, SimLoss, thread::JoinHandle<bool>) {
    let mut sim = SimEngine::new();
    sim.mut_info_for_test().supported_calls = vec![
        ipc::CallKind::Forward(ipc::ForwardMode::Prefill),
        ipc::CallKind::Forward(ipc::ForwardMode::Decode),
    ];
    let mut executor = SimExecutor::new(sim);
    let loss = executor.loss();
    let scheduler = support::scheduler(Box::new(executor), special_tokens(), 32)
        .expect("the scheduler is built over the simulated executor");
    let (tx, rx) = crossbeam_channel::unbounded();
    let commands = CommandSender::new(tx);
    let scheduler_thread = thread::spawn(move || scheduler.run(rx));
    (commands, loss, scheduler_thread)
}

/// Collects `want` tokens, or fails by name after the bound.
fn tokens(events: &mut engine::EventRx, want: usize) -> Vec<u32> {
    let mut tokens = Vec::new();
    let deadline = Instant::now() + DEADLINE;
    while tokens.len() < want {
        assert!(
            Instant::now() < deadline,
            "the request did not reach {want} tokens within the bound"
        );
        match events.try_recv() {
            Ok(Event::TextToken { id, .. }) => tokens.push(id),
            Ok(Event::TextTokens { ids }) => tokens.extend(ids),
            Ok(event) if event.is_terminal() => {
                panic!("the request ended after {} tokens: {event:?}", tokens.len())
            }
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    tokens
}

/// Collects a request's tokens until its terminal event.
fn until_terminal(events: &mut engine::EventRx) -> (Vec<u32>, Event) {
    let mut tokens = Vec::new();
    let deadline = Instant::now() + DEADLINE;
    while Instant::now() < deadline {
        match events.try_recv() {
            Ok(Event::TextToken { id, .. }) => tokens.push(id),
            Ok(Event::TextTokens { ids }) => tokens.extend(ids),
            Ok(event) if event.is_terminal() => return (tokens, event),
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    panic!("the request did not end within the bound");
}

/// A request whose group is lost leaves as migratable, and a continuation
/// carrying its tokens generates the reference's sequence.
///
/// What this case states holds in process: the continuation's tokens equal
/// the uninterrupted reference's, and a killed pid is only how the loss was
/// caused. The other half — that a real process dies, a real one replaces it
/// and requests migrate without failing — is measured end to end on GPUs by
/// the caller, and a simulation cannot stand in for it, so no unit case here
/// claims it.
#[test]
fn a_request_on_a_lost_group_migrates_and_continues_after_the_relaunch() {
    // The reference: the same request id, uninterrupted, over the same group.
    let (commands, _, reference_thread) = serving();
    let mut reference_events = commands
        .submit(request(103, Vec::new(), GENERATED))
        .expect("the reference request is accepted");
    let (reference, terminal) = until_terminal(&mut reference_events);
    assert!(
        matches!(terminal, Event::Finished { .. }) && reference.len() == GENERATED,
        "the reference generated {reference:?} and ended with {terminal:?}"
    );

    // The request, then its group's loss while it runs.
    let (commands, loss, scheduler_thread) = serving();
    let mut events = commands
        .submit(request(103, Vec::new(), GENERATED))
        .expect("the request is accepted");
    let first = tokens(&mut events, 3);
    loss.lose_to_replace(group(), "the rank died holding the request");

    let (rest, terminal) = until_terminal(&mut events);
    let mut generated = first;
    generated.extend(rest);
    assert!(
        matches!(
            terminal,
            Event::Migratable {
                cause: MigrationCause::Loss,
                ..
            }
        ),
        "the lost request did not leave as migratable: {terminal:?}"
    );

    // The group is replaced, and a continuation carrying the tokens the lost
    // request generated finishes with the reference's sequence.
    loss.restore(&group());
    let mut continued = commands
        .submit(request(103, generated.clone(), GENERATED))
        .expect("the continuation is accepted");
    let (tail, finished) = until_terminal(&mut continued);
    generated.extend(tail);
    let finished_well = matches!(
        &finished,
        Event::Finished {
            completion_tokens,
            reason,
            ..
        } if *completion_tokens as usize == GENERATED && *reason != FinishReason::Error
    );
    assert!(
        finished_well,
        "the continuation did not finish at {GENERATED} tokens: {finished:?}"
    );
    assert_eq!(
        generated, reference,
        "the continuation generated {generated:?}, the uninterrupted request {reference:?}"
    );

    commands.shutdown();
    let _ = scheduler_thread.join();
    let _ = reference_thread.join();
}
