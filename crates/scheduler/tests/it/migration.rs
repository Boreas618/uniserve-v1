//! A token request that migrates: the scheduler ends it as migratable, and
//! admits it again with the tokens it generated, continuing its budget, its
//! sampling constraints and its usage.

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use ipc::execution::{Batch, Outcome};
use ipc::{RequestId, SamplingParams};
use proto::request::MigrationCause;
use scheduler::GenerationRequest;
use scheduler::{CommandSender, Event, EventRx, FinishReason, SchedulerConfig};
use testkit::FakeBatches;

use crate::support;
use crate::support::fake;

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
        sampling,
        max_und_tokens,
        include_stop_token: false,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: Default::default(),
        generated_token_ids: generated,
        adapter: 0,
    }
}

/// A scheduler over the fake worker, admitting at most
/// `max_running_requests` requests at once.
fn start(max_running_requests: usize) -> (CommandSender, thread::JoinHandle<()>) {
    let pool = Arc::new(FakeBatches::new(vec![fake::text_pool("model", 8)]));
    let scheduler = fake::scheduler_over(
        pool,
        SchedulerConfig {
            max_running_requests,
            ..Default::default()
        },
        Default::default(),
    )
    .expect("the scheduler is built over the fake");
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
            Ok(Event::TextTokens { ids }) => tokens.extend(ids),
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
    let (handle, thread) = start(8);
    let mut events = handle
        .submit(request(1, vec![1_500, 1_501], SamplingParams::default(), 6))
        .unwrap();
    let (tokens, terminal) = drain(&mut events);
    handle.shutdown();
    thread.join().unwrap();

    // It samples at the positions that follow its prompt and the two tokens
    // it generated before: the draws of an uninterrupted request.
    let continued: Vec<u32> = (5..9).map(|position| fake::token_at(1, position)).collect();
    assert_eq!(tokens, continued);
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

/// Parking ends a running token request after the tokens it generated, and
/// a queued one at once, each with a migratable end in place of a finish,
/// and answers with the scheduler's prefix cache index once nothing runs.
#[test]
fn parking_ends_running_and_queued_requests_as_migratable() {
    let (handle, thread) = start(1);
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
            Ok(Event::TextTokens { ids }) => before += ids.len(),
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    assert!(
        before >= 3,
        "the first request generates before it migrates"
    );
    let parked = handle.park(false);

    let (_, terminal) = drain(&mut running);
    assert!(
        matches!(
            terminal,
            Some(Event::Migratable {
                cause: MigrationCause::Drain,
                ..
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
                cause: MigrationCause::Drain,
                ..
            })
        ),
        "{terminal:?}"
    );
    let parked = parked
        .recv_timeout(Duration::from_secs(10))
        .expect("the scheduler parked")
        .expect("the scheduler parked without an error");
    // The running request published the pages of its prompt and the tokens
    // it generated, which a successor imports; it is the one request handed
    // over with a state, and held the units its pages took. A page it
    // published and its own continuation of that page are two entries of
    // the index over the same units.
    assert!(
        parked.pages.iter().all(|page| !page.units.is_empty()),
        "{parked:?}"
    );
    assert_eq!(parked.requests, 1);
    let units: std::collections::BTreeSet<_> = parked
        .pages
        .iter()
        .flat_map(|page| page.units.iter().copied())
        .collect();
    assert!(parked.held_units as usize >= units.len(), "{parked:?}");
    assert!(handle.resume(false).recv().unwrap().is_ok());
    handle.shutdown();
    thread.join().unwrap();
}

/// The pages of the tokens a request generated serve its migration: the
/// scheduler publishes each page of generated tokens once a decode wrote
/// it, so the migrated request, whose prompt continues with those tokens,
/// computes only what follows the last complete page.
#[test]
fn a_migration_reuses_the_pages_of_the_tokens_it_generated() {
    use std::sync::atomic::Ordering;

    let mut inventory = fake::text_pool("model", 8);
    for cache in fake::cache_of(&mut inventory) {
        for group in &mut cache.groups {
            group.block_size = 4;
        }
    }
    let pool = Arc::new(FakeBatches::new(vec![inventory]));
    let scheduler = fake::scheduler_over(pool, SchedulerConfig::default(), Default::default())
        .expect("the scheduler is built over the fake");
    let stats = scheduler.stats_handle();
    let (tx, rx) = crossbeam_channel::unbounded();
    let thread = thread::spawn(move || {
        let _ = scheduler.run(rx);
    });
    let handle = CommandSender::new(tx);

    // Three prompt tokens and twenty generated ones: the cache holds the
    // first twenty-two, five complete pages of four.
    let mut first = handle
        .submit(request(1, Vec::new(), SamplingParams::default(), 20))
        .unwrap();
    let (generated, terminal) = drain(&mut first);
    assert!(
        matches!(terminal, Some(Event::Finished { .. })),
        "{terminal:?}"
    );
    assert_eq!(generated.len(), 20);
    let hits_before = stats.prefix.hit_tokens.load(Ordering::Relaxed);

    // Its migration's prompt is the twenty-three tokens: it reuses the five
    // pages, all but the prompt's last token being cached, and computes the
    // rest.
    let mut migrated = handle
        .submit(request(2, generated, SamplingParams::default(), 24))
        .unwrap();
    let (tokens, terminal) = drain(&mut migrated);
    handle.shutdown();
    thread.join().unwrap();

    assert!(
        matches!(terminal, Some(Event::Finished { .. })),
        "{terminal:?}"
    );
    assert_eq!(tokens.len(), 4);
    assert_eq!(
        stats.prefix.hit_tokens.load(Ordering::Relaxed) - hits_before,
        20
    );
}

/// How the fake worker fails once told to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fault {
    /// The device fails every batch it holds, as a lost device does.
    Lost,
    /// Every call reports non-finite logits, as a worker whose forward
    /// produced a NaN does (`ErrorCode::NumericalFault`).
    NonFinite,
    /// The worker refuses the request's next calls as invalid input
    /// (`ErrorCode::InputError`), once.
    Refused,
}

/// The fault the fake answers with from the moment it is set, and how many
/// batches a `Refused` fault has refused: one, after which the fault ends.
#[derive(Default)]
struct Faults {
    fault: Option<Fault>,
    refused_batches: usize,
}

/// The fake's answer to `batch` under `faults`: every call completed by its
/// draw, or failed with the fault's code.
fn faulting(batch: &Batch, faults: &Mutex<Faults>) -> Outcome<ipc::execution::BatchOutput> {
    let mut output = testkit::completed(batch, testkit::Tokens::Draw, testkit::Extent::Stated);
    let mut faults = faults.lock().unwrap();
    let code = match faults.fault {
        None | Some(Fault::Lost) => return Outcome::Success(output),
        Some(Fault::NonFinite) => ipc::ErrorCode::NumericalFault,
        Some(Fault::Refused) if faults.refused_batches == 0 => ipc::ErrorCode::InputError,
        Some(Fault::Refused) => return Outcome::Success(output),
    };
    if faults.fault == Some(Fault::Refused) {
        faults.refused_batches += 1;
    }
    for call in &mut output.calls {
        call.status = ipc::CallStatus::Error;
        call.error_code = Some(code);
        call.error_message =
            (code == ipc::ErrorCode::InputError).then(|| "invalid input".to_owned());
        call.tokens.clear();
    }
    Outcome::Success(output)
}

/// What a request run by `fail_while_generating` came to.
struct Faulted {
    /// The tokens the request delivered.
    tokens: Vec<u32>,
    /// Its terminal event.
    terminal: Option<Event>,
    /// Whether the scheduler stopped on a fatal failure.
    stopped: bool,
    /// That failure's first error.
    failure: Option<String>,
    /// The batches a `Refused` fault refused.
    refused_batches: usize,
}

/// Runs a request on a scheduler, on trial or not, until it generated a few
/// tokens, then fails its workers with `fault`.
fn fail_while_generating(trial: bool, fault: Fault, request: GenerationRequest) -> Faulted {
    let faults = Arc::new(Mutex::new(Faults::default()));
    let inventory = fake::text_pool("model", 8);
    let answering = Arc::clone(&faults);
    let pool = Arc::new(
        FakeBatches::new(vec![inventory])
            .answering(Arc::new(move |batch| faulting(batch, &answering))),
    );
    let mut scheduler = fake::scheduler_over(
        Arc::<testkit::FakeBatches>::clone(&pool),
        SchedulerConfig::default(),
        Default::default(),
    )
    .expect("the scheduler is built over the fake");
    scheduler.set_trial(trial);
    let failure = scheduler.failure_handle();
    let (tx, rx) = crossbeam_channel::unbounded();
    let thread = thread::spawn(move || scheduler.run(rx));
    let handle = CommandSender::new(tx);

    let mut events = handle.submit(request).unwrap();
    let (mut tokens, _) = support::wait_for_tokens(&mut events, 4, Duration::from_secs(10));
    assert!(tokens.len() >= 4, "the request generates before the fault");
    if fault == Fault::Lost {
        // The device holds the next batch and then fails everything it holds.
        pool.hold();
        let deadline = Instant::now() + Duration::from_secs(10);
        while pool.held() == 0 && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(1));
        }
        pool.fail_held("the worker was lost");
    } else {
        faults.lock().unwrap().fault = Some(fault);
    }

    let (after, terminal) = drain(&mut events);
    tokens.extend(after);
    // A scheduler on trial that a fault stopped has ended its loop.
    if !(trial && failure.get().is_some()) {
        handle.shutdown();
    }
    let stopped = thread.join().unwrap();
    Faulted {
        tokens,
        terminal,
        stopped,
        failure: failure.get().cloned(),
        refused_batches: faults.lock().unwrap().refused_batches,
    }
}

/// An epoch on trial is not repaired: a lost worker stops its scheduler,
/// which hands every request that resumes from its generated tokens back
/// as migratable for a loss, after every token it delivered, and reports
/// the loss as its first error.
#[test]
fn a_scheduler_on_trial_hands_its_requests_back_when_its_workers_are_lost() {
    let Faulted {
        tokens,
        terminal,
        stopped,
        failure,
        ..
    } = fail_while_generating(
        true,
        Fault::Lost,
        request(1, Vec::new(), SamplingParams::default(), 100_000),
    );
    assert!(stopped, "a lost worker stops a scheduler on trial");
    assert!(tokens.len() >= 4, "{tokens:?}");
    assert!(
        matches!(
            terminal,
            Some(Event::Migratable {
                cause: MigrationCause::Loss,
                ..
            })
        ),
        "{terminal:?}"
    );
    let failure = failure.expect("the scheduler reports its first error");
    assert!(failure.contains("the worker was lost"), "{failure}");
}

/// Non-finite logits fail an epoch on trial as a whole: the scheduler stops
/// and hands the request back as migratable, naming the batch and its
/// requests as its first error. Outside a trial they fail the request
/// alone, as any failed call does, and the scheduler serves on.
#[test]
fn non_finite_logits_fail_a_epoch_on_trial_and_a_request_otherwise() {
    let Faulted {
        tokens,
        terminal,
        stopped,
        failure,
        ..
    } = fail_while_generating(
        true,
        Fault::NonFinite,
        request(1, Vec::new(), SamplingParams::default(), 100_000),
    );
    assert!(stopped, "non-finite logits stop a scheduler on trial");
    assert!(
        matches!(
            terminal,
            Some(Event::Migratable {
                cause: MigrationCause::Loss,
                ..
            })
        ),
        "{terminal:?}"
    );
    let failure = failure.expect("the scheduler reports its first error");
    assert!(
        failure.contains("non-finite logits for requests [1]"),
        "{failure}"
    );
    assert!(tokens.len() >= 4, "{tokens:?}");

    let Faulted {
        terminal,
        stopped,
        failure,
        ..
    } = fail_while_generating(
        false,
        Fault::NonFinite,
        request(1, Vec::new(), SamplingParams::default(), 100_000),
    );
    assert!(!stopped, "a scheduler not on trial serves on");
    assert!(failure.is_none(), "{failure:?}");
    assert!(
        matches!(terminal, Some(Event::Error { .. })),
        "{terminal:?}"
    );
}

/// A request a worker refuses as invalid input fails alone, on trial too:
/// the refusal is the request's, not a worker fault, so the epoch on trial
/// serves on.
#[test]
fn a_refused_request_fails_alone_on_trial() {
    let Faulted {
        terminal,
        stopped,
        failure,
        refused_batches,
        ..
    } = fail_while_generating(
        true,
        Fault::Refused,
        request(1, Vec::new(), SamplingParams::default(), 100_000),
    );
    assert_eq!(refused_batches, 1, "the worker refused one batch");
    assert!(!stopped, "a refusal does not stop a scheduler on trial");
    assert!(failure.is_none(), "{failure:?}");
    assert!(
        matches!(terminal, Some(Event::Error { .. } | Event::Rejected { .. })),
        "{terminal:?}"
    );
}

/// A successor promotes only while the units its own cache holds free hold
/// what the requests its predecessor hands over compute: a request that
/// recomputes more units than are free refuses the promotion, with both
/// counts, as does a request that reserved its whole cache at admission
/// even when the others continue with the state they reached; one
/// continuing with the state it reached needs a page.
#[test]
fn a_promotion_refuses_the_requests_its_cache_cannot_hold() {
    let mut inventory = fake::text_pool("model", 8);
    for cache in fake::cache_of(&mut inventory) {
        cache.num_units = 16;
        for group in &mut cache.groups {
            group.block_size = 16;
        }
    }
    let pool = Arc::new(FakeBatches::new(vec![inventory]));
    let scheduler = fake::scheduler_over(pool, SchedulerConfig::default(), Default::default())
        .expect("the scheduler is built over the fake");
    let (tx, rx) = crossbeam_channel::unbounded();
    let thread = thread::spawn(move || {
        let _ = scheduler.run(rx);
    });
    let handle = CommandSender::new(tx);

    let parked = scheduler::ParkedCache {
        pages: Vec::new(),
        lent: Vec::new(),
        held_units: 64,
        requests: 1,
        reserved_units: 0,
        reserving: 0,
    };
    let refused = handle
        .promote(parked.clone(), true, false)
        .recv_timeout(Duration::from_secs(10))
        .expect("the scheduler answers");
    let reason = refused.expect_err("15 free units cannot hold 65");
    assert!(
        reason.contains("15 free units") && reason.contains("need 65"),
        "{reason}"
    );
    // A request that reserved its whole cache at admission (one with
    // images) reserves it whole again, whether or not the others recompute.
    let reserving = scheduler::ParkedCache {
        reserved_units: 64,
        reserving: 1,
        ..parked.clone()
    };
    let reason = handle
        .promote(reserving, false, false)
        .recv_timeout(Duration::from_secs(10))
        .expect("the scheduler answers")
        .expect_err("15 free units cannot hold a whole reservation of 64");
    assert!(
        reason.contains("15 free units") && reason.contains("need 64"),
        "{reason}"
    );
    assert!(
        handle
            .promote(parked, false, false)
            .recv_timeout(Duration::from_secs(10))
            .expect("the scheduler answers")
            .is_ok()
    );

    handle.shutdown();
    thread.join().unwrap();
}
