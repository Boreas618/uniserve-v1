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
        generated_images: Vec::new(),
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

/// Parking ends a running token request after the tokens it generated, and
/// a queued one at once, each with a migratable end in place of a finish,
/// and answers with the scheduler's prefix cache index once nothing runs.
#[test]
fn parking_ends_running_and_queued_requests_as_migratable() {
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
    let parked = handle.park();

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
    // over with a state, and held the units its pages took.
    assert!(
        parked.pages.iter().all(|page| !page.units.is_empty()),
        "{parked:?}"
    );
    assert_eq!(parked.requests, 1);
    assert!(
        parked.held_units as usize >= parked.pages.len(),
        "{parked:?}"
    );
    assert!(handle.resume().recv().unwrap().is_ok());
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

    let mut sim = SimEngine::new();
    sim.set_text_len(1_000);
    sim.set_block_size(4);
    let scheduler = support::scheduler_with(
        Box::new(SimExecutor::new(sim)),
        SimEngine::special_tokens(),
        SchedulerConfig::default(),
    )
    .unwrap();
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

/// How the workers of [`FaultingExecutor`] fail once told to.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fault {
    /// Every poll fails, as a worker process's exit does.
    Lost,
    /// Every call reports non-finite logits, as a worker whose forward
    /// produced a NaN does (`ErrorCode::NumericalFault`).
    NonFinite,
    /// The worker refuses the request's next calls as invalid input
    /// (`ErrorCode::InputError`); the calls of the batch after them, which
    /// read the refused calls' products, fail for want of them
    /// (`ErrorCode::InvalidDescriptor`), refusing nothing themselves.
    Refused,
}

/// The simulator's workers, which fail as `fault` states from the moment it
/// is set. A batch whose calls computed non-finite logits is reported as
/// workers report one: the failure of the work that depends on its calls
/// first, and the batch's result, its failed calls included, at the next
/// poll.
struct FaultingExecutor {
    inner: SimExecutor,
    fault: std::sync::Arc<std::sync::Mutex<Option<Fault>>>,
    queued: Option<engine::BatchResult>,
    /// The batches a `Refused` fault failed: the refusing one, then the
    /// dependent one. The fault ends after both.
    refused_batches: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    /// The report a `Refused` fault holds back: the refusing batch's until
    /// the batch after it reports, so that the scheduler submitted that one
    /// before it knew of the refusal, then the dependent batch's.
    held: Option<engine::BatchResult>,
}

impl engine::Executor for FaultingExecutor {
    fn info(&self) -> &engine::ExecutorInfo {
        self.inner.info()
    }

    fn is_ready(&self, worker: &engine::WorkerGroupId) -> bool {
        self.inner.is_ready(worker)
    }

    fn has_capacity(&self, worker: &engine::WorkerGroupId) -> bool {
        self.inner.has_capacity(worker)
    }

    fn command_has_capacity(&self, command: &ipc::BatchCommand) -> bool {
        self.inner.command_has_capacity(command)
    }

    fn submit(&mut self, batch: engine::ExecutionBatch) -> Result<(), engine::ExecutorSubmitError> {
        self.inner.submit(batch)
    }

    fn poll(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<engine::BatchResult>, engine::ExecutorError> {
        if let Some(report) = self.queued.take() {
            return Ok(Some(report));
        }
        let fault = *self.fault.lock().unwrap();
        if fault == Some(Fault::Lost) {
            return Err(anyhow::anyhow!("worker process exited"));
        }
        let failed = self
            .refused_batches
            .load(std::sync::atomic::Ordering::SeqCst);
        if fault == Some(Fault::Refused) && failed == 1 {
            // The dependent batch's report, held since the refusal.
            if let Some(held) = self.held.take() {
                return self.fail(held, ipc::ErrorCode::InvalidDescriptor);
            }
        }
        let report = self.inner.poll(timeout)?;
        let Some(report) = report else {
            return Ok(None);
        };
        let code = match fault {
            Some(Fault::NonFinite) => ipc::ErrorCode::NumericalFault,
            Some(Fault::Refused) if failed == 0 && !report.results.is_empty() => {
                match self.held.take() {
                    // The refusing batch waits for the batch after it.
                    None => {
                        self.held = Some(report);
                        return Ok(None);
                    }
                    Some(refusing) => {
                        self.held = Some(report);
                        return self.fail(refusing, ipc::ErrorCode::InputError);
                    }
                }
            }
            _ => return Ok(Some(report)),
        };
        self.fail(report, code)
    }

    fn close(&mut self) -> Result<(), engine::ExecutorError> {
        self.inner.close()
    }
}

impl FaultingExecutor {
    /// Fails every call of `report` with `code` as workers report it: the
    /// failure of the work that depends on its calls now, naming the
    /// requests refused when `code` refuses input, and the report itself at
    /// the next poll.
    fn fail(
        &mut self,
        mut report: engine::BatchResult,
        code: ipc::ErrorCode,
    ) -> Result<Option<engine::BatchResult>, engine::ExecutorError> {
        if *self.fault.lock().unwrap() == Some(Fault::Refused) {
            self.refused_batches
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        for result in &mut report.results {
            result.output.status = ipc::CallStatus::Error;
            result.output.error_code = Some(code);
            result.output.committed_tokens.clear();
        }
        let refusals = report
            .results
            .iter()
            .filter(|_| code == ipc::ErrorCode::InputError)
            .map(|result| (result.output.request_key, "invalid input".to_owned()))
            .collect();
        let failure = engine::WorkerFailure {
            worker_id: engine::WorkerGroupId("sim".to_owned()),
            endpoints: Vec::new(),
            requests: report
                .results
                .iter()
                .map(|result| result.output.request_key)
                .collect(),
            refusals,
            retired: Vec::new(),
            buffers: Vec::new(),
            execution: engine::WorkerExecError::non_finite(
                report.batch_id,
                report.results.iter().map(|result| &result.output),
            ),
            message: "call failed before its dependent work could complete".into(),
        };
        self.queued = Some(report);
        Err(failure.into())
    }
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
    /// The batches a `Refused` fault failed.
    refused_batches: usize,
}

/// Runs a request on a scheduler, on trial or not, until it generated a few
/// tokens, then fails its workers with `fault`.
fn fail_while_generating(trial: bool, fault: Fault, request: GenerationRequest) -> Faulted {
    let mut sim = SimEngine::new();
    sim.set_text_len(1_000_000);
    // A refusal arrives while the request's next batch is in flight.
    if fault == Fault::Refused {
        sim.set_queue_depth(2);
    }
    let faults = std::sync::Arc::new(std::sync::Mutex::new(None));
    let refused_batches = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut scheduler = support::scheduler_with(
        Box::new(FaultingExecutor {
            inner: SimExecutor::new(sim),
            fault: std::sync::Arc::clone(&faults),
            queued: None,
            refused_batches: std::sync::Arc::clone(&refused_batches),
            held: None,
        }),
        SimEngine::special_tokens(),
        SchedulerConfig::default(),
    )
    .unwrap();
    scheduler.set_trial(trial);
    let failure = scheduler.failure_handle();
    let (tx, rx) = crossbeam_channel::unbounded();
    let thread = thread::spawn(move || scheduler.run(rx));
    let handle = CommandSender::new(tx);

    let mut events = handle.submit(request).unwrap();
    let mut tokens = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(10);
    while tokens.len() < 4 && Instant::now() < deadline {
        match events.try_recv() {
            Ok(Event::TextToken { id, .. }) => tokens.push(id),
            Ok(_) => {}
            Err(_) => thread::sleep(Duration::from_millis(1)),
        }
    }
    assert_eq!(tokens.len(), 4, "the request generates before the fault");
    *faults.lock().unwrap() = Some(fault);

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
        refused_batches: refused_batches.load(std::sync::atomic::Ordering::SeqCst),
    }
}

/// A world on trial is not repaired: a lost worker stops its scheduler,
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
    assert!(failure.contains("worker process exited"), "{failure}");
}

/// An image-generating request leaves as migratable for a loss when the
/// workers of a world on trial are lost, whatever it was computing, as a
/// text request does: the daemon submits it again, with its tokens and
/// images, to the previous world.
#[test]
fn a_scheduler_on_trial_hands_its_image_requests_back_when_its_workers_are_lost() {
    let mut image_request = request(1, Vec::new(), SamplingParams::default(), 100_000);
    image_request.constraint = GenerationConstraint::Default;
    image_request.max_images = 3;
    image_request.image = ImageParams {
        steps: 6,
        height: 32,
        width: 32,
        ..ImageParams::default()
    };
    image_request.image_generation = ImageGenerationConfig {
        trigger: engine::ImageTrigger::Token {
            token_id: engine::sim_text_token(RequestId(1), 2),
        },
        requires_text_for_image: false,
        feedback_source: Some(engine::FeedbackSource::DeviceProduct),
        feedback_encoders: vec![engine::ImageEncoderInput {
            encoder: engine::ImageEncoder::Vision,
            num_cache_tokens: None,
            max_cache_tokens: Some(16),
            height: None,
            width: None,
        }],
        num_feedback_positions: 2,
        feedback_next_token: engine::FeedbackNextToken::EndOfImage,
        sample_feedback_continuation: true,
    };
    let Faulted {
        tokens,
        terminal,
        stopped,
        ..
    } = fail_while_generating(true, Fault::Lost, image_request);
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
}

/// Non-finite logits fail a world on trial as a whole: the scheduler stops
/// and hands the request back as migratable, naming the batch and its
/// requests as its first error. Outside a trial they fail the request
/// alone, as any failed call does, and the scheduler serves on.
#[test]
fn non_finite_logits_fail_a_world_on_trial_and_a_request_otherwise() {
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
/// the calls of it already in flight then fail for want of the refused
/// calls' products, and that failure is the refusal's, not a worker
/// fault, so the world on trial serves on.
#[test]
fn a_refused_request_fails_alone_on_trial_with_its_calls_in_flight() {
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
    assert_eq!(
        refused_batches, 2,
        "the refusal and the failure of the batch after it"
    );
    assert!(!stopped, "a refusal does not stop a scheduler on trial");
    assert!(failure.is_none(), "{failure:?}");
    assert!(
        matches!(
            terminal,
            Some(Event::Error { .. } | Event::Rejected { .. })
        ),
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
    let mut sim = SimEngine::new();
    sim.set_num_units(16);
    sim.set_block_size(16);
    let scheduler = support::scheduler_with(
        Box::new(SimExecutor::new(sim)),
        SimEngine::special_tokens(),
        SchedulerConfig::default(),
    )
    .unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let thread = thread::spawn(move || {
        let _ = scheduler.run(rx);
    });
    let handle = CommandSender::new(tx);

    let parked = engine::ParkedCache {
        pages: Vec::new(),
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
    let reserving = engine::ParkedCache {
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
