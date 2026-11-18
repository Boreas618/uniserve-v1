//! Video preemption (SD §8.1): a video request preempted at a step boundary and resumed from its
//! stored step (SD §3.8, §7.3 "Preemption", "Resume"). A drain ends a running video request at a
//! step boundary unless every step of it is scheduled, which runs it to its end; one with a
//! trajectory identity and a completed step stores its trajectory with `Save` once its calls in
//! flight returned, sends its terminal with the stored step at `Saved` and its `Detach` after it,
//! and leaves with no stored step at any other answer. The daemon resubmits it naming that step:
//! it is admitted ahead of the requests that start, its admission's `Restore` brings the stored
//! trajectory back under its key, and it denoises only the steps that remain; a trajectory no
//! epoch stored under its key, or a request naming every step, starts from noise.
//!
//! Every step of every case checks the scheduler's side of A1 to A4.

use std::time::{Duration, Instant};

use engine::batch::Phase;
use engine::scheduler::Message;
use proto::request::{
    DiffusionRequest, Event, FinishReason, MigrationCause, RequestId, Trajectory,
};
use storage::Key;
use storage::key::latent_content;
use storage::runtime::{Cause, Completion, Request};

use crate::harness::{A, Deployment, Harness, LATENT_SEED, STEPS, pipeline, video};

/// The trajectory identity the daemon gives the cases' video requests.
const TRAJECTORY: u64 = 7;

/// The latent seed of the harness's epoch: the one its denoising group states for its weights
/// from the start (SD §2.1 `Capacity`, §4.3).
const SEED: u64 = LATENT_SEED;

/// The host latent units of a stored trajectory: the harness's captured shape states 30 000 bytes
/// for a slot's trajectory, and its latent region's units are 4096 bytes.
const TRAJECTORY_UNITS: u32 = 30_000_u32.div_ceil(4096);

/// Video request `id` of one media unit with the cases' trajectory identity.
fn identified(id: u64) -> DiffusionRequest {
    let mut request = video(id, 1);
    request.trajectory.id = std::num::NonZeroU64::new(TRAJECTORY);
    request
}

/// `request` as the daemon resubmits it after it left with `steps` stored steps.
fn resumed(request: &DiffusionRequest, steps: u32, state_seed: Option<u64>) -> DiffusionRequest {
    DiffusionRequest {
        trajectory: Trajectory {
            id: std::num::NonZeroU64::new(TRAJECTORY),
            denoised_steps: steps,
            state_seed,
            stored_layout: Vec::new(),
        },
        ..request.clone()
    }
}

/// The requests the buffer pool took, each with its number, in the order taken.
fn numbered(harness: &Harness) -> Vec<(u64, Request)> {
    let taken = harness.pool.requests().iter();
    taken
        .map(|(_, number, request)| (number.0, request.clone()))
        .collect()
}

/// The completion the buffer pool wrote for the request it took under `number`.
fn answer_to(harness: &Harness, number: u64) -> Option<Completion> {
    let mut written = harness.pool.completions().iter();
    written
        .find(|(_, answered, _)| answered.0 == number)
        .map(|(_, _, completion)| completion.clone())
}

/// The number of the `Finish` that ended the copy answering the request taken under `copy`, with
/// whether it ran and its answer.
fn finish_of(harness: &Harness, copy: u64) -> Option<(u64, bool, Completion)> {
    numbered(harness)
        .into_iter()
        .find_map(|(number, request)| match request {
            Request::Finish { copy: of, ok, .. } if of.0 == copy => {
                Some((number, ok, answer_to(harness, number)?))
            }
            _ => None,
        })
}

/// Steps until request `id` has ended, the drain's reply included; at most `steps` steps.
fn ended(harness: &mut Harness, id: RequestId, steps: u64) {
    harness.run_until(steps, |harness| harness.terminal(id).is_some());
}

/// Steps until the buffer pool took a `Detach`: a request's `Detach` reaches it once the
/// workers stamped it.
fn detached(harness: &mut Harness) {
    harness.run_until(20, |harness| {
        let taken = harness.taken();
        taken
            .iter()
            .any(|request| matches!(request, Request::Detach { .. }))
    });
}

/// Request `id`'s terminal event, which must be a `Migratable` of a drain: the stored step it
/// names.
fn stored_step(harness: &Harness, id: RequestId) -> u32 {
    let Some(Event::Migratable {
        cause: MigrationCause::Drain,
        denoised_steps,
        stored_layout,
        lend: None,
    }) = harness.terminal(id)
    else {
        panic!(
            "request {} did not leave for the drain: {:?}",
            id.0,
            harness.events(id)
        );
    };
    assert!(
        stored_layout.is_empty(),
        "the key alone finds the trajectory"
    );
    denoised_steps
}

/// Request `id` completed: its artifact, then `Finished` with `Completed`.
fn completed(harness: &Harness, id: RequestId) {
    let events = harness.events(id);
    let [.., Event::Artifact(_), Event::Finished { reason, .. }] = &events[..] else {
        panic!("request {} did not complete: {events:?}", id.0);
    };
    assert_eq!(*reason, FinishReason::Completed);
}

/// Submits `request`, lets it complete one denoising step with its next on the device, and drains
/// the epoch at once; returns the request and the drain's reply.
fn drained_mid_denoising(
    harness: &mut Harness,
    request: DiffusionRequest,
) -> (RequestId, crossbeam_channel::Receiver<()>) {
    let id = harness.submit_video(request);
    harness.denoise_and_hold(id, 1);
    assert!(harness.fake.held() > 0, "a denoising step is on the device");
    let replied = harness.drain(Instant::now(), false);
    (id, replied)
}

/// A drain that meets a video request mid-denoising ends it at its step boundary: once its steps
/// in flight returned, its trajectory at the step it completed is stored with `Save` for its owner
/// on the denoiser, under `latent_content(seed, trajectory, steps)` and in the units the host
/// latent region holds it in; the copy runs, and the request's terminal names that step, sent once
/// the `Save` is answered `Saved` and before its `Detach` (SD §3.7 `release`, `saved`, §3.8,
/// §4.3, §4.4 "Classes").
#[test]
fn a_drain_mid_denoising_saves_the_trajectory_and_sends_the_terminal_at_saved_before_its_detach() {
    let mut harness = Deployment::video(&[&pipeline().phases]).build();
    let (id, replied) = drained_mid_denoising(&mut harness, identified(1));
    harness.fake.resume();

    let saved = |harness: &Harness| {
        harness
            .answers()
            .into_iter()
            .any(|answer| matches!(answer, (Request::Finish { .. }, Completion::Saved)))
    };
    for _ in 0..50 {
        if harness.terminal(id).is_some() {
            break;
        }
        harness.step();
        assert!(
            harness.terminal(id).is_none() || saved(&harness),
            "its terminal was sent before its `Saved`"
        );
    }
    harness.run_until(20, |_| replied.try_recv().is_ok());
    detached(&mut harness);

    // Every step it formed returned and was completed: the trajectory is stored at the last.
    let steps = harness.media_rows(id, Phase::Denoise).len() as u32;
    assert!(0 < steps && steps < STEPS, "{steps}");
    assert_eq!(stored_step(&harness, id), steps);

    let owner = harness.owner(id, A).unwrap();
    let (save, request) = numbered(&harness)
        .into_iter()
        .find(|(_, request)| matches!(request, Request::Save { .. }))
        .expect("the trajectory is saved");
    assert_eq!(
        request,
        Request::Save {
            owner,
            key: Key(latent_content(SEED, TRAJECTORY, steps)),
            units: TRAJECTORY_UNITS,
        }
    );
    assert!(matches!(
        answer_to(&harness, save),
        Some(Completion::Copy { .. })
    ));
    let (finish, ran, answer) = finish_of(&harness, save).expect("the copy ended");
    assert!(ran);
    assert_eq!(answer, Completion::Saved);

    let detached = numbered(&harness).into_iter().any(|(number, request)| {
        matches!(request, Request::Detach { owner: of, .. } if of == owner) && number > finish
    });
    assert!(detached, "its `Detach` follows the `Saved`");
}

/// A preempted request whose `Save` copy fails leaves with no stored step, and gives up its
/// blocks as any other (SD §3.7 `saved`, §3.8; §7.3 "a trajectory that cannot be stored leaves
/// with 0 steps").
#[test]
fn a_save_whose_copy_fails_leaves_the_request_with_no_stored_step() {
    let mut harness = Deployment::video(&[&pipeline().phases]).build();
    harness.fake.fail_copies(true);
    let (id, replied) = drained_mid_denoising(&mut harness, identified(1));
    harness.fake.resume();
    ended(&mut harness, id, 50);
    harness.run_until(20, |_| replied.try_recv().is_ok());

    assert_eq!(stored_step(&harness, id), 0);
    let (save, _) = numbered(&harness)
        .into_iter()
        .find(|(_, request)| matches!(request, Request::Save { .. }))
        .expect("the trajectory is saved");
    let (_, ran, answer) = finish_of(&harness, save).expect("the copy ended");
    assert!(!ran);
    assert_eq!(answer, Completion::Done);
    detached(&mut harness);
}

/// The first life of [`a_resumed_request_restores_its_stored_step_and_denoises_only_the_rest`]
/// and its kin: request 1 preempted mid-denoising with its trajectory stored, and admission open
/// again. Returns the request and the step it stored.
fn stored(harness: &mut Harness) -> (DiffusionRequest, u32) {
    let request = identified(1);
    let (id, replied) = drained_mid_denoising(harness, request.clone());
    harness.fake.resume();
    ended(harness, id, 50);
    harness.run_until(20, |_| replied.try_recv().is_ok());
    let steps = stored_step(harness, id);
    assert!(steps > 0);
    harness.apply(Message::Resume);
    (request, steps)
}

/// The resubmitted life of request `id`: its preparation's and its steps' rows, formed after the
/// batch numbered `from`.
fn second_life(harness: &Harness, id: RequestId, from: usize) -> (Vec<u32>, Vec<u32>) {
    let batches = harness.batches().split_off(from).into_iter();
    let mut prepared = Vec::new();
    let mut denoised = Vec::new();
    for batch in batches {
        let Some(row) = batch.requests.iter().position(|of| *of == id) else {
            continue;
        };
        match batch.phase {
            Phase::LatentPrepare => prepared.push(batch.media[row].step),
            Phase::Denoise => denoised.push(batch.media[row].step),
            _ => {}
        }
    }
    (prepared, denoised)
}

/// A resubmitted request naming its stored step restores the trajectory at its admission: its
/// `Restore` names the key it was saved under, for its owner on the denoiser, and the copy is
/// answered `Restored`; its preparation opens the trajectory at that step and it denoises only the
/// steps that remain, then completes (SD §3.8 "Resume", §7.3 "Resume").
#[test]
fn a_resumed_request_restores_its_stored_step_and_denoises_only_the_rest() {
    let mut harness = Deployment::video(&[&pipeline().phases]).build();
    let (request, steps) = stored(&mut harness);
    let id = request.request_id;
    let from = harness.batches().len();
    harness.submit_video(resumed(&request, steps, None));
    harness.run_until(200, |harness| {
        matches!(harness.terminal(id), Some(Event::Finished { .. }))
    });
    completed(&harness, id);

    let owner = harness.owner(id, A).unwrap();
    let (restore, request) = numbered(&harness)
        .into_iter()
        .find(|(_, request)| matches!(request, Request::Restore { .. }))
        .expect("its admission restores the trajectory");
    assert_eq!(
        request,
        Request::Restore {
            owner,
            key: Key(latent_content(SEED, TRAJECTORY, steps)),
        }
    );
    let (_, ran, answer) = finish_of(&harness, restore).expect("the copy ended");
    assert!(ran);
    assert_eq!(answer, Completion::Restored);

    let (prepared, denoised) = second_life(&harness, id, from);
    assert_eq!(prepared, [steps], "its preparation opens the stored step");
    assert_eq!(denoised, (steps..STEPS).collect::<Vec<_>>());
}

/// A resubmitted request that names the latent seed of another epoch's numerics looks for the
/// trajectory under that seed, which no epoch stored it under: its `Restore` is refused for the
/// key, and it starts from noise and denoises every step (SD §3.5 `Refused`, §3.8 "Resume", §7.3
/// "Resume": the request's `state_seed` overrides the seed).
#[test]
fn a_resumed_request_whose_trajectory_is_missing_starts_from_noise() {
    const OTHER: u64 = 9;
    let mut harness = Deployment::video(&[&pipeline().phases]).build();
    let (request, steps) = stored(&mut harness);
    let id = request.request_id;
    let from = harness.batches().len();
    harness.submit_video(resumed(&request, steps, Some(OTHER)));
    harness.run_until(200, |harness| {
        matches!(harness.terminal(id), Some(Event::Finished { .. }))
    });
    completed(&harness, id);

    let refused = harness.answers().into_iter().any(|answer| {
        matches!(
            answer,
            (Request::Restore { key, .. }, Completion::Refused { cause: Cause::Key })
                if key == Key(latent_content(OTHER, TRAJECTORY, steps))
        )
    });
    assert!(refused, "its `Restore` names the seed it states");
    let (prepared, denoised) = second_life(&harness, id, from);
    assert_eq!(prepared, [0], "it starts from noise");
    assert_eq!(denoised, (0..STEPS).collect::<Vec<_>>());
}

/// A request naming as many denoised steps as the deployment's denoising takes names no step it
/// could continue from: its admission restores nothing and it denoises every step (SD §7.3
/// "Resume": restored when `0 < denoised_steps < steps`).
#[test]
fn a_request_naming_every_step_denoised_starts_from_noise_without_a_restore() {
    let mut harness = Deployment::video(&[&pipeline().phases]).build();
    let id = harness.submit_video(resumed(&identified(1), STEPS, None));
    harness.settle(200);
    completed(&harness, id);

    let restored = harness
        .taken()
        .into_iter()
        .any(|request| matches!(request, Request::Restore { .. }));
    assert!(!restored);
    let (prepared, denoised) = second_life(&harness, id, 0);
    assert_eq!(prepared, [0]);
    assert_eq!(denoised, (0..STEPS).collect::<Vec<_>>());
}

/// A request that resumes a stored trajectory is admitted ahead of a request that starts one,
/// whichever arrived first (SD §3.2 "resume first", §7.2 "The queue's order").
#[test]
fn a_resumed_request_is_admitted_before_one_that_starts() {
    let mut harness = Deployment::video(&[&pipeline().phases]).slots(1).build();
    let running = harness.submit_video(video(1, 1));
    harness.run_until(10, |harness| !harness.events(running).is_empty());
    let starting = harness.submit_video(video(2, 1));
    let resuming = harness.submit_video(resumed(&identified(3), 2, None));
    harness.settle(400);

    let scheduled: Vec<RequestId> = harness
        .sent()
        .filter_map(|(id, event)| matches!(event, Event::Scheduled { .. }).then_some(*id))
        .collect();
    assert_eq!(scheduled, [running, resuming, starting]);
    for id in [running, resuming, starting] {
        completed(&harness, id);
    }
}

/// A drain whose deadline meets a video request with every denoising step scheduled lets it run
/// to its end on this engine: it stores nothing and completes, and the drain is answered after it
/// (SD §3.8 `drain_due`, §7.3 "Preemption": a running request is preempted only while its
/// scheduled steps are fewer than its steps).
#[test]
fn a_request_with_every_step_scheduled_runs_to_its_end_through_a_drain() {
    let mut harness = Deployment::video(&[&pipeline().phases]).build();
    let id = harness.submit_video(identified(1));
    harness.fake.hold();
    harness.run_until(60, |harness| {
        harness.fake.release(1);
        harness.media_rows(id, Phase::Denoise).len() == STEPS as usize
    });

    let replied = harness.drain(Instant::now(), false);
    harness.fake.resume();
    harness.run_until(100, |harness| harness.terminal(id).is_some());
    completed(&harness, id);
    harness.run_until(20, |_| replied.try_recv().is_ok());
    let saved = harness
        .taken()
        .into_iter()
        .any(|request| matches!(request, Request::Save { .. }));
    assert!(!saved);
}

/// A video request that waits when a drain begins leaves at once with no stored step, whatever
/// step it names: it holds nothing on this engine (SD §3.8 `drain`, §7.3 "Preemption": a waiting
/// request leaves with 0 steps).
#[test]
fn a_waiting_video_request_leaves_with_no_stored_step() {
    let mut harness = Deployment::video(&[&pipeline().phases]).slots(1).build();
    let running = harness.submit_video(video(1, 1));
    harness.run_until(10, |harness| !harness.events(running).is_empty());
    let waiting = harness.submit_video(resumed(&identified(2), 2, None));
    harness.step();
    assert!(harness.events(waiting).is_empty(), "one slot: it waits");

    let replied = harness.drain(Instant::now() + Duration::from_secs(60), false);
    let events = harness.events(waiting);
    assert_eq!(events.len(), 1, "{events:?}");
    assert_eq!(stored_step(&harness, waiting), 0);

    harness.settle(200);
    completed(&harness, running);
    harness.run_until(20, |_| replied.try_recv().is_ok());
    let touched = harness
        .taken()
        .into_iter()
        .any(|request| matches!(request, Request::Save { .. } | Request::Restore { .. }));
    assert!(
        !touched,
        "nothing of the waiting request is stored or restored"
    );
}
