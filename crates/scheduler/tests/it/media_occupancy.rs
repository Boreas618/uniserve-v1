//! Media occupancy scheduling and call graphs of video requests, over the
//! production submission door with `testkit::FakeBatches` as the worker.
//!
//! The observation point is the worker's batch interface: the media batches
//! the fake accepts, each with the batches still on the device when it was
//! accepted. A pacer completes one held batch at a time, so batches the
//! scheduler submits together stay in flight together, and what ran at the
//! same time is a function of the scheduler's occupancy rules.

use crate::support;
use crate::support::fake::Pacer;
use crate::support::video::{
    MAX_CONDITION_ROWS, MAX_RASTER, PROMPT, STEPS, video_fake, video_pool, video_request,
};

use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use ipc::MediaCall;
use ipc::execution::{Batches, Inventory};
use ipc::{DiffusionSamplingParams, RequestId};
use proto::video::Canvas;
use scheduler::{CommandSender, SchedulerConfig};
use scheduler::{ConditionExtent, DiffusionRequest, Event, FinishReason, RejectionKind, Request};
use testkit::Media;

/// How long a fake device holds a batch before the pacer completes it.
const DEVICE_STEP: Duration = Duration::from_millis(2);

/// One media call as the worker received it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ObservedCall {
    request: RequestId,
    call: MediaCall,
    /// Media units the call covers: one for a call that is not measured in
    /// units, and for a decode the units from its first to the next round's.
    units: u32,
    /// The first timeline unit a unit-measured call covers, as the mechanism
    /// staged it; zero for every other call.
    first: u32,
}

impl ObservedCall {
    fn is(&self, request: RequestId, call: MediaCall) -> bool {
        self.request == request && self.call == call
    }

    /// Whether the call runs on a rank's bounded host worker.
    fn is_host_task(&self) -> bool {
        matches!(
            self.call,
            MediaCall::VideoEncoding | MediaCall::AudioEncoding | MediaCall::Muxing
        )
    }

    /// Whether the call decodes or encodes media after denoising.
    fn is_media_output(&self) -> bool {
        self.is_host_task()
            || matches!(
                self.call,
                MediaCall::VideoDecoding | MediaCall::AudioDecoding
            )
    }
}

/// One submission as the worker accepted it, with the calls of earlier
/// submissions that were still in flight at that moment.
#[derive(Debug, Clone)]
struct Submission {
    calls: Vec<ObservedCall>,
    in_flight: Vec<ObservedCall>,
}

impl Submission {
    /// Calls that ran at the same time as this submission's calls: those it
    /// carried and those still in flight when it was accepted.
    fn concurrent(&self) -> impl Iterator<Item = &ObservedCall> {
        self.calls.iter().chain(self.in_flight.iter())
    }
}

/// The record of one served workload: what the worker saw and what each
/// request's stream delivered.
struct Served {
    /// Media submissions in the order the worker accepted them.
    submissions: Vec<Submission>,
    /// Every media call, in the order the worker accepted them.
    calls: Vec<ObservedCall>,
    /// Every event each request's stream delivered.
    ended: HashMap<RequestId, Vec<Event>>,
}

impl Served {
    /// Submissions carrying a call of this request and media call, with their index.
    fn submissions_of(
        &self,
        request: RequestId,
        media_call: MediaCall,
    ) -> Vec<(usize, &Submission)> {
        self.submissions
            .iter()
            .enumerate()
            .filter(|(_, submission)| {
                submission
                    .calls
                    .iter()
                    .any(|call| call.is(request, media_call))
            })
            .collect()
    }

    /// Indices into [`Served::calls`] of a request's calls of one kind.
    fn call_indices(&self, request: RequestId, media_call: MediaCall) -> Vec<usize> {
        self.calls
            .iter()
            .enumerate()
            .filter(|(_, call)| call.is(request, media_call))
            .map(|(index, _)| index)
            .collect()
    }

    /// Asserts that a request delivered a video artifact and then finished.
    fn assert_completed(&self, request: RequestId) {
        let events = &self.ended[&request];
        let artifact = events.iter().find_map(|event| match event {
            Event::Artifact(artifact) => Some(artifact),
            _ => None,
        });
        let artifact = artifact
            .unwrap_or_else(|| panic!("request {request:?} delivered no artifact: {events:?}"));
        assert_eq!(artifact.content_type, "video/mp4");
        assert!(!artifact.media.bytes().is_empty());
        assert!(
            matches!(
                events.last(),
                Some(Event::Finished {
                    reason: FinishReason::Completed,
                    ..
                })
            ),
            "request {request:?} did not finish after its artifact: {events:?}"
        );
    }
}

/// A consumer that reads nothing until the request has already finished still
/// receives the artifact followed by the completed terminal event. Events that
/// do not fit the stream's bounded channel wait in the scheduler's per-request
/// output journal until the consumer drains the channel.
#[test]
fn slow_consumer_receives_the_completed_video_before_the_terminal_event() {
    let media = Arc::new(Media::new());
    let fake = video_fake(vec![video_pool(1, 1, 1)], &media);
    let scheduler = support::fake::scheduler_over(
        Arc::clone(&fake) as Arc<dyn Batches>,
        SchedulerConfig {
            max_batch: proto::config::positive(32),
            ..Default::default()
        },
        Default::default(),
    )
    .unwrap();
    let (tx, commands) = crossbeam_channel::unbounded();
    let handle = CommandSender::new(tx);
    let mut stream = handle
        .submit(Request::Diffusion(DiffusionRequest {
            request_id: RequestId(90),
            prompt_token_ids: PROMPT.to_vec(),
            conditioning: b"{}".to_vec(),
            conditions: Vec::new(),
            priority: 0,
            sampling: DiffusionSamplingParams {
                num_frames: 128,
                video_units: 32,
                seed: 1,
                width: 24,
                height: 16,
                condition_rows: 0,
            },
            trajectory: Default::default(),
            encoder_inputs: None,
        }))
        .unwrap();
    let engine = thread::spawn(move || scheduler.run(commands));

    // Let the entire workload finish while the caller retains an unread
    // stream: the final muxing call, the one that hands no encoded round,
    // completes the artifact and with it the request.
    let deadline = Instant::now() + Duration::from_secs(10);
    let finished_muxing = || {
        fake.submitted().iter().any(|batch| {
            batch.media.as_ref().is_some_and(|media| {
                media.computation.call == MediaCall::Muxing
                    && (0..media.inputs.len())
                        .any(|row| media.request(row).is_ok_and(|staged| staged.rounds == 0))
            })
        })
    };
    while !finished_muxing() {
        assert!(
            Instant::now() < deadline,
            "the request never reached its final muxing"
        );
        thread::sleep(Duration::from_millis(1));
    }

    // Only now drain the stream, until its terminal event. The engine keeps
    // running meanwhile, so it can flush journaled events into the channel as
    // reads free room.
    let mut events = Vec::new();
    while Instant::now() < deadline {
        if let Ok(event) = stream.try_recv() {
            let finished = matches!(event, Event::Finished { .. });
            events.push(event);
            if finished {
                break;
            }
        } else {
            thread::sleep(Duration::from_millis(1));
        }
    }
    handle.shutdown();
    // `Scheduler::run` returns `true` only when the engine died rather than
    // shut down.
    assert!(!engine.join().unwrap());

    Served {
        submissions: Vec::new(),
        calls: Vec::new(),
        ended: HashMap::from([(RequestId(90), events)]),
    }
    .assert_completed(RequestId(90));
}

/// Serves the requests, queued together in order, until each stream reaches
/// a terminal event.
fn serve(pool: Inventory, requests: Vec<Request>) -> Served {
    serve_with(pool, Media::new(), requests, None)
}

/// Serves the requests over `pool` whose media calls `media` responses, with an
/// optional bound on waiting request state.
fn serve_with(
    pool: Inventory,
    media: Media,
    requests: Vec<Request>,
    max_queued_requests: Option<usize>,
) -> Served {
    let units: HashMap<RequestId, u32> = requests
        .iter()
        .filter_map(|request| match request {
            Request::Diffusion(request) => Some((request.request_id, request.sampling.video_units)),
            _ => None,
        })
        .collect();
    let fake = video_fake(vec![pool], &Arc::new(media));
    let scheduler = support::fake::scheduler_over(
        Arc::clone(&fake) as Arc<dyn Batches>,
        SchedulerConfig {
            max_batch: proto::config::positive(32),
            max_queued_requests: max_queued_requests.and_then(std::num::NonZeroUsize::new),
            ..Default::default()
        },
        Default::default(),
    )
    .unwrap();
    let (tx, rx) = crossbeam_channel::unbounded();
    let handle = CommandSender::new(tx);

    // Every request is queued before the loop starts, so one admission pass
    // sees them together, running in submission order.
    let mut streams = requests
        .into_iter()
        .map(|request| {
            let id = request.request_id();
            (id, handle.submit(request).expect("submit media request"))
        })
        .collect::<Vec<_>>();
    let pacer = Pacer::start(&fake, DEVICE_STEP);
    let engine_loop = thread::spawn(move || scheduler.run(rx));

    let mut ended: HashMap<RequestId, Vec<Event>> = HashMap::new();
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut terminal = 0;
    while terminal < streams.len() && Instant::now() < deadline {
        let mut idle = true;
        for (id, stream) in streams.iter_mut() {
            while let Ok(event) = stream.try_recv() {
                idle = false;
                if event.is_terminal() {
                    terminal += 1;
                }
                ended.entry(*id).or_default().push(event);
            }
        }
        if idle {
            thread::sleep(Duration::from_millis(1));
        }
    }
    handle.shutdown();
    let died = engine_loop.join().expect("engine loop thread");
    drop(pacer);
    assert!(!died, "the engine loop died");
    assert_eq!(
        terminal,
        streams.len(),
        "every request reaches a terminal event: {ended:?}"
    );

    record(&fake.accepted(), &units, ended)
}

/// The media submissions of `accepted`, each with the calls of the earlier
/// ones still in flight when it was accepted, and the requests' ended.
///
/// A decode covers the units from its round's first unit to the next round's
/// first, and the last round the request's remaining units.
fn record(
    accepted: &[testkit::Accepted],
    units: &HashMap<RequestId, u32>,
    ended: HashMap<RequestId, Vec<Event>>,
) -> Served {
    // Each media batch's calls, with each decode's first unit.
    let mut batches = Vec::new();
    for accepted in accepted {
        let Some(media) = accepted.batch.media.as_ref() else {
            continue;
        };
        let calls = (0..media.inputs.len())
            .map(|row| {
                let staged = media.request(row).expect("a staged media row");
                (
                    staged.request.request_id,
                    media.computation.call,
                    staged.unit,
                )
            })
            .collect::<Vec<_>>();
        batches.push((accepted.batch.id, accepted.in_flight.clone(), calls));
    }

    // A decode round ends where the request's next round starts.
    let mut next_round: HashMap<RequestId, u32> = HashMap::new();
    let mut covered: Vec<Vec<u32>> = vec![Vec::new(); batches.len()];
    for (index, (_, _, calls)) in batches.iter().enumerate().rev() {
        covered[index] = calls
            .iter()
            .rev()
            .map(|(request, call, unit)| {
                if *call != MediaCall::VideoDecoding {
                    return 1;
                }
                let end = next_round
                    .insert(*request, *unit)
                    .unwrap_or_else(|| units.get(request).copied().unwrap_or(*unit + 1));
                end - unit
            })
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
    }

    let mut observed: HashMap<ipc::execution::BatchId, Vec<ObservedCall>> = HashMap::new();
    let mut submissions = Vec::new();
    let mut calls = Vec::new();
    for ((id, in_flight, batch), covered) in batches.into_iter().zip(covered) {
        let batch_calls = batch
            .into_iter()
            .zip(covered)
            .map(|((request, call, first), units)| ObservedCall {
                request,
                call,
                units,
                first,
            })
            .collect::<Vec<_>>();
        submissions.push(Submission {
            calls: batch_calls.clone(),
            in_flight: in_flight
                .iter()
                .filter_map(|earlier| observed.get(earlier))
                .flatten()
                .copied()
                .collect(),
        });
        calls.extend(batch_calls.iter().copied());
        observed.insert(id, batch_calls);
    }

    Served {
        submissions,
        calls,
        ended,
    }
}

#[test]
fn a_younger_request_denoises_only_after_the_older_request_completes_its_steps() {
    let older = RequestId(1);
    let younger = RequestId(2);
    let served = serve(
        video_pool(2, 1, 1),
        vec![video_request(older.0, 2), video_request(younger.0, 2)],
    );
    served.assert_completed(older);
    served.assert_completed(younger);

    let younger_steps = served.submissions_of(younger, MediaCall::Denoising);
    assert_eq!(younger_steps.len(), STEPS as usize);
    for (index, submission) in &younger_steps {
        // The denoiser is held exclusively: no step of the older request is
        // in flight, or dispatched alongside, while the younger request steps.
        assert!(
            !submission
                .concurrent()
                .any(|call| call.is(older, MediaCall::Denoising)),
            "the younger request denoised while the older request held the denoiser: {submission:?}"
        );
        // The denoiser passes to the younger request only once the older one
        // has dispatched every step, so its first step follows the older's
        // last.
        let older_steps_dispatched = served.submissions[..*index]
            .iter()
            .flat_map(|earlier| earlier.calls.iter())
            .filter(|call| call.is(older, MediaCall::Denoising))
            .count();
        assert_eq!(
            older_steps_dispatched, STEPS as usize,
            "the younger request denoised before the older request dispatched every step"
        );
    }

    // Different stages overlap: the older request decodes and encodes
    // its media while the younger request denoises.
    assert!(
        younger_steps.iter().any(|(_, submission)| {
            submission
                .in_flight
                .iter()
                .any(|call| call.request == older && call.is_media_output())
        }),
        "the older request's media output never overlapped the younger request's denoising: {:?}",
        served.submissions
    );
}

#[test]
fn a_distributed_decoder_admits_calls_up_to_its_rank_width_in_media_units() {
    // Each decoder rank reconstructs one media unit per round, so the decoder
    // admits as many units at once as it has ranks.
    const DECODER_RANKS: u32 = 2;

    let older = RequestId(1);
    let younger = RequestId(2);
    let served = serve(
        video_pool(DECODER_RANKS as usize, 1, 1),
        vec![
            video_request(older.0, DECODER_RANKS + 1),
            video_request(younger.0, 1),
        ],
    );
    served.assert_completed(older);
    served.assert_completed(younger);

    // A request with more units than the decoder admits at once decodes in
    // successive rounds: a full round, then the remainder.
    let older_rounds = served
        .calls
        .iter()
        .filter(|call| call.is(older, MediaCall::VideoDecoding))
        .map(|call| call.units)
        .collect::<Vec<_>>();
    assert_eq!(older_rounds, vec![DECODER_RANKS, 1]);

    // The units in flight on the decoder never exceed what it admits.
    for submission in &served.submissions {
        let units = |calls: &[ObservedCall]| {
            calls
                .iter()
                .filter(|call| call.call == MediaCall::VideoDecoding)
                .map(|call| call.units)
                .sum::<u32>()
        };
        let submitted = units(&submission.calls);
        if submitted == 0 {
            continue;
        }
        assert!(
            units(&submission.in_flight) + submitted <= DECODER_RANKS,
            "decode calls exceeded the decoder's media units: {submission:?}"
        );
    }

    // The younger request's decode is admitted while the older request's is
    // in flight, once the older request's remainder round left units free.
    let younger_decodes = served.submissions_of(younger, MediaCall::VideoDecoding);
    assert_eq!(younger_decodes.len(), 1);
    assert!(
        younger_decodes[0]
            .1
            .concurrent()
            .any(|call| call.is(older, MediaCall::VideoDecoding)),
        "the younger request's decode waited for the older request to finish decoding: {:?}",
        served.submissions
    );
}

#[test]
fn a_host_task_pool_admits_no_more_tasks_than_its_worker_advertises() {
    for capacity in [1_usize, 2] {
        let request = RequestId(1);
        let served = serve(
            video_pool(1, capacity as u32, 1),
            vec![video_request(request.0, 1)],
        );
        served.assert_completed(request);

        // Every host task of this worker lands on rank 0's host task pool,
        // whose occupancy never exceeds the advertised capacity. With one rank
        // and one media unit, each host call places exactly one task, so
        // counting calls counts the pool's tasks.
        let mut peak = 0;
        for submission in &served.submissions {
            let submitted = submission
                .calls
                .iter()
                .filter(|call| call.is_host_task())
                .count();
            if submitted == 0 {
                continue;
            }
            let occupied = submission
                .in_flight
                .iter()
                .filter(|call| call.is_host_task())
                .count();
            assert!(
                occupied + submitted <= capacity,
                "host tasks exceeded the pool capacity {capacity}: {submission:?}"
            );
            peak = peak.max(occupied + submitted);
        }

        // The video and audio encodes of one request are the tasks that can
        // overlap: a pool of two holds both, a pool of one serializes them.
        assert_eq!(
            peak,
            capacity.min(2),
            "host task capacity {capacity} was not used up to its bound: {:?}",
            served.submissions
        );
    }
}

#[test]
fn an_admitted_video_request_reports_scheduling_before_its_progress() {
    let request = RequestId(1);
    let served = serve(video_pool(2, 1, 1), vec![video_request(request.0, 2)]);
    served.assert_completed(request);

    // A job's public phases start at admission: the scheduling event carries
    // the queue interval and precedes every progress report.
    let events = &served.ended[&request];
    let scheduled = events
        .iter()
        .position(|event| matches!(event, Event::Scheduled { .. }))
        .unwrap_or_else(|| panic!("the request was never reported as scheduled: {events:?}"));
    let first_progress = events
        .iter()
        .position(|event| matches!(event, Event::MediaProgress { .. }))
        .expect("the request reported progress");
    assert!(scheduled < first_progress, "{events:?}");
    let Event::Scheduled {
        queued_at,
        scheduled_at,
    } = events[scheduled]
    else {
        unreachable!("position matched a scheduling event");
    };
    assert!(queued_at <= scheduled_at);
}

#[test]
fn a_full_waiting_queue_rejects_a_video_request_as_overloaded() {
    let admitted = RequestId(1);
    let refused = RequestId(2);
    let served = serve_with(
        video_pool(2, 1, 1),
        Media::new(),
        vec![video_request(admitted.0, 2), video_request(refused.0, 2)],
        Some(1),
    );
    served.assert_completed(admitted);

    // The second request arrives while the first still waits, so the bound
    // refuses it as retryable overload rather than as an invalid request.
    let events = &served.ended[&refused];
    assert!(
        matches!(
            events.as_slice(),
            [Event::Rejected {
                kind: RejectionKind::Overloaded,
                ..
            }]
        ),
        "{events:?}"
    );
}

/// The resolution of the fixture's conditions.
const CONDITION_RESOLUTION: Canvas = Canvas {
    width: 64,
    height: 32,
};

/// A `ref2va` request with an image reference, a three-unit video reference
/// with its soundtrack and an audio reference.
fn reference_request(id: u64) -> Request {
    let pixels = |frames: u64| {
        frames * u64::from(CONDITION_RESOLUTION.width * CONDITION_RESOLUTION.height) * 3
    };
    let conditions = vec![
        // The image, one frame the conditioner reads as one 2x4 grid.
        ConditionExtent {
            pixel_bytes: pixels(1),
            vision_patches: 8,
            vision_tokens: 2,
            latent_units: vec![2],
            ..ConditionExtent::default()
        },
        // A 40-frame clip whose 39 VAE frames encode in three units, read by
        // the conditioner as a 2x2x4 grid, with a 16000-sample soundtrack.
        ConditionExtent {
            pixel_bytes: pixels(39),
            audio_samples: 16_000,
            vision_patches: 16,
            vision_tokens: 4,
            latent_units: vec![10, 10, 4],
            audio_rows: 40,
        },
        // The same track as an audio reference.
        ConditionExtent {
            audio_samples: 16_000,
            audio_rows: 40,
            ..ConditionExtent::default()
        },
    ];
    // Labels, then one image block and two video blocks with their markers,
    // then the prompt.
    let prompt_tokens = 3 + 4 + 2 + 8 + 3;
    Request::Diffusion(DiffusionRequest {
        request_id: RequestId(id),
        prompt_token_ids: (0..prompt_tokens).collect(),
        conditioning: b"{}".to_vec(),
        conditions,
        priority: 0,
        sampling: DiffusionSamplingParams {
            num_frames: 16,
            video_units: 2,
            seed: id,
            width: 24,
            height: 16,
            condition_rows: 106,
        },
        trajectory: Default::default(),
        encoder_inputs: None,
    })
}

/// A reference request reads its media first, then encodes its vision blocks
/// before the text that splices them in, encodes its visual condition units
/// in rounds and its audio tracks in one further call, and prepares its
/// latents once every encoding completed.
#[test]
fn a_reference_request_reads_encodes_and_prepares_its_conditions() {
    let request = RequestId(5);
    let served = serve(video_pool(2, 2, 1), vec![reference_request(5)]);
    served.assert_completed(request);

    // The media read is the request's first call, alone in its batch.
    assert_eq!(served.call_indices(request, MediaCall::MediaReading), [0]);
    assert_eq!(served.submissions[0].calls.len(), 1);

    // Everything reading the host products waits for the read to complete:
    // no call reading them is submitted while the read is in flight.
    for media_call in [MediaCall::VisionEncoding, MediaCall::LatentEncoding] {
        for (_, submission) in served.submissions_of(request, media_call) {
            assert!(
                !submission
                    .in_flight
                    .iter()
                    .any(|call| call.is(request, MediaCall::MediaReading)),
                "{media_call:?} was submitted while the media read was in flight"
            );
        }
    }
    let vision = served.call_indices(request, MediaCall::VisionEncoding);
    let text = served.call_indices(request, MediaCall::TextEncoding);
    let latents = served.call_indices(request, MediaCall::LatentEncoding);
    assert_eq!(vision.len(), 1);
    assert_eq!(text.len(), 1);
    assert!(
        vision[0] < text[0],
        "the vision tokens precede the text encoding"
    );
    // The text encoding is submitted once its vision features exist, so it
    // never waits in its worker's queue ahead of other calls.
    let (text_submission, text_batch) = served.submissions_of(request, MediaCall::TextEncoding)[0];
    assert!(
        !text_batch
            .in_flight
            .iter()
            .any(|call| call.is(request, MediaCall::VisionEncoding)),
        "the text encoding was submitted while the vision encoding was in flight"
    );

    // The four visual units are encoded in rounds that cover them in order,
    // each on the latent encoder's free ranks, and the audio tracks in one
    // further call that starts at unit zero again; the first round starts
    // beside the vision encoding, before the text encoding.
    const VISUAL_UNITS: u32 = 4;
    let (audio, rounds) = latents.split_last().unwrap();
    let firsts = rounds
        .iter()
        .map(|index| served.calls[*index].first)
        .collect::<Vec<_>>();
    assert_eq!(firsts.first(), Some(&0), "{firsts:?}");
    assert!(
        firsts.windows(2).all(|pair| pair[0] < pair[1])
            && firsts.iter().all(|first| *first < VISUAL_UNITS),
        "the rounds cover the visual units in order: {firsts:?}"
    );
    assert!(
        (2..=VISUAL_UNITS as usize).contains(&rounds.len()),
        "two ranks encode four units in two to four rounds: {firsts:?}"
    );
    assert_eq!(served.calls[*audio].first, 0);
    let dealt = served.submissions_of(request, MediaCall::LatentEncoding);
    assert!(
        dealt[0].0 < text_submission,
        "the first round waited for the text encoding"
    );

    // Latent preparation follows every encoding.
    let preparation = served.call_indices(request, MediaCall::LatentPreparation);
    assert_eq!(preparation.len(), 1);
    assert!(
        latents
            .iter()
            .chain(&text)
            .all(|index| *index < preparation[0])
    );
}

/// A conditioned request is refused when the deployment's denoiser takes no
/// conditions, and its condition rows are refused, naming the product and
/// both sizes, beyond what the latent encoder declares.
#[test]
fn conditioned_requests_beyond_the_deployment_are_refused() {
    let rejected = |served: &Served, id: u64| match served.ended[&RequestId(id)].as_slice() {
        [
            Event::Rejected {
                kind: RejectionKind::Invalid,
                message,
            },
        ] => message.clone(),
        events => panic!("the request was not refused: {events:?}"),
    };

    let mut pool = video_pool(2, 2, 1);
    for rank in &mut pool.ranks {
        if let Some(denoiser) = rank.video_denoiser.as_mut() {
            denoiser.takes_conditions = false;
        }
    }
    let served = serve(pool, vec![reference_request(1)]);
    let message = rejected(&served, 1);
    assert!(message.contains("takes no conditions"), "{message}");

    let Request::Diffusion(mut oversized) = reference_request(2) else {
        unreachable!("the fixture is a video request");
    };
    oversized.conditions[1].latent_units = vec![MAX_CONDITION_ROWS; 3];
    let served = serve(video_pool(2, 2, 1), vec![Request::Diffusion(oversized)]);
    let message = rejected(&served, 2);
    assert!(
        message.contains("condition_video_latents")
            && message.contains(&MAX_CONDITION_ROWS.to_string())
            && message.contains(&(3 * MAX_CONDITION_ROWS + 2).to_string()),
        "{message}"
    );
}

/// A rank that refuses a request's size states why: the request is rejected
/// as invalid with that reason, and a request admitted beside it still
/// completes.
#[test]
fn a_request_the_worker_refuses_is_rejected_with_its_reason() {
    let media = Media::new();
    let reason = "the conditions take 46848 rows, more than the 40960 this worker provisions";
    media.refuse(RequestId(1), reason);
    let served = serve_with(
        video_pool(2, 2, 1),
        media,
        vec![reference_request(1), video_request(2, 2)],
        None,
    );

    let refused = served.ended[&RequestId(1)]
        .iter()
        .find_map(|event| match event {
            Event::Rejected { kind, message } => Some((*kind, message.clone())),
            _ => None,
        });
    assert_eq!(refused, Some((RejectionKind::Invalid, reason.to_owned())));
    served.assert_completed(RequestId(2));
}

/// A `t2va` request of two media units at `resolution`.
fn request_at(id: u64, resolution: Canvas) -> Request {
    let Request::Diffusion(mut request) = video_request(id, 2) else {
        unreachable!("the fixture is a video request");
    };
    request.sampling.width = resolution.width;
    request.sampling.height = resolution.height;
    Request::Diffusion(request)
}

/// Each video request is served at its own resolution when the decoder's
/// declared raster holds it, and a resolution beyond that raster is refused,
/// naming the product and both extents.
#[test]
fn decoded_media_units_are_reserved_at_each_request_resolution() {
    let wide = Canvas {
        width: 24,
        height: 16,
    };
    let tall = Canvas {
        width: 16,
        height: 32,
    };
    let oversized = Canvas {
        width: 48,
        height: 16,
    };
    let served = serve(
        video_pool(2, 2, 1),
        vec![
            request_at(1, wide),
            request_at(2, tall),
            request_at(3, oversized),
        ],
    );

    for id in [1, 2] {
        let request = RequestId(id);
        served.assert_completed(request);
        assert!(
            !served
                .call_indices(request, MediaCall::VideoDecoding)
                .is_empty()
        );
    }

    match served.ended[&RequestId(3)].as_slice() {
        [
            Event::Rejected {
                kind: RejectionKind::Invalid,
                message,
            },
        ] => assert!(
            message.contains("video_units")
                && message.contains(&MAX_RASTER.width.to_string())
                && message.contains(&oversized.width.to_string()),
            "{message}"
        ),
        events => panic!("the oversized resolution was not refused: {events:?}"),
    }
}
