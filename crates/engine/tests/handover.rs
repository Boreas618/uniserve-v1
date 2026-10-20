//! Control-plane handover between two engine cores over the GPU-free
//! simulator: a request a predecessor quiesces mid-generation finishes on
//! its successor exactly as it would have without the handover.
//!
//! The simulator's greedy token at text index `n` of request `r` is
//! `1000 + (7r + n) % 5000` until `SimEngine::set_text_len` tokens, then its
//! EOS; the cache block size is 64 tokens.

#![cfg(test)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::mpsc::error::TryRecvError;
use uniserve_core::{
    EngineCoreOutput, FinishReason, GenerationConstraint, GenerationRequest, ImageGenerationConfig,
    ImageParams, RequestId, SamplingParams,
};
use uniserve_engine::{EngineConfig, EngineCore, EventRx, SimEngine, SimExecutor};

const TEXT_LEN: usize = 400;

fn core() -> EngineCore {
    let mut sim = SimEngine::new();
    sim.set_text_len(TEXT_LEN);
    EngineCore::with_executor(
        EngineConfig::sim("sim-model"),
        Box::new(SimExecutor::new(sim)),
    )
    .unwrap()
}

fn request(id: u64, prompt: Vec<u32>) -> GenerationRequest {
    GenerationRequest {
        request_id: RequestId(id),
        prompt_token_ids: prompt,
        multimodal_inputs: Default::default(),
        negative_prompt_token_ids: Vec::new(),
        constraint: GenerationConstraint::UndOnly,
        sampling: SamplingParams::default(),
        image: ImageParams::default(),
        max_images: 1,
        max_und_tokens: 2 * TEXT_LEN,
        include_stop_token: false,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: Default::default(),
        image_generation: ImageGenerationConfig::default(),
        readout: Vec::new(),
    }
}

/// What a client read of one request: its token ids, its `Scheduled`
/// events, and its terminal event.
#[derive(Debug, Default, PartialEq)]
struct Reply {
    tokens: Vec<u32>,
    scheduled: usize,
    finished: Option<(FinishReason, usize, usize)>,
}

impl Reply {
    fn take(&mut self, event: EngineCoreOutput) {
        match event {
            EngineCoreOutput::TextToken { id, .. } => self.tokens.push(id),
            EngineCoreOutput::TextTokens { ids } => self.tokens.extend(ids),
            EngineCoreOutput::Scheduled { .. } => self.scheduled += 1,
            EngineCoreOutput::Finished {
                reason,
                prompt_tokens,
                completion_tokens,
                ..
            } => self.finished = Some((reason, prompt_tokens, completion_tokens)),
            other => panic!("unexpected event {other:?}"),
        }
    }

    /// Reads `events` until the terminal event.
    fn read_to_end(&mut self, events: &mut EventRx) {
        let deadline = Instant::now() + Duration::from_secs(20);
        while self.finished.is_none() {
            assert!(Instant::now() < deadline, "the request did not finish");
            match events.try_recv() {
                Ok(event) => self.take(event),
                Err(TryRecvError::Empty) => std::thread::sleep(Duration::from_millis(1)),
                Err(TryRecvError::Disconnected) => panic!("the stream closed unfinished"),
            }
        }
    }
}

#[test]
fn a_request_quiesced_mid_generation_finishes_on_the_successor_unchanged() {
    let prompt: Vec<u32> = (0..150).map(|i| i % 97 + 3).collect();

    let reference = core();
    let mut events = reference.submit(request(1, prompt.clone()).into()).unwrap();
    let mut expected = Reply::default();
    expected.read_to_end(&mut events);
    assert_eq!(expected.tokens.len(), TEXT_LEN);
    reference.shutdown();

    // The client reads a few tokens, then stops: the request stalls once
    // its channel and journal are full, mid-generation.
    let predecessor = core();
    let mut client = predecessor
        .submit(request(1, prompt.clone()).into())
        .unwrap();
    let mut reply = Reply::default();
    while reply.tokens.len() < 10 {
        match client.try_recv() {
            Ok(event) => reply.take(event),
            Err(_) => std::thread::sleep(Duration::from_millis(1)),
        }
    }
    let quiesced = predecessor.quiesce().unwrap();
    let snapshot = quiesced.snapshot.clone();
    assert_eq!(snapshot.requests.len(), 1);
    let generated = snapshot.requests[0].generated.len();
    assert!(
        (10..TEXT_LEN).contains(&generated),
        "{generated} tokens generated"
    );
    assert_eq!(snapshot.requests[0].generated, expected.tokens[..generated]);

    let successor = core();
    let stats = Arc::clone(successor.stats());
    let mut streams = successor.resume(snapshot).unwrap();
    assert_eq!(streams.len(), 1);
    assert_eq!(streams[0].0, RequestId(1));

    // The client reads what the predecessor sent, then what its journal
    // held, then the successor's events, as the relay forwards them.
    loop {
        match client.try_recv() {
            Ok(event) => reply.take(event),
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => panic!("the predecessor closed the stream"),
        }
    }
    let [stream] = <[_; 1]>::try_from(quiesced.streams).ok().unwrap();
    for event in stream.pending {
        reply.take(event);
    }
    assert_eq!(reply.tokens, expected.tokens[..generated]);
    reply.read_to_end(&mut streams[0].1);
    assert_eq!(reply, expected);

    // The successor read every complete page of the prompt and the
    // generated tokens the predecessor computed: all but the last
    // generated token.
    let computed = prompt.len() + generated - 1;
    assert_eq!(
        stats
            .prefix
            .hit_tokens
            .load(std::sync::atomic::Ordering::Relaxed),
        (computed / 64 * 64) as u64
    );
    successor.shutdown();
}

/// A request with stop strings migrates with the stop-string decisions its
/// client has not yet made: the client's acknowledgements, which its
/// predecessor's command channel relays to the successor, release the
/// request there, and it finishes with the tokens it generates alone.
#[test]
fn a_request_awaiting_stop_string_decisions_resumes_with_them() {
    let prompt: Vec<u32> = (0..90).map(|i| i % 61 + 9).collect();
    let mut submitted = request(1, prompt);
    submitted.stop_strings = vec!["never matched".into()];

    // The client acknowledges every token it reads.
    let reference = core();
    let mut events = reference.submit(submitted.clone().into()).unwrap();
    let mut expected = Reply::default();
    let deadline = Instant::now() + Duration::from_secs(20);
    while expected.finished.is_none() {
        assert!(Instant::now() < deadline, "the request did not finish");
        match events.try_recv() {
            Ok(event) => {
                expected.take(event);
                events.acknowledge_consumed_prefix();
            }
            Err(_) => std::thread::sleep(Duration::from_millis(1)),
        }
    }
    reference.shutdown();

    // The client acknowledges its first tokens, then reads without
    // acknowledging: the request stops at its bound of undecided calls.
    let predecessor = core();
    let mut client = predecessor.submit(submitted.into()).unwrap();
    let mut reply = Reply::default();
    while reply.tokens.len() < 10 {
        match client.try_recv() {
            Ok(event) => {
                reply.take(event);
                client.acknowledge_consumed_prefix();
            }
            Err(_) => std::thread::sleep(Duration::from_millis(1)),
        }
    }
    let unacknowledged = Instant::now() + Duration::from_millis(200);
    while Instant::now() < unacknowledged {
        match client.try_recv() {
            Ok(event) => reply.take(event),
            Err(_) => std::thread::sleep(Duration::from_millis(1)),
        }
    }
    let quiesced = predecessor.quiesce().unwrap();
    let migrated = &quiesced.snapshot.requests[0];
    assert!(!migrated.decoder_boundaries.is_empty());
    assert!(migrated.generated.len() < TEXT_LEN);

    let successor = core();
    let mut streams = successor.resume(quiesced.snapshot.clone()).unwrap();
    let [stream] = <[_; 1]>::try_from(quiesced.streams).ok().unwrap();

    // The relay: the successor's events go to the client's channel after
    // the journaled ones, and the client's commands to the successor.
    let mut outbox: std::collections::VecDeque<EngineCoreOutput> = stream.pending;
    let deadline = Instant::now() + Duration::from_secs(20);
    while reply.finished.is_none() {
        assert!(
            Instant::now() < deadline,
            "the migrated request did not finish"
        );
        while let Ok(event) = streams[0].1.try_recv() {
            outbox.push_back(event);
        }
        while stream.events.capacity() > 0
            && let Some(event) = outbox.pop_front()
        {
            stream.events.send(event).unwrap();
        }
        // The client's decoder decides on every token it received.
        match client.try_recv() {
            Ok(event) => reply.take(event),
            Err(_) => std::thread::sleep(Duration::from_millis(1)),
        }
        client.acknowledge_consumed_prefix();
        while let Ok(command) = quiesced.commands.try_recv() {
            if let uniserve_engine::Command::Acknowledge {
                request_id,
                output_token_count,
            } = command
            {
                successor
                    .handle()
                    .acknowledge_at(request_id, output_token_count);
            }
        }
    }
    assert_eq!(reply, expected);
    successor.shutdown();
}
