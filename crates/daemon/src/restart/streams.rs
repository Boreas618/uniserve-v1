//! The responses in flight across a hot restart (design `specs/worker.md`
//! §7.5.1).
//!
//! # Mechanism
//!
//! The daemon that is replaced pauses every response it is writing: the
//! response stops at an engine event boundary, with every body frame of the
//! events it consumed already produced, and reports its state — the request's
//! engine identity and the position it consumed, the incremental decoder, the
//! output parsers, the assembled blocks, the chunk layer, the image counters
//! and the HTTP framing — with a descriptor of its socket. The successor takes
//! each response over: it takes the request's engine event stream over from
//! the consumed position ([`crate::engine::client::EngineClient::take_over`],
//! `uniserve.request.Command.Takeover`) so the request keeps generating and no
//! token is computed twice, resumes the session from the handed state, writes
//! the bytes the predecessor did not write, and goes on producing frames on the
//! same socket.
//!
//! # Boundary
//!
//! A response stops between two engine events, never inside one: every chunk
//! an event produced is a frame before the boundary, and the buffered chunk of
//! an update whose metadata has not arrived cannot outlive its event. The
//! keep-alive comment of a streamed response is the connection's, not the
//! body's, so a pause never has to break one.
//!
//! # Ownership
//!
//! A response stays the predecessor's until the successor reports that it took
//! the response over: the predecessor keeps its socket descriptor and writes
//! nothing on it after the pause, and only drops it once the successor's
//! `Adopted` names the response. A response the successor could not take over
//! is resumed here; the predecessor then finishes its handover without waiting
//! for any response to end, and the daemon leaves.
//!
//! # Limits
//!
//! Only the responses item 1 covers are handed over: a chat completion
//! streamed as server-sent events and one answered as a single JSON body, each
//! with a state a successor continues from. A covered response whose state
//! cannot be frozen ends at the boundary instead of being kept here — the
//! process that hands the deployment over does not wait for a response to end.
//! Every route the daemon serves declares that state, so no route is exempt.
//!
//! A video artifact's body is covered too: its state names the job and the
//! byte the predecessor reached, and the successor streams the job's retained
//! media from there. The media is the one the successor adopted with the job,
//! so a response whose media did not come over stays with the predecessor.

use std::os::fd::OwnedFd;
use std::sync::{Arc, Mutex};

use anyhow::Context as _;
use prost::Message as _;
use proto::handoff;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::http::handoff::{Decision, FrozenStream};
use crate::http::stream::ResponseStream;
use crate::leader::Leader;
use crate::openai::video_jobs::VideoJobs;

/// One response the predecessor froze, as it travels to the successor.
///
/// The state is `uniserve.handoff.Stream`'s own encoding: the restart wire's
/// serializer is not a protocol schema, so the state travels as bytes and the
/// successor reads it with the build it serves with.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct StreamHandoff {
    /// The request's identity as the deployment named it, for the record.
    pub(crate) request_id: String,
    /// When the predecessor stopped the response's socket, as `CLOCK_MONOTONIC`
    /// nanoseconds; the successor records its first frame on the same clock,
    /// so the pause both processes report is one measurement.
    pub(crate) frozen_at_ns: u64,
    /// The response's state (`uniserve.handoff.Stream`).
    pub(crate) state: Vec<u8>,
}

impl StreamHandoff {
    /// The handoff state of one frozen response.
    fn of(frozen: &FrozenStream) -> Self {
        Self {
            request_id: frozen.request_id.clone(),
            frozen_at_ns: frozen.frozen_at_ns,
            state: frozen.state.encode_to_vec(),
        }
    }
}

/// A response the successor took over, which its HTTP server serves.
pub(crate) struct Adopted {
    /// The request's identity, as the deployment named it.
    pub(crate) request_id: String,
    /// When the successor took the response's engine stream over, as
    /// `CLOCK_MONOTONIC` nanoseconds.
    ///
    /// This is the moment the engine serves the request to the successor, which
    /// is what the predecessor's `pause_ms` in `Paused::decide` runs to. It is
    /// not §7.5.1's pause, which ends when the successor writes the response's
    /// next event: the successor's own `handoff adopted` record carries that,
    /// and [`adopted`] writes it from the serving path.
    pub(crate) at_ns: u64,
    /// When the predecessor stopped the socket, as `CLOCK_MONOTONIC`
    /// nanoseconds.
    pub(crate) frozen_at_ns: u64,
    /// The response's HTTP framing, with the bytes the predecessor did not
    /// write.
    pub(crate) http: handoff::Http,
    /// The resumed body.
    pub(crate) body: ResponseStream,
    /// A descriptor of the connection's socket.
    pub(crate) descriptor: OwnedFd,
}

/// The responses a predecessor hands over, with the pause records that answer
/// them.
pub(crate) struct Paused {
    /// The states the successor takes over.
    pub(crate) records: Vec<StreamHandoff>,
    /// The sockets they travel with, in the records' order.
    pub(crate) descriptors: Vec<OwnedFd>,
    /// The registrations this process answers, in the same order.
    frozen: Vec<FrozenStream>,
    /// The handoff this pause closed, for the path that gives it back.
    handoff: Option<Arc<crate::http::handoff::Handoff>>,
    /// Whether the responses were answered: a pause that ends any other way
    /// gives the deployment back to this daemon.
    decided: bool,
}

/// A pause that ends without answering its responses — the successor gave up,
/// or the takeover failed before it answered — gives the deployment back: the
/// responses resume here (their decision's sender dies with this struct) and
/// the daemon admits new requests again, instead of closing every connection
/// for a handover that never happened.
impl Drop for Paused {
    fn drop(&mut self) {
        if self.decided {
            return;
        }
        if let Some(handoff) = &self.handoff {
            handoff.release();
        }
    }
}

impl Paused {
    /// No response is in flight.
    fn none() -> Self {
        Self {
            records: Vec::new(),
            descriptors: Vec::new(),
            frozen: Vec::new(),
            handoff: None,
            decided: true,
        }
    }

    /// Answers every frozen response: the ones the successor `adopted` go to
    /// it, and the rest resume here.
    ///
    /// Also writes the predecessor's handoff records — one `handoff stream`
    /// line per response it froze, with the successor's process id and this
    /// process's own view of the pause, which runs to the successor's takeover
    /// of the response's engine stream ([`Adopted::at_ns`]), and one
    /// `handoff streams adopted` line after the last of them — which the
    /// deployment's restart test parses. This is the shorter of the two views:
    /// §7.5.1's pause, which ends when the successor writes the response's next
    /// event, is the successor's [`adopted`] record.
    pub(crate) fn decide(mut self, adopted: &[Taken], successor_pid: u32) {
        self.decided = true;
        let decided_at = crate::http::handoff::now_ns();
        for frozen in std::mem::take(&mut self.frozen) {
            let taken = adopted
                .iter()
                .find(|taken| taken.request_id == frozen.request_id);
            let pause_ms = taken.map_or_else(
                || decided_at.saturating_sub(frozen.frozen_at_ns) / 1_000_000,
                |taken| taken.at_ns.saturating_sub(frozen.frozen_at_ns) / 1_000_000,
            );
            info!(
                "handoff stream {}",
                serde_json::json!({
                    "request_id": frozen.request_id,
                    "successor_pid": successor_pid,
                    "pid": std::process::id(),
                    "pause_ms": pause_ms,
                    "frozen_at_ns": frozen.frozen_at_ns,
                    // When the engine moved this response's stream to the
                    // successor (design §7.5.1 step 4). The ownership window
                    // runs from here to the successor's first new frame, which
                    // its own `handoff adopted` record carries; it is null
                    // when the successor took nothing over. This is where the
                    // stream is no longer the predecessor's, which is what a
                    // successor's death in that window costs.
                    "engine_at_ns": taken.map(|adopted| adopted.at_ns),
                    "adopted": taken.is_some(),
                })
            );
            let taken = taken.is_some();
            if !taken {
                warn!(
                    request_id = %frozen.request_id,
                    "the successor did not take this response over; it resumes here"
                );
            }
            let decision = if taken {
                Decision::Adopt
            } else {
                Decision::Resume
            };
            // A response whose connection already ended needs no decision.
            let _ = frozen.decision.send(decision);
        }
        info!(
            "handoff streams adopted {}",
            serde_json::json!({
                "pid": std::process::id(),
                "successor_pid": successor_pid,
                "at_ns": crate::http::handoff::now_ns(),
            })
        );
    }
}

impl super::Restarter {
    /// Stops every response in flight at its next engine event boundary and
    /// returns what the successor takes over: each response's state with a
    /// descriptor of its socket, and the registrations this process answers.
    ///
    /// A response whose socket cannot be duplicated stays here and resumes
    /// when the successor names the responses it took.
    pub(crate) async fn pause_streams(&self) -> Paused {
        let Some(handoff) = self.handoff() else {
            return Paused::none();
        };
        let frozen = handoff.pause().await;
        let mut records = Vec::with_capacity(frozen.len());
        let mut descriptors = Vec::with_capacity(frozen.len());
        for stream in &frozen {
            match stream.descriptor.try_clone() {
                Ok(descriptor) => {
                    records.push(StreamHandoff::of(stream));
                    descriptors.push(descriptor);
                }
                Err(error) => warn!(
                    request_id = %stream.request_id,
                    %error,
                    "a response's socket cannot be duplicated; it stays here"
                ),
            }
        }
        // Two counts, because they differ exactly when a response cannot be
        // handed over: every registration paused, and the ones whose socket
        // travelled. A paused response with no descriptor resumes here, and
        // the deployment's restart test counts the handed ones.
        info!(
            paused = frozen.len(),
            handed = records.len(),
            "the responses in flight are paused"
        );
        Paused {
            records,
            descriptors,
            frozen,
            handoff: Some(handoff),
            decided: false,
        }
    }
}

/// The responses this process took over, which its HTTP server adopts once it
/// serves.
///
/// A successor takes the responses over before it serves, and serves from
/// another task with the router it builds; the queue is where the two meet.
static ADOPTED: Mutex<Vec<Adopted>> = Mutex::new(Vec::new());

/// Queues the responses this process took over for its HTTP server.
pub(crate) fn queue(adopted: Vec<Adopted>) {
    let mut queue = crate::leader::locked(&ADOPTED);
    queue.extend(adopted);
}

/// Takes the responses this process took over.
pub(crate) fn take_adopted() -> Vec<Adopted> {
    std::mem::take(&mut crate::leader::locked(&ADOPTED))
}

/// One response a successor took over, as it answers the predecessor.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Taken {
    /// The request's identity, as the deployment named it.
    pub(crate) request_id: String,
    /// When the successor took it over, as `CLOCK_MONOTONIC` nanoseconds.
    pub(crate) at_ns: u64,
}

/// What a successor took over of a predecessor's responses.
pub(crate) struct Adoption {
    /// The responses it took over and owns from now on. A response it could
    /// not take over is absent: it stays the predecessor's.
    pub(crate) streams: Vec<Taken>,
}

/// Takes the responses `records` describe over from the predecessor, whose
/// sockets follow the takeover's other descriptors in `next`.
///
/// Each response's engine stream is taken over first, so a response only moves
/// once the engine serves it; a response that cannot be taken over is left
/// with the predecessor, which resumes it.
pub(crate) fn adopt(
    leader: &Arc<Leader>,
    records: Vec<StreamHandoff>,
    jobs: &Arc<VideoJobs>,
    next: &mut impl FnMut() -> anyhow::Result<OwnedFd>,
) -> Adoption {
    let mut taken = Vec::new();
    let mut streams = Vec::new();
    for record in records {
        match take_over(leader, jobs, &record, next) {
            Ok(response) => {
                streams.push(Taken {
                    request_id: response.request_id.clone(),
                    at_ns: response.at_ns,
                });
                taken.push(response);
            }
            Err(error) => {
                warn!(
                    request_id = %record.request_id,
                    error = %error,
                    "a handed-over response cannot be taken over; it stays with the predecessor"
                );
            }
        }
    }
    queue(taken);
    Adoption { streams }
}

/// Takes one handed-over response over: its engine stream first, then its body
/// from the state, then its connection.
fn take_over(
    leader: &Arc<Leader>,
    jobs: &Arc<VideoJobs>,
    record: &StreamHandoff,
    next: &mut impl FnMut() -> anyhow::Result<OwnedFd>,
) -> anyhow::Result<Adopted> {
    let at_ns = crate::http::handoff::now_ns();
    let state = handoff::Stream::decode(record.state.as_slice())
        .context("reading a handed-over response's state")?;
    let http = state
        .http
        .clone()
        .context("a handed-over response states no HTTP framing")?;
    let application = state
        .application
        .clone()
        .context("a handed-over response states no application")?;
    let descriptor = next()?;
    let body = match application {
        handoff::stream::Application::ChatStream(chat) => {
            let events = leader.take_over_request(
                chat.engine_replica,
                &chat.engine_request_id,
                chat.consumed_events,
            )?;
            let started = crate::output::run::started_instant(
                chat.lowering
                    .as_ref()
                    .map_or(0, |lowering| lowering.started_unix_us),
            );
            let outcome = leader.request_outcome(&state.request_id, chat.engine_replica, started);
            let run = crate::output::run::GenerationRun::resume_stream(
                &chat,
                leader.tokenizer(),
                events,
                outcome,
            )?;
            // Server-sent events: the body keeps its connection alive while
            // it waits for the engine.
            ResponseStream::start(Box::new(run), true)
        }
        handoff::stream::Application::ChatResponse(chat) => {
            let events = leader.take_over_request(
                chat.engine_replica,
                &chat.engine_request_id,
                chat.consumed_events,
            )?;
            let started = crate::output::run::started_instant(
                chat.lowering
                    .as_ref()
                    .map_or(0, |lowering| lowering.started_unix_us),
            );
            let outcome = leader.request_outcome(&state.request_id, chat.engine_replica, started);
            let run = crate::output::run::GenerationRun::resume_collection(
                &chat,
                leader.tokenizer(),
                events,
                outcome,
            )?;
            // One JSON body: no server-sent events, so no keep-alive comment.
            ResponseStream::start(Box::new(run), false)
        }
        handoff::stream::Application::VideoContent(content) => {
            // The artifact is the successor's own retained media: the job was
            // adopted from the predecessor before its responses are, so a job
            // that is not here is one whose media did not come over.
            let media = jobs.content(&content.job_id).with_context(|| {
                format!(
                    "the retained media of the handed-over job {} is not here",
                    content.job_id
                )
            })?;
            let run = crate::output::run::VideoContentRun::new(
                content.job_id.clone(),
                content.content_type.clone(),
                media,
            )
            .from_offset(usize::try_from(content.offset).unwrap_or(usize::MAX));
            // Artifact bytes, not events: no server-sent keep-alive.
            ResponseStream::start(Box::new(run), false)
        }
    };
    info!(
        request_id = %record.request_id,
        frozen_at_ns = record.frozen_at_ns,
        "a handed-over response is taken over"
    );
    Ok(Adopted {
        request_id: record.request_id.clone(),
        at_ns,
        frozen_at_ns: record.frozen_at_ns,
        http,
        body,
        descriptor,
    })
}

/// Records that this process is serving one handed-over response, at the moment
/// it has written the predecessor's bytes and is about to write the response's
/// first event of its own.
///
/// The record is one log line whose message is `handoff adopted ` and one JSON
/// object, which the deployment's restart test parses. Exactly one is written
/// per handed-over response, by the process that serves it, so `at_ns` is
/// §7.5.1 step 4's moment and `pause_ms` is the response's pause as §7.5.1
/// defines it. The predecessor's `pause_ms` is its own, shorter view
/// ([`Adopted::at_ns`]) and the artifact does not read it.
pub(crate) fn adopted(request_id: &str, frozen_at_ns: u64) {
    let at_ns = crate::http::handoff::now_ns();
    info!(
        request_id,
        frozen_at_ns, at_ns, "a handed-over response is served here"
    );
    info!(
        "handoff adopted {}",
        serde_json::json!({
            "request_id": request_id,
            "pid": std::process::id(),
            "at_ns": at_ns,
            "frozen_at_ns": frozen_at_ns,
            "pause_ms": at_ns.saturating_sub(frozen_at_ns) / 1_000_000,
        })
    );
}

/// Records that a successor finished writing one adopted response.
///
/// The record is one log line whose message is `handoff finished ` and one
/// JSON object, which the deployment's restart test parses.
pub(crate) fn finished(request_id: &str) {
    info!(
        "handoff finished {}",
        serde_json::json!({
            "request_id": request_id,
            "pid": std::process::id(),
            "at_ns": crate::http::handoff::now_ns(),
        })
    );
}

#[cfg(test)]
mod tests {
    use proto::handoff::stream::Application;
    use scheduler::SimEngine;
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::options::VideoJobConfig;
    use crate::test_support::{configured_tokenizer, leader_over, sim_client, sim_core, sim_model};

    /// A pause that ends without answering its responses — the successor gave
    /// up, or the takeover failed before it answered — gives the deployment
    /// back: the daemon admits requests again rather than closing every
    /// connection for a handover that did not happen, and no round is left
    /// running for a response that registers later. A decided pause does not:
    /// the predecessor is leaving, and its new requests are the successor's.
    #[tokio::test]
    async fn a_pause_dropped_without_a_decision_gives_the_deployment_back() {
        let handoff = crate::http::handoff::Handoff::new();
        assert!(handoff.pause().await.is_empty());
        assert!(
            !handoff.admissions().open(),
            "the handover stopped admissions"
        );
        let undecided = Paused {
            records: Vec::new(),
            descriptors: Vec::new(),
            frozen: Vec::new(),
            handoff: Some(Arc::new(handoff.clone())),
            decided: false,
        };
        drop(undecided);
        assert!(
            handoff.admissions().open(),
            "the daemon admits requests again"
        );
        assert!(!handoff.pausing(), "no round is left running");

        // The successor's answer was given: this process finishes its
        // responses and leaves the deployment's new requests to the successor.
        assert!(handoff.pause().await.is_empty());
        let decided = Paused {
            records: Vec::new(),
            descriptors: Vec::new(),
            frozen: Vec::new(),
            handoff: Some(Arc::new(handoff.clone())),
            decided: true,
        };
        drop(decided);
        assert!(
            !handoff.admissions().open(),
            "a decided handover keeps them closed"
        );
    }

    /// A response the successor cannot read is not taken over: it stays the
    /// predecessor's, it is reported as taken over by nobody, and the
    /// successor does not take its socket out of the handover — it does not
    /// own the connection, so nothing about it moves.
    #[tokio::test]
    async fn a_response_that_cannot_be_read_stays_with_the_predecessor() {
        // The queue of taken-over responses is this process's, so the test
        // starts from what earlier tests left there.
        let _ = take_adopted();
        let core = sim_core(scheduler::sim_args("refused-handoff"), SimEngine::new());
        let client = std::sync::Arc::new(sim_client(vec![core]));
        let leader = std::sync::Arc::new(leader_over(
            sim_model(crate::test_support::CHAT),
            configured_tokenizer(),
            "refused-handoff".into(),
            vec![client],
        ));
        let jobs = std::sync::Arc::new(VideoJobs::new(
            VideoJobConfig::default(),
            CancellationToken::new(),
        ));

        // Bytes that are not a `uniserve.handoff.Stream`.
        let record = StreamHandoff {
            request_id: "refused-handoff-request".to_owned(),
            frozen_at_ns: 1,
            state: b"not a handed-over stream".to_vec(),
        };
        let mut socket_taken = false;
        let adoption = adopt(&leader, vec![record], &jobs, &mut || {
            socket_taken = true;
            anyhow::bail!("a refused response must not take a descriptor")
        });

        assert!(
            adoption.streams.is_empty(),
            "a response that cannot be read is taken over by nobody"
        );
        assert!(take_adopted().is_empty(), "nothing is reported as adopted");
        assert!(
            !socket_taken,
            "the successor takes no socket of a response it did not take over"
        );
    }

    /// The same refusal when the state is readable but names no application:
    /// the record is complete enough to decode and still cannot be resumed.
    #[tokio::test]
    async fn a_response_that_names_no_application_stays_with_the_predecessor() {
        let _ = take_adopted();
        let core = sim_core(scheduler::sim_args("refused-handoff-2"), SimEngine::new());
        let client = std::sync::Arc::new(sim_client(vec![core]));
        let leader = std::sync::Arc::new(leader_over(
            sim_model(crate::test_support::CHAT),
            configured_tokenizer(),
            "refused-handoff-2".into(),
            vec![client],
        ));
        let jobs = std::sync::Arc::new(VideoJobs::new(
            VideoJobConfig::default(),
            CancellationToken::new(),
        ));
        let state = StreamHandoff {
            request_id: "nameless-application".to_owned(),
            frozen_at_ns: 1,
            state: Vec::new(),
        };
        // A decoded stream with no application and no framing.
        let mut record = state;
        record.state = handoff::Stream {
            request_id: "nameless-application".to_owned(),
            http: None,
            application: None::<Application>,
        }
        .encode_to_vec();

        let mut socket_taken = false;
        let adoption = adopt(&leader, vec![record], &jobs, &mut || {
            socket_taken = true;
            anyhow::bail!("a refused response must not take a descriptor")
        });

        assert!(adoption.streams.is_empty());
        assert!(take_adopted().is_empty());
        assert!(!socket_taken, "a response without framing does not move");
    }

    /// §7.5.1's ownership rule, which no run reaches: the responses the
    /// successor reported taking are adopted — it writes the rest of them —
    /// and the ones it did not report resume here, so a response the successor
    /// could not take over is the predecessor's again from the boundary it
    /// stopped at.
    #[tokio::test]
    async fn the_predecessor_adopts_what_the_successor_took_and_resumes_the_rest() {
        let socket = |request_id: &str| {
            let (ours, theirs) = std::os::unix::net::UnixStream::pair().expect("a socket pair");
            // The far end outlives the handoff in a real deployment (it is the
            // client); the test's pauses are microseconds and the socket is
            // never written, so dropping it here changes nothing.
            drop(theirs);
            let (decision, answer) = tokio::sync::oneshot::channel();
            let frozen = FrozenStream {
                request_id: request_id.to_owned(),
                frozen_at_ns: 1_000,
                state: handoff::Stream::default(),
                descriptor: std::os::fd::OwnedFd::from(ours),
                decision,
            };
            (frozen, answer)
        };
        let (taken, mut taken_answer) = socket("taken");
        let (resumed, mut resumed_answer) = socket("resumed");
        let paused = Paused {
            records: Vec::new(),
            descriptors: Vec::new(),
            frozen: vec![taken, resumed],
            handoff: None,
            decided: false,
        };

        paused.decide(
            &[Taken {
                request_id: "taken".to_owned(),
                at_ns: 1_500,
            }],
            4_242,
        );

        assert_eq!(
            taken_answer
                .try_recv()
                .expect("the taken response is answered"),
            Decision::Adopt
        );
        assert_eq!(
            resumed_answer
                .try_recv()
                .expect("the untaken response is answered"),
            Decision::Resume
        );
    }
}
