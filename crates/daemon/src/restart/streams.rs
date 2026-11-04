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
//! The SenseNova and omni readout output filters declare no state; item 2
//! removes their routes from the covered set, and until then a response they
//! write ends at the boundary.

use std::os::fd::OwnedFd;
use std::sync::{Arc, Mutex};

use anyhow::Context as _;
use proto::handoff;
use prost::Message as _;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::http::handoff::{Decision, FrozenStream};
use crate::http::stream::ResponseStream;
use crate::leader::Leader;

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
    /// When the successor took the response over, as `CLOCK_MONOTONIC`
    /// nanoseconds: the predecessor's pause ends here.
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
}

impl Paused {
    /// No response is in flight.
    fn none() -> Self {
        Self {
            records: Vec::new(),
            descriptors: Vec::new(),
            frozen: Vec::new(),
        }
    }

    /// Answers every frozen response: the ones the successor `adopted` go to
    /// it, and the rest resume here.
    ///
    /// Also writes the predecessor's handoff records — one `handoff stream`
    /// line per response it froze, with the successor's process id and the
    /// pause the successor measured, and one `handoff streams adopted` line
    /// after the last of them — which the deployment's restart test parses.
    pub(crate) fn decide(self, adopted: &[Taken], successor_pid: u32) {
        let decided_at = crate::http::handoff::now_ns();
        for frozen in self.frozen {
            let taken = adopted.iter().find(|taken| taken.request_id == frozen.request_id);
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
        info!(count = records.len(), "the responses in flight are paused");
        Paused {
            records,
            descriptors,
            frozen,
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
    next: &mut impl FnMut() -> anyhow::Result<OwnedFd>,
) -> Adoption {
    let mut taken = Vec::new();
    let mut streams = Vec::new();
    for record in records {
        match take_over(leader, &record, next) {
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
    adopted(&taken);
    queue(taken);
    Adoption { streams }
}

/// Takes one handed-over response over: its engine stream first, then its body
/// from the state, then its connection.
fn take_over(
    leader: &Arc<Leader>,
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
            let events = leader.take_over_request(&chat)?;
            let started = crate::output::run::started_instant(
                chat.lowering
                    .as_ref()
                    .map_or(0, |lowering| lowering.started_unix_us),
            );
            let outcome =
                leader.request_outcome(&state.request_id, chat.engine_replica, started);
            let run =
                crate::output::run::GenerationRun::resume(&chat, leader.tokenizer(), events, outcome)?;
            ResponseStream::start(Box::new(run), true)
        }
        other => anyhow::bail!("a handed-over response of an unsupported kind: {other:?}"),
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

/// Records that a successor wrote the first frame of each adopted response.
///
/// The record is one log line whose message is `handoff adopted ` and one
/// JSON object, which the deployment's restart test parses.
pub(crate) fn adopted(adopted: &[Adopted]) {
    for response in adopted {
        info!(
            request_id = %response.request_id,
            frozen_at_ns = response.frozen_at_ns,
            at_ns = response.at_ns,
            "handed-over response adopted"
        );
        info!(
            "handoff adopted {}",
            serde_json::json!({
                "request_id": response.request_id,
                "pid": std::process::id(),
                "at_ns": response.at_ns,
                "frozen_at_ns": response.frozen_at_ns,
                "pause_ms": response.at_ns.saturating_sub(response.frozen_at_ns) / 1_000_000,
            })
        );
    }
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
