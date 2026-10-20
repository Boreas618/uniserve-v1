//! Control-plane hot update: what one engine core hands its successor.
//!
//! A predecessor's engine core quiesces at a step boundary
//! (`EngineCore::quiesce`): it stops admitting, lets every request it cannot
//! migrate run to completion, pauses the rest, publishes each migrating
//! request's computed cache pages to its prefix cache, and snapshots the
//! requests and the prefix cache's index ([`CoreSnapshot`]). Its workers then
//! close, releasing their exclusive storages with the contents intact, and
//! its scheduler hands the migrated requests' event channels and its command
//! channel to a relay ([`Quiesced`]).
//!
//! The successor's engine core started over standby workers that adopted
//! the predecessor's claim and cache units ([`Adoption`]). It resumes the
//! snapshot (`EngineCore::resume`): its workers take the deployment's claim
//! over and are promoted, the prefix cache installs the predecessor's pages,
//! and each migrated request is queued again with its prompt followed by
//! the tokens it generated. Admission finds those tokens' pages in the
//! prefix cache, so the request recomputes at most its last partial page,
//! and it keeps its identity and epoch, so its sampling stream continues.
//!
//! A request migrates when it is a token request that is waiting, or that
//! runs without images (context, generated or canvas) and has no terminal
//! intent. Every other request completes on the predecessor before it
//! quiesces.

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};
use uniserve_core::{EngineCoreOutput, GenerationRequest, RequestId};
use uniserve_ipc::WorkerEndpoint;

use crate::handle::{Command, EventTx};
use crate::prefix_cache::IndexedPage;

/// Version of the request snapshot and the output relay. A successor refuses
/// a predecessor that states another.
pub const SNAPSHOT_VERSION: u32 = 1;

/// What a successor's workers adopt from the predecessor's: the claim under
/// which they stand by, and the cache units of the planes they map.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Adoption {
    /// The deployment's claim at this host's Storage Server.
    pub claim: [u8; 16],
    /// Each worker group's prefix cache units, by group identity: the
    /// planes the successor's workers map hold exactly these.
    pub cache_units: BTreeMap<String, Option<u32>>,
}

/// One engine core's requests and prefix cache index at a step boundary.
///
/// It refers to cache memory only by unit, never by address.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoreSnapshot {
    /// The engine core's next request identity, after which the successor's
    /// engine core issues its own.
    pub next_request_id: u64,
    /// The scheduler's next request epoch.
    pub next_request_epoch: u64,
    /// Migrated requests in the order the successor queues them: the
    /// running ones in admission order, then the waiting ones in queue order.
    pub requests: Vec<MigratedRequest>,
    /// The prefix cache's published pages in eviction order
    /// (`BlockPool::snapshot`).
    pub pages: Vec<IndexedPage>,
    /// The endpoint of each worker group, by group identity, which the
    /// pages name as their source.
    pub endpoints: BTreeMap<String, WorkerEndpoint>,
}

/// One migrated token request: as submitted, and its progress.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MigratedRequest {
    /// The request as it was submitted.
    pub request: GenerationRequest,
    /// The request's epoch, which with its identity keys its sampling
    /// stream.
    pub request_epoch: u64,
    /// The tokens it generated, in order. They follow its prompt when it
    /// resumes.
    pub generated: Vec<u32>,
    /// Public token events its client was sent.
    pub tokens_sent: usize,
    /// Public token counts awaiting the frontend decoder's stop-string
    /// decision, oldest first.
    pub decoder_boundaries: Vec<usize>,
    /// When it entered the waiting queue, in seconds since the Unix epoch.
    pub queued_at: f64,
    /// Whether its client was sent its `Scheduled` event.
    pub scheduled: bool,
}

/// A quiesced engine core's handover: the snapshot to send, and what the
/// predecessor's relay keeps.
pub struct Quiesced {
    pub snapshot: CoreSnapshot,
    /// Each migrated request's event channel to its client, with the events
    /// the client has not yet been sent, in order.
    pub streams: Vec<MigratedStream>,
    /// The engine core's command channel. Its scheduler has stopped; the
    /// relay drains the commands the core's handles still send.
    pub commands: crossbeam_channel::Receiver<Command>,
}

/// The client side of one migrated request on the predecessor.
pub struct MigratedStream {
    pub request_id: RequestId,
    pub events: EventTx,
    /// Events journaled behind a full channel, which precede every event of
    /// the successor.
    pub pending: VecDeque<EngineCoreOutput>,
}
