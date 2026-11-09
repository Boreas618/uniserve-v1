//! Requests and engine commands on the request interface: routing a submission to a scheduler,
//! the per-request commands, the engine commands, a warming's reads, and the event streams each
//! admitted request produces, journaled so that a takeover replays what its consumer has not
//! rendered ([`HANDOVER_JOURNAL`]).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use ipc::UpdateFailure;

use super::MainProcess;
use super::answer;
use super::operations::Owing;

/// How many of a request's events this process keeps for a takeover to replay.
///
/// The event window is the credit a consumer reports in, not the distance it can be behind by:
/// this process writes every event as it is produced, the daemon files them without bound, and the
/// body writes them to a socket the client drains at its own speed, so at a fast decode a response
/// that stops reading is tens of events behind. The pause is that case at its sharpest - a
/// predecessor's consumer stops at the pause while this process goes on producing - and it begins
/// before this process has an attached front to tell it so, so the wide bound is the ordinary one
/// and pruning on a consumer's report is what keeps the journal small in service.
///
/// **It is a memory budget, not a protocol bound.** An event is a token or a scheduling fact,
/// about 100 bytes with its wire form: 4096 of them is roughly 400 KB for one request, so a
/// deployment whose every stream stalls at the bound spends about 50 MB at 128 streams. Raising it
/// buys a longer stall before a takeover is refused; it does not change what the refusal means. A
/// takeover whose position is older than what is kept is refused by name rather than served as a
/// stream with a hole in it.
pub(crate) const HANDOVER_JOURNAL: usize = 4096;

/// One admitted request and the connection its events belong to.
///
/// A request belongs to the daemon front that submitted it until a `Takeover`
/// moves it to another. The events the front's consumer has
/// not reported as consumed are kept, because a takeover from a position names
/// what has to be replayed: the predecessor freezes its response at the pause,
/// so its consumer stops advancing while this process goes on producing, and
/// the successor's body resumes from the position it last rendered.
pub(super) struct Served {
    /// The daemon's identity for the request.
    pub(super) request: u64,
    /// The scheduler rank a command for the request is routed to.
    pub(super) rank: usize,
    /// What the request produced.
    pub(super) events: scheduler::EventRx,
    /// The daemon front it belongs to: 0 is the daemon that launched this
    /// process, and each attached daemon adds one.
    pub(super) front: usize,
    /// Events written to a front so far, which is the position a takeover
    /// names and the consumer reports.
    pub(super) sent: u64,
    /// The events written to a front that a takeover from an earlier position
    /// has to replay, by their position.
    ///
    /// Bounded by [`HANDOVER_JOURNAL`], not by the event window the consumer's
    /// credit names: this process hands every event to the daemon as it is
    /// produced and the daemon files them into a channel it drains at the
    /// client's pace, so the consumer can be further behind the producer than any
    /// window. The events the consumer reported consumed are dropped as they are
    /// reported.
    pub(super) kept: std::collections::VecDeque<KeptEvent>,
}

/// One journaled event: its position in the request's stream, the event, and the artifact bytes it
/// travels with. A replay without the bytes would be an artifact event the daemon cannot read.
type KeptEvent = (
    u64,
    proto::request::wire::Event,
    Option<std::sync::Arc<dyn proto::request::ArtifactMedia>>,
);

/// The most artifact bytes one frame carries, so a frame's size does not grow with the clip.
const ARTIFACT_PART: usize = 1 << 20;

/// Writes the fence probes' device switch (`docs/profiling.md`): `{"enabled": bool, "off":
/// [patterns]}`. A process without probes answers so.
pub(super) fn fence_switch(payload: &str) -> std::result::Result<Vec<String>, String> {
    let payload: serde_json::Value =
        serde_json::from_str(payload).map_err(|error| format!("fence_switch: {error}"))?;
    let enabled = payload["enabled"]
        .as_bool()
        .ok_or_else(|| "fence_switch states no enabled".to_owned())?;
    let off: Vec<String> = match payload.get("off") {
        Some(off) if !off.is_null() => serde_json::from_value(off.clone())
            .map_err(|error| format!("fence_switch off: {error}"))?,
        _ => Vec::new(),
    };
    let switched = worker::profiling::fences::Fences::installed()
        .map(|fences| fences.switch(enabled, &off))
        .transpose()
        .map_err(|error| error.to_string())?;
    Ok(vec![
        serde_json::json!({
            "probes": switched.is_some(),
            "enabled": enabled,
            "off": switched.unwrap_or_default(),
        })
        .to_string(),
    ])
}

/// How long a scheduler takes to answer a warming's read: its next pass.
const WARMING_ANSWER: Duration = Duration::from_secs(2);

/// What a warming operation that copies pages is answered: pages reach a target through the host
/// KV cache, so this process runs no copy of its own.
const PAGES_THROUGH_HOST: &str = "an epoch's pages reach its target through the host KV cache: this \
     process copies none itself";
/// The events a takeover from `from` replays: everything this process produced past the position
/// the successor's consumer states.
///
/// The predecessor freezes its response at the pause, so its consumer stops advancing while this
/// process goes on producing, and the successor resumes from what it last rendered. Without the
/// replay those events are rendered nowhere and the response loses the tokens the pause covered.
fn replay_after(
    kept: &std::collections::VecDeque<KeptEvent>,
    from: u64,
) -> Vec<(
    proto::request::wire::Event,
    Option<std::sync::Arc<dyn proto::request::ArtifactMedia>>,
)> {
    kept.iter()
        .filter(|(position, _, _)| *position > from)
        .map(|(_, event, media)| (event.clone(), media.clone()))
        .collect()
}

/// Journals one event at `position`, dropping the oldest past `bound`.
fn keep_event(
    kept: &mut std::collections::VecDeque<KeptEvent>,
    position: u64,
    event: proto::request::wire::Event,
    media: Option<std::sync::Arc<dyn proto::request::ArtifactMedia>>,
    bound: usize,
) {
    kept.push_back((position, event, media));
    while kept.len() > bound {
        kept.pop_front();
    }
}

/// What the engine answers a command whose owner is the epoch layer: the same operation exists,
/// and it is asked for through the process connection, where the driver applies it.
const EPOCH_LIFECYCLE: &str = "this command is the epoch layer's, applied through the process connection rather than the \
     request connection";
/// What the engine answers a command over the deployment's storage ledgers or transfer budgets:
/// each node's daemon reads them over the claim it holds, not through an epoch's main process.
pub(super) const LEDGERS: &str = "the storage ledgers and transfer budgets are read by each node's daemon over its claim, not \
     by an epoch's main process";
/// The name of an engine command, for a log line about one this epoch does not serve.
pub(super) fn engine_command_name(
    kind: &proto::request::wire::engine_command::Kind,
) -> &'static str {
    use proto::request::wire::engine_command::Kind;
    match kind {
        Kind::Shutdown(_) => "shutdown",
        Kind::Promote(_) => "promote",
        Kind::FlushCache(_) => "flush_cache",
        Kind::Hook(_) => "hook",
        Kind::Configure(_) => "configure",
        Kind::Attach(_) => "attach",
        Kind::ResizeCache(_) => "resize_cache",
        Kind::UpdateWeights(_) => "update_weights",
        Kind::Park(_) => "park",
        Kind::Resume(_) => "resume",
        Kind::Commit(_) => "commit",
        Kind::DescribeCache(_) => "describe_cache",
        Kind::DescribeStorage(_) => "describe_storage",
        Kind::ThrottleTransfers(_) => "throttle_transfers",
        Kind::DescribeRequests(_) => "describe_requests",
        Kind::CacheWarm(_) => "cache_warm",
        Kind::CommitExperts(_) => "commit_experts",
    }
}

impl MainProcess {
    /// Routes a request to the scheduler that serves it (`route`).
    pub(super) fn route(&self, _request: &proto::request::Request) -> usize {
        // The routing policy runs on the scheduler node; a request submitted to
        // a node that holds no scheduler belongs to this node's single rank.
        if self.schedulers.len() <= 1 {
            return 0;
        }
        // Several DP ranks on one node: the mechanism's policy instance picks the rank.
        // Until that call is published the rank holding the fewest requests takes it, because a
        // rank the requests never reach leaves its share of the batch empty.
        let mut held = vec![0_usize; self.schedulers.len()];
        for served in &self.streams {
            if let Some(count) = held.get_mut(served.rank) {
                *count += 1;
            }
        }
        (0..held.len()).min_by_key(|rank| held[*rank]).unwrap_or(0)
    }

    /// Routes a per-request command to the scheduler that holds the request.
    ///
    /// The request's scheduler is the rank the submission was routed to; the
    /// daemon addresses commands to the main process, and the mechanism keeps
    /// the rank of every request it accepted.
    pub(super) fn request_command(
        &mut self,
        front: usize,
        command: proto::request::wire::Command,
    ) -> Result<()> {
        // Two of a request's commands are the connection's bookkeeping rather than a scheduler's,
        // and this process is the connection: it writes every event to the front the request
        // belongs to and keeps what a takeover has to replay. Handling them here keeps the
        // scheduler's refusal, which is right for a scheduler, from ending the epoch on the first
        // long request: the daemon reports a consumer's position every `EVENT_WINDOW / 2` events.
        match command.kind.as_ref() {
            Some(proto::request::wire::command::Kind::Consumed(consumed)) => {
                let Some(served) = self
                    .streams
                    .iter_mut()
                    .find(|served| served.request == command.request)
                else {
                    tracing::debug!(
                        request = command.request,
                        "a consumer's position names a request this epoch no longer serves"
                    );
                    return Ok(());
                };
                // Everything the consumer has read can no longer be replayed, which is what bounds
                // the journal.
                served
                    .kept
                    .retain(|(position, _, _)| *position > consumed.events);
                return Ok(());
            }
            Some(proto::request::wire::command::Kind::Takeover(takeover)) => {
                let Some(index) = self
                    .streams
                    .iter()
                    .position(|served| served.request == command.request)
                else {
                    // A handover's takeover names a request this process kept running for it:
                    // one it does not hold is either a request that ended first or a front that
                    // named the wrong identity, and only the log can say which.
                    tracing::warn!(
                        request = command.request,
                        front,
                        from = takeover.events,
                        "a takeover names a request this epoch no longer serves"
                    );
                    return Ok(());
                };
                // The request is this daemon's from the position its consumer states: what the
                // predecessor produced past it is replayed, because its own consumer stopped
                // reading at the pause.
                //
                // The journal and the position are read out before the process is borrowed again:
                // a refusal writes to a front, and the stream is a field of the same process.
                let kept = self.streams[index].kept.clone();
                let sent = self.streams[index].sent;
                // A position older than the journal cannot be replayed: the events between it and
                // the oldest one kept were dropped, and handing the successor what is left is a
                // stream that skips tokens with nothing saying so. It is refused, naming the two
                // positions and how far behind the consumer was, so the daemon ends the response
                // instead of serving a body with a hole in it.
                if let (Some((oldest, _, _)), true) = (kept.front(), sent > takeover.events) {
                    if takeover.events + 1 < *oldest {
                        let oldest = *oldest;
                        let behind = sent - takeover.events;
                        tracing::error!(
                            request = command.request,
                            front,
                            from = takeover.events,
                            oldest,
                            sent,
                            behind,
                            "a takeover names a position this process no longer holds: the \
                             consumer was more events behind than the journal kept"
                        );
                        self.reject(
                            front,
                            command.request,
                            proto::request::RejectionKind::Overloaded,
                            format!(
                                "the request could not be taken over: its consumer's position \
                                 {} is {} event(s) behind, and this process kept the last {} of \
                                 them from position {} - the handover produced more events of \
                                 this request than a pause may",
                                takeover.events,
                                behind,
                                kept.len(),
                                oldest,
                            ),
                        );
                        return Ok(());
                    }
                }
                let replay = replay_after(&kept, takeover.events);
                self.streams[index].front = front;
                for (event, media) in replay {
                    tracing::debug!(
                        request = command.request,
                        front,
                        "replaying an event a taken-over request produced during the pause"
                    );
                    self.send_stream_event(front, event, media);
                }
                tracing::info!(
                    request = command.request,
                    front,
                    from = takeover.events,
                    "a request was taken over by another daemon front"
                );
                return Ok(());
            }
            _ => {}
        }
        // A command names its request, not the scheduler serving it, so the rank is read from the
        // stream the submission answered with. A request that is not in flight here has already
        // ended: a cancel or a stop for it is a no-op, not a fault, and ending an epoch because a
        // client disconnected a moment too late would be the daemon's most common message.
        let Some(rank) = self
            .streams
            .iter()
            .find(|served| served.request == command.request)
            .map(|served| served.rank)
        else {
            tracing::debug!(
                request = command.request,
                "a request command names a request this epoch no longer serves"
            );
            return Ok(());
        };
        let Some(scheduler) = self.schedulers.get(rank) else {
            tracing::debug!(
                request = command.request,
                rank,
                "a request command names a rank this node does not hold"
            );
            return Ok(());
        };
        answer(
            scheduler.request_command(command),
            self.watchdog.stall_bound(),
            "a request command",
        )
    }

    /// A command to the whole main process: shutdown, drain, park or promote,
    /// and the program-layer commands a single-rank node receives directly.
    pub(super) fn engine_command(
        &mut self,
        front: usize,
        command: proto::request::wire::EngineCommand,
    ) -> Result<Option<UpdateFailure>> {
        // The epoch's lifecycle - Park, Promote, Serve, Drain, Pause, the program layer's
        // stages - arrives on the process connection and is applied by the driver. On this
        // connection the epoch owns exactly one command, `Shutdown`: every request finishes as
        // aborted and the owners close in the order they need (`close`).
        //
        // The data plane runs here too: each of its commands goes to whoever does the work,
        // and the rest of the engine surface is answered with whose it is instead of being dropped.
        // A command a process does not serve is otherwise a caller waiting for a reply that never
        // comes, which is a hang with no name in it.
        let refused = match command.kind {
            Some(proto::request::wire::engine_command::Kind::Shutdown(_)) => {
                self.stopping = true;
                return Ok(None);
            }
            // A lost process's standby is promoted by the daemon that held the two, on this
            // connection, because no epoch switch is running to command it on the process one. The
            // daemon moves the frontend when this process states that it serves.
            Some(proto::request::wire::engine_command::Kind::Promote(_)) => {
                self.promote_in_place()?;
                return Ok(None);
            }
            // A hot restart's successor gives this engine a second daemon front, so
            // the responses in flight can move to it while the process goes on serving. A refusal
            // is a failed handover and not this process's end: the daemon that asked is still
            // serving the requests it took.
            Some(proto::request::wire::engine_command::Kind::Attach(attach)) => {
                let state = self.reported_state;
                let info = self.announced_info();
                match self.control.attach(&attach.requests, state, info) {
                    Ok(front) => tracing::info!(
                        daemon = %attach.requests,
                        front,
                        "a daemon attached to this engine"
                    ),
                    Err(error) => tracing::warn!(
                        daemon = %attach.requests,
                        %error,
                        "attaching a daemon failed; this engine keeps the fronts it has"
                    ),
                }
                return Ok(None);
            }
            // What the data plane changes inside the epoch. None of these waits here: each
            // goes to whoever does the work and is answered by the pass that sees it end.
            Some(proto::request::wire::engine_command::Kind::Hook(hook)) => {
                self.engine_operation(front, hook);
                return Ok(None);
            }
            Some(proto::request::wire::engine_command::Kind::ResizeCache(resize)) => {
                self.resize_cache(front, &resize);
                return Ok(None);
            }
            Some(proto::request::wire::engine_command::Kind::UpdateWeights(update)) => {
                self.start_weight_update(front, update);
                return Ok(None);
            }
            Some(proto::request::wire::engine_command::Kind::CommitExperts(commit)) => {
                let order: BTreeMap<String, Vec<u32>> = commit
                    .order
                    .into_iter()
                    .map(|(module, order)| (module, order.experts))
                    .collect();
                let together = scheduler::HookBarrier::new(self.schedulers.len());
                let answers = self
                    .schedulers
                    .iter()
                    .map(|scheduler| scheduler.commit_experts(order.clone(), Arc::clone(&together)))
                    .collect();
                self.owe(front, commit.id, answers, Owing::Storage);
                return Ok(None);
            }
            // A warming reads how far the requests in flight reached the host KV cache, and limits
            // what they prefill meanwhile; both are every scheduler's own state.
            Some(proto::request::wire::engine_command::Kind::CacheWarm(warm)) => Some(
                proto::request::wire::from_engine::Kind::Hooked(self.warm_cache(warm)),
            ),
            Some(proto::request::wire::engine_command::Kind::FlushCache(_)) => {
                for scheduler in &self.schedulers {
                    scheduler.flush_cache();
                }
                return Ok(None);
            }
            // The worker era's park, resume and trial commit move a worker epoch between
            // processes. The epoch parks, resumes and commits through the process connection, in
            // front of the frontend rather than behind it.
            Some(proto::request::wire::engine_command::Kind::Park(park)) => Some(
                proto::request::wire::from_engine::Kind::Parked(proto::request::wire::Parked {
                    id: park.id,
                    error: EPOCH_LIFECYCLE.to_owned(),
                    index: Vec::new(),
                }),
            ),
            Some(proto::request::wire::engine_command::Kind::Resume(resume)) => Some(
                proto::request::wire::from_engine::Kind::Hooked(proto::request::wire::Hooked {
                    id: resume.id,
                    results: Vec::new(),
                    error: EPOCH_LIFECYCLE.to_owned(),
                }),
            ),
            Some(proto::request::wire::engine_command::Kind::Commit(commit)) => Some(
                proto::request::wire::from_engine::Kind::Hooked(proto::request::wire::Hooked {
                    id: commit.id,
                    results: Vec::new(),
                    error: EPOCH_LIFECYCLE.to_owned(),
                }),
            ),
            // The scheduling limits are the schedulers', applied at their next step through the
            // policy slot the process connection names.
            Some(proto::request::wire::engine_command::Kind::Configure(configure)) => {
                Some(proto::request::wire::from_engine::Kind::Configured(
                    proto::request::wire::Configured {
                        id: configure.id,
                        error: EPOCH_LIFECYCLE.to_owned(),
                    },
                ))
            }
            // The storage ledgers and the transfer budgets are each node's daemon's, read over
            // its claim; an epoch's main process holds no claim of the deployment's.
            Some(proto::request::wire::engine_command::Kind::DescribeStorage(describe)) => Some(
                proto::request::wire::from_engine::Kind::Hooked(proto::request::wire::Hooked {
                    id: describe.id,
                    results: Vec::new(),
                    error: LEDGERS.to_owned(),
                }),
            ),
            Some(proto::request::wire::engine_command::Kind::ThrottleTransfers(throttle)) => Some(
                proto::request::wire::from_engine::Kind::Hooked(proto::request::wire::Hooked {
                    id: throttle.id,
                    results: Vec::new(),
                    error: LEDGERS.to_owned(),
                }),
            ),
            // What the prefix cache holds is answered here, from what this process announced.
            //
            // A standby that starts on these devices maps the serving epoch's segments read-only
            // (the epoch switch's precopy), so reading them moves nothing and is not a lifecycle
            // command. The answer is the description this process gave the daemon when it became
            // ready: the segments are the pool's, and they are the same for as long as the epoch
            // serves. Asking every scheduler afresh is what cannot be done from here - a scheduler
            // holding paused streams does not answer until its stall bound, and a control thread
            // waiting on one cannot serve the handover's own commands (`501fa5e6f`).
            Some(proto::request::wire::engine_command::Kind::DescribeCache(describe)) => {
                let (error, segments) = match &self.announced {
                    Some(info) => (String::new(), info.segments.clone()),
                    None => (
                        "this process has announced no cache segments: it has not become ready"
                            .to_owned(),
                        Vec::new(),
                    ),
                };
                Some(proto::request::wire::from_engine::Kind::CacheDescribed(
                    proto::request::wire::CacheDescribed {
                        id: describe.id,
                        error,
                        segments,
                        storage_epoch: self
                            .storage
                            .first()
                            .map_or(0, |storage| storage.storage_epoch()),
                    },
                ))
            }
            Some(proto::request::wire::engine_command::Kind::DescribeRequests(describe)) => {
                Some(proto::request::wire::from_engine::Kind::RequestsDescribed(
                    proto::request::wire::RequestsDescribed {
                        id: describe.id,
                        error: EPOCH_LIFECYCLE.to_owned(),
                        requests: Vec::new(),
                    },
                ))
            }
            None => None,
        };
        if let Some(kind) = refused {
            self.send_event(front, kind);
        }
        Ok(None)
    }

    /// Answers one operation of an epoch's warming from every scheduler.
    ///
    /// Only the operations that stay inside a scheduler are this process's: the copies of the
    /// worker era's warming ran on its workers' hooks, and an epoch's pages reach a target through
    /// the host KV cache. The operations asked here take no fence, so a scheduler answers at its
    /// next pass; the bound is the one a scheduler that stopped answering is named by.
    fn warm_cache(
        &mut self,
        warm: proto::request::wire::CacheWarm,
    ) -> proto::request::wire::Hooked {
        use storage::manifest::cache::Operation;
        let hooked = |results: Vec<String>, error: String| proto::request::wire::Hooked {
            id: warm.id,
            results,
            error,
        };
        let operation: Operation = match serde_json::from_slice(&warm.operation) {
            Ok(operation) => operation,
            Err(error) => return hooked(Vec::new(), error.to_string()),
        };
        if !matches!(operation, Operation::Prefill { .. } | Operation::Warmed) {
            return hooked(Vec::new(), PAGES_THROUGH_HOST.to_owned());
        }
        let together = scheduler::HookBarrier::new(self.schedulers.len());
        let answers: Vec<_> = self
            .schedulers
            .iter()
            .map(|scheduler| scheduler.cache_warm(operation.clone(), Arc::clone(&together)))
            .collect();
        let mut results = Vec::with_capacity(answers.len());
        for answer in answers {
            match answer.recv_timeout(WARMING_ANSWER) {
                Ok(Ok(result)) => results.extend(result),
                Ok(Err(error)) => return hooked(Vec::new(), error),
                Err(error) => {
                    return hooked(
                        Vec::new(),
                        format!("a scheduler did not answer the warming: {error}"),
                    );
                }
            }
        }
        hooked(results, String::new())
    }

    /// Sends one message of the request interface to the daemon.
    pub(super) fn send_event(
        &mut self,
        front: usize,
        kind: proto::request::wire::from_engine::Kind,
    ) {
        let Some(connection) = self.control.front_mut(front) else {
            // The front this request belongs to detached. Its new owner names itself with a
            // `Takeover`, and until then nobody is reading: the request's events are kept for the
            // takeover to replay.
            tracing::debug!(
                front,
                "a request's front is gone; its events are kept for a takeover"
            );
            return;
        };
        if let Err(error) = connection.send(&proto::request::wire::FromEngine { kind: Some(kind) })
        {
            tracing::warn!(%error, front, "sending an event of an admitted request failed");
        }
    }

    /// Hands one event of a request's stream to its front, with the artifact bytes it travels with.
    ///
    /// A front on this host maps the shared-memory object, whose descriptor rides the event's own
    /// frame. A front on another host, or bytes that no descriptor names, takes bounded parts ahead
    /// of the event, and the daemon writes each part into the artifact's final host storage.
    fn send_stream_event(
        &mut self,
        front: usize,
        event: proto::request::wire::Event,
        media: Option<std::sync::Arc<dyn proto::request::ArtifactMedia>>,
    ) {
        use proto::request::wire::{ArtifactPart, FromEngine, from_engine::Kind};

        let Some(media) = media else {
            self.send_event(front, Kind::Event(event));
            return;
        };
        let Some(connection) = self.control.front_mut(front) else {
            tracing::debug!(
                front,
                "a request's front is gone; its artifact is kept for a takeover"
            );
            return;
        };
        let frame = FromEngine {
            kind: Some(Kind::Event(event)),
        };
        let descriptor = media
            .descriptor()
            .filter(|_| connection.peer_ip().is_none());
        let sent = match descriptor {
            Some(descriptor) => connection.send_with_descriptor(&frame, descriptor),
            None => media
                .bytes()
                .chunks(ARTIFACT_PART)
                .try_for_each(|data| {
                    connection.send(&FromEngine {
                        kind: Some(Kind::ArtifactPart(ArtifactPart {
                            size: media.bytes().len() as u64,
                            data: data.to_vec(),
                        })),
                    })
                })
                .and_then(|()| connection.send(&frame)),
        };
        if let Err(error) = sent {
            tracing::warn!(%error, front, "sending a request's artifact failed");
        }
    }

    /// Ends one request with its refusal: a submission this process will not admit is answered with
    /// the event that says so, because the daemon has no other way to learn it.
    pub(super) fn reject(
        &mut self,
        front: usize,
        request: u64,
        kind: proto::request::RejectionKind,
        message: String,
    ) {
        let event = proto::request::wire::Event {
            request,
            kind: Some(proto::request::wire::event::Kind::Rejected(
                proto::request::wire::Rejected {
                    kind: kind as i32,
                    message,
                },
            )),
        };
        self.send_event(front, proto::request::wire::from_engine::Kind::Event(event));
    }

    /// Carries every admitted request's events to the daemon, and forgets a request whose stream
    /// ended.
    ///
    /// Nothing here waits: a request that produced nothing this pass produces nothing, and the
    /// serve loop comes back to it.
    pub(super) fn forward_events(&mut self) {
        // The streams are taken out so that sending an event can borrow the process: the map is
        // small, and the survivors go back before the pass returns.
        let streams = std::mem::take(&mut self.streams);
        // Whether a handover is open, read once for the pass: a second front that has attached and
        // not yet taken its requests over. The journal's bound is wider while it is, because a
        // predecessor's consumer stops reading at the pause while this process keeps producing.
        let mut running = Vec::with_capacity(streams.len());
        for mut served in streams {
            let id = served.request;
            loop {
                match served.events.try_recv() {
                    Ok(event) => {
                        let terminal = matches!(
                            event,
                            scheduler::Event::Finished { .. }
                                | scheduler::Event::Rejected { .. }
                                | scheduler::Event::Error { .. }
                                | scheduler::Event::Migratable { .. }
                        );
                        let (wire, media) = event.to_wire(proto::request::RequestId(id));
                        // The position this event is written at, and the copy a takeover from an
                        // earlier position replays.
                        served.sent += 1;
                        keep_event(
                            &mut served.kept,
                            served.sent,
                            wire.clone(),
                            media.clone(),
                            HANDOVER_JOURNAL,
                        );
                        self.send_stream_event(served.front, wire, media);
                        if terminal {
                            break;
                        }
                    }
                    // A request that produced nothing yet keeps its stream.
                    Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                        running.push(served);
                        break;
                    }
                    // The scheduler that served the request is gone, which ends it.
                    Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break,
                }
            }
        }
        self.streams = running;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A takeover replays what the predecessor produced past the position the successor's consumer
    /// states, and nothing else.
    ///
    /// The pause freezes the predecessor's response, so its consumer stops advancing while this
    /// process goes on producing; without the replay those events are rendered nowhere and the
    /// response loses the tokens the pause covered.
    #[test]
    fn a_takeover_replays_only_the_events_past_the_position_it_names() {
        let event = |request: u64, message: &str| proto::request::wire::Event {
            request,
            kind: Some(proto::request::wire::event::Kind::Error(
                proto::request::wire::Failed {
                    message: message.to_owned(),
                },
            )),
        };
        let kept: std::collections::VecDeque<_> =
            (1..=4).map(|at| (at, event(9, "e"), None)).collect();

        assert_eq!(replay_after(&kept, 0).len(), 4, "a takeover from the start");
        assert_eq!(
            replay_after(&kept, 2).len(),
            2,
            "only the events past the position"
        );
        assert!(
            replay_after(&kept, 4).is_empty(),
            "a consumer at the last event has nothing left to replay"
        );
    }

    /// A takeover replays a backlog wider than the consumer's event window.
    ///
    /// The window is the credit a consumer reports in, not the distance it can
    /// be behind by: the daemon files this process's events without bound and
    /// writes them at the client's pace, so at a fast decode the body of a
    /// response that pauses is tens of events behind what was produced. A
    /// journal capped at the window drops the oldest of those and the handover
    /// loses exactly them.
    #[test]
    fn a_takeover_replays_a_backlog_wider_than_the_event_window() {
        let event = |request: u64, message: &str| proto::request::wire::Event {
            request,
            kind: Some(proto::request::wire::event::Kind::Error(
                proto::request::wire::Failed {
                    message: message.to_owned(),
                },
            )),
        };
        let backlog = proto::request::EVENT_WINDOW as usize * 3;
        let mut kept = std::collections::VecDeque::new();
        for at in 1..=backlog as u64 {
            keep_event(&mut kept, at, event(9, "e"), None, HANDOVER_JOURNAL);
        }

        assert!(
            kept.len() > proto::request::EVENT_WINDOW as usize,
            "the journal holds the whole backlog: {} of {backlog}",
            kept.len()
        );
        assert_eq!(
            replay_after(&kept, 1).len(),
            backlog - 1,
            "a takeover from the first event replays the backlog whole"
        );
    }
}
