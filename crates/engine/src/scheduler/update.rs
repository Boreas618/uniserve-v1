//! What returns to the scheduler, and how a request ends (`specs/scheduler-design.md` §3.5,
//! §3.7, §7.2, §7.3): a batch's output (`update`, vLLM's `update_from_output`), the buffer pool's
//! completions (`answered`), the end of a request (`finish`, `release`, `free`), and the messages
//! that end or advance a request from outside its batches (`cancel`, `acknowledge`, `consumed`).
//!
//! # A batch's output
//!
//! Each call's output is applied to its request in row order. A request that finished while the
//! call ran has its output dropped (vLLM's rule) and is released once its last call returns. A
//! failed call follows the error rules of SD §7.2; a token call's committed tokens are appended,
//! sent and checked against the stop conditions; a video call reports its progress, and the final
//! muxing call its artifact. `num_computed_tokens` advanced when the call was scheduled, so only a
//! verification's rejected drafts are taken back.
//!
//! # The buffer pool's completions
//!
//! `next_completion` (`mod.rs`) has matched each completion to its request, handed what gives
//! entries to the executors, and applied what it states of the cache (`learn`); `answered` applies
//! the rest to the request the owner is for. What an answer holds of the request's tables, and
//! the requests it calls for, are the memory rules' (`filled`, `memory.rs`); `answered` keeps the
//! request's slots, its admission's count of answers and its transfer.
//!
//! An admission (a request in `WaitingForRemoteKvs` that is still queued) counts its answers in
//! `pending`: every completion of one of its requests but a `Copy`, whose `Finish`'s answer is the
//! request's last, and each request an answer calls for joins it. When the last answer arrives
//! the request is admitted if its entries hold its first call and low watermark (`settled`);
//! otherwise the next step's `admit` asks for the rest in one more round, which makes the hit
//! final. A `Full` returns the request to the waiting queue, detached.
//!
//! # Ending a request
//!
//! `finish` decides a request's end and `release` delivers it: the terminal event is sent as soon
//! as it may be (a `Migratable` once no call of the request is in flight, a `Finished` under stop
//! strings once the daemon's decoder passed every output token), a preempted video request saves
//! its trajectory first, and once nothing of the request is in flight and its terminal is sent,
//! `free` gives up its blocks (each owner's `Detach`, through the executors) and the request is
//! forgotten.

use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use ipc::batch::{ArtifactHandle, Batch, BatchOutput, CallOutput, CallStatus, ErrorCode, Phase};
use ipc::{SharedMedia, SharedMediaError};
use proto::config::WorkerGroupRole;
use proto::request::{
    ArtifactEvent, EVENT_WINDOW, Event, FinishReason, MediaKind, MigrationCause, PositionLogprobs,
    RejectionKind, RequestId, StopReason, TokenLogprob,
};
use scheduling_guest::{End, Event as Change, Nanos, RequestId as PolicyRequestId};
use storage::Key;
use storage::runtime::{self, Cause, Completion, GroupId, OwnerId};

use super::{
    Error, Output, Request, RequestStatus, Scheduler, WARMUP_REQUEST_IDS, WorkerGroup,
    monotonic_nanos, unix_secs,
};

/// The message of a token call that failed for a reason other than its input, a numerical fault or
/// a fatal failure of its rank (`S/generation/mod.rs:737-742`).
const CALL_FAILED: &str = "worker status invalid: call_failed";

/// The message of a video request whose call failed or returned an output its call cannot return
/// (`S/execution/results.rs:185`).
const MEDIA_CALL_FAILED: &str = "media worker call failed";

/// A batch's output (SD §3.7).
impl Scheduler {
    /// Applies a batch's output, once, as it arrives: the group has room for one more batch, and
    /// each call's output is applied to its request in row order. An executor that failed the
    /// batch, or answered it with other calls than it was given, stops the scheduler: the requests
    /// of the epoch end as migratable with the cause `Loss` (SD §7.6). A call that computed
    /// non-finite logits fails its request, and the batch and those requests are reported as
    /// `Output::NumericalFault`.
    pub(super) fn update(&mut self, batch: Batch, output: Result<BatchOutput, Error>) {
        let group = self.group_index(batch.group);
        if let Some(group) = group {
            let group = &mut self.groups[group];
            group.in_flight = group.in_flight.saturating_sub(1);
        }

        let output = match output {
            Ok(output) if output.calls.len() == batch.requests.len() => output,
            Ok(_) => {
                return self.fail(Error::Executor(format!(
                    "batch {} returned other calls than it was given",
                    batch.id.0
                )));
            }
            Err(error) => return self.fail(error),
        };
        self.count_batch(&batch, &output);

        let mut faulted = Vec::new();
        for (row, (&id, call)) in batch.requests.iter().zip(output.calls).enumerate() {
            let Some(request) = self.requests.get_mut(&id) else {
                continue;
            };

            // The call is counted in index order, and the stage it held is given up once no
            // call of its phase remains in flight. A video call's output is judged on the counts
            // before it.
            let before = request
                .completed
                .get(batch.phase.index())
                .copied()
                .unwrap_or(0);
            let valid = media_valid(request, batch.phase, before, &call, &self.groups);
            let index = call_index(request, &batch, row, &self.groups);
            request.complete(batch.phase, index);
            if let Some(group) = group {
                self.groups[group].vacate(batch.phase, request);
            }

            // Cancelled, preempted or finished while the call ran: its output is dropped.
            if request.status == RequestStatus::Finished {
                self.release(id);
                continue;
            }
            if call.status == CallStatus::Error {
                if call.error_code == Some(ErrorCode::NumericalFault) {
                    faulted.push(id);
                }
                self.call_failed(id, &batch, row, &call);
                // A rank that can no longer serve stopped the scheduler: nothing more applies.
                if self.stopped {
                    return;
                }
                continue;
            }
            if !valid {
                self.fail_request(id, MEDIA_CALL_FAILED.to_owned());
                continue;
            }

            request.last_progress = Instant::now();
            self.mark(id);
            self.policy.events.push(Change::Advanced {
                request: PolicyRequestId(id.0),
                phase: batch.phase.id(),
                tokens: tokens_of(&batch, row),
            });

            match batch.phase {
                Phase::Prefill | Phase::Refill => self.update_prefill(id, &batch, row, call),
                Phase::Decode | Phase::Verify => self.update_decode(id, &batch, row, call),
                _ => self.update_media(id, &batch, row, call),
            }
        }

        if !faulted.is_empty() {
            // An engine that no longer listens has stopped this scheduler's epoch.
            let _ = self.output.send(Output::NumericalFault(batch.id, faulted));
        }
    }

    /// A prefill or refill chunk's output. A refill wrote the draft cache alone. A prefill chunk's
    /// complete blocks are committed and the blocks no later call reads are given up (SD §4.4);
    /// its prompt scores go to a request that asked for them; the chunk that ends the prompt
    /// samples the first token, and a request a prefill group served then moves to a decode group.
    fn update_prefill(&mut self, id: RequestId, batch: &Batch, row: usize, call: CallOutput) {
        if batch.phase == Phase::Refill {
            self.record_refill(id, batch, row);
            self.commit_filled(id, batch, row);
            return;
        }

        self.record_drafts(id, batch, row, call.tokens.len(), !call.tokens.is_empty());
        self.commit_filled(id, batch, row);
        self.give_up_behind(id, batch, row);
        if !self.prompt_scores(id, batch, row, &call) {
            return;
        }

        if !call.tokens.is_empty() {
            let eos = self.eos_on(batch.group);
            self.append_tokens(id, &call.tokens, &call, &eos);
        }
        if self.transfers(id) {
            self.start_transfer(id);
        }
    }

    /// A decode's or a verification's output: what it did not commit of the tokens it wrote was
    /// not computed (vLLM: `num_computed_tokens -= num_rejected`), its drafter's proposals are the
    /// next verification's, and its committed tokens are appended; then the blocks they filled are
    /// committed and those no later call reads given up. A call that committed no token resolves
    /// as the first end-of-sequence token (`S/output.rs:260`).
    fn update_decode(&mut self, id: RequestId, batch: &Batch, row: usize, call: CallOutput) {
        let eos = self.eos_on(batch.group);
        let tokens = match (call.tokens.is_empty(), eos.first()) {
            (true, Some(&end)) => vec![end],
            _ => call.tokens.clone(),
        };

        let Some(request) = self.requests.get_mut(&id) else {
            return;
        };
        let rejected = tokens_of(batch, row).saturating_sub(len32(tokens.len()));
        request.num_computed_tokens = request.num_computed_tokens.saturating_sub(rejected);
        request.spec_token_ids.clone_from(&call.draft_token_ids);
        self.count_speculation(batch, row, &call);
        self.record_drafts(id, batch, row, tokens.len(), true);

        self.append_tokens(id, &tokens, &call, &eos);
        if self.runs(id) {
            self.commit_filled(id, batch, row);
            self.give_up_behind(id, batch, row);
        }
    }

    /// A media call's output. A token request's image encoding holds the encoder output it
    /// computed, which is committed under its key (SD §4.3), and so does a video request's text
    /// encoding, whose output is its conditioning. A video request reports its progress after
    /// every applied call, and its final muxing call sends the artifact, then the request's end
    /// (SD §7.3 "Progress and artifact").
    fn update_media(&mut self, id: RequestId, batch: &Batch, row: usize, call: CallOutput) {
        let Some(request) = self.requests.get(&id) else {
            return;
        };
        if !request.is_video() {
            if matches!(batch.phase, Phase::VisionEncode | Phase::LatentEncode) {
                self.commit_encoding(id, batch, row);
                // The request holds the output now; its next encoding is of the next it lacks.
                if let Some(request) = self.requests.get_mut(&id) {
                    skip_held_encodings(request);
                }
            }
            return;
        }
        if batch.phase == Phase::TextEncode {
            self.commit_encoding(id, batch, row);
        }

        let Some(request) = self.requests.get(&id) else {
            return;
        };
        let progress = request.progress(batch.phase, &self.groups);
        let last = batch.phase == Phase::Mux
            && request.completed[Phase::Mux.index()] == request.needed(Phase::Mux, &self.groups);
        if let Some(event) = progress {
            self.send(id, event);
        }
        if !last {
            return;
        }

        let Some(published) = call.media_output else {
            return self.fail_request(id, "media output was not finalized".to_owned());
        };
        match open_artifact(&published) {
            Ok(media) => {
                let artifact = ArtifactEvent {
                    media_kind: MediaKind::Video,
                    content_type: "video/mp4".to_owned(),
                    media,
                };
                self.send(id, Event::Artifact(artifact));
                self.finish(id, self.finished(id, FinishReason::Completed, None));
            }
            // The client's message carries the failed step and its cause, and the call fails
            // (`S/execution/results.rs:57-72`).
            Err(error) => {
                let message = format!("{:#}", anyhow::Error::new(error));
                self.send(id, Event::ArtifactUnavailable { message });
                self.fail_request(id, MEDIA_CALL_FAILED.to_owned());
            }
        }
    }

    /// Appends the tokens a call committed, in order, and sends each one the request publishes;
    /// the first that ends the request finishes it, and the call's later tokens are dropped
    /// (SD §7.2 "Stop conditions", `S/output.rs:794-893`). A token's scores are sent when the
    /// request asks for them, cut to the token and the candidates it asks for. `eos` are the
    /// deployment's end-of-sequence tokens.
    fn append_tokens(&mut self, id: RequestId, tokens: &[u32], call: &CallOutput, eos: &[u32]) {
        let Some(request) = self.requests.get(&id) else {
            return;
        };
        let scored = request
            .generation()
            .filter(|input| input.sampling.generated_logprobs_requested())
            .map(|input| input.sampling.n_logprobs);

        for (at, &token) in tokens.iter().enumerate() {
            let Some(request) = self.requests.get_mut(&id) else {
                return;
            };
            request.token_ids.push(token);
            let stop = request.check_stop(token, eos);

            if request.publishes(token, stop.as_ref(), eos) {
                let candidates = scored.and_then(|count| {
                    let mut entries = call.token_logprobs.get(at)?.clone();
                    requested_candidates(&mut entries, count);
                    (!entries.is_empty()).then_some(entries)
                });
                let logprob = candidates
                    .as_ref()
                    .and_then(|entries| entries.first())
                    .map(|entry| entry.logprob);
                self.send(id, Event::TextToken { id: token, logprob });
                if let Some(candidates) = candidates {
                    self.send(
                        id,
                        Event::TokenLogprobs {
                            id: token,
                            candidates,
                        },
                    );
                }
                self.policy.events.push(Change::Emitted {
                    request: PolicyRequestId(id.0),
                    tokens: 1,
                    at: Nanos(monotonic_nanos(Instant::now())),
                });
            }

            if let Some((reason, stop_reason)) = stop {
                return self.finish(id, self.finished(id, reason, stop_reason));
            }
        }
    }

    /// A call that failed (SD §7.2 "A call's failure", `S/execution/failure.rs:16-114`): a rank
    /// that can no longer serve, or whose control-plane invariant broke, stops the scheduler; an
    /// input the rank refused rejects the request as invalid; anything else, a numerical fault
    /// included, fails the request with a message naming the call.
    fn call_failed(&mut self, id: RequestId, batch: &Batch, row: usize, call: &CallOutput) {
        if let Some(code) = call.error_code.filter(|code| code.fatal()) {
            let message = call.error_message.as_deref().unwrap_or("no message");
            return self.fail(Error::Executor(format!(
                "a worker's control-plane invariant failed in batch {}, call {row}: {code}: \
                 {message}",
                batch.id.0
            )));
        }

        let video = self.requests.get(&id).is_some_and(Request::is_video);
        if call.error_code == Some(ErrorCode::InputError) {
            // A video request whose rank states no reason fails as any media call does.
            let message = match (&call.error_message, video) {
                (Some(message), _) => message.clone(),
                (None, false) => "the worker refused the request's inputs".to_owned(),
                (None, true) => return self.fail_request(id, MEDIA_CALL_FAILED.to_owned()),
            };
            let rejection = Event::Rejected {
                kind: RejectionKind::Invalid,
                message,
            };
            return self.finish(id, rejection);
        }

        let message = if video {
            MEDIA_CALL_FAILED.to_owned()
        } else {
            format!(
                "{CALL_FAILED} (call {}.{row}, {:?})",
                batch.id.0, batch.phase
            )
        };
        self.fail_request(id, message);
    }

    /// Fails a request with `message`: a token request is sent `Error` at once and ends with
    /// `Finished { Error }`, a video request ends with the `Error` (`S/output.rs:976`, the old
    /// scheduler's `Terminal::Failed`).
    fn fail_request(&mut self, id: RequestId, message: String) {
        if self.requests.get(&id).is_some_and(Request::is_video) {
            return self.finish(id, Event::Error { message });
        }
        self.send(id, Event::Error { message });
        self.finish(id, self.finished(id, FinishReason::Error, None));
    }

    /// Sends the prompt scores a prefill chunk returned, to a request that asked for them, each
    /// position cut to its token and the candidates asked for (SD §7.2 "Prompt logprobs"). A chunk
    /// scores each of its tokens whose row names a target, so a prompt is scored at every
    /// position but its first; a chunk that returned another count fails the request
    /// (`S/output.rs:399-411`). False when it ended the request.
    fn prompt_scores(
        &mut self,
        id: RequestId,
        batch: &Batch,
        row: usize,
        call: &CallOutput,
    ) -> bool {
        let Some(count) = self
            .requests
            .get(&id)
            .and_then(Request::generation)
            .filter(|input| input.sampling.prompt_logprobs_requested())
            .map(|input| input.sampling.n_prompt_logprobs)
        else {
            return true;
        };

        let targets = batch
            .columns
            .targets
            .get(token_span(batch, row))
            .unwrap_or_default();
        let scored = targets.iter().filter(|target| **target >= 0).count();
        if call.prompt_logprobs.len() != scored {
            self.finish(id, self.finished(id, FinishReason::Error, None));
            return false;
        }
        if scored == 0 {
            return true;
        }

        let positions = call
            .prompt_logprobs
            .iter()
            .map(|entries| {
                let mut entries = entries.clone();
                requested_candidates(&mut entries, count);
                PositionLogprobs { entries }
            })
            .collect();
        self.send(id, Event::PromptLogprobs { positions });
        true
    }

    /// Advances a speculating request's draft cache over the entries call `row` of `batch` wrote
    /// (`S/speculation.rs:282-315`). An entry is the drafter's evaluation of the target's hidden
    /// state at a position with the token after it, so a call writes those of its positions whose
    /// next token it knows: a prefill chunk without image rows its tokens, but its last when it
    /// neither samples nor knows the token after it; a decode or a verification one per token it
    /// committed. The draft cache counts entries from its start, so a write past a gap, or past
    /// the draft blocks the request holds, counts nothing.
    fn record_drafts(
        &mut self,
        id: RequestId,
        batch: &Batch,
        row: usize,
        committed: usize,
        sampled: bool,
    ) {
        if !self.speculates() {
            return;
        }
        let span = token_span(batch, row);
        let Some(start) = batch
            .columns
            .rows
            .cache_positions
            .get(span.start)
            .and_then(|position| u32::try_from(*position).ok())
        else {
            return;
        };

        let tokens = len32(span.len());
        let written = match batch.phase {
            Phase::Prefill => {
                let images = batch
                    .columns
                    .vision
                    .iter()
                    .any(|block| span.contains(&(block.token as usize)));
                let known = sampled
                    || batch
                        .columns
                        .drafts
                        .get(row)
                        .is_some_and(|draft| draft.lookahead.is_some());
                if images {
                    0
                } else {
                    tokens - u32::from(!known && tokens > 0)
                }
            }
            Phase::Decode | Phase::Verify => len32(committed).max(1),
            _ => 0,
        };
        self.extend_drafts(id, start, start.saturating_add(written));
    }

    /// Advances a request's draft cache over the entries a refill chunk wrote: its tokens from the
    /// position it started at (`DraftWork::refill`).
    fn record_refill(&mut self, id: RequestId, batch: &Batch, row: usize) {
        let Some(request) = self.requests.get(&id) else {
            return;
        };
        let start = batch
            .columns
            .drafts
            .get(row)
            .and_then(|draft| draft.refill)
            .unwrap_or(request.num_draft_tokens);
        self.extend_drafts(id, start, start.saturating_add(tokens_of(batch, row)));
    }

    /// The draft cache's entries `[start, end)` are valid: it reaches `end` when it reached
    /// `start`, and the request holds draft blocks up to `end`.
    fn extend_drafts(&mut self, id: RequestId, start: u32, end: u32) {
        let Some(request) = self.requests.get(&id) else {
            return;
        };
        let held = self.draft_class(request).is_none_or(|(class, tokens)| {
            request
                .blocks
                .get(class)
                .is_some_and(|blocks| blocks.end.saturating_mul(tokens) >= end)
        });
        if let Some(request) = self.requests.get_mut(&id)
            && held
            && request.num_draft_tokens >= start
        {
            request.num_draft_tokens = request.num_draft_tokens.max(end);
        }
    }
}

/// The ends of a request (SD §3.7, §3.8).
impl Scheduler {
    /// Ends a request with `terminal`, once: it leaves the queues, the policy is told, and its
    /// terminal and its blocks are given up as soon as each may be (`release`). A request still
    /// waiting, or admitting, has nothing in flight and is gone at once.
    pub(super) fn finish(&mut self, id: RequestId, terminal: Event) {
        let Some(request) = self.requests.get_mut(&id) else {
            return;
        };
        if request.status == RequestStatus::Finished {
            return;
        }
        request.status = RequestStatus::Finished;
        request.queued = false;
        self.policy.events.push(Change::Finished {
            request: PolicyRequestId(id.0),
            end: end_of(&terminal),
        });
        request.terminal = Some(terminal);

        self.waiting.retain(|waiting| *waiting != id);
        self.running.retain(|running| *running != id);
        self.release(id);
    }

    /// Sends a finished request's terminal event and frees what it holds, each as soon as it may:
    /// called by `finish`, by `update` when a finished request's call returns, by `acknowledge`
    /// and by `cancel`.
    ///
    /// A preempted video request with a committed step first saves its trajectory, once, and its
    /// terminal waits for the answer (`saved`). A `Migratable` waits for the calls in flight,
    /// since the request may come back under the same id; a `Finished` under stop strings waits
    /// for the daemon's decoder to pass every output token, which may turn it into a `Stop`. Once
    /// no call of the request is in flight and its terminal is sent, its blocks are given up and
    /// the request is forgotten.
    pub(super) fn release(&mut self, id: RequestId) {
        let Some(request) = self.requests.get(&id) else {
            return;
        };
        let idle = request.idle();

        // A trajectory the scheduler cannot size leaves with no stored step.
        if idle
            && request.stores_trajectory()
            && let Some(save) = self.save_of(request)
        {
            if !self.saving(&save) {
                self.submit(save);
            }
            return;
        }

        let held = match &request.terminal {
            Some(Event::Migratable { .. }) => !idle,
            Some(Event::Finished { .. }) => self.awaits_acknowledgement(request),
            _ => false,
        };
        if held {
            return;
        }

        // `free` reads the terminal, which commits a migratable request's tails, so the blocks
        // are given up before it is sent.
        if idle {
            self.free(id);
        }
        if let Some(event) = self
            .requests
            .get_mut(&id)
            .and_then(|request| request.terminal.take())
        {
            self.send(id, event);
        }
        if idle {
            self.forget(id);
        }
    }

    /// Gives up everything a request holds, once no call of it is in flight (SD §3.7): for a
    /// request whose terminal, not yet sent, is a `Migratable`, the `Commit` of each partial last
    /// block under its tail key first (§4.3); then the pins `cache` counts for it, and each
    /// owner's `Detach`, which every executor whose device block tables hold the owner's entries
    /// stamps (§3.1). A `Detach` the buffer pool takes while a `Commit` waits for its stamps does
    /// not cancel the commit (`specs/buffer-pool.md` §9, R6).
    pub(super) fn free(&mut self, id: RequestId) {
        let migrates = self
            .requests
            .get(&id)
            .is_some_and(|request| matches!(request.terminal, Some(Event::Migratable { .. })));
        if migrates {
            self.commit_tails(id);
        }
        self.detach(id);
    }

    /// Gives up the pins `cache` counts for request `id` and submits each of its owners'
    /// `Detach`, which every executor whose device block tables hold the owner's entries stamps
    /// (§3.1). `submit` finds those groups from the request's attachments, which the caller
    /// clears only after this.
    fn detach(&mut self, id: RequestId) {
        let Some(request) = self.requests.get(&id) else {
            return;
        };
        let keys = held_keys(request);
        let owners: Vec<OwnerId> = request
            .attachments
            .iter()
            .map(|&(_, owner, _, _)| owner)
            .collect();

        self.unpin(id, &keys);
        for owner in owners {
            self.submit(runtime::Request::Detach {
                owner,
                after: Vec::new(),
            });
        }
    }

    /// Ends a preempted video request that saved its trajectory: `stored` when its `Save` was
    /// answered `Saved`, and the request leaves with that step (`denoised_steps`, with no layout:
    /// the key finds the trajectory under any layout); with any other answer it leaves with no
    /// stored step (SD §3.8). Either way its terminal is sent, its blocks are given up and it is
    /// forgotten.
    pub(super) fn saved(&mut self, id: RequestId, stored: bool) {
        let Some(request) = self.requests.get_mut(&id) else {
            return;
        };
        let steps = if stored {
            request.completed[Phase::Denoise.index()]
        } else {
            0
        };
        if let Some(Event::Migratable {
            denoised_steps,
            stored_layout,
            ..
        }) = &mut request.terminal
        {
            *denoised_steps = steps;
            stored_layout.clear();
        }

        self.free(id);
        if let Some(event) = self
            .requests
            .get_mut(&id)
            .and_then(|request| request.terminal.take())
        {
            self.send(id, event);
        }
        self.forget(id);
    }

    /// Ends a request on the daemon's word (SD §3.8): a waiting one is gone at once, a running one
    /// ends now and its calls in flight run on with their outputs dropped. A stop string the
    /// daemon's decoder matched after the request finished by itself turns the `Finished` it holds
    /// into a `Stop`, which is then sent.
    pub(super) fn cancel(&mut self, id: RequestId, reason: FinishReason) {
        let Some(request) = self.requests.get_mut(&id) else {
            return;
        };
        if request.status != RequestStatus::Finished {
            return self.finish(id, self.finished(id, reason, None));
        }

        if reason == FinishReason::Stop
            && let Some(Event::Finished {
                reason: held,
                stop_reason,
                ..
            }) = &mut request.terminal
        {
            *held = FinishReason::Stop;
            *stop_reason = None;
            self.release(id);
        }
    }

    /// The output tokens of a request the daemon's decoder has passed: a finished request whose
    /// terminal waited for them may end, and a running one may pipeline again.
    pub(super) fn acknowledge(&mut self, id: RequestId, tokens: u32) {
        let Some(request) = self.requests.get_mut(&id) else {
            return;
        };
        request.num_acknowledged_tokens = request.num_acknowledged_tokens.max(tokens);
        if request.status == RequestStatus::Finished {
            self.release(id);
        } else {
            self.mark(id);
        }
    }

    /// Events of a request its consumer took: their credit returns, up to the window.
    pub(super) fn consumed(&mut self, id: RequestId, events: u32) {
        let window = u32::try_from(EVENT_WINDOW).unwrap_or(u32::MAX);
        if let Some(request) = self.requests.get_mut(&id) {
            request.event_credit = request.event_credit.saturating_add(events).min(window);
        }
    }

    /// Sends a request's event to the daemon, one event of its consumer's credit. The events of a
    /// warm-up request are dropped (SD §3.2).
    pub(super) fn send(&mut self, id: RequestId, event: Event) {
        if id.0 >= WARMUP_REQUEST_IDS {
            return;
        }
        if let Some(request) = self.requests.get_mut(&id) {
            request.event_credit = request.event_credit.saturating_sub(1);
        }
        // An engine that no longer listens has stopped this scheduler's epoch.
        let _ = self.output.send(Output::Event(id, event));
    }

    /// The `Finished` event of request `id` for `reason` (SD §2.2 usage): its prompt as it was
    /// submitted, and for a token request admitted here every token it generated, those a resumed
    /// request brought included; a request that never left the waiting queue generated none here.
    pub(super) fn finished(
        &self,
        id: RequestId,
        reason: FinishReason,
        stop_reason: Option<StopReason>,
    ) -> Event {
        let (prompt_tokens, completion_tokens) =
            self.requests
                .get(&id)
                .map_or((0, 0), |request| match &request.input {
                    proto::request::Request::Generation(input) => {
                        let generated = if request.queued {
                            0
                        } else {
                            request.generated() as usize
                        };
                        (input.prompt_token_ids.len(), generated)
                    }
                    proto::request::Request::Diffusion(input) => (input.prompt_token_ids.len(), 0),
                });
        Event::Finished {
            reason,
            stop_reason,
            prompt_tokens,
            completion_tokens,
        }
    }

    /// Returns an admission to the waiting queue at its place (SD §3.5 `Full`, §3.6): placed
    /// nowhere, with nothing held, scheduled or counted. Its pins are given up and every owner of
    /// the admission is detached, so every later completion of the admission is dropped; its next
    /// admission draws new owners.
    pub(super) fn requeue(&mut self, id: RequestId) {
        self.detach(id);
        self.leave_groups(id);

        if let Some(request) = self.requests.get_mut(&id) {
            request.status = RequestStatus::Waiting;
            request.pending = 0;
            request.placement = [None; Phase::COUNT];
            request.attachments.clear();
            request.blocks.clear();
            request.num_cached_tokens = 0;
            request.num_host_tokens = 0;
            request.num_peer_tokens = 0;
            request.num_encoded_inputs = 0;
            for encoding in &mut request.encodings {
                encoding.1 = None;
            }
            request.unpinned.clear();
            // A restored trajectory is restored again by the next admission.
            request.scheduled[Phase::Denoise.index()] = 0;
            request.completed[Phase::Denoise.index()] = 0;
        }
        self.mark(id);
    }

    /// Takes a request out of the scheduler: off every group it is placed on, and its owners out
    /// of `owners`.
    fn forget(&mut self, id: RequestId) {
        self.leave_groups(id);
        if let Some(request) = self.requests.remove(&id) {
            for (_, owner, _, _) in &request.attachments {
                self.owners.remove(owner);
            }
        }
        self.waiting.retain(|waiting| *waiting != id);
        self.running.retain(|running| *running != id);
    }

    /// Takes a request off each group it is placed on, once per group, as its `placement` reads.
    fn leave_groups(&mut self, id: RequestId) {
        let Some(request) = self.requests.get(&id) else {
            return;
        };
        let mut placed: Vec<GroupId> = Vec::new();
        for group in request.placement.iter().flatten() {
            if !placed.contains(group) {
                placed.push(*group);
            }
        }
        for group in placed {
            if let Some(index) = self.groups.iter().position(|entry| entry.id() == group) {
                self.groups[index].leave(request);
            }
        }
    }

    /// Whether a finished request's `Finished` waits for the daemon's decoder (SD §7.2 "Stop
    /// strings", `S/control.rs:838-856`): the request has stop strings, ended by itself rather
    /// than by an error, a cancel or the decoder's own stop, and has published output tokens the
    /// decoder has not passed. A final end-of-sequence token, and a stop token without
    /// `include_stop_token`, were never published.
    fn awaits_acknowledgement(&self, request: &Request) -> bool {
        let Some(input) = request.generation() else {
            return false;
        };
        let Some(Event::Finished {
            reason,
            stop_reason,
            ..
        }) = &request.terminal
        else {
            return false;
        };
        let decoded = matches!(
            reason,
            FinishReason::Error | FinishReason::Cancelled | FinishReason::Aborted
        ) || (*reason == FinishReason::Stop && stop_reason.is_none());
        if input.stop_strings.is_empty() || decoded {
            return false;
        }

        let output = len32(request.token_ids.len()).saturating_sub(request.num_prompt_tokens);
        let eos = self.eos_of(request);
        let end = (reason.clone(), stop_reason.clone());
        let unpublished = request
            .token_ids
            .last()
            .filter(|_| output > 0)
            .is_some_and(|&last| !request.publishes(last, Some(&end), &eos));
        output.saturating_sub(u32::from(unpublished)) > request.num_acknowledged_tokens
    }

    /// The `Save` of a preempted video request's trajectory at its committed step, for its owner
    /// on the group that denoises it, sized in units of the host latent region (SD §3.8, §4.4).
    fn save_of(&self, request: &Request) -> Option<runtime::Request> {
        Some(runtime::Request::Save {
            owner: request.owner_on(Phase::Denoise)?,
            key: request.trajectory_key(self.latent_seed)?,
            units: self.trajectory_units(request)?,
        })
    }

    /// Whether a `Save` for the owner `save` names is held or outstanding: one `Save` per request.
    fn saving(&self, save: &runtime::Request) -> bool {
        let runtime::Request::Save { owner, .. } = save else {
            return false;
        };
        let of = |request: &runtime::Request| match request {
            runtime::Request::Save { owner: other, .. } => other == owner,
            _ => false,
        };
        self.held.iter().any(of) || self.outstanding.values().any(|(_, request)| of(request))
    }
}

/// The buffer pool's completions (SD §3.5 `answered`, §4.1).
impl Scheduler {
    /// Applies one completion, as `next_completion` returned it: the number of the request it
    /// answers, the owner that request acts for, the request and the completion. A refusal that
    /// is no miss names something the scheduler chose from its own counts, which is its defect:
    /// the scheduler stops. A block committed, or stored on the host, is stored on as the write
    /// policy says (`write_policy`), whether or not its owner is still live. Completions of
    /// requests that act for no owner change the scheduler's regions and its admissions'
    /// prefetches; the rest apply to the request the owner is for, as an admission's, a
    /// transfer's or a running request's answer.
    pub(super) fn answered(
        &mut self,
        _: runtime::RequestId,
        owner: Option<OwnerId>,
        request: runtime::Request,
        completion: Completion,
    ) {
        if let Completion::Refused { cause } = completion
            && !refusal_is_miss(&request, cause)
        {
            return self.fail(Error::Storage(format!(
                "the buffer pool refused {request:?} for its {cause:?}"
            )));
        }

        if let Some(store) = self.write_policy(&request, &completion) {
            self.submit(store);
        }
        match (&request, &completion) {
            (runtime::Request::Resize { region, .. }, Completion::Resized { units, .. }) => {
                return self.resized(*region, *units);
            }
            (runtime::Request::Prefetch { .. }, Completion::Prefetched { .. }) => {
                return self.prefetched(&request, &completion);
            }
            _ => {}
        }

        let Some(owner) = owner else {
            return;
        };
        let Some(&id) = self.owners.get(&owner) else {
            return;
        };
        self.fill(id, owner, &request, &completion);
        // What the answer holds of the request's tables, and the requests it calls for (§4.4).
        // An encoder output a `Pin` found is held, and the next encoding is of one it lacks.
        let follow = self.filled(id, &request, &completion);
        if let (runtime::Request::Pin { .. }, Some(row)) = (&request, self.requests.get_mut(&id)) {
            skip_held_encodings(row);
        }

        let Some(row) = self.requests.get(&id) else {
            return;
        };
        match row.status {
            RequestStatus::WaitingForRemoteKvs if row.queued => {
                self.admission_answered(id, owner, &request, &completion, follow);
            }
            RequestStatus::WaitingForRemoteKvs => {
                self.transfer_answered(id, owner, &request, &completion);
            }
            _ => {
                for ask in follow {
                    self.submit(ask);
                }
                self.running_answered(id, &request, &completion);
            }
        }
    }

    /// The last answer of an admission arrived and its entries hold what its first call needs: it
    /// runs from its hit, and its draft cache from the draft blocks its hit pinned (SD §3.5,
    /// §4.4); its queue wait and hit are counted (§7.5).
    pub(super) fn admitted(&mut self, id: RequestId) {
        let now = Instant::now();
        let drafted = self
            .requests
            .get(&id)
            .map_or(0, |request| self.draft_tokens(request));
        let Some(request) = self.requests.get_mut(&id) else {
            return;
        };
        request.num_computed_tokens = request.num_cached_tokens;
        request.num_draft_tokens = drafted;
        request.status = RequestStatus::Running;
        request.queued = false;
        request.last_progress = now;
        let queued_at = unix_secs(request.arrival);
        self.waiting.retain(|waiting| *waiting != id);
        self.running.push(id);

        self.count_admitted(id, now);
        self.send(
            id,
            Event::Scheduled {
                queued_at,
                scheduled_at: unix_secs(now),
            },
        );
        self.policy.events.push(Change::Admitted {
            request: PolicyRequestId(id.0),
        });
    }

    /// Starts moving a request a prefill group served to a decode group, once its prompt is
    /// computed and nothing of it is in flight (SD §3.5 "Prefill to decode transfer"): the decode
    /// group is placed as any phase is, and the request's owner on the prefill group is moved
    /// there by `Transfer`, which the prefill group's executor stamps. The request waits in
    /// `WaitingForRemoteKvs` for the answer, out of every batch and of admission. With no decode
    /// group able to take it now, it stays where it is and a later step tries again
    /// (`start_transfers`).
    pub(super) fn start_transfer(&mut self, id: RequestId) {
        let Some(request) = self.requests.get(&id) else {
            return;
        };
        let Some(owner) = request.owner_on(Phase::Prefill) else {
            return;
        };
        let candidates = self.candidates(request, Phase::Decode);
        let Some(target) = self.choose(request, &candidates, None) else {
            return;
        };
        let Some(index) = self.group_index(target) else {
            return;
        };
        let verifies = self.groups[index].serves(Phase::Verify);

        if let Some(request) = self.requests.get_mut(&id) {
            request.placement[Phase::Decode.index()] = Some(target);
            if verifies {
                request.placement[Phase::Verify.index()] = Some(target);
            }
            request.status = RequestStatus::WaitingForRemoteKvs;
            request.pending = 1;
        }
        self.groups[index].join();
        self.submit(runtime::Request::Transfer {
            owner,
            group: target,
            after: Vec::new(),
        });
        self.mark(id);
    }

    /// Starts the transfer of every running request a prefill group finished that no decode group
    /// could take when it finished: called once per pass of the loop, after the capacity words
    /// are read.
    pub(super) fn start_transfers(&mut self) {
        let finished: Vec<RequestId> = self
            .running
            .iter()
            .copied()
            .filter(|&id| self.transfers(id))
            .collect();
        for id in finished {
            self.start_transfer(id);
        }
    }

    /// Whether a request moves to a decode group now: it runs on a group of the `Prefill` role
    /// with no decode group placed, its prompt is computed and no call of it is in flight.
    fn transfers(&self, id: RequestId) -> bool {
        let Some(request) = self.requests.get(&id) else {
            return false;
        };
        let prefills = request.placement[Phase::Prefill.index()]
            .and_then(|group| self.group(group))
            .is_some_and(|group| group.config.role == WorkerGroupRole::Prefill);
        request.status == RequestStatus::Running
            && prefills
            && request.placement[Phase::Decode.index()].is_none()
            && request.num_computed_tokens >= request.num_prompt_tokens
            && request.idle()
    }

    /// What an answer states of the request beyond its tables, whatever the request is doing:
    /// the slot of an owner's `Attach`, the step a `Restore` brought back (SD §3.8), and the keys
    /// of the blocks a `Commit` or a `Load` left private to the request, answered `Duplicate`,
    /// which `cache` does not count it a holder of (BP §9, R4).
    fn fill(
        &mut self,
        id: RequestId,
        owner: OwnerId,
        request: &runtime::Request,
        completion: &Completion,
    ) {
        let Some(row) = self.requests.get_mut(&id) else {
            return;
        };
        match (request, completion) {
            (runtime::Request::Attach { .. }, Completion::Attached { slot }) => {
                for attachment in &mut row.attachments {
                    if attachment.1 == owner {
                        attachment.2 = *slot;
                    }
                }
            }
            (runtime::Request::Commit { key, .. }, Completion::Duplicate) => {
                row.unpinned.push(*key);
            }
            (runtime::Request::Load { keys, .. }, Completion::Duplicate) => {
                row.unpinned.extend_from_slice(keys);
            }
            // The slot holds the stored trajectory: denoising continues at its step.
            (runtime::Request::Restore { .. }, Completion::Restored) => {
                let steps = row
                    .video()
                    .map_or(0, |video| video.trajectory.denoised_steps);
                row.scheduled[Phase::Denoise.index()] = steps;
                row.completed[Phase::Denoise.index()] = steps;
            }
            _ => {}
        }
    }

    /// An answer to a request of an admission (SD §3.5): a `Full` returns the request to the
    /// waiting queue; the requests the answer calls for (`follow`, from `filled`) join the
    /// admission; and the last answer settles it.
    fn admission_answered(
        &mut self,
        id: RequestId,
        owner: OwnerId,
        request: &runtime::Request,
        completion: &Completion,
        follow: Vec<runtime::Request>,
    ) {
        if let Completion::Full { .. } = completion {
            // An owner whose `Attach` found no room was never created, and takes no `Detach`.
            if let runtime::Request::Attach { .. } = request {
                if let Some(row) = self.requests.get_mut(&id) {
                    row.attachments.retain(|attachment| attachment.1 != owner);
                }
                self.owners.remove(&owner);
            }
            return self.requeue(id);
        }

        for ask in follow {
            self.ask(id, ask);
        }
        // A copy's last answer is the answer to its `Finish`.
        if !matches!(completion, Completion::Copy { .. }) {
            self.answer_arrived(id);
        }
    }

    /// An answer to a transfer's `Transfer` (SD §3.5): `Transferred` makes the owner the
    /// request's attachment to the decode group, with its slot there, and the request runs;
    /// `Failed` leaves the owner attached to the prefill group, holding blocks no device block
    /// table holds, so it is detached with no stamp and the request ends as migratable with the
    /// cause `Loss`. Other answers of the owner (the commits of its last chunk) change nothing.
    fn transfer_answered(
        &mut self,
        id: RequestId,
        owner: OwnerId,
        request: &runtime::Request,
        completion: &Completion,
    ) {
        let runtime::Request::Transfer { group: target, .. } = *request else {
            return;
        };
        match completion {
            Completion::Transferred { slot, .. } => self.transferred(id, owner, target, *slot),
            Completion::Failed => {
                // The attachment goes first, so that `submit` finds no group to stamp the
                // `Detach` (the owner's entries left every device block table).
                if let Some(row) = self.requests.get_mut(&id) {
                    row.attachments.retain(|attachment| attachment.1 != owner);
                }
                self.submit(runtime::Request::Detach {
                    owner,
                    after: Vec::new(),
                });
                self.finish(id, migratable(MigrationCause::Loss));
            }
            _ => {}
        }
    }

    /// The owner moved to group `target` with slot `slot`: the request leaves the prefill group,
    /// whose placement entries it clears, and the decode group sees it for the first time, so its
    /// first decode carries a `NewSlot`.
    fn transferred(&mut self, id: RequestId, owner: OwnerId, target: GroupId, slot: u32) {
        let Some((source, _)) = self.requests.get(&id).and_then(|row| row.attachment(owner)) else {
            return;
        };
        let index = self.group_index(source);
        if let (Some(index), Some(row)) = (index, self.requests.get(&id)) {
            self.groups[index].leave(row);
        }

        let Some(row) = self.requests.get_mut(&id) else {
            return;
        };
        for placed in &mut row.placement {
            if *placed == Some(source) {
                *placed = None;
            }
        }
        for attachment in &mut row.attachments {
            if attachment.1 == owner {
                *attachment = (target, owner, slot, false);
            }
        }
        row.status = RequestStatus::Running;
        row.pending = 0;
        row.last_progress = Instant::now();
        self.mark(id);
    }

    /// An answer to a running or finished request: a running request's `Allocate` that found no
    /// room leaves its class short, and the request preempts (SD §3.6); a `Save` ends its
    /// request.
    fn running_answered(
        &mut self,
        id: RequestId,
        request: &runtime::Request,
        completion: &Completion,
    ) {
        use runtime::Request as R;

        match (request, completion) {
            (R::Allocate { class, .. }, Completion::Full { region }) => {
                let running = self.runs(id);
                if let Some(blocks) = self
                    .requests
                    .get_mut(&id)
                    .and_then(|row| row.blocks.get_mut(usize::from(*class)))
                {
                    blocks.asked = blocks.end;
                }
                let index = self
                    .region_index(*region)
                    .and_then(|index| u8::try_from(index).ok());
                if let (true, Some(index)) = (running, index) {
                    self.preempt(id, index);
                }
            }
            (R::Allocate { .. }, Completion::Allocated { .. }) => self.mark(id),
            (R::Save { .. }, Completion::Saved) => self.saved(id, true),
            (R::Save { .. }, Completion::Done | Completion::Full { .. }) => self.saved(id, false),
            _ => {}
        }
    }

    /// One more answer of an admission arrived. At the last, the request is admitted when its
    /// entries hold its first call and low watermark (`settled`); otherwise the next step's
    /// `admit` asks for the rest in one more round, which makes the hit final (§4.4).
    fn answer_arrived(&mut self, id: RequestId) {
        let Some(row) = self.requests.get_mut(&id) else {
            return;
        };
        row.pending = row.pending.saturating_sub(1);
        if row.pending > 0 {
            return;
        }

        let settled = self.requests.get(&id).is_some_and(|row| self.settled(row));
        if settled {
            self.admitted(id);
        } else {
            self.mark(id);
        }
    }

    /// A `Prefetch` ended (SD §3.5): the found keys are on the host (`learn` recorded them). An
    /// admission waiting for it is the one whose lookup keys of the class hold its first key; it
    /// counts the tokens a peers' `Prefetch` found (`filled`), and its next round loads them. A
    /// prefetch the policy asked for a waiting request answers no admission.
    fn prefetched(&mut self, request: &runtime::Request, completion: &Completion) {
        let runtime::Request::Prefetch { class, keys, .. } = request else {
            return;
        };
        let Some(first) = keys.first() else {
            return;
        };
        let id = self.waiting.iter().copied().find(|id| {
            self.requests.get(id).is_some_and(|row| {
                row.status == RequestStatus::WaitingForRemoteKvs
                    && row.pending > 0
                    && row
                        .keys
                        .get(usize::from(*class))
                        .is_some_and(|chain| chain.contains(first))
            })
        });
        let Some(id) = id else {
            return;
        };

        for ask in self.filled(id, request, completion) {
            self.ask(id, ask);
        }
        self.answer_arrived(id);
    }

    /// A region's units after a `Resize` (SD §3.8): the reply owed to a `Message::Resize` for the
    /// region, or, for the shrink of a drain, `Output::Shrunk`, which the engine divides among the
    /// target epoch's schedulers.
    fn resized(&mut self, region: u8, units: u32) {
        if let Some(index) = self.region_index(region)
            && let Some(capacity) = self.capacity.get_mut(index)
        {
            *capacity = units;
        }
        match self.resizing.iter().position(|(owed, _)| *owed == region) {
            Some(at) => {
                let (_, reply) = self.resizing.remove(at);
                // A requester that stopped waiting needs no reply.
                let _ = reply.send(());
            }
            None => {
                let _ = self.output.send(Output::Shrunk(region, units));
            }
        }
    }
}

/// Helpers of a request's token calls.
impl Scheduler {
    /// Whether the request is `Running`: admitted, and in no transfer.
    fn runs(&self, id: RequestId) -> bool {
        self.requests
            .get(&id)
            .is_some_and(|request| request.status == RequestStatus::Running)
    }

    /// Whether the deployment speculates: a group verifies drafts.
    fn speculates(&self) -> bool {
        self.groups.iter().any(|group| group.serves(Phase::Verify))
    }

    /// The end-of-sequence tokens the executors of `group` state.
    fn eos_on(&self, group: GroupId) -> Vec<u32> {
        self.group(group)
            .map(|group| group.capacity.eos_token_ids.clone())
            .unwrap_or_default()
    }

    /// The end-of-sequence tokens of the group that decodes `request`, or that prefilled it.
    fn eos_of(&self, request: &Request) -> Vec<u32> {
        let placed =
            request.placement[Phase::Decode.index()].or(request.placement[Phase::Prefill.index()]);
        match placed {
            Some(group) => self.eos_on(group),
            None => Vec::new(),
        }
    }
}

impl Request {
    /// Counts the applied output of call `index` of `phase` (SD §3.3): `completed` counts calls in
    /// index order, so an output ahead of the next uncounted call waits in `early` until every call
    /// before it is counted. An index already counted changes nothing.
    pub(super) fn complete(&mut self, phase: Phase, index: u32) {
        let Some(completed) = self.completed.get_mut(phase.index()) else {
            return;
        };
        if index > *completed {
            if !self.early.contains(&(phase, index)) {
                self.early.push((phase, index));
            }
            return;
        }
        if index < *completed {
            return;
        }

        *completed += 1;
        while let Some(at) = self
            .early
            .iter()
            .position(|&(early, call)| early == phase && call == *completed)
        {
            self.early.swap_remove(at);
            *completed += 1;
        }
    }
}

/// The `Migratable` a request leaves with for `cause`, naming no stored step: a preempted video
/// request's stored step is written into it when its `Save` is answered.
pub(super) fn migratable(cause: MigrationCause) -> Event {
    Event::Migratable {
        cause,
        denoised_steps: 0,
        stored_layout: Vec::new(),
    }
}

/// How the policy is told a request ended (`S/terminal.rs:79-93`).
fn end_of(terminal: &Event) -> End {
    match terminal {
        Event::Migratable { .. } => End::Migratable,
        Event::Rejected { .. } => End::Rejected,
        Event::Error { .. } => End::Failed,
        Event::Finished {
            reason: FinishReason::Cancelled,
            ..
        } => End::Cancelled,
        Event::Finished {
            reason: FinishReason::Error,
            ..
        } => End::Failed,
        _ => End::Completed,
    }
}

/// Whether a refusal is a miss the request goes on from (SD §3.5 `Refused`): a key `cache`
/// believed held that the buffer pool does not hold, for a `Clone` or a `Restore`; a host copy of
/// another layout, for a `Load` or a `Restore`; and a `List` whose listing exceeds what one
/// completion holds, after which the scheduler knows the cache as it did.
fn refusal_is_miss(request: &runtime::Request, cause: Cause) -> bool {
    matches!(
        (request, cause),
        (runtime::Request::Clone { .. }, Cause::Key)
            | (runtime::Request::Restore { .. }, Cause::Key | Cause::Class)
            | (runtime::Request::Load { .. }, Cause::Class)
            | (runtime::Request::List, Cause::Request)
    )
}

/// The keys of the shared blocks a request holds, whose pins `cache` counts: per cache group (class
/// `c` is cache group `c`), its key chain's blocks from `first` to `shared`, and every encoder
/// output it holds.
fn held_keys(request: &Request) -> Vec<Key> {
    let mut keys = Vec::new();
    for (blocks, chain) in request.blocks.iter().zip(&request.keys) {
        let end = (blocks.shared as usize).min(chain.len());
        let start = (blocks.first as usize).min(end);
        keys.extend_from_slice(&chain[start..end]);
    }
    keys.extend(
        request
            .encodings
            .iter()
            .filter(|(_, position)| position.is_some())
            .map(|(key, _)| *key),
    );
    keys
}

/// Moves a token request's next encoding to the first encoder input whose output it lacks.
fn skip_held_encodings(request: &mut Request) {
    let next = request
        .encodings
        .iter()
        .position(|(_, position)| position.is_none())
        .unwrap_or(request.encodings.len());
    request.num_encoded_inputs = len32(next);
}

/// The index of call `row` of `batch` among its request's calls of the batch's phase: a video
/// decode or encode round and a denoising step state their own (the round's first unit
/// `MediaRow::unit`, `step`); every other call returns in the order it was scheduled, after the
/// ones counted.
fn call_index(request: &Request, batch: &Batch, row: usize, groups: &[WorkerGroup]) -> u32 {
    match (batch.phase, batch.media.get(row)) {
        (Phase::VideoDecode | Phase::VideoEncode, Some(media)) => {
            request.round_of(media.unit, groups)
        }
        (Phase::Denoise, Some(media)) => media.step,
        (phase, _) => request.completed.get(phase.index()).copied().unwrap_or(0),
    }
}

/// Whether the output of a video request's call of `phase` is one its call can return (SD §7.3
/// "Denoising", `S/execution/results.rs:120-143`), judged on `completed`, the calls of the phase
/// counted before it: only the final muxing call carries the artifact, a denoising step advances
/// the trajectory past the steps completed, and a latent preparation opens it at the step its
/// admission restored (0 from noise). Every call of a token request is.
fn media_valid(
    request: &Request,
    phase: Phase,
    completed: u32,
    call: &CallOutput,
    groups: &[WorkerGroup],
) -> bool {
    if !request.is_video() || call.status != CallStatus::Ok {
        return true;
    }
    let last =
        phase == Phase::Mux && completed.saturating_add(1) >= request.needed(Phase::Mux, groups);
    if call.media_output.is_some() && !last {
        return false;
    }
    match phase {
        Phase::Denoise => call.step_index > completed,
        Phase::LatentPrepare => call.step_index == request.completed[Phase::Denoise.index()],
        _ => true,
    }
}

/// The token rows of call `row` of a token batch, as indices into its token columns: from the
/// query offset of its first row to that of the row after its last. Empty for a media call.
fn token_span(batch: &Batch, row: usize) -> Range<usize> {
    let rows = &batch.columns.rows;
    let offset = |index: u32| {
        rows.query_offsets
            .get(index as usize)
            .and_then(|offset| usize::try_from(*offset).ok())
    };
    rows.calls
        .get(row)
        .and_then(|call| Some(offset(call.rows.start)?..offset(call.rows.end)?))
        .unwrap_or(0..0)
}

/// The tokens call `row` of `batch` took from its step.
fn tokens_of(batch: &Batch, row: usize) -> u32 {
    len32(token_span(batch, row).len())
}

/// Keeps of one scored position the candidates its request asks for (`S/completion.rs:232-244`):
/// the position's own token, then the first `count` of the position's most likely tokens without
/// the token itself, which is vLLM's set (`{token} ∪ top-count`, the token first). A rank reports
/// the deployment's `max_logprobs` candidates whatever the request asked.
fn requested_candidates(entries: &mut Vec<TokenLogprob>, count: u32) {
    let Some(own) = entries.first().map(|entry| entry.token_id) else {
        return;
    };
    let mut kept = 1;
    for index in 1..entries.len().min(1 + count as usize) {
        if entries[index].token_id != own {
            entries.swap(kept, index);
            kept += 1;
        }
    }
    entries.truncate(kept);
}

/// Opens the artifact a final muxing call published: the scheduler is the one opener of the
/// name, which it unlinks, and it maps the bytes.
fn open_artifact(
    published: &ipc::batch::MediaOutput,
) -> Result<Arc<SharedMedia>, SharedMediaError> {
    let ArtifactHandle::PosixShm { name } = &published.handle;
    // SAFETY: the media host wrote the object's bytes before it relinquished the object and never
    // writes or resizes it after, which `SharedMedia::open` requires; the scheduler opens each
    // published name once, from the one output that names it.
    let media = unsafe { SharedMedia::open(name, published.bytes)? };
    Ok(Arc::new(media))
}

/// A count as the `u32` the request's counters hold, saturated.
fn len32(count: usize) -> u32 {
    u32::try_from(count).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    //! `update`, `answered` and the ends of a request over the scheduler of
    //! `scheduler::tests::bench`: request 1, attached to group A in slot row 3 under `OWNER_A`,
    //! with a prompt of three tokens and room for four more. Its requests hold no keys and no
    //! blocks, so no call commits or gives up a block.

    use crossbeam_channel::Receiver;
    use ipc::batch::{
        Batch, BatchId, BatchOutput, CallOutput, CallStatus, Columns, ErrorCode, Extent, Phase,
        Row, Shape, TimingCounters, Variant,
    };
    use proto::config::WorkerGroupRole;
    use proto::request::{
        Event, FinishReason, MigrationCause, RejectionKind, RequestId, StopReason, TokenLogprob,
    };
    use storage::runtime::{self, Cause, Completion, OwnerId};

    use super::super::cache::DEVICE;
    use super::super::tests::{A, B, Bench, OWNER_A, bench};
    use super::super::{Blocks, Cached, Output, RequestStatus};

    const ID: RequestId = RequestId(1);
    const EOS: u32 = 2;
    const OWNER_B: OwnerId = OwnerId(11);

    /// The bench, with the scheduler's outputs to read.
    fn scheduler() -> (Bench, Receiver<Output>) {
        let mut bench = bench(4096);
        let (output, outputs) = crossbeam_channel::unbounded();
        bench.scheduler.output = output;
        (bench, outputs)
    }

    /// Request 1 running on group A, which serves `phases`, with its prompt computed.
    fn running(bench: &mut Bench, phases: &[Phase]) {
        bench.attach(&[(A, OWNER_A, 3)]);
        let scheduler = &mut bench.scheduler;
        let group = &mut scheduler.groups[0];
        group.capacity.phases = phases.iter().map(|phase| 1 << phase.index()).sum();
        group.capacity.eos_token_ids = vec![EOS];

        let request = scheduler.requests.get_mut(&ID).unwrap();
        request.status = RequestStatus::Running;
        request.queued = false;
        for phase in phases {
            request.placement[phase.index()] = Some(A);
        }
        request.num_computed_tokens = 3;
        scheduler.running.push(ID);
    }

    /// Hands `calls` calls of `phase` of request 1 to group A, each of `tokens` tokens, as
    /// `schedule` does.
    fn schedule(bench: &mut Bench, phase: Phase, calls: u32, tokens: u32) {
        let request = bench.scheduler.requests.get_mut(&ID).unwrap();
        request.scheduled[phase.index()] += calls;
        request.num_computed_tokens += calls * tokens;
        bench.scheduler.groups[0].in_flight += calls;
    }

    /// A batch of group A with request 1's call of `phase`: `tokens` tokens from cache position
    /// `start`.
    fn batch(phase: Phase, start: u32, tokens: u32) -> Batch {
        let mut columns = Columns::default();
        columns.rows.calls.push(Row {
            slot: 3,
            kind: phase,
            cache_group: 0,
            rows: 0..1,
        });
        columns.rows.query_offsets = vec![0, tokens as i32];
        columns.rows.positions = (start..start + tokens).map(|at| at as i32).collect();
        columns.rows.cache_positions = columns.rows.positions.clone();
        columns.rows.seq_lens = vec![(start + tokens) as i32];
        columns.rows.logits_rows = vec![tokens as i32 - 1];
        columns.input_ids = vec![1; tokens as usize];
        Batch {
            id: BatchId(7),
            group: A,
            phase,
            generation: 0,
            shape: Shape {
                rows: 1,
                tokens,
                variant: Variant::DecodeGreedy,
                extent: Extent::None,
            },
            requests: vec![ID],
            columns,
            media: Vec::new(),
            new_slots: Vec::new(),
        }
    }

    fn output(tokens: &[u32]) -> CallOutput {
        CallOutput {
            status: CallStatus::Ok,
            error_code: None,
            error_message: None,
            tokens: tokens.to_vec(),
            step_index: 0,
            cache_computed_len: 0,
            cache_visible_len: 0,
            token_logprobs: Vec::new(),
            prompt_logprobs: Vec::new(),
            media_output: None,
            draft_token_ids: Vec::new(),
        }
    }

    fn failed(code: ErrorCode, message: Option<&str>) -> CallOutput {
        CallOutput {
            status: CallStatus::Error,
            error_code: Some(code),
            error_message: message.map(str::to_owned),
            ..output(&[])
        }
    }

    /// Applies `call` as the output of `batch`.
    fn apply(bench: &mut Bench, batch: Batch, call: CallOutput) {
        let output = BatchOutput {
            calls: vec![call],
            timing: TimingCounters::default(),
            forward_stats: None,
        };
        bench.scheduler.update(batch, Ok(output));
    }

    /// The events sent to request 1 so far.
    fn events(outputs: &Receiver<Output>) -> Vec<Event> {
        outputs
            .try_iter()
            .filter_map(|output| match output {
                Output::Event(id, event) if id == ID => Some(event),
                _ => None,
            })
            .collect()
    }

    /// Writes the requests the scheduler holds, answers them in order with `completions`, and
    /// applies the answers; returns the requests written.
    fn answer(bench: &mut Bench, completions: &[Completion]) -> Vec<runtime::Request> {
        bench.scheduler.post().unwrap();
        let taken = bench.take();
        for ((id, _), completion) in taken.iter().zip(completions) {
            bench.complete(*id, completion);
        }
        while let Some((id, owner, request, completion)) =
            bench.scheduler.next_completion().unwrap()
        {
            bench.scheduler.answered(id, owner, request, completion);
        }
        taken.into_iter().map(|(_, request)| request).collect()
    }

    fn detach(owner: OwnerId) -> runtime::Request {
        runtime::Request::Detach {
            owner,
            after: Vec::new(),
        }
    }

    fn input(bench: &mut Bench) -> &mut proto::request::GenerationRequest {
        let request = bench.scheduler.requests.get_mut(&ID).unwrap();
        match &mut request.input {
            proto::request::Request::Generation(input) => input,
            proto::request::Request::Diffusion(_) => unreachable!("request 1 is a token request"),
        }
    }

    /// A decode's token is appended and sent; an end-of-sequence token is not sent, ends the
    /// request with the usage it generated, and its owner is detached through its group's
    /// executor once no call of it is in flight.
    #[test]
    fn a_decode_sends_its_token_and_an_end_of_sequence_token_finishes_the_request() {
        let (mut bench, outputs) = scheduler();
        running(&mut bench, &[Phase::Prefill, Phase::Decode]);

        schedule(&mut bench, Phase::Decode, 1, 1);
        apply(&mut bench, batch(Phase::Decode, 3, 1), output(&[7]));
        let sent = events(&outputs);
        assert!(matches!(
            sent[..],
            [Event::TextToken {
                id: 7,
                logprob: None
            }]
        ));
        let request = &bench.scheduler.requests[&ID];
        assert_eq!(request.token_ids, [1, 2, 3, 7]);
        assert_eq!(request.completed[Phase::Decode.index()], 1);
        assert_eq!(bench.scheduler.groups[0].in_flight, 0);

        schedule(&mut bench, Phase::Decode, 1, 1);
        apply(&mut bench, batch(Phase::Decode, 4, 1), output(&[EOS]));
        let sent = events(&outputs);
        assert!(matches!(
            sent[..],
            [Event::Finished {
                reason: FinishReason::Eos,
                stop_reason: None,
                prompt_tokens: 3,
                completion_tokens: 2,
            }]
        ));
        assert!(bench.scheduler.requests.is_empty());
        assert!(bench.scheduler.running.is_empty());
        assert_eq!(bench.stamps(), [(A, detach(OWNER_A))]);
    }

    /// A request that finished keeps what it holds while a call of it is in flight; that call's
    /// output is dropped when it returns, and only then is its owner detached.
    #[test]
    fn a_finished_requests_call_in_flight_is_dropped_and_then_its_blocks_are_given_up() {
        let (mut bench, outputs) = scheduler();
        running(&mut bench, &[Phase::Prefill, Phase::Decode]);
        schedule(&mut bench, Phase::Decode, 2, 1);

        apply(&mut bench, batch(Phase::Decode, 3, 1), output(&[EOS]));
        assert!(matches!(
            events(&outputs)[..],
            [Event::Finished {
                reason: FinishReason::Eos,
                ..
            }]
        ));
        assert_eq!(
            bench.scheduler.requests[&ID].status,
            RequestStatus::Finished
        );
        assert!(bench.stamps().is_empty());

        apply(&mut bench, batch(Phase::Decode, 4, 1), output(&[9]));
        assert!(events(&outputs).is_empty());
        assert!(bench.scheduler.requests.is_empty());
        assert_eq!(bench.stamps(), [(A, detach(OWNER_A))]);
    }

    /// A verification took its committed token and two drafts; the one draft it rejected was not
    /// computed, and its drafter's proposals are the next verification's.
    #[test]
    fn a_verification_takes_back_the_drafts_it_rejected() {
        let (mut bench, outputs) = scheduler();
        running(&mut bench, &[Phase::Prefill, Phase::Verify]);
        schedule(&mut bench, Phase::Verify, 1, 3);
        assert_eq!(bench.scheduler.requests[&ID].num_computed_tokens, 6);

        let call = CallOutput {
            draft_token_ids: vec![11, 12],
            ..output(&[5, 6])
        };
        apply(&mut bench, batch(Phase::Verify, 3, 3), call);

        let request = &bench.scheduler.requests[&ID];
        assert_eq!(request.num_computed_tokens, 5);
        assert_eq!(request.token_ids, [1, 2, 3, 5, 6]);
        assert_eq!(request.spec_token_ids, [11, 12]);
        assert_eq!(events(&outputs).len(), 2);
    }

    /// A token's scores go to a request that asked for them, cut to the token and the one most
    /// likely candidate it asked for; the token's own score is its `logprob`.
    #[test]
    fn a_token_carries_the_scores_its_request_asked_for() {
        let (mut bench, outputs) = scheduler();
        running(&mut bench, &[Phase::Prefill, Phase::Decode]);
        input(&mut bench).sampling.n_logprobs = 1;
        schedule(&mut bench, Phase::Decode, 1, 1);

        let entry = |token_id, logprob| TokenLogprob {
            token_id,
            logprob,
            rank: 1,
        };
        let call = CallOutput {
            token_logprobs: vec![vec![
                entry(7, -0.5),
                entry(8, -0.4),
                entry(7, -0.5),
                entry(9, -2.0),
            ]],
            ..output(&[7])
        };
        apply(&mut bench, batch(Phase::Decode, 3, 1), call);

        let sent = events(&outputs);
        assert!(matches!(
            sent[..],
            [
                Event::TextToken {
                    id: 7,
                    logprob: Some(-0.5)
                },
                Event::TokenLogprobs { id: 7, .. }
            ]
        ));
        let Event::TokenLogprobs { candidates, .. } = &sent[1] else {
            unreachable!()
        };
        let kept: Vec<u32> = candidates.iter().map(|entry| entry.token_id).collect();
        assert_eq!(kept, [7, 8]);
    }

    /// The chunk that ends a prompt sends the prompt's scores at every position but the first,
    /// then the token it sampled; a chunk that scored another count of positions fails the
    /// request.
    #[test]
    fn a_prompt_is_scored_at_every_position_but_its_first() {
        let scored = |positions: usize| {
            let (mut bench, outputs) = scheduler();
            running(&mut bench, &[Phase::Prefill, Phase::Decode]);
            let request = bench.scheduler.requests.get_mut(&ID).unwrap();
            request.num_computed_tokens = 0;
            input(&mut bench).sampling.n_prompt_logprobs = 1;
            schedule(&mut bench, Phase::Prefill, 1, 3);

            let mut prefill = batch(Phase::Prefill, 0, 3);
            prefill.columns.targets = vec![2, 3, -1];
            let entries = vec![
                TokenLogprob {
                    token_id: 2,
                    logprob: -0.1,
                    rank: 1,
                },
                TokenLogprob {
                    token_id: 4,
                    logprob: -0.2,
                    rank: 2,
                },
                TokenLogprob {
                    token_id: 5,
                    logprob: -0.3,
                    rank: 3,
                },
            ];
            let call = CallOutput {
                prompt_logprobs: vec![entries; positions],
                ..output(&[7])
            };
            apply(&mut bench, prefill, call);
            events(&outputs)
        };

        let sent = scored(2);
        let [
            Event::PromptLogprobs { positions },
            Event::TextToken { id: 7, .. },
        ] = &sent[..]
        else {
            panic!("{sent:?}");
        };
        assert_eq!(positions.len(), 2);
        assert!(positions.iter().all(|position| position.entries.len() == 2));

        assert!(matches!(
            scored(1)[..],
            [Event::Finished {
                reason: FinishReason::Error,
                ..
            }]
        ));
    }

    /// The error rules of a call: an input the rank refused rejects the request as invalid; any
    /// other failure, a numerical fault included, fails it with a message naming the call, and a
    /// numerical fault reports the batch; a rank that can no longer serve stops the scheduler.
    #[test]
    fn a_failed_call_ends_its_request_by_its_class() {
        let fail = |call: CallOutput| {
            let (mut bench, outputs) = scheduler();
            running(&mut bench, &[Phase::Prefill, Phase::Decode]);
            schedule(&mut bench, Phase::Decode, 1, 1);
            apply(&mut bench, batch(Phase::Decode, 3, 1), call);
            assert!(bench.scheduler.requests.is_empty() || bench.scheduler.stopped);
            outputs.try_iter().collect::<Vec<Output>>()
        };

        let rejected = fail(failed(ErrorCode::InputError, Some("bad image")));
        assert!(matches!(
            &rejected[..],
            [Output::Event(ID, Event::Rejected { kind: RejectionKind::Invalid, message })]
                if message == "bad image"
        ));

        let computed = fail(failed(ErrorCode::ComputeError, None));
        assert!(matches!(
            &computed[..],
            [
                Output::Event(ID, Event::Error { message }),
                Output::Event(ID, Event::Finished { reason: FinishReason::Error, .. }),
            ] if message == "worker status invalid: call_failed (call 7.0, Decode)"
        ));

        let faulted = fail(failed(ErrorCode::NumericalFault, None));
        assert!(matches!(
            &faulted[..],
            [
                Output::Event(ID, Event::Error { .. }),
                Output::Event(ID, Event::Finished { reason: FinishReason::Error, .. }),
                Output::NumericalFault(BatchId(7), requests),
            ] if requests == &[ID]
        ));

        let lost = fail(failed(ErrorCode::FatalWorkerFailure, None));
        assert!(
            lost.iter()
                .any(|output| matches!(output, Output::Failed(_)))
        );
    }

    /// An output that names other calls than its batch is the executor's defect.
    #[test]
    fn an_output_of_other_calls_than_its_batch_stops_the_scheduler() {
        let (mut bench, outputs) = scheduler();
        running(&mut bench, &[Phase::Prefill, Phase::Decode]);
        schedule(&mut bench, Phase::Decode, 1, 1);
        let output = BatchOutput {
            calls: Vec::new(),
            timing: TimingCounters::default(),
            forward_stats: None,
        };
        bench
            .scheduler
            .update(batch(Phase::Decode, 3, 1), Ok(output));
        assert!(
            outputs
                .try_iter()
                .any(|output| matches!(output, Output::Failed(_)))
        );
    }

    /// Under stop strings a request's `Finished` waits until the daemon's decoder passed every
    /// token it published, the unpublished end-of-sequence token aside; a stop string the decoder
    /// matched meanwhile turns it into a `Stop`.
    #[test]
    fn stop_strings_hold_the_finish_for_the_decoder() {
        let finish = || {
            let (mut bench, outputs) = scheduler();
            running(&mut bench, &[Phase::Prefill, Phase::Decode]);
            input(&mut bench).stop_strings = vec!["end".to_owned()];
            schedule(&mut bench, Phase::Decode, 1, 1);
            apply(&mut bench, batch(Phase::Decode, 3, 1), output(&[7]));
            schedule(&mut bench, Phase::Decode, 1, 1);
            apply(&mut bench, batch(Phase::Decode, 4, 1), output(&[EOS]));
            assert!(matches!(
                events(&outputs)[..],
                [Event::TextToken { id: 7, .. }]
            ));
            assert!(bench.scheduler.requests.contains_key(&ID));
            (bench, outputs)
        };

        let (mut bench, outputs) = finish();
        bench.scheduler.acknowledge(ID, 1);
        assert!(matches!(
            events(&outputs)[..],
            [Event::Finished {
                reason: FinishReason::Eos,
                ..
            }]
        ));
        assert!(bench.scheduler.requests.is_empty());

        let (mut bench, outputs) = finish();
        bench.scheduler.cancel(ID, FinishReason::Stop);
        assert!(matches!(
            events(&outputs)[..],
            [Event::Finished {
                reason: FinishReason::Stop,
                stop_reason: None,
                ..
            }]
        ));
        assert!(bench.scheduler.requests.is_empty());
    }

    /// A cancelled waiting request is gone at once and generated nothing here; a cancelled
    /// running request's `Finished` is sent at once and its blocks wait for its call in flight.
    #[test]
    fn a_cancel_ends_a_request_at_once() {
        let (mut bench, outputs) = scheduler();
        bench.attach(&[]);
        bench.scheduler.requests.get_mut(&ID).unwrap().status = RequestStatus::Waiting;
        bench.scheduler.waiting.push_back(ID);
        bench.scheduler.cancel(ID, FinishReason::Cancelled);
        assert!(matches!(
            events(&outputs)[..],
            [Event::Finished {
                reason: FinishReason::Cancelled,
                prompt_tokens: 3,
                completion_tokens: 0,
                ..
            }]
        ));
        assert!(bench.scheduler.requests.is_empty() && bench.scheduler.waiting.is_empty());

        let (mut bench, outputs) = scheduler();
        running(&mut bench, &[Phase::Prefill, Phase::Decode]);
        schedule(&mut bench, Phase::Decode, 1, 1);
        bench.scheduler.cancel(ID, FinishReason::Aborted);
        assert!(matches!(
            events(&outputs)[..],
            [Event::Finished {
                reason: FinishReason::Aborted,
                ..
            }]
        ));
        assert!(bench.stamps().is_empty());
        apply(&mut bench, batch(Phase::Decode, 3, 1), output(&[7]));
        assert!(events(&outputs).is_empty());
        assert_eq!(bench.stamps(), [(A, detach(OWNER_A))]);
    }

    /// An admission records each owner's slot from its `Attached`; a `Full` returns the request
    /// to its place in the waiting queue, placed nowhere and holding nothing, with every owner of
    /// the admission that exists detached.
    #[test]
    fn a_full_returns_an_admission_to_the_queue_detached() {
        let (mut bench, _outputs) = scheduler();
        bench.attach(&[(A, OWNER_A, 0), (B, OWNER_B, 0)]);
        let scheduler = &mut bench.scheduler;
        scheduler.waiting.push_back(ID);
        let request = scheduler.requests.get_mut(&ID).unwrap();
        request.pending = 2;
        request.placement[Phase::Prefill.index()] = Some(A);
        request.placement[Phase::Decode.index()] = Some(B);
        scheduler.groups[0].num_requests = 1;
        scheduler.groups[1].num_requests = 1;
        for (owner, group) in [(OWNER_A, A), (OWNER_B, B)] {
            scheduler.submit(runtime::Request::Attach { owner, group });
        }

        // Both `Attach`es are written together; the first is answered.
        bench.scheduler.post().unwrap();
        let [(first, _), (second, _)] = bench.take()[..] else {
            panic!("the two `Attach`es are written");
        };
        bench.complete(first, &Completion::Attached { slot: 5 });
        answer(&mut bench, &[]);
        let request = &bench.scheduler.requests[&ID];
        assert_eq!(request.attachment(OWNER_A), Some((A, 5)));
        assert_eq!(request.pending, 1);

        // The second `Attach`'s answer arrives now.
        bench.complete(second, &Completion::Full { region: 7 });
        while let Some((id, owner, request, completion)) =
            bench.scheduler.next_completion().unwrap()
        {
            bench.scheduler.answered(id, owner, request, completion);
        }

        let scheduler = &bench.scheduler;
        let request = &scheduler.requests[&ID];
        assert_eq!(request.status, RequestStatus::Waiting);
        assert!(request.queued && request.attachments.is_empty() && request.pending == 0);
        assert!(request.placement.iter().all(Option::is_none));
        assert_eq!(scheduler.waiting, [ID]);
        assert!(scheduler.owners.is_empty());
        assert_eq!(scheduler.groups[0].num_requests, 0);
        assert_eq!(scheduler.groups[1].num_requests, 0);
        assert_eq!(bench.stamps(), [(A, detach(OWNER_A))]);
    }

    /// An admitted request runs from its hit, which it is told it was scheduled with, and leaves
    /// the waiting queue.
    #[test]
    fn an_admitted_request_runs_from_its_hit() {
        let (mut bench, outputs) = scheduler();
        bench.attach(&[(A, OWNER_A, 3)]);
        let scheduler = &mut bench.scheduler;
        scheduler.waiting.push_back(ID);
        scheduler.requests.get_mut(&ID).unwrap().num_cached_tokens = 2;

        scheduler.admitted(ID);
        let request = &scheduler.requests[&ID];
        assert_eq!(request.status, RequestStatus::Running);
        assert_eq!(request.num_computed_tokens, 2);
        assert!(!request.queued);
        assert!(scheduler.waiting.is_empty());
        assert_eq!(scheduler.running, [ID]);
        let sent = events(&outputs);
        let [
            Event::Scheduled {
                queued_at,
                scheduled_at,
            },
        ] = sent[..]
        else {
            panic!("{sent:?}");
        };
        assert!(0.0 < queued_at && queued_at <= scheduled_at);
    }

    /// A request a prefill group served moves to a decode group once its prompt is computed: its
    /// owner's `Transfer` is stamped by the prefill group, and at `Transferred` the owner is its
    /// attachment to the decode group, with the slot there, which has not seen it yet.
    #[test]
    fn a_prefill_group_hands_its_request_to_a_decode_group() {
        let (mut bench, outputs) = scheduler();
        prefill_and_decode(&mut bench);

        let transfer = runtime::Request::Transfer {
            owner: OWNER_A,
            group: B,
            after: Vec::new(),
        };
        assert_eq!(bench.stamps(), [(A, transfer)]);
        assert!(matches!(
            events(&outputs)[..],
            [Event::TextToken { id: 7, .. }]
        ));
        let request = &bench.scheduler.requests[&ID];
        assert_eq!(request.status, RequestStatus::WaitingForRemoteKvs);
        assert_eq!(request.placement[Phase::Decode.index()], Some(B));

        stamp_transfer(&mut bench);
        let answered = answer(
            &mut bench,
            &[Completion::Transferred {
                slot: 6,
                runs: Vec::new(),
            }],
        );
        assert!(matches!(answered[..], [runtime::Request::Transfer { .. }]));

        let scheduler = &bench.scheduler;
        let request = &scheduler.requests[&ID];
        assert_eq!(request.status, RequestStatus::Running);
        assert_eq!(request.attachments, [(B, OWNER_A, 6, false)]);
        assert_eq!(request.placement[Phase::Prefill.index()], None);
        assert_eq!(request.placement[Phase::Decode.index()], Some(B));
        assert_eq!(scheduler.groups[0].num_requests, 0);
        assert_eq!(scheduler.groups[1].num_requests, 1);
    }

    /// A transfer that failed leaves the owner on the prefill group with blocks no device block
    /// table holds: its `Detach` is written with no stamp and the request leaves as migratable.
    #[test]
    fn a_failed_transfer_ends_the_request_as_migratable_and_detaches_it_unstamped() {
        let (mut bench, outputs) = scheduler();
        prefill_and_decode(&mut bench);
        bench.stamps();
        events(&outputs);

        stamp_transfer(&mut bench);
        answer(&mut bench, &[Completion::Failed]);
        assert!(matches!(
            events(&outputs)[..],
            [Event::Migratable {
                cause: MigrationCause::Loss,
                denoised_steps: 0,
                ..
            }]
        ));
        assert!(bench.scheduler.requests.is_empty());
        assert!(bench.stamps().is_empty());
        assert_eq!(answer(&mut bench, &[]), [detach(OWNER_A)]);
        assert_eq!(bench.scheduler.groups[1].num_requests, 0);
    }

    /// A request whose decode groups have no free slot stays on its prefill group, and moves once
    /// a later step finds one.
    #[test]
    fn a_transfer_that_finds_no_decode_group_waits_for_one() {
        let (mut bench, _outputs) = scheduler();
        bench.scheduler.available[3] = 0;
        prefill_and_decode(&mut bench);
        assert!(bench.stamps().is_empty());
        assert_eq!(bench.scheduler.requests[&ID].status, RequestStatus::Running);

        bench.scheduler.start_transfers();
        assert!(bench.stamps().is_empty());
        bench.scheduler.available[3] = 4;
        bench.scheduler.start_transfers();
        assert!(matches!(
            bench.stamps()[..],
            [(A, runtime::Request::Transfer { group: B, .. })]
        ));
    }

    /// Request 1 prefilled on group A, a prefill group, beside group B, a decode group: its
    /// prompt's one chunk returned its first token.
    fn prefill_and_decode(bench: &mut Bench) {
        running(bench, &[Phase::Prefill]);
        let scheduler = &mut bench.scheduler;
        scheduler.groups[0].config.role = WorkerGroupRole::Prefill;
        scheduler.groups[0].num_requests = 1;
        scheduler.groups[1].config.role = WorkerGroupRole::Decode;
        scheduler.groups[1].capacity.phases = 1 << Phase::Decode.index();
        scheduler.requests.get_mut(&ID).unwrap().num_computed_tokens = 0;

        schedule(bench, Phase::Prefill, 1, 3);
        apply(bench, batch(Phase::Prefill, 0, 3), output(&[7]));
    }

    /// Group A's executor returns request 1's `Transfer`, stamped.
    fn stamp_transfer(bench: &mut Bench) {
        bench.scheduler.stamped(vec![runtime::Request::Transfer {
            owner: OWNER_A,
            group: B,
            after: vec![(A, 9)],
        }]);
    }

    /// A running request's `Allocate` that found no room asks for nothing beyond what it holds.
    #[test]
    fn an_allocate_that_found_no_room_leaves_its_class_short() {
        let (mut bench, _outputs) = scheduler();
        running(&mut bench, &[Phase::Prefill, Phase::Decode]);
        let blocks = &mut bench.scheduler.requests.get_mut(&ID).unwrap().blocks[0];
        blocks.end = 2;
        blocks.asked = 3;
        bench.scheduler.submit(runtime::Request::Allocate {
            owner: OWNER_A,
            class: 0,
            at: 2,
            count: 1,
            units: 0,
        });
        answer(&mut bench, &[Completion::Full { region: 4 }]);
        let blocks = bench.scheduler.requests[&ID].blocks[0];
        assert_eq!((blocks.end, blocks.asked), (2, 2));
    }

    /// A refusal of what the scheduler chose from its own counts stops it; a key `cache` believed
    /// held is a miss the admission goes on from.
    #[test]
    fn a_refusal_is_a_miss_only_for_what_cache_believed() {
        let (mut bench, outputs) = scheduler();
        bench.attach(&[(A, OWNER_A, 3)]);
        bench.scheduler.requests.get_mut(&ID).unwrap().pending = 2;
        bench.scheduler.submit(runtime::Request::Clone {
            owner: OWNER_A,
            class: 0,
            at: 0,
            key: super::super::tests::key(1),
        });
        answer(&mut bench, &[Completion::Refused { cause: Cause::Key }]);
        assert_eq!(bench.scheduler.requests[&ID].pending, 1);
        assert!(!bench.scheduler.stopped);

        bench.scheduler.submit(runtime::Request::Allocate {
            owner: OWNER_A,
            class: 0,
            at: 0,
            count: 1,
            units: 0,
        });
        answer(
            &mut bench,
            &[Completion::Refused {
                cause: Cause::Position,
            }],
        );
        assert!(bench.scheduler.stopped);
        assert!(
            outputs
                .try_iter()
                .any(|output| matches!(output, Output::Failed(_)))
        );
    }

    /// A `Resize` the engine asked for is answered at its `Resized`, which sets the region's
    /// capacity.
    #[test]
    fn a_resize_is_answered_at_its_resized() {
        let (mut bench, outputs) = scheduler();
        let (reply, replied) = crossbeam_channel::bounded(1);
        bench.scheduler.resizing.push((5, reply));
        bench.scheduler.submit(runtime::Request::Resize {
            region: 5,
            units: 32,
        });
        answer(
            &mut bench,
            &[Completion::Resized {
                units: 30,
                evicted: Vec::new(),
            }],
        );
        assert_eq!(replied.try_recv(), Ok(()));
        assert_eq!(bench.scheduler.capacity[1], 30);
        assert!(bench.scheduler.resizing.is_empty());
        assert!(
            !outputs
                .try_iter()
                .any(|output| matches!(output, Output::Shrunk(..)))
        );
    }

    /// The events a consumer took return their credit, never beyond the window.
    #[test]
    fn consumed_events_return_their_credit() {
        let (mut bench, _outputs) = scheduler();
        running(&mut bench, &[Phase::Prefill, Phase::Decode]);
        bench.scheduler.requests.get_mut(&ID).unwrap().event_credit = 60;
        bench.scheduler.send(
            ID,
            Event::TextToken {
                id: 7,
                logprob: None,
            },
        );
        assert_eq!(bench.scheduler.requests[&ID].event_credit, 59);
        bench.scheduler.consumed(ID, 3);
        assert_eq!(bench.scheduler.requests[&ID].event_credit, 62);
        bench.scheduler.consumed(ID, 10);
        assert_eq!(bench.scheduler.requests[&ID].event_credit, 64);
    }

    /// Outputs of one phase are counted in the order their calls were scheduled, those that
    /// returned ahead of an earlier call once it returned.
    #[test]
    fn outputs_are_counted_in_call_order() {
        let (mut bench, _outputs) = scheduler();
        bench.attach(&[]);
        let request = bench.scheduler.requests.get_mut(&ID).unwrap();
        request.complete(Phase::VideoEncode, 1);
        request.complete(Phase::VideoEncode, 2);
        assert_eq!(request.completed[Phase::VideoEncode.index()], 0);
        request.complete(Phase::VideoEncode, 0);
        assert_eq!(request.completed[Phase::VideoEncode.index()], 3);
        assert!(request.early.is_empty());
        request.complete(Phase::VideoEncode, 1);
        assert_eq!(request.completed[Phase::VideoEncode.index()], 3);
    }

    /// A block a request's `Commit` left private to it, answered `Duplicate`, was never counted as
    /// held by it (BP §9, R4): giving the request up leaves the cached block to its other holder,
    /// out of the eviction order.
    #[test]
    fn a_duplicate_is_given_up_without_unpinning_the_cached_block() {
        let (mut bench, _outputs) = scheduler();
        running(&mut bench, &[Phase::Prefill, Phase::Decode]);
        // Another request of the scheduler holds the block cached under key 1 in group A.
        let key = super::super::tests::key(1);
        let cached = Cached {
            key,
            class: 0,
            tokens: 16,
            tiers: DEVICE,
            groups: 1 << A.0,
            pins: 1,
            last_used: 1,
        };
        bench.scheduler.cache.insert(Cached::id(&key), cached);
        let request = bench.scheduler.requests.get_mut(&ID).unwrap();
        request.keys = vec![vec![key]];
        request.blocks[0] = Blocks {
            first: 0,
            shared: 1,
            end: 1,
            asked: 1,
        };

        let commit = |after| runtime::Request::Commit {
            owner: OWNER_A,
            class: 0,
            at: 0,
            key,
            tokens: 16,
            after,
        };
        bench.scheduler.submit(commit(Vec::new()));
        bench.stamps();
        bench.scheduler.stamped(vec![commit(vec![(A, 1)])]);
        assert_eq!(answer(&mut bench, &[Completion::Duplicate]).len(), 1);

        bench.scheduler.cancel(ID, FinishReason::Cancelled);
        bench.scheduler.install_victims();
        let written = answer(&mut bench, &[]);
        assert!(
            !written
                .iter()
                .any(|request| matches!(request, runtime::Request::Victims { .. })),
            "the block is still held: {written:?}"
        );
    }

    /// A stop token ends the request with `Stop`, unpublished without `include_stop_token`.
    #[test]
    fn a_stop_token_ends_the_request_unpublished() {
        let (mut bench, outputs) = scheduler();
        running(&mut bench, &[Phase::Prefill, Phase::Decode]);
        input(&mut bench).stop_token_ids = vec![9];
        schedule(&mut bench, Phase::Decode, 1, 1);
        apply(&mut bench, batch(Phase::Decode, 3, 1), output(&[9]));
        assert!(matches!(
            events(&outputs)[..],
            [Event::Finished {
                reason: FinishReason::Stop,
                stop_reason: Some(StopReason::Token(9)),
                completion_tokens: 1,
                ..
            }]
        ));
    }
}
