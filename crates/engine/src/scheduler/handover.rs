//! The scheduler's part in a control-plane handover (`crate::handover`).
//!
//! Quiescing (`Command::Quiesce`) stops token admission: every waiting token
//! request migrates as it is. Requests that cannot migrate (media requests,
//! token requests with images, readouts, canvases, and requests whose
//! generation has ended) run to completion. Once only migrating requests
//! run, the scheduler pauses them: it forms no batch, and lets the batches
//! and lifecycle commands in flight drain. At that step boundary it hands
//! over: it publishes each running request's computed cache pages, snapshots
//! the requests and the prefix cache index, closes the executor and replies
//! with the snapshot, the migrated requests' event channels and its command
//! channel. Its loop then ends.
//!
//! Resuming (`Command::Resume`) promotes the executor's workers, installs
//! the predecessor's pages, and queues each migrated request with its
//! generated tokens after its prompt, in the snapshot's order, ahead of any
//! request submitted later.

use super::*;
use crate::handover::{Adoption, CoreSnapshot, MigratedRequest, MigratedStream, Quiesced};

impl Scheduler {
    /// Prepares a handover: see `ExecutorHandover::prepare`. An executor
    /// whose workers keep no storage adopts nothing.
    pub(super) fn prepare_handover(&mut self) -> anyhow::Result<Option<Adoption>> {
        self.executor
            .handover()
            .map(|handover| handover.prepare())
            .transpose()
    }

    /// The successor did not take over: see `ExecutorHandover::abandon`.
    pub(super) fn abandon_handover(&mut self) {
        if let Some(handover) = self.executor.handover() {
            handover.abandon();
        }
    }

    /// Whether a running token request migrates: it generates text only from
    /// a text prompt, its generation has not ended, any prompt scores it
    /// asked for were all published, and no host page tier read of its
    /// prompt is outstanding (the tier counts those pages computed before
    /// they are read).
    fn migrates(&self, state: &RequestState) -> bool {
        self.config.canvas.is_none()
            && !state.req.is_readout()
            && !state.req.generates_images()
            && !state.has_context_images()
            && !state.terminal_intent.is_terminal()
            && state.failure.is_none()
            && state.generated_token_ids.len() == state.num_generated_tokens
            && (!state.req.sampling.prompt_logprobs_requested()
                || !matches!(state.phase, Phase::Prefill))
            && self
                .tiers
                .as_ref()
                .is_none_or(|tiers| tiers.hit_pages(state.req.request_id).is_empty())
    }

    /// Whether the scheduler is quiescing.
    pub(super) fn quiescing(&self) -> bool {
        self.quiesce.is_some()
    }

    /// Whether a quiescing scheduler pauses its requests: every request that
    /// does not migrate has completed.
    pub(super) fn paused(&self) -> bool {
        self.quiescing()
            && self.running_media.is_empty()
            && self.waiting_media.is_empty()
            && self.running.values().all(|state| self.migrates(state))
    }

    /// Whether a quiescing scheduler is at the step boundary it hands over
    /// at: paused, with no batch, submission or lifecycle command in flight
    /// and no request retiring.
    pub(super) fn quiescent(&self) -> bool {
        self.paused()
            && self.inflight.pending_batches.is_empty()
            && self.inflight.pending_submissions.is_empty()
            && self.inflight.pending_commands.is_empty()
            && self.retiring_requests.is_empty()
    }

    /// Hands the engine core over at a quiescent step boundary and replies
    /// to the quiesce command; the caller's loop then ends. `commands` is
    /// the core's command channel, which the reply hands to the relay.
    pub(super) fn hand_over(&mut self, commands: &Receiver<Command>) {
        let Some(reply) = self.quiesce.take() else {
            return;
        };
        let quiesced = self.quiesced(commands);
        if let Err(error) = &quiesced {
            tracing::error!("the engine core could not hand over: {error:#}");
            self.abort_all_requests();
            let _ = self.executor.close();
        }
        let _ = reply.send(quiesced);
    }

    fn quiesced(&mut self, commands: &Receiver<Command>) -> anyhow::Result<Quiesced> {
        let running: Vec<RequestId> = self
            .running_order
            .iter()
            .copied()
            .filter(|id| self.running.contains_key(id))
            .collect();
        for &id in &running {
            self.publish_computed(id);
        }
        let pages = self
            .storage
            .cache
            .as_ref()
            .map(|cache| cache.block_pool.snapshot())
            .unwrap_or_default();
        let endpoints = self
            .executor
            .info()
            .workers
            .iter()
            .map(|(id, info)| (id.to_string(), info.endpoint.clone()))
            .collect();

        // Running requests first, in admission order, then waiting ones.
        let mut states = Vec::with_capacity(running.len() + self.waiting.len());
        for id in running {
            if let Some(state) = self.running.remove(&id) {
                states.push(state);
            }
        }
        self.running_order.clear();
        states.extend(self.waiting.drain(..));
        let mut requests = Vec::with_capacity(states.len());
        let mut streams = Vec::with_capacity(states.len());
        for state in states {
            let (request, stream) = migrated(state);
            requests.push(request);
            streams.push(stream);
        }
        let migrated_requests = requests.len();

        // The tier's pins end with its connection; its pending writes are
        // dropped. The workers close, releasing their exclusive storages
        // with the contents intact.
        self.tiers = None;
        self.executor.close()?;
        // Finished requests' undelivered events go to their receivers.
        self.output.hand_off_retired();
        tracing::info!(
            requests = migrated_requests,
            pages = pages.len(),
            "the engine core quiesced and handed its requests over"
        );
        Ok(Quiesced {
            snapshot: CoreSnapshot {
                // The engine core states its own next request identity.
                next_request_id: 0,
                next_request_epoch: self.next_request_epoch,
                requests,
                pages,
                endpoints,
            },
            streams,
            commands: commands.clone(),
        })
    }

    /// Publishes the cache pages a running request has computed over its
    /// prompt and generated tokens, keyed by its prefill worker, so that the
    /// successor finds them when it queues the request again. A request
    /// whose cache policy writes nothing publishes nothing and recomputes.
    fn publish_computed(&mut self, id: RequestId) {
        let Some(state) = self.running.get(&id) else {
            return;
        };
        let key = RequestKey::new(self.engine_id, id, state.request_epoch);
        let source = self
            .placement
            .affinity
            .get(&(key, DEFAULT_COMPONENT.to_owned()))
            .and_then(|worker| {
                self.executor
                    .info()
                    .workers
                    .iter()
                    .find(|(id, _)| id == worker)
            })
            .map(|(_, worker)| Arc::new(worker.endpoint.clone()));
        let (Some(source), Some(state), Some(cache)) = (
            source,
            self.running.get_mut(&id),
            self.storage.cache.as_ref(),
        ) else {
            return;
        };

        // The cache holds the prompt's tokens, then the generated ones but
        // the last, which the next forward would have read. A verifier's
        // rejected drafts are initialized past the visible extent and are
        // not the request's tokens.
        let mut sequence = state.req.prompt_token_ids.clone();
        sequence.extend_from_slice(&state.generated_token_ids[state.resumed_tokens..]);
        let computed = state.cache_visible_len.min(state.cache_computed_len) as usize;
        let hashes = cache
            .coordinator
            .prefix_hashes(&sequence, state.req.cache.isolation_key);
        let cache_write = state.req.cache.write;
        let mut published = std::mem::take(&mut state.prefix_published);
        if let Some(allocation) = state.prefix_cache_mut() {
            cache.coordinator.publish_prompt(
                &cache.block_pool,
                allocation,
                &sequence,
                &hashes,
                computed,
                &mut published,
                cache_write,
                &source,
            );
        }
        state.prefix_published = published;
    }

    /// Resumes a predecessor's requests: see the module documentation.
    /// `events` holds each migrated request's event channel, in the
    /// snapshot's order. A request this engine core refuses is rejected on
    /// its channel.
    ///
    /// Fails when the workers cannot be promoted.
    pub(super) fn resume(
        &mut self,
        snapshot: CoreSnapshot,
        events: Vec<EventTx>,
    ) -> anyhow::Result<()> {
        if let Some(handover) = self.executor.handover() {
            handover.promote()?;
        }

        // The predecessor's pages name its workers' endpoints; this core's
        // groups of the same identity hold their units now.
        let endpoints: HashMap<&uniserve_ipc::WorkerEndpoint, Arc<uniserve_ipc::WorkerEndpoint>> =
            snapshot
                .endpoints
                .iter()
                .filter_map(|(group, before)| {
                    let (_, info) = self
                        .executor
                        .info()
                        .workers
                        .iter()
                        .find(|(id, _)| id.to_string() == *group)?;
                    Some((before, Arc::new(info.endpoint.clone())))
                })
                .collect();
        if let Some(cache) = self.storage.cache.as_ref() {
            let restored = cache
                .block_pool
                .restore(&snapshot.pages, |source| endpoints.get(source).cloned());
            tracing::info!(
                pages = restored,
                of = snapshot.pages.len(),
                "installed the predecessor's prefix cache pages"
            );
        }

        self.next_request_epoch = self.next_request_epoch.max(snapshot.next_request_epoch);
        for (migrated, event_tx) in snapshot.requests.into_iter().zip(events) {
            let max_cache_tokens = match self.validate_submission(&migrated.request) {
                Ok(tokens) => tokens,
                Err(message) => {
                    let _ = event_tx.send(EngineCoreOutput::Rejected {
                        kind: RejectionKind::Invalid,
                        message,
                    });
                    continue;
                }
            };
            let MigratedRequest {
                mut request,
                request_epoch,
                generated,
                tokens_sent,
                decoder_boundaries,
                queued_at,
                scheduled,
            } = migrated;
            let resumed_tokens = generated.len();
            request.prompt_token_ids.extend_from_slice(&generated);
            // An admitted request's prompt scores were published.
            if scheduled {
                request.sampling.return_prompt_logprobs = false;
                request.sampling.n_prompt_logprobs = 0;
            }
            let mut state = self.request_state(request, event_tx, max_cache_tokens);
            state.request_epoch = request_epoch;
            state.resumed_tokens = resumed_tokens;
            state.num_generated_tokens = resumed_tokens;
            state.generated_token_ids = generated;
            state.output.tokens_sent = tokens_sent;
            state.output.decoder_boundaries = decoder_boundaries.into();
            state.queued_at = queued_at;
            state.scheduled_sent = scheduled;
            self.waiting.push_back(state);
        }
        Ok(())
    }
}

/// A request leaving the engine core: its snapshot entry and its client
/// side.
fn migrated(state: RequestState) -> (MigratedRequest, MigratedStream) {
    let mut request = state.req;
    let prompt_tokens = request.prompt_token_ids.len() - state.resumed_tokens;
    request.prompt_token_ids.truncate(prompt_tokens);
    let id = request.request_id;
    let (events, pending) = state.output.events.into_parts();
    (
        MigratedRequest {
            request,
            request_epoch: state.request_epoch,
            generated: state.generated_token_ids,
            tokens_sent: state.output.tokens_sent,
            decoder_boundaries: state.output.decoder_boundaries.into_iter().collect(),
            queued_at: state.queued_at,
            scheduled: state.scheduled_sent,
        },
        MigratedStream {
            request_id: id,
            events,
            pending,
        },
    )
}
