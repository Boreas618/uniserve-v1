//! Result reconciliation: `apply_result` validates every returned call identity
//! against the in-flight window before any completion is applied, stages
//! completions, and applies them in request-local dependency order; a diffusion
//! request's results advance its media flow until it finishes.

use super::*;
use ipc::{ForwardMode, MediaCall, TransferMode};

/// Describes a rejected worker result for the request's terminal error and
/// its error finish's log line: the rejection, then the call it answered.
fn rejection_cause(call: &Call, error: &generation::GenerationResultError) -> String {
    format!(
        "{error} (call {}.{}, {:?})",
        call.call_id.batch_id, call.call_id.request_index, call.kind
    )
}

impl Scheduler {
    /// Holds one validated completion until earlier calls for the request are applied.
    ///
    /// A completion that names no in-flight call of the request, or repeats
    /// one already queued, is dropped and fails its request: a media request
    /// records a failure intent, a running token request finishes with an
    /// error. `apply_result` releases queued completions in dependency order
    /// through `Inflight::take_ready_completions`.
    pub(in crate::scheduler) fn queue_completion(
        &mut self,
        mut record: ipc::CallOutput,
        media: Result<Option<Arc<SharedMedia>>, ipc::SharedMediaError>,
    ) {
        let id = record.request_key.request_id;
        let call_id = record.call_id;
        let known = self.inflight.pending_calls.get(&id).is_some_and(|queue| {
            queue.iter().any(|inflight| {
                inflight.call.request_key == record.request_key && inflight.call.call_id == call_id
            })
        });
        let duplicate = self
            .inflight
            .pending_completions
            .get(&id)
            .is_some_and(|pending| pending.contains_key(&call_id));
        if !known || duplicate {
            if let Some(state) = self.media_state_mut(id) {
                state.terminal_intent =
                    TerminalIntent::Failure("worker returned an unknown media call".to_string());
            } else if self.running.contains_key(&id) {
                self.finish(id, FinishReason::Error);
            }
            return;
        }

        // Output storage the executor could not claim fails the call rather
        // than dropping its completion, so request-local ordering still sees
        // the call return.
        let media = match media {
            Ok(media) => media,
            Err(error) => {
                // The client's message carries the failed step and its cause.
                let message = format!("{:#}", anyhow::Error::new(error));
                let event = Event::ArtifactUnavailable { message };
                if let Some(state) = self.media_state_mut(id) {
                    let _ = state.output.enqueue(event);
                } else {
                    self.emit(id, event);
                }
                record.status = CallStatus::Error;
                record.error_code = Some(ipc::ErrorCode::ResourceError);
                None
            }
        };

        let arrival_seq = self.inflight.next_arrival();
        self.inflight
            .pending_completions
            .entry(id)
            .or_default()
            .insert(
                call_id,
                PendingCompletion {
                    record,
                    media,
                    arrival_seq,
                },
            );
    }

    /// Validates a video media call result and advances only the request
    /// fields that call completes.
    ///
    /// A result is valid when it succeeded, names the submitted call, carries
    /// an artifact exactly when it is the final muxing call, and, for a
    /// denoising step, advances the trajectory's completed steps the worker
    /// reports. The first valid result marks
    /// the request registered with its workers. An invalid result records a
    /// failure intent instead of advancing anything. A valid result of a
    /// request that has not already failed frees the products the call
    /// consumed. Once no call of the request remains in flight, a terminal
    /// intent, a closed output or a muxed artifact retires the request
    /// through `finish_media`.
    pub(in crate::scheduler) fn process_diffusion_result(
        &mut self,
        call: Call,
        decode: Option<DecodeRange>,
        record: ipc::CallOutput,
        media: Option<Arc<SharedMedia>>,
    ) {
        let id = record.request_key.request_id;
        let Some(state) = self.media_state(id) else {
            return;
        };
        let mut consumed_products = Vec::new();
        let already_failed = matches!(
            state.terminal_intent,
            TerminalIntent::Failure(_) | TerminalIntent::Rejected(_)
        );

        // Only the final muxing call, the one that carries no media units,
        // returns the artifact; every other call returns none.
        let media_output_valid = if call.kind == CallKind::Media(MediaCall::Muxing) {
            media.is_some() == call.inputs.is_empty()
        } else {
            media.is_none()
        };
        // The worker reports the steps it has completed; a denoising result
        // that does not advance the trajectory would be scheduled again
        // without end.
        let step_valid = call.kind != CallKind::Media(MediaCall::Denoising)
            || record.step_index > state.denoising.completed();
        let valid = record.status == CallStatus::Ok
            && record.request_key == call.request_key
            && record.call_id == call.call_id
            && media_output_valid
            && step_valid;
        if valid
            && let Some(state) = self.media_state_mut(id)
            && state.admission_state == WorkerRegistration::InFlight
        {
            state.admission_state = WorkerRegistration::Registered;
        }

        // A request that already failed only drains its remaining calls. A
        // worker that refuses the request states why, and the client is told.
        if !already_failed {
            if !valid {
                let refusal = (record.error_code == Some(ipc::ErrorCode::InputError))
                    .then(|| record.error_message.clone())
                    .flatten();
                if let Some(state) = self.media_state_mut(id) {
                    state.terminal_intent = match refusal {
                        Some(reason) => TerminalIntent::Rejected(reason),
                        None => TerminalIntent::Failure("media worker call failed".to_string()),
                    };
                }
            } else if let Some(state) = self.media_state_mut(id) {
                // A condition product retires once the calls reading it have
                // completed: the vision patches with the vision encoding, the
                // vision features with the text encoding, the pixels with the
                // last visual latent round, the PCM with the audio latent
                // encoding, and the condition rows with latent preparation.
                match call.kind {
                    CallKind::Media(MediaCall::VisionEncoding) => {
                        consumed_products
                            .extend(state.condition_media.remove(product::VISION_PIXELS));
                    }
                    CallKind::Media(MediaCall::TextEncoding) => {
                        consumed_products.extend(state.vision_features.take());
                    }
                    CallKind::Media(MediaCall::LatentEncoding) => {
                        let audio = state
                            .condition_audio_latents
                            .as_ref()
                            .is_some_and(|product| product.producer_call_id == call.call_id);
                        if audio {
                            consumed_products
                                .extend(state.condition_media.remove(product::CONDITION_SAMPLES));
                        } else if let Some(range) = decode.as_ref() {
                            state.encoded_condition_units += range.max_units;
                            if state.encoded_condition_units
                                == state.graph.condition_units().len() as u32
                            {
                                consumed_products.extend(
                                    state.condition_media.remove(product::CONDITION_PIXELS),
                                );
                            }
                        } else {
                            self.invariant_broken("a submitted latent round keeps its unit range");
                            return;
                        }
                    }
                    CallKind::Media(MediaCall::LatentPreparation) => {
                        state.denoising.open(None);
                        consumed_products.extend(state.condition_video_latents.iter().cloned());
                        consumed_products.extend(state.condition_audio_latents.iter().cloned());
                    }
                    // `valid` requires a denoising result to advance the trajectory.
                    CallKind::Media(MediaCall::Denoising) => {
                        let accepted = state.denoising.accept(record.step_index, None);
                        debug_assert!(accepted, "a valid step advances the trajectory");
                    }
                    CallKind::Media(MediaCall::VideoEncoding) => {
                        let Some(range) = decode.as_ref() else {
                            self.invariant_broken("a submitted media write keeps its input range");
                            return;
                        };
                        state.encoded_ready.insert(range.cursor, range.max_units);
                        if let Some((_, product)) = state.decoded_units.remove(&range.cursor) {
                            consumed_products.push(product);
                        }
                    }
                    CallKind::Media(MediaCall::AudioEncoding) => {
                        state.audio_encoded = true;
                        if let Some(product) = state.audio.take() {
                            consumed_products.push(product);
                        }
                    }
                    CallKind::Media(MediaCall::Muxing) => {
                        // The muxer has taken these units into the container,
                        // so the encoded products it consumed retire with the
                        // call; the artifact arrives with the final call.
                        state.muxing_in_flight = false;
                        consumed_products.append(&mut state.muxing_inputs);
                        if media.is_some() {
                            state.muxed = true;
                        }
                    }
                    _ => {}
                }

                let phase = match call.kind {
                    CallKind::Media(
                        MediaCall::MediaReading
                        | MediaCall::VisionEncoding
                        | MediaCall::LatentEncoding
                        | MediaCall::TextEncoding,
                    ) => "preparing",
                    CallKind::Media(MediaCall::LatentPreparation)
                    | CallKind::Media(MediaCall::Denoising) => {
                        if state.denoising.is_complete() {
                            "decoding"
                        } else {
                            "denoising"
                        }
                    }
                    CallKind::Media(
                        MediaCall::VideoDecoding
                        | MediaCall::AudioDecoding
                        | MediaCall::VideoEncoding
                        | MediaCall::AudioEncoding,
                    ) => "decoding",
                    CallKind::Media(MediaCall::Muxing) => "finalizing",
                    _ => unreachable!("media request completed a non-media call"),
                };
                let _ = state.output.enqueue(Event::MediaProgress {
                    phase: phase.to_owned(),
                    completed_steps: state.denoising.completed(),
                });
                if let Some(media) = media {
                    state.artifact = Some(ArtifactEvent {
                        media_kind: MediaKind::Video,
                        content_type: "video/mp4".to_string(),
                        media,
                    });
                }
            }
        }

        if !consumed_products.is_empty() {
            self.free_buffers(
                consumed_products
                    .into_iter()
                    .map(|product| product.buffer_id()),
            );
        }

        // Retire the request only once none of its calls remain in flight. A
        // failure outranks a finish, which outranks a closed output; a muxed
        // request completes with its artifact.
        let terminal = self.media_state(id).and_then(|state| {
            if self.inflight.has_pending_calls(id) {
                return None;
            }
            if let TerminalIntent::Failure(message) = &state.terminal_intent {
                Some(DiffusionTerminal::Failed(message.clone()))
            } else if let TerminalIntent::Rejected(message) = &state.terminal_intent {
                Some(DiffusionTerminal::Rejected(message.clone()))
            } else if let TerminalIntent::Finish(reason) = &state.terminal_intent {
                Some(DiffusionTerminal::Finished(reason.clone()))
            } else if state.output.is_closed() {
                Some(DiffusionTerminal::Finished(FinishReason::Cancelled))
            } else if state.muxed {
                let event = state.artifact.clone().map_or_else(
                    || DiffusionTerminal::Failed("media output was not finalized".to_string()),
                    DiffusionTerminal::Completed,
                );
                Some(event)
            } else {
                None
            }
        });
        if let Some(event) = terminal {
            self.finish_media(id, event);
        }
    }

    /// Emits terminal media output and releases all request-owned resources.
    ///
    /// A request whose admission was never placed in a batch frees its
    /// allocations at once. Otherwise a `Finish` command is queued and the
    /// allocations wait in `retiring_requests` until `acknowledge_commands`
    /// sees that command complete.
    pub(in crate::scheduler) fn finish_media(&mut self, id: RequestId, event: DiffusionTerminal) {
        let Some(mut state) = self.take_media_state(id) else {
            return;
        };
        self.running_order.retain(|candidate| *candidate != id);
        match event {
            DiffusionTerminal::Completed(artifact) => {
                let _ = state.output.enqueue(Event::Artifact(artifact));
                let _ = state.output.enqueue(Event::Finished {
                    reason: FinishReason::Completed,
                    stop_reason: None,
                    prompt_tokens: state.request.prompt_token_ids.len(),
                    completion_tokens: 0,
                    images: 0,
                });
            }
            DiffusionTerminal::Failed(message) => {
                let _ = state.output.enqueue(Event::Error { message });
            }
            DiffusionTerminal::Rejected(message) => {
                let _ = state.output.enqueue(Event::Rejected {
                    kind: RejectionKind::Invalid,
                    message,
                });
            }
            DiffusionTerminal::Finished(reason) => {
                let _ = state.output.enqueue(Event::Finished {
                    reason,
                    stop_reason: None,
                    prompt_tokens: state.request.prompt_token_ids.len(),
                    completion_tokens: 0,
                    images: 0,
                });
            }
        }
        self.output.retire(id, state.output);

        // No worker holds state for a request whose admission was never
        // placed in a batch, so there is nothing to retire remotely.
        if state.admission_state == WorkerRegistration::Unsubmitted {
            if let Err(unknown) = state.allocations.free(&mut self.storage) {
                self.invariant_broken(&unknown.to_string());
            }
            return;
        }
        let request_key = state.admission.request_key;
        self.inflight
            .pending_commands
            .push_back(BatchCommand::Finish {
                request_key,
                retained_buffers: Vec::new(),
            });
        self.retiring_requests.insert(
            id,
            RetiringRequest {
                request_key,
                allocations: Vec::new(),
                media_allocations: Some(state.allocations),
                buffers: HashMap::new(),
            },
        );
    }

    /// Removes the in-flight call a worker result names, releasing its
    /// transfer reservation and its domain credit.
    ///
    /// The call must be at the front of its request's queue, unless it is an
    /// independent media call (media input that does not advance state).
    /// Returns `None` for any other call, an unknown call, or a call id with
    /// batch zero.
    pub(in crate::scheduler) fn pop_inflight(
        &mut self,
        request_key: RequestKey,
        call_id: CallId,
    ) -> Option<(Call, InflightInput)> {
        let inflight = self.inflight.pop_pending_call(request_key, call_id)?;
        self.reclaim_domain_credit(inflight.call.kind, false);
        Some((inflight.call, inflight.input))
    }

    /// Revokes completed buffer identities while retaining allocations until
    /// release acknowledgement.
    ///
    /// Queues one `Free` command per distinct buffer. The allocation of a
    /// buffer tracked in `encoder_buffers` moves to `pending_buffer_frees`
    /// until `acknowledge_commands` sees its `Free` complete; a buffer already
    /// pending there is an accounting defect that latches engine-fatal.
    pub(in crate::scheduler) fn free_buffers(
        &mut self,
        buffers: impl IntoIterator<Item = ipc::BufferId>,
    ) {
        let mut released = HashSet::new();
        for buffer in buffers {
            if !released.insert(buffer) {
                continue;
            }
            if let Some(allocation) = self.storage.take_encoder_buffer(buffer)
                && let Some(previous) = self.storage.pending_buffer_frees.insert(buffer, allocation)
            {
                tracing::error!(?buffer, "buffer free identity was already pending");
                self.storage.buffer_pool.free(previous);
                self.fatal = true;
            }
            self.inflight
                .pending_commands
                .push_back(BatchCommand::Free { buffer });
        }
    }
}

impl Scheduler {
    /// Reconciles one physical batch result with logical calls, state, and ownership.
    ///
    /// Stages, in order:
    /// 1. Validate: every returned call identity must be expected by its
    ///    pending batch exactly once, and a batch reported done must have
    ///    returned every call. Any violation latches engine-fatal and fails
    ///    every running request (`fail_all_running`) before any completion is
    ///    applied.
    /// 2. Account: fold domain and batch timing; on the batch's final result,
    ///    settle its lifecycle command receipts.
    /// 3. Queue and apply: hold each completion until
    ///    `Inflight::take_ready_completions` releases it in request-local
    ///    dependency order, then reclaim resources and resolve the public
    ///    effects in lifecycle priority and arrival order.
    pub(in crate::scheduler) fn apply_result(&mut self, report: BatchResult) {
        let result_batch_id = report.batch_id;
        // Aggregate timings by the stable public metric groups, one slot per
        // `ExecutionDomainStats::groups` entry. Multiple concrete calls in one
        // group count as one returned batch, as do multiple requests.
        let mut returned_groups: [Option<TimingCounters>;
            crate::scheduler::stats::ExecutionDomainStats::COUNT] =
            [None; crate::scheduler::stats::ExecutionDomainStats::COUNT];
        let mut invalid_result = false;

        // A completion can mutate state only while its logical batch remains owned
        // by the in-flight window.
        if !self.inflight.pending_batches.contains_key(&result_batch_id) {
            tracing::error!(
                batch_id = result_batch_id,
                "executor returned a result for an unknown logical batch"
            );
            self.fatal = true;
            self.fail_all_running("executor returned a result for an unknown logical batch");
            return;
        }

        // Consume each expected call identity exactly once while folding the
        // rank-local timing records into their scheduler domains.
        for result in &report.results {
            let record = &result.output;
            let identity = (record.request_key, record.call_id);
            let removed = self
                .inflight
                .pending_batches
                .get_mut(&result_batch_id)
                .is_some_and(|pending| pending.calls.remove(&identity));
            if !removed {
                tracing::error!(
                    batch_id = result_batch_id,
                    request_id = record.request_key.request_id.0,
                    call_id = ?record.call_id,
                    "executor returned a duplicate or unknown call result"
                );
                invalid_result = true;
                continue;
            }
            let Some(kind) = self
                .inflight
                .pending_calls
                .get(&record.request_key.request_id)
                .and_then(|calls| {
                    calls
                        .iter()
                        .find(|inflight| inflight.call.call_id == record.call_id)
                })
                .map(|inflight| inflight.call.kind)
            else {
                invalid_result = true;
                continue;
            };
            // A group's batch timing is the maximum of each phase over its calls.
            let component = returned_groups
                [crate::scheduler::stats::ExecutionDomainStats::index(kind)]
            .get_or_insert(TimingCounters::default());
            component.queued_us = component.queued_us.max(record.timing_counters.queued_us);
            component.device_us = component.device_us.max(record.timing_counters.device_us);
            component.copy_us = component.copy_us.max(record.timing_counters.copy_us);
            component.host_us = component.host_us.max(record.timing_counters.host_us);
        }

        let calls_complete = self
            .inflight
            .pending_batches
            .get(&result_batch_id)
            .is_some_and(|batch| batch.calls.is_empty());
        let batch_complete = report.done;
        if batch_complete && !calls_complete {
            invalid_result = true;
        }
        // The page offloads the batch carried are committed when every call
        // ran, and aborted otherwise.
        if batch_complete && let Some(kv_offload) = self.kv_offload.as_mut() {
            let succeeded = report
                .results
                .iter()
                .all(|result| result.output.status != ipc::CallStatus::Error);
            kv_offload.completed(result_batch_id, succeeded && !invalid_result);
        }
        if invalid_result {
            self.fatal = true;
            self.fail_all_running("executor returned an invalid call result");
            return;
        }

        // Publish domain and batch accounting before individual request state is
        // advanced, so every accepted completion contributes exactly once.
        let forward_stats = report.forward_stats;
        for result in &report.results {
            let record = &result.output;
            if let Some(kind) = self
                .inflight
                .pending_calls
                .get(&record.request_key.request_id)
                .and_then(|calls| {
                    calls
                        .iter()
                        .find(|inflight| inflight.call.call_id == record.call_id)
                })
                .map(|inflight| inflight.call.kind)
            {
                self.record_domain_completion(kind, record.status);
            }
        }
        for (index, returned) in returned_groups.into_iter().enumerate() {
            let Some(timing) = returned else {
                continue;
            };
            let (_, stats) = self.stats.domains.groups()[index];
            Self::record_domain_batch(stats, timing);
        }

        let std::collections::hash_map::Entry::Occupied(mut owned) =
            self.inflight.pending_batches.entry(result_batch_id)
        else {
            self.invariant_broken("a validated batch remains owned until reconciliation");
            return;
        };
        let pending = owned.get_mut();
        pending.worker_exec_us = report
            .worker_exec_us
            .into_iter()
            .fold(pending.worker_exec_us, u64::saturating_add);
        let batch_roundtrip_us = pending.started.elapsed().as_micros() as u64;
        // A partial result leaves the batch owned. The final one releases it
        // and settles its command receipts: the recorded commands exclude
        // admissions (`Inflight::register_pending_batch`), matching how
        // `command_index` counts. A failed `Finish` or `Free` keeps its
        // physical ownership and is queued again.
        let worker_us = if batch_complete {
            let pending = owned.remove();
            for (index, command) in pending.commands.into_iter().enumerate() {
                let failed = report.command_results.iter().any(|result| {
                    result.command_index == index as u32
                        && result.outcome == crate::executor::CommandOutcome::Failed
                });
                if failed {
                    match command {
                        BatchCommand::Finish { .. } | BatchCommand::Free { .. } => {
                            self.inflight.pending_commands.push_back(command);
                        }
                        BatchCommand::Start { .. } => {}
                    }
                } else {
                    self.acknowledge_commands(std::slice::from_ref(&command));
                }
            }
            pending.worker_exec_us
        } else {
            0
        };

        if batch_complete {
            self.stats
                .timing
                .last_worker_exec_us
                .store(worker_us, Ordering::Relaxed);
        }
        if batch_complete {
            self.stats
                .timing
                .worker_exec_us_total
                .fetch_add(worker_us, Ordering::Relaxed);
            self.stats
                .timing
                .batch_roundtrip_us_total
                .fetch_add(batch_roundtrip_us, Ordering::Relaxed);
            self.stats
                .timing
                .batch_timing_count
                .fetch_add(1, Ordering::Relaxed);
        }

        for stats in &forward_stats {
            self.record_worker_forward_stats(Some(stats));
        }

        // Queuing decouples executor arrival order from request-local dependency
        // order. `take_ready_completions` releases only completions whose call
        // is at the front of its request's queue, or independent media calls
        // whose input producers have resolved; applying one can make others
        // ready, so the loop repeats until none are.
        for result in report.results {
            self.queue_completion(result.output, result.media);
        }

        loop {
            let completions = self.inflight.take_ready_completions();
            if completions.is_empty() {
                break;
            }

            let mut to_resolve = Vec::with_capacity(completions.len());
            for completion in completions {
                let PendingCompletion {
                    record,
                    media,
                    arrival_seq,
                } = completion;
                let id = record.request_key.request_id;
                let call_id = record.call_id;
                let Some((call, input)) = self.pop_inflight(record.request_key, call_id) else {
                    if let Some(state) = self.media_state_mut(id) {
                        state.terminal_intent = TerminalIntent::Failure(
                            "worker returned an out-of-order media call".to_string(),
                        );
                    } else if self.running.contains_key(&id) {
                        self.finish(id, FinishReason::Error);
                    }
                    continue;
                };

                let call_variant = call.kind;

                // Media calls update their independent call progress immediately;
                // generation calls continue through semantic validation.
                let image_cache = match input {
                    InflightInput::Media { decode } => {
                        self.process_diffusion_result(call, decode, record, media);
                        continue;
                    }
                    InflightInput::Generation { image_cache } => image_cache,
                };

                // A request already finishing with an error discards the
                // result's products.
                if self
                    .inflight
                    .pending_finishes
                    .get(&id)
                    .is_some_and(|finish| finish.reason == FinishReason::Error)
                {
                    self.free_buffers(call.output_buffers());
                    self.finish_pending_if_idle(id);
                    continue;
                }

                // A descendant submitted before its chain was invalidated is
                // discarded: `num_cache_units_sent` drops by its `max_cache_units`
                // bound, so the next dispatch declares those units again, its
                // products are freed, and the chain stays marked while later
                // descendants remain in flight.
                let discard_invalidated_descendant = self
                    .running
                    .get(&id)
                    .is_some_and(|state| state.speculative_chain_invalidated);
                if discard_invalidated_descendant {
                    let has_unresolved_descendants = self.inflight.has_pending_calls(id);
                    if let Some(state) = self.running.get_mut(&id) {
                        state.num_cache_units_sent = state
                            .num_cache_units_sent
                            .saturating_sub(u64::from(call.bounds.max_cache_units));
                        state.speculative_chain_invalidated = has_unresolved_descendants;
                    }
                    self.free_buffers(call.output_buffers());
                    self.finish_pending_if_idle(id);
                    continue;
                }

                let Some(state) = self
                    .running
                    .get(&id)
                    .filter(|state| state.request_epoch == call.request_key.request_epoch)
                else {
                    // Finish already owns retirement for a closed epoch, including
                    // outputs still being produced when cancellation arrived.
                    continue;
                };
                if let Err(error) = generation::validate_generation_result(
                    &call,
                    image_cache,
                    state,
                    &record,
                    self.config.canvas.as_ref(),
                ) {
                    self.fail_after_inflight(id, rejection_cause(&call, &error));
                    continue;
                }

                // Completion products used only as device predicates can be released
                // once no in-flight descendant refers to them.
                if record.status == CallStatus::Ok && !call.advances_state() {
                    let completion_has_device_consumer =
                        self.inflight.pending_calls.get(&id).is_some_and(|queue| {
                            queue.iter().any(|inflight| {
                                inflight
                                    .call
                                    .predicate
                                    .as_ref()
                                    .as_ref()
                                    .is_some_and(|predicate| {
                                        predicate.producer_call_id == call.call_id
                                    })
                            })
                        });
                    if !completion_has_device_consumer {
                        let completed_predicates = call
                            .completion_output
                            .iter()
                            .chain(call.transition_output.iter())
                            .cloned()
                            .collect::<Vec<_>>();
                        self.free_buffers(
                            completed_predicates
                                .into_iter()
                                .map(|product| product.buffer_id()),
                        );
                    }
                }

                // A request with a terminal intent or a deferred finish takes no
                // further semantic effect; an image decode still releases the
                // request latent it closes.
                let semantic_blocked = self
                    .running
                    .get(&id)
                    .is_some_and(|state| state.terminal_intent.is_terminal())
                    || self.inflight.pending_finishes.contains_key(&id);
                if semantic_blocked {
                    if call.kind == CallKind::Media(MediaCall::ImageDecoding) {
                        self.free_request_latent(id);
                    }
                    self.finish_pending_if_idle(id);
                    continue;
                }

                let progress_result = if record.status == CallStatus::Ok {
                    self.running
                        .get_mut(&id)
                        .map(|state| state.process_generation_result(&call, &record))
                } else {
                    None
                };
                if let Some(Err(error)) = progress_result {
                    self.fail_after_inflight(id, rejection_cause(&call, &error));
                    continue;
                }
                // An accepted hand-off install releases the prefill group's
                // export it read.
                if progress_result.is_some()
                    && call.kind == CallKind::Transfer(TransferMode::CacheInstall)
                {
                    self.free_buffers(call.cache_input);
                }
                // An accepted forward extends the request's prefix cache: publish the
                // prompt pages it completed, then retire sliding-window pages
                // no later reader needs.
                if progress_result.is_some() && matches!(call.kind, CallKind::Forward(_)) {
                    self.advance_request_cache(id);
                }

                // Apply accepted progress directly; the output decoder separately
                // decides the user-visible prefix and stop-string boundary.
                let advanced = record.status == CallStatus::Ok && call.advances_state();
                let latest_token = if advanced {
                    call.token_output.clone()
                } else {
                    None
                };
                // The resolved token product stays available as the next
                // decode's device input (`can_reuse_resolved_token_product`)
                // only when no call of the request is still in flight.
                let retain_device_token = !self.inflight.has_pending_calls(id);
                if let Some(state) = self.running.get_mut(&id)
                    && advanced
                {
                    state.last_state_call_id = record.call_id;
                    state.latest_token = if retain_device_token {
                        latest_token
                    } else {
                        None
                    };
                }

                // Reclaim resources whose lifetime ends at this transition before
                // making its public output eligible for resolution.
                let free_guidance_prefix = call_variant == CallKind::Media(MediaCall::Denoising)
                    && record.status == CallStatus::Ok
                    && self
                        .running
                        .get(&id)
                        .is_some_and(|state| record.step_index >= state.denoising.steps());
                if call_variant == CallKind::Media(MediaCall::Denoising)
                    && record.status == CallStatus::Ok
                    && let Some(prefix) = self
                        .running
                        .get_mut(&id)
                        .and_then(|state| state.guidance_prefix.as_mut())
                {
                    prefix.diffusion_finalized = true;
                }
                if call.kind == CallKind::Media(MediaCall::ImageDecoding) {
                    self.free_request_latent(id);
                }
                if free_guidance_prefix {
                    self.free_guidance_prefix(id);
                }
                if matches!(
                    call_variant,
                    CallKind::Media(MediaCall::Denoising)
                        | CallKind::Media(MediaCall::ImageDecoding)
                ) {
                    let consumed_latents = call.latent_input.iter().cloned().collect::<Vec<_>>();
                    if !consumed_latents.is_empty() {
                        self.free_buffers(
                            consumed_latents
                                .into_iter()
                                .map(|product| product.buffer_id()),
                        );
                    }
                }

                let priority = completion_priority(call_variant);
                let public_tokens_before = self
                    .running
                    .get(&id)
                    .map_or(0, |state| state.output.tokens_sent);
                // A predicated call was skipped because its device predicate
                // was false, so it is rolled back like an invalidated
                // descendant and marks any later descendants invalidated.
                // A step queued behind the step that stopped its block is the
                // exception: the host knew it for a no-op once it accepted the
                // stopping step, and every call queued since, the block's
                // commit and the next block's steps, was planned from that
                // host-observed stop (`follows_unconditionally`).
                if record.status == CallStatus::Predicated {
                    let has_unresolved_descendants = self.inflight.has_pending_calls(id);
                    if let Some(state) = self.running.get_mut(&id) {
                        state.num_cache_units_sent = state
                            .num_cache_units_sent
                            .saturating_sub(u64::from(call.bounds.max_cache_units));
                        let stop_observed = state.phase == Phase::CommitCanvas
                            && call
                                .canvas
                                .is_some_and(|step| step.block == state.canvas_block);
                        if !stop_observed {
                            state.speculative_chain_invalidated = has_unresolved_descendants;
                        }
                    }
                    self.free_buffers(call.output_buffers());
                    self.finish_pending_if_idle(id);
                } else {
                    to_resolve.push((
                        priority,
                        arrival_seq,
                        id,
                        call,
                        record,
                        media,
                        public_tokens_before,
                    ));
                }
            }

            // Resolve semantic effects in lifecycle priority and stable arrival
            // order, independent of the executor's physical completion order.
            to_resolve.sort_by_key(|(priority, seq_index, ..)| (*priority, *seq_index));
            for (_priority, _seq_index, id, call, record, media, public_tokens_before) in to_resolve
            {
                let token_call = matches!(
                    call.kind,
                    CallKind::Forward(ForwardMode::Prefill)
                        | CallKind::Forward(ForwardMode::Decode)
                        | CallKind::Forward(ForwardMode::Verify)
                ) || call.canvas.is_some();
                // With stop strings, push a placeholder decoder boundary before
                // resolving, so a non-error finish reached while resolving
                // waits for the frontend decoder's stop-string decision on this
                // call's tokens (see `finish_after_inflight`). Afterwards the
                // placeholder becomes the end of the tokens this call
                // published, or is removed when it published none.
                let awaiting_decoder = token_call
                    && match self.running.get_mut(&id) {
                        Some(state) if !state.req.stop_strings.is_empty() => {
                            state.output.decoder_boundaries.push_back(usize::MAX);
                            true
                        }
                        _ => false,
                    };
                if self.running.contains_key(&id)
                    && !self.inflight.pending_finishes.contains_key(&id)
                {
                    self.resolve(id, call, record, media.as_deref());
                }
                if awaiting_decoder && let Some(state) = self.running.get_mut(&id) {
                    // Cancellation clears the boundaries while resolving, which
                    // leaves no reserved boundary to settle.
                    if state.output.tokens_sent > public_tokens_before {
                        if let Some(boundary) = state.output.decoder_boundaries.back_mut() {
                            *boundary = state.output.tokens_sent;
                        }
                    } else {
                        state.output.decoder_boundaries.pop_back();
                    }
                }
                self.finish_pending_if_idle(id);
            }
        }
    }

    /// Merges optional worker forward-pass counters into scheduler statistics.
    pub(in crate::scheduler) fn record_worker_forward_stats(&self, stats: Option<&ForwardStats>) {
        let Some(stats) = stats else {
            return;
        };
        add_worker_forward_map(&self.stats.worker.forward_mode_counts, &stats.mode_counts);
        add_worker_forward_map(&self.stats.worker.forward_mode_tokens, &stats.mode_tokens);
        add_worker_forward_map(&self.stats.worker.forward_mode_us, &stats.mode_us);
        add_worker_forward_map(&self.stats.worker.forward_component_us, &stats.component_us);
        add_worker_forward_map(
            &self.stats.worker.attention_backend_counts,
            &stats.attention_backend_counts,
        );
        add_worker_forward_map(
            &self.stats.worker.cuda_graph_runtime_mode_counts,
            &stats.cuda_graph_runtime_mode_counts,
        );
        self.stats
            .worker
            .attention_launches
            .fetch_add(stats.attention_launches, Ordering::Relaxed);
        self.stats
            .worker
            .attention_us
            .fetch_add(stats.attention_us, Ordering::Relaxed);
        self.stats
            .worker
            .cuda_graph_captures
            .fetch_add(stats.cuda_graph_captures, Ordering::Relaxed);
        self.stats
            .worker
            .cuda_graph_replays
            .fetch_add(stats.cuda_graph_replays, Ordering::Relaxed);
        self.stats
            .worker
            .cuda_graph_misses
            .fetch_add(stats.cuda_graph_misses, Ordering::Relaxed);
        self.stats
            .worker
            .cuda_graph_fallbacks
            .fetch_add(stats.cuda_graph_fallbacks, Ordering::Relaxed);
        self.stats
            .worker
            .cuda_graph_unpadded_tokens
            .fetch_add(stats.cuda_graph_unpadded_tokens, Ordering::Relaxed);
        self.stats
            .worker
            .cuda_graph_padded_tokens
            .fetch_add(stats.cuda_graph_padded_tokens, Ordering::Relaxed);
        self.stats
            .worker
            .text_decode_token_relay_hits
            .fetch_add(stats.text_decode_token_relay_hits, Ordering::Relaxed);
        self.stats
            .worker
            .text_decode_token_relay_misses
            .fetch_add(stats.text_decode_token_relay_misses, Ordering::Relaxed);
        self.stats
            .worker
            .text_decode_position_relay_hits
            .fetch_add(stats.text_decode_position_relay_hits, Ordering::Relaxed);
        self.stats
            .worker
            .text_decode_position_relay_misses
            .fetch_add(stats.text_decode_position_relay_misses, Ordering::Relaxed);
        self.stats
            .worker
            .flashinfer_decode_plan_calls
            .fetch_add(stats.flashinfer_decode_plan_calls, Ordering::Relaxed);
        self.stats
            .worker
            .flashinfer_decode_plan_reuses
            .fetch_add(stats.flashinfer_decode_plan_reuses, Ordering::Relaxed);
        self.stats
            .worker
            .flashinfer_decode_plan_rows
            .fetch_add(stats.flashinfer_decode_plan_rows, Ordering::Relaxed);
        self.stats
            .worker
            .flashinfer_decode_plan_indices
            .fetch_add(stats.flashinfer_decode_plan_indices, Ordering::Relaxed);
        self.stats
            .worker
            .flashinfer_decode_graph_plan_calls
            .fetch_add(stats.flashinfer_decode_graph_plan_calls, Ordering::Relaxed);
        self.stats
            .worker
            .flashinfer_decode_graph_plan_reuses
            .fetch_add(stats.flashinfer_decode_graph_plan_reuses, Ordering::Relaxed);
        self.stats
            .worker
            .spec_verify_rows
            .fetch_add(stats.spec_verify_rows, Ordering::Relaxed);
        self.stats
            .worker
            .spec_verify_draft_tokens
            .fetch_add(stats.spec_verify_draft_tokens, Ordering::Relaxed);
        self.stats
            .worker
            .spec_verify_accepted_tokens
            .fetch_add(stats.spec_verify_accepted_tokens, Ordering::Relaxed);
        self.stats
            .worker
            .spec_verify_rejected_tokens
            .fetch_add(stats.spec_verify_rejected_tokens, Ordering::Relaxed);
        self.stats
            .worker
            .spec_verify_committed_tokens
            .fetch_add(stats.spec_verify_committed_tokens, Ordering::Relaxed);
        if !stats.spec_verify_path_counts.is_empty() {
            let mut path_counts = self
                .stats
                .worker
                .spec_verify_path_counts
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for (path, count) in &stats.spec_verify_path_counts {
                *path_counts.entry(path.clone()).or_default() += *count;
            }
        }
    }
}
