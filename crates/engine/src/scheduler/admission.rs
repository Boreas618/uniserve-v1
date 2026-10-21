//! Request admission, resource reservation, and waiting-queue insertion.
//!
//! Submission (`enqueue`, `enqueue_media`) validates a request and checks it
//! against the loaded worker's capabilities and the waiting-queue bound. It
//! either queues the request or rejects it synchronously with
//! `Event::Rejected` on the request's event channel. Admission
//! (`admit`, `admit_media`), which the scheduling pass runs before and
//! between batch assembly, moves requests from the head of the token and
//! media waiting queues into the running set while `max_running_requests` and
//! storage allow. A head that does not fit yet blocks the requests behind it
//! in its queue, and running requests are never preempted to make room.
//!
//! A token request reserves a request row and one cache table per cache group;
//! every group draws units from one pool, so admission compares the sum of
//! the groups' unit demand with the free units. Its prefill worker is
//! recorded in `Placement::affinity`. A media request
//! reserves a request row and every declared result buffer on each worker of
//! its route.

use super::*;
use ipc::ForwardMode;

/// One product a video request reserves: its declared name, the component
/// output it is reserved under, and its request-sized shape.
pub(super) struct MediaOutput {
    pub(super) name: &'static str,
    pub(super) component: String,
    pub(super) index: u32,
    pub(super) dtype: DType,
    pub(super) shape: ShapeBound,
}

/// Why the loaded media components cannot hold a request's product.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(super) enum MediaOutputError {
    /// No loaded component serves the call that publishes the product.
    #[error("no loaded component serves {0:?}")]
    Unserved(MediaCall),
    /// The serving component has no placement candidate.
    #[error("component {0} has no loaded candidate")]
    Unplaced(String),
    /// The candidate's worker does not bind the component.
    #[error("component {0} is not bound on its worker")]
    Unbound(String),
    /// The component declares no output of the product's name.
    #[error("component {component} declares no {product} output")]
    Undeclared {
        component: String,
        product: &'static str,
    },
    /// A raster-laid output lacks a static axis for one side of the canvas.
    #[error("{product} of component {component} has no static {side} axis")]
    RasterAxis {
        component: String,
        product: &'static str,
        side: &'static str,
    },
    /// The request's canvas side is zero or exceeds the declared extent.
    #[error(
        "{product} of component {component} holds a {side} of at most {max} pixels, the request's canvas has {extent}"
    )]
    Raster {
        component: String,
        product: &'static str,
        side: &'static str,
        max: u32,
        extent: u32,
    },
    /// A request-sized product's output has no device-sized axis.
    #[error("{product} of component {component} has no request-sized axis")]
    RequestAxis {
        component: String,
        product: &'static str,
    },
    /// The request's extent is zero or exceeds the device-sized maximum.
    #[error("{product} of component {component} holds 1 to {max}, the request needs {extent}")]
    Request {
        component: String,
        product: &'static str,
        max: u32,
        extent: u32,
    },
}

impl Scheduler {
    /// Validates and queues one token-generation request or rejects it synchronously.
    ///
    /// Rejects with `RejectionKind::Invalid` when `validate_submission`
    /// refuses the request, and with `RejectionKind::Overloaded` when the
    /// waiting bound is reached. A rejection is sent on `event_tx` and its
    /// send result is ignored.
    pub(super) fn enqueue(&mut self, req: GenerationRequest, event_tx: EventTx) {
        let max_cache_tokens = match self.validate_submission(&req) {
            Ok(tokens) => tokens,
            Err(message) => {
                let _ = event_tx.send(Event::Rejected {
                    kind: RejectionKind::Invalid,
                    message,
                });
                return;
            }
        };
        // Backpressure: under overload the new submission is rejected rather
        // than letting queued requests grow without bound and exhaust process
        // memory. The bound covers both waiting queues and retired requests
        // whose events are still undelivered; `enqueue_media` applies the
        // same count.
        let waiting = self.pending_request_count() + self.output.retained_len();
        if self
            .config
            .max_queued_requests
            .is_some_and(|bound| waiting >= bound)
        {
            let _ = event_tx.send(Event::Rejected {
                kind: RejectionKind::Overloaded,
                message: "scheduler waiting queue is full".into(),
            });
            return;
        }
        let st = self.request_state(req, event_tx, max_cache_tokens);

        // Admission examines only the queue head, so the insertion position
        // fixes the admission order. Under `Priority` a request goes after
        // every queued request with a lower priority value, or an equal one
        // and an earlier or equal arrival, so ties stay FIFO.
        let position = match self.config.policy {
            SchedulingPolicy::Fcfs => self.waiting.len(),
            SchedulingPolicy::Priority => self.waiting.partition_point(|queued| {
                (queued.req.priority, queued.queued_at) <= (st.req.priority, st.queued_at)
            }),
        };
        self.waiting.insert(position, st);
    }

    /// Checks a token-generation request against the loaded workers and
    /// returns its worst-case cache token count.
    ///
    /// Refuses, with the client's message, a request when the runtime has no
    /// prefix cache, the request fails validation, a unit the scheduler never
    /// splits (a readout row, a generation canvas, an input image's vision
    /// block) exceeds the step token budget, the worker lacks a required
    /// generation feature, or `validate_resources` or `max_cache_tokens`
    /// fails against `generation_limits`.
    pub(super) fn validate_submission(&self, req: &GenerationRequest) -> Result<usize, String> {
        if self.storage.cache.is_none() {
            return Err("generation request requires worker KV resources".into());
        }
        if let Err(error) = req.validate() {
            return Err(format!("invalid generation request: {error:?}"));
        }
        // A canvas pass holds whole rows, so every readout row must fit one
        // scheduling step's token budget.
        if let Some(row) = req
            .readout
            .iter()
            .find(|row| row.token_ids.len() > self.config.max_num_batched_tokens)
        {
            return Err(format!(
                "a {}-token readout canvas exceeds the {}-token step budget",
                row.token_ids.len(),
                self.config.max_num_batched_tokens
            ));
        }
        // A request the deployment generates in canvases generates text only
        // and names the seed its draws follow.
        if self.config.canvas.is_some() && !req.is_readout() {
            let error = if req.generates_images() {
                Some(GenerationRequestError::CanvasGeneratesTextOnly)
            } else if req.sampling.seed.is_none() {
                Some(GenerationRequestError::UnseededCanvas)
            } else {
                None
            };
            if let Some(error) = error {
                return Err(format!("invalid generation request: {error:?}"));
            }
            // A canvas's committed blocks are tokens, but the denoising state
            // of the block in progress is not.
            if req.is_migrated() {
                return Err(
                    "a block-diffusion request does not resume from generated tokens".into(),
                );
            }
        }
        // A context prefill writes an input image's vision block whole, as
        // one attention block of one call, so the block's exact cache length
        // must fit one scheduling step's token budget.
        if let Some(tokens) = req
            .multimodal_inputs
            .images
            .iter()
            .flat_map(|image| &image.encoders)
            .filter(|input| input.encoder == crate::ImageEncoder::Vision)
            .filter_map(|input| input.num_cache_tokens)
            .find(|&tokens| tokens as usize > self.config.max_num_batched_tokens)
        {
            return Err(format!(
                "a {}-token input image vision block exceeds the {}-token step budget",
                tokens, self.config.max_num_batched_tokens
            ));
        }
        if let Some(feature) = self.missing_required_feature(req) {
            return Err(format!(
                "generation request requires worker feature `{feature}`, but the worker does not support it"
            ));
        }
        // Resource validation includes the cache bound, so a request that passes
        // it has a worst-case cache token count.
        req.validate_resources(&self.generation_limits)
            .and_then(|()| req.max_cache_tokens(&self.generation_limits))
            .map(|tokens| tokens.saturating_add(self.image_unconditional_tokens(req)))
            .map_err(|error| format!("invalid generation resource requirements: {error}"))
    }

    /// The state of a validated request, at the next request epoch, which
    /// publishes through `event_tx`. `max_cache_tokens` is its worst-case
    /// cache token count (`validate_submission`).
    pub(super) fn request_state(
        &mut self,
        mut req: GenerationRequest,
        event_tx: EventTx,
        max_cache_tokens: usize,
    ) -> RequestState {
        // A migrated request's generated tokens continue its prompt: the
        // engine computes their cache as it computes the prompt's and samples
        // after the last of them, at the position the request had reached.
        // They count as generated from the start, so the token limit,
        // `min_tokens`, the penalties and the bad words continue where they
        // were; its usage charges only its own prompt as prompt
        // (`RequestState::charged_prompt_tokens`).
        req.prompt_token_ids
            .extend_from_slice(&req.generated_token_ids);
        let generated_token_ids = req.generated_token_ids.clone();
        // Multimodal requests reserve their configured bounded cache envelope at
        // admission so excess concurrency queues instead of exhausting the prefix cache.
        let reserve_worstcase = !req.multimodal_inputs.images.is_empty() || req.generates_images();
        let finish_token_ids = finish_token_ids(&req, &self.special_tokens.eos);
        let st = RequestState {
            finish_token_ids,
            allocations: None,
            guidance_prefix: None,
            request_epoch: self.next_request_epoch,
            last_state_call_id: CallId::default(),
            latest_token: None,
            speculative_chain_invalidated: false,
            // Every request starts in prefill. A request with context images
            // encodes each image's blocks ahead of the context prefill that
            // writes them into their marker gap with the text around them; a
            // latent block is written by a call of its own.
            phase: Phase::Prefill,
            num_computed_prompt_tokens: 0,
            num_ingested_images: 0,
            image_encoder_index: 0,
            input_image_features: None,
            context_features: Vec::new(),
            round_closing: false,
            logical_position: 0,
            cache_visible_len: 0,
            cache_computed_len: 0,
            next_token: 0,
            num_generated_tokens: generated_token_ids.len(),
            image_id: 0,
            num_generated_images: 0,
            image_reservation_pending: false,
            denoising: Denoising::new(u32::from(req.image.steps)),
            image_conditioning: None,
            feedback_encoder_index: 0,
            feedback_source: None,
            feedback_features: None,
            worker_registered: false,
            num_cache_units_sent: 0,
            reserve_worstcase,
            // `admit` reserves the units of this worst-case extent only for
            // requests with `reserve_worstcase` set.
            max_reserved_cache_tokens: max_cache_tokens,
            prefix_page_hashes: Vec::new(),
            prefix_published: Vec::new(),
            page_keys: Vec::new(),
            offload_boundary: 0,
            generated_token_ids,
            round_token_ids: Vec::new(),
            feedback_image_b64: None,
            replayable: true,
            encoder_cache_pins: Vec::new(),
            transient_encoder_products: Vec::new(),
            readout_rows: 0,
            readout_logprobs: Vec::with_capacity(req.readout_candidates()),
            canvas_block: 0,
            canvas_step: 0,
            canvas_commit: Vec::new(),
            output: RequestOutput::new(event_tx),
            queued_at: now(),
            terminal_intent: super::TerminalIntent::None,
            failure: None,
            req,
        };
        self.next_request_epoch = self.next_request_epoch.saturating_add(1);
        st
    }

    /// Resolves the products a video request reserves at admission.
    ///
    /// Each product of the request's graph is found by name among the
    /// outputs of the component serving its call, as bound on that call's
    /// first placement candidate. A product with a request extent takes it
    /// as the extent of its output's device-sized axis, which must not exceed
    /// the declared maximum; a product without one keeps its declared
    /// extent there. An output declaring raster axes takes the request's
    /// canvas height and width in them, each at most the declared extent.
    /// Returns one entry per product: its name, component, output index,
    /// dtype and shape bound.
    ///
    /// # Errors
    ///
    /// Returns a [`MediaOutputError`] naming the product when no component
    /// serves its call, the component declares no output of that name, the output has
    /// no device-sized axis for a request extent, or the request's extent or
    /// canvas is zero or exceeds the declared maximum. `enqueue_media`
    /// rejects such a request, and `admit_media` calls this again to size the
    /// reservation.
    fn media_outputs(&self, graph: &VideoGraph) -> Result<Vec<MediaOutput>, MediaOutputError> {
        let mut outputs = Vec::new();
        for product in graph.products() {
            let name = self
                .info
                .media_components
                .get(&product.call)
                .ok_or(MediaOutputError::Unserved(product.call))?;
            let (_, bound, info) = self
                .placement
                .component_candidates(self.executor.as_ref(), CallKind::Media(product.call), name)
                .next()
                .ok_or_else(|| MediaOutputError::Unplaced(name.clone()))?;
            let component = info
                .components
                .iter()
                .find(|component| component.name == bound)
                .ok_or_else(|| MediaOutputError::Unbound(bound.to_owned()))?;
            let (index, output) = component
                .outputs
                .iter()
                .enumerate()
                .find(|(_, output)| output.name == product.name)
                .ok_or_else(|| MediaOutputError::Undeclared {
                    component: name.clone(),
                    product: product.name,
                })?;
            let mut shape = output.shape_bound.clone();
            // A product laid out at the request's raster holds the canvas in
            // its raster axes, whose declared extents are the largest the
            // worker admits.
            if let Some(raster) = output.raster_axes {
                let canvas = graph.canvas();
                for (axis, extent, side) in [
                    (raster.height, canvas.height, "height"),
                    (raster.width, canvas.width, "width"),
                ] {
                    let Some(DimBound::Static(max)) = shape.dims.get(axis as usize).copied() else {
                        return Err(MediaOutputError::RasterAxis {
                            component: name.clone(),
                            product: product.name,
                            side,
                        });
                    };
                    if extent == 0 || extent > max {
                        return Err(MediaOutputError::Raster {
                            component: name.clone(),
                            product: product.name,
                            side,
                            max,
                            extent,
                        });
                    }
                    shape.dims[axis as usize] = DimBound::Static(extent);
                }
            }
            if let Some(extent) = product.extent {
                let dim = shape
                    .dims
                    .iter_mut()
                    .find(|dim| matches!(dim, DimBound::Device { .. }))
                    .ok_or_else(|| MediaOutputError::RequestAxis {
                        component: name.clone(),
                        product: product.name,
                    })?;
                let DimBound::Device { max } = *dim else {
                    unreachable!("the selected axis is device-sized");
                };
                if extent == 0 || extent > max {
                    return Err(MediaOutputError::Request {
                        component: name.clone(),
                        product: product.name,
                        max,
                        extent,
                    });
                }
                *dim = DimBound::Static(extent);
            }
            outputs.push(MediaOutput {
                name: product.name,
                component: name.clone(),
                index: index as u32,
                dtype: output.dtype,
                shape,
            });
        }
        Ok(outputs)
    }

    /// Chooses one physical owner for every component a media request's
    /// calls run on.
    ///
    /// The denoiser is selected first because it defines the expensive replica
    /// residency. Components co-located with that worker follow it; shared
    /// components such as a TP text encoder and the host codec worker retain
    /// their own independently bounded request rows. Otherwise the candidate
    /// with the fewest distinct requests placed on it in `Placement::affinity`
    /// wins, then the one with the most free request rows.
    ///
    /// Returns a map from component name to worker, or `None` when some
    /// component has no ready candidate that is already on the route or has
    /// a free request row. When the executor reports admissible video
    /// codecs for the video decoder's chosen worker, the video codec's
    /// candidates are further limited to those.
    fn media_routes(&self, graph: &VideoGraph) -> Option<HashMap<String, crate::WorkerGroupId>> {
        use ipc::MediaCall;

        let mut required = graph
            .calls()
            .iter()
            .filter_map(|call| {
                self.info
                    .media_components
                    .get(call)
                    .map(|component| (*call, component.clone()))
            })
            .collect::<Vec<_>>();
        // Routing order: the denoiser first, then the calls that feed it, then
        // decoders before encoders, since the video codec's admissible
        // workers depend on the route chosen for the video decoder.
        required.sort_by_key(|(call, _)| match call {
            MediaCall::Denoising => 0,
            MediaCall::MediaReading
            | MediaCall::VisionEncoding
            | MediaCall::LatentEncoding
            | MediaCall::TextEncoding
            | MediaCall::LatentPreparation => 1,
            MediaCall::VideoDecoding | MediaCall::AudioDecoding => 2,
            MediaCall::VideoEncoding | MediaCall::AudioEncoding | MediaCall::Muxing => 3,
            _ => 4,
        });
        let mut routes = HashMap::new();
        let mut selected_workers = HashSet::new();
        for (call, component) in required {
            // A component serving several media calls is routed once.
            if routes.contains_key(&component) {
                continue;
            }
            // A request's media units are encoded where they were decoded, so
            // its encoder is one of the replicas admissible for its decoder.
            let admissible = (call == MediaCall::VideoEncoding)
                .then(|| {
                    let decoder = self.info.media_components.get(&MediaCall::VideoDecoding)?;
                    let worker = routes.get(decoder)?;
                    self.executor.info().video_codecs.get(worker)
                })
                .flatten();
            let candidates = self
                .placement
                .component_candidates(self.executor.as_ref(), CallKind::Media(call), &component)
                .filter(|(worker, _, _)| {
                    admissible.is_none_or(|encoders| encoders.contains(*worker))
                        && self.executor.is_ready(worker)
                        && (selected_workers.contains(*worker)
                            || self
                                .storage
                                .media_storage
                                .get(*worker)
                                .is_some_and(|storage| storage.requests.available() > 0))
                })
                .collect::<Vec<_>>();

            // A worker already on the route is preferred, so co-located
            // components follow the denoiser. It needs no further free row
            // because `reserve_media` takes one request row per distinct
            // worker.
            let resident = candidates
                .iter()
                .copied()
                .find(|(worker, _, _)| selected_workers.contains(*worker));
            let chosen = resident.or_else(|| {
                candidates.into_iter().min_by_key(|(worker, _, _)| {
                    let active = self
                        .placement
                        .affinity
                        .iter()
                        .filter_map(|((request, _), owner)| (owner == *worker).then_some(*request))
                        .collect::<HashSet<_>>()
                        .len();
                    let available = self.storage.media_storage[*worker].requests.available();
                    (active, usize::MAX - available)
                })
            })?;
            selected_workers.insert(chosen.0.clone());
            routes.insert(component.clone(), chosen.0.clone());
        }
        Some(routes)
    }

    /// Validates and queues one video request.
    ///
    /// Rejects with `RejectionKind::Invalid` when the request fails
    /// validation; the worker reports no muxing component or no video
    /// denoiser; the denoiser does not serve the request's task; its call
    /// graph cannot be built; or `media_outputs` cannot bound its products. A
    /// served task's calls all have components: the executor checked them at
    /// startup. The two resident request slots a video decoder needs are the
    /// rank's to require: it refuses to start with fewer. Rejects with
    /// `RejectionKind::Overloaded` when the waiting bound is reached. A
    /// rejection's send result is ignored.
    pub(super) fn enqueue_media(
        &mut self,
        request: DiffusionRequest,
        event_tx: EventTx,
        queued_at: f64,
    ) {
        let reject = |kind: RejectionKind, message: String| {
            let _ = event_tx.send(Event::Rejected { kind, message });
        };
        if let Err(message) = request.validate() {
            reject(RejectionKind::Invalid, message.to_string());
            return;
        }
        // A worker reports the component serving each media call it
        // implements, so a deployment that assembles an artifact is the one
        // that can serve a video request; reporting some call is not enough.
        if !self
            .info
            .media_components
            .contains_key(&ipc::MediaCall::Muxing)
        {
            reject(
                RejectionKind::Invalid,
                "worker does not provide the video media components".to_string(),
            );
            return;
        }
        let Some(denoiser) = self.info.video_denoiser.as_ref() else {
            reject(
                RejectionKind::Invalid,
                "worker reports no video denoiser".to_string(),
            );
            return;
        };
        let Some(task) = denoiser.task(&request.task) else {
            reject(
                RejectionKind::Invalid,
                format!("the deployment's denoiser does not serve {}", request.task),
            );
            return;
        };
        if task.conditioned == request.conditions.is_empty() {
            reject(
                RejectionKind::Invalid,
                format!("the conditions do not fit the task {}", request.task),
            );
            return;
        }
        let graph = match VideoGraph::new(&request) {
            Ok(graph) => graph,
            Err(error) => {
                reject(RejectionKind::Invalid, error.to_string());
                return;
            }
        };
        if let Err(error) = self.media_outputs(&graph) {
            reject(
                RejectionKind::Invalid,
                format!("loaded media components cannot hold the request: {error}"),
            );
            return;
        }
        let waiting = self.waiting.len() + self.waiting_media.len() + self.output.retained_len();
        if self
            .config
            .max_queued_requests
            .is_some_and(|bound| waiting >= bound)
        {
            reject(
                RejectionKind::Overloaded,
                "scheduler waiting queue is full".to_string(),
            );
            return;
        }
        self.waiting_media.push_back(PendingMedia {
            request,
            graph,
            event_tx,
            queued_at,
        });
    }

    /// Reserves a media request's row and output buffers on every routed worker.
    ///
    /// Every declared output is reserved in full in the buffer pool of each
    /// rank that backs it on each routed worker, not only on the worker whose
    /// component produces it; spans are sized from the shape bound's maximum
    /// element count and 256-byte aligned.
    ///
    /// Returns `Ok(None)`, with nothing reserved, when a request row or buffer
    /// span cannot be allocated on some rank now; `admit_media` then leaves
    /// the request at the head of the queue. Returns `Err` when a routed
    /// worker has no media storage, after releasing what was already reserved
    /// on known workers.
    fn reserve_media(
        &mut self,
        route_workers: &HashSet<crate::WorkerGroupId>,
        request_key: RequestKey,
        outputs: &[MediaOutput],
    ) -> Result<Option<MediaAllocations>, UnknownMediaWorker> {
        let mut reserved = MediaAllocations {
            request_slots: HashMap::new(),
            tensors: HashMap::new(),
        };
        for worker in route_workers {
            let Some(storage) = self.storage.media_storage.get_mut(worker) else {
                reserved.free(&mut self.storage)?;
                return Err(UnknownMediaWorker(worker.clone()));
            };
            let Ok(slot) = storage.requests.allocate() else {
                reserved.free(&mut self.storage)?;
                return Ok(None);
            };
            reserved.request_slots.insert(worker.clone(), slot);
        }
        for output in outputs {
            let (component, index) = (output.component.clone(), output.index);
            let bytes = output
                .shape
                .max_elements()
                .saturating_mul(output.dtype.element_bytes());
            let mut tensor = MediaTensorAllocation {
                allocations: HashMap::new(),
                dtype: output.dtype,
                shape_bound: output.shape.clone(),
            };
            for worker in route_workers {
                let Some(storage) = self.storage.media_storage.get_mut(worker) else {
                    reserved.tensors.insert((component, index), tensor);
                    reserved.free(&mut self.storage)?;
                    return Err(UnknownMediaWorker(worker.clone()));
                };
                for rank in storage.holders(&component).to_vec() {
                    let Ok(buffer) =
                        storage.buffers[rank as usize].allocate(request_key, bytes, 256)
                    else {
                        reserved.tensors.insert((component, index), tensor);
                        reserved.free(&mut self.storage)?;
                        return Ok(None);
                    };
                    tensor.allocations.insert((worker.clone(), rank), buffer);
                }
            }
            reserved.tensors.insert((component, index), tensor);
        }
        Ok(Some(reserved))
    }

    /// Admits queued media requests while request and product storage remain available.
    ///
    /// Media and token requests share the `max_running_requests` running bound.
    /// Admission stops at the first request that cannot be routed or
    /// reserved, leaving it at the queue head. A broken scheduler invariant
    /// also requeues the request, latches engine-fatal, and stops.
    pub(super) fn admit_media(&mut self) {
        // A closing scheduler admits nothing more.
        if self.closing {
            return;
        }
        while self.running_request_count() < self.config.max_running_requests {
            let Some(submission) = self.waiting_media.pop_front() else {
                break;
            };
            // The denoiser's fixed schedule sets every request's step count;
            // submission refused requests while the worker reports none.
            let Some(steps) = self
                .info
                .video_denoiser
                .as_ref()
                .map(|video| video.denoise_steps)
            else {
                self.waiting_media.push_front(submission);
                self.invariant_broken("a queued media request has a video denoiser");
                break;
            };
            let id = submission.request.request_id;
            let request_epoch = self.next_request_epoch;
            let request_key = RequestKey::new(self.engine_id, id, request_epoch);
            let Some(routes) = self.media_routes(&submission.graph) else {
                self.waiting_media.push_front(submission);
                break;
            };
            // Submission validated these bounds against the same loaded
            // components, so a request reaching admission has them.
            let Ok(outputs) = self.media_outputs(&submission.graph) else {
                self.waiting_media.push_front(submission);
                self.invariant_broken("a queued media request has valid output bounds");
                break;
            };
            let route_workers = routes.values().cloned().collect::<HashSet<_>>();
            let allocations = match self.reserve_media(&route_workers, request_key, &outputs) {
                Ok(Some(allocations)) => allocations,
                Ok(None) => {
                    self.waiting_media.push_front(submission);
                    break;
                }
                Err(unknown) => {
                    self.waiting_media.push_front(submission);
                    self.invariant_broken(&unknown.to_string());
                    break;
                }
            };

            // `allocations` holds the request's row on every routed worker;
            // the admission carries the denoiser's row, and a call placed on
            // another worker uses that worker's row.
            let primary_component = self.info.media_components[&MediaCall::Denoising].as_str();
            let request_pool_idx = allocations.request_slot(&routes[primary_component]);
            // Submission validated the prompt, sampling and conditions this
            // admission carries.
            let admission = match NewRequest::new_media(
                request_key,
                request_pool_idx,
                submission.request.prompt_token_ids.clone(),
                submission.request.sampling,
                ipc::VideoAdmission {
                    conditioning: submission.request.conditioning.clone(),
                },
            ) {
                Ok(admission) => admission,
                Err(error) => {
                    let released = allocations.free(&mut self.storage);
                    self.waiting_media.push_front(submission);
                    self.invariant_broken(&format!(
                        "a queued media request forms a valid admission: {error}"
                    ));
                    if let Err(unknown) = released {
                        self.invariant_broken(&unknown.to_string());
                    }
                    break;
                }
            };

            // The admission commits here; nothing below requeues the request.
            for (component, worker) in &routes {
                self.placement
                    .affinity
                    .insert((request_key, component.clone()), worker.clone());
            }
            self.next_request_epoch = self.next_request_epoch.saturating_add(1);
            let root = CallId::new(0, 0);
            let mut state = MediaFlowState {
                request: submission.request,
                graph: submission.graph,
                output: output::EventJournal::new(submission.event_tx),
                allocations,
                reservations: outputs
                    .iter()
                    .map(|output| (output.name, (output.component.clone(), output.index)))
                    .collect(),
                buffer_bindings: HashMap::new(),
                condition_media: HashMap::new(),
                vision_features: None,
                scheduled_condition_units: 0,
                encoded_condition_units: 0,
                condition_video_latents: Vec::new(),
                condition_audio_latents: None,
                conditioning: None,
                latents: Vec::new(),
                decoded_units: BTreeMap::new(),
                encoded_units: BTreeMap::new(),
                encoded_ready: BTreeMap::new(),
                handed_units: 0,
                muxing_inputs: Vec::new(),
                audio: None,
                admission,
                admission_state: WorkerRegistration::Unsubmitted,
                media_reading_scheduled: false,
                vision_encoding_scheduled: false,
                text_encoding_scheduled: false,
                latent_preparation_scheduled: false,
                denoising: Denoising::new(steps),
                scheduled_decode_units: 0,
                scheduled_encode_units: 0,
                audio_decoding_scheduled: false,
                audio_encoding_scheduled: false,
                audio_encoded: false,
                muxing_in_flight: false,
                final_muxing_scheduled: false,
                muxed: false,
                predecessor: root,
                terminal_intent: TerminalIntent::None,
                artifact: None,
            };

            // The interval from receipt to admission is the queue wait. The
            // engine and rank own the preparation that follows, which the
            // request's public phases report from here on.
            let scheduled_at = now();
            self.record_queue_wait(submission.queued_at, scheduled_at);
            if !state.output.enqueue(Event::Scheduled {
                queued_at: submission.queued_at,
                scheduled_at,
            }) {
                state.terminal_intent.finish(FinishReason::Cancelled);
            }
            self.running_media.insert(id, state);
            self.running_order.push(id);
        }
    }

    /// Returns the number of latent positions of an image of `ip`'s size.
    ///
    /// Computes `(height / d) * (width / d)` with floor division, where `d` is
    /// `latent_downsample` (at least 1). Denoising call planning adds
    /// `commit_marker_tokens` to it for the call's query length.
    pub(super) fn num_vae(&self, ip: &ipc::ImageParams) -> u64 {
        let dl = u64::from(self.generation_limits.latent_downsample).max(1);
        (ip.height as u64 / dl) * (ip.width as u64 / dl)
    }

    /// Returns a required runtime feature that the worker lacks.
    pub(super) fn missing_required_feature(
        &self,
        request: &GenerationRequest,
    ) -> Option<crate::GenerationFeatures> {
        // A block-diffusion runtime generates every request it does not read
        // out in canvases.
        let features = if self.config.canvas.is_some() && !request.is_readout() {
            request.canvas_features()
        } else {
            request.required_features()
        };
        self.generation_limits.covers(features).err()
    }

    /// Returns whether the worker tracks image-latent capacity.
    pub(super) fn worker_tracks_image_latent(&self) -> bool {
        self.info.latent_page_units > 0 && self.info.latent_pages > 1
    }

    /// Computes image-latent capacity required by a request.
    ///
    /// The result counts latent positions: with `d = latent_downsample` (at
    /// least 1), it is `ceil(height / d) * ceil(width / d)`, each side
    /// counting at least one position. Unlike `num_vae`, partial positions
    /// round up.
    pub(super) fn worker_image_latent_units_for(&self, st: &RequestState) -> u64 {
        let downsample = (self.generation_limits.latent_downsample as u64).max(1);
        let (height, width) = (st.req.image.height, st.req.image.width);
        ceil_div_u64((height as u64).max(1), downsample)
            * ceil_div_u64((width as u64).max(1), downsample)
    }

    /// Materializes the cache prefixes multi-branch guidance attends to.
    ///
    /// The text-unconditional branch attends to the negative prompt tokens,
    /// and the image-unconditional branch to the prefix the worker states
    /// for its model, or to the request's own context when it states none;
    /// two branches with the same tokens share one prefix. Each prefix gets
    /// its own request row and cache tables sized to its tokens, stored in
    /// `RequestState::guidance_prefix`. Returns `true` when the prefixes are
    /// in place, the request needs none, or the request is not running.
    /// Returns `false`, holding nothing new, when no request row or cache
    /// capacity is free, or (after latching engine-fatal) the prefix cache
    /// is missing; `reserve_generation_resources` then declines the
    /// denoising call.
    pub(super) fn ensure_guidance_prefix(&mut self, id: RequestId) -> bool {
        let Some(state) = self.running.get(&id) else {
            return true;
        };
        if state.guidance_prefix.is_some() {
            return true;
        }
        let negative = &state.req.negative_prompt_token_ids;
        let text = state.req.image.guides_text().then_some(negative);
        let image = self
            .info
            .image_unconditional_token_ids
            .as_ref()
            .filter(|_| state.req.image.guides_image());
        let mut lengths = Vec::new();
        let text_index = text.map(|tokens| {
            lengths.push(tokens.len() as u32);
            lengths.len() - 1
        });
        let image_index = image.map(|tokens| match (text, text_index) {
            (Some(text), Some(index)) if text == tokens => index,
            _ => {
                lengths.push(tokens.len() as u32);
                lengths.len() - 1
            }
        });
        if lengths.is_empty() {
            return true;
        }

        let mut prefixes = Vec::with_capacity(lengths.len());
        for tokens in lengths {
            match self.allocate_guidance_prefix(tokens) {
                Some(prefix) => prefixes.push(prefix),
                None => {
                    for prefix in prefixes {
                        prefix.allocations.free(&mut self.storage);
                    }
                    return false;
                }
            }
        }
        let Some(state) = self.running.get_mut(&id) else {
            for prefix in prefixes {
                prefix.allocations.free(&mut self.storage);
            }
            return false;
        };
        state.guidance_prefix = Some(GuidancePrefixState {
            prefixes,
            text: text_index,
            image: image_index,
            diffusion_finalized: false,
        });
        true
    }

    /// Allocates one guidance prefix of `tokens` tokens: a request row and
    /// cache tables. `None`, holding nothing, when either is unavailable.
    fn allocate_guidance_prefix(&mut self, tokens: u32) -> Option<GuidancePrefix> {
        let request_slot = self.storage.request_pool.allocate().ok()?;
        let Some(cache) = self.storage.cache() else {
            self.storage.request_pool.free(request_slot);
            self.invariant_broken("a running token request has a prefix cache");
            return None;
        };
        let Ok(prefix_cache) = cache.allocate(tokens) else {
            self.storage.request_pool.free(request_slot);
            return None;
        };
        let new_units = prefix_cache
            .tables
            .iter()
            .map(|table| (table.group_id() as u32, table.unit_ids()))
            .collect();
        Some(GuidancePrefix {
            allocations: RequestAllocations {
                request_slot,
                prefix_cache,
                latent: None,
                buffers: HashMap::new(),
            },
            new_units,
            tokens,
        })
    }

    /// The cache tokens a request's image-unconditional guidance prefix
    /// holds, beside those `GenerationRequest::max_cache_tokens` counts.
    pub(super) fn image_unconditional_tokens(&self, request: &GenerationRequest) -> usize {
        self.info
            .image_unconditional_token_ids
            .as_ref()
            .filter(|_| request.generates_images() && request.image.guides_image())
            .map_or(0, Vec::len)
    }

    /// Releases the request's guidance-prefix allocations.
    pub(super) fn free_guidance_prefix(&mut self, id: RequestId) {
        let prefix = self
            .running
            .get_mut(&id)
            .and_then(|state| state.guidance_prefix.take());
        for allocations in prefix
            .into_iter()
            .flat_map(GuidancePrefixState::into_allocations)
        {
            allocations.free(&mut self.storage);
        }
    }

    /// Reaps terminal requests after all submitted descendants resolve.
    pub(super) fn reap_cancellations(&mut self) {
        // A cancelled request closes only after every submitted descendant has
        // resolved. Host cache ownership then remains pinned through the close
        // acknowledgement and ordered worker request retirement.
        let cancelled: Vec<(RequestId, TerminalIntent)> = self
            .running
            .iter()
            .filter(|(id, state)| {
                state.terminal_intent.is_terminal() && !self.inflight.has_pending_calls(**id)
            })
            .map(|(id, state)| (*id, state.terminal_intent.clone()))
            .collect();
        for (id, intent) in cancelled {
            let reason = match intent {
                TerminalIntent::Finish(reason) => reason,
                TerminalIntent::Migrate(cause) => {
                    self.migrate_request(id, cause);
                    continue;
                }
                TerminalIntent::Failure(_) | TerminalIntent::Rejected(_) => FinishReason::Error,
                TerminalIntent::None => continue,
            };
            self.finish(id, reason);
        }
        let media = self
            .running_order
            .iter()
            .filter_map(|id| {
                self.media_state(*id)
                    .filter(|state| {
                        state.terminal_intent.is_terminal()
                            && !self.inflight.has_pending_calls(state.request.request_id)
                    })
                    .map(|state| (state.request.request_id, state.terminal_intent.clone()))
            })
            .collect::<Vec<_>>();
        for (id, intent) in media {
            let event = match intent {
                TerminalIntent::Failure(message) => DiffusionTerminal::Failed(message),
                TerminalIntent::Rejected(message) => DiffusionTerminal::Rejected(message),
                TerminalIntent::Finish(reason) => DiffusionTerminal::Finished(reason),
                // `migrate_requests` marks token requests alone.
                TerminalIntent::Migrate(_) => {
                    DiffusionTerminal::Failed("a media request does not migrate".to_owned())
                }
                TerminalIntent::None => continue,
            };
            self.finish_media(id, event);
        }
    }

    /// Admits requests from the queue head while sequence and storage budgets allow.
    ///
    /// Requests configured for worst-case reservation acquire their full cache capacity here,
    /// keeping that capacity resident for the request lifetime.
    ///
    /// A worst-case request is admitted when the free cache units cover the
    /// units of `max_reserved_cache_tokens` in every group and its
    /// encoder-cache entries fit the budget; any other request needs only
    /// the units of its first prefill chunk beyond its cached prefix, summed
    /// over every cache group. A head that can never fit the total cache
    /// capacity is rejected with `RejectionKind::Invalid`. Admission stops
    /// when the queue is empty, `max_running_requests` is reached, no request row is
    /// free, the head does not fit yet or finds no ready prefill worker, or a
    /// scheduler invariant breaks.
    pub(super) fn admit(&mut self) {
        // A closing scheduler admits nothing more.
        if self.closing {
            return;
        }
        loop {
            if self.running_request_count() >= self.config.max_running_requests
                || self.storage.request_pool.is_empty()
            {
                break;
            }
            let Some(head) = self.waiting.front() else {
                break;
            };
            // Submission rejects token requests when the worker has no prefix
            // cache, so a queued one always finds it.
            let Some(cache) = self.storage.cache() else {
                self.invariant_broken("a queued token request has a prefix cache");
                break;
            };

            if head.reserve_worstcase {
                let tokens = head.max_reserved_cache_tokens;
                let need = cache.coordinator.units_for_tokens(tokens);
                let encoder_entries = head.req.num_encoder_cache_entries();
                let encoder_ok = self
                    .storage
                    .reserved_encoder_entries
                    .saturating_add(encoder_entries)
                    <= self.storage.encoder_cache.budget();
                if need > self.storage.usable_units() {
                    let Some(st) = self.waiting.pop_front() else {
                        break;
                    };
                    let _ = st.output.events.event_tx.send(Event::Rejected {
                        kind: RejectionKind::Invalid,
                        message: "request exceeds total KV capacity".into(),
                    });
                    continue;
                }
                if self.storage.free_units() >= need && encoder_ok {
                    let Some((target, _)) = self.prefill_target(head) else {
                        break;
                    };
                    let Some(st) = self.waiting.pop_front() else {
                        break;
                    };
                    let reservation = match self.reserve_admission(&target.0) {
                        Ok(reservation) => reservation,
                        Err(invariant) => {
                            self.waiting.push_front(st);
                            self.invariant_broken(invariant);
                            break;
                        }
                    };
                    let id = st.req.request_id;
                    self.admit_running(st, target, reservation);
                    // Grow the tables to the whole worst case now, from token
                    // zero, so later admissions cannot take these units. The
                    // result is not checked here; when an image branch opens,
                    // `promote_gen_branch_reservation` finishes the request
                    // with an error if a table covers fewer than `tokens`
                    // tokens.
                    self.ensure_request_capacity(id, 0, tokens);
                    continue;
                }
            } else {
                let n = head.req.prompt_token_ids.len();
                let Some((target, prefix_hit)) = self.prefill_target(head) else {
                    break;
                };

                // Units the first prefill chunk needs beyond the cached
                // prefix. The chunk is bounded by `chunked_prefill_size`
                // and `max_num_batched_tokens` and holds at least one token.
                let chunk_limit = self
                    .config
                    .chunked_prefill_size
                    .min(self.config.max_num_batched_tokens)
                    .max(1);
                let cached_tokens = prefix_hit.cached_tokens.min(n);
                // The host KV cache may hold the reusable pages that follow
                // the device match; the head waits for its pins.
                let mut host_tokens = 0;
                if let Some(kv_offload) = self.kv_offload.as_mut()
                    && let Some(step) = kv_offload.step(&target.0)
                    && head.req.cache.read
                    && !head.has_context_images()
                {
                    // Reusable pages leave one prompt token to compute, as
                    // in the device cache.
                    let limit = n.saturating_sub(1) / step * step;
                    if limit > cached_tokens {
                        let keys = kv_offload.page_keys(
                            &target.0,
                            &head.req.prompt_token_ids,
                            head.req.cache.isolation_key,
                            limit,
                        );
                        match kv_offload.admission(
                            head.req.request_id,
                            &target.0,
                            &keys,
                            cached_tokens,
                            limit,
                        ) {
                            Some(boundary) => host_tokens = boundary - cached_tokens,
                            None => break,
                        }
                    }
                }
                let prefix_tokens = cached_tokens + host_tokens;
                let first_chunk = n.saturating_sub(prefix_tokens).min(chunk_limit).max(1);
                let first_chunk_units = cache
                    .coordinator
                    .units_after_prefix(cached_tokens, prefix_tokens + first_chunk);

                // The prompt alone must fit the pool: every full-attention page
                // and each sliding-window group's pages of one chunk.
                if cache.coordinator.min_units_for_prompt(n, chunk_limit)
                    > self.storage.usable_units()
                {
                    let Some(st) = self.waiting.pop_front() else {
                        break;
                    };
                    let _ = st.output.events.event_tx.send(Event::Rejected {
                        kind: RejectionKind::Invalid,
                        message: "request exceeds total KV capacity".into(),
                    });
                    continue;
                }

                // Hit pages that are currently unreferenced count as free
                // until acquisition takes them, so they are subtracted from
                // the free units.
                let capacity_available = self
                    .storage
                    .free_units()
                    .saturating_sub(prefix_hit.cached_free_units)
                    >= first_chunk_units;
                if capacity_available {
                    let Some(st) = self.waiting.pop_front() else {
                        break;
                    };
                    let reservation = match self.reserve_admission(&target.0) {
                        Ok(reservation) => reservation,
                        Err(invariant) => {
                            self.waiting.push_front(st);
                            self.invariant_broken(invariant);
                            break;
                        }
                    };
                    let id = st.req.request_id;
                    let worker = target.0.clone();
                    self.admit_running(st, target, reservation);
                    self.admit_offloaded_prefix(id, &worker);
                    continue;
                }
            }

            // The head does not fit yet. Resident requests keep their physical
            // cache units and are never preempted or relocated to make room, and
            // later requests do not bypass the head, so admission waits for
            // capacity to free.
            break;
        }
    }

    /// Probes the prefix cache on the configured prefill owner.
    ///
    /// Picks the first ready worker that serves prefill and returns it with
    /// the component name it binds, plus the prefix hit on that worker's
    /// endpoint (prefix-cache residency is keyed by endpoint). The hit is an
    /// estimate; `acquire_cached_prefix` takes the actual prefix. Returns
    /// `None` when the runtime has no prefix cache or no prefill worker is ready.
    fn prefill_target(
        &self,
        state: &RequestState,
    ) -> Option<((crate::WorkerGroupId, String), crate::kv_cache::PrefixHit)> {
        let cache = self.storage.cache()?;
        self.placement
            .worker_candidates(
                self.executor.as_ref(),
                &self.info,
                CallKind::Forward(ForwardMode::Prefill),
            )
            .filter(|(worker, _, _)| self.executor.is_ready(worker))
            .map(|(worker, component, info)| {
                let hit = cache.coordinator.probe_prefix(
                    &cache.block_pool,
                    &state.req.prompt_token_ids,
                    state.req.cache.read,
                    state.has_context_images(),
                    state.req.cache.isolation_key,
                    &info.endpoint,
                );
                ((worker.clone(), component.to_owned()), hit)
            })
            .next()
    }

    /// Accumulates one request's interval from queue entry to admission.
    ///
    /// Timestamps are in seconds; the statistics are in microseconds, and a
    /// negative interval counts as zero.
    fn record_queue_wait(&self, queued_at: f64, scheduled_at: f64) {
        let queue_wait_us = ((scheduled_at - queued_at).max(0.0) * 1_000_000.0) as u64;
        self.stats
            .timing
            .queue_wait_count
            .fetch_add(1, Ordering::Relaxed);
        self.stats
            .timing
            .queue_wait_us_total
            .fetch_add(queue_wait_us, Ordering::Relaxed);
        self.stats
            .timing
            .queue_wait_us_max
            .fetch_max(queue_wait_us, Ordering::Relaxed);
    }

    /// Reserves what admission to `target` needs before the request commits.
    ///
    /// Admission checked request-row capacity and chose `target` among loaded
    /// workers, and the text prefix cache exists for every queued token request, so
    /// an error names the scheduler invariant that no longer holds. Nothing
    /// stays reserved on error.
    fn reserve_admission(
        &mut self,
        target: &crate::WorkerGroupId,
    ) -> Result<AdmissionReservation, &'static str> {
        let source = self
            .executor
            .info()
            .workers
            .iter()
            .find(|(worker, _)| worker == target)
            .map(|(_, worker)| worker.endpoint.clone())
            .ok_or("the selected prefill worker remains loaded")?;
        let cache = self
            .storage
            .cache
            .as_ref()
            .ok_or("a queued token request has a prefix cache")?;
        let request_slot = self
            .storage
            .request_pool
            .allocate()
            .map_err(|_| "admission checked request-row capacity")?;
        // Admission owns the request row and an initially empty table for every
        // cache group before the request enters the runnable set.
        let Ok(prefix_cache) = cache.allocate(0) else {
            self.storage.request_pool.free(request_slot);
            return Err("an empty cache allocation always fits");
        };
        Ok(AdmissionReservation {
            source,
            request_slot,
            prefix_cache,
        })
    }

    /// Moves one validated request into the runnable set with its reservation.
    ///
    /// Records `target`'s worker as the owner of its component for the
    /// request, emits the public `Scheduled` event (a closed receiver marks
    /// the request cancelled), adds its encoder-cache entries to the reserved
    /// count, and acquires its cached prefix, which sets the computed-prompt
    /// cursors to the reused tokens. A failed acquisition latches
    /// engine-fatal.
    pub(super) fn admit_running(
        &mut self,
        mut st: RequestState,
        target: (crate::WorkerGroupId, String),
        reservation: AdmissionReservation,
    ) {
        let id = st.req.request_id;
        let request_key = RequestKey::new(self.engine_id, id, st.request_epoch);
        let AdmissionReservation {
            source,
            request_slot,
            prefix_cache,
        } = reservation;
        self.placement
            .affinity
            .insert((request_key, target.1), target.0);
        st.allocations = Some(RequestAllocations {
            request_slot,
            prefix_cache,
            latent: None,
            buffers: HashMap::new(),
        });

        // Queue latency is finalized at the same timestamp exposed through the
        // public scheduling event.
        let q = st.queued_at;
        let scheduled_at = now();
        self.record_queue_wait(q, scheduled_at);

        let encoder_entries = st.req.num_encoder_cache_entries();
        if !st.output.events.enqueue(Event::Scheduled {
            queued_at: q,
            scheduled_at,
        }) {
            st.terminal_intent = TerminalIntent::Finish(FinishReason::Cancelled);
        }

        // Running order, resource reservations, and runtime ownership advance
        // together so the next scheduling pass observes one coherent admission.
        self.running.insert(id, st);
        self.running_order.push(id);
        self.storage.reserved_encoder_entries = self
            .storage
            .reserved_encoder_entries
            .saturating_add(encoder_entries);

        // Prefix-cache acquisition pins every reused block to this request's
        // newly installed block tables.
        let acquired = match (self.running.get_mut(&id), self.storage.cache.as_ref()) {
            (Some(st), Some(prefix_cache)) => acquire_cached_prefix(
                &prefix_cache.coordinator,
                st,
                &prefix_cache.block_pool,
                &self.stats,
                &source,
            ),
            _ => Err("an admitted request runs with a prefix cache"),
        };
        if let Err(invariant) = acquired {
            self.invariant_broken(invariant);
        }
    }
}

impl Scheduler {
    /// Apply the host KV cache to an admitted request on `worker`: record
    /// the keys its published pages are offloaded under, and extend its
    /// cached prefix to the host KV cache's hit, whose units are allocated
    /// now and loaded by its first call. A hit that no longer follows the
    /// device prefix the request acquired is released.
    fn admit_offloaded_prefix(&mut self, id: RequestId, worker: &crate::WorkerGroupId) {
        let Some(kv_offload) = self.kv_offload.as_mut() else {
            return;
        };
        let Some(state) = self.running.get_mut(&id) else {
            return;
        };
        if !state.req.cache.read || state.has_context_images() {
            return;
        }
        state.page_keys = kv_offload.page_keys(
            worker,
            &state.req.prompt_token_ids,
            state.req.cache.isolation_key,
            state.req.prompt_token_ids.len(),
        );
        let computed = state.num_computed_prompt_tokens as usize;
        state.offload_boundary = computed;
        let Some((device, boundary)) = kv_offload.admit(id) else {
            return;
        };
        if computed != device {
            kv_offload.abandon(id);
            return;
        }
        // The first call continues from the boundary: a sliding window's
        // pages before its window there are not read again, and the window's
        // pages the device lacks are allocated for the loads.
        let released = match (self.storage.cache.as_ref(), state.prefix_cache_mut()) {
            (Some(cache), Some(allocation)) => {
                cache.coordinator.release_window(allocation, boundary)
            }
            _ => false,
        };
        if !released || !self.ensure_request_capacity(id, computed, boundary) {
            if let Some(kv_offload) = self.kv_offload.as_mut() {
                kv_offload.abandon(id);
            }
            return;
        }
        let Some(state) = self.running.get_mut(&id) else {
            return;
        };
        // The hit's pages are computed once loaded: the first call starts
        // after them, and they are published to the device cache when that
        // call's result is accepted.
        state.num_computed_prompt_tokens = boundary as u32;
        state.logical_position = state.num_computed_prompt_tokens;
        state.cache_visible_len = state.num_computed_prompt_tokens;
        state.cache_computed_len = state.num_computed_prompt_tokens;
        state.offload_boundary = boundary;
        self.stats
            .prefix
            .hit_tokens
            .fetch_add((boundary - computed) as u64, Ordering::Relaxed);
    }
}

/// Resources a token request holds from the moment admission commits it.
pub(super) struct AdmissionReservation {
    /// The prefill worker's endpoint, which keys its prefix-cache residency.
    source: ipc::WorkerEndpoint,
    request_slot: RequestSlot,
    prefix_cache: CacheAllocation,
}

/// Acquires the longest prefix every cache group can reuse and records cache
/// accounting on the request.
///
/// Fails, acquiring nothing, when the request holds no block tables or its
/// tables do not match the pool's cache groups.
fn acquire_cached_prefix(
    coordinator: &PrefixCacheCoordinator,
    state: &mut RequestState,
    pool: &BlockPool,
    stats: &SchedulerStats,
    source: &ipc::WorkerEndpoint,
) -> Result<(), &'static str> {
    let prompt = state.req.prompt_token_ids.clone();
    let has_context_images = state.has_context_images();
    let cache_read = state.req.cache.read;
    let isolation_key = state.req.cache.isolation_key;
    let allocation = state
        .prefix_cache_mut()
        .ok_or("an admitted request holds its block tables")?;
    let hit = coordinator
        .acquire_prefix(
            pool,
            allocation,
            &prompt,
            cache_read,
            has_context_images,
            isolation_key,
            source,
        )
        .ok_or("an admitted request's block tables cover every cache group")?;

    // Queries count the tokens eligible for reuse under the same rule as the
    // prefix selection: a multiple of the largest block size that leaves at
    // least one prompt token to compute, so prefill produces logits.
    let step = coordinator
        .groups()
        .iter()
        .map(|group| group.block_size)
        .max()
        .unwrap_or(1);
    let query_tokens = prompt.len().saturating_sub(1) / step * step;
    stats
        .prefix
        .query_tokens
        .fetch_add(query_tokens as u64, Ordering::Relaxed);
    stats
        .prefix
        .hit_tokens
        .fetch_add(hit.cached_tokens as u64, Ordering::Relaxed);

    // The reused tokens are already in the prefix cache, so every prompt and cache cursor
    // starts past them, and their pages are already published.
    state.prefix_published = coordinator
        .groups()
        .iter()
        .map(|group| hit.cached_tokens / group.block_size)
        .collect();
    state.prefix_page_hashes = hit.page_hashes;
    state.num_computed_prompt_tokens = hit.cached_tokens as u32;
    state.logical_position = state.num_computed_prompt_tokens;
    state.cache_visible_len = state.num_computed_prompt_tokens;
    state.cache_computed_len = state.num_computed_prompt_tokens;
    Ok(())
}
