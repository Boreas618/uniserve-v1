//! Request arrival, the waiting queue and admission.
//!
//! [`Scheduler::add_request`] validates a request against what the loaded
//! workers serve and against the waiting-queue bound. It either queues the
//! request or rejects it synchronously with `Event::Rejected` on the
//! request's event channel. A token request and a video request wait in one
//! queue: the requests that resume work another engine began first, then
//! arrival order.
//!
//! Admission (`admit`) runs once per step on the policy's output. It refuses
//! the requests the policy rejects and tries the ones it admits in the
//! queue's order, while `max_running_requests` and storage allow. The head
//! that does not fit yet blocks the requests behind it; when running
//! requests themselves cannot grow, the policy names a victim
//! (`Scheduler::preempt_shortage`).
//!
//! A token request reserves a request slot and one cache table per cache
//! group; every group draws units from one pool, so admission compares the
//! sum of the groups' unit demand with the free units. Its prefill worker is
//! recorded in `Dispatch::affinity`. A video request reserves a request slot
//! and every declared output buffer on each worker of its route.

use super::*;
use crate::kv_cache::{KvOffloadManager, PrefixScope};
use crate::request::Request;
use ipc::ForwardMode;
use scheduling::guest::Plan;
use std::ops::ControlFlow;

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
    /// The serving component has no dispatch candidate.
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
    /// Takes one request the daemon submitted: validates it against what the
    /// loaded workers serve and queues it, or rejects it synchronously.
    ///
    /// Rejects with `RejectionKind::Invalid` when the request's own
    /// validation (`validate_submission`, `validate_video`) refuses it, and
    /// with `RejectionKind::Overloaded` when the waiting bound is reached. A
    /// rejection is sent on `event_tx` and its send result is ignored.
    pub fn add_request(&mut self, request: crate::Request, event_tx: EventTx) {
        let reject = |event_tx: &EventTx, kind: RejectionKind, message: String| {
            let _ = event_tx.send(Event::Rejected { kind, message });
        };
        let full = "scheduler waiting queue is full".to_owned();
        // The request's reader wakes this scheduler when it frees room.
        event_tx.served_by(self.worker.waker());
        let graph = match request {
            crate::Request::Generation(mut req) => {
                let max_cache_tokens = match self.validate_submission(&req) {
                    Ok(tokens) => tokens,
                    Err(message) => return reject(&event_tx, RejectionKind::Invalid, message),
                };
                if self.queue_full() {
                    return reject(&event_tx, RejectionKind::Overloaded, full);
                }
                if req.cache.resume.is_none() {
                    req.cache.resume = Some(proto::request::CacheResume {
                        epoch: self.epoch,
                        request_id: req.request_id.0,
                    });
                }
                Graph::Tokens(Box::new(self.token_state(*req, event_tx, max_cache_tokens)))
            }
            crate::Request::Diffusion(request) => {
                let (graph, steps) = match self.validate_video(&request) {
                    Ok(validated) => validated,
                    Err(message) => return reject(&event_tx, RejectionKind::Invalid, message),
                };
                if self.queue_full() {
                    return reject(&event_tx, RejectionKind::Overloaded, full);
                }
                let request_key =
                    RequestKey::new(self.engine_id, request.request_id, self.next_request_epoch);
                self.next_request_epoch = self.next_request_epoch.saturating_add(1);
                Graph::Video(Box::new(VideoState::new(
                    request,
                    request_key,
                    graph,
                    event_tx,
                    steps,
                )))
            }
        };
        let request = Request::waiting(graph);
        let id = request.id();
        self.queue(request);

        // The policy sees the request as it arrived: nothing has happened to it
        // yet, so every feature but the submitted ones is empty.
        if let Some(view) = views::view_of(self, id, self.recorder.recorded()) {
            self.recorder.record(views::arrived(view));
        }
    }

    /// Whether the waiting bound is reached.
    ///
    /// Backpressure: under overload a new submission is rejected rather than
    /// letting waiting requests grow without bound and exhaust process
    /// storage. The bound covers the waiting queue and the retired requests
    /// whose events are still undelivered.
    fn queue_full(&self) -> bool {
        let waiting = self.pending_request_count() + self.output.retained_len();
        self.config
            .max_queued_requests
            .is_some_and(|bound| waiting >= bound.get())
    }

    /// Queues `request` for admission: a request that resumes work another
    /// engine began behind the others that resume and ahead of every request
    /// that starts, and any other request last.
    ///
    /// A resumed request arrived first, and what it resumes from stays needed
    /// until it is admitted: a stored trajectory until its latent preparation
    /// restores it, so taking the resumed requests first keeps the
    /// trajectories an epoch still has to restore to those of its latest
    /// preemption, at most its request slots, which the host latent pools
    /// are sized by.
    fn queue(&mut self, request: Request) {
        let id = request.id();
        let position = if request.resumes() {
            self.waiting
                .iter()
                .position(|queued| !self.requests.get(queued).is_some_and(Request::resumes))
                .unwrap_or(self.waiting.len())
        } else {
            self.waiting.len()
        };
        self.requests.insert(request);
        self.waiting.insert(position, id);
    }

    /// Checks a token-generation request against the loaded workers and
    /// returns its worst-case cache token count.
    ///
    /// Refuses, with the client's message, a request when the runtime has no
    /// prefix cache, the request fails validation, a unit the scheduler never
    /// splits (a generation canvas, an input image's vision block) exceeds the step token budget, the worker lacks a required
    /// generation feature, or `validate_resources` or `max_cache_tokens`
    /// fails against `generation_limits`.
    pub(super) fn validate_submission(&self, req: &GenerationRequest) -> Result<usize, String> {
        if self.storage.cache.is_none() {
            return Err("generation request requires worker KV resources".into());
        }
        if let Err(error) = req.validate() {
            return Err(format!("invalid generation request: {error:?}"));
        }
        // A request the deployment generates in canvases names the seed its
        // draws follow.
        if self.config.canvas.is_some() {
            if req.sampling.seed.is_none() {
                return Err(format!(
                    "invalid generation request: {:?}",
                    GenerationRequestError::UnseededCanvas
                ));
            }
            // A canvas's committed blocks are tokens, but the denoising state
            // of the block in progress is not.
            if req.is_migrated() {
                return Err(
                    "a block-diffusion request does not resume from generated tokens".into(),
                );
            }
        }
        // A context prefill writes the prompt and every input image's vision
        // block in one call, each block whole as one attention block of that
        // call, so the whole context the prefill reaches must fit one
        // scheduling step's token budget. The bound is the context, not the
        // block: a block that fits the budget alone still does not fit beside
        // the prompt it is embedded in.
        let blocks: usize = req
            .multimodal_inputs
            .images
            .iter()
            .flat_map(|image| &image.encoders)
            .filter(|input| input.encoder == crate::ImageEncoder::Vision)
            .filter_map(|input| input.num_cache_tokens)
            .map(|tokens| tokens as usize)
            .sum();
        let context = req.prompt_token_ids.len().saturating_add(blocks);
        if blocks > 0 && context > self.config.max_num_batched_tokens {
            return Err(format!(
                "a {context}-token context, whose {blocks} tokens of input image vision block \
                 are written whole by one context prefill, exceeds the {}-token step budget",
                self.config.max_num_batched_tokens
            ));
        }
        // A vision block's features are one product of the rows it writes, which the rank's
        // feature storage holds whole.
        if let Some(row_bytes) = self
            .info()
            .image_encoders
            .get(crate::encodings::VISION_FEATURES)
            .map(|output| output.row_bytes)
        {
            let largest = req
                .multimodal_inputs
                .images
                .iter()
                .flat_map(|image| &image.encoders)
                .filter(|input| input.encoder == crate::ImageEncoder::Vision)
                .filter_map(|input| input.num_cache_tokens)
                .max()
                .unwrap_or(0);
            let bytes = u64::from(largest).saturating_mul(row_bytes);
            if bytes > self.generation_limits.max_vision_feature_bytes {
                return Err(format!(
                    "an input image's vision block of {largest} tokens takes {bytes} bytes of \
                     features, beyond the {} bytes one encoding holds",
                    self.generation_limits.max_vision_feature_bytes
                ));
            }
        }
        if let Some(feature) = self.missing_required_feature(req) {
            return Err(format!(
                "generation request requires worker feature `{feature}`, but the worker does not support it"
            ));
        }
        if let Some(refusal) = self.uncaptured_latent(req) {
            return Err(refusal);
        }
        // A prefill group hands a request's cache to a decode group once,
        // after its prompt: such a deployment generates text.
        if self.splits_prefill() && self.config.canvas.is_some() {
            return Err("a deployment of prefill and decode groups generates text only".into());
        }
        // Resource validation includes the cache bound, so a request that passes
        // it has a worst-case cache token count.
        req.validate_resources(&self.generation_limits)
            .and_then(|()| req.max_cache_tokens(&self.generation_limits))
            .map_err(|error| format!("invalid generation resource requirements: {error}"))
    }

    /// The state of a validated request, at the next request epoch, which
    /// publishes through `event_tx`. `max_cache_tokens` is its worst-case
    /// cache token count (`validate_submission`).
    pub(super) fn token_state(
        &mut self,
        mut req: GenerationRequest,
        event_tx: EventTx,
        max_cache_tokens: usize,
    ) -> TokenState {
        let prompt_tokens = req.prompt_token_ids.len();
        let generated = req.generated_token_ids.len();
        // A migrated request's generated tokens continue its prompt: the
        // engine computes their cache as it computes the prompt's and samples
        // after the last of them, at the position the request had reached.
        // They count as generated from the start, so the token limit,
        // `min_tokens`, the penalties and the bad words continue where they
        // were; its usage charges only its own prompt as prompt
        // (`TokenState::charged_prompt_tokens`).
        req.prompt_token_ids
            .extend_from_slice(&req.generated_token_ids);
        let generated_token_ids = req.generated_token_ids.clone();
        // Multimodal requests reserve their configured bounded cache envelope at
        // admission so excess concurrency queues instead of exhausting the prefix cache.
        let reserve_worstcase = !req.multimodal_inputs.images.is_empty();
        let finish_token_ids = finish_token_ids(&req, &self.special_tokens.eos);
        let st = TokenState {
            finish_token_ids,
            allocations: None,
            state_resident: false,
            sampling_resident: false,
            request_epoch: self.next_request_epoch,
            last_state_call_id: CallId::default(),
            latest_token: None,
            speculative_chain_invalidated: false,
            draft: Default::default(),
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
            logical_position: 0,
            cache_visible_len: 0,
            cache_computed_len: 0,
            // A request draws its first token at position zero of its stream.
            sampling_position: 0,
            next_token: 0,
            num_generated_tokens: generated,
            hand_off: generation::HandOff::None,
            worker_registered: false,
            num_cache_units_sent: 0,
            reserve_worstcase,
            // `admit` reserves the units of this worst-case extent only for
            // requests with `reserve_worstcase` set.
            max_reserved_cache_tokens: max_cache_tokens,
            prefix_page_hashes: Vec::new(),
            prefix_published: Vec::new(),
            host_keyed: false,
            offload_boundary: 0,
            generated_token_ids,
            replayable: true,
            encoder_cache_pins: Vec::new(),
            host_encoding: None,
            transient_encoder_products: Vec::new(),
            canvas_block: 0,
            canvas_step: 0,
            canvas_commit: Vec::new(),
            output: RequestOutput::new(event_tx),
            failure: None,
            prompt_tokens,
            req,
        };
        self.next_request_epoch = self.next_request_epoch.saturating_add(1);
        st
    }

    /// Resolves the products a video request reserves at admission.
    ///
    /// Each product of the request's graph is found by name among the
    /// outputs of the component serving its call, as bound on that call's
    /// first dispatch candidate. A product with a request extent takes it
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
    /// canvas is zero or exceeds the declared maximum. `add_request`
    /// rejects such a request, and `admit_video` calls this again to size the
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
                .dispatch
                .component_candidates(&self.worker, CallKind::Media(product.call), name)
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
    /// with the fewest distinct requests placed on it in `Dispatch::affinity`
    /// wins, then the one with the most free request rows.
    ///
    /// Returns a map from component name to worker, or `None` when some
    /// component has no ready candidate that is already on the route or has
    /// a free request row. When the worker reports admissible video
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
                    self.worker.info().video_codecs.get(worker)
                })
                .flatten();
            let candidates = self
                .dispatch
                .component_candidates(&self.worker, CallKind::Media(call), &component)
                .filter(|(worker, _, _)| {
                    admissible.is_none_or(|encoders| encoders.contains(*worker))
                        && self.worker.is_ready(worker)
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
                        .dispatch
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

    /// Checks a video request against the loaded workers and returns its
    /// call graph with the denoiser's step count.
    ///
    /// Refuses, with the client's message, a request that fails validation;
    /// when the worker reports no muxing stage or no video denoiser; when
    /// the denoiser does not serve the request's task; when its call graph
    /// cannot be built; or when `media_outputs` cannot bound its products. A
    /// served task's calls all have stages: the worker checked them at
    /// startup. The two request slots a video decoder needs are the rank's
    /// to require: it refuses to start with fewer.
    fn validate_video(&self, request: &DiffusionRequest) -> Result<(VideoGraph, u32), String> {
        request.validate().map_err(|error| error.to_string())?;
        // A worker reports the stage serving each media call it implements,
        // so a deployment that assembles an artifact is the one that can
        // serve a video request; reporting some call is not enough.
        if !self
            .info
            .media_components
            .contains_key(&ipc::MediaCall::Muxing)
        {
            return Err("worker does not provide the video media components".to_owned());
        }
        let denoiser = self
            .info
            .video_denoiser
            .as_ref()
            .ok_or_else(|| "worker reports no video denoiser".to_owned())?;
        if !denoiser.takes_conditions && !request.conditions.is_empty() {
            return Err("the deployment's denoiser takes no conditions".to_owned());
        }
        let graph = VideoGraph::new(request).map_err(|error| error.to_string())?;
        self.media_outputs(&graph)
            .map_err(|error| format!("loaded media components cannot hold the request: {error}"))?;
        // The denoiser's fixed schedule sets every request's step count.
        Ok((graph, denoiser.denoise_steps))
    }

    /// Reserves a media request's row and output buffers on every routed worker.
    ///
    /// Every declared output is reserved in full in the buffer pool of each
    /// rank that backs it on each routed worker, not only on the worker whose
    /// component produces it; spans are sized from the shape bound's maximum
    /// element count and 256-byte aligned.
    ///
    /// Returns `Ok(None)`, with nothing reserved, when a request row or buffer
    /// span cannot be allocated on some rank now; `admit_video` then leaves
    /// the request at the head of the queue. Returns `Err` when a routed
    /// worker has no media storage, after releasing what was already reserved
    /// on known workers.
    fn reserve_media(
        &mut self,
        route_workers: &HashSet<crate::WorkerGroupId>,
        request_key: RequestKey,
        outputs: &[MediaOutput],
    ) -> Result<Option<MediaAllocations>, UnknownMediaGroup> {
        let mut reserved = MediaAllocations {
            request_slots: HashMap::new(),
            tensors: HashMap::new(),
        };
        for worker in route_workers {
            let Some(storage) = self.storage.media_storage.get_mut(worker) else {
                reserved.free(&mut self.storage)?;
                return Err(UnknownMediaGroup(worker.clone()));
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
                    return Err(UnknownMediaGroup(worker.clone()));
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

    /// Admits the waiting video request `id`, the head of the queue, when its
    /// route and its storage are available.
    ///
    /// Answers `Break` when the head waits: it cannot be routed or reserved
    /// now, or a scheduler invariant broke (which also latches engine-fatal).
    /// Nothing stays reserved for a head that waits.
    fn admit_video(&mut self, id: RequestId) -> ControlFlow<()> {
        let Some(state) = self.requests.get(&id).and_then(Request::video) else {
            self.invariant_broken("a waiting video request is in the request table");
            return ControlFlow::Break(());
        };
        let request_key = state.request_key;
        let Some(routes) = self.media_routes(&state.graph) else {
            return ControlFlow::Break(());
        };
        // Submission validated these bounds against the same loaded
        // stages, so a request reaching admission has them.
        let Ok(outputs) = self.media_outputs(&state.graph) else {
            self.invariant_broken("a queued media request has valid output bounds");
            return ControlFlow::Break(());
        };
        let prompt_token_ids = state.request.prompt_token_ids.clone();
        let sampling = state.request.sampling;
        let conditioning = state.request.conditioning.clone();
        let route_workers = routes.values().cloned().collect::<HashSet<_>>();
        let allocations = match self.reserve_media(&route_workers, request_key, &outputs) {
            Ok(Some(allocations)) => allocations,
            Ok(None) => return ControlFlow::Break(()),
            Err(unknown) => {
                self.invariant_broken(&unknown.to_string());
                return ControlFlow::Break(());
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
            prompt_token_ids,
            sampling,
            ipc::VideoAdmission { conditioning },
        ) {
            Ok(admission) => admission,
            Err(error) => {
                let released = allocations.free(&mut self.storage);
                self.invariant_broken(&format!(
                    "a queued media request forms a valid admission: {error}"
                ));
                if let Err(unknown) = released {
                    self.invariant_broken(&unknown.to_string());
                }
                return ControlFlow::Break(());
            }
        };

        // The admission commits here; nothing below leaves the request waiting.
        for (component, worker) in &routes {
            self.dispatch
                .affinity
                .insert((request_key, component.clone()), worker.clone());
        }
        self.waiting.retain(|queued| *queued != id);
        let Some(request) = self.requests.get_mut(&id) else {
            return ControlFlow::Break(());
        };
        request.status = RequestStatus::Running;
        let queued_us = request.queued_us;
        // The interval from receipt to admission is the queue wait. The
        // engine and rank own the preparation that follows, which the
        // request's public phases report from here on.
        let scheduled_us = crate::clock::now_monotonic_us();
        if let Some(state) = request.video_mut() {
            state.allocations = allocations;
            state.reservations = outputs
                .iter()
                .map(|output| (output.name, (output.component.clone(), output.index)))
                .collect();
            state.admission = Some(admission);
            if !state.output.enqueue(Event::Scheduled {
                queued_at: crate::clock::unix_secs_at(queued_us),
                scheduled_at: crate::clock::unix_secs_at(scheduled_us),
            }) {
                request
                    .status
                    .finish(Terminal::finish(FinishReason::Cancelled));
            }
        }
        self.record_queue_wait(queued_us, scheduled_us);
        self.running.push(id);
        self.tell_admitted(id);
        // A resumed request's stored trajectory is pinned while its
        // conditions are prepared, and every request's encoder outputs
        // are looked up before its encoders are offered.
        let restoring = self.start_restore(id);
        if let Some(state) = self.media_state_mut(id) {
            state.host_trajectory.restoring = restoring;
        }
        self.look_up_encoder_outputs(id);
        ControlFlow::Continue(())
    }

    /// Tells the policy that `id` was admitted, and starts the starvation
    /// bound's clock for it.
    ///
    /// The policy's queue is its own, so an admission it is not told of is a
    /// name it offers in `admit` again, and one the mechanism then refuses
    /// as not waiting. A running request that never advances is a policy
    /// failure, whatever its graph.
    fn tell_admitted(&mut self, id: RequestId) {
        let admitted_at =
            scheduling::guest::Nanos(crate::clock::now_monotonic_us().saturating_mul(1_000));
        let request = scheduling::guest::RequestId(id.0);
        self.recorder.admitted(request, admitted_at);
        self.watchdog.admitted(request, admitted_at);
    }

    /// Whether some worker group prefills only (`WorkerGroupRole::Prefill`),
    /// its requests decoding on decode groups.
    fn splits_prefill(&self) -> bool {
        !self.dispatch.prefill_groups.is_empty()
    }

    /// Why the latent encoder cannot encode one of `req`'s input images, if
    /// so: under CUDA graphs it encodes only the image sizes it captured
    /// (`MainProcessInfo::captured_latents`), and a latent block states the size
    /// its image resizes to.
    fn uncaptured_latent(&self, req: &GenerationRequest) -> Option<String> {
        let captured = self.info.captured_latents.as_ref()?;
        let block = req
            .multimodal_inputs
            .images
            .iter()
            .flat_map(|image| &image.encoders)
            .filter(|input| input.encoder == crate::ImageEncoder::Latent)
            .find(|input| match (input.height, input.width) {
                (Some(height), Some(width)) => !captured.contains(&[height, width]),
                _ => true,
            })?;
        let sizes: Vec<String> = captured
            .iter()
            .map(|[height, width]| format!("{width}x{height}"))
            .collect();
        let size = match (block.height, block.width) {
            (Some(height), Some(width)) => format!("{width}x{height}"),
            _ => "an unstated size".to_owned(),
        };
        Some(format!(
            "an input image resizes to {size} for latent encoding, which the workers encode \
             under CUDA graphs only at the sizes they captured: {}; add the size to \
             cuda_graph_config.flow_shapes to encode it",
            if sizes.is_empty() {
                "none".to_owned()
            } else {
                sizes.join(", ")
            }
        ))
    }

    /// Returns a required runtime feature that the worker lacks.
    pub(super) fn missing_required_feature(
        &self,
        request: &GenerationRequest,
    ) -> Option<crate::GenerationFeatures> {
        // A block-diffusion runtime generates every request in canvases.
        let features = if self.config.canvas.is_some() {
            request.canvas_features()
        } else {
            request.required_features()
        };
        self.generation_limits.covers(features).err()
    }

    /// Retires the requests whose end is decided once every call they
    /// submitted has resolved.
    ///
    /// A finished request closes only after every submitted descendant has
    /// resolved. Host cache ownership then remains pinned through the close
    /// acknowledgement and ordered worker request retirement.
    pub(super) fn reap_cancellations(&mut self) {
        let finished: Vec<(RequestId, Terminal)> = self
            .running
            .iter()
            .filter(|id| !self.inflight.has_pending_calls(**id))
            .filter_map(|id| Some((*id, self.requests.status(id)?.terminal()?.clone())))
            .collect();
        for (id, terminal) in finished {
            if self.requests.video(&id).is_some() {
                match terminal {
                    // A preempted request is at a step boundary: its trajectory
                    // is stored before it leaves.
                    Terminal::Migrated { cause, .. } => {
                        self.retire_preempted(id, cause);
                    }
                    terminal => self.finish_media(id, terminal),
                }
                continue;
            }
            let reason = match terminal {
                Terminal::Finished { reason, .. } => reason,
                Terminal::Migrated { cause, .. } => {
                    self.migrate_request(id, cause);
                    continue;
                }
                Terminal::Failed(_) | Terminal::Rejected(_) => FinishReason::Error,
                Terminal::Completed(_) => FinishReason::Completed,
            };
            self.finish(id, reason);
        }
    }

    /// Executes the policy's admission decisions: the requests it rejects
    /// are refused, and the ones it admits are tried in the queue's order,
    /// the mechanism stopping at the first the hard limits refuse so that a
    /// later request never bypasses one ahead of it.
    ///
    /// The policy names a *set*, and a waiting request it does not name
    /// waits for a later step. Which of the named requests the mechanism
    /// spends its resources on first is the queue's order, which carries a
    /// rule a policy cannot see (`Self::queue`): the named requests move
    /// ahead of the rest by a stable partition, never a re-sort.
    ///
    /// Admission stops when `max_running_requests` is reached or the head
    /// waits: a token request for a slot, for the units of its first prefill
    /// chunk beyond its cached prefix, or for a ready prefill worker; a
    /// video request for its route and its output buffers.
    pub(super) fn admit(&mut self, plan: &Plan) {
        // A closing scheduler admits nothing more, nor does one that sleeps.
        if self.draining || self.admission_sleeps() {
            return;
        }
        for id in &plan.reject {
            self.reject_waiting(ipc::RequestId(id.0));
        }
        let named: HashSet<ipc::RequestId> =
            plan.admit.iter().map(|id| ipc::RequestId(id.0)).collect();
        self.waiting
            .make_contiguous()
            .sort_by_key(|id| !named.contains(id));
        let count = self.waiting.iter().filter(|id| named.contains(id)).count();
        for _ in 0..count {
            if self.running_request_count() >= self.config.max_running_requests {
                break;
            }
            let Some(&head) = self.waiting.front() else {
                break;
            };
            let admitted = match self.requests.get(&head).map(|request| &request.graph) {
                Some(Graph::Tokens(_)) => self.admit_tokens(head),
                Some(Graph::Video(_)) => self.admit_video(head),
                None => {
                    self.invariant_broken("a waiting request is in the request table");
                    ControlFlow::Break(())
                }
            };
            if admitted.is_break() {
                break;
            }
        }
    }

    /// Refuses one waiting request the policy rejected.
    ///
    /// The policy named it, and the mechanism still tells it the request
    /// ended: the queue it keeps is the policy's own, and an end it is not
    /// told of leaves a name behind that a later view change makes
    /// admissible again, which the mechanism would then refuse as not
    /// waiting.
    fn reject_waiting(&mut self, id: ipc::RequestId) {
        self.end_waiting(
            id,
            scheduling::guest::event::End::Rejected,
            Event::Rejected {
                kind: RejectionKind::Overloaded,
                message: "the scheduling policy rejected the request as over capacity".into(),
            },
        );
    }

    /// Ends the waiting request `id` with `event`, telling the policy how it
    /// ended. A request that is not waiting is left alone.
    pub(super) fn end_waiting(
        &mut self,
        id: ipc::RequestId,
        end: scheduling::guest::event::End,
        event: Event,
    ) {
        if let Some(request) = self.remove_waiting(id) {
            self.announce_end(id, end);
            let _ = request.event_tx().send(event);
        }
    }

    /// Whether the scheduler still holds the requests that continue from
    /// its predecessor's pages; once the wait ends, they return to the head
    /// of the queue, in their order.
    fn awaits_index(&mut self) -> bool {
        if self
            .awaiting_index
            .is_some_and(|until| std::time::Instant::now() < until)
        {
            return true;
        }
        self.release_held_for_index();
        false
    }

    /// Ends the wait for the predecessor's index: the requests held for it
    /// return to the head of the queue, in their order.
    pub(super) fn release_held_for_index(&mut self) {
        self.awaiting_index = None;
        for held in std::mem::take(&mut self.held_for_index).into_iter().rev() {
            self.waiting.push_front(held);
        }
    }

    /// Admits the waiting token request `id`, the head of the queue, when a
    /// slot and the cache units of its first call are free.
    ///
    /// A request that reserves its worst case (one with context images) is
    /// admitted when the free cache units cover the units of
    /// `max_reserved_cache_tokens` in every group and its encoder-cache
    /// entries fit the budget; any other request needs only the units of its
    /// first prefill chunk beyond its cached prefix, summed over every cache
    /// group. A head that can never fit the total cache capacity is rejected
    /// with `RejectionKind::Invalid`.
    ///
    /// Answers `Continue` when the head left the queue (admitted, rejected,
    /// or set aside for its predecessor's index) and `Break` when it waits:
    /// no slot is free, a running request waits for units, the head does not
    /// fit yet or finds no ready prefill worker, or an invariant broke.
    fn admit_tokens(&mut self, id: RequestId) -> ControlFlow<()> {
        // A running request the last assembly could not grow takes the units
        // that come free before any request is admitted: a request admitted
        // into them would be the first one the preemption gives up.
        if self.storage.request_pool.is_empty() || !self.starved.is_empty() {
            return ControlFlow::Break(());
        }
        let Some(continues) = self
            .requests
            .get(&id)
            .and_then(Request::tokens)
            .map(|head| head.req.cache.resume.is_some())
        else {
            self.invariant_broken("a waiting token request is in the request table");
            return ControlFlow::Break(());
        };
        // A request that continues from the drained predecessor's pages
        // waits for their index, which arrives once that epoch parked: a
        // request admitted before it computes again what the pages hold.
        // It steps aside, and the requests behind it are admitted.
        if continues && self.awaits_index() {
            self.waiting.retain(|queued| *queued != id);
            self.held_for_index.push(id);
            return ControlFlow::Continue(());
        }
        let Some(head) = self.requests.get(&id).and_then(Request::tokens) else {
            return ControlFlow::Break(());
        };
        // Submission rejects token requests when the worker has no prefix
        // cache, so a waiting one always finds it.
        let Some(cache) = self.storage.cache() else {
            self.invariant_broken("a queued token request has a prefix cache");
            return ControlFlow::Break(());
        };

        if head.reserve_worstcase {
            let tokens = head.max_reserved_cache_tokens;
            let need = cache.units_for_tokens(tokens);
            let encoder_entries = head.req.num_encoder_cache_entries();
            let encoder_ok = self
                .storage
                .reserved_encoder_entries
                .saturating_add(encoder_entries)
                <= self.storage.encoder_cache.budget();
            if need > self.storage.capacity_units() {
                self.reject_oversized(id);
                return ControlFlow::Continue(());
            }
            if self.storage.free_units() < need || !encoder_ok {
                return ControlFlow::Break(());
            }
            let Some((target, _)) = self.prefill_target(head) else {
                return ControlFlow::Break(());
            };
            let reservation = match self.reserve_admission(&target.0) {
                Ok(reservation) => reservation,
                Err(invariant) => {
                    self.invariant_broken(invariant);
                    return ControlFlow::Break(());
                }
            };
            self.admit_running(id, target, reservation);
            // Grow the tables to the whole worst case now, from token
            // zero, so later admissions cannot take these units. The
            // result is not checked here: a later call that needs
            // more units than the tables cover waits for them.
            self.ensure_request_capacity(id, 0, tokens);
            return ControlFlow::Continue(());
        }

        let n = head.req.prompt_token_ids.len();
        let Some((target, prefix_hit)) = self.prefill_target(head) else {
            return ControlFlow::Break(());
        };

        // Units the first prefill chunk needs beyond the cached
        // prefix. The chunk is bounded by `chunked_prefill_size`
        // and `max_num_batched_tokens` and holds at least one token.
        let chunk_limit = self
            .config
            .prefill_chunk()
            .min(self.config.max_num_batched_tokens)
            .max(1);
        let cached_tokens = prefix_hit.cached_tokens.min(n);
        // The host KV cache may hold the reusable pages that follow
        // the device match; the head waits for its pins. A request
        // continuing another epoch's state reads none: the host's
        // pages are keyed by the epoch that serves.
        let mut host_tokens = 0;
        let mut host = host_prefix(
            &mut self.kv_offload,
            self.storage.cache(),
            head,
            &target,
            cached_tokens,
            n,
            0,
        );
        // A lookup is a round trip to this host's Storage Server, which answers well
        // within a millisecond. Taking the answer at the next step instead admitted a burst
        // of prompts one per step: GSM8K at 64 slots ran 1320 prefill batches with the
        // host tier on against 1101 with it off. So the head waits a bounded moment for
        // its own answer; a prefetch from the KV storage, which takes longer, still waits
        // across steps.
        if matches!(host, HostHit::Pending) {
            let deadline = std::time::Instant::now() + HOST_ANSWER_WAIT;
            while matches!(host, HostHit::Pending) && std::time::Instant::now() < deadline {
                if let Some(kv_offload) = self.kv_offload.as_mut() {
                    kv_offload.poll();
                }
                std::thread::yield_now();
                host = host_prefix(
                    &mut self.kv_offload,
                    self.storage.cache(),
                    head,
                    &target,
                    cached_tokens,
                    n,
                    0,
                );
            }
        }
        match host {
            HostHit::Serving(boundary) => host_tokens = boundary - cached_tokens,
            // The host is looking up this request's pages: the head
            // waits for the pins.
            HostHit::Pending => return ControlFlow::Break(()),
            HostHit::Nothing => {}
        }
        // The two facts of the hit that no view can compute, recorded
        // where admission measured them.
        self.recorder.prefix_hit(
            scheduling::guest::RequestId(id.0),
            u32::try_from(cached_tokens).unwrap_or(u32::MAX),
            u32::try_from(host_tokens).unwrap_or(u32::MAX),
        );
        // One `Hit` per page the mechanism re-uses: a policy's LRU
        // orders evictions by these, so a page that was hit must move.
        for head_unit in &prefix_hit.matched_pages {
            self.recorder
                .hit(scheduling::guest::PageId(u64::from(*head_unit)));
        }
        let prefix_tokens = cached_tokens + host_tokens;
        let first_chunk = n.saturating_sub(prefix_tokens).min(chunk_limit).max(1);
        let first_chunk_units =
            cache.units_after_prefix(cached_tokens, prefix_tokens + first_chunk);

        // The prompt alone must fit the pool: every full-attention page
        // and each sliding-window group's pages of one chunk.
        if cache.min_units_for_prompt(n, chunk_limit) > self.storage.capacity_units() {
            self.reject_oversized(id);
            return ControlFlow::Continue(());
        }

        // Hit pages that are currently unreferenced count as free
        // until acquisition takes them, so they are subtracted from
        // the free units. A host hit holds the units of all its pages
        // from its admission, so the head's whole prompt fits beside
        // what the running requests' prompts still take: requests that hold
        // every unit with none for their prompts' tails would wait for
        // one another. What running requests generate is not reserved: one
        // that cannot grow in decode is relieved by the preemption, which
        // gives up the request admitted last to be computed again from
        // its tokens (`Scheduler::preempt_shortage`).
        let head_units = cache
            .units_after_prefix(cached_tokens, n)
            .min(cache.min_units_for_prompt(n, chunk_limit))
            .max(first_chunk_units);
        let capacity_available = self
            .storage
            .free_units()
            .saturating_sub(prefix_hit.cached_free_units)
            .saturating_sub(self.units_running_still_need(chunk_limit))
            >= head_units;
        // The head does not fit yet. Later requests do not bypass it, so
        // admission waits for capacity to free.
        if !capacity_available {
            return ControlFlow::Break(());
        }
        let reservation = match self.reserve_admission(&target.0) {
            Ok(reservation) => reservation,
            Err(invariant) => {
                self.invariant_broken(invariant);
                return ControlFlow::Break(());
            }
        };
        let worker = target.0.clone();
        self.admit_running(id, target, reservation);
        self.admit_offloaded_prefix(id, &worker);
        ControlFlow::Continue(())
    }

    /// Rejects the waiting request `id`, whose prompt can never fit the
    /// cache.
    ///
    /// The mechanism ends a request the policy still queues, so it says so:
    /// a policy learns a request from `Arrived` and an end from this, and a
    /// name it is not told has gone is one it offers in `admit` again.
    fn reject_oversized(&mut self, id: RequestId) {
        self.end_waiting(
            id,
            scheduling::guest::event::End::Rejected,
            Event::Rejected {
                kind: RejectionKind::Invalid,
                message: "request exceeds total KV capacity".into(),
            },
        );
    }

    /// The units the running requests still take to compute their whole prompts in chunks of
    /// `chunk`.
    fn units_running_still_need(&self, chunk: usize) -> usize {
        let Some(cache) = self.storage.cache() else {
            return 0;
        };
        self.requests
            .running_tokens()
            .map(|(_, _, state)| state)
            .filter(|state| {
                (state.num_computed_prompt_tokens as usize) < state.req.prompt_token_ids.len()
            })
            .filter_map(|state| {
                let held: usize = state
                    .block_tables()?
                    .iter()
                    .map(BlockTable::held_units)
                    .sum();
                let prompt = state.req.prompt_token_ids.len();
                Some(
                    cache
                        .min_units_for_prompt(prompt, chunk)
                        .saturating_sub(held),
                )
            })
            .sum()
    }

    /// Probes the prefix cache on the configured prefill owner.
    ///
    /// Picks the first ready worker that serves prefill and returns it with
    /// the component name it binds, plus the prefix hit on that worker's
    /// endpoint (prefix-cache residency is keyed by endpoint). The hit is an
    /// estimate; `acquire_cached_prefix` takes the actual prefix. Returns
    /// `None` when the runtime has no prefix cache or no prefill worker is ready.
    pub(super) fn prefill_target(
        &self,
        state: &TokenState,
    ) -> Option<((crate::WorkerGroupId, String), crate::kv_cache::PrefixHit)> {
        let cache = self.storage.cache()?;
        self.dispatch
            .worker_candidates(
                &self.worker,
                &self.info,
                CallKind::Forward(ForwardMode::Prefill),
            )
            .filter(|(worker, _, _)| self.worker.is_ready(worker))
            .map(|(worker, component, info)| {
                let hit = cache.probe_prefix(
                    &state.req.prompt_token_ids,
                    state.req.cache.read,
                    state.has_context_images(),
                    PrefixScope::of(&state.req.cache),
                    &info.endpoint,
                );
                ((worker.clone(), component.to_owned()), hit)
            })
            .next()
    }

    /// Accumulates one request's interval from queue entry to admission.
    ///
    /// Both stamps are monotonic microseconds (`clock::now_monotonic_us`), so
    /// a wall-clock step does not skew the statistics.
    fn record_queue_wait(&self, queued_us: u64, scheduled_us: u64) {
        let queue_wait_us = scheduled_us.saturating_sub(queued_us);
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
            .worker
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

    /// Admits the waiting token request `id` with its reservation: it leaves
    /// the queue and runs.
    ///
    /// Records `target`'s worker as the owner of its stage for the request,
    /// emits the public `Scheduled` event (a closed receiver marks the
    /// request cancelled), adds its encoder-cache entries to the reserved
    /// count, and acquires its cached prefix, which sets the computed-prompt
    /// cursors to the reused tokens. A failed acquisition latches
    /// engine-fatal.
    pub(super) fn admit_running(
        &mut self,
        id: RequestId,
        target: (crate::WorkerGroupId, String),
        reservation: AdmissionReservation,
    ) {
        self.tell_admitted(id);
        self.waiting.retain(|queued| *queued != id);
        let AdmissionReservation {
            source,
            request_slot,
            prefix_cache,
        } = reservation;
        let scheduled_us = crate::clock::now_monotonic_us();
        let Some(Request {
            status,
            queued_us,
            graph: Graph::Tokens(st),
        }) = self.requests.get_mut(&id)
        else {
            self.invariant_broken("an admitted token request is in the request table");
            return;
        };
        *status = RequestStatus::Running;
        let queued_us = *queued_us;
        let request_key = RequestKey::new(self.engine_id, id, st.request_epoch);
        if st
            .allocations
            .replace(RequestAllocations {
                request_slot,
                prefix_cache,
                buffers: HashMap::new(),
            })
            .is_none()
        {
            // The request holds a slot from now until its execution ends,
            // which is what `Room::slots` counts without visiting the
            // running requests.
            self.rows_held = self.rows_held.saturating_add(1);
        }
        // Queue latency is finalized at the same timestamp exposed through the
        // public scheduling event.
        let encoder_entries = st.req.num_encoder_cache_entries();
        if !st.output.events.enqueue(Event::Scheduled {
            queued_at: crate::clock::unix_secs_at(queued_us),
            scheduled_at: crate::clock::unix_secs_at(scheduled_us),
        }) {
            status.finish(Terminal::finish(FinishReason::Cancelled));
        }
        let resume = st.req.cache.resume;

        // Running order, resource reservations, and runtime ownership advance
        // together so the next scheduling pass observes one coherent admission.
        self.dispatch
            .affinity
            .insert((request_key, target.1), target.0);
        self.record_queue_wait(queued_us, scheduled_us);
        self.running.push(id);
        self.storage.reserved_encoder_entries = self
            .storage
            .reserved_encoder_entries
            .saturating_add(encoder_entries);

        // Prefix-cache acquisition pins every reused block to this request's
        // newly installed block tables.
        let acquired = match (self.requests.tokens_mut(&id), self.storage.cache.as_ref()) {
            (Some(st), Some(prefix_cache)) => {
                acquire_cached_prefix(prefix_cache, st, &self.stats, &source)
            }
            _ => Err("an admitted request runs with a prefix cache"),
        };
        if let Err(invariant) = acquired {
            self.invariant_broken(invariant);
        }
        if let Some(resume) = resume {
            self.warming.claimed(resume);
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
        let Some(state) = self.requests.tokens_mut(&id) else {
            return;
        };
        if !state.req.cache.read || state.has_context_images() {
            return;
        }
        state.host_keyed = kv_offload.serves(worker);
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
            (Some(cache), Some(allocation)) => cache.release_window(allocation, boundary, None),
            _ => false,
        };
        if !released || !self.ensure_request_capacity(id, computed, boundary) {
            if let Some(kv_offload) = self.kv_offload.as_mut() {
                kv_offload.abandon(id);
            }
            return;
        }
        let Some(state) = self.requests.tokens_mut(&id) else {
            return;
        };
        // The draft cache pages the hit loads into units are valid once
        // loaded, as its other pages are.
        let draft = self.storage.cache.as_ref().and_then(|cache| {
            let group = cache.draft_group()?;
            let pages = self.kv_offload.as_ref()?.hit_pages(id).get(group)?.clone();
            let block = cache.groups()[group].block_size;
            let (start, end) = (pages.start * block, pages.end * block);
            let allocation = &state.allocations()?.prefix_cache;
            cache
                .holds_draft(allocation, start, end)
                .then_some((start as u32, end as u32))
        });
        if let Some((start, end)) = draft {
            state.draft.record(start, end);
        }
        // The hit's pages are computed once loaded: the first call starts
        // after them, and they are published to the device cache when that
        // call's result is accepted.
        state.num_computed_prompt_tokens = boundary as u32;
        state.logical_position = state.num_computed_prompt_tokens;
        state.cache_visible_len = state.num_computed_prompt_tokens;
        state.cache_computed_len = state.num_computed_prompt_tokens;
        state.offload_boundary = boundary;
        let loaded = (boundary - computed) as u64;
        self.stats
            .prefix
            .hit_tokens
            .fetch_add(loaded, Ordering::Relaxed);
        self.stats
            .prefix
            .host_hit_tokens
            .fetch_add(loaded, Ordering::Relaxed);
        let peer = self
            .kv_offload
            .as_mut()
            .map_or(0, |kv_offload| kv_offload.take_peer_tokens(id));
        self.stats
            .prefix
            .peer_hit_tokens
            .fetch_add(peer as u64, Ordering::Relaxed);
    }
}

/// Resources a token request holds from the moment admission commits it.
pub(super) struct AdmissionReservation {
    /// The prefill worker's endpoint, which keys its prefix-cache residency.
    source: ipc::MainProcessEndpoint,
    request_slot: RequestSlot,
    prefix_cache: CacheAllocation,
}

/// How long admission waits for the head request's host KV lookup within one step.
const HOST_ANSWER_WAIT: std::time::Duration = std::time::Duration::from_millis(2);

/// What the host KV cache says about a request's prompt beyond its device hit.
pub(super) enum HostHit {
    /// Nothing to look for: no host cache serves that worker, the request
    /// reads another epoch's pages or continues context images, or its prompt
    /// is on the device already.
    Nothing,
    /// The host serves the prompt up to this token boundary.
    Serving(usize),
    /// The host is looking the pages up, and the request waits for the pins.
    Pending,
}

/// Starts the host KV cache looking for the pages a request can reuse beyond
/// the device hit it already has, up to `limit` tokens of its prompt.
///
/// Admission asks for the boundary it needs; a plan's `prefetch` asks again
/// later, and the answer grows as host pages arrive. Reusable pages leave one
/// prompt token to compute, as in the device cache.
pub(super) fn host_prefix(
    kv_offload: &mut Option<KvOffloadManager>,
    coordinator: Option<&PrefixCacheCoordinator>,
    state: &TokenState,
    target: &(crate::WorkerGroupId, String),
    device_hit: usize,
    limit: usize,
    min_deepening: usize,
) -> HostHit {
    let (Some(kv_offload), Some(coordinator)) = (kv_offload.as_mut(), coordinator) else {
        return HostHit::Nothing;
    };
    let Some(step) = kv_offload.step(&target.0) else {
        return HostHit::Nothing;
    };
    if !state.req.cache.read
        || !state.req.cache.state_seeds.is_empty()
        || state.has_context_images()
    {
        return HostHit::Nothing;
    }
    let limit = limit.min(state.req.prompt_token_ids.len().saturating_sub(1)) / step * step;
    if limit <= device_hit {
        return HostHit::Nothing;
    }
    // The host pools hold a page under the key the device cache chains it
    // under: one chain, one key per page at every level.
    let keys = coordinator.prefix_hashes(
        &state.req.prompt_token_ids[..limit.min(state.req.prompt_token_ids.len())],
        PrefixScope::of(&state.req.cache),
    );
    match kv_offload.admission(
        state.req.request_id,
        &target.0,
        &keys,
        device_hit,
        limit,
        min_deepening,
    ) {
        Some(boundary) => HostHit::Serving(boundary),
        None => HostHit::Pending,
    }
}

/// Acquires the longest prefix every cache group can reuse and records cache
/// accounting on the request.
///
/// Fails, acquiring nothing, when the request holds no block tables or its
/// tables do not match the pool's cache groups.
fn acquire_cached_prefix(
    coordinator: &PrefixCacheCoordinator,
    state: &mut TokenState,
    stats: &SchedulerStats,
    source: &ipc::MainProcessEndpoint,
) -> Result<(), &'static str> {
    let prompt = state.req.prompt_token_ids.clone();
    let has_context_images = state.has_context_images();
    let cache_read = state.req.cache.read;
    let cache_policy = state.req.cache.clone();
    let scope = PrefixScope::of(&cache_policy);
    let allocation = state
        .prefix_cache_mut()
        .ok_or("an admitted request holds its block tables")?;
    let hit = coordinator
        .acquire_prefix(
            allocation,
            &prompt,
            cache_read,
            has_context_images,
            scope,
            source,
        )
        .ok_or("an admitted request's block tables cover every cache group")?;

    // A stored partial page can cover any prefix while leaving the final
    // prompt token to compute the logits. Count that same lookup extent.
    let query_tokens = prompt.len().saturating_sub(1);
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
    // The draft cache's pages publish as the request's entries complete
    // them; the reused ones, which the prefix holds, are its valid prefix.
    state.prefix_published = coordinator
        .groups()
        .iter()
        .map(|group| {
            if group.draft() {
                hit.draft_tokens / group.block_size
            } else {
                hit.cached_tokens / group.block_size
            }
        })
        .collect();
    state.draft = Default::default();
    state.draft.valid = hit.draft_tokens as u32;
    state.prefix_page_hashes = hit.page_hashes;
    state.num_computed_prompt_tokens = hit.cached_tokens as u32;
    state.logical_position = state.num_computed_prompt_tokens;
    state.cache_visible_len = state.num_computed_prompt_tokens;
    state.cache_computed_len = state.num_computed_prompt_tokens;
    Ok(())
}
