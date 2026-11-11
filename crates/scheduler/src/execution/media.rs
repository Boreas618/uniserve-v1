//! Video media scheduling: a video request runs the calls of its graph
//! (`graph::VideoGraph`, built at admission from its task and conditions),
//! offered by `ready_calls` as their inputs become available. Calls occupy
//! their stages' capacity on their workers and, for media reading,
//! encoding and muxing, host task pools, tracked by the media occupancy
//! ([`Occupancy`]); `prepare_media_call` builds one call with its products,
//! buffer bindings, latent pages and reading stages, and
//! `assemble_batch` schedules a media call of the step's output onto the batch of its
//! phase and stage.

use super::*;
use ipc::MediaCall;

/// A video request's graph and what it holds: the calls of its chain
/// (`graph::VideoGraph`), the products its calls bound, its storage and its
/// output. Built when the request arrives; admission fills what the request
/// holds (`allocations`, `reservations`, `admission`).
pub(crate) struct VideoState {
    pub(crate) request: DiffusionRequest,
    /// The request's identity on the workers, fixed when it arrives.
    pub(crate) request_key: RequestKey,
    /// The calls the request runs, the products connecting them and where
    /// each call stands (`graph::NodeState`), which readiness is derived from.
    pub(crate) graph: VideoGraph,
    pub(crate) output: output::EventJournal,
    /// The rows and output buffers admission reserved on the routed workers;
    /// empty while the request waits.
    pub(crate) allocations: MediaAllocations,
    /// Each product the request reserved, by its declared name, with the
    /// stage output it is reserved under.
    pub(crate) reservations: HashMap<&'static str, (String, u32)>,
    /// Logical products mapped to their reserved stage output and unit slice.
    pub(crate) buffer_bindings: HashMap<BufferId, (String, u32, u32)>,
    /// The media reader's products, by name, once its call is scheduled.
    pub(crate) condition_media: HashMap<&'static str, TensorRef>,
    /// The vision encoder's token features, once its call is scheduled.
    pub(crate) vision_features: Option<TensorRef>,
    /// Visual condition units whose latent encoding is scheduled, and of
    /// those, the units whose encoding completed.
    pub(crate) scheduled_condition_units: u32,
    pub(crate) encoded_condition_units: u32,
    /// Each visual latent encoding round's rows, in unit order.
    pub(crate) condition_video_latents: Vec<TensorRef>,
    /// The audio latent encoding's rows, once its call is scheduled.
    pub(crate) condition_audio_latents: Option<TensorRef>,
    pub(crate) conditioning: Option<TensorRef>,
    pub(crate) latents: Vec<TensorRef>,
    /// Decoded media units by the cursor of the round that produced them,
    /// until their encode round retires them.
    pub(crate) decoded_units: BTreeMap<u32, (u32, TensorRef)>,
    /// Encoded media units by the cursor of the round that produced them,
    /// until the muxer is handed them.
    pub(crate) encoded_units: BTreeMap<u32, (u32, TensorRef)>,
    /// Media unit count of each encode round that has completed and not yet
    /// been handed to the muxer, by the round's cursor.
    pub(crate) encoded_ready: BTreeMap<u32, u32>,
    /// Media units handed to the muxer so far; the next round it takes
    /// starts here.
    pub(crate) handed_units: u32,
    /// Encoded products carried by the muxing call in flight, retired with it.
    pub(crate) muxing_inputs: Vec<TensorRef>,
    pub(crate) audio: Option<TensorRef>,
    /// Static worker admission, sent as a `BatchCommand::Start` in the batch
    /// of the request's first scheduled call; `None` while the request waits.
    pub(crate) admission: Option<NewRequest>,
    pub(crate) admission_state: WorkerRegistration,
    /// Solver progress of the request's latent trajectory.
    pub(crate) denoising: Denoising,
    pub(crate) scheduled_decode_units: u32,
    pub(crate) scheduled_encode_units: u32,
    pub(crate) predecessor: CallId,
    pub(crate) artifact: Option<ArtifactEvent>,
    /// The request's trajectory in the host latent pools.
    pub(crate) host_trajectory: HostTrajectory,
}

impl VideoState {
    /// The state of a video request that just arrived: its graph, with
    /// nothing scheduled and nothing held.
    pub(crate) fn new(
        request: DiffusionRequest,
        request_key: RequestKey,
        graph: VideoGraph,
        event_tx: EventTx,
        steps: u32,
    ) -> Self {
        Self {
            request,
            request_key,
            graph,
            output: output::EventJournal::new(event_tx),
            allocations: MediaAllocations {
                request_slots: HashMap::new(),
                tensors: HashMap::new(),
            },
            reservations: HashMap::new(),
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
            admission: None,
            admission_state: WorkerRegistration::Unsubmitted,
            denoising: Denoising::new(steps),
            scheduled_decode_units: 0,
            scheduled_encode_units: 0,
            predecessor: CallId::new(0, 0),
            artifact: None,
            host_trajectory: HostTrajectory::default(),
        }
    }
}

/// Denoising steps one video request keeps in flight: the step its
/// denoiser runs and the next, queued behind it so that the denoiser never
/// waits for the host between steps. A preemption (a park, a sleep, a
/// weight update) therefore meets the request at a step boundary within
/// that many steps, rather than after every step of its schedule.
pub(crate) const DENOISING_LOOKAHEAD: usize = 2;

/// A video request's encoder output `name` in the host encoder pools: a
/// video request has one output of each name.
fn video_output(name: &'static str) -> crate::encoder_outputs::OutputId {
    crate::encoder_outputs::OutputId { name, ordinal: 0 }
}

/// What one media call occupies while it is in flight.
///
/// On a stage whose capacity is measured in media units the call
/// carries `units`; a demand with zero `units` holds its stage
/// exclusively, one request at a time. `cpu_workers` names the host task
/// pools the call places tasks on.
pub(crate) struct Demand {
    /// The worker and stage whose capacity the call occupies, taken
    /// from the request's dispatch (`Dispatch::affinity`). Admission places
    /// a media request on a worker for every media stage, so a request's
    /// first call is checked against its stage too. `None` when no
    /// stage serves the call or the request holds no dispatch for it.
    stage: Option<(crate::WorkerGroupId, String)>,
    units: u32,
    /// Host task pools this call occupies, each named by its worker and rank
    /// with the number of tasks the call places on it: one per media unit
    /// the rank encodes, one for any other host call.
    cpu_workers: Vec<(crate::WorkerGroupId, usize, u32)>,
}

/// What the media calls in flight occupy of their stages and of the
/// ranks' host task pools.
///
/// `media_occupancy` rebuilds it from the in-flight calls at the start of each
/// media pass, and at the first inline-image encoder call of a generation
/// pass; the pass charges each call it selects.
#[derive(Default)]
pub(crate) struct Occupancy {
    /// The request holding each exclusively held stage.
    exclusive: HashMap<(crate::WorkerGroupId, String), RequestId>,
    /// Media units in flight on each unit-measured stage.
    units: HashMap<(crate::WorkerGroupId, String), u32>,
    /// Each rank owns one host task pool, so occupancy is counted per rank
    /// of each worker.
    host: HashMap<(crate::WorkerGroupId, usize), u32>,
}

impl Occupancy {
    /// Returns whether one request's call fits what it occupies.
    ///
    /// Every host task pool the call places tasks on must have room for
    /// them. An exclusively held stage admits further calls of the
    /// request already holding it.
    pub(crate) fn admits(
        &self,
        demand: &Demand,
        request: RequestId,
        scheduler: &Scheduler,
    ) -> bool {
        if demand.cpu_workers.iter().any(|(worker, rank, tasks)| {
            let capacity = scheduler.host_task_capacity(worker);
            let key = (worker.clone(), *rank);
            self.host.get(&key).copied().unwrap_or(0) + tasks > capacity
        }) {
            return false;
        }
        let Some((worker, stage)) = demand.stage.as_ref() else {
            return true;
        };
        if demand.units > 0 {
            let capacity = scheduler.stage_units(worker, stage).unwrap_or(1);
            self.units
                .get(&(worker.clone(), stage.clone()))
                .copied()
                .unwrap_or(0)
                + demand.units
                <= capacity
        } else {
            self.exclusive
                .get(&(worker.clone(), stage.clone()))
                .is_none_or(|holder| *holder == request)
        }
    }

    /// Marks what one request's call occupies until it completes.
    pub(crate) fn occupy(&mut self, demand: &Demand, request: RequestId) {
        for (worker, rank, tasks) in &demand.cpu_workers {
            *self.host.entry((worker.clone(), *rank)).or_default() += tasks;
        }
        let Some((worker, stage)) = demand.stage.as_ref() else {
            return;
        };
        let key = (worker.clone(), stage.clone());
        if demand.units > 0 {
            *self.units.entry(key).or_default() += demand.units;
        } else {
            self.exclusive.insert(key, request);
        }
    }
}

impl Scheduler {
    /// The cursors of the completed encode rounds the muxer can take next:
    /// the contiguous run that starts where the units it holds end.
    fn ready_encode_rounds(state: &VideoState) -> Vec<u32> {
        let mut rounds = Vec::new();
        let mut cursor = state.handed_units;
        while let Some(units) = state.encoded_ready.get(&cursor) {
            rounds.push(cursor);
            cursor += units;
        }
        rounds
    }

    /// The calls of request `id`'s graph that are ready now, in the graph's
    /// order: the first is its next phase, which the policy reads.
    ///
    /// A token request's loop has at most one (`token_next_call`); a video
    /// request's chain offers every call whose inputs are done
    /// (`video_ready_calls`). Empty when the request is not running or has
    /// nothing ready.
    pub(crate) fn ready_calls(&self, id: RequestId) -> Vec<CallKind> {
        match self.media_state(id) {
            Some(state) => self
                .video_ready_calls(state)
                .into_iter()
                .map(CallKind::Media)
                .collect(),
            None => self.token_next_call(id).into_iter().collect(),
        }
    }

    /// Lists every call of one video request that can be scheduled now.
    ///
    /// A request is a graph of calls with data dependencies, so a step offers
    /// all of them and the media occupancy decides which ones fit. At most one
    /// call of each kind is ready at once: the cursors this returns against
    /// advance as calls are scheduled, so a unit group only becomes ready
    /// once its predecessor is submitted.
    ///
    /// A conditioned request first reads its media on a CPU worker. The calls
    /// reading those host products wait until the read has completed, so a
    /// product a call names always exists. The encoders, the latent
    /// preparation, the denoising steps and the decoders become
    /// ready once their predecessor is scheduled; they name their inputs by
    /// product reference and do not wait here for the producer to complete.
    /// Video and audio encoding wait until the call producing their input has
    /// completed, and muxing waits for completed encode rounds.
    pub(crate) fn video_ready_calls(&self, state: &VideoState) -> Vec<MediaCall> {
        let sampling = state.request.sampling;
        let graph = &state.graph;
        // A product is produced once its producing call has left the
        // request's in-flight calls.
        let produced = |product: &TensorRef| {
            !self
                .inflight
                .pending_calls
                .get(&state.request.request_id)
                .is_some_and(|calls| {
                    calls
                        .iter()
                        .any(|call| call.call.call_id == product.producer_call_id)
                })
        };

        // The calls before latent preparation, then latent preparation and
        // the denoising steps, are scheduled in full before any later call
        // is offered.
        if graph.reads_media() {
            if graph.pending(MediaCall::MediaReading) {
                return vec![MediaCall::MediaReading];
            }
            if !state.condition_media.values().all(produced) {
                return Vec::new();
            }
        }
        // An encoder is offered once its output's lookup in the host encoder
        // pool resolved, so that every call of the output reads it or every
        // call computes it.
        let resolved = |name: &'static str| {
            self.encoder_outputs
                .as_ref()
                .is_none_or(|store| store.resolved(state.request_key, video_output(name)))
        };
        let mut ready = Vec::new();
        if graph.encodes_vision() && graph.pending(MediaCall::VisionEncoding) {
            if resolved(product::VISION_FEATURES) {
                ready.push(MediaCall::VisionEncoding);
            }
        } else if graph.pending(MediaCall::TextEncoding)
            && state.vision_features.as_ref().is_none_or(produced)
            && resolved(product::CONDITIONING)
        {
            ready.push(MediaCall::TextEncoding);
        }
        // Latent encoding is offered while a rank of the latent encoder is
        // free to take units (`latent_round`), from the first tick: the ranks
        // that encode no prompt start on the condition units beside the
        // vision encoding, and ranks holding both encoders take units only
        // once the prompt's encoding, whose ranks synchronize in its
        // collectives, is placed ahead of them.
        let visual_units = graph.condition_units().len() as u32;
        let encoding_remains = state.scheduled_condition_units < visual_units
            || (graph.encodes_condition_audio() && state.condition_audio_latents.is_none());
        let latents = if state.scheduled_condition_units < visual_units {
            product::CONDITION_VIDEO_LATENTS
        } else {
            product::CONDITION_AUDIO_LATENTS
        };
        if encoding_remains && self.latent_round(state).is_some() && resolved(latents) {
            ready.push(MediaCall::LatentEncoding);
        }
        // Latent preparation reads the text features and every condition
        // latent, so nothing later is offered while the text encoding waits
        // for its vision features or a latent encoding call for a rank.
        if !ready.is_empty() || encoding_remains || graph.pending(MediaCall::TextEncoding) {
            return ready;
        }
        // A resumed request's preparation waits until its stored
        // trajectory's pages are pinned, or found missing.
        if graph.pending(MediaCall::LatentPreparation) {
            if state.host_trajectory.restoring == Restoring::Pinning {
                return Vec::new();
            }
            return vec![MediaCall::LatentPreparation];
        }
        if self.scheduled_steps(state.request.request_id, &state.denoising)
            < state.denoising.steps()
        {
            // A request keeps at most `DENOISING_LOOKAHEAD` steps in flight,
            // so a preemption meets it at a step boundary within that many
            // steps.
            let in_flight = self
                .inflight
                .pending_calls
                .get(&state.request.request_id)
                .into_iter()
                .flatten()
                .filter(|pending| pending.call.kind == CallKind::Media(MediaCall::Denoising))
                .count();
            if in_flight >= DENOISING_LOOKAHEAD {
                return Vec::new();
            }
            return vec![MediaCall::Denoising];
        }

        // Consume completed decoder outputs first. Audio and video retain
        // independent readiness and capacity when the other branch is busy.
        if let Some((_, product)) = state.decoded_units.get(&state.scheduled_encode_units)
            && produced(product)
        {
            ready.push(MediaCall::VideoEncoding);
        }
        if graph.pending(MediaCall::AudioEncoding) && state.audio.as_ref().is_some_and(produced) {
            ready.push(MediaCall::AudioEncoding);
        }
        if state.scheduled_decode_units < sampling.video_units {
            ready.push(MediaCall::VideoDecoding);
        }
        if graph.pending(MediaCall::AudioDecoding) {
            ready.push(MediaCall::AudioDecoding);
        }
        // The muxer consumes media units as they arrive: a muxing call takes
        // the encode rounds that have completed since the last one, in
        // order, and a final call carrying no units assembles the artifact
        // once every unit and the audio track are in. It waits for completed
        // writes, never just their scheduled counts.
        if graph.pending(MediaCall::Muxing) {
            let next_round_ready = state.encoded_ready.contains_key(&state.handed_units);
            let everything_in =
                graph.done(MediaCall::AudioEncoding) && state.handed_units == sampling.video_units;
            if next_round_ready || everything_in {
                ready.push(MediaCall::Muxing);
            }
        }
        ready
    }

    /// Looks up admitted video request `id`'s encoder outputs in the host
    /// encoder pools (`crate::encoder_outputs`): each output its graph
    /// computes, keyed by the digest of what its encoder reads, which the
    /// request names (`DiffusionRequest::encoder_inputs`), and its rows of
    /// static width as the request reserved them. A request that names no
    /// digests, or an output the pools do not serve, computes them.
    pub(crate) fn look_up_encoder_outputs(&mut self, id: RequestId) {
        let Some(state) = self.media_state(id) else {
            return;
        };
        let Some(inputs) = state.request.encoder_inputs.as_ref() else {
            return;
        };
        let request_key = state.request_key;
        // The visual condition latents are one output, of every unit.
        let units = (!inputs.video_units.is_empty()).then(|| {
            let mut digest = storage::key::Digest::new(b"encoder-units");
            for unit in &inputs.video_units {
                digest.bytes(unit);
            }
            digest.finish()
        });
        let outputs = [
            (
                MediaCall::VisionEncoding,
                product::VISION_FEATURES,
                inputs.vision,
            ),
            (
                MediaCall::TextEncoding,
                product::CONDITIONING,
                Some(inputs.text),
            ),
            (
                MediaCall::LatentEncoding,
                product::CONDITION_VIDEO_LATENTS,
                units,
            ),
            (
                MediaCall::LatentEncoding,
                product::CONDITION_AUDIO_LATENTS,
                inputs.audio,
            ),
        ];
        let mut lookups = Vec::new();
        for (media_call, name, digest) in outputs {
            let Some(digest) = digest else {
                continue;
            };
            let Some(rows) = state.graph.product(name).and_then(|product| product.extent) else {
                continue;
            };
            let Some(reservation) = state.reservations.get(name) else {
                continue;
            };
            let reserved = &state.allocations.tensors[reservation];
            // A row's bytes, when every extent but the rows' is static.
            let Some(row_bytes) = reserved.shape_bound.dims[1..].iter().try_fold(
                reserved.dtype.element_bytes(),
                |bytes, dim| match dim {
                    DimBound::Static(extent) => Some(bytes * u64::from(*extent)),
                    DimBound::Device { .. } => None,
                },
            ) else {
                continue;
            };
            let Some(worker) = self
                .info
                .media_stages
                .get(&media_call)
                .and_then(|stage| self.dispatch.affinity.get(&(request_key, stage.clone())))
            else {
                continue;
            };
            lookups.push((worker.clone(), name, digest, rows, row_bytes));
        }
        let Some(store) = self.encoder_outputs.as_mut() else {
            return;
        };
        for (worker, name, digest, rows, row_bytes) in lookups {
            store.lookup(
                request_key,
                &worker,
                video_output(name),
                &digest,
                rows,
                row_bytes,
                crate::encoder_outputs::Access::FULL,
            );
        }
    }

    /// Returns the steps of a request's trajectory covered once its submitted
    /// calls complete.
    pub(crate) fn scheduled_steps(&self, id: RequestId, progress: &Denoising) -> u32 {
        progress.scheduled(
            self.inflight
                .pending_calls
                .get(&id)
                .into_iter()
                .flatten()
                .filter_map(|pending| match pending.call.kind {
                    CallKind::Media(media_call) => Some((
                        media_call,
                        pending
                            .call
                            .latent_restore
                            .as_ref()
                            .map_or(0, |restore| restore.steps),
                    )),
                    _ => None,
                }),
        )
    }

    /// Returns the stage that serves one media call.
    fn media_stage(&self, media_call: MediaCall) -> Option<String> {
        self.info.media_stages.get(&media_call).cloned()
    }

    /// Returns the worker serving `stage` for one request, with the
    /// stage's dispatch there.
    ///
    /// The worker is the request's owner of `stage` when it has one,
    /// otherwise the first candidate that can execute `work`. Returns `None`
    /// when no such worker is loaded or its bound stage has no binding
    /// on that worker.
    fn placed_stage<'a>(
        &'a self,
        request: RequestKey,
        work: CallKind,
        stage: &'a str,
    ) -> Option<(&'a crate::WorkerGroupId, &'a crate::BlockConfig)> {
        let owner = self.dispatch.affinity.get(&(request, stage.to_owned()));
        let (worker, bound, _) = self
            .dispatch
            .stage_candidates(&self.worker, work, stage)
            .find(|(worker, _, _)| owner.is_none_or(|owner| *worker == owner))?;
        let config = self.worker.info().stage_config(worker, bound)?;
        Some((worker, config))
    }

    /// Returns how many media units one call of this kind covers at once.
    ///
    /// A distributed stage reconstructs `units_per_rank` media units on
    /// each of its ranks per round; any other stage counts one unit per
    /// rank, on the worker `placed_stage` names.
    ///
    /// Returns `None` when the stage has no dispatch or the width does
    /// not fit in `u32`.
    fn stage_width(&self, request: RequestKey, work: CallKind, stage: &str) -> Option<u32> {
        let (_, config) = self.placed_stage(request, work, stage)?;
        let units_per_rank = if config.distribution.is_some() {
            config.units_per_rank.max(1)
        } else {
            1
        };
        u32::try_from(config.ranks.len().saturating_mul(units_per_rank)).ok()
    }

    /// Returns the next latent encoding call of one request: the condition
    /// units it covers and the ranks that hold them.
    ///
    /// The call is a visual round while condition units remain unscheduled
    /// and otherwise the request's audio tracks, which are one unit.
    /// Condition units are encoded independently of one another, so a
    /// distributed latent encoder deals a call only to its free ranks, in
    /// the stage's rank order: those that hold none of the request's
    /// calls in flight and that the prompt's encoding, until its text
    /// encoding is placed, does not take. A rank still encoding the prompt
    /// or an earlier round thus takes no unit that a free rank would
    /// otherwise wait beside. Returns `None` while every rank is held; each
    /// completing call frees its ranks.
    ///
    /// A latent encoder whose members encode together has no deal: its call
    /// covers the stage's width and names no rank.
    ///
    /// Returns `None` as well when no stage or worker serves latent
    /// encoding.
    fn latent_round(&self, state: &VideoState) -> Option<(u32, Vec<u32>)> {
        let request = state.request_key;
        let work = CallKind::Media(MediaCall::LatentEncoding);
        let stage = self.media_stage(MediaCall::LatentEncoding)?;
        let (worker, config) = self.placed_stage(request, work, &stage)?;
        let remaining =
            state.graph.condition_units().len() as u32 - state.scheduled_condition_units;
        if config.distribution.is_none() {
            let width = u32::try_from(config.ranks.len()).ok()?;
            let units = if remaining > 0 {
                width.min(remaining)
            } else {
                1
            };
            return Some((units, Vec::new()));
        }

        // A call in flight on this worker holds the ranks it was dealt to,
        // or every rank of its stage when it names none.
        let mut held = std::collections::HashSet::new();
        // The vision and text encodings run one after the other on the text
        // encoder's ranks, which are theirs from the start: between the two
        // calls no call of the prompt is in flight there.
        if state.graph.pending(MediaCall::TextEncoding)
            && let Some(name) = self.media_stage(MediaCall::TextEncoding)
            && let Some((owner, prompt)) =
                self.placed_stage(request, CallKind::Media(MediaCall::TextEncoding), &name)
            && owner == worker
        {
            held.extend(prompt.ranks.iter().copied());
        }
        for pending in self
            .inflight
            .pending_calls
            .get(&state.request.request_id)
            .into_iter()
            .flatten()
        {
            let name = &pending.call.stage;
            if self.dispatch.affinity.get(&(request, name.clone())) != Some(worker) {
                continue;
            }
            match &pending.input {
                InflightInput::Media {
                    decode: Some(range),
                    ..
                } if !range.ranks.is_empty() => {
                    held.extend(range.ranks.iter().map(|rank| *rank as usize));
                }
                _ => held.extend(
                    self.worker
                        .info()
                        .stage_config(worker, name)
                        .into_iter()
                        .flat_map(|config| config.ranks.iter().copied()),
                ),
            }
        }

        let free = config
            .ranks
            .iter()
            .filter(|rank| !held.contains(*rank))
            .map(|rank| *rank as u32)
            .collect::<Vec<_>>();
        if free.is_empty() {
            return None;
        }
        let per_rank = config.units_per_rank.max(1) as u32;
        let units = if remaining > 0 {
            remaining.min(free.len() as u32 * per_rank)
        } else {
            1
        };
        let holders = units.div_ceil(per_rank) as usize;
        Some((units, free[..holders].to_vec()))
    }

    /// Returns a stage's capacity on `worker` in media units.
    ///
    /// A stage that distributes its work measures its capacity in the
    /// units its ranks reconstruct together; every other stage is held
    /// exclusively and admits one request at a time.
    ///
    /// Returns `None` for an exclusively held stage, and when `worker`
    /// was launched without a stage named `stage`.
    fn stage_units(&self, worker: &crate::WorkerGroupId, stage: &str) -> Option<u32> {
        let config = self.worker.info().stage_config(worker, stage)?;
        config
            .distribution
            .as_ref()
            .map(|_| (config.ranks.len() * config.units_per_rank.max(1)) as u32)
    }

    /// Returns what one media call occupies and how much of each.
    ///
    /// Only decoding calls on a unit-measured stage charge media units;
    /// every other placed call holds its stage exclusively. Media
    /// reading, encoding and muxing calls also place tasks on host task
    /// pools.
    pub(crate) fn media_demand(
        &self,
        request: RequestKey,
        media_call: MediaCall,
        units: u32,
    ) -> Demand {
        let stage = self.media_stage(media_call);
        let placed_stage = stage.as_ref().and_then(|stage| {
            self.dispatch
                .affinity
                .get(&(request, stage.clone()))
                .map(|worker| (worker.clone(), stage.clone()))
        });
        let device_units = match media_call {
            // A decode round occupies the stage's media units. Encoding
            // a media unit is admitted on its rank's host task pool, so it
            // overlaps the decode round that follows it rather than
            // excluding it, and the post-processing it begins with rides on
            // the same admission.
            MediaCall::VideoDecoding | MediaCall::AudioDecoding => placed_stage
                .as_ref()
                .and_then(|(worker, name)| self.stage_units(worker, name))
                .map_or(0, |_| units.max(1)),
            _ => 0,
        };
        let cpu_workers = match media_call {
            MediaCall::MediaReading
            | MediaCall::VideoEncoding
            | MediaCall::AudioEncoding
            | MediaCall::Muxing => self.host_task_workers(request, stage.as_deref(), units),
            _ => Vec::new(),
        };
        Demand {
            stage: placed_stage,
            units: device_units,
            cpu_workers,
        }
    }

    /// Returns the host task pools one call of a stage occupies, each
    /// named by the worker holding the stage and the rank within it,
    /// with the tasks the call places there.
    ///
    /// A distributed stage's round runs on as many of its ranks as it has
    /// media units, each rank taking one task per unit it encodes; any other
    /// stage runs one task on each of its ranks.
    ///
    /// The worker is the request's owner of the stage, or the first
    /// loaded worker holding it before the request is placed. Returns no
    /// pools when `stage` is `None` or that worker is not loaded or does
    /// not hold the stage.
    fn host_task_workers(
        &self,
        request: RequestKey,
        stage: Option<&str>,
        units: u32,
    ) -> Vec<(crate::WorkerGroupId, usize, u32)> {
        let Some(stage) = stage else {
            return Vec::new();
        };
        let owner = self.dispatch.affinity.get(&(request, stage.to_owned()));
        let worker = self.worker.info();
        let Some((worker, config)) = worker.workers.iter().find_map(|(id, _)| {
            if owner.is_some_and(|owner| id != owner) {
                return None;
            }
            worker.stage_config(id, stage).map(|config| (id, config))
        }) else {
            return Vec::new();
        };
        let ranks = &config.ranks;
        let per_rank = config.units_per_rank.max(1) as u32;
        let (width, tasks): (usize, Box<dyn Fn(usize) -> u32>) = match config.distribution {
            Some(_) => (
                (units.max(1) as usize)
                    .div_ceil(per_rank as usize)
                    .min(ranks.len()),
                Box::new(move |position: usize| {
                    let start = position as u32 * per_rank;
                    units.max(1).saturating_sub(start).min(per_rank)
                }),
            ),
            None => (ranks.len(), Box::new(|_| 1)),
        };
        ranks[..width]
            .iter()
            .enumerate()
            .map(|(position, rank)| (worker.clone(), *rank, tasks(position)))
            .collect()
    }

    /// Returns how many tasks the host task pool of each rank of one worker
    /// admits at once.
    fn host_task_capacity(&self, worker: &crate::WorkerGroupId) -> u32 {
        self.worker
            .info()
            .workers
            .iter()
            .find(|(id, _)| id == worker)
            .map_or(1, |(_, info)| info.host_task_capacity.max(1))
    }

    /// Returns the host task pools one encoder call carrying inline image
    /// bytes occupies.
    ///
    /// The worker prepares an inline input image (decoding, resizing,
    /// normalization) as one host task on each rank of the stage that
    /// encodes it, and the call holds those tasks until its result returns.
    /// The stage is the one the worker routes `media_call` to, or the
    /// default stage, as `Dispatch::worker_target` resolves it.
    pub(crate) fn image_demand(&self, request: RequestKey, media_call: MediaCall) -> Demand {
        let stage = self
            .media_stage(media_call)
            .unwrap_or_else(|| DEFAULT_BLOCK.to_owned());
        Demand {
            stage: None,
            units: 0,
            cpu_workers: self.host_task_workers(request, Some(&stage), 1),
        }
    }

    /// Accumulates what the media calls in flight occupy, including the host
    /// tasks preparing the inline images of generation encoder calls.
    pub(crate) fn media_occupancy(&self) -> Occupancy {
        let mut occupancy = Occupancy::default();
        for (id, queue) in &self.inflight.pending_calls {
            for inflight in queue {
                let CallKind::Media(media_call) = inflight.call.kind else {
                    continue;
                };
                if inflight.call.input_image.is_some() {
                    occupancy.occupy(
                        &self.image_demand(inflight.call.request_key, media_call),
                        *id,
                    );
                    continue;
                }
                let units = match &inflight.input {
                    InflightInput::Media {
                        decode: Some(range),
                    } => range.max_units,
                    _ => 1,
                };
                occupancy.occupy(
                    &self.media_demand(inflight.call.request_key, media_call, units),
                    *id,
                );
            }
        }
        // A denoiser holds one request's trajectory in its columns from the request's latent
        // preparation to its last step, so the request keeps the denoiser across the gaps between
        // its calls as well: another request's preparation or step in such a gap would open its own
        // trajectory over the first one's. A preempted request keeps it until it leaves, which is
        // after its trajectory was written to the host latent pool.
        for (id, _, state) in self.requests.running_video() {
            if !state.graph.pending(MediaCall::LatentPreparation)
                && state.denoising.completed() < state.denoising.steps()
            {
                occupancy.occupy(
                    &self.media_demand(state.request_key, MediaCall::Denoising, 1),
                    id,
                );
            }
        }
        occupancy
    }

    /// Returns whether the rank group that owns one call can accept a batch.
    ///
    /// Returns `false` when no stage serves `media_call` or
    /// `Dispatch::stage_target` selects no worker. Residency narrows
    /// that choice: a request already placed on the stage is checked only
    /// against its owning worker's queue, and one running request on a candidate
    /// worker for another stage only against that worker's queue.
    pub(crate) fn call_has_queue_capacity(
        &self,
        request: RequestKey,
        media_call: MediaCall,
    ) -> bool {
        self.info
            .media_stages
            .get(&media_call)
            .is_some_and(|stage| {
                self.dispatch
                    .stage_target(
                        &self.worker,
                        request,
                        CallKind::Media(media_call),
                        stage,
                        &dispatch::Choice::Existing,
                    )
                    .is_some()
            })
    }

    /// Returns the media units one scheduled call of this kind would occupy.
    ///
    /// A video decode round occupies its stage's whole capacity
    /// (`stage_units`) unless the track has fewer units left; an audio
    /// decode round always claims the whole capacity. The capacity counts as
    /// one when the stage is not measured in media units or the request
    /// is not yet placed on it.
    pub(crate) fn call_units(&self, state: &VideoState, media_call: MediaCall) -> u32 {
        let width = || {
            self.media_stage(media_call)
                .as_deref()
                .and_then(|name| {
                    let worker = self
                        .dispatch
                        .affinity
                        .get(&(state.request_key, name.to_owned()))?;
                    self.stage_units(worker, name)
                })
                .unwrap_or(1)
        };
        match media_call {
            MediaCall::VideoDecoding => {
                width().min(state.request.sampling.video_units - state.scheduled_decode_units)
            }
            // A visual round covers the condition units its free ranks
            // encode together; the audio tracks are one call.
            MediaCall::LatentEncoding => self.latent_round(state).map_or(1, |(units, _)| units),
            MediaCall::AudioDecoding => width(),
            // An encode round covers the media units the decode round produced.
            MediaCall::VideoEncoding => state
                .decoded_units
                .get(&state.scheduled_encode_units)
                .map_or(1, |(units, _)| *units),
            _ => 1,
        }
    }

    /// Builds one selected media call and records what it schedules.
    ///
    /// Selection checked the request's readiness for `media_call`, so an error
    /// names the readiness invariant that no longer holds. A request's first
    /// call also queues its worker admission onto `admissions`.
    ///
    /// Preparing advances the request's scheduling cursors, selects the worker
    /// (recording the request's residency there), binds every product to its
    /// reserved buffer on that worker, names the stages that read the
    /// call's products from the request's graph, and registers the call as in
    /// flight. The cursors advance before the worker is selected, so an error
    /// can leave the request partially prepared; `assemble_batch` latches
    /// engine-fatal on any error.
    ///
    /// A latent encoding call is a visual round while condition units remain
    /// unscheduled, covering the next units its free ranks encode together,
    /// and otherwise the request's one audio call (`latent_round`).
    pub(crate) fn prepare_media_call(
        &mut self,
        id: RequestId,
        media_call: MediaCall,
        call_id: CallId,
        admissions: &mut Vec<NewRequest>,
    ) -> Result<(Call, RequestDispatch), &'static str> {
        let work = CallKind::Media(media_call);
        let stage = self.info.media_stages[&media_call].clone();
        let state = self.media_state(id).ok_or("a media candidate is running")?;
        let request_key = state.request_key;
        let stateful = work.advances_state();

        // A video request advances one denoising step per call.
        let step = self.scheduled_steps(id, &state.denoising);
        let last_step = media_call == MediaCall::Denoising && state.denoising.ends(step);
        let units = state.graph.condition_units();
        let visual_round = media_call == MediaCall::LatentEncoding
            && state.scheduled_condition_units < units.len() as u32;
        let latent_round = if media_call == MediaCall::LatentEncoding {
            Some(
                self.latent_round(state)
                    .ok_or("a scheduled latent encoding call has a free rank")?,
            )
        } else {
            None
        };
        // Freeze the actual decode/write interval before advancing scheduled
        // counters. Completion consumes this same range from the submission.
        let decode = match media_call {
            MediaCall::VideoDecoding => Some(DecodeRange {
                request_key,
                call_id,
                cursor: state.scheduled_decode_units,
                max_units: self
                    .stage_width(request_key, work, &stage)
                    .ok_or("a scheduled decoder has a loaded owner")?
                    .min(state.request.sampling.video_units - state.scheduled_decode_units),
                ranks: Vec::new(),
            }),
            MediaCall::VideoEncoding => Some(DecodeRange {
                request_key,
                call_id,
                cursor: state.scheduled_encode_units,
                max_units: state.decoded_units[&state.scheduled_encode_units].0,
                ranks: Vec::new(),
            }),
            MediaCall::AudioDecoding => Some(DecodeRange {
                request_key,
                call_id,
                cursor: 0,
                max_units: self
                    .stage_width(request_key, work, &stage)
                    .ok_or("a scheduled decoder has a loaded owner")?,
                ranks: Vec::new(),
            }),
            MediaCall::AudioEncoding => Some(DecodeRange {
                request_key,
                call_id,
                cursor: 0,
                max_units: 1,
                ranks: Vec::new(),
            }),
            // A visual round covers the next condition units on the latent
            // encoder's free ranks; the audio call is one unit on the first.
            MediaCall::LatentEncoding => {
                let (max_units, ranks) =
                    latent_round.ok_or("a latent encoding call has its round")?;
                Some(DecodeRange {
                    request_key,
                    call_id,
                    cursor: if visual_round {
                        state.scheduled_condition_units
                    } else {
                        0
                    },
                    max_units,
                    ranks,
                })
            }
            _ => None,
        };

        // A state-advancing call is predicated on its predecessor's completion
        // product when the predecessor declares one and is still in flight;
        // otherwise the call carries no predicate.
        let predicate = if stateful {
            self.inflight
                .pending_calls
                .get(&id)
                .and_then(|calls| {
                    calls
                        .iter()
                        .find(|call| call.call.call_id == state.predecessor)
                })
                .and_then(|call| call.call.completion_output.as_ref())
                .cloned()
        } else {
            None
        };

        let media_input = |name: &str| {
            state
                .condition_media
                .get(name)
                .cloned()
                .ok_or("the media reader has declared the condition input")
        };
        let inputs = match media_call {
            MediaCall::VisionEncoding => vec![media_input(product::VISION_PIXELS)?],
            MediaCall::TextEncoding => state.vision_features.iter().cloned().collect(),
            MediaCall::LatentEncoding if visual_round => {
                vec![media_input(product::CONDITION_PIXELS)?]
            }
            MediaCall::LatentEncoding => vec![media_input(product::CONDITION_SAMPLES)?],
            // The text features, then the visual condition rows in unit
            // order, then the audio condition rows.
            MediaCall::LatentPreparation => std::iter::once(
                state
                    .conditioning
                    .clone()
                    .ok_or("text encoding has declared its conditioning output")?,
            )
            .chain(state.condition_video_latents.iter().cloned())
            .chain(state.condition_audio_latents.iter().cloned())
            .collect(),
            MediaCall::VideoDecoding => vec![state.latents[0].clone()],
            MediaCall::AudioDecoding => vec![state.latents[1].clone()],
            MediaCall::VideoEncoding => {
                let range = decode.as_ref().ok_or("a video write has an input range")?;
                vec![state.decoded_units[&range.cursor].1.clone()]
            }
            MediaCall::AudioEncoding => {
                vec![
                    state
                        .audio
                        .as_ref()
                        .ok_or("audio is ready to encode")?
                        .clone(),
                ]
            }
            // The muxer takes the completed encode rounds that follow the
            // last it was handed, in media unit order; the final call
            // carries none.
            MediaCall::Muxing => Self::ready_encode_rounds(state)
                .into_iter()
                .map(|cursor| state.encoded_units[&cursor].1.clone())
                .collect(),
            _ => Vec::new(),
        };

        // Declare the call's products against the request's reservations,
        // by the names their stage declares them under. A slice holds
        // the product's leading extent and its first unit within the
        // reservation.
        let mut declared: Vec<(&'static str, Option<(u32, u32)>)> = Vec::new();
        match media_call {
            MediaCall::MediaReading => {
                for name in [
                    product::CONDITION_PIXELS,
                    product::CONDITION_SAMPLES,
                    product::VISION_PIXELS,
                ] {
                    if state.graph.product(name).is_some() {
                        declared.push((name, None));
                    }
                }
            }
            MediaCall::VisionEncoding => declared.push((product::VISION_FEATURES, None)),
            MediaCall::TextEncoding => declared.push((product::CONDITIONING, None)),
            // A visual round writes the rows of its own units, which follow
            // the rows of every earlier unit.
            MediaCall::LatentEncoding if visual_round => {
                let range = decode.as_ref().ok_or("a latent round has a unit range")?;
                let cursor = range.cursor as usize;
                let first: u32 = units[..cursor].iter().sum();
                let rows: u32 = units[cursor..cursor + range.max_units as usize]
                    .iter()
                    .sum();
                declared.push((product::CONDITION_VIDEO_LATENTS, Some((rows, first))));
            }
            MediaCall::LatentEncoding => declared.push((product::CONDITION_AUDIO_LATENTS, None)),
            MediaCall::Denoising if last_step => {
                declared.push((product::VIDEO_LATENTS, None));
                declared.push((product::AUDIO_LATENTS, None));
            }
            // A media unit round writes its own slice of the track's
            // reservation, whether the slice holds decoded media units or
            // the rows encoded from them.
            MediaCall::VideoDecoding | MediaCall::VideoEncoding => {
                let range = decode.as_ref().ok_or("a media round has a unit range")?;
                let name = if media_call == MediaCall::VideoDecoding {
                    product::VIDEO_UNITS
                } else {
                    product::ENCODED_UNITS
                };
                declared.push((name, Some((range.max_units, range.cursor))));
            }
            MediaCall::AudioDecoding => declared.push((product::AUDIO_SAMPLES, None)),
            _ => {}
        }
        let mut outputs = Vec::new();
        let mut output_names = Vec::new();
        let mut output_starts = Vec::new();
        for (name, slice) in declared {
            let reservation = state
                .reservations
                .get(name)
                .ok_or("every product of the graph is reserved")?;
            let reserved = &state.allocations.tensors[reservation];
            let mut shape_bound = reserved.shape_bound.clone();
            let start = match slice {
                Some((extent, start)) => {
                    shape_bound.dims[0] = DimBound::Static(extent);
                    start
                }
                None => 0,
            };
            outputs.push(TensorRef {
                request_key,
                producer_call_id: call_id,
                output_index: reservation.1 as u16,
                generation: 1,
                dtype: reserved.dtype,
                shape_bound,
            });
            output_names.push(name);
            output_starts.push((reservation.clone(), start));
        }

        // A resumed request's preparation opens the stored trajectory its
        // restore pinned.
        let latent_restore = match &state.host_trajectory.restoring {
            Restoring::Ready(restore) if media_call == MediaCall::LatentPreparation => {
                restore.clone()
            }
            _ => None,
        };
        // An encoder call's output in the host encoder pool: the output, the
        // rows of it the call's output holds, and whether no other call of
        // it follows. A visual latent round holds its own units' rows.
        let encoded = |name: &'static str| {
            state
                .graph
                .product(name)
                .and_then(|product| product.extent)
                .map(|rows| (name, [0, rows], true))
        };
        let host_output = match media_call {
            MediaCall::VisionEncoding => encoded(product::VISION_FEATURES),
            MediaCall::TextEncoding => encoded(product::CONDITIONING),
            MediaCall::LatentEncoding if visual_round => decode.as_ref().map(|range| {
                let cursor = range.cursor as usize;
                let end = cursor + range.max_units as usize;
                let first: u32 = units[..cursor].iter().sum();
                let rows: u32 = units[cursor..end].iter().sum();
                (
                    product::CONDITION_VIDEO_LATENTS,
                    [first, first + rows],
                    end == units.len(),
                )
            }),
            MediaCall::LatentEncoding => encoded(product::CONDITION_AUDIO_LATENTS),
            _ => None,
        };

        let completion_output = if outputs.is_empty() && stateful {
            Some(self.media_completion_product(request_key, call_id)?)
        } else {
            None
        };

        let mut call = Call::new(request_key, call_id, work);
        call.stage = stage.to_owned();
        call.inputs = inputs;
        call.outputs = outputs;
        call.completion_output = completion_output;
        call.predicate = predicate;
        call.latent_restore = latent_restore;
        call.host_output = match (host_output, self.encoder_outputs.as_mut()) {
            (Some((name, rows, last)), Some(store)) => {
                store.prepare_call(request_key, video_output(name), call_id, rows, last)
            }
            _ => None,
        };

        // Record what this call schedules: the admission it carries, the
        // reservation slice each product binds to, and the cursors that
        // `ready_calls` reads.
        let state = self
            .media_state_mut(id)
            .ok_or("a media candidate is running")?;
        if state.admission_state == WorkerRegistration::Unsubmitted {
            admissions.push(
                state
                    .admission
                    .clone()
                    .ok_or("an admitted video request states its admission")?,
            );
            state.admission_state = WorkerRegistration::InFlight;
        }
        for (product, ((stage, index), start)) in call.outputs.iter().zip(output_starts) {
            let replaced = state
                .buffer_bindings
                .insert(product.buffer_id(), (stage, index, start));
            debug_assert!(replaced.is_none(), "media buffer identity was reused");
        }
        match media_call {
            MediaCall::MediaReading => {
                for (name, product) in output_names.iter().zip(&call.outputs) {
                    state.condition_media.insert(name, product.clone());
                }
            }
            MediaCall::VisionEncoding => {
                state.vision_features = call.outputs.first().cloned();
            }
            MediaCall::TextEncoding => {
                state.conditioning = call.outputs.first().cloned();
            }
            MediaCall::LatentEncoding => {
                let product = call
                    .outputs
                    .first()
                    .cloned()
                    .ok_or("latent encoding declares its rows")?;
                if visual_round {
                    let range = decode.as_ref().ok_or("a latent round has a unit range")?;
                    state.scheduled_condition_units += range.max_units;
                    state.condition_video_latents.push(product);
                } else {
                    state.condition_audio_latents = Some(product);
                }
            }
            MediaCall::LatentPreparation => {
                state.host_trajectory.restoring = Restoring::Prepared;
            }
            // Submitted steps are derived from the pending denoising calls.
            MediaCall::Denoising => {}
            MediaCall::VideoDecoding => {
                let range = decode.as_ref().ok_or("a video decode has an input range")?;
                state.scheduled_decode_units += range.max_units;
                state
                    .decoded_units
                    .insert(range.cursor, (range.max_units, call.outputs[0].clone()));
            }
            MediaCall::AudioDecoding => {
                state.audio = Some(call.outputs[0].clone());
            }
            MediaCall::VideoEncoding => {
                let range = decode.as_ref().ok_or("a video write has an input range")?;
                state.scheduled_encode_units += range.max_units;
                state
                    .encoded_units
                    .insert(range.cursor, (range.max_units, call.outputs[0].clone()));
            }
            MediaCall::AudioEncoding => {}
            MediaCall::Muxing => {
                // The final round carries no media units: it leaves
                // `muxing_inputs` empty, which its completion reads.
                let rounds = Self::ready_encode_rounds(state);
                for cursor in rounds {
                    let units = state
                        .encoded_ready
                        .remove(&cursor)
                        .ok_or("a ready encode round was counted")?;
                    let (_, product) = state
                        .encoded_units
                        .remove(&cursor)
                        .ok_or("a ready encode round has its product")?;
                    state.handed_units += units;
                    state.muxing_inputs.push(product);
                }
            }
        }
        state.graph.schedule(media_call);
        if stateful {
            state.predecessor = call_id;
        }
        if last_step {
            state.latents = call.outputs.clone();
        }

        // Place the call, recording the request's residency on the selected
        // worker, then bind each distinct input and output buffer to its
        // reserved span in the buffer pool of each rank that backs it.
        let (worker, selected_stage) = self
            .dispatch
            .select_worker(&self.worker, &self.info, &call, &dispatch::Choice::Existing)
            .ok_or("a prepared call retains an executable stage")?;
        call.stage = selected_stage;
        let state = self
            .media_state(id)
            .ok_or("a selected media request is running")?;
        let mut bound = HashSet::new();
        let buffers = call
            .buffer_inputs()
            .chain(call.buffer_outputs())
            .filter(|product| bound.insert(product.buffer_id()))
            .flat_map(|product| {
                let (stage, index, start) = &state.buffer_bindings[&product.buffer_id()];
                state.allocations.tensors[&(stage.clone(), *index)].bind(product, *start, &worker)
            })
            .collect();

        // Request slots are allocated per worker; the denoiser derives the
        // latent pages of its slot from its own pool.
        let request_pool_idx = state.allocations.request_slot(&worker);

        // Admission routed every stage of the request, so the replicas
        // reading this call's products are known here: the producer names
        // the ranks of the stages its request's graph reads them with,
        // on the replicas the request was routed to.
        let readers = state
            .graph
            .consumers(media_call)
            .iter()
            .filter_map(|consumer| {
                let stage = self.info.media_stages.get(consumer)?;
                let routed = self.dispatch.affinity.get(&(request_key, stage.clone()))?;
                Some((stage.clone(), routed.clone()))
            })
            .collect();

        let dispatch = RequestDispatch {
            worker,
            request_pool_idx: Some(request_pool_idx),
            decode,
            buffers,
            readers: Some(readers),
        };
        self.register_inflight(
            call.clone(),
            InflightInput::Media {
                decode: dispatch.decode.clone(),
            },
            0,
        );
        Ok((call, dispatch))
    }
}
