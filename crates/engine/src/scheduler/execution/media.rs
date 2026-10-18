//! Video media scheduling: a video request runs the calls of its graph
//! (`graph::VideoGraph`, built at admission from its task and conditions),
//! offered by `ready_calls` as their inputs become available. Calls occupy
//! device lanes and, for media reading, encoding and muxing, host lanes,
//! tracked by the lane ledger ([`LaneLedger`]); `plan_media_call` builds one
//! call with its products, buffer bindings, latent pages and reading
//! components, and `prepare_media_batches` forms the media batches a refill
//! dispatches.

use super::*;
use uniserve_ipc::MediaCall;

/// The lanes one media call occupies while it is in flight.
///
/// A component lane measured in media units carries `units`; a demand with
/// zero `units` holds its component lane exclusively, one request at a time.
/// `host_ranks` names the host lanes the call places tasks on.
pub(in crate::scheduler) struct LaneDemand {
    /// The worker and component whose device lane the call occupies, taken
    /// from the request's placement (`Placement::affinity`). Admission places
    /// a media request on a worker for every media component, so a request's
    /// first call is checked against its lane too. `None` when no component
    /// serves the call or the request holds no placement for it.
    component: Option<(crate::WorkerId, String)>,
    units: u32,
    /// Host lanes this call occupies, each named by its worker and rank with
    /// the number of tasks the call places on it: one per media unit the
    /// rank encodes, one for any other host call.
    host_ranks: Vec<(crate::WorkerId, usize, u32)>,
}

/// Lane occupancy across the media calls in flight.
///
/// `lane_occupancy` rebuilds it from the in-flight calls at the start of each
/// media pass, and at the first inline-image encoder call of a generation
/// pass; the pass charges each call it selects.
#[derive(Default)]
pub(in crate::scheduler) struct LaneLedger {
    /// The request holding each exclusive component lane.
    exclusive: HashMap<(crate::WorkerId, String), RequestId>,
    /// Media units in flight on each unit-measured component lane.
    units: HashMap<(crate::WorkerId, String), u32>,
    /// A host rank owns one host lane, so occupancy is counted per rank of
    /// each worker.
    host: HashMap<(crate::WorkerId, usize), u32>,
}

impl LaneLedger {
    /// Returns whether one request's call fits the lanes it occupies.
    ///
    /// Every host rank must have room for the call's tasks. An exclusive
    /// component lane admits further calls of the request already holding it.
    pub(in crate::scheduler) fn admits(
        &self,
        demand: &LaneDemand,
        request: RequestId,
        scheduler: &Scheduler,
    ) -> bool {
        if demand.host_ranks.iter().any(|(worker, rank, tasks)| {
            let capacity = scheduler.host_lane_capacity(worker);
            let key = (worker.clone(), *rank);
            self.host.get(&key).copied().unwrap_or(0) + tasks > capacity
        }) {
            return false;
        }
        let Some((worker, component)) = demand.component.as_ref() else {
            return true;
        };
        if demand.units > 0 {
            let capacity = scheduler.device_lane_units(worker, component).unwrap_or(1);
            self.units
                .get(&(worker.clone(), component.clone()))
                .copied()
                .unwrap_or(0)
                + demand.units
                <= capacity
        } else {
            self.exclusive
                .get(&(worker.clone(), component.clone()))
                .is_none_or(|holder| *holder == request)
        }
    }

    /// Marks the lanes one request's call occupies until it completes.
    pub(in crate::scheduler) fn occupy(&mut self, demand: &LaneDemand, request: RequestId) {
        for (worker, rank, tasks) in &demand.host_ranks {
            *self.host.entry((worker.clone(), *rank)).or_default() += tasks;
        }
        let Some((worker, component)) = demand.component.as_ref() else {
            return;
        };
        let key = (worker.clone(), component.clone());
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
    fn ready_encode_rounds(state: &MediaFlowState) -> Vec<u32> {
        let mut rounds = Vec::new();
        let mut cursor = state.handed_units;
        while let Some(units) = state.encoded_ready.get(&cursor) {
            rounds.push(cursor);
            cursor += units;
        }
        rounds
    }

    /// Lists every call of one request that can be scheduled now.
    ///
    /// A request is a graph of calls with data dependencies, so a tick offers
    /// all of them and the lane ledger decides which ones fit. At most one
    /// call of each kind is ready at once: the cursors this returns against
    /// advance as calls are scheduled, so a unit group only becomes ready
    /// once its predecessor is submitted.
    ///
    /// A conditioned request first reads its media on a host rank. The calls
    /// reading those host products wait until the read has completed, so a
    /// model worker's queue never holds a call blocked on a host task. Then
    /// the vision encoding precedes the text encoding that splices its
    /// features in, on the same component, while the latent encoding covers
    /// the visual condition units in rounds and then the audio tracks.
    /// Latent preparation follows once every one of them is scheduled.
    ///
    /// Latent preparation, the denoising steps and the two decoders become
    /// ready once their predecessor is scheduled; they name their inputs by
    /// product reference and do not wait here for the producer to complete.
    /// Video and audio encoding wait until the call producing their input has
    /// completed, and muxing waits for completed encode rounds.
    fn ready_calls(&self, state: &MediaFlowState) -> Vec<MediaCall> {
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
            if !state.media_reading_scheduled {
                return vec![MediaCall::MediaReading];
            }
            if !state.condition_media.values().all(produced) {
                return Vec::new();
            }
        }
        let mut ready = Vec::new();
        if graph.encodes_vision() && !state.vision_encoding_scheduled {
            ready.push(MediaCall::VisionEncoding);
        } else if !state.text_encoding_scheduled {
            ready.push(MediaCall::TextEncoding);
        }
        let visual_units = graph.condition_units().len() as u32;
        if state.scheduled_condition_units < visual_units
            || (graph.encodes_condition_audio() && state.condition_audio_latents.is_none())
        {
            ready.push(MediaCall::LatentEncoding);
        }
        if !ready.is_empty() {
            return ready;
        }
        if !state.latent_preparation_scheduled {
            return vec![MediaCall::LatentPreparation];
        }
        if self.scheduled_steps(state.request.request_id, &state.denoising)
            < state.denoising.steps()
        {
            return vec![MediaCall::Denoising];
        }

        // Consume completed decoder outputs first. Audio and video retain
        // independent readiness and capacity when the other branch is busy.
        if let Some((_, product)) = state.decoded_units.get(&state.scheduled_encode_units)
            && produced(product)
        {
            ready.push(MediaCall::VideoEncoding);
        }
        if !state.audio_encoding_scheduled && state.audio.as_ref().is_some_and(produced) {
            ready.push(MediaCall::AudioEncoding);
        }
        if state.scheduled_decode_units < sampling.video_units {
            ready.push(MediaCall::VideoDecoding);
        }
        if !state.audio_decoding_scheduled {
            ready.push(MediaCall::AudioDecoding);
        }
        // The muxer consumes media units as they arrive: a muxing call takes
        // the encode rounds that have completed since the last one, in
        // order, and a final call carrying no units assembles the artifact
        // once every unit and the audio track are in. It waits for completed
        // writes, never just their scheduled counts.
        if !state.muxing_in_flight && !state.muxed {
            let next_round_ready = state.encoded_ready.contains_key(&state.handed_units);
            let everything_in = state.audio_encoded
                && state.handed_units == sampling.video_units
                && !state.final_muxing_scheduled;
            if next_round_ready || everything_in {
                ready.push(MediaCall::Muxing);
            }
        }
        ready
    }

    /// Returns the steps of a request's trajectory covered once its submitted
    /// calls complete.
    pub(in crate::scheduler) fn scheduled_steps(&self, id: RequestId, progress: &Denoising) -> u32 {
        progress.scheduled(
            self.inflight
                .pending_calls
                .get(&id)
                .into_iter()
                .flatten()
                .filter_map(|pending| match pending.call.kind {
                    CallKind::Media(media_call) => Some(media_call),
                    _ => None,
                }),
        )
    }

    /// Returns the component whose lanes one media call occupies.
    fn media_component(&self, media_call: MediaCall) -> Option<String> {
        self.info.media_components.get(&media_call).cloned()
    }

    /// Returns how many media units one call of this kind covers at once.
    ///
    /// A distributed component reconstructs `units_per_rank` media units on
    /// each of its ranks per round; any other component counts one unit per
    /// rank. The worker is the request's owner of `component` when it has
    /// one, otherwise the first candidate that can execute `work`.
    ///
    /// Returns `None` when no such worker is loaded, when the worker was
    /// launched without its bound component, or when the width does not fit
    /// in `u32`.
    fn component_width(&self, request: RequestKey, work: CallKind, component: &str) -> Option<u32> {
        let owner = self
            .placement
            .affinity
            .get(&(request, component.to_owned()));
        let (worker, bound, _) = self
            .placement
            .component_candidates(self.executor.as_ref(), work, component)
            .find(|(worker, _, _)| owner.is_none_or(|owner| *worker == owner))?;
        let config = self.executor.info().component_config(worker, bound)?;
        let units_per_rank = if config.distribution.is_some() {
            config.units_per_rank.max(1)
        } else {
            1
        };
        u32::try_from(config.ranks.len().saturating_mul(units_per_rank)).ok()
    }

    /// Returns a component's device lane capacity in media units.
    ///
    /// A component that distributes its work measures its lane in the units its
    /// ranks reconstruct together; every other component's lane is exclusive
    /// and admits one request at a time.
    ///
    /// Returns `None` for an exclusive lane, and when `worker` was launched
    /// without a component named `component`.
    fn device_lane_units(&self, worker: &crate::WorkerId, component: &str) -> Option<u32> {
        let config = self.executor.info().component_config(worker, component)?;
        config
            .distribution
            .as_ref()
            .map(|_| (config.ranks.len() * config.units_per_rank.max(1)) as u32)
    }

    /// Returns the lanes one media call occupies and how much of each.
    ///
    /// Only decoding calls on a unit-measured lane charge media units; every
    /// other placed call holds its component lane exclusively. Media
    /// reading, encoding and muxing calls also place tasks on host lanes.
    fn lane_demand(&self, request: RequestKey, media_call: MediaCall, units: u32) -> LaneDemand {
        let component = self.media_component(media_call);
        let placed_component = component.as_ref().and_then(|component| {
            self.placement
                .affinity
                .get(&(request, component.clone()))
                .map(|worker| (worker.clone(), component.clone()))
        });
        let device_units = match media_call {
            // A device lane is occupied by a decode round. Encoding a media
            // unit is admitted on its rank's host lane, so it overlaps the
            // decode round that follows it rather than excluding it, and the
            // post-processing it begins with rides on the same admission.
            MediaCall::VideoDecoding | MediaCall::AudioDecoding => placed_component
                .as_ref()
                .and_then(|(worker, name)| self.device_lane_units(worker, name))
                .map_or(0, |_| units.max(1)),
            _ => 0,
        };
        let host_ranks = match media_call {
            MediaCall::MediaReading
            | MediaCall::VideoEncoding
            | MediaCall::AudioEncoding
            | MediaCall::Muxing => self.host_lane_ranks(request, component.as_deref(), units),
            _ => Vec::new(),
        };
        LaneDemand {
            component: placed_component,
            units: device_units,
            host_ranks,
        }
    }

    /// Returns the host lanes one call of a component occupies, each named by
    /// the worker holding the component and the rank within it, with the
    /// tasks the call places there.
    ///
    /// A distributed component's round runs on as many of its ranks as it has
    /// media units, each rank taking one task per unit it encodes; any other
    /// component runs one task on each of its ranks.
    ///
    /// The worker is the request's owner of the component, or the first
    /// loaded worker holding it before the request is placed. Returns no
    /// lanes when `component` is `None` or that worker is not loaded or does
    /// not hold the component.
    fn host_lane_ranks(
        &self,
        request: RequestKey,
        component: Option<&str>,
        units: u32,
    ) -> Vec<(crate::WorkerId, usize, u32)> {
        let Some(component) = component else {
            return Vec::new();
        };
        let owner = self
            .placement
            .affinity
            .get(&(request, component.to_owned()));
        let executor = self.executor.info();
        let Some((worker, config)) = executor.workers.iter().find_map(|(id, _)| {
            if owner.is_some_and(|owner| id != owner) {
                return None;
            }
            executor
                .component_config(id, component)
                .map(|config| (id, config))
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

    /// Returns the host lane capacity of one worker's ranks.
    fn host_lane_capacity(&self, worker: &crate::WorkerId) -> u32 {
        self.executor
            .info()
            .workers
            .iter()
            .find(|(id, _)| id == worker)
            .map_or(1, |(_, info)| info.host_lane_capacity.max(1))
    }

    /// Returns the host lanes one encoder call carrying inline image bytes
    /// occupies.
    ///
    /// The worker prepares an inline input image (decoding, resizing,
    /// normalization) as one host task on each rank of the component that
    /// encodes it, and the call holds those tasks until its result returns.
    /// The component is the one the worker routes `media_call` to, or the
    /// default component, as `Placement::worker_target` resolves it.
    pub(in crate::scheduler) fn image_lane_demand(
        &self,
        request: RequestKey,
        media_call: MediaCall,
    ) -> LaneDemand {
        let component = self
            .media_component(media_call)
            .unwrap_or_else(|| DEFAULT_COMPONENT.to_owned());
        LaneDemand {
            component: None,
            units: 0,
            host_ranks: self.host_lane_ranks(request, Some(&component), 1),
        }
    }

    /// Accumulates the lanes the media calls in flight occupy, including the
    /// host tasks preparing the inline images of generation encoder calls.
    pub(in crate::scheduler) fn lane_occupancy(&self) -> LaneLedger {
        let mut ledger = LaneLedger::default();
        for (id, queue) in &self.inflight.pending_calls {
            for inflight in queue {
                let CallKind::Media(media_call) = inflight.call.kind else {
                    continue;
                };
                if inflight.call.input_image.is_some() {
                    ledger.occupy(
                        &self.image_lane_demand(inflight.call.request_key, media_call),
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
                ledger.occupy(
                    &self.lane_demand(inflight.call.request_key, media_call, units),
                    *id,
                );
            }
        }
        ledger
    }

    /// Returns whether the rank group that owns one call can accept a batch.
    ///
    /// Returns `false` when no component serves `media_call` or
    /// `Placement::component_target` selects no worker. Residency narrows
    /// that choice: a request already placed on the component is checked only
    /// against its owning worker's queue, and one resident on a candidate
    /// worker for another component only against that worker's queue.
    fn call_has_queue_capacity(&self, request: RequestKey, media_call: MediaCall) -> bool {
        self.info
            .media_components
            .get(&media_call)
            .is_some_and(|component| {
                self.placement
                    .component_target(
                        self.executor.as_ref(),
                        request,
                        CallKind::Media(media_call),
                        component,
                    )
                    .is_some()
            })
    }

    /// Returns the media units one scheduled call of this kind would occupy.
    ///
    /// A video decode round occupies its component's whole lane
    /// (`device_lane_units`) unless the track has fewer units left; an audio
    /// decode round always claims the whole lane. The lane width counts as
    /// one when the component has no unit-measured lane or the request is not
    /// yet placed on it.
    fn call_units(&self, state: &MediaFlowState, media_call: MediaCall) -> u32 {
        let width = || {
            self.media_component(media_call)
                .as_deref()
                .and_then(|name| {
                    let worker = self
                        .placement
                        .affinity
                        .get(&(state.admission.request_key, name.to_owned()))?;
                    self.device_lane_units(worker, name)
                })
                .unwrap_or(1)
        };
        match media_call {
            MediaCall::VideoDecoding => {
                width().min(state.request.sampling.video_units - state.scheduled_decode_units)
            }
            // A visual round covers the condition units its component's
            // ranks encode together; the audio tracks are one call.
            MediaCall::LatentEncoding => {
                let remaining =
                    state.graph.condition_units().len() as u32 - state.scheduled_condition_units;
                if remaining > 0 {
                    width().min(remaining)
                } else {
                    1
                }
            }
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
    /// Planning advances the request's scheduling cursors, selects the worker
    /// (recording the request's residency there), binds every product to its
    /// reserved buffer on that worker, names the components that read the
    /// call's products from the request's graph, and registers the call as in
    /// flight. The cursors advance before the worker is selected, so an error
    /// can leave the request partially planned; `prepare_media_batches`
    /// latches engine-fatal on any error.
    ///
    /// A latent encoding call is a visual round while condition units remain
    /// unscheduled, covering the next units its component's ranks encode
    /// together, and otherwise the request's one audio call.
    fn plan_media_call(
        &mut self,
        id: RequestId,
        media_call: MediaCall,
        call_id: CallId,
        admissions: &mut Vec<NewRequest>,
    ) -> Result<(Call, RequestPlacement), &'static str> {
        let work = CallKind::Media(media_call);
        let component = self.info.media_components[&media_call].clone();
        let state = self.media_state(id).ok_or("a media candidate is running")?;
        let request_key = state.admission.request_key;
        let stateful = work.advances_state();

        // A video request advances its ladder one step per call.
        let step = self.scheduled_steps(id, &state.denoising);
        let last_step = media_call == MediaCall::Denoising && state.denoising.ends(step);
        let units = state.graph.condition_units();
        let visual_round = media_call == MediaCall::LatentEncoding
            && state.scheduled_condition_units < units.len() as u32;
        // Freeze the actual decode/write interval before advancing scheduled
        // counters. Completion consumes this same range from the submission.
        let decode = match media_call {
            MediaCall::VideoDecoding => Some(DecodeRange {
                request_key,
                call_id,
                cursor: state.scheduled_decode_units,
                max_units: self
                    .component_width(request_key, work, &component)
                    .ok_or("a scheduled decoder has a loaded owner")?
                    .min(state.request.sampling.video_units - state.scheduled_decode_units),
            }),
            MediaCall::VideoEncoding => Some(DecodeRange {
                request_key,
                call_id,
                cursor: state.scheduled_encode_units,
                max_units: state.decoded_units[&state.scheduled_encode_units].0,
            }),
            MediaCall::AudioDecoding => Some(DecodeRange {
                request_key,
                call_id,
                cursor: 0,
                max_units: self
                    .component_width(request_key, work, &component)
                    .ok_or("a scheduled decoder has a loaded owner")?,
            }),
            MediaCall::AudioEncoding => Some(DecodeRange {
                request_key,
                call_id,
                cursor: 0,
                max_units: 1,
            }),
            // A visual round covers the next condition units the latent
            // encoder's ranks encode together; the audio call is one unit.
            MediaCall::LatentEncoding if visual_round => Some(DecodeRange {
                request_key,
                call_id,
                cursor: state.scheduled_condition_units,
                max_units: self
                    .component_width(request_key, work, &component)
                    .ok_or("a scheduled latent encoder has a loaded owner")?
                    .min(units.len() as u32 - state.scheduled_condition_units),
            }),
            MediaCall::LatentEncoding => Some(DecodeRange {
                request_key,
                call_id,
                cursor: 0,
                max_units: 1,
            }),
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
        // by the names their component declares them under. A slice holds
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

        let completion_output = if outputs.is_empty() && stateful {
            Some(self.media_completion_product(request_key, call_id)?)
        } else {
            None
        };

        let mut call = Call::new(request_key, call_id, work);
        call.component = component.to_owned();
        call.inputs = inputs;
        call.outputs = outputs;
        call.completion_output = completion_output;
        call.predicate = predicate;

        // Record what this call schedules: the admission it carries, the
        // reservation slice each product binds to, and the cursors that
        // `ready_calls` reads.
        let state = self
            .media_state_mut(id)
            .ok_or("a media candidate is running")?;
        if state.admission_state == WorkerRegistration::Unsubmitted {
            admissions.push(state.admission.clone());
            state.admission_state = WorkerRegistration::InFlight;
        }
        for (product, ((component, index), start)) in call.outputs.iter().zip(output_starts) {
            let replaced = state
                .buffer_bindings
                .insert(product.buffer_id(), (component, index, start));
            debug_assert!(replaced.is_none(), "media buffer identity was reused");
        }
        match media_call {
            MediaCall::MediaReading => {
                state.media_reading_scheduled = true;
                for (name, product) in output_names.iter().zip(&call.outputs) {
                    state.condition_media.insert(name, product.clone());
                }
            }
            MediaCall::VisionEncoding => {
                state.vision_features = call.outputs.first().cloned();
                state.vision_encoding_scheduled = true;
            }
            MediaCall::TextEncoding => {
                state.conditioning = call.outputs.first().cloned();
                state.text_encoding_scheduled = true;
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
            MediaCall::LatentPreparation => state.latent_preparation_scheduled = true,
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
                state.audio_decoding_scheduled = true;
                state.audio = Some(call.outputs[0].clone());
            }
            MediaCall::VideoEncoding => {
                let range = decode.as_ref().ok_or("a video write has an input range")?;
                state.scheduled_encode_units += range.max_units;
                state
                    .encoded_units
                    .insert(range.cursor, (range.max_units, call.outputs[0].clone()));
            }
            MediaCall::AudioEncoding => state.audio_encoding_scheduled = true,
            MediaCall::Muxing => {
                let rounds = Self::ready_encode_rounds(state);
                if rounds.is_empty() {
                    state.final_muxing_scheduled = true;
                }
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
                state.muxing_in_flight = true;
            }
            MediaCall::ImageDecoding => {
                unreachable!("a video request's graph has no image decoding")
            }
        }
        if stateful {
            state.predecessor = call_id;
        }
        if last_step {
            state.latents = call.outputs.clone();
        }

        // Place the call, recording the request's residency on the selected
        // worker, then bind each distinct input and output buffer to its
        // reserved span in the buffer pool of each rank that backs it.
        let (worker, selected_component) = self
            .placement
            .select_worker(self.executor.as_ref(), &self.info, &call)
            .ok_or("a planned call retains an executable component")?;
        call.component = selected_component;
        let state = self
            .media_state(id)
            .ok_or("a selected media request is running")?;
        let mut bound = HashSet::new();
        let buffers = call
            .buffer_inputs()
            .chain(call.buffer_outputs())
            .filter(|product| bound.insert(product.buffer_id()))
            .flat_map(|product| {
                let (component, index, start) = &state.buffer_bindings[&product.buffer_id()];
                state.allocations.tensors[&(component.clone(), *index)]
                    .bind(product, *start, &worker)
            })
            .collect();

        // Request slots are allocated per worker; the denoiser derives the
        // latent pages of its slot from its own pool.
        let request_pool_idx = state.allocations.request_slot(&worker);

        // Admission routed every component of the request, so the replicas
        // reading this call's products are known here: the producer names
        // the ranks of the components its request's graph reads them with,
        // on the replicas the request was routed to.
        let readers = state
            .graph
            .consumers(media_call)
            .iter()
            .filter_map(|consumer| {
                let component = self.info.media_components.get(consumer)?;
                let routed = self
                    .placement
                    .affinity
                    .get(&(request_key, component.clone()))?;
                Some((component.clone(), routed.clone()))
            })
            .collect();

        let placement = RequestPlacement {
            worker,
            request_pool_idx: Some(request_pool_idx),
            block_tables: Vec::new(),
            new_cache_units: Vec::new(),
            forward: uniserve_ipc::ForwardBatch::default(),
            latent: None,
            decode,
            buffers,
            readers: Some(readers),
        };
        self.register_inflight(
            call.clone(),
            InflightInput::Media {
                decode: placement.decode.clone(),
            },
            0,
        );
        Ok((call, placement))
    }

    /// Selects the ready calls of running video requests and builds one
    /// media pass: one batch per selected call kind, followed by a
    /// command-only batch for any lifecycle commands the pass takes.
    ///
    /// Independent audio and video branches are linked by tensor inputs, not
    /// by a state predecessor. Returns no batches when nothing fits, and when
    /// planning finds a broken invariant, which latches engine-fatal.
    pub(in crate::scheduler) fn prepare_media_batches(&mut self) -> Vec<ExecutionBatch> {
        // Every resident request, in `running_order`, offers each call
        // `ready_calls` lists, and a call is selected when the lanes it
        // occupies have free capacity and its worker's queue can accept a
        // batch. No per-request cap applies beyond the lanes and rank queues.
        let mut ledger = self.lane_occupancy();
        let mut candidates = Vec::new();
        for id in self.running_order.clone() {
            let Some(state) = self.media_state(id) else {
                continue;
            };
            // A request offers nothing further while the call carrying its
            // worker admission is in flight; `process_diffusion_result` marks
            // it registered on that call's valid result.
            if state.terminal_intent.is_terminal()
                || state.admission_state == WorkerRegistration::InFlight
            {
                continue;
            }
            for media_call in self.ready_calls(state) {
                let demand = self.lane_demand(
                    state.admission.request_key,
                    media_call,
                    self.call_units(state, media_call),
                );
                if !ledger.admits(&demand, id, self)
                    || !self.call_has_queue_capacity(state.admission.request_key, media_call)
                {
                    continue;
                }
                ledger.occupy(&demand, id);
                candidates.push((id, media_call));
            }
        }
        if candidates.is_empty() {
            return Vec::new();
        }

        let submit_at = Instant::now();
        // The pass also takes the queued commands of its candidate requests
        // and every queued `Finish`, except those `take_commands` defers.
        let candidate_requests = candidates.iter().map(|(id, _)| *id).collect::<HashSet<_>>();
        let commands = self.take_commands(|command| {
            candidate_requests.contains(&command.request_key().request_id)
                || matches!(command, BatchCommand::Finish { .. })
        });

        // One batch per media call: a batch is one numerical call on one component,
        // so a rank receives it as a single homogeneous group. Batches keep the
        // order in which their first call was selected.
        let mut call_order = Vec::new();
        let mut call_batches: HashMap<MediaCall, (u64, Vec<(Call, RequestPlacement)>)> =
            HashMap::new();
        for (_, media_call) in &candidates {
            if !call_batches.contains_key(media_call) {
                call_batches.insert(*media_call, (self.inflight.next_batch_id(), Vec::new()));
                call_order.push(*media_call);
            }
        }

        // A call's index within its batch is the request index of its call id.
        let mut admissions = Vec::new();
        for (id, media_call) in candidates.into_iter() {
            let Some((batch_id, calls)) = call_batches.get_mut(&media_call) else {
                self.invariant_broken("every selected media call has its batch");
                return Vec::new();
            };
            let planned = match u32::try_from(calls.len()) {
                Ok(request_index) => self.plan_media_call(
                    id,
                    media_call,
                    CallId::new(*batch_id, request_index),
                    &mut admissions,
                ),
                Err(_) => Err("a batch's request count fits the IPC index"),
            };
            match planned {
                Ok(planned) => calls.push(planned),
                Err(invariant) => {
                    self.invariant_broken(invariant);
                    return Vec::new();
                }
            }
        }

        // A request's admission travels with the batch of the first call that
        // uses it.
        let mut starts: HashMap<MediaCall, Vec<BatchCommand>> = HashMap::new();
        for request in admissions {
            let owner = call_order.iter().copied().find(|media_call| {
                call_batches[media_call]
                    .1
                    .iter()
                    .any(|(call, _)| call.request_key == request.request_key)
            });
            let Some(owner) = owner else {
                self.invariant_broken("an admitted media request has a selected call");
                return Vec::new();
            };
            starts.entry(owner).or_default().push(BatchCommand::Start {
                request: Box::new(request),
            });
        }

        let mut batches = Vec::with_capacity(call_order.len() + 1);
        for media_call in call_order {
            let Some((batch_id, calls)) = call_batches.remove(&media_call) else {
                self.invariant_broken("every selected media call has its batch");
                return Vec::new();
            };
            let batch_commands = starts.remove(&media_call).unwrap_or_default();
            let batch = ExecutionBatch::new(batch_id, calls, batch_commands, Vec::new());
            self.inflight.register_pending_batch(&batch, submit_at);
            batches.push(batch);
        }

        // Retirement is its own batch, submitted after the calls of this pass.
        // Its result is the retirement acknowledgement, so no call's result
        // waits for storage the round no longer needs.
        if !commands.is_empty() {
            let batch = ExecutionBatch::new(
                self.inflight.next_batch_id(),
                Vec::new(),
                commands,
                Vec::new(),
            );
            self.inflight.register_pending_batch(&batch, submit_at);
            batches.push(batch);
        }
        batches
    }
}
