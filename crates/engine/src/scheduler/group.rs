//! The worker groups a scheduler places requests on (`specs/scheduler-design.md` §2.1
//! `WorkerGroup` and `Capacity`).
//!
//! A worker group is the ranks that run the same blocks under one queue depth. Its
//! [`WorkerGroup`] row holds what the deployment configures for it, what its executors stated
//! when they were ready ([`Capacity`]), how the epoch's session with the buffer pool states it
//! (its [`GroupId`], memory, slot region and per class its device regions), and the occupancy the
//! scheduler counts. A scheduler's groups are its own: the data-parallel replicas of a configured
//! group under it, its prefill and decode groups, and its media stages. A group of the `Experts`
//! role serves no phase and is never a placement.
//!
//! The rules here are SD §3.4 and §3.5:
//!
//! - **Phases and roles.** A group serves a phase when its executors stated it and its role
//!   allows it: a `Prefill` group never decodes or verifies, a `Decode` group does nothing else.
//! - **Occupancy.** The stage that serves a phase holds at most `Capacity::rows` requests at once.
//!   A request holds the stage while a call of the phase is in flight, so its further calls (a
//!   pipelined decode, the next denoising step) run beside it; the denoiser is held from the
//!   request's latent preparation to its last applied step, since its trajectory lives there
//!   across the gaps between its calls (SD §7.3).
//! - **Buckets.** A batch is formed only while it fits a bucket the executors captured.
//! - **Placement.** The candidates of a request's phase are the groups that serve it and have a
//!   free slot or hold the request already; among them the request's own group, then the
//!   policy's choice for a request placed nowhere, then the least loaded group (SD §3.5, §7.3).

use std::cmp::Reverse;

pub use ipc::Capacity;
use ipc::batch::Phase;
pub(super) use ipc::capacity::variants;
use proto::config::{VIDEO_CODEC_BLOCK, VIDEO_DECODER_BLOCK, WorkerGroupConfig, WorkerGroupRole};
use storage::runtime::{self, GroupId};

use super::{Request, Scheduler};

/// One worker group of a scheduler.
///
/// Invariants: `in_flight <= config.queue_depth` and `occupied[p] <= capacity.rows[p]` for every
/// phase index `p`.
#[derive(Clone, Debug)]
pub struct WorkerGroup {
    /// The group as the deployment configures it: its role, ranks and queue depth.
    pub config: WorkerGroupConfig,
    /// What the group's executors serve, as they stated it once they were ready.
    pub capacity: Capacity,
    /// The group as the epoch's session with the buffer pool states it: the [`GroupId`] batches,
    /// requests and stamps name it by, the device memory its ranks address, its slot region, and
    /// per class the device regions its owners' blocks are taken from, as indices into the
    /// session's regions (one per class, one per backing rank of a product, none for a class the
    /// group lacks).
    pub layout: runtime::Group,
    /// Per phase, by [`Phase::index`], the requests the stage serving it holds now.
    pub occupied: [u32; Phase::COUNT],
    /// Batches submitted to the group whose output has not been applied.
    pub in_flight: u32,
    /// Requests placed on the group.
    pub num_requests: u32,
}

impl WorkerGroup {
    /// A group with nothing placed on it and nothing in flight.
    pub fn new(config: WorkerGroupConfig, capacity: Capacity, layout: runtime::Group) -> Self {
        Self {
            config,
            capacity,
            layout,
            occupied: [0; Phase::COUNT],
            in_flight: 0,
            num_requests: 0,
        }
    }

    /// The number the session gives the group.
    pub fn id(&self) -> GroupId {
        self.layout.id
    }

    /// The device regions the group's owners take blocks of `class` from, as indices into the
    /// session's regions; empty for a class the group lacks.
    pub fn regions(&self, class: u8) -> &[u8] {
        self.layout
            .regions
            .get(usize::from(class))
            .map_or(&[], Vec::as_slice)
    }

    /// Whether the group runs calls of `phase`: its executors stated the phase, and its role
    /// allows it. A `Prefill` group runs no decode or verification, which a request reaches on a
    /// decode group by the transfer of its blocks; a `Decode` group runs nothing else; an
    /// `Experts` group serves the deployment's schedulers and runs no phase of a request.
    pub fn serves(&self, phase: Phase) -> bool {
        let decodes = matches!(phase, Phase::Decode | Phase::Verify);
        let allowed = match self.config.role {
            WorkerGroupRole::Model => true,
            WorkerGroupRole::Prefill => !decodes,
            WorkerGroupRole::Decode => decodes,
            WorkerGroupRole::Experts => false,
        };
        allowed && self.capacity.serves(phase)
    }

    /// Whether the stage that serves `phase` can take the next call of `request`: the request
    /// holds the stage already, or the stage holds fewer requests than its rows. `request` is as
    /// it stands before the call is counted in `scheduled`.
    pub fn has_room(&self, phase: Phase, request: &Request) -> bool {
        let stage = self.stage(phase).index();
        match (self.occupied.get(stage), self.capacity.rows.get(stage)) {
            (Some(occupied), Some(rows)) => self.holding(phase, request) || occupied < rows,
            _ => false,
        }
    }

    /// Counts a scheduled call of `phase` of `request`, whose `scheduled` already counts it: the
    /// request takes the stage when the call is its only one of the phase in flight, or, on the
    /// denoiser, when the call is its latent preparation. Its further calls take nothing more.
    pub fn occupy(&mut self, phase: Phase, request: &Request) {
        let stage = self.stage(phase);
        let takes = match stage {
            Phase::Denoise => phase == Phase::LatentPrepare,
            _ => request.in_flight(phase) == 1,
        };
        if takes && let Some(occupied) = self.occupied.get_mut(stage.index()) {
            *occupied += 1;
        }
    }

    /// Counts the applied output of a call of `phase` of `request`, whose `completed` already
    /// counts it: the request gives the stage up when no call of the phase remains in flight, or,
    /// on the denoiser, when the output is its last step. An output that returned ahead of an
    /// earlier call keeps the stage until that call returns.
    pub fn vacate(&mut self, phase: Phase, request: &Request) {
        let stage = self.stage(phase);
        let gives_up = match stage {
            Phase::Denoise => {
                phase == Phase::Denoise
                    && request.completed[Phase::Denoise.index()] == self.capacity.denoise_steps
            }
            _ => request.in_flight(phase) == 0,
        };
        if gives_up {
            self.release(stage);
        }
    }

    /// Counts a request placed on the group; [`WorkerGroup::leave`] takes it out again.
    pub fn join(&mut self) {
        self.num_requests += 1;
    }

    /// Takes `request` off the group, once for each [`WorkerGroup::join`]: when it is forgotten,
    /// returned to the waiting queue, or transferred to another group. Every stage it still holds
    /// on the group is given up, the denoiser of a request that ended before its last step
    /// included, which it holds until it leaves since its stored step is copied from there.
    /// `request` is read as the scheduler counts it, with its `placement` on this group.
    pub fn leave(&mut self, request: &Request) {
        self.num_requests = self.num_requests.saturating_sub(1);

        let id = Some(self.id());
        for phase in Phase::ALL {
            // Each stage once: a latent preparation's hold is the denoiser's.
            let placed = request.placement[phase.index()] == id;
            if placed && self.stage(phase) == phase && self.holding(phase, request) {
                self.release(phase);
            }
        }
    }

    /// The stage whose occupancy a call of `phase` counts in: a latent preparation opens the
    /// request's trajectory in the denoiser it runs on, so it counts as the denoiser's.
    fn stage(&self, phase: Phase) -> Phase {
        if phase == Phase::LatentPrepare && self.capacity.serves(Phase::Denoise) {
            Phase::Denoise
        } else {
            phase
        }
    }

    /// Whether `request` holds the stage of `phase` on this group: the denoiser from its latent
    /// preparation until its last step is applied, any other stage while a call of the phase is
    /// in flight.
    fn holding(&self, phase: Phase, request: &Request) -> bool {
        match self.stage(phase) {
            Phase::Denoise => {
                request.scheduled[Phase::LatentPrepare.index()] > 0
                    && request.completed[Phase::Denoise.index()] < self.capacity.denoise_steps
            }
            _ => request.in_flight(phase) > 0,
        }
    }

    fn release(&mut self, stage: Phase) {
        if let Some(occupied) = self.occupied.get_mut(stage.index()) {
            *occupied = occupied.saturating_sub(1);
        }
    }
}

/// Whether the video codec of `encoder` encodes every unit the video decoder of `decoder`
/// decodes on the host that decoded it, where the encoder reads the decoded unit in place
/// (`S/worker/codecs.rs:25`). A round deals its units to each block's ranks in order,
/// `units_per_rank` consecutive positions each: position `p` is decoded by the decoder's rank
/// `p / per_decoder` and encoded by the encoder's rank `p / per_encoder`. The pairing holds when
/// the codec covers a whole round and those two ranks share a host at every position. Groups
/// that state neither block run the two in one place, and pair.
pub(super) fn encodes_on_host(decoder: &WorkerGroupConfig, encoder: &WorkerGroupConfig) -> bool {
    let (Some(decoding), Some(encoding)) = (
        decoder.blocks.get(VIDEO_DECODER_BLOCK),
        encoder.blocks.get(VIDEO_CODEC_BLOCK),
    ) else {
        return true;
    };
    let per_decoder = decoding.units_per_rank.max(1);
    let per_encoder = encoding.units_per_rank.max(1);
    let round = decoding.ranks.len() * per_decoder;
    if encoding.ranks.len() * per_encoder < round {
        return false;
    }

    // The host of a block's rank, by its index into the group's ranks.
    fn host<'a>(group: &'a WorkerGroupConfig, rank: Option<&usize>) -> Option<&'a str> {
        let rank = group.ranks.get(*rank?)?;
        Some(rank.host.as_str())
    }
    (0..round).all(|position| {
        host(decoder, decoding.ranks.get(position / per_decoder))
            == host(encoder, encoding.ranks.get(position / per_encoder))
    })
}

/// Placement (SD §3.5 `place`, §7.3 "Routes"). The caller places a request's phases in the order
/// of its route (a video request's denoiser first, then its feeders, decoders, encoders and
/// muxer), so that the stages that follow the denoiser land beside it and the video encoder
/// beside the decoder; it records each choice in `placement` and calls
/// [`WorkerGroup::join`] once per group the request is newly placed on.
impl Scheduler {
    /// The groups that can run `phase` of `request`, in the order they are configured: those that
    /// serve it ([`WorkerGroup::serves`]) and that the request is placed on already or whose slot
    /// region has a free slot. A video encode is offered only the groups whose codec encodes on
    /// the hosts of the request's decoder ([`encodes_on_host`]). Empty when nothing can place it
    /// now.
    pub(super) fn candidates(&self, request: &Request, phase: Phase) -> Vec<GroupId> {
        let decoder = (phase == Phase::VideoEncode)
            .then(|| request.placement[Phase::VideoDecode.index()])
            .flatten()
            .and_then(|id| self.group(id));

        self.groups
            .iter()
            .filter(|group| group.serves(phase))
            .filter(|group| {
                request.placement.contains(&Some(group.id())) || self.free_slots(group) > 0
            })
            .filter(|group| {
                decoder.is_none_or(|decoder| encodes_on_host(&decoder.config, &group.config))
            })
            .map(WorkerGroup::id)
            .collect()
    }

    /// The group, among `candidates` (as [`Scheduler::candidates`] lists them), that a phase of
    /// `request` is placed on: a candidate the request is placed on already; else, for a request
    /// placed nowhere yet, the policy's choice `named`, an index into `candidates` (one that is
    /// not is passed over); else the candidate with the fewest placed requests, then the most free
    /// slots, then the first configured. `None` when there is no candidate.
    pub(super) fn choose(
        &self,
        request: &Request,
        candidates: &[GroupId],
        named: Option<usize>,
    ) -> Option<GroupId> {
        let placed = |id: &GroupId| request.placement.contains(&Some(*id));
        if let Some(&held) = candidates.iter().find(|id| placed(id)) {
            return Some(held);
        }

        let nowhere = request.placement.iter().all(Option::is_none);
        if nowhere && let Some(&named) = named.and_then(|index| candidates.get(index)) {
            return Some(named);
        }

        // `min_by_key` keeps the first of equal keys: the first configured.
        candidates.iter().copied().min_by_key(|id| {
            self.group(*id).map_or((u32::MAX, Reverse(0)), |group| {
                (group.num_requests, Reverse(self.free_slots(group)))
            })
        })
    }

    /// The free slots of `group`: the units its slot region has available, one per slot row.
    fn free_slots(&self, group: &WorkerGroup) -> u32 {
        self.available
            .get(usize::from(group.layout.slots))
            .copied()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::sync::atomic::AtomicU64;
    use std::time::Instant;

    use ipc::batch::{Batch, BatchId, Columns, Extent, Phase, Shape, Variant};
    use proto::config::{
        BlockConfig, BlockDistribution, SchedulerConfig, VIDEO_CODEC_BLOCK, VIDEO_DECODER_BLOCK,
        WorkerGroupConfig, WorkerGroupId, WorkerGroupRole, WorkerRank,
    };
    use proto::request::{GenerationRequest, RequestId};
    use storage::QueuePair;
    use storage::runtime::{self, Completion, GroupId, Region, Tier};

    use super::super::{Blocks, Executor, Request, RequestStatus, Scheduler};
    use super::{Capacity, WorkerGroup};

    struct Idle;

    impl Executor for Idle {
        fn submit(&self, _: Batch) {}

        fn map(
            &self,
            _: GroupId,
            _: u32,
            _: runtime::RequestId,
            _: runtime::Request,
            _: Completion,
        ) {
        }

        fn stamp(&self, _: GroupId, _: Vec<runtime::Request>) {}
    }

    /// A group of `role` named `id` whose ranks lie on `hosts`, one per rank.
    fn config(id: &str, role: WorkerGroupRole, hosts: &[&str]) -> WorkerGroupConfig {
        WorkerGroupConfig {
            id: WorkerGroupId(id.to_owned()),
            role,
            replicas: 1,
            ranks: hosts
                .iter()
                .map(|host| WorkerRank {
                    host: (*host).to_owned(),
                    device: "cpu".to_owned(),
                })
                .collect(),
            blocks: BTreeMap::new(),
            queue_depth: 2,
            mem_fraction_static: None,
        }
    }

    /// Executors that state `phases`, each served by a stage of `rows` requests, over 8 slots.
    fn capacity(phases: &[Phase], rows: u32) -> Capacity {
        let mut capacity = Capacity {
            phases: 0,
            slots: 8,
            kv_cache: None,
            cache_groups: Vec::new(),
            seeds: Vec::new(),
            buckets: Vec::new(),
            rows: [0; Phase::COUNT],
            max_model_len: 4096,
            denoise_steps: 2,
            bos_token_id: 0,
            eos_token_ids: Vec::new(),
            feature_bytes: 0,
            media_bytes: Vec::new(),
            call_window: 2,
        };
        for phase in phases {
            capacity.phases |= 1 << phase.index();
            capacity.rows[phase.index()] = rows;
        }
        capacity
    }

    fn group(config: WorkerGroupConfig, capacity: Capacity) -> WorkerGroup {
        let layout = runtime::Group {
            id: GroupId(0),
            memory: "memory".to_owned(),
            slots: 0,
            regions: Vec::new(),
        };
        WorkerGroup::new(config, capacity, layout)
    }

    /// A scheduler over `groups`, numbered from 0 in order, each with a slot region of its own
    /// whose available units are `free`.
    fn scheduler(groups: Vec<(WorkerGroupConfig, Capacity)>, free: &[u32]) -> Scheduler {
        let ids: Vec<GroupId> = (0..groups.len())
            .map(|index| GroupId(u8::try_from(index).unwrap()))
            .collect();
        let layout = runtime::Open {
            version: runtime::VERSION,
            epoch: 1,
            trial: false,
            regions: ids
                .iter()
                .map(|id| Region {
                    name: format!("slots-{}", id.0),
                    tier: Tier::Device,
                    space: format!("slots-{}", id.0),
                    unit_bytes: 256,
                    part_bytes: 256,
                    reserved: 1,
                    units: 9,
                    limit: 9,
                    segments: Vec::new(),
                })
                .collect(),
            classes: Vec::new(),
            trajectory: 0,
            peers: Vec::new(),
            groups: ids
                .iter()
                .map(|id| runtime::Group {
                    id: *id,
                    memory: format!("memory-{}", id.0),
                    slots: id.0,
                    regions: Vec::new(),
                })
                .collect(),
            schedulers: vec![ids.clone()],
        };
        let opened = runtime::Opened {
            regions: ids.iter().map(|id| (id.0 + 1, 9)).collect(),
            segments: Vec::new(),
            registers: vec![runtime::Registers {
                completed: Vec::new(),
                capacity: Vec::new(),
            }],
        };

        let (output, _) = crossbeam_channel::unbounded();
        let mut scheduler = Scheduler::new(
            SchedulerConfig::default(),
            proto::request::GenerationLimits::default(),
            &layout,
            &opened,
            0,
            QueuePair::create(4096, 1).unwrap(),
            groups,
            crate::scheduler::tests::idle_policy(),
            scheduling::Parameters::default(),
            Box::new(Idle),
            output,
            Arc::new(AtomicU64::new(0)),
        )
        .unwrap();
        scheduler.available = free.to_vec();
        scheduler
    }

    fn request(id: u64) -> Request {
        let input = GenerationRequest {
            request_id: RequestId(id),
            prompt_token_ids: vec![1, 2, 3],
            multimodal_inputs: Default::default(),
            sampling: Default::default(),
            max_und_tokens: 4,
            stop_strings: Vec::new(),
            stop_token_ids: Vec::new(),
            priority: 0,
            cache: Default::default(),
            include_stop_token: false,
            generated_token_ids: Vec::new(),
            adapter: 0,
        };
        let now = Instant::now();
        Request {
            id: RequestId(id),
            input: input.into(),
            arrival: now,
            status: RequestStatus::Running,
            placement: [None; Phase::COUNT],
            attachments: Vec::new(),
            scheduled: [0; Phase::COUNT],
            completed: [0; Phase::COUNT],
            early: Vec::new(),
            blocks: vec![Blocks::default()],
            queued: false,
            pending: 0,
            token_ids: vec![1, 2, 3],
            num_prompt_tokens: 3,
            num_computed_tokens: 0,
            num_cached_tokens: 0,
            num_host_tokens: 0,
            num_peer_tokens: 0,
            keys: Vec::new(),
            spec_token_ids: Vec::new(),
            num_draft_tokens: 0,
            num_encoded_inputs: 0,
            encodings: Vec::new(),
            unpinned: Vec::new(),
            num_acknowledged_tokens: 0,
            artifact: None,
            terminal: None,
            event_credit: 64,
            last_progress: now,
            dirty: false,
        }
    }

    /// The scheduler's side of one call of `phase`: counted when scheduled, as the stage counts
    /// it.
    fn schedule(group: &mut WorkerGroup, request: &mut Request, phase: Phase) {
        assert!(group.has_room(phase, request));
        request.scheduled[phase.index()] += 1;
        group.occupy(phase, request);
    }

    /// The scheduler's side of the output of a call of `phase`, applied in order.
    fn complete(group: &mut WorkerGroup, request: &mut Request, phase: Phase) {
        request.completed[phase.index()] += 1;
        group.vacate(phase, request);
    }

    fn batch(phase: Phase, calls: u64) -> Batch {
        Batch {
            id: BatchId(1),
            group: GroupId(0),
            phase,
            generation: 0,
            shape: Shape::decode(0, Variant::DecodeGreedy),
            requests: (0..calls).map(RequestId).collect(),
            columns: Columns::default(),
            media: Vec::new(),
            new_slots: Vec::new(),
        }
    }

    fn bucket(rows: u32, variant: Variant) -> Shape {
        Shape {
            rows,
            tokens: rows,
            variant,
            extent: Extent::None,
        }
    }

    #[test]
    fn a_role_bounds_the_phases_a_group_serves() {
        let stated = [Phase::Prefill, Phase::Decode, Phase::Verify];
        let served = |role| {
            let group = group(config("g", role, &["a"]), capacity(&stated, 1));
            Phase::ALL
                .into_iter()
                .filter(|phase| group.serves(*phase))
                .collect::<Vec<_>>()
        };

        assert_eq!(served(WorkerGroupRole::Model), stated);
        assert_eq!(served(WorkerGroupRole::Prefill), [Phase::Prefill]);
        assert_eq!(
            served(WorkerGroupRole::Decode),
            [Phase::Decode, Phase::Verify]
        );
        assert!(served(WorkerGroupRole::Experts).is_empty());
    }

    #[test]
    fn an_exclusive_stage_serves_one_request_and_all_of_its_calls() {
        let mut group = group(
            config("g", WorkerGroupRole::Model, &["a"]),
            capacity(&[Phase::Decode], 1),
        );
        let (mut first, second) = (request(1), request(2));

        // The first request's pipelined decodes run beside each other; the second waits for both.
        schedule(&mut group, &mut first, Phase::Decode);
        schedule(&mut group, &mut first, Phase::Decode);
        assert!(!group.has_room(Phase::Decode, &second));

        complete(&mut group, &mut first, Phase::Decode);
        assert!(!group.has_room(Phase::Decode, &second));
        complete(&mut group, &mut first, Phase::Decode);
        assert!(group.has_room(Phase::Decode, &second));
    }

    #[test]
    fn an_output_ahead_of_an_earlier_call_keeps_the_stage_until_that_call_returns() {
        let mut group = group(
            config("g", WorkerGroupRole::Model, &["a"]),
            capacity(&[Phase::VideoEncode], 1),
        );
        let (mut first, second) = (request(1), request(2));
        schedule(&mut group, &mut first, Phase::VideoEncode);
        schedule(&mut group, &mut first, Phase::VideoEncode);

        // The second round returns first: it waits in `early` and counts once the first returns.
        first.early.push((Phase::VideoEncode, 1));
        group.vacate(Phase::VideoEncode, &first);
        assert!(!group.has_room(Phase::VideoEncode, &second));

        first.early.clear();
        first.completed[Phase::VideoEncode.index()] = 2;
        group.vacate(Phase::VideoEncode, &first);
        assert!(group.has_room(Phase::VideoEncode, &second));
    }

    #[test]
    fn the_denoiser_is_held_from_latent_preparation_to_the_last_applied_step() {
        let mut group = group(
            config("g", WorkerGroupRole::Model, &["a"]),
            capacity(&[Phase::LatentPrepare, Phase::Denoise], 1),
        );
        let (mut first, second) = (request(1), request(2));

        schedule(&mut group, &mut first, Phase::LatentPrepare);
        assert!(!group.has_room(Phase::LatentPrepare, &second));

        // Between its calls nothing of the first request is in flight, and it keeps the denoiser.
        complete(&mut group, &mut first, Phase::LatentPrepare);
        schedule(&mut group, &mut first, Phase::Denoise);
        complete(&mut group, &mut first, Phase::Denoise);
        assert!(!group.has_room(Phase::LatentPrepare, &second));

        schedule(&mut group, &mut first, Phase::Denoise);
        complete(&mut group, &mut first, Phase::Denoise);
        assert!(group.has_room(Phase::LatentPrepare, &second));
    }

    #[test]
    fn a_request_that_ends_before_its_last_step_holds_the_denoiser_until_it_leaves() {
        let mut group = group(
            config("g", WorkerGroupRole::Model, &["a"]),
            capacity(&[Phase::LatentPrepare, Phase::Denoise], 1),
        );
        let (mut first, second) = (request(1), request(2));
        first.placement[Phase::LatentPrepare.index()] = Some(group.id());
        first.placement[Phase::Denoise.index()] = Some(group.id());
        group.join();

        schedule(&mut group, &mut first, Phase::LatentPrepare);
        complete(&mut group, &mut first, Phase::LatentPrepare);
        schedule(&mut group, &mut first, Phase::Denoise);
        complete(&mut group, &mut first, Phase::Denoise);
        first.status = RequestStatus::Finished;
        assert!(!group.has_room(Phase::LatentPrepare, &second));

        group.leave(&first);
        assert!(group.has_room(Phase::LatentPrepare, &second));
        assert_eq!(group.num_requests, 0);
    }

    #[test]
    fn a_batch_grows_within_the_largest_bucket_of_its_phase_and_the_slots() {
        let mut capacity = capacity(&[], 1);
        capacity.slots = 3;
        capacity.buckets = vec![
            bucket(4, Variant::DecodeSampled),
            bucket(2, Variant::PrefillOutput),
            bucket(1, Variant::Denoise),
            bucket(1, Variant::Mux),
        ];

        // Decode rows stop at the slots, prefill rows at the prefill buckets.
        assert!(capacity.holds(&batch(Phase::Decode, 2), Phase::Decode));
        assert!(!capacity.holds(&batch(Phase::Decode, 3), Phase::Decode));
        assert!(capacity.holds(&batch(Phase::Prefill, 1), Phase::Prefill));
        assert!(!capacity.holds(&batch(Phase::Prefill, 2), Phase::Prefill));
        assert!(!capacity.holds(&batch(Phase::Denoise, 1), Phase::Denoise));

        // A phase no bucket names runs one call per batch; a host stage takes every call.
        assert!(capacity.holds(&batch(Phase::Verify, 0), Phase::Verify));
        assert!(!capacity.holds(&batch(Phase::Verify, 1), Phase::Verify));
        assert!(capacity.holds(&batch(Phase::Mux, 5), Phase::Mux));
    }

    #[test]
    fn a_request_stays_on_the_group_it_runs_on() {
        let model = || {
            (
                config("model", WorkerGroupRole::Model, &["a"]),
                capacity(&[Phase::Prefill, Phase::Decode], 1),
            )
        };
        let mut scheduler = scheduler(vec![model(), model()], &[4, 4]);
        scheduler.groups[1].num_requests = 3;
        let mut placed = request(1);
        placed.placement[Phase::Prefill.index()] = Some(GroupId(1));

        // Neither the load nor the policy moves it.
        let candidates = scheduler.candidates(&placed, Phase::Decode);
        assert_eq!(candidates, [GroupId(0), GroupId(1)]);
        assert_eq!(
            scheduler.choose(&placed, &candidates, Some(0)),
            Some(GroupId(1))
        );
    }

    #[test]
    fn a_new_request_goes_where_the_policy_names_or_to_the_least_loaded_group() {
        let model = || {
            (
                config("model", WorkerGroupRole::Model, &["a"]),
                capacity(&[Phase::Prefill], 1),
            )
        };
        let mut scheduler = scheduler(vec![model(), model(), model()], &[5, 3, 4]);
        scheduler.groups[0].num_requests = 2;
        scheduler.groups[1].num_requests = 1;
        scheduler.groups[2].num_requests = 1;
        let fresh = request(1);
        let candidates = scheduler.candidates(&fresh, Phase::Prefill);

        assert_eq!(
            scheduler.choose(&fresh, &candidates, Some(0)),
            Some(GroupId(0))
        );
        // An index that names no candidate is passed over: the fewest placed requests, then the
        // most free slots.
        assert_eq!(
            scheduler.choose(&fresh, &candidates, Some(3)),
            Some(GroupId(2))
        );

        // Equal load and free slots go to the first configured.
        scheduler.available = vec![4, 4, 4];
        assert_eq!(
            scheduler.choose(&fresh, &candidates, None),
            Some(GroupId(1))
        );
    }

    #[test]
    fn a_group_without_a_free_slot_takes_only_the_requests_it_holds() {
        let model = || {
            (
                config("model", WorkerGroupRole::Model, &["a"]),
                capacity(&[Phase::Prefill, Phase::Decode], 1),
            )
        };
        let scheduler = scheduler(vec![model(), model()], &[0, 2]);
        let mut held = request(1);

        assert_eq!(scheduler.candidates(&held, Phase::Prefill), [GroupId(1)]);
        held.placement[Phase::Prefill.index()] = Some(GroupId(0));
        assert_eq!(
            scheduler.candidates(&held, Phase::Decode),
            [GroupId(0), GroupId(1)]
        );
    }

    #[test]
    fn prefill_and_decode_groups_split_a_request_and_experts_take_none() {
        let stated = [Phase::Prefill, Phase::Decode, Phase::Verify];
        let scheduler = scheduler(
            vec![
                (
                    config("prefill", WorkerGroupRole::Prefill, &["a"]),
                    capacity(&stated, 1),
                ),
                (
                    config("decode", WorkerGroupRole::Decode, &["a"]),
                    capacity(&stated, 1),
                ),
                (
                    config("experts", WorkerGroupRole::Experts, &["a"]),
                    capacity(&stated, 1),
                ),
            ],
            &[4, 4, 4],
        );
        let mut prefilled = request(1);
        prefilled.placement[Phase::Prefill.index()] = Some(GroupId(0));

        assert_eq!(
            scheduler.candidates(&prefilled, Phase::Prefill),
            [GroupId(0)]
        );
        let candidates = scheduler.candidates(&prefilled, Phase::Decode);
        assert_eq!(candidates, [GroupId(1)]);
        assert_eq!(
            scheduler.choose(&prefilled, &candidates, None),
            Some(GroupId(1))
        );
    }

    #[test]
    fn a_video_unit_is_encoded_on_the_host_that_decoded_it() {
        // Two decoding ranks, one unit each, on hosts `a` and `b`.
        let block = |ranks: Vec<usize>| BlockConfig {
            distribution: Some(BlockDistribution::TemporalUnits),
            ..BlockConfig::parallel(ranks)
        };
        let with = |mut config: WorkerGroupConfig, name: &str, ranks: Vec<usize>| {
            config.blocks.insert(name.to_owned(), block(ranks));
            config
        };
        let decoder = with(
            config("decoder", WorkerGroupRole::Model, &["a", "b"]),
            VIDEO_DECODER_BLOCK,
            vec![0, 1],
        );
        // The codec on the decoder's hosts in order; in the other order; on one rank only.
        let beside = with(
            config("beside", WorkerGroupRole::Model, &["a", "b"]),
            VIDEO_CODEC_BLOCK,
            vec![0, 1],
        );
        let crossed = with(
            config("crossed", WorkerGroupRole::Model, &["b", "a"]),
            VIDEO_CODEC_BLOCK,
            vec![0, 1],
        );
        let narrow = with(
            config("narrow", WorkerGroupRole::Model, &["a"]),
            VIDEO_CODEC_BLOCK,
            vec![0],
        );

        let scheduler = scheduler(
            vec![
                (decoder, capacity(&[Phase::VideoDecode], 1)),
                (crossed, capacity(&[Phase::VideoEncode], 1)),
                (beside, capacity(&[Phase::VideoEncode], 1)),
                (narrow, capacity(&[Phase::VideoEncode], 1)),
            ],
            &[4, 4, 4, 4],
        );
        let mut video = request(1);
        video.placement[Phase::VideoDecode.index()] = Some(GroupId(0));

        assert_eq!(
            scheduler.candidates(&video, Phase::VideoEncode),
            [GroupId(2)]
        );
    }
}
