//! The harness every area of the suite builds on: a deployment stated in a few parameters, the
//! scheduler over the fake executors of its groups and the in-process buffer pool, a step that
//! moves all three as the engine's loop would, and readers of what the scheduler reported.
//!
//! - [`Deployment`] states the worker groups (their roles and phases), the cache groups of the
//!   model, speculation, chat images or video, and the sizes of the session; [`Deployment::build`]
//!   opens the session at the buffer pool, builds the scheduler with the `arrival` policy this tree
//!   builds (which the scheduler instantiates on its own snapshot, which states the phases its
//!   groups serve), and answers its startup `List`.
//! - [`Harness::step`] is one pass: the buffer pool's turn (it takes the requests the last pass
//!   wrote, and writes the completions that are due), every message of the scheduler's inbox
//!   applied (the outputs of the batches the workers ran, the requests they stamped), then the
//!   scheduler's step. The fake answers a batch as it is submitted, from its own rule and the
//!   tokens a case scripts ([`Harness::sample`]); the answer reaches the scheduler at the next
//!   pass, as real workers' would, and the copies queued on a group run with its next batch.
//!   Every pass ends with the fake's record of A1 to A4 empty.
//! - The pace of the buffer pool: [`InProcessPool::delay`] and [`InProcessPool::withhold`] on
//!   [`Harness::pool`], and [`Harness::reorder`] for completions reordered within a window.
//!
//! The session's layout, by region index: per group `g` its KV units (`kv-g`) and its slot rows
//! (`slots-g`); the device buffers every group shares, which hold the sized classes (encoder
//! outputs and media products); the host's KV pool when the deployment has a host tier; the host
//! latent pool of stored trajectories; and for a model that keeps a recurrent state, per group
//! `g` its state units (`state-g`), one checkpoint each, as a rank states its state region
//! (`specs/engine-worker-port.md` §1.6). Its classes, in order: one per cache group of the model
//! (class `c` is cache group `c`), each block one unit of its region, the encoder outputs, one per
//! media product kind, and the stored trajectories.

use engine::batch::Generations;
use std::collections::{HashMap, HashSet, VecDeque};
use std::num::{NonZeroU32, NonZeroUsize};
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use crossbeam_channel::{Receiver, unbounded};
use engine::batch::Capacity;
use engine::batch::cache::CacheGroupKind;
use engine::batch::{
    Batch, BatchOutput, CallOutput, CallStatus, ErrorCode, Extent, MediaRow, Phase, Shape, Variant,
};
use engine::scheduler::{Message, Output, Scheduler, policy};
use proto::config::{
    BlockConfig, KvCacheConfig, SchedulerConfig, Tuning, WorkerGroupConfig, WorkerGroupId,
    WorkerGroupRole, WorkerRank,
};
use proto::report::SchedulerStats;
use proto::request::{
    DiffusionRequest, DiffusionSamplingParams, Event, FinishReason, GenerationFeatures,
    GenerationLimits, GenerationRequest, ImageEncoder, ImageEncoderInput, ImageInput,
    MigrationCause, RequestId, Trajectory,
};
use service::lock;
use storage::protocol::BlockState;
use storage::runtime::{self, Class, Completion, GroupId, OwnerId, Region, Tier};
use storage::{Key, QueuePair};
use testkit::executor::output_of;
use testkit::text::scored;
use testkit::{FakeExecutor, InProcessPool};

/// The event window the harness states as the consumer of every request it submits, as a daemon
/// states its own in its welcome.
const EVENT_WINDOW: std::num::NonZeroU32 = std::num::NonZeroU32::new(64).unwrap();

/// The first two worker groups of a deployment, as the session numbers them.
pub(crate) const A: GroupId = GroupId(0);
pub(crate) const B: GroupId = GroupId(1);

/// The tokens of a block of a cache group the deployment does not size otherwise: small, so that
/// a request crosses blocks, and asks for more of them, within a few calls.
pub(crate) const BLOCK: u32 = 4;

/// The end-of-sequence token of every deployment of the harness.
pub(crate) const EOS: u32 = testkit::EOS;

/// The denoising steps of a video deployment.
pub(crate) const STEPS: u32 = 4;

/// The latent seed a group that denoises states for the weights it serves
/// (`Capacity::latent_seed`), which keys the trajectories it stores.
pub(crate) const LATENT_SEED: u64 = 0x5eed;

/// The media pipeline a video deployment's groups serve and state (`Capacity::pipeline`):
/// MiniMax-H3's, as its family declares it. Its phases are every phase of a video request, which
/// a video deployment of one group serves; its products are the classes a session states between
/// the encoder outputs and the stored trajectories, which the scheduler finds as the classes just
/// before the stored trajectories'.
pub(crate) fn pipeline() -> nn::pipeline::Pipeline {
    models::minimax_h3::family::pipeline()
}

/// The raster and timeline of every video request of the harness, which the denoising shapes of a
/// video deployment capture.
const FRAMES: u32 = 121;
const WIDTH: u32 = 64;
const HEIGHT: u32 = 32;

/// The prompt rows a captured denoising shape holds.
const TEXT_ROWS: u32 = 128;

/// The bytes of one token of a chat image's encoder output.
const FEATURE_BYTES: u32 = 64;

/// The bytes of a unit of each kind of region.
const KV_UNIT: u64 = 4096;
const STATE_UNIT: u64 = 1 << 20;
const BUFFER_UNIT: u64 = 256;
const LATENT_UNIT: u64 = 4096;

/// A deployment of one scheduler, stated in a few parameters; [`Deployment::build`] makes the
/// harness over it.
#[derive(Clone, Debug)]
pub(crate) struct Deployment {
    /// Per worker group, in session order from [`A`]: its role and the phases it serves.
    groups: Vec<(WorkerGroupRole, Vec<Phase>)>,
    /// The model's cache groups, class `c` being cache group `c`: kind and block size in tokens.
    cache: Vec<(CacheGroupKind, u32)>,
    /// The drafts a verification carries; 0 for a deployment that does not speculate.
    drafts: u32,
    /// The KV units of each group's region, and of the host's KV pool (0: no host tier).
    kv_units: u32,
    host_units: u32,
    /// The checkpoints of each group's state region, for a model that keeps a recurrent state;
    /// `None`: two per request slot, which no load of a case runs short of.
    state_units: Option<u32>,
    /// The decoded tokens between two checkpoints of a request's recurrent state
    /// (`KvCacheConfig::state_interval`).
    state_interval: NonZeroU32,
    /// The units of the host latent pool, which holds stored trajectories.
    latent_units: u32,
    /// The request slots of each group.
    slots: u32,
    /// The rows of the largest captured bucket, and the tokens of the widest prefill bucket.
    rows: u32,
    tokens: u32,
    /// The prefill chunk the step values state.
    chunk: u32,
    /// Whether the step values state the step's token budget (the widest prefill bucket's
    /// tokens); otherwise the scheduler takes the widest captured bucket's.
    states_budget: bool,
    /// The bound of the waiting queue the step values state.
    queued: Option<usize>,
    queue_depth: usize,
    /// The bytes of each ring of the queue pair.
    ring_bytes: usize,
    /// Whether the deployment keeps its host copies in the remote store, and has peers whose
    /// host copies a prefetch may read.
    remote: bool,
    peers: bool,
    /// The identity of the weights the groups that encode state for their encoder outputs
    /// (`Capacity::encoder_identity`); empty: none.
    encoder_identity: String,
    /// Blocks of the groups' configurations: per entry the group's index, the block's name and
    /// the block.
    blocks: Vec<(usize, String, BlockConfig)>,
    /// The ranks of the groups that have more than one: per entry the group's index and its
    /// ranks, all on this host.
    ranks: Vec<(usize, usize)>,
}

impl Deployment {
    /// A token deployment: group [`A`] of the `Model` role serving prefills and decodes, over a
    /// model of one full-attention cache group of [`BLOCK`] tokens; 256 KV units, 8 slots, buckets
    /// of 8 rows and 64 tokens, chunks of 64 tokens, queue depth 2, no host tier.
    pub(crate) fn tokens() -> Self {
        Self {
            groups: vec![(WorkerGroupRole::Model, vec![Phase::Prefill, Phase::Decode])],
            cache: vec![(CacheGroupKind::Full, BLOCK)],
            drafts: 0,
            kv_units: 256,
            host_units: 0,
            state_units: None,
            state_interval: proto::config::DEFAULT_STATE_INTERVAL,
            latent_units: 1 << 12,
            slots: 8,
            rows: 8,
            tokens: 64,
            chunk: 64,
            states_budget: true,
            queued: None,
            queue_depth: 2,
            ring_bytes: 1 << 16,
            remote: false,
            peers: false,
            encoder_identity: String::new(),
            blocks: Vec::new(),
            ranks: Vec::new(),
        }
    }

    /// A video deployment: one group per entry of `groups`, of the `Model` role, serving the
    /// phases it lists (the [`pipeline`]'s phases on one group serve every phase), with [`STEPS`]
    /// denoising steps on the group that denoises and the captured shapes of the harness's video
    /// requests ([`video`]).
    pub(crate) fn video(groups: &[&[Phase]]) -> Self {
        Self {
            groups: groups
                .iter()
                .map(|phases| (WorkerGroupRole::Model, phases.to_vec()))
                .collect(),
            cache: Vec::new(),
            ..Self::tokens()
        }
    }

    /// `count` replicas of group [`A`], each with the same roles and phases.
    pub(crate) fn replicas(mut self, count: usize) -> Self {
        let first = self.groups[0].clone();
        self.groups = vec![first; count];
        self
    }

    /// Prefill and decode on two groups: [`A`] of the `Prefill` role serving every phase of the
    /// deployment but decodes and verifications, [`B`] of the `Decode` role serving those.
    pub(crate) fn disaggregated(mut self) -> Self {
        let (_, phases) = self.groups[0].clone();
        let decodes = |phase: &Phase| matches!(phase, Phase::Decode | Phase::Verify);
        let prefill = phases.iter().copied().filter(|phase| !decodes(phase));
        let decode = phases.iter().copied().filter(decodes);
        self.groups = vec![
            (WorkerGroupRole::Prefill, prefill.collect()),
            (WorkerGroupRole::Decode, decode.collect()),
        ];
        self
    }

    /// The model's cache groups, each with its kind and block size in tokens: full attention,
    /// a sliding window, a recurrent state.
    pub(crate) fn cache(mut self, groups: &[(CacheGroupKind, u32)]) -> Self {
        let draft = self.cache.iter().copied().find(|(kind, _)| kind.is_draft());
        self.cache = groups.to_vec();
        self.cache.extend(draft);
        self
    }

    /// A deployment that speculates: its token groups verify `drafts` drafts per call and refill
    /// their draft caches, held in a draft cache group of [`BLOCK`] tokens, the model's last; the
    /// drafter proposes `drafts` tokens with every sampled one, and each verification accepts
    /// every draft it carried ([`Harness::sample`] still names what it samples).
    pub(crate) fn speculative(mut self, drafts: u32) -> Self {
        self.drafts = drafts;
        for (_, phases) in &mut self.groups {
            if phases.contains(&Phase::Prefill) {
                phases.push(Phase::Refill);
            }
            if phases.contains(&Phase::Decode) {
                phases.push(Phase::Verify);
            }
        }
        self.cache.retain(|(kind, _)| !kind.is_draft());
        self.cache.push((CacheGroupKind::Draft, BLOCK));
        self
    }

    /// A speculating deployment's token groups as a speculating rank states them: verifications
    /// in place of decodes, since the rank captures verification buckets and no decode bucket.
    pub(crate) fn verifying(mut self) -> Self {
        for (_, phases) in &mut self.groups {
            phases.retain(|phase| *phase != Phase::Decode);
        }
        self
    }

    /// Chat images: the groups that prefill also encode images ([`chat_image`]), each token of an
    /// encoder output taking [`FEATURE_BYTES`].
    pub(crate) fn images(mut self) -> Self {
        for (_, phases) in &mut self.groups {
            if phases.contains(&Phase::Prefill) {
                phases.push(Phase::VisionEncode);
            }
        }
        self
    }

    /// Chat images encoded on a group of their own: one more group, of the `Model` role, that
    /// serves vision encodes alone; the token groups read the encoder outputs it writes.
    pub(crate) fn encoder(mut self) -> Self {
        self.groups
            .push((WorkerGroupRole::Model, vec![Phase::VisionEncode]));
        self
    }

    /// The identity of the weights that compute encoder outputs, which every group that encodes
    /// (a chat image, a video's conditioning) states.
    pub(crate) fn encoder_identity(mut self, identity: &str) -> Self {
        self.encoder_identity = identity.to_owned();
        self
    }

    /// Block `name` of group `index`'s configuration, over the group's ranks.
    pub(crate) fn block(mut self, index: usize, name: &str, block: BlockConfig) -> Self {
        self.blocks.push((index, name.to_owned(), block));
        self
    }

    /// `count` ranks of group `index`, all on this host; a group has one otherwise.
    pub(crate) fn ranks(mut self, index: usize, count: usize) -> Self {
        self.ranks.push((index, count));
        self
    }

    /// The units of the host latent pool, which holds stored trajectories.
    pub(crate) fn latents(mut self, units: u32) -> Self {
        self.latent_units = units;
        self
    }

    /// The KV units of each group's region.
    pub(crate) fn kv_units(mut self, units: u32) -> Self {
        self.kv_units = units;
        self
    }

    /// A host tier of `units` KV units: committed full-attention blocks are stored there.
    pub(crate) fn host(mut self, units: u32) -> Self {
        self.host_units = units;
        self
    }

    /// The checkpoints of each group's state region, for a model that keeps a recurrent state:
    /// a region of its own beside the KV region, one checkpoint a unit.
    pub(crate) fn state_units(mut self, units: u32) -> Self {
        self.state_units = Some(units);
        self
    }

    /// The decoded tokens between two checkpoints of a request's recurrent state.
    pub(crate) fn state_interval(mut self, tokens: NonZeroU32) -> Self {
        self.state_interval = tokens;
        self
    }

    /// A remote store beside the host tier: every block stored on the host is stored on to it,
    /// and an admission prefetches from it what it holds beyond the host.
    pub(crate) fn remote(mut self) -> Self {
        self.remote = true;
        self
    }

    /// Peers beside the host tier: an admission whose prompt reaches far enough beyond what this
    /// host holds prefetches from their host copies.
    pub(crate) fn peers(mut self) -> Self {
        self.peers = true;
        self
    }

    /// The request slots of each group, which bound the running requests.
    pub(crate) fn slots(mut self, slots: u32) -> Self {
        self.slots = slots;
        self
    }

    /// Buckets of `rows` rows and `tokens` prefill tokens, which bound a step's calls and tokens.
    pub(crate) fn batch(mut self, rows: u32, tokens: u32) -> Self {
        self.rows = rows;
        self.tokens = tokens;
        self
    }

    /// The prefill chunk the policy declares.
    pub(crate) fn chunk(mut self, tokens: u32) -> Self {
        self.chunk = tokens;
        self
    }

    /// A policy package that states no step token budget: the scheduler budgets each step by its
    /// configuration's, the widest prefill bucket's tokens (SD §7.1).
    pub(crate) fn unstated_budget(mut self) -> Self {
        self.states_budget = false;
        self
    }

    /// The bound of the waiting queue the policy declares.
    pub(crate) fn max_queued(mut self, requests: usize) -> Self {
        self.queued = Some(requests);
        self
    }

    /// The batches each group holds in flight.
    pub(crate) fn queue_depth(mut self, depth: usize) -> Self {
        self.queue_depth = depth;
        self
    }

    /// The bytes of each ring of the queue pair (a power of two of at least 64).
    pub(crate) fn ring_bytes(mut self, bytes: usize) -> Self {
        self.ring_bytes = bytes;
        self
    }

    /// Opens the session at a new in-process buffer pool, builds the scheduler over the fake
    /// executors of its groups with the `arrival` policy, and answers the scheduler's startup `List`
    /// (an empty buffer pool lists nothing).
    ///
    /// # Panics
    ///
    /// When the buffer pool refuses the layout or the scheduler refuses the deployment.
    pub(crate) fn build(self) -> Harness {
        let layout = self.layout();
        let ids: Vec<GroupId> = (0..self.groups.len()).map(group_id).collect();

        let (inbox, messages) = unbounded();
        let fake = FakeExecutor::new(&ids, self.written(), inbox);
        let fake = match self.class_of(|kind| kind.is_draft()) {
            Some(class) => fake.with_draft(class, BLOCK),
            None => fake,
        };
        let fake = match self.class_of(|kind| kind.is_state()) {
            Some(class) => fake.with_checkpoints(class),
            None => fake,
        };
        let rule = Arc::new(Mutex::new(Rule {
            proposals: self.drafts,
            ..Rule::default()
        }));
        let responding = Arc::clone(&rule);
        let fake = fake.with_output(Arc::new(move |batch: &Batch| respond(&responding, batch)));

        let mut pool = InProcessPool::new(self.ring_bytes);
        let (opened, mut ends) = pool
            .open(&layout, vec![fake.clone()])
            .unwrap_or_else(|refusal| panic!("the buffer pool refuses the layout: {refusal}"));

        let groups: Vec<(WorkerGroupConfig, Capacity)> = (0..self.groups.len())
            .map(|index| (self.config(index), self.capacity(index)))
            .collect();
        // The scheduler instantiates the policy on its own snapshot, which states the phases its
        // groups serve.
        let module = Arc::new(policy::load(&testkit::packaged_module()).unwrap());
        let (output, outputs) = unbounded();
        let config = self.scheduler_config();
        let tuning = Arc::new(ArcSwap::from_pointee(self.tuning()));
        let scheduler = Scheduler::new(
            config.clone(),
            limits(),
            &layout,
            &opened,
            0,
            ends.remove(0),
            groups,
            module,
            testkit::packaged_parameters(),
            Box::new(fake.clone()),
            output,
            Arc::new(Generations::default()),
            Arc::clone(&tuning),
        )
        .unwrap_or_else(|refusal| panic!("the scheduler refuses the deployment: {refusal}"));

        let mut harness = Harness {
            scheduler,
            config,
            tuning,
            pool,
            fake,
            inbox: messages,
            outputs,
            record: Vec::new(),
            rule,
            videos: HashSet::new(),
            submitted: Vec::new(),
            window: None,
            steps: 0,
            layout,
            other: None,
        };
        // The scheduler writes its `List` at its first step, and reads its `Listing` at its
        // second: every case starts with the startup answered.
        harness.step();
        harness.step();
        harness
    }

    /// The session's layout ([`Harness`]'s module documentation states it).
    fn layout(&self) -> runtime::Open {
        let region = |name: String, tier, unit_bytes, reserved: u32, units: u32| Region {
            space: name.clone(),
            name,
            tier,
            unit_bytes,
            part_bytes: unit_bytes,
            reserved,
            units,
            limit: reserved + units,
            segments: Vec::new(),
            memory: String::new(),
        };
        let mut regions = Vec::new();
        for index in 0..self.groups.len() {
            regions.push(region(
                format!("kv-{index}"),
                Tier::Device,
                KV_UNIT,
                1,
                self.kv_units,
            ));
            regions.push(region(
                format!("slots-{index}"),
                Tier::Device,
                1,
                1,
                self.slots,
            ));
        }
        let buffers = index(regions.len());
        regions.push(region(
            "buffers".into(),
            Tier::Device,
            BUFFER_UNIT,
            0,
            1 << 16,
        ));
        let host = (self.host_units > 0).then(|| {
            regions.push(region(
                "host".into(),
                Tier::Host,
                KV_UNIT,
                0,
                self.host_units,
            ));
            index(regions.len() - 1)
        });
        let latents = index(regions.len());
        regions.push(region(
            "latents".into(),
            Tier::Host,
            LATENT_UNIT,
            0,
            self.latent_units,
        ));

        // Per group, the state region of a model that keeps a recurrent state: no unit reserved,
        // as a rank states it.
        let state_class = self.class_of(|kind| kind.is_state());
        let state_regions: Vec<u8> = match state_class {
            Some(_) => (0..self.groups.len())
                .map(|at| {
                    regions.push(region(
                        format!("state-{at}"),
                        Tier::Device,
                        STATE_UNIT,
                        0,
                        self.state_units.unwrap_or(2 * self.slots),
                    ));
                    index(regions.len() - 1)
                })
                .collect(),
            None => Vec::new(),
        };

        // A KV block is one unit of its region, a checkpoint one of the state region; a sized
        // block states its units when it is asked for. A recurrent state's class holds its live
        // state in the slot, as a rank states it.
        let class = |units, host, state_in_slot| Class {
            units,
            host,
            state_in_slot,
            tensors: [0; 32],
            encoding: [0; 32],
            source: None,
        };
        let mut classes: Vec<Class> = self
            .cache
            .iter()
            .map(|(kind, _)| class(1, host.map(|host| (host, 1)), kind.is_state()))
            .collect();
        // The class of encoder outputs, then one per product of a video deployment's pipeline;
        // a token deployment holds no product.
        classes.push(class(0, None, false));
        classes.extend(self.products().map(|_| class(0, None, false)));
        classes.push(class(0, Some((latents, 0)), false));
        let trajectory = index(classes.len() - 1);

        let groups = (0..self.groups.len())
            .map(|at| {
                let kv = index(2 * at);
                let per_class = (0..classes.len()).map(|class| {
                    if Some(class) == state_class.map(usize::from) {
                        vec![state_regions[at]]
                    } else if class < self.cache.len() {
                        vec![kv]
                    } else if class < usize::from(trajectory) {
                        vec![buffers]
                    } else {
                        Vec::new()
                    }
                });
                runtime::Group {
                    id: group_id(at),
                    memory: format!("gpu-{at}"),
                    slots: index(2 * at + 1),
                    regions: per_class.collect(),
                    receives: Vec::new(),
                    parts: Vec::new(),
                }
            })
            .collect();

        runtime::Open {
            version: runtime::VERSION,
            epoch: 1,
            trial: false,
            regions,
            classes,
            trajectory,
            peers: Vec::new(),
            groups,
            schedulers: vec![(0..self.groups.len()).map(group_id).collect()],
        }
    }

    /// The classes a prefill, decode or verification row writes at the block of each of its cache
    /// positions, with their block sizes: every cache group but a recurrent state, whose
    /// checkpoints rows name, and the draft cache, which refills write.
    fn written(&self) -> Vec<(u8, u32)> {
        let classes = (0..).zip(&self.cache);
        let written = classes.filter(|(_, (kind, _))| !kind.is_state() && !kind.is_draft());
        written.map(|(class, (_, block))| (class, *block)).collect()
    }

    /// The class of the first cache group of a kind.
    fn class_of(&self, kind: impl Fn(&CacheGroupKind) -> bool) -> Option<u8> {
        let found = self.cache.iter().position(|(of, _)| kind(of))?;
        u8::try_from(found).ok()
    }

    /// Group `index`'s configuration: its ranks on this host, one unless stated
    /// ([`Deployment::ranks`]), with the deployment's queue depth and the blocks stated for it
    /// ([`Deployment::block`]).
    fn config(&self, index: usize) -> WorkerGroupConfig {
        let ranks = self
            .ranks
            .iter()
            .rev()
            .find(|(group, _)| *group == index)
            .map_or(1, |&(_, count)| count);
        let rank = WorkerRank {
            host: "localhost".to_owned(),
            device: "cpu".parse().unwrap(),
        };
        WorkerGroupConfig {
            id: WorkerGroupId(format!("group-{index}")),
            role: self.groups[index].0,
            replicas: 1,
            ranks: vec![rank; ranks],
            blocks: self
                .blocks
                .iter()
                .filter(|(group, ..)| *group == index)
                .map(|(_, name, block)| (name.clone(), block.clone()))
                .collect(),
            queue_depth: self.queue_depth,
            mem_fraction_static: None,
        }
    }

    /// What group `index`'s workers state: its phases, its slots (every stage holds that many
    /// requests), the model's cache groups for a group that runs token calls, a bucket of the
    /// deployment's rows for each phase it serves (and its tokens for prefills), for a group that
    /// denoises the steps and the shapes of the harness's video requests, and for a group that
    /// encodes the deployment's encoder identity. A group of a model that keeps a recurrent state
    /// serves no refill, as its ranks state none: a refill would evaluate positions behind its
    /// slot's own from a state the runner has none of.
    fn capacity(&self, index: usize) -> Capacity {
        let stateful = self.class_of(|kind| kind.is_state()).is_some();
        let phases: Vec<Phase> = self.groups[index]
            .1
            .iter()
            .copied()
            .filter(|phase| *phase != Phase::Refill || !stateful)
            .collect();
        let serves = |phase| phases.contains(&phase);
        let tokens = [Phase::Prefill, Phase::Decode, Phase::Verify, Phase::Refill]
            .into_iter()
            .any(serves);

        let mut buckets = Vec::new();
        if serves(Phase::Prefill) || serves(Phase::Refill) {
            buckets.push(Shape::prefill(
                self.rows,
                self.tokens,
                Variant::PrefillOutput,
            ));
        }
        if serves(Phase::Decode) {
            buckets.push(Shape::decode(self.rows, Variant::DecodeSampled));
        }
        if serves(Phase::Verify) {
            let tokens = self.rows * (1 + self.drafts);
            buckets.push(Shape::prefill(self.rows, tokens, Variant::VerifySampled));
        }
        if serves(Phase::VisionEncode) {
            buckets.push(Shape::prefill(
                self.rows,
                self.tokens,
                Variant::ConditionEncode,
            ));
        }

        let denoises = serves(Phase::Denoise);
        let products: Vec<u64> = (1..=self.products().count() as u64)
            .map(|kind| 256 * kind)
            .collect();
        let media_bytes = if denoises {
            vec![
                (denoising(TEXT_ROWS, 0), 30_000, products.clone()),
                (denoising(TEXT_ROWS, 4096), 40_000, products),
            ]
        } else {
            Vec::new()
        };

        Capacity {
            phases: phases
                .iter()
                .fold(0, |bits, phase| bits | 1 << phase.index()),
            slots: self.slots,
            kv_cache: tokens.then(|| KvCacheConfig {
                block_size: self.block_size(),
                host_tier: self.host_units > 0,
                state_interval: self.state_interval,
                mooncake: self.remote.then(|| storage::connector::Settings {
                    master: "localhost:50051".parse().unwrap(),
                    metadata: storage::connector::Metadata::P2pHandshake,
                    transport: storage::connector::Transport::Tcp,
                    device: String::new(),
                    segment_gib: 0,
                }),
                peers: if self.peers {
                    vec!["peer-1:7600".to_owned()]
                } else {
                    Vec::new()
                },
                ..KvCacheConfig::default()
            }),
            cache_groups: if tokens {
                self.cache.clone()
            } else {
                Vec::new()
            },
            seeds: Vec::new(),
            buckets,
            rows: [self.slots; Phase::COUNT],
            max_model_len: 4096,
            denoise_steps: if denoises { STEPS } else { 0 },
            bos_token_id: 1,
            eos_token_ids: vec![EOS],
            token_history: 0,
            feature_bytes: if serves(Phase::VisionEncode) {
                FEATURE_BYTES
            } else {
                0
            },
            encoder_identity: if [Phase::TextEncode, Phase::VisionEncode, Phase::LatentEncode]
                .into_iter()
                .any(serves)
            {
                self.encoder_identity.clone()
            } else {
                String::new()
            },
            latent_seed: if denoises { LATENT_SEED } else { 0 },
            media_bytes,
            pipeline: self.media().then(pipeline),
            call_window: 2,
        }
    }

    /// Whether the deployment serves video requests: a group serves a phase of the media
    /// pipeline, and every group then states the pipeline.
    fn media(&self) -> bool {
        let pipeline = pipeline();
        self.groups
            .iter()
            .any(|(_, phases)| phases.iter().any(|phase| pipeline.phases.contains(phase)))
    }

    /// The products the deployment's session states classes of: the pipeline's for a video
    /// deployment, none for a token one.
    fn products(&self) -> impl Iterator<Item = nn::pipeline::ProductSpec> {
        let products = if self.media() {
            pipeline().products
        } else {
            Vec::new()
        };
        products.into_iter()
    }

    /// The deployment's block size: that of the cache group with the widest token rows, the
    /// largest of the model's that is not the draft cache's; `None` for a model of no cache.
    fn block_size(&self) -> Option<NonZeroU32> {
        self.cache
            .iter()
            .filter(|(kind, _)| !kind.is_draft())
            .map(|(_, block)| *block)
            .max()
            .and_then(NonZeroU32::new)
    }

    /// The scheduler's configuration: as many running requests as the groups hold slots for
    /// together in the phase fewest slots serve, the step's limits of the buckets, and the
    /// deployment's block size, which the groups' capacities state too (as
    /// `SchedulerConfig::of` resolves it from the engine's configuration).
    fn scheduler_config(&self) -> SchedulerConfig {
        let mut served: HashMap<Phase, u32> = HashMap::new();
        for (_, phases) in &self.groups {
            for phase in phases {
                *served.entry(*phase).or_default() += self.slots;
            }
        }
        let running = served.values().copied().min().unwrap_or(self.slots);
        SchedulerConfig {
            max_batch: NonZeroUsize::new(self.rows as usize).unwrap_or(NonZeroUsize::MIN),
            max_num_batched_tokens: self.tokens as usize,
            max_running_requests: running as usize,
            draft_tokens: self.drafts as usize,
            block_tokens: self
                .block_size()
                .map_or(proto::config::DEFAULT_BLOCK_SIZE, NonZeroU32::get),
            ..SchedulerConfig::default()
        }
    }

    /// The step values of the deployment: its chunk, step tokens (unless it leaves them to the
    /// widest captured bucket), calls and queue bound, of revision 0.
    fn tuning(&self) -> Tuning {
        Tuning {
            revision: 0,
            chunked_prefill_size: NonZeroUsize::new(self.chunk as usize),
            max_batch: NonZeroUsize::new(self.rows as usize).unwrap_or(NonZeroUsize::MIN),
            max_num_batched_tokens: self
                .states_budget
                .then(|| NonZeroUsize::new(self.tokens as usize))
                .flatten(),
            max_queued_requests: self.queued.and_then(NonZeroUsize::new),
            queue_depths: Default::default(),
            prefix_caching: true,
            host_tier: true,
            copy_budget: proto::config::DEFAULT_COPY_BUDGET,
            report_period: proto::config::DEFAULT_REPORT_PERIOD,
            watchdog_timeout: proto::config::DEFAULT_WATCHDOG_TIMEOUT,
            compare_timeout: proto::config::DEFAULT_COMPARE_TIMEOUT,
            nvtx_ranges: false,
        }
    }
}

/// The generation limits of every deployment: room for the images and scores a case asks for.
fn limits() -> GenerationLimits {
    GenerationLimits {
        features: GenerationFeatures::all(),
        max_vae_grid_tokens: 1 << 16,
        max_vit_grid_tokens: 1 << 16,
        max_latent_feature_bytes: 1 << 30,
        max_vision_feature_bytes: 1 << 30,
        encoder_cache_entries: 64,
        max_logprobs: 8,
    }
}

/// A captured denoising shape of the harness's video raster and timeline.
fn denoising(text_rows: u32, condition_rows: u32) -> Shape {
    Shape {
        rows: 1,
        tokens: 0,
        variant: Variant::Denoise,
        extent: Extent::Request {
            frames: FRAMES,
            width: WIDTH,
            height: HEIGHT,
            text_rows,
            condition_rows,
        },
    }
}

fn group_id(index: usize) -> GroupId {
    GroupId(index as u8)
}

fn index(at: usize) -> u8 {
    u8::try_from(at).unwrap()
}

/// What the fake's batches return beside its own rule (`testkit::executor::output_of`).
#[derive(Default)]
struct Rule {
    /// Per request, the tokens its sampling calls sample, in order; the fake's draw after them.
    text: HashMap<RequestId, VecDeque<u32>>,
    /// Per request, the failure every later call of it returns.
    failing: HashMap<RequestId, (ErrorCode, Option<String>)>,
    /// The drafts the drafter proposes with every token a call samples.
    proposals: u32,
}

/// A batch's output: the fake's rule, with each call of a failing request failed, a
/// verification's drafts accepted, a scripted token in place of the one the call samples, and the
/// drafter's proposals beside every sampled token.
fn respond(rule: &Mutex<Rule>, batch: &Batch) -> Result<BatchOutput, engine::Error> {
    let mut output = output_of(batch, None);
    let mut rule = lock(rule);
    let token = matches!(batch.phase, Phase::Prefill | Phase::Decode | Phase::Verify);

    for (index, call) in output.calls.iter_mut().enumerate() {
        let Some(id) = batch.requests.get(index).copied() else {
            continue;
        };
        if let Some((code, message)) = rule.failing.get(&id).cloned() {
            *call = CallOutput {
                status: CallStatus::Error,
                error_code: Some(code),
                error_message: message,
                tokens: Vec::new(),
                token_logprobs: Vec::new(),
                prompt_logprobs: Vec::new(),
                draft_token_ids: Vec::new(),
                ..call.clone()
            };
            continue;
        }
        if !token || call.tokens.is_empty() {
            continue;
        }

        let mut tokens = if batch.phase == Phase::Verify {
            drafts_of(batch, index)
        } else {
            Vec::new()
        };
        let sampled = rule
            .text
            .get_mut(&id)
            .and_then(VecDeque::pop_front)
            .unwrap_or(call.tokens[0]);
        tokens.push(sampled);
        call.token_logprobs = tokens.iter().map(|token| scored(*token)).collect();
        call.tokens = tokens;
        call.draft_token_ids = (0..rule.proposals).map(|draft| 900 + draft).collect();
    }
    Ok(output)
}

/// The drafts call `index` of a verification carried: its input ids after its first.
fn drafts_of(batch: &Batch, index: usize) -> Vec<u32> {
    let rows = &batch.columns.rows;
    let Some(call) = rows.calls.get(index) else {
        return Vec::new();
    };
    let offset = |row: u32| {
        let at = rows.query_offsets.get(row as usize).copied().unwrap_or(0);
        usize::try_from(at).unwrap_or(0)
    };
    let span = offset(call.rows.start)..offset(call.rows.end);
    let ids = batch.columns.input_ids.get(span).unwrap_or_default();
    ids.iter()
        .skip(1)
        .filter_map(|id| u32::try_from(*id).ok())
        .collect()
}

/// The scheduler of a [`Deployment`], the fake executors of its groups, the buffer pool, and the
/// record of everything the scheduler reported.
pub(crate) struct Harness {
    pub(crate) scheduler: Scheduler,
    /// The configuration the scheduler was built with.
    pub(crate) config: SchedulerConfig,
    /// The step values the scheduler reads at every step, as its engine shares them.
    tuning: Arc<ArcSwap<Tuning>>,
    pub(crate) pool: InProcessPool,
    pub(crate) fake: FakeExecutor,
    /// The scheduler's inbox, which the fake's outputs and stamped requests reach.
    inbox: Receiver<Message>,
    outputs: Receiver<Output>,
    /// Every output of the scheduler, in the order it was sent.
    record: Vec<Output>,
    rule: Arc<Mutex<Rule>>,
    /// The video requests submitted, whose `Event::Error` ends them.
    videos: HashSet<RequestId>,
    /// Every request submitted, in order.
    submitted: Vec<RequestId>,
    /// The steps over which completions are held back and then written in reverse
    /// ([`Harness::reorder`]).
    window: Option<u64>,
    /// The steps taken.
    pub(crate) steps: u64,
    /// The session's layout.
    layout: runtime::Open,
    /// Another engine epoch's session, of the same layout opened at its first use
    /// ([`Harness::elsewhere`]) or of another layout ([`Harness::beside`]): its end of its queue
    /// pair and the last number it drew.
    other: Option<(QueuePair, u64)>,
}

impl Harness {
    /// One pass of the engine's loop over the scheduler: the buffer pool's turn, every message of
    /// the scheduler's inbox applied, then the scheduler's step; returns whether the step formed
    /// a batch. Ends with the scheduler's side of A1 to A4 checked (SD §10, L5).
    ///
    /// # Panics
    ///
    /// When the fake recorded a breach of A1 to A4, naming each.
    pub(crate) fn step(&mut self) -> bool {
        if let Some(window) = self.window
            && self.steps.is_multiple_of(window)
        {
            // The completions held over the window fall due together, in reverse.
            self.pool.release();
            self.pool.withhold(|_, _| true);
        }
        self.pool.turn().unwrap();
        while let Ok(message) = self.inbox.try_recv() {
            self.scheduler.apply(message);
        }
        let formed = self.scheduler.step();
        self.steps += 1;
        self.record.extend(self.outputs.try_iter());

        let violations = self.fake.violations();
        assert!(
            violations.is_empty(),
            "the scheduler broke A1 to A4 by step {}: {violations:#?}",
            self.steps
        );
        formed
    }

    /// One step on a device that runs each batch through the step after the one that formed it:
    /// the batches on the device before this step are answered once it ends, so their outputs
    /// reach the scheduler at the next step, while the batches this step formed stay on the
    /// device until then. The case holds the fake's batches first ([`FakeExecutor::hold`]).
    /// Returns whether the step formed a batch.
    pub(crate) fn step_slow(&mut self) -> bool {
        let on_device = self.fake.held();
        let formed = self.step();
        self.fake.release(on_device);
        formed
    }

    /// Takes `steps` steps.
    pub(crate) fn run(&mut self, steps: u64) {
        for _ in 0..steps {
            self.step();
        }
    }

    /// Steps until `done` holds, at most `steps` steps; returns the steps taken.
    ///
    /// # Panics
    ///
    /// When `done` does not hold after `steps` steps.
    pub(crate) fn run_until(&mut self, steps: u64, mut done: impl FnMut(&mut Self) -> bool) -> u64 {
        for taken in 0..steps {
            if done(self) {
                return taken;
            }
            self.step();
        }
        assert!(
            done(self),
            "not reached in {steps} steps: {}",
            self.summary()
        );
        steps
    }

    /// What a failed case reports: per request submitted, its tokens and its last event; the
    /// requests the buffer pool owes answers to and withholds; the scheduler's failure.
    pub(crate) fn summary(&self) -> String {
        let mut lines = Vec::new();
        for id in &self.submitted {
            let events = self.events(*id);
            lines.push(format!(
                "request {}: {} tokens, last event {:?}",
                id.0,
                self.tokens(*id).len(),
                events.last()
            ));
        }
        let owed = self.pool.requests().iter().filter(|(_, id, _)| {
            !self
                .pool
                .completions()
                .iter()
                .any(|(_, answered, _)| answered == id)
        });
        let owed: Vec<String> = owed.map(|(_, _, request)| format!("{request:?}")).collect();
        lines.push(format!("owed by the buffer pool: {owed:#?}"));
        lines.push(format!("withheld: {}", self.pool.withheld()));
        lines.push(format!("failure: {:?}", self.failure()));
        lines.join("\n")
    }

    /// Steps until every request submitted has ended and the buffer pool holds nothing of any of
    /// them: no slot row and no private block. At most `steps` steps.
    ///
    /// # Panics
    ///
    /// As [`Harness::run_until`].
    pub(crate) fn settle(&mut self, steps: u64) -> u64 {
        self.run_until(steps, |harness| {
            harness
                .submitted
                .iter()
                .all(|id| harness.terminal(*id).is_some())
                && harness.held() == 0
        })
    }

    /// Submits a token request, as the daemon does (`Message::AddRequest`), under the event window
    /// the harness states as its consumer ([`EVENT_WINDOW`]).
    pub(crate) fn submit(&mut self, request: GenerationRequest) -> RequestId {
        self.add(request.into())
    }

    /// Hands the scheduler `request` as its warm-up request, as the engine does when admission
    /// opens (`Message::WarmUp`), under the harness's event window. The scheduler gives it its id.
    pub(crate) fn warm_up(&mut self, request: GenerationRequest) {
        self.apply(Message::WarmUp(Box::new(request), EVENT_WINDOW));
    }

    /// Submits a video request, as the daemon does (`Message::AddRequest`).
    pub(crate) fn submit_video(&mut self, request: DiffusionRequest) -> RequestId {
        self.add(proto::request::Request::Diffusion(request))
    }

    fn add(&mut self, request: proto::request::Request) -> RequestId {
        let id = request.request_id();
        if matches!(request, proto::request::Request::Diffusion(_)) {
            self.videos.insert(id);
        }
        self.submitted.push(id);
        self.apply(Message::AddRequest(request, EVENT_WINDOW));
        id
    }

    /// Applies a message from the engine or the daemon between two steps, as the loop applies each
    /// one it takes (`Cancel`, `Acknowledge`, `Consumed`, `Pause`, `Drain`, ...), and records what
    /// the scheduler reported at once.
    /// Replaces the step values by `change` of them, as the engine's `Tune` does: the scheduler
    /// reads them from its next step on.
    pub(crate) fn tune(&mut self, change: impl FnOnce(&mut Tuning)) {
        let mut tuning = Tuning::clone(&self.tuning.load());
        change(&mut tuning);
        self.tuning.store(Arc::new(tuning));
    }

    pub(crate) fn apply(&mut self, message: Message) {
        self.scheduler.apply(message);
        self.record.extend(self.outputs.try_iter());
    }

    /// The tokens request `id`'s next sampling calls sample, one each, in order; after them the
    /// fake's draw. A verification samples the token after the drafts it accepts.
    pub(crate) fn sample(&self, id: RequestId, tokens: &[u32]) {
        let mut rule = lock(&self.rule);
        rule.text.entry(id).or_default().extend(tokens);
    }

    /// Fails every later call of request `id` with `code` and, for an input error, the message
    /// the rank states; `None` stops failing it.
    pub(crate) fn fail_calls(&self, id: RequestId, failure: Option<(ErrorCode, Option<String>)>) {
        let mut rule = lock(&self.rule);
        match failure {
            Some(failure) => rule.failing.insert(id, failure),
            None => rule.failing.remove(&id),
        };
    }

    /// Reorders the buffer pool's completions within windows of `steps` steps: every completion
    /// is withheld, and at the start of each window those withheld are written together, in the
    /// reverse of the order they were produced. `None` writes them as they fall due again. Takes
    /// the buffer pool's withholding over while it is set.
    pub(crate) fn reorder(&mut self, steps: Option<u64>) {
        self.window = steps.map(|steps| steps.max(1));
        self.pool.reorder(steps.is_some());
        match steps {
            Some(_) => self.pool.withhold(|_, _| true),
            None => self.pool.release(),
        }
    }

    /// Submits `requests` on the queue pair of another engine epoch of the same layout (epoch 2,
    /// whose session is opened at the first use), takes one turn of the buffer pool, and returns
    /// the completions that turn wrote for them, in order: what another epoch does to the blocks
    /// the two share, which this scheduler learns only from the answers to its own requests (SD
    /// §4.5). The turn serves this scheduler's queue pair too, as a turn between two of its steps.
    /// The other epoch's workers stamp nothing: a request it submits names `after` itself.
    ///
    /// # Panics
    ///
    /// When the buffer pool refuses the session, or a request finds no room in its queue.
    pub(crate) fn elsewhere(&mut self, requests: &[runtime::Request]) -> Vec<Completion> {
        if self.other.is_none() {
            let layout = runtime::Open {
                epoch: 2,
                ..self.layout.clone()
            };
            let groups: Vec<GroupId> = layout.groups.iter().map(|group| group.id).collect();
            let (inbox, _) = unbounded();
            let executor = FakeExecutor::new(&groups, Vec::new(), inbox);
            let (_, mut ends) = self
                .pool
                .open(&layout, vec![executor])
                .unwrap_or_else(|refusal| panic!("the buffer pool refuses epoch 2: {refusal}"));
            self.other = Some((ends.remove(0), 0));
        }
        let Some((queue, number)) = self.other.as_mut() else {
            unreachable!("the other epoch's session is open");
        };

        let mut record = Vec::new();
        for request in requests {
            *number += 1;
            record.clear();
            request.encode(runtime::RequestId(*number), &mut record);
            assert!(queue.submit(&record).unwrap(), "epoch 2's queue is full");
        }
        self.pool.turn().unwrap();
        let mut completions = Vec::new();
        while queue.next_completion(&mut record).unwrap() {
            completions.push(Completion::decode(&record).unwrap().1);
        }
        completions
    }

    /// Opens the session of an engine epoch of another layout (epoch 2) beside this scheduler's,
    /// as an epoch update's target of another layout starts: its region of the memory space of
    /// this session's region `name` is a region of its own, `{name}-beside`, which takes `units`
    /// units of that space at its opening, as far as the space's free memory goes, and its slot
    /// rows lie in regions of its own (`slots-g-beside`, in memory of their own). Returns the
    /// number of `{name}-beside`; [`Harness::elsewhere`] submits on this epoch's queue pair from
    /// then on.
    ///
    /// # Panics
    ///
    /// When the session's layout states no region `name`, or the buffer pool refuses the session.
    pub(crate) fn beside(&mut self, name: &str, units: u32) -> u8 {
        self.open_target(name, units, false)
    }

    /// [`Harness::beside`] with the target's region `{name}-beside` over a memory space of its
    /// own, which holds `units` units from its opening: the target's groups state the memories of
    /// this session's, so a block of the target is valid in the memory of a group of this
    /// session, but none of its units lies in a region of this session.
    ///
    /// # Panics
    ///
    /// When the session's layout states no region `name`, or the buffer pool refuses the session.
    pub(crate) fn apart(&mut self, name: &str, units: u32) -> u8 {
        self.open_target(name, units, true)
    }

    /// Opens the session of [`Harness::beside`], its region `{name}-beside` over the space of
    /// `name` or, with `own_space`, over a space of its own.
    fn open_target(&mut self, name: &str, units: u32, own_space: bool) -> u8 {
        let renamed = format!("{name}-beside");
        let mut layout = runtime::Open {
            epoch: 2,
            ..self.layout.clone()
        };
        for slots in layout
            .regions
            .iter_mut()
            .filter(|region| region.name.starts_with("slots-") && region.name != name)
        {
            let own = format!("{}-beside", slots.name);
            slots.space.clone_from(&own);
            slots.name = own;
        }
        let region = layout
            .regions
            .iter_mut()
            .find(|region| region.name == name)
            .unwrap_or_else(|| panic!("the layout states no region {name}"));
        // Over the same space, the two epochs' regions share its memory.
        region.name.clone_from(&renamed);
        region.units = units;
        if own_space {
            region.space.clone_from(&renamed);
        }

        let groups: Vec<GroupId> = layout.groups.iter().map(|group| group.id).collect();
        let (inbox, _) = unbounded();
        let executor = FakeExecutor::new(&groups, Vec::new(), inbox);
        let (_, mut ends) = self
            .pool
            .open(&layout, vec![executor])
            .unwrap_or_else(|refusal| panic!("the buffer pool refuses epoch 2: {refusal}"));
        self.other = Some((ends.remove(0), 0));
        self.region(&renamed)
    }

    /// The number the buffer pool gives the session's region `name` (`kv-0`, `slots-0`,
    /// `buffers`, `host`, `latents`), which `Message::Resize` and the capacity words name.
    pub(crate) fn region(&mut self, name: &str) -> u8 {
        self.pool.pool().region_number(name).unwrap()
    }

    /// Resizes the session's region `name` to `units` as the engine does (`Message::Resize`),
    /// stepping until the reply, and returns the units the region reached.
    ///
    /// # Panics
    ///
    /// When no reply arrives within 20 steps.
    pub(crate) fn resize(&mut self, name: &str, units: u32) -> u32 {
        let region = self.region(name);
        let (reply, replied) = crossbeam_channel::bounded(1);
        self.apply(Message::Resize(region, units, reply));
        let mut reached = None;
        self.run_until(20, |_| {
            reached = replied.try_recv().ok();
            reached.is_some()
        });
        reached.unwrap_or_default()
    }

    /// Every output the scheduler sent, in order.
    pub(crate) fn outputs(&self) -> &[Output] {
        &self.record
    }

    /// Every event the scheduler sent, with its request, in the order it sent them.
    pub(crate) fn sent(&self) -> impl Iterator<Item = &(RequestId, Event)> {
        self.record.iter().flat_map(|output| match output {
            Output::Events(events) => events.as_slice(),
            _ => &[],
        })
    }

    /// The events the scheduler sent request `id`, in order.
    pub(crate) fn events(&self, id: RequestId) -> Vec<Event> {
        let of = |(to, event): &(RequestId, Event)| (*to == id).then(|| event.clone());
        self.sent().filter_map(of).collect()
    }

    /// The tokens the scheduler sent request `id` (`Event::TextToken`), in order.
    pub(crate) fn tokens(&self, id: RequestId) -> Vec<u32> {
        let token = |event: Event| match event {
            Event::TextToken { id, .. } => Some(id),
            _ => None,
        };
        self.events(id).into_iter().filter_map(token).collect()
    }

    /// The event that ended request `id`'s last life: its last event when that is a `Finished`,
    /// a `Rejected` or a `Migratable`, or for a video request an `Error`; `None` while it has not
    /// ended, a request resubmitted under its id after a `Migratable` included.
    pub(crate) fn terminal(&self, id: RequestId) -> Option<Event> {
        let video = self.videos.contains(&id);
        let last = self.events(id).pop()?;
        match last {
            Event::Finished { .. } | Event::Rejected { .. } | Event::Migratable { .. } => {
                Some(last)
            }
            Event::Error { .. } if video => Some(last),
            _ => None,
        }
    }

    /// The error the scheduler stopped with (`Output::Failed`), as it reads.
    pub(crate) fn failure(&self) -> Option<String> {
        self.record.iter().find_map(|output| match output {
            Output::Failed(error) => Some(error.to_string()),
            _ => None,
        })
    }

    /// Every statistics report the scheduler sent (`Output::Stats`), in order.
    pub(crate) fn reports(&self) -> Vec<&SchedulerStats> {
        let reports = self.record.iter().filter_map(|output| match output {
            Output::Stats(stats) => Some(stats),
            _ => None,
        });
        reports.collect()
    }

    /// Takes steps until the scheduler sent its next statistics report, which it sends once a
    /// second has passed since the last, and returns the calls it left out of a step because
    /// their blocks had not arrived (`calls_without_blocks`), over every report it sent.
    ///
    /// # Panics
    ///
    /// When no report is sent within five seconds.
    pub(crate) fn calls_without_blocks(&mut self) -> u64 {
        self.next_report();
        let reports = self.reports().into_iter();
        reports.map(|report| report.calls_without_blocks).sum()
    }

    /// Takes steps, 100 ms apart, until the scheduler sent its next statistics report, which it
    /// sends once a second has passed since the last, and returns that report.
    ///
    /// # Panics
    ///
    /// When no report is sent within five seconds.
    pub(crate) fn next_report(&mut self) -> SchedulerStats {
        let sent = self.reports().len();
        let started = Instant::now();
        while self.reports().len() == sent {
            assert!(
                started.elapsed() < Duration::from_secs(5),
                "no report was sent"
            );
            std::thread::sleep(Duration::from_millis(100));
            self.step();
        }
        let reports = self.reports();
        reports[sent].clone()
    }

    /// The requests the buffer pool took whose completions it has not written: what the
    /// scheduler is owed.
    pub(crate) fn outstanding(&self) -> usize {
        let written: HashSet<u64> = self
            .pool
            .completions()
            .iter()
            .map(|(_, id, _)| id.0)
            .collect();
        let taken = self.pool.requests().iter();
        taken.filter(|(_, id, _)| !written.contains(&id.0)).count()
    }

    /// The batches the scheduler handed to the executor, in order.
    pub(crate) fn batches(&self) -> Vec<Batch> {
        self.fake.submitted()
    }

    /// The batches that carried a call of request `id`, in order.
    pub(crate) fn batches_of(&self, id: RequestId) -> Vec<Batch> {
        let carries = |batch: &Batch| batch.requests.contains(&id);
        self.batches().into_iter().filter(carries).collect()
    }

    /// The batches of `phase` that carried a call of request `id`, in order.
    pub(crate) fn batches_of_phase(&self, id: RequestId, phase: Phase) -> Vec<Batch> {
        let batches = self.batches_of(id).into_iter();
        batches.filter(|batch| batch.phase == phase).collect()
    }

    /// The requests the buffer pool took, in order.
    pub(crate) fn taken(&self) -> Vec<runtime::Request> {
        let taken = self.pool.requests().iter();
        taken.map(|(_, _, request)| request.clone()).collect()
    }

    /// The completions the buffer pool wrote, each with the request it answers, in order.
    pub(crate) fn answers(&self) -> Vec<(runtime::Request, Completion)> {
        let requests: HashMap<u64, &runtime::Request> = self
            .pool
            .requests()
            .iter()
            .map(|(_, id, request)| (id.0, request))
            .collect();
        let answered = self.pool.completions().iter();
        answered
            .filter_map(|(_, id, completion)| {
                let request = requests.get(&id.0)?;
                Some(((*request).clone(), completion.clone()))
            })
            .collect()
    }

    /// The owner request `id` holds on `group`: the owner whose `Attached` named the slot of the
    /// request's last batch on the group. `None` before a batch of it ran there.
    pub(crate) fn owner(&self, id: RequestId, group: GroupId) -> Option<OwnerId> {
        let slot = self.batches_of(id).into_iter().rev().find_map(|batch| {
            let row = batch.requests.iter().position(|of| *of == id)?;
            (batch.group == group).then(|| {
                let token = batch.columns.rows.calls.get(row).map(|call| call.slot);
                token.or_else(|| batch.media.get(row).map(|media| media.slot))
            })?
        })?;
        self.fake
            .mapped()
            .into_iter()
            .rev()
            .find_map(|mapped| match mapped {
                (
                    of,
                    _,
                    _,
                    runtime::Request::Attach { owner, .. },
                    Completion::Attached { slot: at },
                ) if of == group && at == slot => Some(owner),
                _ => None,
            })
    }

    /// Sends `Message::Drain` for `MigrationCause::Drain`, as the engine drains its serving epoch,
    /// with `deadline` and `shrink` (give the device memory back), and returns the receiver of
    /// its reply.
    pub(crate) fn drain(&mut self, deadline: Instant, shrink: bool) -> Receiver<()> {
        let (reply, replied) = crossbeam_channel::bounded(1);
        self.apply(Message::Drain(
            MigrationCause::Drain,
            deadline,
            shrink,
            reply,
        ));
        replied
    }

    /// The media rows of request `id`'s calls of `phase`, in the order their batches were
    /// submitted.
    pub(crate) fn media_rows(&self, id: RequestId, phase: Phase) -> Vec<MediaRow> {
        let batches = self.batches_of(id).into_iter();
        let of_phase = batches.filter(|batch| batch.phase == phase);
        of_phase
            .filter_map(|batch| {
                let row = batch.requests.iter().position(|of| *of == id)?;
                batch.media.get(row).cloned()
            })
            .collect()
    }

    /// The denoising steps video request `id` completed, as its last progress event states them
    /// (`MediaProgress::completed_steps`); 0 before its first.
    pub(crate) fn denoised(&self, id: RequestId) -> u32 {
        let mut steps = self.events(id).into_iter().filter_map(|event| match event {
            Event::MediaProgress {
                completed_steps, ..
            } => Some(completed_steps),
            _ => None,
        });
        steps.next_back().unwrap_or(0)
    }

    /// Holds every batch from now on and lets one end per step, oldest first, until video request
    /// `id` reported `steps` denoising steps completed: the batches formed meanwhile and not let
    /// end, the request's next denoising steps among them while it has steps left, stay on the
    /// device until the case releases them (`fake.release`, `fake.resume`).
    ///
    /// # Panics
    ///
    /// When the request has not completed `steps` steps within 60 steps.
    pub(crate) fn denoise_and_hold(&mut self, id: RequestId, steps: u32) {
        self.fake.hold();
        for _ in 0..60 {
            if self.denoised(id) >= steps {
                return;
            }
            self.fake.release(1);
            self.step();
        }
        panic!(
            "request {} did not complete {steps} denoising steps: {}",
            id.0,
            self.summary()
        );
    }

    /// The keys of the `Commit`s of `owner` the buffer pool took, in order.
    pub(crate) fn committed(&self, owner: OwnerId) -> Vec<Key> {
        let commits = self
            .taken()
            .into_iter()
            .filter_map(|request| match request {
                runtime::Request::Commit { owner: of, key, .. } if of == owner => Some(key),
                _ => None,
            });
        commits.collect()
    }

    /// The first cache position request `id`'s first prefill after its `skip` earlier batches
    /// wrote: where the computation of that life of it started, its prefix hit.
    pub(crate) fn first_written(&self, id: RequestId, skip: usize) -> Option<i32> {
        let batches = self.batches_of(id).into_iter().skip(skip);
        let prefill = batches
            .into_iter()
            .find(|batch| batch.phase == Phase::Prefill)?;
        let row = prefill.requests.iter().position(|of| *of == id)?;
        let rows = &prefill.columns.rows;
        let first = *rows
            .query_offsets
            .get(rows.calls.get(row)?.rows.start as usize)?;
        rows.cache_positions
            .get(usize::try_from(first).ok()?)
            .copied()
    }

    /// Whether request `id`'s life ended with `Finished` for `reason`.
    pub(crate) fn finished_with(&self, id: RequestId, reason: FinishReason) -> bool {
        matches!(self.terminal(id), Some(Event::Finished { reason: of, .. }) if of == reason)
    }

    /// The blocks the buffer pool holds for requests: slot rows and private blocks, in every
    /// region.
    pub(crate) fn held(&mut self) -> u32 {
        let counts = self.pool.pool().counts();
        let held = counts
            .iter()
            .filter(|((_, _, state), _)| *state == BlockState::Private);
        held.map(|(_, blocks)| *blocks).sum()
    }
}

/// The span of request `id`'s row in token batch `batch`, in the batch's token columns: its
/// first token to its end. `None` when the batch carries no token row of it.
pub(crate) fn row_span(batch: &Batch, id: RequestId) -> Option<Range<usize>> {
    let rows = &batch.columns.rows;
    let row = batch.requests.iter().position(|of| *of == id)?;
    let call = rows.calls.get(row)?;
    let offset = |at: u32| {
        let offset = rows.query_offsets.get(at as usize)?;
        usize::try_from(*offset).ok()
    };
    Some(offset(call.rows.start)?..offset(call.rows.end)?)
}

/// The cache positions request `id`'s row in token batch `batch` writes, `-1` where it writes
/// none; empty when the batch carries no token row of it.
pub(crate) fn row_positions(batch: &Batch, id: RequestId) -> Vec<i32> {
    let positions = &batch.columns.rows.cache_positions;
    let span = row_span(batch, id).and_then(|span| positions.get(span));
    span.unwrap_or_default().to_vec()
}

/// The prompt of request `id` of `tokens` tokens: ids below the fake's sampled window, which no
/// other request's prompt of that length shares.
pub(crate) fn prompt(id: u64, tokens: u32) -> Vec<u32> {
    (0..tokens)
        .map(|at| 10 + (id as u32 * 97 + at) % 900)
        .collect()
}

/// Token request `id` that may generate `max_tokens` tokens, whose prompt is request `of`'s
/// prompt of `prompt_tokens` tokens: it reads the blocks that request committed.
pub(crate) fn sharing(
    id: u64,
    of: u64,
    prompt_tokens: u32,
    max_tokens: usize,
) -> GenerationRequest {
    GenerationRequest {
        prompt_token_ids: prompt(of, prompt_tokens),
        ..token(id, prompt_tokens, max_tokens)
    }
}

/// Token request `id` of a prompt of `prompt_tokens` tokens ([`prompt`]) that may generate
/// `max_tokens` tokens, sampled greedily, with the default cache policy.
pub(crate) fn token(id: u64, prompt_tokens: u32, max_tokens: usize) -> GenerationRequest {
    GenerationRequest {
        request_id: RequestId(id),
        prompt_token_ids: prompt(id, prompt_tokens),
        multimodal_inputs: Default::default(),
        sampling: Default::default(),
        max_und_tokens: max_tokens,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: Default::default(),
        include_stop_token: false,
        generated_token_ids: Vec::new(),
        adapter: 0,
    }
}

/// Chat request `id`: [`token`] with one image in the middle of its prompt, whose vision encoder
/// writes a block of `image_tokens` cache rows.
pub(crate) fn chat_image(
    id: u64,
    prompt_tokens: u32,
    image_tokens: u32,
    max_tokens: usize,
) -> GenerationRequest {
    let mut request = token(id, prompt_tokens, max_tokens);
    let mut digest = [0; 32];
    digest[..8].copy_from_slice(&id.to_le_bytes());
    request.multimodal_inputs.images = vec![ImageInput {
        digest,
        encoded: b"image".to_vec(),
        position: prompt_tokens / 2,
        num_positions: image_tokens,
        encoders: vec![ImageEncoderInput {
            encoder: ImageEncoder::Vision,
            num_cache_tokens: Some(image_tokens),
            max_cache_tokens: None,
            height: None,
            width: None,
        }],
    }];
    request
}

/// Video request `id` of `units` media units, a prompt of 40 tokens and no conditions, at the
/// raster and timeline a video deployment captured.
pub(crate) fn video(id: u64, units: u32) -> DiffusionRequest {
    DiffusionRequest {
        request_id: RequestId(id),
        prompt_token_ids: prompt(id, 40),
        conditioning: proto::video::Conditions {
            prompt_tokens: 40,
            ..proto::video::Conditions::default()
        }
        .encode(),
        conditions: Vec::new(),
        media: Vec::new(),
        priority: 0,
        sampling: DiffusionSamplingParams {
            num_frames: FRAMES,
            video_units: units,
            seed: id,
            width: WIDTH,
            height: HEIGHT,
            condition_rows: 0,
        },
        trajectory: Trajectory::default(),
        encoder_inputs: None,
    }
}
