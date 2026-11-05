//! How the parked main process is built from one epoch's build config.
//!
//! `src/engine_main.rs` is the main process itself: the control thread, the device executors, the
//! schedulers and the hot update. Everything it needs *before* it can start a thread comes from an
//! epoch's configuration - which devices, which buckets are captured, how much of each device the
//! graphs may map, which blocking pools exist, which Program Server socket and targets this rank
//! speaks to - and none of that belongs in `daemon::epoch::BuildConfig`: the config fixes the epoch
//! and must not learn how the executor assembles its devices, its buckets or its graphs. It also
//! does not hold every input, because half of what the main process needs is a property of the node
//! it runs on (`EngineArgs`) or of the devices it measured.
//!
//! So the derivation lives here, on the consumer's side of the daemon/engine seam: [`EpochPlan`] is
//! an epoch's build config resolved against the node's arguments, and the accessors below are the
//! whole of what the main process asks of it. [`EpochPlan::runners`] is the one place in the tree
//! that names a concrete model class (§4.3): the capability dispatch every other line obeys has
//! exactly one entry, and this is it.
//!
//! The capacity plan - what the rank's resident memory is - is not here: it is resolved from a
//! measured device against the reported capacity, which is the capacity negotiation's own area
//! (`executor::memory::Memory::new`'s caller).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context as _, Result, bail};
use executor::blocking::{BlockingKind, LaneLimits, Limits};
use executor::capacity::{graph_storage_budget_bytes, graph_storage_bytes, quota};
use executor::device::GraphStorage;
use executor::memory::Memory;
use executor::plan::{Buckets, Variant};
use executor::runner::{Binding, NamedRunner, RunnerId};
use executor::runners::causal_lm::{CausalLmRunner, Geometry, KvProvider, Shared};
use executor::runners::resident::ResidentCache;
use executor::stream::EventPool;
use ipc::PhaseId;
use ops::DType;
use ops::Generation;
use platform::DeviceId;
use program_server::Target;
use proto::request::{GenerationFeatures, GenerationLimits, GraphPolicy};

use crate::engine_main::Launch;

/// One epoch's build config, resolved against the node the main process runs on.
/// The catalogue of prefill buckets a rank captures, and where its bound came from (§4.5).
///
/// The bound is the largest token count any captured bucket holds, which is also the ceiling a
/// policy's `max_num_batched_tokens` is checked against (§4.4): a policy that wants a wider step
/// than the rank's graphs can serve is refused, and the refusal is only actionable when it says
/// which figure came from which place. `stated` separates the deployment's own
/// `cuda_graph.prefill_token_sizes` from the build's default catalogue, because a deployment that
/// configured none cannot find its bound in its own launch file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PrefillCatalogue {
    /// The largest prefill token count any captured bucket holds.
    pub(crate) bound: u32,
    /// Whether the deployment stated the sizes rather than taking the build's default catalogue.
    pub(crate) stated: bool,
}

/// The model class a checkpoint declares, and the deployment this build serves it as (4.3, 9).
///
/// **One place maps a checkpoint's `architectures` entry to a class.** The class is the checkpoint's
/// own statement (`config.json`), not the deployment's name for it, and every other site branches on
/// this value rather than on a string: a second `match` on a model name would be a second statement
/// of which deployments this build serves, and the two would drift the first time one changed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ModelClass {
    /// A causal language model: prefill and decode forwards over a paged key/value cache.
    CausalLm,
    /// The media pipeline: an encode of a request's conditions, denoising steps of a latent layout,
    /// and a clip decoded from latents. It keeps no paged key/value cache.
    Media,
}

impl ModelClass {
    /// The class `checkpoint` declares, or the refusal that names what it declares.
    fn of(checkpoint: &std::path::Path) -> Result<Self> {
        let declared = architectural_class(checkpoint)?;
        match declared.as_str() {
            "Qwen3MoeForCausalLM" => Ok(Self::CausalLm),
            "MiniMaxH3ModularPipeline" => Ok(Self::Media),
            other => Err(anyhow::anyhow!(
                "{} declares model class {other}, which this build cannot serve: it serves \
                 Qwen3MoeForCausalLM as a causal language model and MiniMaxH3ModularPipeline as the \
                 media pipeline",
                checkpoint.display()
            )),
        }
    }
}

/// What a rank of a class holds, once its model is loaded and its resident memory opened.
///
/// The class was resolved at the same moment, so the value carries both: a caller that has a
/// [`Served`] cannot ask it for a figure of the other class, which a bare `Arc<…>` would allow.
pub(crate) enum Served {
    /// The causal language model, one `Arc` per rank: every device's runner reads the same weights.
    Text(Arc<qwen3_moe::Qwen3Moe>),
    /// The media pipeline, whose blocks the runners take one `Arc` each.
    Media(Arc<minimax_h3::load::MinimaxH3>),
}

impl Served {
    /// The checkpoint-named class this value was loaded as.
    pub(crate) fn class(&self) -> ModelClass {
        match self {
            Self::Text(_) => ModelClass::CausalLm,
            Self::Media(_) => ModelClass::Media,
        }
    }
}

pub(crate) struct EpochPlan {
    /// The epoch's build config, which the daemon wrote (§7.2).
    build: daemon::epoch::BuildConfig,
    /// The checkpoint this rank serves.
    checkpoint: PathBuf,
    /// This rank's devices, as the executor's names for them.
    devices: Vec<DeviceId>,
    /// The Program Server's socket.
    program_server: PathBuf,
    /// The targets this rank resolves a module for (§5.3).
    targets: Vec<Target>,
    /// The bucket catalog every device of this rank captures.
    buckets: Buckets,
    /// This rank's ordinal inside its group, which the storage claims and the cache plan's
    /// agreement with its group are keyed by.
    rank: u32,
    /// The world the epoch executes, which keys the storage it creates.
    world: u64,
    /// Ranks in the deployment, whose minimum the cache plan's block size and slot count are
    /// agreed by (§4.1).
    world_size: usize,
    /// Batches the rank's schedulers keep in flight, which the capacity plan relays through.
    queue_depth: usize,
    /// This rank's executor group, which the rank facts name as its identity.
    group: String,
    /// The watchdog the launch file states apart from the build config (§2.4), when it states one.
    watchdog_override: Option<Duration>,
    /// This rank's part of the epoch's parallel shape (§4.3, §5.1), resolved with the plan.
    placement: Placement,
}

/// The refusal a stated graph tier earns, if any (§4.4, §4.5).
///
/// The two bounds are not the same kind of thing, and this is where that is decided:
///
/// - **The token bound is a contract a policy is checked against.** `ceilings_of` gives the
///   scheduler's policy `max_num_batched_tokens` from the catalogue's captured token bound
///   (`crates/engine/src/scheduler/policy.rs`), and a policy that declares more is refused by the
///   mechanism. A token tier capture drops therefore changes what a deployment's policy may
///   declare, and the deployment would find that out at its precopy - after weights, from a
///   sentence about two numbers its own file does not contain. It is refused here instead.
/// - **The row bound only sizes a batch.** A decode tier above the request bound changes no
///   policy's declaration: `Scheduler::new` lowers the limit, names both figures and keeps serving
///   (`crates/engine/src/scheduler/run.rs`), which is what a deployment whose limits move in place
///   relies on - its tiers outlive one setting of the bound. It refuses nothing.
///
/// A zero states no size at all and is dropped, as it always was.
fn tier_refusal(
    decode: Option<&[u32]>,
    rows: u32,
    prefill: &[u32],
    longest: u32,
) -> Option<String> {
    // Named for the reader, decided as below: rows within the bound or not, this is not a refusal.
    let _ = (decode, rows);
    let stated = prefill
        .iter()
        .copied()
        .find(|size| *size != 0 && *size > longest)?;
    Some(format!(
        "cuda_graph_config.prefill_token_sizes states {stated}, above the {longest} that \
         max_model_len admits: the bucket would be captured and never filled, and the catalogue a \
         policy's token budget is checked against is the one capture keeps - so this rank would \
         refuse a policy against a bound the deployment never named"
    ))
}

impl EpochPlan {
    /// The epoch `build` builds, with this node's arguments resolved against it.
    ///
    /// The devices and the Program Server socket are the node's, not the epoch's: the daemon
    /// starts every rank of a node with the ordinals that rank serves, and a main process that
    /// guessed its devices from the epoch's parallelism would capture graphs on another rank's
    /// device.
    pub(crate) fn of(launch: &Launch, build: &daemon::epoch::BuildConfig) -> Result<Self> {
        let placement = Placement::of(launch, build)?;
        if launch.devices.is_empty() {
            bail!(
                "rank {} of group {} was started with no device to serve",
                launch.rank,
                launch.group
            );
        }
        let mut devices = Vec::with_capacity(launch.devices.len());
        for ordinal in &launch.devices {
            let ordinal = i32::try_from(*ordinal).with_context(|| {
                format!("device ordinal {ordinal} is outside the driver's range")
            })?;
            devices.push(DeviceId(ordinal));
        }
        // §4.5: a rank without graphs serves nothing, so `off` is a configuration this build
        // cannot honour - there is no eager path to fall back to.
        if build.cuda_graph.policy == GraphPolicy::Off {
            bail!(
                "cuda_graph.policy is off: this build serves causal-language batches from captured \
                 graphs only, so an epoch that captures none cannot run"
            );
        }
        let rows: Option<Vec<u32>> = build
            .cuda_graph
            .decode_batch_sizes
            .as_ref()
            .map(|sizes| sizes.iter().map(|size| size.get()).collect());
        let tokens: Option<Vec<u32>> = build
            .cuda_graph
            .prefill_token_sizes
            .as_ref()
            .map(|sizes| sizes.iter().map(|size| size.get()).collect());
        // The bounds a batch may reach: requests in flight, and the context a prefill chunk
        // cannot exceed. A bucket above either would be captured and never filled.
        let rows_bound = u32::try_from(build.max_running_requests.get())
            .context("more request slots than the protocol addresses")?;
        let tokens_bound = match build.max_model_len {
            Some(length) => length.get(),
            None => u32::MAX,
        };
        // §4.5: a stated token tier above the bound that admits it is refused here, before the
        // catalogue is built and long before any weight is mapped; a stated decode tier above the
        // request bound is the clamp in `Scheduler::new`, not a refusal. `tier_refusal` is where
        // that asymmetry is stated and argued.
        if let (Some(sizes), Some(length)) = (&tokens, build.max_model_len) {
            if let Some(refusal) = tier_refusal(rows.as_deref(), rows_bound, sizes, length.get()) {
                bail!("{refusal}");
            }
        }
        let buckets = Buckets::for_batch(
            rows.as_deref(),
            tokens.as_deref(),
            rows_bound,
            tokens_bound,
            true,
        )?;
        if buckets.is_empty() {
            bail!("the build config captures no graph bucket");
        }
        let targets = targets(&devices)?;
        Ok(Self {
            build: build.clone(),
            checkpoint: launch.model.clone(),
            devices,
            program_server: PathBuf::from(&launch.program_server),
            targets,
            buckets,
            rank: launch.rank,
            world: launch.world,
            world_size: launch.world_size as usize,
            queue_depth: launch.queue_depth as usize,
            group: launch.group.clone(),
            watchdog_override: launch.watchdog_timeout_s.map(Duration::from_secs),
            placement,
        })
    }

    /// The catalogue of prefill buckets this rank captures, and whether the deployment stated it.
    ///
    /// `Buckets::bounds` answers the largest prefill token count among the buckets this rank
    /// captures, skipping media computations whose extents are not a language-model batch size, so
    /// the figure is the ceiling the mechanism admits and the one a policy's limit is checked
    /// against - not the build config's, which may be wider than the bound that dropped it.
    pub(crate) fn prefill_catalogue(&self) -> PrefillCatalogue {
        let (_, bound) = self.buckets.bounds();
        PrefillCatalogue {
            bound,
            stated: self.build.cuda_graph.prefill_token_sizes.is_some(),
        }
    }

    /// The blocking pools' admission bounds, one lane per kind (§2.4).
    ///
    /// The build config sizes no pool - a blocking lane's width follows the work it carries, not
    /// the epoch - so the bounds are derived from what the node has: the checkpoint's own reader
    /// count for the lane that maps weights and installs programs, and the host's parallelism for
    /// the lanes that read transfers, reclaim storage and handle media. A lane whose bound is
    /// narrower than the work in flight queues it; it never drops it.
    pub(crate) fn blocking_limits(&self) -> Limits {
        let available = std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(4);
        let shared = (available / 2).clamp(1, 8);
        let lanes = BlockingKind::ALL.map(|kind| match kind {
            BlockingKind::ModuleLoad => {
                let workers = self
                    .build
                    .load
                    .threads
                    .map_or_else(|| shared.clamp(1, 8), |threads| threads.get() as usize);
                LaneLimits {
                    workers,
                    capacity: workers.saturating_mul(2),
                }
            }
            _ => LaneLimits {
                workers: shared,
                capacity: shared.saturating_mul(4),
            },
        });
        Limits::new(lanes)
    }

    /// The Program Server's socket, which the rank's program client connects to.
    pub(crate) fn program_server(&self) -> &Path {
        &self.program_server
    }

    /// This rank's devices, in the order the daemon named them.
    pub(crate) fn devices(&self) -> &[DeviceId] {
        &self.devices
    }

    /// The device's completion events: one pool per rank, shared by its executors (§2.4).
    pub(crate) fn events(&self) -> Arc<EventPool> {
        EventPool::new()
    }

    /// The device's graph storage: the buckets it captures and the pools' reservations (4.1).
    ///
    /// The figures are capacity planning, so the derivation stays in the executor's capacity
    /// module and this reads its two answers: the graph reservation is the share of a device a
    /// rank may map for captured graphs, and the budget is the rank's quota of the device. The
    /// scratch reservation is a fraction of the graph pool, which the paired forward's page set and
    /// the shadow pair's region are carved from (7.3). The resident pool is reserved by whoever
    /// carves it - `Memory`, through `Executor::pools()` - so the graph storage asks for none: a
    /// second figure here would be a second claim on the same bytes.
    /// `snapshot` is the state group's snapshot (`CachePlan::state_snapshot_bytes`, design 4.5),
    /// which the scratch pool has to hold beside the shadow region. The **reservation** is one
    /// figure - the pool is reserved once at construction, and a second reservation of the same
    /// pool kind is a no-op the allocator refuses - and the storage states both tenants' shares, so
    /// the paired forward's twins take `shadow` and `Snapshot::allocate` keeps the rest:
    /// `shadow + snapshot == scratch`, and the order the two take their bytes in does not matter.
    pub(crate) fn graph(
        &self,
        device: DeviceId,
        resident: u64,
        snapshot: u64,
        streams: usize,
    ) -> Result<GraphStorage> {
        let handle = platform::Device::new(device)
            .with_context(|| format!("opening device {}", device.0))?;
        let storage = handle.memory_info()?;
        let total = storage.total;
        // The graph reservation is the design's share of the whole device (4.1), so its base is
        // the device's total and not what is left of it. Two pools are reserved and, per
        // `online-operations.md`, that is one for the serving generation and one for a preparation
        // beside it - the successor recaptures on the same stream, so its plans map beside the
        // serving generation's and its buckets stay addressable through a revert.
        //
        // A reservation is virtual address space, not a booking (`capacity::graph_storage_bytes`'s
        // own contract): the bytes are booked when a graph maps, so reserving the second pool costs
        // a device nothing until an update needs it, and the refusal for a preparation with no room
        // comes from the ledger's book at capture time - the update's failure, not the rank's.
        let bytes = graph_storage_bytes(i64::try_from(total)?, true, false);
        if bytes == 0 {
            bail!("device {} reports no memory to capture graphs in", device.0);
        }
        // The shadow region §7.3 step 2 runs a replaced operator in, plus the state group's
        // snapshot §4.5 copies units through: one scratch pool, sized once, never allocated per
        // step or per batch.
        let shadow = graph_storage_budget_bytes(i64::try_from(total)?) / 8;
        let scratch = shadow.checked_add(snapshot).with_context(|| {
            format!(
                "device {} reserves a {shadow}-byte shadow region and a {snapshot}-byte state snapshot, more than one scratch pool holds",
                device.0
            )
        })?;
        // Every pool on this device is booked against `budget`, so `budget` is what the device has
        // left to give: the epoch's weights are device-resident retained storage, mapped before
        // this call (4.1), and an allowance drawn from the total would let the graph, scratch and
        // resident pools map past the device by exactly the model's size. The deployment's
        // fraction still caps the pools' share of the whole device, and the smaller of the two
        // rules is the allowance: a deployment that asks for a smaller share gets it, and a device
        // already holding the weights gets only what remains.
        let budget = quota(total, self.build.mem_fraction_static).min(storage.free);
        // What a rank must be able to map to serve at all, which is every pool the allocator
        // books at its first capture: the graph range in full and the scratch. The graph range is
        // mapped whole and not a pool at a time - `arena::Pools::new` divides the range among the
        // streams and `Pools::pool` allocates a stream's share in one buffer - so a rank that
        // holds room for one of the two pools and not both is refused here
        // rather than at its first capture, after the model load. The resident pool is a
        // reservation of addresses and not of storage: its bytes are the KV cache the capacity
        // plan fitted into the same free bytes below, and they map through the Storage Server's
        // ledger, so charging them here as well would refuse a deployment whose cache fills the
        // device's free bytes - which is what the fit is meant to do.
        let needs = bytes + scratch;
        if budget < needs {
            bail!(
                "device {} has {budget} bytes left of {total} once the model is mapped, at \
                 mem_fraction_static {}, less than the {needs} bytes of graph range and scratch a \
                 rank maps to serve",
                device.0,
                self.build.mem_fraction_static,
            );
        }
        Ok(GraphStorage {
            bytes,
            budget,
            // The range is divided among the streams that will map it, and `streams` is that
            // count: one per runner the device serves plus one for a preparation, which walks its
            // own generation on a stream of its own. `bytes` above is the design's figure for the
            // device - one graph pool, two while a preparation is alive - so a stream's share is
            // `bytes / streams` and a device that serves more runners takes a smaller share each.
            // The figure is NOT scaled by the runner count: 10% of the device is the budget for a
            // device's graphs, and a media rank with five runners may not claim 50%.
            //
            // A count short of the streams that actually map is not a tuning error: the first
            // streams take the range whole and the next one is refused at `Pools::pool` with
            // "cannot give out N more", which reads as an allocator fault rather than a plan.
            // `Executor::spawn` therefore refuses a device whose runners outnumber the streams the
            // range was divided for, by name, before any capture.
            streams,
            scratch,
            // The shadow region alone, which the paired forward's twins allocate; `scratch` keeps
            // the reservation that also holds the state group's snapshot (7.3, 4.5).
            shadow,
            // The KV pool: the units the capacity plan fitted, each the size of a cache unit. This
            // rank keeps its keys and values in the process - that is what a prefix cache is - so
            // the figure is the plan's and not zero. A pool whose figure is zero is not reserved,
            // and a KV provider that asks for it is refused by name.
            resident,
        })
    }

    /// How long a job may run without rank progress before it is reported `Stalled` (§2.4): the
    /// launch file's when it states one apart from the build config, and the build config's else.
    pub(crate) fn watchdog_timeout(&self) -> Duration {
        self.watchdog_override
            .unwrap_or(self.build.worker.watchdog_timeout)
    }

    /// The launch-file and build-config facts a scheduler derives an epoch's configuration from.
    ///
    /// Every fact this process can state is read from the source that owns it: the launch file the
    /// daemon wrote (`group`, `rank`, `world`, `world_size`, `devices`, the watchdog), the epoch's
    /// build config (`dtype`, `max_running_requests`, `max_model_len`, the host cache), the captured
    /// catalog (`max_batch`, `max_num_batched_tokens`, `graph_tiers`; a batch larger than the
    /// largest bucket is refused before a kernel runs, so the catalog is the admission bound), and,
    /// for the page size and the cache dtype, what was opened rather than what was asked for.
    ///
    /// Two facts are answered only as far as a launch file can answer them. `hosts` is one host
    /// name per rank in rank order: a world of one states this node's own, because one node is one
    /// host and the name is read from the machine rather than invented, while a world of more ranks
    /// leaves the fact absent - the real names are the deployment's, and a launch file, which is per
    /// node, does not carry them (`Launch::peers` is this node's peer addresses).
    ///
    /// The rest stay `None`, and that is the answer rather than a gap to fill with a guess: `bos`,
    /// `eos`, `end_of_image` and `generation_limits` are the checkpoint's tokenizer and generation
    /// facts, which neither the launch file nor the model's configuration states.
    /// `engine::EngineConfig::of` refuses each absent fact by name, so a deployment that needs one
    /// fails with the fact it is missing rather than planning against a made-up value.
    pub(crate) fn rank_facts(&self, memory: &Memory, served: &Served) -> Result<engine::RankFacts> {
        let Served::Text(model) = served else {
            return self.media_rank_facts();
        };
        let (rows, tokens) = self.buckets.bounds();
        let control = checkpoint_tokens(&self.checkpoint)?;
        // The graph sizes this rank captures, which is the tier a plan's batch must fit (§4.5).
        let graph_tiers = self
            .buckets
            .iter()
            .map(|bucket| bucket.tokens)
            .collect::<std::collections::BTreeSet<u32>>()
            .into_iter()
            .collect();
        Ok(engine::RankFacts {
            model: Some(self.checkpoint.display().to_string()),
            dtype: Some(self.build.dtype.clone()),
            world: Some(self.world),
            group: Some(self.group.clone()),
            rank: Some(self.rank as usize),
            world_size: Some(self.world_size),
            devices: Some(
                self.devices
                    .iter()
                    .map(|device| device.0.to_string())
                    .collect(),
            ),
            // The engine's facts want one host name per rank, and the launch file states the
            // rank's devices but not which host each rank sits on - it cannot, a launch file is
            // per node. A world of one therefore has exactly one host and this node is it; the
            // name is the machine's own rather than an invented string, so a report, a log or a
            // transfer that names the host names the host.
            //
            // A world of more ranks needs the real names for placement and for cross-host weight
            // transfer, and a guess there would be worse than the refusal: the fact stays absent
            // and `EngineConfig::of` names what the launch file owes it.
            hosts: match self.world_size {
                1 => Some(vec![host_name()?]),
                _ => None,
            },
            peers: None,
            max_running_requests: Some(self.build.max_running_requests.get()),
            max_batch: Some(rows as usize),
            max_num_batched_tokens: Some(tokens as usize),
            block_size: Some(cache_page_size(memory, model)?),
            kv_cache_dtype: Some(cache_dtype(model, self.build.kv_cache.dtype)?),
            host_cache: match (self.build.kv_cache.host_ratio, self.build.kv_cache.host_gib) {
                // A deployment that states neither keeps no host tier of its own, which is what a
                // storage grant of `None` means.
                (None, None) => None,
                (ratio, gib) => Some(engine::HostCacheSize::new(
                    ratio,
                    gib.map(|gib| gib.get() << 30),
                )?),
            },
            graph_tiers: Some(graph_tiers),
            // The tokenizer facts are the checkpoint's (`checkpoint_tokens`): a launch file states
            // where the checkpoint is, not what its vocabulary means, and the engine schedules
            // against the ids themselves.
            bos: Some(control.bos),
            eos: Some(control.eos),
            // A causal LM encodes no image, so no token closes encoded image content. The wire
            // field carries no presence bit: it is a plain `uint32`, and the daemon writes `0`
            // for a profile that states no generation controls. So `0` stands for the absence of
            // a token rather than for an invented one, and a reader must not take it as an id.
            end_of_image: Some(0),
            // What one request may be given to generate on this model; see `understanding_limits`.
            generation_limits: Some(understanding_limits()),
            max_model_len: self.build.max_model_len.map(|length| length.get()),
            watchdog_timeout: Some(self.watchdog_timeout()),
        })
    }

    /// The runners a media rank serves `device` with: one per block the pipeline holds.
    ///
    /// **Three runners, one stream each.** An encoder writes the hidden states a denoise step reads,
    /// a denoiser reads them and publishes the trajectory, and a decoder reads that trajectory as a
    /// clip, so each is a separate stream on the one device - a stream per runner is what lets the
    /// store order the crossings between them.
    ///
    /// **The buckets are the deployment's canvases and prompt capacities**, not the checkpoint's: a
    /// bucket is a captured graph, and a deployment that captures a canvas it does not serve would
    /// pay a capture for nothing.
    fn media_runners(
        &self,
        device: DeviceId,
        pipeline: &Arc<minimax_h3::load::MinimaxH3>,
        generation: &Arc<Generation>,
        blocking: &executor::blocking::ThreadPool,
    ) -> Result<Vec<NamedRunner>> {
        use executor::runners::decoder as media_decoder;
        use executor::runners::denoiser as media_denoiser;
        use executor::runners::encoder as media_encoder;
        use executor::runners::media as media_host;
        use nn::capability::Denoiser as _;

        if Some(&device) != self.devices.first() {
            bail!(
                "this build maps one model per rank, on device {}; a runner on device {} would \
                 forward from another device's addresses",
                self.devices.first().map_or(-1, |first| first.0),
                device.0
            );
        }
        let longest = self.build.max_model_len.map(|length| length.get()).context(
            "a deployment of the media class states no `max_model_len`, and its prompt capacities \
             are checked against it",
        )?;
        let capacities = prompt_capacities(&self.build.video, longest)?;
        // The checkpoint's own statement of what it serves, read once: the candidates are per task,
        // and a task the checkpoint does not serve gets no graph.
        let config = minimax_h3::Config::read(&self.checkpoint, None)?;
        let candidates = media_candidates(&config, &self.build.video, &capacities)?;

        // The buckets the deployment captures: an encode bucket per (rows, prompt capacity) and a
        // decode bucket per canvas. A denoiser derives its own from the candidates it walks, so the
        // set here is what the encoder and the decoder are bound with.
        let mut encode_buckets: Vec<executor::plan::Bucket> = Vec::new();
        let rows = u32::try_from(self.build.max_running_requests.get())
            .context("more request slots than the protocol addresses")?;
        for capacity in &capacities {
            for row in 1..=rows {
                encode_buckets.push(executor::plan::Bucket::prefill(
                    row,
                    *capacity,
                    Variant::Encode,
                ));
            }
        }
        // **A decode bucket is one decode unit, not the timeline.** The video decoder decodes a clip
        // of the checkpoint's own window - `decode_latents` latent frames, `frames_out` native ones -
        // and a timeline longer than one is spliced from its units. A bucket naming the deployment's
        // whole clip asked one decode call for a raster no decoder produces.
        let window = minimax_h3::video_vae::decode_latents(&config.video_vae)
            * config.video_vae.temporal_compression();
        let mut decode_buckets: Vec<executor::plan::Bucket> = Vec::new();
        for candidate in &candidates {
            let bucket =
                executor::plan::Bucket::clip(window, candidate.height, candidate.width);
            if !decode_buckets.contains(&bucket) {
                decode_buckets.push(bucket);
            }
        }
        let hidden = config.text_encoder.hidden_size;
        let text = pipeline.text_encoder().context(
            "the checkpoint holds no text encoder, so a media request's prompt could not be \
             encoded",
        )?;
        let denoiser = pipeline
            .transformer()
            .context("the checkpoint holds no transformer, so no denoising step could run")?;
        let decoder = pipeline
            .video_decoder()
            .context("the checkpoint holds no video decoder, so no clip could be decoded")?;
        // The encoder writes its hidden states in the deployment's effective type, which the daemon
        // resolved against the checkpoint before it wrote this build config (a `dtype` of `auto`
        // never reaches a main process as `auto`).
        let dtype = match self.build.dtype {
            proto::request::ModelDtype::Float16 => DType::F16,
            proto::request::ModelDtype::BFloat16 => DType::BF16,
            proto::request::ModelDtype::Float32 => DType::F32,
        };

        let stream = |name: &str| -> Result<platform::Stream> {
            platform::Stream::new(device)
                .with_context(|| format!("creating device {}'s {name} stream", device.0))
        };
        let encode_set = Buckets::new(encode_buckets);
        let decode_set = Buckets::new(decode_buckets);
        let encoder = media_encoder::EncoderRunner::new(
            Binding {
                id: RunnerId(0),
                phases: served_phases(&encode_set),
                device,
                stream: stream("encoder")?,
            },
            Arc::clone(text),
            encode_set,
            media_encoder::EncoderShape { hidden, dtype },
            Arc::clone(generation),
        )?;
        let warm = executor::runners::warm::distinct(&**denoiser, &candidates);
        // The denoiser's own buckets are the layouts those candidates collapse to: its phase is
        // derived from them rather than named, so a rank that captures no denoise bucket states no
        // denoise phase.
        let denoise_set = Buckets::new(warm.iter().map(|candidate| {
            executor::plan::Bucket::denoise(1, denoiser.layout(&candidate.view()))
        }));
        // One audio decode bucket per distinct track those layouts carry. A `t2va` clip is a picture
        // **and** a soundtrack, so a deployment that captured clips alone would produce video with
        // no track; the track's extent is the layout the denoiser already states, and two canvases
        // of one duration collapse onto one track. A deployment that prepares no audio at all states
        // none, and then no audio runner is bound rather than one that could serve nothing.
        let audio_buckets: Vec<executor::plan::Bucket> = {
            let mut seen: Vec<(u32, u32)> = Vec::new();
            let mut buckets = Vec::new();
            for bucket in denoise_set.iter() {
                let executor::plan::Extent::Latents(layout) = bucket.extent else {
                    continue;
                };
                let Some(audio) = layout.audio else {
                    continue;
                };
                if seen.contains(&(audio.frames, audio.channels)) {
                    continue;
                }
                seen.push((audio.frames, audio.channels));
                buckets.push(executor::plan::Bucket::audio(1, layout));
            }
            buckets
        };
        let denoise = media_denoiser::DenoiserRunner::new(
            Binding {
                id: RunnerId(1),
                phases: served_phases(&denoise_set),
                device,
                stream: stream("denoiser")?,
            },
            Arc::clone(denoiser),
            warm,
            media_denoiser::DenoiserShape {
                stack: dtype,
                sample: minimax_h3::denoiser::SAMPLE_DTYPE,
                width: hidden,
            },
            Arc::clone(generation),
        )?;
        let decode = media_decoder::DecoderRunner::new(
            Binding {
                id: RunnerId(2),
                phases: served_phases(&decode_set),
                device,
                stream: stream("decoder")?,
            },
            Arc::clone(decoder),
            decode_set,
            Arc::clone(generation),
            // The `video_decoder` component's second entry point. A checkpoint that holds none
            // serves a clip rather than a timeline, and the runner then publishes the clip.
            pipeline
                .video_postprocessor()
                .map(|block| Arc::clone(block) as Arc<dyn nn::capability::Decoder + Send + Sync>),
        )?;
        let mut runners = vec![
            NamedRunner::new(RunnerId(0), Box::new(encoder)),
            NamedRunner::new(RunnerId(1), Box::new(denoise)),
            NamedRunner::new(RunnerId(2), Box::new(decode)),
        ];
        let audio_decoder = pipeline.audio_decoder().context(
            "the checkpoint holds no audio decoder, so no track could be decoded or encoded and \
             every response would be a clip with no sound",
        )?;
        if !audio_buckets.is_empty() {
            let audio_set = Buckets::new(audio_buckets);
            let audio = executor::runners::AudioDecoderRunner::new(
                Binding {
                    id: RunnerId(3),
                    phases: served_phases(&audio_set),
                    device,
                    stream: stream("audio decoder")?,
                },
                Arc::clone(audio_decoder),
                audio_set,
                Arc::clone(generation),
            )?;
            runners.push(NamedRunner::new(RunnerId(3), Box::new(audio)));
        }
        // The artifact's three calls. They are a runner of this device because the products they
        // stage live in this device's memory and their copies are ordered on this device's stream,
        // and they own the process's media lane because encoding and muxing are not device work
        // (2.4). The two numbers they work at are the checkpoint's: the frame rate the timeline is
        // measured in and the sample clock the track was decoded at.
        let media = media_host::MediaStage::new(
            Binding {
                id: RunnerId(4),
                phases: host_phases(),
                device,
                stream: stream("media")?,
            },
            Arc::new(media_host::Output::from_env(media_host::Rates {
                frame_rate: minimax_h3::config::FPS,
                audio_rate: audio_decoder.sampling_rate(),
            })?),
            blocking.lane::<media_host::MediaWork>(executor::blocking::BlockingKind::Media)?,
            // A video unit's frames are the model's schedule, not this plan's arithmetic: the
            // postprocessor states the one home of `CLIP_FRAMES * n + OVERLAP_FRAMES`.
            video_units,
        );
        runners.push(NamedRunner::new(RunnerId(4), Box::new(media)));
        Ok(runners)
    }

    /// What a media rank advertises of itself: the facts its scheduler is configured from.
    ///
    /// **A media rank keeps no paged key/value cache**, so the three facts that describe one are
    /// absent rather than zero: `block_size`, `kv_cache_dtype` and the host tier describe a cache
    /// this class does not hold, and stating a zero page would make a scheduler plan against a
    /// pool that never exists. What replaces them is the deployment's media statement: the widest
    /// batch it admits, the prompt capacity its encoder captures, and the length it states.
    ///
    /// Every figure here is the deployment's, not a checkpoint's: a pipeline does not imply a
    /// prompt capacity or a clip length, and a default invented here would be a second statement of
    /// the deployment's own file.
    fn media_rank_facts(&self) -> Result<engine::RankFacts> {
        let longest = self
            .build
            .max_model_len
            .map(|length| length.get())
            .context(
                "a deployment of the media class states no `max_model_len`: the pipeline keeps no \
                 key/value cache, so no checkpoint figure implies one and the deployment's own is \
                 what its prompt capacities are checked against",
            )?;
        let tokens = prompt_capacity(&self.build.video, longest)?;
        Ok(engine::RankFacts {
            model: Some(self.checkpoint.display().to_string()),
            dtype: Some(self.build.dtype.clone()),
            world: Some(self.world),
            group: Some(self.group.clone()),
            rank: Some(self.rank as usize),
            world_size: Some(self.world_size),
            devices: Some(
                self.devices
                    .iter()
                    .map(|device| device.0.to_string())
                    .collect(),
            ),
            hosts: match self.world_size {
                1 => Some(vec![host_name()?]),
                _ => None,
            },
            peers: None,
            max_running_requests: Some(self.build.max_running_requests.get()),
            // One media computation per request: a batch's rows are the requests it carries, and
            // the deployment's admission bound is the number of them.
            max_batch: Some(self.build.max_running_requests.get()),
            max_num_batched_tokens: Some(tokens as usize),
            // No paged cache: the three facts below describe one, and absence is their statement.
            block_size: None,
            kv_cache_dtype: None,
            host_cache: None,
            // A media rank captures graphs per bucket, and a bucket of this class is not sized by
            // tokens: the tiers a policy is checked against do not exist here, which is what an
            // empty list states (`SchedulerConfig::graph_tiers`).
            graph_tiers: Some(Vec::new()),
            // The media class decodes no tokens, so no tokenizer fact reaches a scheduler call: `0`
            // stands for the absence of a token, as a causal LM's `end_of_image` does, and the list
            // of terminating tokens is empty.
            bos: Some(0),
            eos: Some(Vec::new()),
            end_of_image: Some(0),
            // A media request generates a clip, not tokens, so no understanding feature is served;
            // the engine still needs the two lower bounds, as the frontend's arm for this class
            // states them (`crates/frontend/src/model/generation.rs`).
            generation_limits: Some(GenerationLimits {
                latent_downsample: 1,
                max_cfg_branches: 1,
                ..GenerationLimits::default()
            }),
            max_model_len: self.build.max_model_len.map(|length| length.get()),
            watchdog_timeout: Some(self.watchdog_timeout()),
        })
    }

    /// The Program Server's targets this rank resolves a module for (§5.3).
    pub(crate) fn targets(&self) -> &[Target] {
        &self.targets
    }

    /// The streams `device` maps graphs on: one per runner it serves, plus one for a preparation.
    ///
    /// The count is the number of `NamedRunner`s [`EpochPlan::runners`] builds for this class, and
    /// the two are checked against each other where the runners exist: `Executor::spawn` refuses a
    /// device whose runners outnumber the streams its graph range was divided for, naming both
    /// figures. That refusal is what keeps this constant honest - a runner added without the count
    /// following it stops at the spawn instead of at a capture nobody can read.
    ///
    /// A preparation walks its own generation on a stream of its own, so it takes one more range
    /// than the serving generation does, whether or not an update is in flight when the process
    /// starts: a range that appeared only at the first precopy would be mapped after the model and
    /// the cache, and the device would have nothing left to give it.
    fn streams_on(served: &Served) -> usize {
        let serving = match served {
            // The text class serves one causal-language runner (4.3).
            Served::Text(_) => 1,
            // The media class serves its encoder, its denoise step, its decoder, its audio
            // decoder and its host stage: five runners, one stream each.
            Served::Media(_) => 5,
        };
        serving + 1
    }

    /// The runners `device` serves: one per block of the loaded model (§4.3).
    ///
    /// The model's class was resolved once, when it was loaded and its resident memory opened
    /// ([`EpochPlan::load_and_open`]), so this builds the runner that class needs from the model it
    /// is handed. Everything downstream - the buckets, the arena, the sampler, the graphs - is
    /// written against `nn`'s capability traits.
    pub(crate) fn runners(
        &self,
        device: DeviceId,
        served: &Served,
        memory: &mut Memory,
        generation: &Arc<Generation>,
        blocking: &executor::blocking::ThreadPool,
    ) -> Result<Vec<NamedRunner>> {
        let Served::Text(model) = served else {
            let Served::Media(pipeline) = served else {
                unreachable!("Served has two variants");
            };
            return self.media_runners(device, pipeline, generation, blocking);
        };
        let mut runners: Vec<NamedRunner> = Vec::new();
        // One model per rank is loaded and mapped on the rank's first device, and a kernel on a
        // second device would read the first device's addresses: a node whose rank serves two
        // devices needs the model loaded per device. Refusing says so; serving would emit wrong
        // tokens and look like a kernel defect.
        if Some(&device) != self.devices.first() {
            bail!(
                "this build maps one model per rank, on device {}; a runner on device {} would \
                 forward from another device's addresses",
                self.devices.first().map_or(-1, |first| first.0),
                device.0
            );
        }
        {
            {
                let geometry = Geometry {
                    hidden: narrow(model.config().hidden_size, "hidden_size")?,
                    hidden_dtype: model.dtype(),
                    // A logit row is written in f32: the sampler and the read-back both read it,
                    // and the head's own precision would decide a token from the last bit.
                    logits_dtype: DType::F32,
                    vocabulary: narrow(model.config().vocab_size, "vocab_size")?,
                    slots: u32::try_from(self.build.max_running_requests.get())
                        .context("more request slots than the protocol addresses")?,
                    // A request may ask for any top-k it likes, so the candidate width is the
                    // vocabulary: a narrower buffer would silently sample a different set.
                    width: narrow(model.config().vocab_size, "vocab_size")?,
                    // The page size the resident cache was planned with, not the one the model
                    // asked for: the capacity plan may have had to take a smaller page, and a
                    // runner that read a different one would address another request's pages.
                    // A plan that disagrees with the model's stated page size is refused here,
                    // naming both figures, rather than served.
                    page_size: cache_page_size(memory, &model)?,
                    max_kv_len: self.max_kv_len(served)?,
                };
                let stream = platform::Stream::new(device)
                    .with_context(|| format!("creating device {}'s stream", device.0))?;
                let rows = self
                    .buckets
                    .largest(Variant::DecodeGreedy)
                    .map_or(1, |bucket| bucket.rows);
                let cache = memory.shared_cache().context(
                    "a causal-language rank reads a paged cache, and this rank opened none",
                )?;
                let provider: Box<dyn KvProvider> =
                    Box::new(ResidentCache::new(cache, device, rows)?);
                let runner = CausalLmRunner::new(
                    Binding {
                        id: RunnerId(0),
                        phases: served_phases(&self.buckets),
                        device,
                        stream,
                    },
                    Shared::new(Arc::clone(model)),
                    Arc::clone(generation),
                    self.buckets.clone(),
                    geometry,
                    provider,
                    // A seed that changes with the process, so two runs of one deployment are not
                    // the same run.
                    seed(),
                )?;
                runners.push(NamedRunner::new(RunnerId(0), Box::new(runner)));
            }
        }
        Ok(runners)
    }

    /// Loads the rank's model and opens the resident owners it leaves room for (§4.3, §8, 4.1).
    ///
    /// **This is the single match on model class in the tree**, and the arm loads the class it
    /// names: the checkpoint is opened, its config and partition read, the weights mapped through
    /// `weights` (a weight converted once is published to retained storage and mapped read-only
    /// afterwards), and the model built. It then opens the memory, because the capacity plan
    /// measures a device the weights are already mapped on - what is left, not what was there
    /// before the load - and returns both, so the caller holds no model class of its own and no
    /// knowledge of the order the two must be done in.
    /// The deployment page this epoch runs, as the launcher placed it.
    ///
    /// The media class's parallel shape is the deployment's, and the page states it beside the
    /// component it splits; the same reader the launcher used answers it here, so a rank and the
    /// engine that started it cannot disagree about the group's degrees.
    fn deployment_groups(&self) -> Result<Option<Vec<engine::ExecutorGroupConfig>>> {
        let Some(path) = self.build.worker_groups.as_ref() else {
            return Ok(None);
        };
        let text = path
            .to_str()
            .with_context(|| format!("the deployment file {} is not UTF-8", path.display()))?;
        let deployment = engine::Deployment::read(text)
            .with_context(|| format!("reading the deployment file {}", path.display()))?;
        let host = self
            .build
            .engine_host
            .clone()
            .unwrap_or_else(|| "localhost".to_owned());
        deployment
            .worker_groups(&host)
            .map(Some)
            .with_context(|| format!("placing the deployment file {} on {host}", path.display()))
    }

    pub(crate) fn load_and_open(
        &self,
        weights: &mut dyn nn::RetainedWeights,
        storage: &mut dyn executor::memory::retained::ExclusiveStorage,
    ) -> Result<(Served, Memory, ipc::MainProcessInfo, GraphStorage)> {
        let placement = &self.placement;
        match ModelClass::of(&self.checkpoint)? {
            ModelClass::CausalLm => {
                // §8: the checkpoint's own representation stands unless the deployment fixed
                // another, which is the build config's `dtype` and `quantization`.
                let opened = nn::Checkpoint::open(&self.checkpoint)?;
                let mut loader = nn::WeightLoader::new(&opened, weights, None, None);
                let config = qwen3_moe::Config::read(&self.checkpoint)?;
                let partition = qwen3_moe::Partition::new(
                    &config,
                    placement.tensor_parallel as usize,
                    placement.tensor_rank as usize,
                    placement.expert_parallel as usize,
                    placement.expert_rank as usize,
                )?;
                // The checkpoint states its own representation: a calibrated export is read packed,
                // and a dense one dense. The layout is the one the serving programs are built for.
                let model = qwen3_moe::load::load_checkpoint(
                    &self.checkpoint,
                    partition,
                    &mut loader,
                    placement.load_options,
                    ops::fp4::ScaleLayout::Linear,
                )?;
                let served = Served::Text(Arc::new(model));
                let (memory, info, graph) = self.open_memory(&served, storage)?;
                Ok((served, memory, info, graph))
            }
            ModelClass::Media => {
                // §8: the deployment's own representation, which the daemon resolved against the
                // checkpoint before it wrote this build config (`dtype = "auto"` never reaches a
                // main process as `auto`: it arrives as the checkpoint's own type).
                let opened = nn::Checkpoint::open(&self.checkpoint)?;
                let mut loader = nn::WeightLoader::new(&opened, weights, None, None);
                let config = minimax_h3::Config::read(&self.checkpoint, None)?;
                // **The media class's parallel shape is the deployment's, and the deployment file
                // is where it is stated.** The build config carries the tensor-parallel width and
                // no sequence degree, so a rank that read its tensor place alone would either
                // refuse every wider deployment or serve four whole denoisers. The page that
                // placed this group states both beside the component they split, and this reads it
                // through the same reader the launcher used.
                let groups = self.deployment_groups()?;
                let (ulysses, tensor_parallel, tensor_rank, sequence_rank) = media_parallel(
                    groups.as_deref(),
                    &self.group,
                    self.rank,
                    placement.tensor_parallel,
                    placement.tensor_rank,
                    self.world_size,
                )?;
                // **The partition serves the task the checkpoint states, and a checkpoint that
                // states several is refused here.** One partition carries one task's block set
                // (the condition encoders are placed for the conditioned tasks) and one task's
                // condition capacity, so two tasks are two partitions and one deployment holds one.
                // A construction stating their union would serve both; the model offers none, and
                // this does not invent one.
                // **What a deployment of this checkpoint serves, not everything its architecture
                // can express**: a distilled export serves the one task its contract states, and a
                // partition carries one task's block set and condition capacity.
                let task = served_task(&config.serving(), &self.checkpoint)?;
                // **The postprocessor is placed here, though `serving` names only what holds
                // weights.** Its arithmetic is the clip splice and it names no checkpoint
                // directory, so it is not a component a task's weights place - but the decode
                // runner takes it as the video decoder's second entry point, and a rank that held
                // the decoder without it would publish a timeline's first clip and say nothing.
                let partition = minimax_h3::Partition::serving(
                    &config,
                    task,
                    ulysses,
                    tensor_parallel,
                    tensor_rank,
                    sequence_rank,
                )?;
                let partition = partition.placing(
                    partition
                        .blocks
                        .with(minimax_h3::Component::VideoPostprocessor),
                );
                let dtype = match self.build.dtype {
                    proto::request::ModelDtype::Float16 => DType::F16,
                    proto::request::ModelDtype::BFloat16 => DType::BF16,
                    proto::request::ModelDtype::Float32 => DType::F32,
                };
                let canvases = deployment_canvases(&self.build.video)?;
                let model =
                    minimax_h3::load(&opened, partition, Some(dtype), &canvases, &mut loader)?;
                let served = Served::Media(Arc::new(model));
                let (memory, info, graph) = self.open_memory(&served, storage)?;
                Ok((served, memory, info, graph))
            }
        }
    }

    /// Opens the rank's resident owners from the capacity plan (4.1, 8).
    ///
    /// Called after the epoch's model is loaded and before any thread that borrows the owners: the
    /// plan measures a device the weights are already mapped on, so what it computes is what is
    /// left, not what was there before the load. Every owner is booked against the device's one
    /// allowance as it maps, which is what makes the first `mapped <= budget` reading after warmup
    /// count every byte the plan charged. The device's pool set is opened here, first, because the
    /// owners carve their storage from it and the runner is built from them.
    pub(crate) fn open_memory(
        &self,
        served: &Served,
        storage: &mut dyn executor::memory::retained::ExclusiveStorage,
    ) -> Result<(Memory, ipc::MainProcessInfo, GraphStorage)> {
        let device = *self
            .devices
            .first()
            .context("this rank serves no device to open its resident memory on")?;
        let host = host_name()?;
        // What one batch may be, in the class's own terms: a causal-language rank admits up to the
        // largest bucket its catalog captured (`Graphs` refuses anything larger before a kernel
        // runs, so the catalog is its bound and not an estimate), and a media rank admits the
        // deployment's requests at the widest prompt capacity it captures.
        let (rows, tokens) =
            match served {
                Served::Text(_) => self.buckets.bounds(),
                Served::Media(_) => {
                    let longest = self.build.max_model_len.map(|length| length.get()).context(
                    "a deployment of the media class states no `max_model_len`, and its prompt \
                     capacities are checked against it",
                )?;
                    (
                        u32::try_from(self.build.max_running_requests.get())
                            .context("more request slots than the protocol addresses")?,
                        prompt_capacity(&self.build.video, longest)?,
                    )
                }
            };
        // Everything the engine states about the rank, and what the rank reports of itself. The two
        // differ in one field between the classes: a rank that keeps a paged cache states its
        // geometry and a rank that keeps none states its absence, and each planner refuses the
        // other's value by name.
        let (facts, report) = match served {
            Served::Text(model) => {
                // The KV geometry is the model's own: its layer count, the heads each layer stores
                // and the width of a head. Every figure comes from `config()`, so a checkpoint whose
                // shape this build cannot state is refused by name rather than planned with a
                // literal.
                let config = model.config();
                let cache = executor::capacity::CacheGeometry {
                    layers: narrow(config.num_hidden_layers, "num_hidden_layers")?,
                    kv_heads: narrow(config.num_key_value_heads, "num_key_value_heads")?,
                    head_dim: narrow(config.head_dim, "head_dim")?,
                };
                let facts = executor::capacity::RankFacts {
                    device: executor::config::Device::Cuda(device),
                    host: &host,
                    world: self.world,
                    world_size: self.world_size,
                    queue_depth: self.queue_depth,
                    // The model's page is the preset: the plan returns it or fails, and a plan that
                    // lowered it would leave every runner addressing pages the pool does not hold.
                    block_size: Some(model.page_tokens()),
                    max_batch_calls: rows as usize,
                    max_batch_tokens: tokens as usize,
                    max_request_pool_size: self.build.max_running_requests.get(),
                    max_model_len: self.max_kv_len(served)? as usize,
                    mem_fraction_static: self.build.mem_fraction_static,
                    cache: Some(cache),
                    kv_dtype: cache_dtype(model, self.build.kv_cache.dtype)?,
                };
                (facts, rank_report(device, &host)?)
            }
            Served::Media(_) => {
                let facts = executor::capacity::RankFacts {
                    device: executor::config::Device::Cuda(device),
                    host: &host,
                    world: self.world,
                    world_size: self.world_size,
                    queue_depth: self.queue_depth,
                    // No page and no cache: the media class addresses no paged request, so the
                    // three figures a cache is described by are absent. `kv_dtype` is the
                    // deployment's own setting, kept here because the struct states one and the
                    // media plan fits no cache to apply it to.
                    block_size: None,
                    max_batch_calls: rows as usize,
                    max_batch_tokens: tokens as usize,
                    max_request_pool_size: self.build.max_running_requests.get(),
                    max_model_len: self.max_kv_len(served)? as usize,
                    mem_fraction_static: self.build.mem_fraction_static,
                    cache: None,
                    // The media plan fits no cache, so this figure is never read: it states the
                    // element type the rank's states are held in, which is the deployment's own
                    // model type rather than a cache setting the deployment did not make.
                    kv_dtype: match self.build.dtype {
                        proto::request::ModelDtype::Float16 => ipc::PrefixCacheDtype::Float16,
                        proto::request::ModelDtype::BFloat16 => ipc::PrefixCacheDtype::BFloat16,
                        proto::request::ModelDtype::Float32 => ipc::PrefixCacheDtype::Float32,
                    },
                };
                (facts, media_rank_report(device, &host)?)
            }
        };
        // One rank's host group: this rank alone, which is what a world of one rank is. A
        // deployment of several ranks forms its group over the peers' rendezvous at start, and
        // until that exists a plan that agreed a block size over one member would be a plan this
        // rank cannot honour - so it is refused rather than served.
        if self.world_size != 1 {
            bail!(
                "this rank is one of {} ranks and the host group they agree capacity over is not \
                 formed yet; capacity is planned for a world of one rank",
                self.world_size
            );
        }
        let group = executor::parallel::host::HostGroup::single(self.rank as usize);
        let planned = match served {
            Served::Text(_) => executor::capacity::plan_rank(&facts, report, &group)?,
            Served::Media(_) => executor::capacity::plan_media(&facts, report, &group)?,
        };
        // The keys and values the plan fitted are the process's resident pool: a unit is a cache
        // unit, and the plan charged them all against the same device allowance the pools are
        // booked against. A plan that fitted no cache (a rank that serves none) states zero.
        let resident = match (&planned.plan.cache, planned.plan.info.prefix_cache.as_ref()) {
            (Some(cache), Some(info)) => u64::from(info.num_units) * cache.unit_bytes,
            // A media rank keeps no cache, and its resident pool holds the tables its model
            // declares per layout. The reservation is addresses and not storage: a table books the
            // allowance as it maps, so the range is what the device has left to give.
            _ if matches!(served, Served::Media(_)) => {
                platform::Device::new(device)
                    .with_context(|| format!("opening device {}", device.0))?
                    .memory_info()?
                    .free
            }
            _ => 0,
        };
        let snapshot = planned.state_snapshot_bytes();
        let graph = self.graph(device, resident, snapshot, Self::streams_on(served))?;
        let pools = executor::device::Executor::open(device, &graph)?;
        let stream = platform::Stream::new(device)
            .with_context(|| format!("creating device {}'s stream", device.0))?;
        let memory = executor::memory::open_memory(
            executor::memory::OpenMemory {
                plan: &planned.plan,
                // A causal-language rank keeps no per-slot bank: its per-request state is the KV
                // pages in the cache, and the sampler's parameters travel in the batch's columns.
                slots: &[],
                device: Some(device),
                stream: Some(&stream),
                width: planned.width,
                devices: vec![executor::memory::allocation::MemoryKind::Cuda(device)],
                // A rank that takes no other rank's export states that as the absence of a
                // program: a conversion that reaches it is a mistake, and it says so by name.
                conversion: Arc::new(executor::memory::prefix_cache::conversion::Unimported),
                events: self.events(),
                pools,
            },
            storage,
        )
        .map_err(|error| error)
        .context("opening the rank's resident memory")?;
        // The plan's own account of this rank is what the scheduler is told about its pool: the
        // capacities it fitted, the cache it holds and where it is. It is the same `MainProcessInfo` the
        // deployment's other ranks report, so the scheduler plans against one description of a
        // rank rather than a second one derived here. A media rank adds what its denoiser serves,
        // which the frontend plans a request against and no capacity figure implies.
        let mut info = planned.plan.info.clone();
        if let Served::Media(pipeline) = served {
            info.video_denoiser = Some(denoiser_report(pipeline, &self.build.video)?);
        }
        Ok((memory, info, graph))
    }

    /// Requests this rank's scheduler may have in flight (launch: `scheduling.max_running_requests`).
    pub(crate) fn queue_depth(&self) -> usize {
        self.queue_depth
    }

    /// The ranks of this process's group; one process holds one pool, so it is the deployment's.
    pub(crate) fn world_size(&self) -> usize {
        self.world_size
    }

    /// The keys a row reads at most: the cache's bound, from the epoch's context length.
    pub(crate) fn max_kv_len(&self, served: &Served) -> Result<u32> {
        if let Some(length) = self.build.max_model_len {
            return Ok(length.get());
        }
        match served {
            Served::Text(model) => u32::try_from(model.config().max_position_embeddings)
                .context("the checkpoint's context length is wider than the protocol's"),
            Served::Media(_) => bail!(
                "this rank serves the media class and the deployment states no `max_model_len`: \
                 the pipeline keeps no key/value cache, so a length that decides one is the \
                 deployment's to state rather than a checkpoint's to imply"
            ),
        }
    }
}

/// What this rank advertises of itself: the report the capacity plan is made against (§2.3, §4).
///
/// A causal-language rank states one component - the model's single block - the two forward calls
/// it serves, and the three identities that name this process. It publishes no products (`logits`
/// returns through the batch, not as an exported tensor), so no device product is exported and no
/// media component is routed: the deployment is one class of computation, and saying otherwise
/// would make the engine reserve storage for exports that never happen.
pub(crate) fn rank_report(device: DeviceId, host: &str) -> Result<executor::capacity::Report> {
    let at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos() as u64);
    Ok(executor::capacity::Report {
        endpoint: ipc::MainProcessEndpoint {
            host: host.to_owned(),
            // The address space this rank's tensors live in: one main process, one pid.
            address_space: std::process::id().to_string(),
            // This process's start, which the engine compares to notice a relaunched rank.
            incarnation: format!("{at}"),
        },
        device_exports: false,
        fabric_handles: platform::HandleType::probe(device)? == platform::HandleType::Fabric,
        supported_calls: vec![
            ipc::CallKind::Forward(ipc::ForwardMode::Prefill),
            ipc::CallKind::Forward(ipc::ForwardMode::Decode),
        ],
        media_components: std::collections::BTreeMap::new(),
        // The component this rank holds, named as the placement names it (`schedulers` places one
        // component over every rank). The scheduler routes a call by component name, so a rank that
        // advertised a name of its own would hold nothing any pool asks for.
        components: vec![ipc::ComponentInfo {
            name: ipc::DEFAULT_COMPONENT.to_owned(),
            outputs: Vec::new(),
        }],
    })
}

/// What a media rank advertises of itself: the calls its blocks serve, and which component serves
/// each.
///
/// **A media rank serves media calls, not forwards**, so the two calls a causal-language report
/// states (prefill and decode) are not among them: a scheduler routes a request's encode, its
/// denoising steps and its decode by call, and each is served by one block of the pipeline.
///
/// **No device product is exported**, for the same reason the text class exports none: what this
/// class hands on - the encoded conditioning rows, the latent trajectory, the decoded clip - stays
/// in the process, through the product store, and a report that claimed otherwise would have the
/// capacity plan reserve storage for transfers no rank makes.
pub(crate) fn media_rank_report(
    device: DeviceId,
    host: &str,
) -> Result<executor::capacity::Report> {
    use ipc::MediaCall;
    let at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos() as u64);
    // The component names are the deployment's own, which the pipeline's blocks answer to: a
    // scheduler routes a call by component name, and a rank that advertised names of its own would
    // hold nothing any pool asks for.
    let mut media_components = std::collections::BTreeMap::new();
    media_components.insert(MediaCall::TextEncoding, "text_encoder".to_owned());
    media_components.insert(MediaCall::LatentPreparation, "transformer".to_owned());
    media_components.insert(MediaCall::Denoising, "transformer".to_owned());
    media_components.insert(MediaCall::VideoDecoding, "video_decoder".to_owned());
    media_components.insert(MediaCall::AudioDecoding, "audio_decoder".to_owned());
    media_components.insert(MediaCall::VideoEncoding, "video_codec".to_owned());
    media_components.insert(MediaCall::AudioEncoding, "muxer".to_owned());
    media_components.insert(MediaCall::Muxing, "muxer".to_owned());
    let supported_calls: Vec<ipc::CallKind> = media_components
        .keys()
        .copied()
        .map(ipc::CallKind::Media)
        .collect();
    let components: Vec<ipc::ComponentInfo> = media_components
        .values()
        .map(|name| ipc::ComponentInfo {
            name: name.clone(),
            outputs: Vec::new(),
        })
        .collect();
    Ok(executor::capacity::Report {
        endpoint: ipc::MainProcessEndpoint {
            host: host.to_owned(),
            address_space: std::process::id().to_string(),
            incarnation: format!("{at}"),
        },
        device_exports: false,
        fabric_handles: platform::HandleType::probe(device)? == platform::HandleType::Fabric,
        supported_calls,
        media_components,
        components,
    })
}

/// The control tokens a checkpoint states of itself: what begins its sequences and what ends them.
#[derive(Debug)]
struct ControlTokens {
    /// The token that begins a sequence.
    bos: u32,
    /// The tokens that terminate generation, the primary one first.
    eos: Vec<u32>,
}

/// Reads the control tokens from the checkpoint the launch file names.
///
/// A launch file states where the checkpoint is, not what its vocabulary means, so the ids come
/// from the checkpoint's own files, in the order the engine reads them:
///
/// - `generation_config.json` states the generation defaults, and its `eos_token_id` is the primary
///   end of generation;
/// - `config.json` states the architecture's `bos_token_id` and `eos_token_id`, which may be one id
///   or a list;
/// - `tokenizer_config.json` names its special tokens and maps each to an id in
///   `added_tokens_decoder`.
///
/// Qwen3 states no beginning-of-sequence token (`"bos_token_id": null`), so the beginning a text
/// model has is the tokenizer's own end of text, `<|endoftext|>`, read from the file that names it
/// rather than written here as a constant - and it joins the eos list for the same reason: a
/// completion that emits the vocabulary's end of text must stop, and the engine acts only on the
/// ids it is given.
///
/// The primary end of generation is `eos[0]`, because that is the token the engine gives a
/// completion that commits none (`crates/daemon/src/leader.rs::special_token_ids`, whose rule this
/// follows); the rest follow ascending.
fn checkpoint_tokens(dir: &Path) -> Result<ControlTokens> {
    let config = read_json(&dir.join("config.json"))?;
    // Both files are optional: a checkpoint may state its generation defaults in the architecture's
    // configuration alone, and an architecture that states a beginning states it there.
    let generation = read_json(&dir.join("generation_config.json")).ok();
    let tokenizer = read_json(&dir.join("tokenizer_config.json")).ok();
    let end_of_text = tokenizer
        .as_ref()
        .and_then(|value| added_token(value, "<|endoftext|>"));

    let bos = token_id(config.get("bos_token_id"))
        .or_else(|| {
            generation
                .as_ref()
                .and_then(|value| token_id(value.get("bos_token_id")))
        })
        .or(end_of_text)
        .with_context(|| {
            format!(
                "{} states no beginning-of-sequence token: `config.json` and \
                 `generation_config.json` state no `bos_token_id`, and `tokenizer_config.json` \
                 names no `<|endoftext|>`",
                dir.display()
            )
        })?;

    let mut eos = Vec::new();
    for value in [
        generation
            .as_ref()
            .and_then(|value| value.get("eos_token_id")),
        config.get("eos_token_id"),
    ]
    .into_iter()
    .flatten()
    {
        for id in token_ids(value) {
            if !eos.contains(&id) {
                eos.push(id);
            }
        }
    }
    if let Some(end_of_text) = end_of_text {
        if !eos.contains(&end_of_text) {
            eos.push(end_of_text);
        }
    }
    anyhow::ensure!(
        !eos.is_empty(),
        "{} states no end-of-generation token: neither `config.json` nor \
         `generation_config.json` states `eos_token_id`, and `tokenizer_config.json` names no \
         `<|endoftext|>`",
        dir.display()
    );
    Ok(ControlTokens { bos, eos })
}

/// Reads one JSON file of a checkpoint.
fn read_json(path: &Path) -> Result<serde_json::Value> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

/// One token id, whether the file writes it as a number or as a list of numbers.
fn token_id(value: Option<&serde_json::Value>) -> Option<u32> {
    let value = value?;
    match value {
        serde_json::Value::Number(number) => number.as_u64().and_then(|id| u32::try_from(id).ok()),
        serde_json::Value::Array(ids) => ids
            .first()
            .and_then(|id| id.as_u64())
            .and_then(|id| u32::try_from(id).ok()),
        _ => None,
    }
}

/// Every token id a field states, in the order it writes them; a single number is a list of one.
fn token_ids(value: &serde_json::Value) -> Vec<u32> {
    match value {
        serde_json::Value::Number(number) => number
            .as_u64()
            .and_then(|id| u32::try_from(id).ok())
            .into_iter()
            .collect(),
        serde_json::Value::Array(ids) => ids
            .iter()
            .filter_map(|id| id.as_u64())
            .filter_map(|id| u32::try_from(id).ok())
            .collect(),
        _ => Vec::new(),
    }
}

/// The id a tokenizer gives the special token whose text is `content`.
///
/// The tokenizer states its added tokens as a map from id to token in `added_tokens_decoder`, and
/// older checkpoints state them as a list of `{id, content}` in `added_tokens`; both are read, so a
/// checkpoint of either shape resolves the same token.
fn added_token(tokenizer: &serde_json::Value, content: &str) -> Option<u32> {
    if let Some(decoder) = tokenizer
        .get("added_tokens_decoder")
        .and_then(serde_json::Value::as_object)
    {
        for (id, token) in decoder {
            if token.get("content").and_then(serde_json::Value::as_str) == Some(content) {
                return id.parse::<u32>().ok();
            }
        }
    }
    tokenizer
        .get("added_tokens")
        .and_then(serde_json::Value::as_array)?
        .iter()
        .find(|token| token.get("content").and_then(serde_json::Value::as_str) == Some(content))
        .and_then(|token| token.get("id"))
        .and_then(serde_json::Value::as_u64)
        .and_then(|id| u32::try_from(id).ok())
}

/// What one request may be given to generate on a text model.
///
/// This model understands a prompt and generates tokens, and it encodes no vision features and
/// denoises no latents, so the served feature is `GenerationFeatures::UNDERSTANDING` and every
/// **capacity** is zero: `max_latent_units`, `latent_bytes_per_unit`, `max_vae_grid_tokens`,
/// `max_vit_grid_tokens`, `max_latent_feature_bytes`, `max_vision_feature_bytes`,
/// `commit_marker_tokens` and `encoder_cache_entries`. A capacity of zero declares no capacity at
/// all, which is what a model that encodes and denoises nothing has.
///
/// The two nonzero figures are **lower bounds that must be at least one**, not capacities, and they
/// are the Qwen3 profile's own (`crates/frontend/src/model/generation.rs`, its `Qwen3` arm:
/// `GenerationLimits { latent_downsample: 1, max_cfg_branches: 1, ..Default::default() }`):
/// `latent_downsample` is a divisor, so zero would divide a pixel grid by it, and
/// `max_cfg_branches` counts the branches a request may run, so one branch is the minimum a
/// deployment admits rather than a media fact about this model.
fn understanding_limits() -> GenerationLimits {
    GenerationLimits {
        features: GenerationFeatures::UNDERSTANDING,
        latent_downsample: 1,
        max_cfg_branches: 1,
        ..GenerationLimits::default()
    }
}

/// The element type the cache pool stores, from the deployment when it states one.
///
/// An unstated cache dtype is the model's own activation dtype, which is what its layers compute
/// in until the deployment asks for a narrower cache. A model whose activations have no cache
/// element type cannot have its pool planned, and says so rather than taking the nearest.
fn cache_dtype(
    model: &qwen3_moe::Qwen3Moe,
    stated: Option<proto::request::PrefixCacheDtype>,
) -> Result<proto::request::PrefixCacheDtype> {
    if let Some(dtype) = stated {
        return Ok(dtype);
    }
    match model.dtype() {
        DType::BF16 => Ok(proto::request::PrefixCacheDtype::BFloat16),
        DType::F16 => Ok(proto::request::PrefixCacheDtype::Float16),
        DType::F32 => Ok(proto::request::PrefixCacheDtype::Float32),
        DType::F8E4M3 => Ok(proto::request::PrefixCacheDtype::Float8E4m3Fn),
        other => bail!("this build plans no KV cache in a model whose activations are {other:?}"),
    }
}

/// This host's name, as `gethostname(2)` gives it: the report's host identity (§2.3).
fn host_name() -> Result<String> {
    let mut name = [0_u8; 256];
    // SAFETY: gethostname writes at most `name.len()` bytes into `name`.
    if unsafe { libc::gethostname(name.as_mut_ptr().cast(), name.len()) } != 0 {
        return Err(std::io::Error::last_os_error()).context("reading the host name");
    }
    let length = name
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(name.len());
    String::from_utf8(name[..length].to_vec()).context("the host name is not UTF-8")
}

/// The page size the resident cache was planned with, checked against the model's (§4.5, §8).
///
/// The model states the page it reads; the capacity plan states the page it could afford, and the
/// two must be the same page. A runner that served the model's figure while the pool held another
/// would read another request's tokens out of the cache - wrong tokens with no error - so this is a
/// refusal at spawn, naming both figures, and the block table and the sampler read the same one.
fn cache_page_size(memory: &Memory, model: &qwen3_moe::Qwen3Moe) -> Result<u32> {
    let cache = memory.shared_cache().context(
        "the model reads a paged cache and this rank opened none",
    )?;
    let cache = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let planned = cache
        .layer_planes(0, 0)
        .context("reading the resident cache's planned page size")?
        .block_size;
    let stated = model.page_tokens();
    if planned != stated {
        bail!(
            "the model reads {stated} tokens per cache page and the resident cache was planned \
             with {planned}; the pool would hold pages the runner does not address"
        );
    }
    Ok(planned)
}

/// The targets this rank resolves a module for: the architecture-specific instruction set of every
/// device it serves (§5.3).
///
/// The architecture-suffixed name (`sm_100a`) is the one the kernels are built for: the plain name
/// omits the family's instructions, and a module built for it would run the slice's attention at a
/// fraction of the device.
fn targets(devices: &[DeviceId]) -> Result<Vec<Target>> {
    let mut targets = Vec::with_capacity(devices.len());
    for device in devices {
        let handle = platform::Device::new(*device)
            .with_context(|| format!("opening device {}", device.0))?;
        let (major, minor) = handle.architecture();
        let target = Target::new(format!("sm_{major}{minor}a"));
        if !targets.contains(&target) {
            targets.push(target);
        }
    }
    Ok(targets)
}

/// A seed for the runner's draws: the process's clock, so two runs differ.
fn seed() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos() as u64);
    (nanos ^ u64::from(std::process::id())) | 1
}

/// `value` as the protocol's width, or the name of the field that does not fit it.
fn narrow(value: usize, name: &str) -> Result<u32> {
    u32::try_from(value).with_context(|| format!("{name} of {value} exceeds the protocol's width"))
}

/// The prompt capacities a media deployment captures: its own statement, or the default ladder.
///
/// The deployment's `text_capacities`, when stated, must be strictly increasing and below the
/// length it admits - a capacity above it would capture an encoder graph no request can fill. The
/// default is the one its options document states: `1024`, then steps of `2048`, below the length.
fn prompt_capacities(video: &daemon::options::VideoOptions, longest: u32) -> Result<Vec<u32>> {
    let stated: Vec<u32> = match &video.text_capacities {
        Some(capacities) => capacities.iter().map(|value| value.get()).collect(),
        None => {
            let mut ladder = Vec::new();
            let mut capacity = 1024;
            while capacity < longest {
                ladder.push(capacity);
                capacity += 2048;
            }
            ladder
        }
    };
    if stated.is_empty() {
        bail!(
            "this deployment's video options state no prompt capacity below the {longest} tokens \
             it admits, so no encoder bucket could be captured"
        );
    }
    for pair in stated.windows(2) {
        if pair[0] >= pair[1] {
            bail!(
                "this deployment's prompt capacities {} are not strictly increasing, so a request \
                 of {} tokens would select two captured graphs",
                stated
                    .iter()
                    .map(|value| value.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
                pair[0]
            );
        }
    }
    if let Some(above) = stated.iter().copied().find(|capacity| *capacity > longest) {
        bail!(
            "this deployment states the prompt capacity {above} above the {longest} tokens it \
             admits: the graph would be captured and never filled"
        );
    }
    Ok(stated)
}

/// The widest prompt capacity a deployment captures: what one media batch may present.
fn prompt_capacity(video: &daemon::options::VideoOptions, longest: u32) -> Result<u32> {
    prompt_capacities(video, longest)?
        .into_iter()
        .max()
        .ok_or_else(|| anyhow::anyhow!("this deployment captures no prompt capacity"))
}

/// The condition counts a task's warm graph is captured with.
///
/// A warm request states the **most** a request of its task carries, because the graph is captured
/// once and a request may carry fewer: `fl2va` anchors a first and a last frame, and `t2va` carries
/// no condition media at all. `ref2va`'s count is the deployment's `references` limit - a reference
/// video's length is a request's, so the number is not the task's - and it is `None` rather than a
/// guess.
fn task_conditions(task: nn::VideoTask) -> Option<(u32, u32, u32)> {
    Some(match task {
        nn::VideoTask::TextToVideoAudio => (0, 0, 0),
        nn::VideoTask::FirstLastToVideoAudio => (1, 1, 0),
        nn::VideoTask::ReferenceToVideoAudio => return None,
    })
}

/// The least legal timeline of at least `frames` frames (`17 * n + 5`).
///
/// A timeline is one 17-frame body per clip plus a 5-frame tail (`VideoPostprocessor::splices_of`),
/// so the legal counts are 22, 39, 56, … A deployment states a duration in seconds and the model
/// admits the least of those covering it: 4 s is 96 frames and snaps to 107, 15 s is 360 and snaps
/// to 362.
fn snap_frames(frames: u32) -> Result<u32> {
    use minimax_h3::config::CLIP_FRAMES;
    // The tail is the timeline's own: `Ref2Va`'s overlap frames, the same five the splice fades.
    let tail = minimax_h3::output::Rgb::OVERLAP_FRAMES;
    if frames < CLIP_FRAMES + tail {
        bail!(
            "this deployment's longest clip is {frames} frame(s) and a timeline is at least {} \
             ({CLIP_FRAMES} body frames and a {tail}-frame tail)",
            CLIP_FRAMES + tail
        );
    }
    let beyond = frames - tail;
    Ok(tail + beyond.div_ceil(CLIP_FRAMES) * CLIP_FRAMES)
}

/// The warm candidates a media deployment binds its denoiser with.
///
/// **One candidate per (task, canvas, prompt capacity) the deployment serves.** A denoiser bucket is
/// a layout, and a layout counts the prompt's rows and the request's conditions, so a deployment
/// that captures three prompt capacities at two canvases has six layouts to walk - and
/// `warm::distinct` keeps one candidate per layout, because two candidates that the model's padding
/// collapses would capture the same graph twice.
///
/// **The tasks are the checkpoint's.** The candidates are one per (task, canvas, prompt capacity)
/// for every task `Config::tasks` states, with the condition counts of that task: a deployment of a
/// text-to-video-and-audio export captures the one graph its requests fill, and an export that
/// serves first/last-frame requests captures those too. A task whose warm graph needs a count this
/// function cannot state is refused by name rather than captured with a guess.
///
/// The canvas set is the intersection of what the deployment prepares and what the checkpoint
/// serves, and the frames are the deployment's longest clip at the model's frame rate - the model
/// snaps them to a legal count itself, which is why the request states the plain figure.
fn media_candidates(
    config: &minimax_h3::Config,
    video: &daemon::options::VideoOptions,
    capacities: &[u32],
) -> Result<Vec<executor::runners::denoiser::WarmRequest>> {
    use minimax_h3::config::{FPS, NAMED_CANVASES};
    let mut candidates = Vec::new();
    let frames = (video.max_seconds * f64::from(FPS)).round();
    if !(frames.is_finite() && frames >= 1.0) {
        bail!(
            "this deployment's longest clip is {} seconds, which is not a positive number of \
             frames at {FPS} frames per second",
            video.max_seconds
        );
    }
    // **The timeline rule is the checkpoint's** (`17 * n + 5`): a deployment states a duration, and
    // the frames it admits are the least legal timeline covering it. An unsnapped count captured a
    // layout the splice could not write and the decoder could not decode.
    let frames = snap_frames(frames as u32)?;
    for edge in &video.resolutions {
        let short_edge: u32 = edge
            .strip_suffix('p')
            .and_then(|value| value.parse().ok())
            .with_context(|| {
                format!("the deployment's resolution class {edge:?} does not name a short edge in pixels")
            })?;
        for ratio in &video.aspect_ratios {
            let (width, height) = ratio
                .split_once(':')
                .and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?)))
                .with_context(|| format!("the deployment's aspect ratio {ratio:?} is not W:H"))?;
            let canvas = NAMED_CANVASES
                .iter()
                .find(|canvas| canvas.short_edge == short_edge && canvas.aspect == (width, height))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "this deployment prepares {edge} at {ratio}, and the checkpoint serves no \
                         such canvas: it serves {}",
                        NAMED_CANVASES
                            .iter()
                            .map(|canvas| format!(
                                "{}p {}:{}",
                                canvas.short_edge, canvas.aspect.0, canvas.aspect.1
                            ))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })?;
            // **The tasks are the checkpoint's, not this function's.** A deployment captures a
            // graph per (task, canvas, prompt capacity): the checkpoint states which tasks it
            // serves (`Config::tasks`), the daemon refuses a request of any other task against that
            // statement, and a candidate stated here for a task the checkpoint does not serve would
            // be a graph nothing fills.
            for task in config.tasks() {
                let (first_frames, last_frames, references) =
                    task_conditions(task).with_context(|| {
                        format!(
                            "this checkpoint serves {task:?} and a warm graph of that task states \
                             its reference count, which is the deployment's `references` limit \
                             rather than a number of the task's"
                        )
                    })?;
                for capacity in capacities {
                    candidates.push(executor::runners::denoiser::WarmRequest {
                        task,
                        width: canvas.width,
                        height: canvas.height,
                        frames,
                        fps: FPS,
                        first_frames,
                        last_frames,
                        references,
                        text_tokens: *capacity,
                        conditions: Vec::new(),
                        vision_spans: Vec::new(),
                    });
                }
            }
        }
    }
    if candidates.is_empty() {
        bail!(
            "this deployment prepares no resolution class at any aspect ratio, so a media runner \
             would be bound with no canvas to serve"
        );
    }
    Ok(candidates)
}

/// The rasters this deployment's video decoder serves, in pixels, in the order it names them.
///
/// **A canvas is the deployment's statement, not the checkpoint's**: `video_config.resolutions` and
/// `video_config.aspect_ratios` name it, and the checkpoint is asked which raster that pair is
/// (`NAMED_CANVASES`). **One decoder serves the whole set**, because the constants a decode reads
/// are a tile's (its voxel permutation, its register and zero-row offsets) or a width's (the zero
/// row, the rotary frequencies) - the same for every canvas, so a canvas of the set is a lookup and
/// costs no device storage. The one per-canvas figure is the attention block's row bound, taken as
/// the largest.
fn deployment_canvases(video: &daemon::options::VideoOptions) -> Result<Vec<(u32, u32)>> {
    let mut canvases: Vec<(u32, u32)> = Vec::new();
    for canvas in named_canvases(video)? {
        if !canvases.contains(&(canvas.width, canvas.height)) {
            canvases.push((canvas.width, canvas.height));
        }
    }
    Ok(canvases)
}

/// What the placed denoisers serve, as the frontend plans a request against it.
///
/// The tasks, the schedule and the shifts are the checkpoint's. The frame rule is the clip rule, the
/// rates are the model's output clock, and the canvases are the ones this deployment prepares.
fn denoiser_report(
    pipeline: &minimax_h3::load::MinimaxH3,
    video: &daemon::options::VideoOptions,
) -> Result<ipc::VideoDenoiserInfo> {
    use minimax_h3::{AUDIO_SAMPLE_RATE, CLIP_FRAMES, CLIP_LATENTS, FPS};
    use models::video::{
        Canvas, DenoiserInfo, FrameClips, ReferenceLimits, ServedCanvas, VideoTask,
    };
    let denoisers: Vec<_> = [pipeline.transformer(), pipeline.transformer_ref()]
        .into_iter()
        .flatten()
        .collect();
    let Some(first) = denoisers.first() else {
        bail!("this media rank places no denoiser, so it serves no task");
    };
    let served = |task: VideoTask| {
        let named = match task {
            VideoTask::T2va => nn::VideoTask::TextToVideoAudio,
            VideoTask::Fl2va => nn::VideoTask::FirstLastToVideoAudio,
            VideoTask::Ref2va => nn::VideoTask::ReferenceToVideoAudio,
        };
        denoisers
            .iter()
            .any(|denoiser| denoiser.config().tasks.contains(&named))
    };
    let tasks: Vec<VideoTask> = VideoTask::ALL.into_iter().filter(|task| served(*task)).collect();
    let steps = first.schedule().num_steps()?;
    let shifts = &first.config().schedule.shifts;
    // The fewest frames are one whole clip and the tail, and the tail's latent frames are what the
    // timeline holds past the clip's.
    let min_frames = CLIP_FRAMES + CLIP_LATENTS;
    let latent_frames = minimax_h3::packing::video_latent_frames(min_frames)?;
    let report = DenoiserInfo {
        tasks: tasks.iter().map(|task| task.as_str().to_owned()).collect(),
        schedule_points: steps + 1,
        video_shift: f64::from(shifts.video),
        audio_shift: f64::from(shifts.audio),
        frame_rate: FPS,
        audio_sample_rate: AUDIO_SAMPLE_RATE,
        frames: FrameClips {
            min_frames,
            clip_frames: CLIP_FRAMES,
            clip_latent_frames: CLIP_LATENTS,
            tail_latent_frames: latent_frames - CLIP_LATENTS,
        },
        canvases: named_canvases(video)?
            .into_iter()
            .map(|canvas| ServedCanvas {
                short_edge: canvas.short_edge,
                aspect_ratio: canvas.name(),
                canvas: Canvas {
                    width: canvas.width,
                    height: canvas.height,
                },
            })
            .collect(),
        references: if tasks.contains(&VideoTask::Ref2va) {
            // The counts the checkpoint documents for a reference request.
            ReferenceLimits {
                images: 9,
                videos: 3,
                audios: 3,
                total: 12,
                image_short_edge: 2048,
            }
        } else {
            ReferenceLimits::default()
        },
    };
    report
        .validate()
        .map_err(|reason| anyhow::anyhow!("the denoiser's report: {reason}"))?;
    Ok(ipc::VideoDenoiserInfo {
        tasks: tasks
            .iter()
            .map(|task| ipc::VideoTaskInfo {
                name: task.as_str().to_owned(),
                conditioned: *task != VideoTask::T2va,
            })
            .collect(),
        denoise_steps: steps,
        description: report.encode().context("encoding the denoiser's report")?,
    })
}

/// The named canvases this deployment prepares, each resolution class at each aspect ratio.
fn named_canvases(
    video: &daemon::options::VideoOptions,
) -> Result<Vec<&'static minimax_h3::Canvas>> {
    use minimax_h3::config::NAMED_CANVASES;
    let mut canvases: Vec<&'static minimax_h3::Canvas> = Vec::new();
    for edge in &video.resolutions {
        let short_edge: u32 = edge
            .strip_suffix('p')
            .and_then(|value| value.parse().ok())
            .with_context(|| {
                format!("the deployment's resolution class {edge:?} does not name a short edge in pixels")
            })?;
        for ratio in &video.aspect_ratios {
            let (width, height) = ratio
                .split_once(':')
                .and_then(|(w, h)| Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?)))
                .with_context(|| format!("the deployment's aspect ratio {ratio:?} is not W:H"))?;
            let canvas = NAMED_CANVASES
                .iter()
                .find(|canvas| canvas.short_edge == short_edge && canvas.aspect == (width, height))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "this deployment prepares {edge} at {ratio}, and the checkpoint serves no \
                         such canvas: it serves {}",
                        NAMED_CANVASES
                            .iter()
                            .map(|canvas| format!(
                                "{}p {}:{}",
                                canvas.short_edge, canvas.aspect.0, canvas.aspect.1
                            ))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })?;
            canvases.push(canvas);
        }
    }
    if canvases.is_empty() {
        bail!("this deployment prepares no resolution class, so no raster could be decoded");
    }
    Ok(canvases)
}

/// The model class a checkpoint declares, as the checkpoint itself states it.
///
/// A transformers checkpoint states it in `config.json`'s `architectures`; a diffusers modular
/// pipeline states it in `modular_model_index.json`'s `_class_name` and has no `config.json` at
/// its root at all. Both are read here, so every caller keeps branching on the declared class
/// rather than on which file a checkpoint happens to carry.
fn architectural_class(checkpoint: &Path) -> Result<String> {
    let config_path = checkpoint.join("config.json");
    if config_path.is_file() {
        let text = std::fs::read_to_string(&config_path)
            .with_context(|| format!("reading {}", config_path.display()))?;
        let config: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("parsing {}", config_path.display()))?;
        let class = config
            .get("architectures")
            .and_then(|architectures| architectures.as_array())
            .and_then(|architectures| architectures.first())
            .and_then(|class| class.as_str())
            .ok_or_else(|| anyhow::anyhow!("{} declares no model class", config_path.display()))?;
        return Ok(class.to_owned());
    }

    let index_path = checkpoint.join("modular_model_index.json");
    if index_path.is_file() {
        let text = std::fs::read_to_string(&index_path)
            .with_context(|| format!("reading {}", index_path.display()))?;
        let index: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("parsing {}", index_path.display()))?;
        let class = index
            .get("_class_name")
            .and_then(|class| class.as_str())
            .ok_or_else(|| {
                anyhow::anyhow!("{} declares no `_class_name`", index_path.display())
            })?;
        return Ok(class.to_owned());
    }

    bail!(
        "{} declares no model class: it has neither {} (a transformers checkpoint's \
         `architectures`) nor {} (a modular pipeline's `_class_name`)",
        checkpoint.display(),
        config_path.display(),
        index_path.display()
    )
}

/// How this rank takes part in the deployment's parallel shape (§4.3, §5.1), as the build config
/// fixes it and the daemon's launch places it.
pub(crate) struct Placement {
    tensor_parallel: u32,
    tensor_rank: u32,
    expert_parallel: u32,
    expert_rank: u32,
    load_options: qwen3_moe::LoadOptions,
}

/// The one task this deployment serves, from the checkpoint's own statement.
///
/// A partition carries one task's block set and one task's condition capacity: the condition encoders
/// are placed for the conditioned tasks and the capacity is [`minimax_h3::partition::condition_rows`]
/// of the task, so two tasks are two partitions and one deployment holds one. The checkpoint states
/// which it serves (`Config::tasks`, read from the export's own record), and a checkpoint that states
/// several is refused here naming them, rather than served with one task's arrangement for another
/// task's request.
fn served_task(tasks: &[nn::VideoTask], checkpoint: &Path) -> Result<nn::VideoTask> {
    match tasks {
        [] => bail!(
            "the checkpoint {} states no task, so no denoiser partition can be sized for one",
            checkpoint.display()
        ),
        [task] => Ok(*task),
        several => bail!(
            "the checkpoint states {} tasks ({}) and one partition serves one task's block set and \
             condition capacity: a deployment serving several needs a construction that states their \
             union",
            several.len(),
            several
                .iter()
                .map(|task| format!("{task:?}"))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// This rank's sequence split, and its place in it, as the deployment page states them.
///
/// **The page is the one statement of the parallel shape, and this reads it rather than a second
/// copy.** A main process knows its own devices and its rank number, and the build config states the
/// tensor-parallel width; the sequence degree and the head projection are the deployment file's,
/// written beside the component they split. So the group is found by the name the launch states, its
/// one sequence-split component is read, and the rank's position inside that component is its
/// sequence rank.
///
/// It refuses what would otherwise be assumed: a group with two sequence-split components is not one
/// partition, a projection this stack does not implement is refused rather than served at degree
/// one, and a mode that is not Ulysses is refused by name because the denoiser's sequence exchange
/// is Ulysses' alone.
/// The component a media rank serves: the denoiser's node in the deployment page's group.
///
/// It is the page's key (`[worker_groups.components.transformer]`), and it is what tells one engine's
/// group from another's when the name the launch states is the daemon's process group.
const DENOISER_COMPONENT: &str = "transformer";

fn media_parallel(
    groups: Option<&[engine::ExecutorGroupConfig]>,
    group: &str,
    rank: u32,
    tensor_parallel: u32,
    tensor_rank: u32,
    world_size: usize,
) -> Result<(minimax_h3::partition::Ulysses, usize, usize, usize)> {
    let tensor_parallel =
        usize::try_from(tensor_parallel).context("a tensor-parallel width the process addresses")?;
    let tensor_rank = usize::try_from(tensor_rank).context("a tensor rank the process addresses")?;
    let rank = usize::try_from(rank).context("a rank number the process addresses")?;
    // A deployment that states no file places one rank per device, and with one rank there is no
    // other rank for a shape to disagree with: the width the build config states is the whole of it.
    let Some(groups) = groups else {
        if world_size == 1 {
            return Ok((minimax_h3::partition::Ulysses::single(), 1, 0, 0));
        }
        bail!(
            "this deployment states no worker-group file and serves {world_size} ranks, so no \
             sequence degree is stated for them: a file is what places a rank's group and its split"
        );
    };
    // **The name the launch states is the daemon's process group, not the page's id.** Every engine's
    // main process is launched into one process group (`ENGINE_GROUP = "engine"`), which is the
    // epoch's bookkeeping name, while the page names the groups it places (`model`, `text`, `flow`).
    // A rank that looked itself up by the name it was launched under found neither and refused - so
    // the name is preferred when the page has it, and otherwise the page is asked the question it can
    // answer: **which group places this class's component?** A media rank serves the denoiser, and a
    // group that places no transformer is another engine's.
    let named = groups.iter().find(|candidate| candidate.id.0 == group);
    let denoisers: Vec<&engine::ExecutorGroupConfig> = groups
        .iter()
        .filter(|candidate| candidate.components.contains_key(DENOISER_COMPONENT))
        .collect();
    // Both sides as a pair in either refusal: the launch's name is the daemon's roster label and the
    // page's ids are the deployment's groups, and a reader who sees only one side cannot tell which
    // of the two statements to correct.
    let placed = match (named, denoisers.as_slice()) {
        (Some(named), _) => named,
        (None, [only]) => *only,
        (None, []) => {
            bail!(
                "the launch file states worker group {group:?} and the deployment page places \
                 neither it nor a group that places {DENOISER_COMPONENT:?}: the page places {}",
                groups
                    .iter()
                    .map(|group| format!("{:?}", group.id.0))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
        (None, many) => {
            bail!(
                "this rank serves worker group {group:?}, which the deployment does not place, \
                 and {} groups place {DENOISER_COMPONENT:?} ({}): one partition needs one of them",
                many.len(),
                many.iter()
                    .map(|group| format!("{:?}", group.id.0))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    };
    let mut splits = placed.components.iter().filter(|(_, component)| {
        !matches!(
            component.parallel_config.sequence_parallel,
            ipc::SequenceParallel::Local
        )
    });
    let (_, component) = match (splits.next(), splits.next()) {
        // No split: the sequence is this rank's own, and the tensor degree is the whole shape.
        (None, _) => return Ok((minimax_h3::partition::Ulysses::single(), tensor_parallel, tensor_rank, rank)),
        (Some(one), None) => one,
        (Some(first), Some(second)) => bail!(
            "worker group {group:?} states a sequence split on two components ({:?} and {:?}) and \
             one partition describes one split",
            first.0,
            second.0
        ),
    };
    // A component that states no member list holds every rank of its replica, so a rank's position
    // in it is the rank's own ordinal.
    let sequence_rank = if component.ranks.is_empty() {
        rank
    } else {
        component
            .ranks
            .iter()
            .position(|member| *member == rank)
            .with_context(|| {
                format!(
                    "this rank is {rank} and the group's sequence-split component holds ranks {:?}",
                    component.ranks
                )
            })?
    };
    let ulysses = match &component.parallel_config.sequence_parallel {
        ipc::SequenceParallel::Ulysses {
            ulysses_degree,
            projection,
            ..
        } => minimax_h3::partition::Ulysses::of(
            *ulysses_degree,
            match projection {
                ipc::HeadProjection::Exchange => minimax_h3::partition::Projection::Exchange,
                // The stack loads whole attention projections on every rank and sends each member
                // its head shard, which is the exchange mode: a gathered projection would be served
                // weights its own page says it does not hold.
                ipc::HeadProjection::Gather => bail!(
                    "worker group {group:?} states the `gather` head projection, and this stack \
                     loads whole attention projections on every rank: it implements `exchange`"
                ),
            },
        ),
        other => bail!(
            "worker group {group:?} states the sequence split {other:?}, and this class's denoiser \
             exchanges its sequence over Ulysses ranks"
        ),
    };
    Ok((ulysses, tensor_parallel, tensor_rank, sequence_rank))
}

impl Placement {
    /// This rank's part of the epoch's parallel shape.
    ///
    /// Tensor parallelism splits every layer of the group's ranks; expert parallelism splits the
    /// routed experts of the deployment's schedulers. A rank's ordinal places it in both: its
    /// tensor rank is its position inside its scheduler's group, and its expert rank is which
    /// scheduler's share of the experts it keeps.
    fn of(launch: &Launch, build: &daemon::epoch::BuildConfig) -> Result<Self> {
        let tensor_parallel = u32::try_from(build.tensor_parallel_size.get())
            .context("a tensor-parallel degree larger than u32::MAX")?;
        if tensor_parallel == 0 {
            bail!("a tensor-parallel degree of zero splits nothing");
        }
        let data_parallel = u32::try_from(build.data_parallel_size.get())
            .context("a data-parallel degree larger than u32::MAX")?;
        let expert_parallel = if build.expert_parallel {
            data_parallel
        } else {
            1
        };
        Ok(Self {
            tensor_parallel,
            tensor_rank: launch.rank % tensor_parallel,
            expert_parallel,
            expert_rank: (launch.rank / tensor_parallel) % expert_parallel,
            // The deployment's page size is the cache's (`kv_cache_config.block_size`), and the
            // attention program's workspace is the load config's; neither is ours to guess, so the
            // model's own defaults stand until the loader takes them from the build config.
            load_options: qwen3_moe::LoadOptions::default(),
        })
    }
}

/// The phases this node's runner serves, in the numbering the scheduler plans with: one id per phase
/// kind this plan's own buckets are launched as.
///
/// The kinds come from the buckets - what this rank's plan can launch - and the ids from
/// `engine::phase_id`, the table the scheduler's `Plan` numbers a request's work with. Never from a
/// literal: a plan that stated `PhaseId(0)` for its device (text encoding) served no phase the
/// scheduler ever planned, and the executor refused the deployment's first completion with `no
/// device of this rank serves phase PhaseId(3)`.
/// The frames of each video unit of a `frames`-frame timeline, as the model states them.
///
/// The host stage is handed this rather than given the arithmetic: a unit's length is the
/// checkpoint's (`CLIP_FRAMES` and the seam a successor fades against), the model's postprocessor is
/// its one home, and an executor that recomputed it would be a second.
fn video_units(frames: u32) -> executor::Result<Vec<u32>> {
    minimax_h3::output::VideoPostprocessor::splices_of(frames)
        .map(|splices| splices.into_iter().map(|splice| splice.keep).collect())
        .map_err(|error| executor::Error::Invalid(format!("a video unit's frames: {error}")))
}

/// The phases the artifact's three host calls are planned as.
///
/// They serve no graph, so no bucket states them: the mechanism plans each call by its phase, and a
/// device that did not state these three would refuse the first encode of every request.
fn host_phases() -> Vec<PhaseId> {
    [
        engine::PhaseKind::VideoEncode,
        engine::PhaseKind::AudioEncode,
        engine::PhaseKind::Mux,
        engine::PhaseKind::LatentPrepare,
    ]
    .into_iter()
    .map(|kind| PhaseId(u16::try_from(engine::phase_id(kind).0).expect("a phase id fits")))
    .collect()
}

fn served_phases(buckets: &Buckets) -> Vec<PhaseId> {
    let mut kinds: Vec<engine::PhaseKind> = buckets
        .iter()
        .map(|bucket| match bucket.variant {
            Variant::PrefillWrite | Variant::PrefillOutput => engine::PhaseKind::Prefill,
            Variant::DecodeGreedy | Variant::DecodeSampled => engine::PhaseKind::Decode,
            Variant::Encode => engine::PhaseKind::TextEncoding,
            Variant::Denoise => engine::PhaseKind::Denoise,
            // The scheduler plans a clip's decode as `VideoDecode` and a track's as `AudioDecode`
            // (`call_phase`), so a device that advertised `ImageDecode` for either served no phase
            // the scheduler ever planned - the same defect this function's doc records.
            Variant::Decode => engine::PhaseKind::VideoDecode,
            Variant::AudioDecode => engine::PhaseKind::AudioDecode,
            // The artifact's three host calls are calls the mechanism submits and waits on, so the
            // device states the phases they run in even though no graph serves them.
            Variant::VideoEncode => engine::PhaseKind::VideoEncode,
            Variant::AudioEncode => engine::PhaseKind::AudioEncode,
            Variant::Mux => engine::PhaseKind::Mux,
            Variant::LatentPrepare => engine::PhaseKind::LatentPrepare,
        })
        .collect();
    // A refill chunk is a prefill-class chunk of a decoding request and runs wherever a prefill
    // chunk does, so the device that serves prefill states it as well: the scheduler may name a
    // refill and the batch then routes here.
    if kinds.contains(&engine::PhaseKind::Prefill) {
        kinds.push(engine::PhaseKind::Refill);
    }
    kinds.sort_by_key(|kind| engine::phase_id(*kind).0);
    kinds.dedup();
    kinds
        .into_iter()
        // The scheduler numbers a phase with the world's `u32` and the executor names one with the
        // wire's `u16`: the plan is where the two meet, and a kind that did not fit would be a table
        // this process cannot state rather than a phase to serve quietly.
        .map(|kind| PhaseId(u16::try_from(engine::phase_id(kind).0).expect("a phase id fits")))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every checked-in H3 deployment parses, places and validates, and places no rank on a CPU.
    ///
    /// The defect this pins, measured on the tree: ten of the thirteen files placed the media
    /// components on `cpu*N` / `cpu` groups, and a node refuses such a rank by design - "host
    /// {host} runs a rank on \"cpu\": a main process serves CUDA devices"
    /// (`crates/daemon/src/engine/control.rs`). Every one of those deployments was unlaunchable as
    /// written, and nothing checked the files.
    ///
    /// It also holds each file to the resolver's own rules - a component's membership equals its
    /// tensor, sequence and pipeline degrees, a distributed component's degrees are one, every
    /// group's queue depth is positive - because a page that violates one of them parses and then
    /// fails on the node that reads it.
    #[test]
    fn every_h3_deployment_parses_places_and_validates() {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("configs/minimax_h3");
        let mut files = 0;
        for entry in std::fs::read_dir(&directory).expect("the deployment directory") {
            let path = entry.expect("an entry").path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("toml") {
                continue;
            }
            let name = path.file_name().expect("a name").to_string_lossy().into_owned();
            let text = std::fs::read_to_string(&path).expect("the file reads");
            let page = engine::Deployment::parse(&text)
                .unwrap_or_else(|error| panic!("{name} does not parse: {error:#}"));
            // The hosts the file names: a rank is `[HOST/]DEVICE`, and a file that names no host
            // places on the local one.
            let mut hosts: Vec<String> = Vec::new();
            for piece in text.split(['[', ',', ']']) {
                let piece = piece.trim().trim_matches('"');
                let Some((host, device)) = piece.rsplit_once('/') else {
                    continue;
                };
                let placed = device.starts_with("cuda") || device.starts_with("cpu");
                if placed && !hosts.iter().any(|seen| seen == host) {
                    hosts.push(host.to_owned());
                }
            }
            if hosts.is_empty() {
                hosts.push("localhost".to_owned());
            }
            for host in &hosts {
                let groups = page
                    .worker_groups(host)
                    .unwrap_or_else(|error| panic!("{name} does not place on {host}: {error:#}"));
                assert!(!groups.is_empty(), "{name} places no group on {host}");
                for group in &groups {
                    group.validate().unwrap_or_else(|error| {
                        panic!("{name}: worker group {:?} is invalid: {error:#}", group.id.0)
                    });
                    for rank in &group.ranks {
                        assert!(
                            !rank.device.starts_with("cpu"),
                            "{name}: worker group {:?} places rank {} on {:?}, and a main process \
                             serves CUDA devices",
                            group.id.0,
                            rank.host,
                            rank.device
                        );
                    }
                }
            }
            files += 1;
        }
        assert_eq!(files, 13, "every checked-in H3 deployment is checked");
    }

    /// A deployment page as written, parsed and placed on this host.
    fn page(toml: &str) -> Vec<engine::ExecutorGroupConfig> {
        engine::Deployment::parse(toml)
            .expect("the page parses")
            .worker_groups("localhost")
            .expect("the page places")
    }

    /// A deployment page holding one group named `model`, with the components it names.
    fn groups(components: &str) -> Vec<engine::ExecutorGroupConfig> {
        page(&format!(
            "[[worker_groups]]\nid = \"model\"\nranks = [\"cuda:0-3\"]\nqueue_depth = 2\n{components}"
        ))
    }

    /// **The name the launch states is the daemon's process group, and the page's ids are its own.**
    ///
    /// Every engine's main process is launched into one process group (`engine`), which is the
    /// epoch's bookkeeping name rather than the deployment's; the page names what it places (`model`,
    /// `text`, `flow`). A media rank that looked itself up by the name it was launched under found
    /// neither and refused - which is what the first end-to-end serve run reported. The page can
    /// answer the question it states: the group that places the denoiser is this rank's. A page where
    /// two groups place a denoiser does not say which rank is which, and is refused rather than
    /// guessed at.
    #[test]
    fn the_group_is_the_one_that_places_the_denoiser_when_the_launch_name_is_not_on_the_page() {
        let placed = groups(
            "[worker_groups.components.transformer]\nsequence_parallel = { kind = \"ulysses\", \
             ulysses_degree = 4, projection = \"exchange\" }\n",
        );
        let (ulysses, _, _, sequence_rank) =
            media_parallel(Some(&placed), "engine", 1, 4, 1, 4).expect(
                "the group that places the denoiser is this rank's, whatever the launch calls it",
            );
        assert_eq!(ulysses.degree, 4, "its split is the one this rank serves");
        assert_eq!(sequence_rank, 1);

        let two = page(
            "[[worker_groups]]\nid = \"text\"\nranks = [\"cuda:0-3\"]\nqueue_depth = 2\n\
             [worker_groups.components.text_encoder]\n\
             [[worker_groups]]\nid = \"flow\"\nranks = [\"cuda:0-3\"]\nqueue_depth = 2\n\
             [worker_groups.components.transformer]\n\
             [[worker_groups]]\nid = \"other\"\nranks = [\"cuda:0-3\"]\nqueue_depth = 2\n\
             [worker_groups.components.transformer]\n",
        );
        let error = media_parallel(Some(&two), "engine", 0, 1, 0, 1)
            .expect_err("two groups placing a denoiser is not one partition");
        assert!(
            format!("{error:#}").contains("flow") && format!("{error:#}").contains("other"),
            "the refusal names both groups: {error:#}"
        );
    }

    /// The rank's sequence split is the deployment's, and it is read from the page that states it.
    ///
    /// The defect this pins: the arm refused every rank above one because the build config carries no
    /// sequence degree, and the alternative - a partition of `1` over four ranks - would have served
    /// four whole denoisers. The page states the degree beside the component it splits, so the rank
    /// that reads it serves the split it was launched in.
    #[test]
    fn the_sequence_split_comes_from_the_group_the_rank_serves() {
        let placed = groups(
            "[worker_groups.components.transformer]\nsequence_parallel = { kind = \"ulysses\", \
             ulysses_degree = 4, projection = \"exchange\" }\n\
             [worker_groups.components.text_encoder]\ntensor_parallel_size = 4\n",
        );
        let (ulysses, tensor_parallel, tensor_rank, sequence_rank) =
            media_parallel(Some(&placed), "model", 2, 4, 2, 4)
                .expect("a four-way exchange partition");
        assert_eq!(ulysses.degree, 4, "the page states the degree");
        assert_eq!(
            ulysses.projection,
            minimax_h3::partition::Projection::Exchange,
            "the page states the projection this stack implements"
        );
        assert_eq!((tensor_parallel, tensor_rank), (4, 2), "the tensor place is the rank's");
        assert_eq!(sequence_rank, 2, "a component of every rank places the rank at its ordinal");
    }

    /// A deployment with no worker-group file serves one rank, and refuses to guess above one.
    ///
    /// The shorthand deployment (no `--worker-groups`) states a tensor-parallel width and no
    /// sequence degree. With one rank no other rank can disagree with the shape, so one rank's
    /// sequence is the whole of it; with several, a shape read from the width alone would be an
    /// assumption, and the refusal names the file that would state it.
    #[test]
    fn a_deployment_without_a_file_serves_one_rank_or_refuses() {
        let (ulysses, tensor_parallel, tensor_rank, sequence_rank) =
            media_parallel(None, "model", 0, 1, 0, 1).expect("one rank's sequence");
        assert_eq!(ulysses.degree, 1);
        assert_eq!((tensor_parallel, tensor_rank, sequence_rank), (1, 0, 0));
        let error = media_parallel(None, "model", 2, 4, 2, 4).expect_err("four ranks need a file");
        assert!(
            format!("{error:#}").contains("worker-group file"),
            "the refusal names what would state the shape: {error:#}"
        );
    }

    /// A page that states no split serves one rank's sequence, which is not a refusal.
    #[test]
    fn a_group_with_no_sequence_split_serves_one_rank() {
        let placed = groups("[worker_groups.components.transformer]\n");
        let (ulysses, _, _, sequence_rank) =
            media_parallel(Some(&placed), "model", 1, 1, 0, 1).expect("one rank's sequence");
        assert_eq!(ulysses.degree, 1);
        assert_eq!(sequence_rank, 1);
    }

    /// The two refusals the page can earn, by name rather than served at degree one.
    ///
    /// A gathered projection would be served weights this stack does not load (it holds whole
    /// attention projections on every rank and sends each member its head shard), and two
    /// sequence-split components are two splits where a partition describes one.
    #[test]
    fn a_projection_or_a_second_split_the_stack_cannot_serve_is_refused_by_name() {
        let gather = groups(
            "[worker_groups.components.transformer]\nsequence_parallel = { kind = \"ulysses\", \
             ulysses_degree = 2, projection = \"gather\" }\n",
        );
        let error =
            media_parallel(Some(&gather), "model", 0, 2, 0, 2).expect_err("gather is unimplemented");
        assert!(
            format!("{error:#}").contains("exchange"),
            "the refusal names the mode this stack implements: {error:#}"
        );
        let two = groups(
            "[worker_groups.components.transformer]\nsequence_parallel = { kind = \"ulysses\", \
             ulysses_degree = 2 }\n\
             [worker_groups.components.text_encoder]\nsequence_parallel = { kind = \"ulysses\", \
             ulysses_degree = 2 }\n",
        );
        let error =
            media_parallel(Some(&two), "model", 0, 2, 0, 2).expect_err("one partition, one split");
        assert!(
            format!("{error:#}").contains("transformer") && format!("{error:#}").contains("text_encoder"),
            "the refusal names both components: {error:#}"
        );
    }

    /// One partition serves one task, and the checkpoint's own statement decides which.
    ///
    /// The condition encoders are placed for the conditioned tasks and the condition capacity is that
    /// task's, so a second task is a second partition. A checkpoint that states several is refused
    /// naming them, rather than served with one task's arrangement for another task's request.
    #[test]
    fn the_served_task_is_the_one_the_checkpoint_states() {
        let checkpoint = Path::new("/models/fasth3");
        let one = served_task(&[nn::VideoTask::TextToVideoAudio], checkpoint)
            .expect("a checkpoint that states one task serves it");
        assert_eq!(one, nn::VideoTask::TextToVideoAudio);
        let none = served_task(&[], checkpoint).expect_err("no task is no partition");
        assert!(
            format!("{none:#}").contains("states no task"),
            "the refusal says the checkpoint states none: {none:#}"
        );
        let several = served_task(
            &[
                nn::VideoTask::TextToVideoAudio,
                nn::VideoTask::FirstLastToVideoAudio,
            ],
            checkpoint,
        )
        .expect_err("two tasks are two partitions");
        let sentence = format!("{several:#}");
        assert!(
            sentence.contains("TextToVideoAudio") && sentence.contains("FirstLastToVideoAudio"),
            "the refusal names the tasks: {sentence}"
        );
    }

    /// A plan serves the phases its buckets are launched as, in the scheduler's numbering, and no
    /// phase it does not run.
    ///
    /// The defect this pins: the plan stated `PhaseId(0)` for its device, so the executor refused
    /// the deployment's first completion with `no device of this rank serves phase PhaseId(3)`.
    /// `PhaseId(3)` is a whole-prompt forward and `PhaseId(0)` is text encoding: the plan and the
    /// scheduler were numbering phases in two tables.
    #[test]
    fn a_plan_serves_the_phases_its_buckets_are_planned_as() {
        let buckets = Buckets::causal_lm(&[8], &[64], true);
        let phases = served_phases(&buckets);
        let id = |kind| PhaseId(u16::try_from(engine::phase_id(kind).0).expect("a phase id fits"));
        assert!(
            phases.contains(&id(engine::PhaseKind::Prefill)),
            "a prefill batch routes to this node's device: {phases:?}"
        );
        assert!(
            phases.contains(&id(engine::PhaseKind::Decode)),
            "a decode batch routes to this node's device: {phases:?}"
        );
        assert!(
            !phases.contains(&id(engine::PhaseKind::TextEncoding)),
            "this node encodes no request itself: {phases:?}"
        );
        assert!(
            !phases.contains(&id(engine::PhaseKind::Denoise)),
            "a batch this plan has no bucket for is not served: {phases:?}"
        );
    }

    /// A checkpoint directory holding the files a test states.
    fn checkpoint(files: &[(&str, &str)]) -> tempfile::TempDir {
        let directory = tempfile::tempdir().expect("a temporary checkpoint directory");
        for (name, body) in files {
            std::fs::write(directory.path().join(name), body).expect("writing a checkpoint file");
        }
        directory
    }

    #[test]
    fn a_qwen3_checkpoint_states_its_ends_and_its_tokenizer_names_the_beginning() {
        let directory = checkpoint(&[
            (
                "config.json",
                r#"{"bos_token_id": null, "eos_token_id": 151645}"#,
            ),
            (
                "tokenizer_config.json",
                r#"{"added_tokens_decoder": {
                    "151643": {"content": "<|endoftext|>"},
                    "151645": {"content": "<|im_end|>"}
                }}"#,
            ),
        ]);

        let control = checkpoint_tokens(directory.path()).expect("the checkpoint's control tokens");

        // The architecture states no beginning, so the tokenizer's own end of text is the
        // beginning a text model has; the stated end of generation comes first and the vocabulary's
        // end of text follows it, because a completion that emits it must stop.
        assert_eq!(control.bos, 151643);
        assert_eq!(control.eos, vec![151645, 151643]);
    }

    #[test]
    fn the_stated_beginning_wins_and_the_generation_default_orders_the_ends() {
        let directory = checkpoint(&[
            (
                "config.json",
                r#"{"bos_token_id": 5, "eos_token_id": [3, 2]}"#,
            ),
            ("generation_config.json", r#"{"eos_token_id": 2}"#),
        ]);

        let control = checkpoint_tokens(directory.path()).expect("the checkpoint's control tokens");

        assert_eq!(
            control.bos, 5,
            "the architecture's own beginning is the one it means"
        );
        // `eos[0]` is what the engine gives a completion that commits none, so the generation
        // default leads and the architecture's remaining ids follow, each stated once.
        assert_eq!(control.eos, vec![2, 3]);
    }

    #[test]
    fn a_checkpoint_that_states_no_end_of_generation_is_refused_by_name() {
        let directory = checkpoint(&[("config.json", r#"{"bos_token_id": 5}"#)]);

        let error = checkpoint_tokens(directory.path()).expect_err("an unstated eos token");
        let message = format!("{error:#}");

        assert!(
            message.contains("states no end-of-generation token"),
            "the refusal names what is missing: {message}"
        );
        assert!(
            message.contains("config.json"),
            "the refusal names the file that was read: {message}"
        );
    }

    /// A deployment whose stated token tier is longer than its model is refused before weights,
    /// and the refusal names all three figures: the option, the size it states and the bound.
    #[test]
    fn a_stated_prefill_tier_above_the_model_is_refused_naming_all_three_figures() {
        let refusal = tier_refusal(None, 8, &[256, 512, 1024], 512)
            .expect("a stated prefill tier above the model was accepted");

        assert!(
            refusal.contains("cuda_graph_config.prefill_token_sizes"),
            "the option is not named: {refusal}"
        );
        assert!(
            refusal.contains("states 1024"),
            "the stated figure is not named: {refusal}"
        );
        assert!(
            refusal.contains("above the 512 that max_model_len admits"),
            "the captured figure or its key is not named: {refusal}"
        );
    }

    /// A deployment whose stated decode tiers exceed its request bound still starts: that bound
    /// sizes a batch, nothing checks a policy against it, and the limit moves in place.
    #[test]
    fn a_stated_decode_tier_above_the_request_bound_is_not_refused() {
        assert_eq!(
            tier_refusal(Some(&[1, 2, 4, 8, 16]), 4, &[], 4096),
            None,
            "a decode tier above the request bound was refused, which the live-update deployment \
             states deliberately"
        );
        // And a zero in a stated catalogue is a size of nothing, not a refusal.
        assert_eq!(
            tier_refusal(None, 8, &[0, 4, 512], 512),
            None,
            "a zero in a stated prefill catalogue was refused"
        );
    }

    #[test]
    fn a_text_model_states_understanding_and_no_media_capacity() {
        let limits = understanding_limits();

        assert!(limits.features.contains(GenerationFeatures::UNDERSTANDING));
        assert!(
            !limits
                .features
                .contains(GenerationFeatures::IMAGE_GENERATION)
        );
        assert!(!limits.features.contains(GenerationFeatures::VISION_ENCODE));
        // The two figures the Qwen3 profile states as one, because zero divides and admits nothing.
        assert_eq!(limits.latent_downsample, 1);
        assert_eq!(limits.max_cfg_branches, 1);
        assert_eq!(limits.max_latent_units, 0);
        assert_eq!(limits.max_vit_grid_tokens, 0);
    }
}

#[cfg(test)]
mod class_tests {
    use super::ModelClass;

    /// A checkpoint's class is read from whichever file the checkpoint's own format states it in:
    /// a transformers checkpoint in `config.json`, a diffusers modular pipeline in
    /// `modular_model_index.json` with no `config.json` at its root.
    #[test]
    fn a_class_is_read_from_the_file_that_states_it() {
        let root = std::env::temp_dir().join(format!("class-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();

        std::fs::write(
            root.join("modular_model_index.json"),
            r#"{"_class_name": "MiniMaxH3ModularPipeline"}"#,
        )
        .unwrap();
        assert_eq!(ModelClass::of(&root).unwrap(), ModelClass::Media);

        std::fs::write(root.join("config.json"), r#"{"architectures": ["Qwen3MoeForCausalLM"]}"#)
            .unwrap();
        assert_eq!(ModelClass::of(&root).unwrap(), ModelClass::CausalLm);

        std::fs::remove_file(root.join("config.json")).unwrap();
        std::fs::remove_file(root.join("modular_model_index.json")).unwrap();
        let error = ModelClass::of(&root).unwrap_err().to_string();
        assert!(error.contains("declares no model class"), "{error}");
        assert!(error.contains("modular_model_index.json"), "{error}");
        let _ = std::fs::remove_dir_all(&root);
    }
}
