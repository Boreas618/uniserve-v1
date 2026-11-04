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
use executor::capacity::{graph_storage_budget_bytes, quota};
use executor::device::GraphStorage;
use executor::memory::Memory;
use executor::plan::{Buckets, Variant};
use executor::runner::{NamedRunner, RunnerId};
use executor::runners::causal_lm::{CausalLmRunner, Geometry, KvProvider, Shared};
use executor::runners::resident::ResidentCache;
use executor::stream::EventPool;
use ipc::PhaseId;
use ops::DType;
use ops::Generation;
use platform::DeviceId;
use proto::request::GraphPolicy;
use program_server::Target;

use crate::engine_main::Launch;

/// One epoch's build config, resolved against the node the main process runs on.
pub struct EpochPlan {
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
}

impl EpochPlan {
    /// The epoch `build` builds, with this node's arguments resolved against it.
    ///
    /// The devices and the Program Server socket are the node's, not the epoch's: the daemon
    /// starts every rank of a node with the ordinals that rank serves, and a main process that
    /// guessed its devices from the epoch's parallelism would capture graphs on another rank's
    /// device.
    pub fn of(launch: &Launch, build: &daemon::epoch::BuildConfig) -> Result<Self> {
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
        })
    }

    /// The epoch's build config, for an accessor that needs another of its fields.
    pub fn build(&self) -> &daemon::epoch::BuildConfig {
        &self.build
    }

    /// The checkpoint this rank serves.
    pub fn checkpoint(&self) -> &Path {
        &self.checkpoint
    }

    /// The bucket catalog every device captures, built once from `cuda_graph`.
    ///
    /// A catalog is a property of the epoch, not of a device, so `graph` and `runners` share it
    /// rather than each building one: two catalogs of one epoch would be two answers to what a
    /// batch may be, and a batch that fits one but not the other would have no graph to replay.
    pub fn buckets(&self) -> &Buckets {
        &self.buckets
    }

    /// The blocking pools' admission bounds, one lane per kind (§2.4).
    ///
    /// The build config sizes no pool - a blocking lane's width follows the work it carries, not
    /// the epoch - so the bounds are derived from what the node has: the checkpoint's own reader
    /// count for the lane that maps weights and installs programs, and the host's parallelism for
    /// the lanes that read transfers, reclaim storage and handle media. A lane whose bound is
    /// narrower than the work in flight queues it; it never drops it.
    pub fn blocking_limits(&self) -> Limits {
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
    pub fn program_server(&self) -> &Path {
        &self.program_server
    }

    /// This rank's devices, in the order the daemon named them.
    pub fn devices(&self) -> &[DeviceId] {
        &self.devices
    }

    /// The device's completion events: one pool per rank, shared by its executors (§2.4).
    pub fn events(&self) -> Arc<EventPool> {
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
    pub fn graph(&self, device: DeviceId) -> Result<GraphStorage> {
        let handle = platform::Device::new(device)
            .with_context(|| format!("opening device {}", device.0))?;
        let total = handle.total_memory();
        let bytes = graph_storage_budget_bytes(i64::try_from(total)?);
        if bytes == 0 {
            bail!("device {} reports no memory to capture graphs in", device.0);
        }
        let scratch = bytes / 8;
        let budget = quota(total, self.build.mem_fraction_static);
        if budget < bytes + scratch {
            bail!(
                "device {} grants {budget} bytes at mem_fraction_static {}, less than the graph and \
                 scratch reservation of {}",
                device.0,
                self.build.mem_fraction_static,
                bytes + scratch
            );
        }
        Ok(GraphStorage {
            bytes,
            budget,
            // One stream per device in this epoch: one causal-language runner serves every bucket
            // of it, so the graph reservation is not divided among streams.
            streams: 1,
            scratch,
            resident: 0,
        })
    }

    /// How long a job may run without rank progress before it is reported `Stalled` (§2.4).
    pub fn watchdog_timeout(&self) -> Duration {
        self.build.worker.watchdog_timeout
    }

    /// The Program Server's targets this rank resolves a module for (§5.3).
    pub fn targets(&self) -> &[Target] {
        &self.targets
    }

    /// The runners `device` serves: one per block of the loaded model (§4.3).
    ///
    /// This is the single match on model class in the tree. Everything downstream of it - the
    /// buckets, the arena, the sampler, the graphs - is written against `nn`'s capability traits,
    /// so a second causal-language class joins as a variant here and changes nothing else.
    pub fn runners(
        &self,
        device: DeviceId,
        model: Arc<qwen3_moe::Qwen3Moe>,
        memory: &mut Memory,
        generation: &Arc<Generation>,
    ) -> Result<Vec<NamedRunner>> {
        let class = architectural_class(&self.checkpoint)?;
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
        match class.as_str() {
            "Qwen3MoeForCausalLM" => {
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
                    max_kv_len: self.max_kv_len(&model)?,
                };
                let stream = platform::Stream::new(device)
                    .with_context(|| format!("creating device {}'s stream", device.0))?;
                let rows = self
                    .buckets
                    .largest(Variant::DecodeGreedy)
                    .map_or(1, |bucket| bucket.rows);
                let provider: Box<dyn KvProvider> =
                    Box::new(ResidentCache::new(memory.shared_cache(), device, rows)?);
                let runner = CausalLmRunner::new(
                    RunnerId(0),
                    vec![PhaseId(0)],
                    Shared::new(model),
                    Arc::clone(generation),
                    self.buckets.clone(),
                    device,
                    stream,
                    geometry,
                    provider,
                    // A seed that changes with the process, so two runs of one deployment are not
                    // the same run.
                    seed(),
                )?;
                runners.push(NamedRunner::new(RunnerId(0), Box::new(runner)));
            }
            other => bail!(
                "{} declares model class {other}, which this build cannot serve",
                self.checkpoint.display()
            ),
        }
        Ok(runners)
    }

    /// The keys a row reads at most: the cache's bound, from the epoch's context length.
    fn max_kv_len(&self, model: &qwen3_moe::Qwen3Moe) -> Result<u32> {
        match self.build.max_model_len {
            Some(length) => Ok(length.get()),
            None => u32::try_from(model.config().max_position_embeddings)
                .context("the checkpoint's context length is wider than the protocol's"),
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
pub(crate) fn rank_report(device: DeviceId) -> Result<executor::capacity::Report> {
    let at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos() as u64);
    Ok(executor::capacity::Report {
        endpoint: ipc::WorkerEndpoint {
            host: host_name()?,
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
        components: vec![ipc::ComponentInfo {
            name: "backbone".to_owned(),
            outputs: Vec::new(),
        }],
    })
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
    let cache = memory.shared_cache();
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

/// The model class a checkpoint declares, as its `config.json` states it.
fn architectural_class(checkpoint: &Path) -> Result<String> {
    let path = checkpoint.join("config.json");
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let config: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("parsing {}", path.display()))?;
    let class = config
        .get("architectures")
        .and_then(|architectures| architectures.as_array())
        .and_then(|architectures| architectures.first())
        .and_then(|class| class.as_str())
        .ok_or_else(|| anyhow::anyhow!("{} declares no model class", path.display()))?;
    Ok(class.to_owned())
}
