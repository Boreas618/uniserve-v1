//! A rank's device memory: its graph storage, and the resident owners the capacity plan opens.

use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use uniserve_tensor::DType;
use platform::DeviceId;
use worker::capacity::{graph_scratch_bytes, graph_storage_bytes, quota};
use worker::device::GraphStorage;
use worker::memory::Memory;

use super::report::{MediaBounds, media_rank_report, rank_report};
use super::video::prompt_capacity;
use super::{Loaded, ModelRunner};

/// Tokens one page of a causal language model's paged cache holds: the largest page the attention
/// programs read. The model reads it from the cache it is handed (`nn::KvGroup::page_size`).
pub(crate) const PAGE_TOKENS: u32 = 64;

impl ModelRunner {
    /// The device's graph storage: the buckets it captures and the pools' reservations.
    ///
    /// The figures are capacity planning, so the derivation stays in the worker's capacity
    /// module and this reads its two answers: the graph reservation is the share of a device a
    /// rank may map for captured graphs, and the budget is the rank's quota of the device. The
    /// scratch reservation is a fraction of the graph pool, which the paired forward's page set and
    /// the shadow pair's region are carved from. The resident pool is reserved by whoever
    /// carves it - `Memory`, through `Worker::pools()` - so the graph storage asks for none: a
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
        // The graph reservation is the design's share of the whole device, so its base is
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
        // The shadow region an update's shadow step runs a replaced operator in, plus the state
        // group's snapshot copies units through: one scratch pool, sized once, never allocated per
        // step or per batch.
        let shadow = graph_scratch_bytes(i64::try_from(total)?);
        let scratch = shadow.checked_add(snapshot).with_context(|| {
            format!(
                "device {} reserves a {shadow}-byte shadow region and a {snapshot}-byte state snapshot, more than one scratch pool holds",
                device.0
            )
        })?;
        // Every pool on this device is booked against `budget`, so `budget` is what the device has
        // left to give: the epoch's weights are device-resident retained storage, mapped before
        // this call, and an allowance drawn from the total would let the graph, scratch and
        // resident pools map past the device by exactly the model's size. The deployment's
        // fraction still caps the pools' share of the whole device, and the smaller of the two
        // rules is the allowance: a deployment that asks for a smaller share gets it, and a device
        // already holding the weights gets only what remains.
        let budget = quota(total, self.engine_config().mem_fraction_static).min(storage.free);
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
                self.engine_config().mem_fraction_static,
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
            // `Worker::spawn` therefore refuses a device whose runners outnumber the streams the
            // range was divided for, by name, before any capture.
            streams,
            scratch,
            // The shadow region alone, which the paired forward's twins allocate; `scratch` keeps
            // the reservation that also holds the state group's snapshot.
            shadow,
            // The KV pool: the units the capacity plan fitted, each the size of a cache unit. This
            // rank keeps its keys and values in the process - that is what a prefix cache is - so
            // the figure is the plan's and not zero. A pool whose figure is zero is not reserved,
            // and a KV provider that asks for it is refused by name.
            resident,
        })
    }

    /// Opens the rank's resident owners from the capacity plan.
    ///
    /// Called after the epoch's model is loaded and before any thread that borrows the owners: the
    /// plan measures a device the weights are already mapped on, so what it computes is what is
    /// left, not what was there before the load. Every owner is booked against the device's one
    /// allowance as it maps, which is what makes the first `mapped <= budget` reading after warmup
    /// count every byte the plan charged. The device's pool set is opened here, first, because the
    /// owners carve their storage from it and the runner is built from them.
    pub(crate) fn open_memory(
        &self,
        loaded: &Loaded,
        storage: &mut dyn worker::memory::retained::ExclusiveStorage,
        group: &worker::parallel::host::HostGroup,
    ) -> Result<(Memory, ipc::MainProcessInfo, GraphStorage)> {
        let device = *self
            .devices
            .first()
            .context("this rank serves no device to open its resident memory on")?;
        // The rank's host identity is the daemon's, from the launch file: the storage claim, the
        // deployment file's host names and the scheduler's storages are keyed by it, and a rank
        // that reported the machine's name instead would read its own Storage Server as another
        // host's wherever the two names differ.
        let host = self.launch.host.as_str();
        // What one batch may be, in the class's own terms: a causal-language rank admits up to the
        // largest bucket its catalog captured (`Graphs` refuses anything larger before a kernel
        // runs, so the catalog is its bound and not an estimate), and a media rank admits the
        // deployment's requests at the widest prompt capacity it captures.
        let (rows, tokens) = match loaded {
            Loaded::Text(_) => self.buckets.bounds(),
            Loaded::Video(_) => {
                let longest = self.engine_config().model.max_model_len;
                (
                    u32::try_from(self.engine_config().max_running_requests.get())
                        .context("more request slots than the protocol addresses")?,
                    prompt_capacity(&self.engine_config().video, longest)?,
                )
            }
        };
        // What the rank states to the memory planner, and what it reports of itself. The two classes
        // differ in the cache: a rank that keeps a paged cache states the one its model reads, and a
        // rank that keeps none states its absence.
        let (request, report) = match loaded {
            Loaded::Text(model) => {
                // The KV layout is the model's attention layer count, the heads each layer stores on
                // this rank and the width of a head, in the deployment's pages; a model that keeps a
                // recurrent state per request slot adds the state group its checkpoints fill.
                let shape = model.cache();
                let dtype = cache_dtype(model.dtype(), self.engine_config().kv_cache.dtype)?;
                let state = model.state();
                let cache = match &state {
                    None => worker::memory::CachePlan::dense(
                        PAGE_TOKENS,
                        shape.layers,
                        shape.kv_heads,
                        shape.head_dim,
                        dtype,
                    )
                    .validated(),
                    Some(state) => worker::memory::CachePlan::hybrid(
                        PAGE_TOKENS,
                        shape.layers,
                        shape.kv_heads,
                        shape.head_dim,
                        dtype,
                        state,
                    ),
                }
                .context("describing the model's KV cache")?;
                // A deployment that speculates keeps its drafter's draft cache in the same pool.
                let drafts = self.engine_config().speculative.num_draft_tokens;
                let cache = match model.drafter().filter(|_| drafts > 0) {
                    Some(drafter) => {
                        let shape = drafter.cache();
                        cache
                            .with_draft(shape.layers, shape.kv_heads, shape.head_dim)
                            .context("describing the drafter's draft cache")?
                    }
                    None => cache,
                };
                // A deployment that reads other hosts' host KV cache pages (`kv_cache_config.peers`)
                // into an FP8 pool also loads the pages they computed in the dense compute dtype, by
                // converting them as they load (OD-3: never the other direction). A pool whose
                // groups have no conversion (a state or draft group) loads only its own shape.
                let kv = &self.engine_config().kv_cache;
                let cache =
                    if kv.host_tier && kv.prefix_caching && !kv.peers.is_empty() && cache.quantized
                    {
                        match cache.clone().with_conversion() {
                            Ok(converting) => converting,
                            Err(why) => {
                                tracing::info!(%why, "the KV cache loads no converted peer page");
                                cache
                            }
                        }
                    } else {
                        cache
                    };
                // The request slots' state banks are the runner's, carved from the resident pool
                // beside the cache.
                let slots = u32::try_from(self.engine_config().max_running_requests.get())
                    .context("more request slots than the protocol addresses")?;
                // A speculating rank's state also keeps a verification's trail: every slot's drafts
                // and the token before them.
                let trail = if drafts > 0 {
                    slots.saturating_mul(drafts + 1)
                } else {
                    0
                };
                let state_bytes = state.as_ref().map_or(0, |state| {
                    worker::runners::state::bank_bytes(state, slots, trail)
                });
                // A model that takes images keeps its feature slots beside the banks.
                let feature_bytes = image_features(model.as_ref())
                    .map_or(0, |(row_bytes, rows)| {
                        worker::runners::vision::feature_bytes(rows, row_bytes)
                    });
                let request = worker::capacity::MemoryRequest {
                    device: worker::config::Device::Cuda(device),
                    epoch: self.cache_epoch.unwrap_or(self.launch.epoch),
                    queue_depth: self.queue_depth(),
                    max_batch_calls: rows as usize,
                    max_batch_tokens: tokens as usize,
                    cache: Some(Arc::new(cache)),
                    segments: self.adopted_cache.clone(),
                    transport_bytes: self.transport_bytes(device, model.as_ref())?,
                    carved_bytes: state_bytes + feature_bytes,
                };
                let images = image_features(model.as_ref()).is_some();
                (request, rank_report(device, host, drafts > 0, images)?)
            }
            Loaded::Video(pipeline) => {
                // No cache: a video rank addresses no paged request.
                let request = worker::capacity::MemoryRequest {
                    device: worker::config::Device::Cuda(device),
                    epoch: self.launch.epoch,
                    queue_depth: self.queue_depth(),
                    max_batch_calls: rows as usize,
                    max_batch_tokens: tokens as usize,
                    cache: None,
                    segments: None,
                    transport_bytes: 0,
                    carved_bytes: 0,
                };
                let longest = pipeline.timeline(self.engine_config().video.max_seconds)?;
                let states = proto::tensor::DType::from(self.engine_config().model.dtype);
                let bounds = MediaBounds {
                    prompt_tokens: self.engine_config().model.max_model_len,
                    video_units: u32::try_from(pipeline.video_units(longest)?.len())
                        .context("more decode units than the protocol addresses")?,
                    conditions: pipeline.condition_bounds(),
                    conditioning_row: (states, pipeline.text_width()),
                };
                (
                    request,
                    media_rank_report(device, host, bounds, pipeline.denoiser_stage())?,
                )
            }
        };
        // The ranks of this process agree over the group the main process formed at start, and a
        // world of one rank is a group of one member.
        let planned = worker::capacity::plan_rank(self.engine_config(), &request, report, group)?;
        // The keys and values the plan fitted are the process's resident pool: a unit is a cache
        // unit, and the plan charged them all against the same device allowance the pools are
        // booked against. A plan that fitted no cache (a rank that serves none) states zero.
        let resident = match (&planned.cache, planned.info.prefix_cache.as_ref()) {
            // The recurrent state's banks are carved from the same pool, beside the block tables.
            (Some(cache), Some(info)) => {
                u64::from(info.num_units) * cache.unit_bytes + request.carved_bytes
            }
            // A media rank keeps no cache, and its resident pool holds the tables its model
            // declares per layout. The reservation is addresses and not storage: a table books the
            // allowance as it maps, so the range is what the device has left to give.
            _ if matches!(loaded, Loaded::Video(_)) => {
                platform::Device::new(device)
                    .with_context(|| format!("opening device {}", device.0))?
                    .memory_info()?
                    .free
            }
            _ => 0,
        };
        let snapshot = planned.state_snapshot_bytes();
        let graph = self.graph(device, resident, snapshot, Self::streams_on(loaded))?;
        let pools = worker::device::Worker::open(device, &graph)?;
        let stream = platform::Stream::new(device)
            .with_context(|| format!("creating device {}'s stream", device.0))?;
        let memory = worker::memory::Memory::new(
            worker::memory::OpenMemory {
                plan: &planned,
                device: Some(device),
                stream: Some(&stream),
                pools,
            },
            storage,
        )
        .context("opening the rank's resident memory")?;
        // The plan's own account of this rank is what the scheduler is told about its pool: the
        // capacities it fitted, the cache it holds and where it is. It is the same `MainProcessInfo` the
        // deployment's other ranks report, so the scheduler plans against one description of a
        // rank rather than a second one derived here. A media rank adds what its denoiser serves,
        // which the frontend plans a request against and no capacity figure implies.
        let mut info = planned.info.clone();
        // A text rank's sampler reports the deployment's candidates at every position it scores
        // (`CausalLmRunner::reporting`), which bounds what a request may ask for.
        if let Loaded::Text(_) = loaded {
            info.max_logprobs = self.engine_config().max_logprobs;
        }
        // A rank that takes images states its encodings: the feature row an image block's
        // encoding writes, and the slots that hold them, each the largest image's.
        if let Loaded::Text(model) = loaded
            && let Some((row_bytes, rows)) = image_features(model.as_ref())
        {
            info.image_encoders.insert(
                ipc::product::VISION_FEATURES.to_owned(),
                ipc::ImageEncoderOutput {
                    row_bytes,
                    draws: false,
                },
            );
            let entry = u64::from(rows) * row_bytes;
            info.encoder_entry_bytes = entry;
            info.encoder_cache_entries = worker::runners::vision::FEATURE_ENTRIES;
            info.buffer_pool_bytes = entry * u64::from(worker::runners::vision::FEATURE_ENTRIES);
        }
        // Where the planes the Storage Server retains hold each unit, which a copy between
        // processes (a prefill/decode hand-off) addresses.
        if let (Some(cache), Some(prefix_cache)) =
            (info.prefix_cache.as_mut(), memory.prefix_cache.as_ref())
        {
            let prefix_cache = prefix_cache
                .lock()
                .map_err(|_| anyhow::anyhow!("the prefix cache's lock is poisoned"))?;
            cache.plane_runs = prefix_cache.planes().runs().to_vec();
        }
        if let Loaded::Video(pipeline) = loaded {
            let description = pipeline.description(&self.video_serving()?)?;
            info.video_denoiser = Some(ipc::VideoDenoiserInfo {
                takes_conditions: description.takes_conditions,
                denoise_steps: description.denoise_steps,
                description: description.details,
            });
        }
        Ok((memory, info, graph))
    }

    /// The streams `device` maps graphs on: one per runner it serves, plus one for a preparation.
    ///
    /// The count is the number of `NamedRunner`s [`ModelRunner::runners`] builds for this class, and
    /// the two are checked against each other where the runners exist: `Worker::spawn` refuses a
    /// device whose runners outnumber the streams its graph range was divided for, naming both
    /// figures. That refusal is what keeps this constant honest - a runner added without the count
    /// following it stops at the spawn instead of at a capture nobody can read.
    ///
    /// A preparation walks its own generation on a stream of its own, so it takes one more range
    /// than the serving generation does, whether or not an update is in flight when the process
    /// starts: a range that appeared only at the first precopy would be mapped after the model and
    /// the cache, and the device would have nothing left to give it.
    pub(super) fn streams_on(loaded: &Loaded) -> usize {
        let serving = match loaded {
            // A text rank serves one causal-language runner.
            Loaded::Text(_) => 1,
            // A video rank serves at most its encoder, its denoise step, its decoder, its audio
            // decoder and its host stage: five runners, one stream each.
            Loaded::Video(_) => 5,
        };
        serving + 1
    }
}

/// What a model's image input occupies on its rank: the bytes of one feature row and the rows of
/// the largest image's encoding; `None` for a model without a vision tower.
pub(super) fn image_features(model: &dyn nn::family::TextModel) -> Option<(u64, u32)> {
    let vision = model.vision()?;
    let row_bytes = u64::from(model.hidden()) * ops::abi::bytes(model.dtype())?;
    Some((row_bytes, vision.max_image_rows()))
}

/// The element type the cache pool stores, from the deployment when it states one.
///
/// An unstated cache dtype is the model's own activation dtype, which is what its layers compute
/// in until the deployment asks for a narrower cache. A model whose activations have no cache
/// element type cannot have its pool planned, and says so rather than taking the nearest.
pub(super) fn cache_dtype(
    activations: DType,
    stated: Option<proto::request::PrefixCacheDtype>,
) -> Result<proto::request::PrefixCacheDtype> {
    if let Some(dtype) = stated {
        return Ok(dtype);
    }
    match activations {
        DType::BF16 => Ok(proto::request::PrefixCacheDtype::BFloat16),
        DType::F16 => Ok(proto::request::PrefixCacheDtype::Float16),
        DType::F32 => Ok(proto::request::PrefixCacheDtype::Float32),
        DType::F8E4M3 => Ok(proto::request::PrefixCacheDtype::Float8E4m3Fn),
        other => bail!("this build plans no KV cache in a model whose activations are {other:?}"),
    }
}
