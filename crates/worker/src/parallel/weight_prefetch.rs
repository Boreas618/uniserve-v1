//! The worker's share of layer weights copied into device slots ahead of
//! their use: the memory and the byte movement.
//!
//! Two owners stream weights this way. Distributed expert weights (DWDP,
//! [`DwdpManager`]) read the other ranks' shards of each expert layer from
//! peer memory into two reusable slots; host weights ([`HostWeights`])
//! stream the projections of a repeated layer stack from page-locked host
//! memory into a ring of device slots. Here the worker allocates and maps
//! that memory and, for DWDP, plans each layer's peer reads as one batch of
//! copies. The library's schedule (`uniserve.weight_prefetch`) owns the rest:
//! which layer goes to which slot, the copy stream and the events that order
//! copies against the layers' reads, and, under graph capture, each copy as
//! the graph's break work. The worker releases the memory only after the
//! schedule drained its copy stream.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cuda::{CopyBatch, DeviceAllocation, DeviceId, PeerMapping, SourceOrder, Stream, memcpy_async};
use indexmap::IndexMap;
use ipc::ComponentWeights;
use tvm_ffi::{Any, AnyView};

use super::world::World;
use crate::config::placement::Components;
use crate::config::{Device, WorkerLaunch};
use crate::ffi::{Backing, DType, Handle, Interpreter};
use crate::memory::allocation::{Storage, StorageView};
use crate::stream::Streams;
use crate::worker::LoadedModel;
use crate::{Error, Result, lock};

/// One sharded expert-weight field of one layer: this rank's contiguous
/// shard of `rows` experts (the field's leading axis), on the device.
pub(crate) struct ShardField {
    /// The field's name, unique within its layer (`up_gate.values`).
    pub name: String,
    /// Device address and bytes of this rank's shard.
    pub address: u64,
    pub bytes: usize,
}

/// Storage that must outlive every copy and every reader of the composite
/// weight views: published shards, reusable remote slots and the composite
/// mappings.
pub(crate) struct WeightStorage {
    published: Vec<crate::transfer::PeerStorage>,
    slots: Vec<DeviceAllocation>,
    composites: Vec<Arc<Storage>>,
}

/// Distributed expert weights: every rank keeps its expert shard resident
/// and reads the other ranks' shards from peer memory into two reusable
/// slots per field before each layer computes.
///
/// The mechanism follows TensorRT-LLM's composite-VA weight buffer. For each
/// field, every rank publishes a page-granular allocation holding its shard
/// (all ranks the same size, so every peer maps them back to back); a
/// composite virtual range maps `[pre slot | resident pages | post slot]`
/// contiguously, so the numerical layer reads the full field as one
/// contiguous tensor. The slots of layer `i` are shared with layer `i + 2`.
/// Bytes of peers' shards inside this rank's resident pages (page edges)
/// are copied once at setup; every other remote byte is copied by layer
/// `i`'s plan, which the library's schedule submits before the layer
/// computes. Only setup is collective: inference waits on local copy and
/// compute events alone.
pub(crate) struct ExpertWeights {
    device: DeviceId,
    /// Layer `i`'s peer reads into its slots, as one batch.
    plans: Vec<CopyBatch>,
    /// Every address the plans and the composite views use; `None` once
    /// released.
    storage: Mutex<Option<WeightStorage>>,
    /// Bytes of resident pages, reusable slots and per-invocation copies.
    pub resident_bytes: usize,
    pub buffer_bytes: usize,
    pub transfer_bytes: usize,
}

impl ExpertWeights {
    /// Publish and map every layer's fields, collectively over `group`.
    ///
    /// `layers[i]` holds layer `i`'s sharded fields in the same order on
    /// every rank. Returns the owner and, per layer and field, the full
    /// field as a contiguous byte view (`bytes * group size` long). Setup
    /// copies run on `stream`, which is drained before returning. On
    /// failure every mapping is retained until the process exits, because
    /// peers may already read published pages.
    pub(crate) fn build(
        group: &super::host::HostGroup,
        device: DeviceId,
        host: &str,
        layers: &[Vec<ShardField>],
        stream: &Stream,
    ) -> Result<(Self, Vec<Vec<StorageView>>)> {
        let mut storage = WeightStorage {
            published: Vec::new(),
            slots: Vec::new(),
            composites: Vec::new(),
        };
        match Self::map(group, device, host, layers, stream, &mut storage) {
            Ok((copies, views, (resident_bytes, buffer_bytes, transfer_bytes))) => Ok((
                Self {
                    device,
                    plans: copies.into_iter().map(plan).collect(),
                    storage: Mutex::new(Some(storage)),
                    resident_bytes,
                    buffer_bytes,
                    transfer_bytes,
                },
                views,
            )),
            Err(error) => {
                // Peers may read the pages a failed setup published.
                crate::leak(storage);
                Err(error)
            }
        }
    }

    #[expect(clippy::type_complexity, reason = "one setup's complete result")]
    fn map(
        group: &super::host::HostGroup,
        device: DeviceId,
        host: &str,
        layers: &[Vec<ShardField>],
        stream: &Stream,
        storage: &mut WeightStorage,
    ) -> Result<(
        Vec<Vec<(usize, usize, usize, usize)>>,
        Vec<Vec<StorageView>>,
        (usize, usize, usize),
    )> {
        let (rank, size) = (group.rank(), group.size());
        let page = DeviceAllocation::granularity(device)?;
        let mut slots: HashMap<(usize, String, &'static str, usize), usize> = HashMap::new();
        let mut copies = Vec::with_capacity(layers.len());
        let mut views = Vec::with_capacity(layers.len());
        let (mut resident_bytes, mut buffer_bytes, mut transfer_bytes) = (0, 0, 0);
        for (index, fields) in layers.iter().enumerate() {
            let mut layer_copies = Vec::new();
            let mut layer_views = Vec::new();
            for field in fields {
                let shard = field.bytes;
                let total = shard * size;
                let (begin, end) = (rank * shard, (rank + 1) * shard);
                let first = begin / page * page;
                // Every rank publishes the same extent, even when its shard
                // starts at another offset within a page; fabric handles map
                // only whole allocations.
                let capacity = (0..size)
                    .map(|peer| (peer * shard % page + shard).div_ceil(page) * page)
                    .max()
                    .unwrap_or(page);
                let last = first + capacity;
                let published =
                    crate::transfer::PeerStorage::allocate(device, capacity, host, group)?;
                if published.bytes_each() != capacity {
                    return Err(Error::Invariant(
                        "published expert pages disagree with their capacity".into(),
                    ));
                }
                let local = published
                    .local()
                    .allocation()
                    .ok_or_else(|| Error::Invariant("published pages are device memory".into()))?
                    .address();
                // SAFETY: the shard is a live device tensor of `shard` bytes
                // the caller keeps through this setup; the destination lies
                // inside this rank's published allocation.
                unsafe {
                    memcpy_async(local + (begin - first) as u64, field.address, shard, stream)
                }?;
                stream.synchronize()?;
                // Publish: no rank writes its resident pages again.
                group.barrier()?;
                resident_bytes += capacity;

                let mut parts: Vec<usize> = Vec::new();
                let post = total.div_ceil(page) * page;
                for (side, length) in [("pre", first), ("post", post.saturating_sub(last))] {
                    if length == 0 {
                        continue;
                    }
                    let key = (index % 2, field.name.clone(), side, length);
                    let slot = match slots.get(&key) {
                        Some(&slot) => slot,
                        None => {
                            storage
                                .slots
                                .push(DeviceAllocation::exportable(device, length, &[])?);
                            buffer_bytes += length;
                            slots.insert(key, storage.slots.len() - 1);
                            storage.slots.len() - 1
                        }
                    };
                    parts.push(slot);
                }
                let resident = published
                    .local()
                    .allocation()
                    .ok_or_else(|| Error::Invariant("published pages are device memory".into()))?;
                let mut pieces: Vec<&DeviceAllocation> = Vec::new();
                let mut part = parts.iter();
                if first > 0 {
                    pieces.push(
                        &storage.slots[*part.next().ok_or_else(|| {
                            Error::Invariant("expert weight pre slot is missing".into())
                        })?],
                    );
                }
                pieces.push(resident);
                if let Some(&slot) = part.next() {
                    pieces.push(&storage.slots[slot]);
                }
                let composite = Storage::mapped(PeerMapping::alias(&pieces)?);
                let whole = composite.address();
                let peers = published.peers().address();

                // Edge bytes share immutable resident pages: copy them once
                // now. Sources always lie inside the peer's own shard.
                for peer in (0..size).filter(|&peer| peer != rank) {
                    let (peer_begin, peer_end) = (peer * shard, (peer + 1) * shard);
                    let source_begin = peers + (peer * capacity + peer_begin % page) as u64;
                    for (low, high, persistent) in [
                        (0, first, false),
                        (first, begin, true),
                        (end, last.min(total), true),
                        (last, total, false),
                    ] {
                        let (start, stop) = (low.max(peer_begin), high.min(peer_end));
                        if start >= stop {
                            continue;
                        }
                        let source = source_begin + (start - peer_begin) as u64;
                        let destination = whole + start as u64;
                        if persistent {
                            // SAFETY: both ranges lie inside live mappings
                            // retained by `storage`.
                            unsafe { memcpy_async(destination, source, stop - start, stream) }?;
                        } else {
                            layer_copies.push((
                                peer,
                                destination as usize,
                                source as usize,
                                stop - start,
                            ));
                            transfer_bytes += stop - start;
                        }
                    }
                }
                stream.synchronize()?;
                layer_views.push(StorageView::contiguous(
                    Arc::clone(&composite) as Arc<dyn Backing>,
                    0,
                    DType::U8,
                    &[total as i64],
                )?);
                storage.composites.push(composite);
                storage.published.push(published);
            }
            copies.push(layer_copies);
            views.push(layer_views);
        }
        Ok((
            copies,
            views,
            (resident_bytes, buffer_bytes, transfer_bytes),
        ))
    }

    /// Submit layer `layer`'s peer reads into its slots on `stream`. No host
    /// synchronization is performed.
    ///
    /// The caller orders the reads after the last reader of the slots they
    /// overwrite, and drains `stream` before the weights are released.
    fn submit(&self, layer: usize, stream: &Stream) -> Result<()> {
        // Held through the submission, so a release cannot interleave.
        let storage = lock(&self.storage);
        if storage.is_none() {
            return Err(Error::State("distributed expert weights are closed"));
        }
        let plan = self
            .plans
            .get(layer)
            .ok_or_else(|| Error::Invalid(format!("expert layer {layer} has no copy plan")))?;
        // SAFETY: `storage` retains every address the plan uses until it is
        // released, after the caller drained the streams it submitted on.
        unsafe { plan.submit(stream) }?;
        Ok(())
    }

    /// Take the storage for destruction outside the owner, once no copy
    /// can still address it. Later submissions fail.
    fn release(&self) -> Option<WeightStorage> {
        lock(&self.storage).take()
    }
}

/// Page-interleave one layer's peer reads into one batch.
fn plan(copies: Vec<(usize, usize, usize, usize)>) -> CopyBatch {
    // As in TensorRT-LLM's batched DWDP reads, interleave 2 MiB slices from
    // different peers. Destinations are disjoint, allowing DMA pipelining.
    let mut peers = IndexMap::<usize, Vec<(usize, usize, usize)>>::new();
    const CHUNK: usize = 2 << 20;

    for (peer, destination, source, bytes) in copies {
        let slices = peers.entry(peer).or_default();
        for offset in (0..bytes).step_by(CHUNK) {
            slices.push((
                destination + offset,
                source + offset,
                CHUNK.min(bytes - offset),
            ));
        }
    }

    let rounds = peers.values().map(Vec::len).max().unwrap_or(0);
    let mut batch = CopyBatch::new(SourceOrder::Stream);
    for round in 0..rounds {
        for &(destination, source, bytes) in peers.values().filter_map(|peer| peer.get(round)) {
            batch.push(destination as u64, source as u64, bytes);
        }
    }
    batch
}

/// The library module of expert exchanges and distributed expert weights.
const MODULE: &str = "uniserve.distributed.experts";

// Distributed expert weights of a rank (DWDP): setup is collective over the
// expert union, after loading and before capacity planning. `weight_fields`
// places each layer's weights for its kernels and lists this rank's shards;
// [`ExpertWeights::build`] publishes them and maps every field's full
// extent; `bind_weights` installs the full views as the layers' parameters,
// and `bind_prefetch` returns the library's prefetch schedule over the
// weights' copy submission, which execution contexts pass to their expert
// calls.

/// The rank's distributed expert weights and their prefetch schedule.
pub(crate) struct DwdpManager {
    weights: Arc<ExpertWeights>,
    /// `bind_prefetch`'s object (`WeightPrefetch`), which the executor
    /// passes to every binding's expert calls.
    prefetch: Handle,
    closed: bool,
}

/// A worker error raised back into the calling numerical code.
fn ffi_error(error: &Error) -> tvm_ffi::Error {
    tvm_ffi::Error::new(tvm_ffi::RUNTIME_ERROR, &error.to_string(), "")
}

fn count(value: &AnyView<'_>) -> std::result::Result<usize, tvm_ffi::Error> {
    let value = i64::try_from(*value)?;
    usize::try_from(value)
        .map_err(|_| tvm_ffi::Error::new(tvm_ffi::VALUE_ERROR, "expected a count", ""))
}

impl DwdpManager {
    /// Set up the rank's distributed expert weights, collectively over the
    /// expert union. `None` unless the union prefetches weights.
    pub(crate) fn start(
        launch: &WorkerLaunch,
        world: &World,
        model: &LoadedModel,
        streams: &Streams,
        host: &str,
    ) -> Result<Option<Self>> {
        let Some(union) = &launch.expert_parallel else {
            return Ok(None);
        };
        if !union.prefetches_weights() {
            return Ok(None);
        }
        let Device::Cuda(device) = launch.device() else {
            return Err(Error::Unsupported("DWDP requires CUDA peer memory".into()));
        };
        let stream = streams
            .preparation(device)
            .ok_or(Error::State("DWDP requires a CUDA stream"))?;
        let stream = stream.stream()?.clone();
        let union_group = world
            .union
            .as_ref()
            .ok_or(Error::State("DWDP requires its union's host group"))?;

        // The number of fields of each layer, then every field's name and
        // shard, the layers back to back.
        let function = format!("{MODULE}.weight_fields");
        let mismatch = |error: tvm_ffi::Error| Error::numerical(&function, &error);
        let listed: tvm_ffi::Array<Any> =
            Interpreter::import(&function)?.call_as(&[model.handle().view()])?;
        let part = |index: usize| listed.get(index).map_err(mismatch);
        let counts = crate::ffi::integers(part(0)?).map_err(mismatch)?;
        let names = tvm_ffi::Array::<tvm_ffi::String>::try_from(part(1)?).map_err(mismatch)?;
        // Shard tensors stay referenced by `shards` through the setup copies.
        let shards = tvm_ffi::Array::<tvm_ffi::Tensor>::try_from(part(2)?).map_err(mismatch)?;
        let mut fields = names.iter().zip(shards.iter()).map(|(name, shard)| {
            let bits = usize::from(shard.dtype().bits) * usize::from(shard.dtype().lanes);
            ShardField {
                name: name.as_str().to_owned(),
                address: shard.data_ptr() as u64,
                bytes: shard.numel() * bits / 8,
            }
        });
        let layers: Vec<Vec<ShardField>> = counts
            .iter()
            .map(|&count| fields.by_ref().take(count as usize).collect())
            .collect();
        let (weights, views) = ExpertWeights::build(union_group, device, host, &layers, &stream)?;
        drop(shards);
        let bind = Interpreter::import(&format!("{MODULE}.bind_weights"))?;
        for (layer, (fields, views)) in layers.iter().zip(views).enumerate() {
            // The views' wrappings and the call take one handoff.
            Interpreter::execute(|| -> Result<()> {
                let mut arguments = vec![
                    model.handle().to_any(),
                    Any::from(layer as i64),
                    Any::from(crate::ffi::strings(fields.iter().map(|field| &field.name))),
                ];
                for view in &views {
                    arguments.push(view.handle()?.to_any());
                }
                bind.call_values(&arguments).map(drop)
            })?;
        }
        tracing::info!(
            "DWDP expert weights: {} layers, {} resident and {} slot bytes, {} bytes read per invocation",
            layers.len(),
            weights.resident_bytes,
            weights.buffer_bytes,
            weights.transfer_bytes,
        );

        let weights = Arc::new(weights);
        let prefetch = Handle::new(
            Interpreter::import(&format!("{MODULE}.bind_prefetch"))?
                .call_values(&[model.handle().to_any(), Any::from(copy_function(&weights))])?,
        )?;
        Ok(Some(Self {
            weights,
            prefetch,
            closed: false,
        }))
    }

    /// The object bindings pass to their expert calls.
    pub(crate) fn prefetch(&self) -> &Handle {
        &self.prefetch
    }

    /// Drain the schedule's copies and release the weights after every
    /// binding and graph retired. An aborted close keeps every mapping
    /// until exit: peers may still read this rank's pages.
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        if std::mem::replace(&mut self.closed, true) {
            return Ok(());
        }
        if aborted {
            crate::leak(Arc::clone(&self.weights));
            return Ok(());
        }
        let drained = Interpreter::import(&format!("{MODULE}.WeightPrefetch.close"))
            .and_then(|close| close.invoke(&[self.prefetch.view()]));
        let storage = self.weights.release();
        if drained.is_err() {
            // Copies may still read peers' pages and write the slots.
            crate::leak(storage);
        }
        drained
    }
}

impl Drop for DwdpManager {
    fn drop(&mut self) {
        let _ = self.close(true);
    }
}

/// `bind_prefetch`'s `copy(layer, stream)` as a TVM-FFI function closed over
/// `weights`: it submits layer `layer`'s peer reads on the CUDA stream
/// handle `stream` and blocks on no peer. Graph replays call it as break
/// work, so it holds the weights until the schedule and its graphs release
/// it.
fn copy_function(weights: &Arc<ExpertWeights>) -> tvm_ffi::Function {
    let weights = Arc::clone(weights);
    tvm_ffi::Function::from_packed(move |args: &[AnyView<'_>]| {
        let [layer, stream] = args else {
            return Err(ffi_error(&Error::Invalid(
                "copy takes a layer and a stream".into(),
            )));
        };
        // SAFETY: the schedule names its copy stream on the weights' device,
        // which it keeps until it drained it at close.
        let stream = unsafe { crate::stream::external(weights.device, count(stream)?) }
            .map_err(|error| ffi_error(&error))?;
        weights
            .submit(count(layer)?, &stream)
            .map(|()| Any::default())
            .map_err(|error| ffi_error(&error))
    })
}

/// The library module of host-resident layer weights.
const HOST_WEIGHTS: &str = "uniserve.loading.host_weights";

/// Where the host regions of host-weight stacks come from when the
/// worker's weights are retained at the Storage Server.
pub(crate) enum HostRetention<'a> {
    /// The worker's weights are not retained: each region is page-locked
    /// memory of the worker's own.
    None,
    /// The worker loaded: each region is new retained host memory that joins
    /// its published weights, filled by binding.
    Publish(&'a mut crate::memory::weights::Published),
    /// The worker bound retained weights: each region is a retained host
    /// storage among these, already holding the matrices.
    Bound(&'a [Arc<Storage>]),
}

/// The dense projection weights of one repeated layer stack, kept in
/// page-locked host memory and streamed through a ring of device slots.
///
/// The worker allocates the host region and the slots; the library rebinds
/// every layer's projection matrices to views of its slot and copies the
/// layers into the slots ahead of their projection calls, on a stream of its
/// own (`uniserve.loading.host_weights.HostStack`). The library's tensors
/// retain the memory; the worker drains the stack's copies at close.
pub(crate) struct HostWeights {
    /// The library's `HostStack` of the stack.
    stack: Handle,
}

impl HostWeights {
    /// Drain the stack's copies after every reader retired; the host region
    /// and the slots go with the library's last reference to them. A failed
    /// drain keeps the stack, and with it the memory, until exit.
    pub(crate) fn close(self) -> Result<()> {
        let drained = Interpreter::import(&format!("{HOST_WEIGHTS}.HostStack.close"))
            .and_then(|close| close.invoke(&[self.stack.view()]));
        if drained.is_err() {
            // Copies may still write the slots.
            crate::leak(self);
        }
        drained
    }

    /// Stream the layer stacks of this rank's components placed with
    /// `"weights": "host"`.
    ///
    /// Runs after loading and before any numerical operator binds the
    /// layers' parameters and before capacity planning, so the device
    /// memory the streamed matrices release becomes capacity. Each outermost
    /// declared module path of such a component streams its repeated layer
    /// stack.
    ///
    /// # Errors
    /// [`Error::Unsupported`] when a component off CUDA keeps host weights or
    /// a module cannot stream its layers.
    pub(crate) fn start(
        model: &LoadedModel,
        components: &Components,
        retention: &mut HostRetention<'_>,
    ) -> Result<Vec<Self>> {
        let mut owners = Vec::new();
        for placed in components.placed() {
            if placed.config.weights != ComponentWeights::Host {
                continue;
            }
            let Device::Cuda(device) = placed.device else {
                return Err(Error::Unsupported(format!(
                    "component {} streams host weights to a CUDA device, not {:?}",
                    placed.name, placed.device
                )));
            };
            for path in model.declarations.roots(&placed.name) {
                let owner = Self::stream(model, device, &path, retention).map_err(|error| {
                    Error::Unsupported(format!(
                        "component {} cannot stream {path:?} from host memory: {error}",
                        placed.name
                    ))
                })?;
                owners.push(owner);
            }
        }
        Ok(owners)
    }

    fn stream(
        model: &LoadedModel,
        device: DeviceId,
        path: &str,
        retention: &mut HostRetention<'_>,
    ) -> Result<Self> {
        let at_path = || Any::from(tvm_ffi::String::from(path));
        // `name(model, path)` of the library module, with the function's
        // name for errors.
        let call = |name: &str| -> Result<(String, Any)> {
            let function = format!("{HOST_WEIGHTS}.{name}");
            let value = Interpreter::import(&function)?
                .call_values(&[model.handle().to_any(), at_path()])?;
            Ok((function, value))
        };
        let integer = |name: &str| -> Result<i64> {
            let (function, value) = call(name)?;
            value
                .try_into()
                .map_err(|error| Error::numerical(&function, &error))
        };
        let count = |value: i64| {
            usize::try_from(value)
                .map_err(|_| Error::Interface("a streamed stack's layout holds counts".into()))
        };
        let layout = || -> Result<(usize, usize)> {
            let (function, value) = call("layout")?;
            let layout =
                crate::ffi::integers(value).map_err(|error| Error::numerical(&function, &error))?;
            let [layers, layer_bytes] = layout[..] else {
                return Err(Error::Interface(
                    "a streamed stack's layout names its layers and layer bytes".into(),
                ));
            };
            Ok((count(layers)?, count(layer_bytes)?))
        };

        // One page-locked region of exactly every layer's bytes: retained
        // at the Storage Server when the worker's weights are, and then
        // already holding the matrices when the worker bound them.
        let (host, layers, layer_bytes, filled) = match retention {
            HostRetention::Bound(storages) => {
                let layers = count(integer("layers")?)?;
                let address = integer("address")? as u64;
                let host = storages
                    .iter()
                    .find(|storage| {
                        storage.address() <= address
                            && address < storage.address() + storage.bytes() as u64
                    })
                    .cloned()
                    .ok_or_else(|| {
                        Error::Interface(
                            "no retained host region holds the bound streamed matrices".into(),
                        )
                    })?;
                if layers == 0 || host.bytes() % layers != 0 {
                    return Err(Error::Interface(
                        "a retained host region disagrees with its layer count".into(),
                    ));
                }
                let layer_bytes = host.bytes() / layers;
                (host, layers, layer_bytes, true)
            }
            HostRetention::Publish(published) => {
                let (layers, layer_bytes) = layout()?;
                let (host, descriptor) = Storage::host_retainable(device, layers * layer_bytes)?;
                published.add_host(model.handle(), path, Arc::clone(&host), descriptor)?;
                (host, layers, layer_bytes, false)
            }
            HostRetention::None => {
                let (layers, layer_bytes) = layout()?;
                let host = Storage::pinned(device, layers * layer_bytes)?;
                (host, layers, layer_bytes, false)
            }
        };

        // As many device slots as the library's ring streams the stack
        // through.
        let slots = Interpreter::import(&format!("{HOST_WEIGHTS}.slots"))?
            .call_as::<i64>(&[(&Any::from(layers as i64)).into()])?;
        let slots = (0..count(slots)?)
            .map(|_| Storage::device(device, layer_bytes))
            .collect::<Result<Vec<_>>>()?;
        let view = |memory: &Arc<Storage>| {
            StorageView::whole(Arc::clone(memory) as Arc<dyn Backing>, DType::U8)
        };
        let host_view = view(&host)?;
        let slot_views = slots.iter().map(view).collect::<Result<Vec<_>>>()?;

        let bind = Interpreter::import(&format!("{HOST_WEIGHTS}.bind"))?;
        // The views' wrappings and the call take one handoff.
        let stack = Interpreter::execute(|| -> Result<Handle> {
            let mut arguments = vec![
                model.handle().to_any(),
                at_path(),
                Any::from(filled),
                host_view.handle()?.to_any(),
            ];
            for slot in &slot_views {
                arguments.push(slot.handle()?.to_any());
            }
            Handle::new(bind.call_values(&arguments)?)
        })?;
        tracing::info!(
            "streaming {layers} layers of {path:?} from host memory through {} device slots of {layer_bytes} bytes",
            slots.len()
        );
        Ok(Self { stack })
    }
}
