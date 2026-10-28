//! The worker's share of layer weights copied into device slots ahead of
//! their use: the storage and the byte movement.
//!
//! Two owners stream weights this way. Distributed expert weights (DWDP,
//! [`DwdpManager`]) read the other ranks' shards of each expert layer from
//! peer storage into two reusable slots; host weights ([`HostWeights`])
//! stream the projections of a repeated layer stack from page-locked host
//! storage into a ring of device slots. Here the worker allocates and maps
//! that storage and, for DWDP, plans each layer's peer reads as one batch of
//! copies. The library's schedule (`uniserve.weight_prefetch`) owns the rest:
//! which layer goes to which slot, the copy stream and the events that order
//! copies against the layers' reads, and, under graph capture, each copy as
//! the graph's break work. The worker releases the storage only after the
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
use crate::storage::allocation::{Storage, StorageView};
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
    shards: Vec<Vec<(StorageView, usize)>>,
    copies: Vec<Vec<(usize, usize, usize, usize)>>,
    plans: Vec<CopyBatch>,
}

/// Distributed expert weights: every rank keeps its expert shard resident
/// and reads the other ranks' shards from peer storage into two reusable
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
            shards: Vec::new(),
            copies: Vec::new(),
            plans: Vec::new(),
        };
        match Self::map(group, device, host, layers, stream, &mut storage) {
            Ok((copies, views, (resident_bytes, buffer_bytes, transfer_bytes))) => {
                storage.plans = copies.iter().cloned().map(plan).collect();
                storage.copies = copies;
                Ok((
                    Self {
                        device,
                        storage: Mutex::new(Some(storage)),
                        resident_bytes,
                        buffer_bytes,
                        transfer_bytes,
                    },
                    views,
                ))
            }
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
            let mut shard_views = Vec::new();
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
                    .ok_or_else(|| Error::Invariant("published pages are device storage".into()))?
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
                    .ok_or_else(|| Error::Invariant("published pages are device storage".into()))?;
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
                shard_views.push((
                    StorageView::whole(
                        Arc::clone(published.local()) as Arc<dyn Backing>,
                        DType::U8,
                    )?,
                    begin - first,
                ));
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
            storage.shards.push(shard_views);
        }
        // Every peer has finished the persistent edge copies before a rank
        // can publish or discard this completed generation.
        group.barrier()?;
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
        let storage = storage
            .as_ref()
            .ok_or(Error::State("distributed expert weights are closed"))?;
        let plan = storage
            .plans
            .get(layer)
            .ok_or_else(|| Error::Invalid(format!("expert layer {layer} has no copy plan")))?;
        // SAFETY: `storage` retains every address the plan uses until it is
        // released, after the caller drained the streams it submitted on.
        unsafe { plan.submit(stream) }?;
        Ok(())
    }

    /// Borrow the true resident shards, before full composite tensors replace
    /// the model's local parameters. Numerical manifests use these views;
    /// reusable prefetch slots never describe immutable source weights.
    fn resident(&self) -> Result<Vec<Vec<(Handle, usize)>>> {
        let storage = lock(&self.storage);
        let storage = storage
            .as_ref()
            .ok_or(Error::State("distributed expert weights are closed"))?;
        storage
            .shards
            .iter()
            .map(|layer| {
                layer
                    .iter()
                    .map(|(view, offset)| Ok((view.handle()?.clone(), *offset)))
                    .collect()
            })
            .collect()
    }

    pub(crate) fn resident_storages(&self) -> Result<Vec<Arc<Storage>>> {
        let storage = lock(&self.storage);
        let storage = storage
            .as_ref()
            .ok_or(Error::State("distributed expert weights are closed"))?;
        Ok(storage
            .published
            .iter()
            .map(|field| Arc::clone(field.local()))
            .collect())
    }

    /// Install a fully prepared successor at the existing composite addresses.
    /// All ranks must drain compute and prefetch before entering this local
    /// operation, and resume only after every rank committed. The successor's
    /// peer mappings and creating allocations stay owned by this instance.
    pub(crate) fn replace(&self, successor: Self) -> Result<()> {
        let mut active = lock(&self.storage);
        let current = active
            .as_ref()
            .ok_or(Error::State("distributed expert weights are closed"))?;
        let mut incoming = successor
            .release()
            .ok_or(Error::State("the successor expert weights are live"))?;
        if current.composites.len() != incoming.composites.len()
            || current
                .composites
                .iter()
                .zip(&incoming.composites)
                .any(|(target, source)| target.bytes() != source.bytes())
        {
            return Err(Error::Invalid(
                "expert replacement must preserve field extents".into(),
            ));
        }
        for copies in &mut incoming.copies {
            for (_, destination, _, _) in copies {
                let (target, source) = current
                    .composites
                    .iter()
                    .zip(&incoming.composites)
                    .find(|(_, source)| {
                        let start = source.address() as usize;
                        (start..start + source.bytes()).contains(destination)
                    })
                    .ok_or(Error::State("a peer read writes a composite weight field"))?;
                *destination =
                    target.address() as usize + (*destination - source.address() as usize);
            }
        }
        incoming.plans = incoming.copies.iter().cloned().map(plan).collect();
        for (target, source) in current.composites.iter().zip(&incoming.composites) {
            if let Err(error) = target.remap_alias(source) {
                // Other ranks may already have installed this generation's
                // peer mappings. Keep their creating handles until exit.
                crate::leak(incoming);
                return Err(error);
            }
        }
        incoming.composites.clone_from(&current.composites);
        *active = Some(incoming);
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
    update: Arc<DwdpUpdate>,
    /// `bind_prefetch`'s object (`WeightPrefetch`), which the executor
    /// passes to every binding's expert calls.
    prefetch: Handle,
    closed: bool,
}

/// Online redistribution's access to the distributed weight owner.
/// Its collective channel and stream are independent of serving, so staging
/// a successor does not park token execution or interleave control messages.
pub(crate) struct DwdpUpdate {
    weights: Arc<ExpertWeights>,
    group: super::host::HostGroup,
    host: String,
    stream: Stream,
    layout: Handle,
    names: Vec<Vec<String>>,
}

impl DwdpUpdate {
    /// Additional physical bytes while both generations coexist. Staging
    /// includes reusable prefetch slots as well as immutable resident pages.
    pub(crate) fn required_bytes(&self) -> usize {
        self.weights.resident_bytes + self.weights.buffer_bytes
    }

    /// The layout and owning allocations of the live resident expert shards.
    pub(crate) fn layout(&self, orders: &str) -> Result<Handle> {
        Handle::new(
            Interpreter::import("uniserve.loading.expert_weights.ResidentExperts.describe")?
                .call_values(&[
                    self.layout.to_any(),
                    Any::from(tvm_ffi::String::from(orders)),
                ])?,
        )
    }

    pub(crate) fn storages(&self) -> Result<Vec<Arc<Storage>>> {
        self.weights.resident_storages()
    }
    /// Build a successor from completed borrowed shard views. Collective over
    /// every expert rank; run concurrently on all ranks on background threads.
    /// The caller retains the input allocations until this method returns.
    pub(crate) fn prepare(&self, retained: &Handle) -> Result<ExpertWeights> {
        let listed: tvm_ffi::Array<Any> =
            Interpreter::import("uniserve.loading.expert_weights.ResidentExperts.fields")?
                .call_values(&[self.layout.to_any(), retained.to_any()])?
                .try_into()
                .map_err(|error| Error::numerical("distributed expert fields", &error))?;
        let fields = shard_fields(&listed)?;
        let (weights, _) = ExpertWeights::build(
            &self.group,
            self.weights.device,
            &self.host,
            &fields,
            &self.stream,
        )?;
        Ok(weights)
    }

    /// Break a failed generation's collective I/O before retiring the worker.
    /// Collective failure requires group recovery; this channel is not reused.
    pub(crate) fn cancel(&self) {
        self.group.close();
    }

    /// Publish a staged generation locally at a common drained step boundary.
    /// This call does not perform a collective; the engine holds all ranks
    /// until each has installed its aliases and logical routing tables.
    pub(crate) fn commit(&self, successor: ExpertWeights) -> Result<()> {
        self.weights.replace(successor)?;
        bind_resident(&self.layout, &self.names, self.weights.resident()?)
    }
}

fn shard_fields(listed: &tvm_ffi::Array<Any>) -> Result<Vec<Vec<ShardField>>> {
    let mismatch = |error: tvm_ffi::Error| Error::numerical("distributed expert fields", &error);
    let part = |index: usize| listed.get(index).map_err(mismatch);
    let counts = crate::ffi::integers(part(0)?).map_err(mismatch)?;
    let names = tvm_ffi::Array::<tvm_ffi::String>::try_from(part(1)?).map_err(mismatch)?;
    let shards = tvm_ffi::Array::<tvm_ffi::Tensor>::try_from(part(2)?).map_err(mismatch)?;
    let mut fields = names.iter().zip(shards.iter()).map(|(name, shard)| {
        let bits = usize::from(shard.dtype().bits) * usize::from(shard.dtype().lanes);
        ShardField {
            name: name.as_str().to_owned(),
            address: shard.data_ptr() as u64,
            bytes: shard.numel() * bits / 8,
        }
    });
    Ok(counts
        .iter()
        .map(|&count| fields.by_ref().take(count as usize).collect())
        .collect())
}

fn bind_resident(
    layout: &Handle,
    names: &[Vec<String>],
    fields: Vec<Vec<(Handle, usize)>>,
) -> Result<()> {
    let bind = Interpreter::import("uniserve.loading.expert_weights.ResidentExperts.bind")?;
    for (index, (names, fields)) in names.iter().zip(fields).enumerate() {
        let mut arguments = vec![
            layout.to_any(),
            Any::from(index as i64),
            Any::from(crate::ffi::strings(names)),
            Any::from(crate::ffi::integer_list(
                fields.iter().map(|(_, offset)| *offset as i64),
            )),
        ];
        arguments.extend(fields.iter().map(|(view, _)| view.to_any()));
        bind.call_values(&arguments)?;
    }
    Ok(())
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
            return Err(Error::Unsupported("DWDP requires CUDA peer storage".into()));
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
        let listed: tvm_ffi::Array<Any> =
            Interpreter::import(&function)?.call_as(&[model.handle().view()])?;
        let layers = shard_fields(&listed)?;
        let (weights, views) = ExpertWeights::build(union_group, device, host, &layers, &stream)?;
        drop(listed);
        let resident = Handle::new(
            Interpreter::import("uniserve.loading.expert_weights.ResidentExperts")?
                .call_values(&[model.handle().to_any()])?,
        )?;
        let names: Vec<Vec<String>> = layers
            .iter()
            .map(|fields| fields.iter().map(|field| field.name.clone()).collect())
            .collect();
        bind_resident(&resident, &names, weights.resident()?)?;
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
        let rendezvous = union
            .rendezvous
            .as_ref()
            .ok_or(Error::State("DWDP requires its union rendezvous"))?;
        let update = Arc::new(DwdpUpdate {
            weights: Arc::clone(&weights),
            group: union_group.duplicate(&rendezvous.address)?,
            host: host.to_owned(),
            stream: Stream::new(device)?,
            layout: resident,
            names,
        });
        let prefetch = Handle::new(
            Interpreter::import(&format!("{MODULE}.bind_prefetch"))?
                .call_values(&[model.handle().to_any(), Any::from(copy_function(&weights))])?,
        )?;
        Ok(Some(Self {
            weights,
            update,
            prefetch,
            closed: false,
        }))
    }

    /// The object bindings pass to their expert calls.
    pub(crate) fn prefetch(&self) -> &Handle {
        &self.prefetch
    }

    pub(crate) fn update(&self) -> Arc<DwdpUpdate> {
        Arc::clone(&self.update)
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
    /// storage of the worker's own.
    None,
    /// The worker loaded: each region is new retained host storage that joins
    /// its published weights, filled by binding.
    Publish(&'a mut crate::storage::weights::Published),
    /// The worker bound retained weights: each region is a retained host
    /// storage among these, already holding the matrices.
    Bound(&'a [Arc<Storage>]),
}

/// The dense projection weights of one repeated layer stack, kept in
/// page-locked host storage and streamed through a ring of device slots.
///
/// The worker allocates the host region and the slots; the library rebinds
/// every layer's projection matrices to views of its slot and copies the
/// layers into the slots ahead of their projection calls, on a stream of its
/// own (`uniserve.loading.host_weights.HostStack`). The library's tensors
/// retain the storage; the worker drains the stack's copies at close.
pub(crate) struct HostWeights {
    /// The library's `HostStack` of the stack.
    stack: Handle,
}

impl HostWeights {
    /// Drain the stack's copies after every reader retired; the host region
    /// and the slots go with the library's last reference to them. A failed
    /// drain keeps the stack, and with it the storage, until exit.
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
    /// storage the streamed matrices release becomes capacity. Each outermost
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
                        "component {} cannot stream {path:?} from host storage: {error}",
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
        let view = |storage: &Arc<Storage>| {
            StorageView::whole(Arc::clone(storage) as Arc<dyn Backing>, DType::U8)
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
            "streaming {layers} layers of {path:?} from host storage through {} device slots of {layer_bytes} bytes",
            slots.len()
        );
        Ok(Self { stack })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parallel::host::{HostGroup, Rendezvous, STARTUP_TIMEOUT};
    use cuda::{PinnedBuffer, memset_d8};
    use std::net::TcpListener;

    #[test]
    #[ignore = "requires two CUDA devices"]
    fn distributed_weights_replacement_preserves_borrowed_fields() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let ranks = [
            Rendezvous::serve(&address, listener),
            Rendezvous::connect(&address),
        ];
        let threads: Vec<_> = ranks
            .into_iter()
            .enumerate()
            .map(|(rank, rendezvous)| {
                std::thread::spawn(move || -> Result<()> {
                    let device = DeviceId(rank as i32);
                    let group = HostGroup::connect(rendezvous, rank, 2, STARTUP_TIMEOUT)?;
                    let page = DeviceAllocation::granularity(device)?;
                    // A shard boundary inside a page exercises persistent neighbor
                    // bytes as well as the reusable remote prefetch slots.
                    let bytes = page + 256;
                    let stream = Stream::new(device)?;
                    let make =
                        |owner: usize| -> Result<(Vec<DeviceAllocation>, Vec<Vec<ShardField>>)> {
                            let allocations = (0..4)
                                .map(|layer| {
                                    let allocation = DeviceAllocation::linear(device, bytes)?;
                                    // SAFETY: the new allocation owns the filled range.
                                    unsafe {
                                        memset_d8(
                                            allocation.address(),
                                            (11 + owner * 4 + layer) as u8,
                                            bytes,
                                            &stream,
                                        )
                                    }?;
                                    Ok(allocation)
                                })
                                .collect::<Result<Vec<_>>>()?;
                            stream.synchronize()?;
                            let fields = allocations
                                .iter()
                                .map(|allocation| {
                                    vec![ShardField {
                                        name: "up_gate.weight".into(),
                                        address: allocation.address(),
                                        bytes,
                                    }]
                                })
                                .collect();
                            Ok((allocations, fields))
                        };
                    let (original, fields) = make(rank)?;
                    let (weights, views) =
                        ExpertWeights::build(&group, device, "local", &fields, &stream)?;
                    drop(original);
                    let host = PinnedBuffer::new(device, bytes * 2)?;
                    let verify =
                        |swapped: bool| -> Result<()> {
                            for (layer, fields) in views.iter().enumerate() {
                                weights.submit(layer, &stream)?;
                                // SAFETY: the owner retains every borrowed field and
                                // the readback is drained before host inspection.
                                unsafe {
                                    memcpy_async(
                                        host.address(),
                                        fields[0].address(),
                                        bytes * 2,
                                        &stream,
                                    )
                                }?;
                                stream.synchronize()?;
                                // SAFETY: the preceding stream synchronization completed
                                // every write to this pinned buffer.
                                let observed = unsafe { host.as_slice() };
                                for peer in 0..2 {
                                    let logical = if swapped { peer ^ 1 } else { peer };
                                    assert!(observed[peer * bytes..(peer + 1) * bytes].iter()
                                .all(|value| *value == (11 + logical * 4 + layer) as u8));
                                }
                            }
                            Ok(())
                        };
                    verify(false)?;
                    let (incoming, fields) = make(rank ^ 1)?;
                    let (successor, _) =
                        ExpertWeights::build(&group, device, "local", &fields, &stream)?;
                    drop(incoming);
                    group.barrier()?;
                    weights.replace(successor)?;
                    group.barrier()?;
                    verify(true)?;
                    // No rank retires peer-readable allocations while another
                    // rank's final readback is still in flight.
                    group.barrier()?;
                    Ok(())
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap().unwrap();
        }
    }
}
