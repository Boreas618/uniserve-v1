//! Layer weights copied into a ring of device slots ahead of their use.
//!
//! Two owners share one mechanism, [`SlotCopies`]: a copy stream, one copy
//! plan per layer and, per slot, an event the copy records once the slot is
//! readable and an event a computing stream records once it has read the
//! slot. Layer `i` lives in slot `i % slots`. Distributed expert weights
//! (DWDP, [`DwdpManager`]) read the other ranks' shards of the next expert
//! layer from peer memory into two slots; host weights ([`HostWeights`])
//! stream the projections of a repeated layer stack from page-locked host
//! memory into a few slots.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, PoisonError};

use indexmap::IndexMap;
use uniserve_cuda::{
    CaptureStatus, CopyBatch, DeviceGuard, DeviceId, Event, EventFlags, SourceOrder, Stream,
};
use uniserve_ipc::ComponentWeights;

use std::sync::Arc;

use tvm_ffi::{Any, AnyView};

use super::world::World;
use crate::config::placement::Components;
use crate::config::{Device, WorkerLaunch};
use crate::cuda_graph::GraphBreaks;
use crate::ffi::{Backing, DType, Function, Handle, Interpreter};
use crate::storage::memory::{Storage, StorageView};
use crate::stream::Streams;
use crate::worker::LoadedModel;
use crate::{Error, Result, lock};

/// Copies of each layer's weights into its slot, ordered by events against
/// the streams that read the slots.
///
/// `copies[i]` fills layer `i`'s slot. A copy waits for the slot's last
/// recorded read and records the slot's readiness; a reader waits for that
/// readiness and records its read. `backing` retains every address the
/// copies use until [`close`](Self::close) drains the copy stream.
struct SlotCopies<B> {
    device: DeviceId,
    copies: Vec<CopyBatch>,
    stream: Stream,
    /// Recorded on the copy stream once a slot holds its newest copy.
    ready: Vec<Event>,
    /// Recorded on a computing stream once it has read a slot.
    consumed: Vec<Event>,
    /// `None` once closed.
    backing: Mutex<Option<B>>,
}

impl<B> SlotCopies<B> {
    fn new(device: DeviceId, slots: usize, copies: Vec<CopyBatch>, backing: B) -> Result<Self> {
        if slots == 0 || copies.is_empty() {
            return Err(Error::Invalid(
                "slot copies need a slot and one copy plan per layer".into(),
            ));
        }
        // External events order the slots across CUDA graph segments.
        let external = || {
            Event::new(EventFlags {
                external: true,
                ..EventFlags::default()
            })
        };
        Ok(Self {
            device,
            copies,
            stream: Stream::new(device)?,
            ready: (0..slots).map(|_| external()).collect(),
            consumed: (0..slots).map(|_| external()).collect(),
            backing: Mutex::new(Some(backing)),
        })
    }

    fn slot(&self, index: usize) -> usize {
        index % self.ready.len()
    }

    fn closed(&self) -> bool {
        lock(&self.backing).is_none()
    }

    /// Copy layer `index` into its slot after the slot's last read. No host
    /// synchronization is performed.
    fn copy(&self, index: usize) -> Result<()> {
        if self.closed() {
            return Err(Error::State("layer weight copies are closed"));
        }
        let copy = self
            .copies
            .get(index)
            .ok_or_else(|| Error::Invalid("layer is outside the copy plan".into()))?;
        let slot = self.slot(index);
        self.stream.wait_event(&self.consumed[slot])?;
        // SAFETY: `backing` retains every source and destination until close
        // drains this stream.
        unsafe { copy.submit(&self.stream) }?;
        self.ready[slot].record(&self.stream)?;
        Ok(())
    }

    /// Make layer `index`'s slot readable on `compute`.
    fn wait(&self, index: usize, compute: &Stream) -> Result<()> {
        compute.wait_event(&self.ready[self.slot(index)])?;
        Ok(())
    }

    /// Release layer `index`'s slot after its reads on `compute`.
    fn release(&self, index: usize, compute: &Stream) -> Result<()> {
        self.consumed[self.slot(index)].record(compute)?;
        Ok(())
    }

    /// Order every slot's next copy after the work `compute` has queued.
    fn release_all(&self, compute: &Stream) -> Result<()> {
        for event in &self.consumed {
            event.record(compute)?;
        }
        Ok(())
    }

    /// The caller retires the slots' readers first. Drain the copy stream
    /// and return the backing, for destruction outside owner locks.
    fn close(&self) -> Result<Option<B>> {
        if self.closed() {
            return Ok(None);
        }
        self.stream.synchronize()?;
        Ok(lock(&self.backing).take())
    }
}

impl<B> Drop for SlotCopies<B> {
    fn drop(&mut self) {
        // After a failed close the copies' completion is unknown: the memory
        // they address stays until the process exits.
        if self.close().is_err() {
            let backing = self
                .backing
                .get_mut()
                .unwrap_or_else(PoisonError::into_inner);
            crate::leak(backing.take());
        }
    }
}

#[derive(Default)]
struct Call {
    started: bool,
    queued: Option<usize>,
    last: Option<usize>,
}

/// Prefetch the next expert layer into alternating remote-weight slots.
///
/// The numerical backend supplies immutable source mappings and destination
/// views. Calls serialize on their execution stream. External events protect
/// both copy-before-read and read-before-overwrite across CUDA graph segments.
pub(crate) struct WeightPrefetch {
    layers: HashMap<usize, usize>,
    slots: SlotCopies<WeightStorage>,
    calls: Mutex<Vec<Call>>,
}

impl WeightPrefetch {
    /// Copy tuples are `(peer, destination, source, bytes)`, grouped by layer.
    /// `backing` retains every allocation used by those addresses.
    pub(crate) fn new(
        device: DeviceId,
        layers: Vec<usize>,
        copies: Vec<Vec<(usize, usize, usize, usize)>>,
        backing: WeightStorage,
    ) -> Result<Self> {
        if layers.len() != copies.len() || layers.is_empty() {
            return Err(Error::Invalid(
                "each expert layer needs one copy plan".into(),
            ));
        }

        Ok(Self {
            layers: layers
                .into_iter()
                .enumerate()
                .map(|(i, layer)| (layer, i))
                .collect(),
            // The slots of layer `i` serve layer `i + 2`.
            slots: SlotCopies::new(device, 2, copies.into_iter().map(plan).collect(), backing)?,
            calls: Mutex::new(Vec::new()),
        })
    }

    fn calls(&self) -> MutexGuard<'_, Vec<Call>> {
        lock(&self.calls)
    }

    /// The device the weights and their copy stream live on.
    pub(crate) fn device(&self) -> DeviceId {
        self.slots.device
    }

    fn index(&self, module: usize) -> Result<usize> {
        self.layers
            .get(&module)
            .copied()
            .ok_or_else(|| Error::Invalid("module has no prefetched expert weights".into()))
    }

    pub(crate) fn begin(&self) -> Result<()> {
        if self.slots.closed() {
            return Err(Error::State("DWDP weight owner is closed"));
        }
        self.calls().push(Call::default());
        Ok(())
    }

    /// Join a speculative tail copy even if the call omitted its last layer.
    pub(crate) fn end(&self, compute: &Stream) -> Result<()> {
        let last = self
            .calls()
            .pop()
            .ok_or(Error::State("DWDP expert call requires an active context"))?
            .last;
        if let Some(last) = last {
            self.slots.wait(last, compute)?;
        }
        Ok(())
    }

    /// `submit` either copies now or records a copy between graph segments.
    /// It runs outside the state lock and may call back into this owner; its
    /// failure is returned as is.
    pub(crate) fn before(
        &self,
        module: usize,
        compute: &Stream,
        mut submit: impl FnMut(usize) -> Result<()>,
    ) -> Result<()> {
        let index = self.index(module)?;
        let (started, queued) = {
            let mut calls = self.calls();
            let call = calls
                .last_mut()
                .ok_or(Error::State("DWDP expert call requires an active context"))?;
            let started = std::mem::replace(&mut call.started, true);
            (started, call.queued)
        };

        // A default-stream handle resolves against the current context.
        let _device = DeviceGuard::new(self.device())?;
        if !started {
            self.slots.release_all(compute)?;
        }

        if queued != Some(index) {
            self.prefetch(index, &mut submit)?;
        }

        self.slots.wait(index, compute)?;

        let next = (index + 1 < self.slots.copies.len()).then_some(index + 1);
        self.calls()
            .last_mut()
            .ok_or(Error::State("DWDP expert call requires an active context"))?
            .queued = next;
        if let Some(next) = next {
            return self.prefetch(next, &mut submit);
        }
        Ok(())
    }

    fn prefetch(&self, index: usize, submit: &mut impl FnMut(usize) -> Result<()>) -> Result<()> {
        self.calls()
            .last_mut()
            .ok_or(Error::State("DWDP expert call requires an active context"))?
            .last = Some(index);
        submit(index)
    }

    pub(crate) fn after(&self, module: usize, compute: &Stream) -> Result<()> {
        let index = self.index(module)?;
        self.slots.release(index, compute)
    }

    /// Replay a recorded peer read, preserving the same two-slot dependencies
    /// as eager execution. No host synchronization is performed.
    pub(crate) fn copy(&self, index: usize) -> Result<()> {
        if self.slots.closed() {
            return Err(Error::State("DWDP weight owner is closed"));
        }
        self.slots.copy(index)
    }

    /// The caller retires numerical readers first. Drain the copy stream and
    /// return backing for destruction outside owner locks and binding borrows.
    pub(crate) fn close(&self) -> Result<Option<WeightStorage>> {
        if !self.calls().is_empty() {
            return Err(Error::State("cannot close active weight prefetch"));
        }
        self.slots.close()
    }
}

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
    slots: Vec<uniserve_cuda::DeviceAllocation>,
    composites: Vec<std::sync::Arc<crate::storage::memory::Storage>>,
}

/// Distributed expert weights: every rank keeps its expert shard resident
/// and reads the other ranks' shards from peer memory into two reusable
/// slots per field, prefetching the next layer while one computes.
///
/// The mechanism follows TensorRT-LLM's composite-VA weight buffer. For each
/// field, every rank publishes a page-granular allocation holding its shard
/// (all ranks the same size, so every peer maps them back to back); a
/// composite virtual range maps `[pre slot | resident pages | post slot]`
/// contiguously, so the numerical layer reads the full field as one
/// contiguous tensor. The slots of layer `i` are shared with layer `i + 2`.
/// Bytes of peers' shards inside this rank's resident pages (page edges)
/// are copied once at setup; every other remote byte is copied before its
/// layer computes. Only setup is collective: inference waits on local copy
/// and compute events alone.
pub(crate) struct ExpertWeights {
    prefetch: WeightPrefetch,
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
    ) -> Result<(Self, Vec<Vec<crate::storage::memory::StorageView>>)> {
        let mut storage = WeightStorage {
            published: Vec::new(),
            slots: Vec::new(),
            composites: Vec::new(),
        };
        let built = Self::map(group, device, host, layers, stream, &mut storage);
        match built {
            Ok((copies, views, (resident_bytes, buffer_bytes, transfer_bytes))) => {
                let prefetch =
                    WeightPrefetch::new(device, (0..layers.len()).collect(), copies, storage)?;
                Ok((
                    Self {
                        prefetch,
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
        Vec<Vec<crate::storage::memory::StorageView>>,
        (usize, usize, usize),
    )> {
        use std::collections::HashMap;
        use std::sync::Arc;

        use uniserve_cuda::{DeviceAllocation, PeerMapping, memcpy_async};

        use crate::ffi::{Backing, DType};
        use crate::storage::memory::{Storage, StorageView};

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

    /// The device the weights live on.
    pub(crate) fn device(&self) -> DeviceId {
        self.prefetch.device()
    }

    /// Open one invocation.
    pub(crate) fn begin(&self) -> Result<()> {
        self.prefetch.begin()
    }

    /// Close one invocation on `compute`, joining a speculative tail copy.
    pub(crate) fn end(&self, compute: &Stream) -> Result<()> {
        self.prefetch.end(compute)
    }

    /// Make layer `layer` readable on `compute` and prefetch the next one.
    ///
    /// Eagerly, the copies are submitted now. On a capturing stream, copies
    /// cannot be recorded: `record(index)` receives each copy in submission
    /// order and must end the current graph segment, schedule
    /// [`copy`](Self::copy) of `index` and begin the next segment. The order
    /// matters: the wait for this layer's slots is captured after the
    /// segment boundary that submits their copy.
    pub(crate) fn before(
        &self,
        layer: usize,
        compute: &Stream,
        mut record: impl FnMut(usize) -> Result<()>,
    ) -> Result<()> {
        let capturing = compute.capture_status()? != uniserve_cuda::CaptureStatus::None;
        self.prefetch.before(layer, compute, |index| {
            if capturing {
                record(index)
            } else {
                self.prefetch.copy(index)
            }
        })
    }

    /// Release layer `layer`'s slots after its last reader on `compute`.
    pub(crate) fn after(&self, layer: usize, compute: &Stream) -> Result<()> {
        self.prefetch.after(layer, compute)
    }

    /// Submit one recorded copy (graph replay between segments).
    pub(crate) fn copy(&self, index: usize) -> Result<()> {
        self.prefetch.copy(index)
    }

    /// Drain the copy stream and release the storage after every reader and
    /// graph retired.
    pub(crate) fn close(&self) -> Result<()> {
        drop(self.prefetch.close()?);
        Ok(())
    }
}

/// The library module of expert exchanges and distributed expert weights.
const MODULE: &str = "uniserve.distributed.experts";

// Distributed expert weights of a rank (DWDP): setup is collective over the
// expert union, after loading and before capacity planning. `weight_fields`
// places each layer's weights for its kernels and lists this rank's shards;
// [`ExpertWeights::build`] publishes them and maps every field's full
// extent; `bind_weights` installs the full views as the layers' parameters,
// and `bind_prefetch` returns the object execution contexts pass to their
// expert calls, over the manager's hooks. A prefetch a captured forward
// reaches is a graph break: the hook ends the segment and notes the copy,
// which the forward's graph performs at every launch.

/// The rank's distributed expert weights and their prefetch object.
pub(crate) struct DwdpManager {
    weights: Arc<ExpertWeights>,
    /// `bind_prefetch`'s object, passed to every runner's constructor.
    prefetch: Handle,
    /// The weight copies noted as graph breaks of the forward being
    /// captured.
    breaks: GraphBreaks,
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

/// The compute stream an expert call names by its handle.
fn compute(
    weights: &ExpertWeights,
    handle: &AnyView<'_>,
) -> std::result::Result<Stream, tvm_ffi::Error> {
    // SAFETY: the binding names the stream its expert call runs on, which the
    // rank's lanes own until the weights close.
    unsafe { crate::stream::external(weights.device(), count(handle)?) }
        .map_err(|error| ffi_error(&error))
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
        if union.exchange != "dwdp" {
            return Ok(None);
        }
        let Device::Cuda(device) = launch.device() else {
            return Err(Error::Unsupported("DWDP requires CUDA peer memory".into()));
        };
        let stream = match streams.lanes() {
            [] => streams.preparation(device),
            [lane] => Some(&lane.stream),
            _ => {
                return Err(Error::Unsupported(
                    "DWDP runs every forward on exactly one execution lane".into(),
                ));
            }
        }
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
        let breaks = GraphBreaks::default();
        let break_graph = Interpreter::import("uniserve.cuda_graph.break_graph")?;
        let prefetch = Handle::new(
            Interpreter::import(&format!("{MODULE}.bind_prefetch"))?.call_values(
                &std::iter::once(model.handle().to_any())
                    .chain(hooks(&weights, &break_graph, &breaks))
                    .collect::<Vec<_>>(),
            )?,
        )?;
        Ok(Some(Self {
            weights,
            prefetch,
            breaks,
            closed: false,
        }))
    }

    /// The object bindings pass to their expert calls.
    pub(crate) fn prefetch(&self) -> &Handle {
        &self.prefetch
    }

    /// The graph breaks of the forward being captured: whoever captures a
    /// forward that reads these weights takes them for its graph.
    pub(crate) fn breaks(&self) -> &GraphBreaks {
        &self.breaks
    }

    /// Drain the copy stream and release the weights after every binding
    /// and graph retired. An aborted close keeps every mapping until exit:
    /// peers may still read this rank's pages.
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        if std::mem::replace(&mut self.closed, true) {
            return Ok(());
        }
        if aborted {
            crate::leak(Arc::clone(&self.weights));
            return Ok(());
        }
        self.weights.close()
    }
}

impl Drop for DwdpManager {
    fn drop(&mut self) {
        let _ = self.close(true);
    }
}

/// The prefetch hooks as TVM-FFI functions closed over `weights`, in the
/// order `bind_prefetch` takes them after the model: `begin()`,
/// `end(stream)`, `before(layer, stream)` and `after(layer, stream)`. They
/// block on no peer.
///
/// Under graph capture a copy is a graph break: `before` ends the segment
/// with `break_graph` and notes the copy in `breaks`, and the runtime
/// performs it at every launch.
fn hooks(weights: &Arc<ExpertWeights>, break_graph: &Function, breaks: &GraphBreaks) -> Vec<Any> {
    type Hook = dyn Fn(&ExpertWeights, &[AnyView<'_>]) -> std::result::Result<Any, tvm_ffi::Error>;
    let function = |call: Box<Hook>| {
        let weights = Arc::clone(weights);
        tvm_ffi::Function::from_packed(move |args: &[AnyView<'_>]| call(&weights, args))
    };
    let unit = |result: Result<()>| {
        result
            .map(|()| Any::default())
            .map_err(|error| ffi_error(&error))
    };
    let begin = Any::from(function(Box::new(move |weights, _| unit(weights.begin()))));
    let end = Any::from(function(Box::new(move |weights, args| {
        let [stream] = args else {
            return Err(ffi_error(&Error::Invalid("end takes a stream".into())));
        };
        unit(weights.end(&compute(weights, stream)?))
    })));
    let before = Any::from({
        let owner = Arc::clone(weights);
        let break_graph = break_graph.clone();
        let breaks = breaks.clone();
        function(Box::new(move |weights, args| {
            let [layer, stream] = args else {
                return Err(ffi_error(&Error::Invalid(
                    "before takes a layer and a stream".into(),
                )));
            };
            let stream = compute(weights, stream)?;
            unit(weights.before(count(layer)?, &stream, |index| {
                break_graph.invoke(&[])?;
                let owner = Arc::clone(&owner);
                breaks.note(Arc::new(move || owner.copy(index)));
                Ok(())
            }))
        }))
    });
    let after = Any::from(function(Box::new(move |weights, args| {
        let [layer, stream] = args else {
            return Err(ffi_error(&Error::Invalid(
                "after takes a layer and a stream".into(),
            )));
        };
        unit(weights.after(count(layer)?, &compute(weights, stream)?))
    })));
    vec![begin, end, before, after]
}

/// The library module of host-resident layer weights.
const HOST_WEIGHTS: &str = "uniserve.loading.host_weights";

/// Which layer each slot of a host-weight stack holds, and which layers may
/// still read their slots.
///
/// Evaluation order makes two layers live at once: a layer's projections
/// start while the previous layer's feed-forward still produces rows, and
/// that layer completes before the next one attends. With `slots` slots,
/// `slots - 2` layers can therefore be copied ahead. `slots` divides the
/// layer count, so a repeated pass (the next denoising step) finds every
/// layer in its own slot; a copy never targets a slot whose layer is live,
/// and a layer whose copy was not enqueued in time (the first pass, or one
/// after an interrupted pass) is copied when first used. A pass acquires
/// its layers in order, so a first use that does not follow the newest live
/// layer starts a new pass.
#[derive(Debug)]
struct Ring {
    layers: usize,
    slots: usize,
    /// The layer whose copy each slot last received.
    loaded: Vec<Option<usize>>,
    /// The layers whose projections may still run, newest last.
    live: Vec<usize>,
}

impl Ring {
    /// The fewest slots, at least four, that divide `layers`.
    fn new(layers: usize) -> Result<Self> {
        let slots = (4..=layers)
            .find(|count| layers.is_multiple_of(*count))
            .ok_or_else(|| {
                Error::Unsupported(format!(
                    "no slot count of at least four divides {layers} streamed layers"
                ))
            })?;
        Ok(Self {
            layers,
            slots,
            loaded: vec![None; slots],
            live: Vec::new(),
        })
    }

    fn occupied(&self, slot: usize) -> bool {
        self.live.iter().any(|layer| layer % self.slots == slot)
    }

    /// Begin `layer`'s use. Returns the layers to copy, in order (`layer`
    /// itself first when its slot does not hold it), then the following
    /// layers whose slots no live layer reads; `None` when `layer` is
    /// already live and its slot was made readable.
    fn acquire(&mut self, layer: usize) -> Result<Option<Vec<usize>>> {
        if layer >= self.layers {
            return Err(Error::Invalid("layer is outside the streamed stack".into()));
        }
        if self.live.contains(&layer) {
            return Ok(None);
        }
        if self
            .live
            .last()
            .is_some_and(|newest| layer != (newest + 1) % self.layers)
        {
            // Another layer starts a new pass: the previous one, complete
            // or interrupted, has enqueued every read of its layers.
            self.live.clear();
        }
        let slot = layer % self.slots;
        if self.occupied(slot) {
            return Err(Error::Invariant(
                "a streamed layer shares its slot with a live layer".into(),
            ));
        }

        let mut copies = Vec::new();
        if self.loaded[slot] != Some(layer) {
            self.loaded[slot] = Some(layer);
            copies.push(layer);
        }
        let previous = self.live.last().copied();
        self.live = previous.into_iter().chain([layer]).collect();
        // Copy the following layers into slots no live layer reads; every
        // read of their previous layers is already enqueued.
        for step in 1..=self.slots - 2 {
            let following = (layer + step) % self.layers;
            let target = following % self.slots;
            if self.loaded[target] != Some(following) && !self.occupied(target) {
                self.loaded[target] = Some(following);
                copies.push(following);
            }
        }
        Ok(Some(copies))
    }
}

/// Where the host regions of host-weight stacks come from when the
/// worker's weights are retained at the Storage Server.
pub(crate) enum HostRetention<'a> {
    /// The worker's weights are not retained: each region is page-locked
    /// memory of the worker's own.
    None,
    /// The worker loaded: each region is new retained host memory that joins
    /// its published weights, filled by binding.
    Publish(&'a mut crate::storage::weights::Published),
    /// The worker bound retained weights: each region is a retained host
    /// storage among these, already holding the matrices.
    Bound(&'a [Arc<Storage>]),
}

/// What the copies of a host-weight stack address: its page-locked host
/// copy of every layer and the device slots.
struct HostStorage {
    _host: Arc<Storage>,
    _slots: Vec<Arc<Storage>>,
}

/// The dense projection weights of one repeated layer stack, kept in
/// page-locked host memory and streamed through a ring of device slots.
///
/// The library rebinds every layer's projection matrices to views of its
/// slot (`uniserve.loading.host_weights.bind`); its hooks call
/// [`before`](Self::before) ahead of each projection call and
/// [`after`](Self::after) after it, on the computing stream. The stack's
/// calls cannot be captured: copies are scheduled by the host as layers are
/// reached.
pub(crate) struct HostWeights {
    copies: SlotCopies<HostStorage>,
    ring: Mutex<Ring>,
}

impl HostWeights {
    /// Make `layer` readable on `compute` and copy the layers after it
    /// ahead.
    fn before(&self, layer: usize, compute: &Stream) -> Result<()> {
        if compute.capture_status()? != CaptureStatus::None {
            return Err(Error::Unsupported(
                "host-streamed layer weights cannot be captured".into(),
            ));
        }
        let Some(copies) = lock(&self.ring).acquire(layer)? else {
            return Ok(());
        };
        // A default-stream handle resolves against the current context.
        let _device = DeviceGuard::new(self.copies.device)?;
        for index in copies {
            self.copies.copy(index)?;
        }
        self.copies.wait(layer, compute)
    }

    /// Record that `compute` has read `layer`'s slot.
    fn after(&self, layer: usize, compute: &Stream) -> Result<()> {
        self.copies.release(layer, compute)
    }

    /// Drain the copy stream and release the host copy and the slots, after
    /// every reader retired.
    pub(crate) fn close(&self) -> Result<()> {
        drop(self.copies.close()?);
        Ok(())
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
    ) -> Result<Vec<Arc<Self>>> {
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
    ) -> Result<Arc<Self>> {
        let call = |name: &str| -> Result<Vec<i64>> {
            let function = format!("{HOST_WEIGHTS}.{name}");
            let value = Interpreter::import(&function)?.call_values(&[
                model.handle().to_any(),
                Any::from(tvm_ffi::String::from(path)),
            ])?;
            if name == "layers" {
                let layers: i64 = value
                    .try_into()
                    .map_err(|error| Error::numerical(&function, &error))?;
                return Ok(vec![layers]);
            }
            crate::ffi::integers(value).map_err(|error| Error::numerical(&function, &error))
        };
        let count = |value: i64| {
            usize::try_from(value)
                .map_err(|_| Error::Interface("a streamed stack's layout holds counts".into()))
        };

        // One page-locked region of exactly every layer's bytes: retained
        // at the Storage Server when the worker's weights are, and then
        // already holding the matrices when the worker bound them.
        let (host, layers, layer_bytes, filled) = match retention {
            HostRetention::Bound(storages) => {
                let layers = count(call("layers")?[0])?;
                let address = call("address")?[0] as u64;
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
                let layout = call("layout")?;
                let [layers, layer_bytes] = layout[..] else {
                    return Err(Error::Interface(
                        "a streamed stack's layout names its layers and layer bytes".into(),
                    ));
                };
                let (layers, layer_bytes) = (count(layers)?, count(layer_bytes)?);
                let (host, descriptor) = Storage::host_retainable(device, layers * layer_bytes)?;
                published.add_host(model.handle(), path, Arc::clone(&host), descriptor)?;
                (host, layers, layer_bytes, false)
            }
            HostRetention::None => {
                let layout = call("layout")?;
                let [layers, layer_bytes] = layout[..] else {
                    return Err(Error::Interface(
                        "a streamed stack's layout names its layers and layer bytes".into(),
                    ));
                };
                let (layers, layer_bytes) = (count(layers)?, count(layer_bytes)?);
                let host = Storage::pinned(device, layers * layer_bytes)?;
                (host, layers, layer_bytes, false)
            }
        };
        let ring = Ring::new(layers)?;

        // One device slot per ring position.
        let slots = (0..ring.slots)
            .map(|_| Storage::device(device, layer_bytes))
            .collect::<Result<Vec<_>>>()?;
        let view = |memory: &Arc<Storage>| {
            StorageView::whole(Arc::clone(memory) as Arc<dyn Backing>, DType::U8)
        };
        let host_view = view(&host)?;
        let slot_views = slots.iter().map(view).collect::<Result<Vec<_>>>()?;
        let copies = (0..layers)
            .map(|layer| {
                let mut batch = CopyBatch::new(SourceOrder::Stream);
                batch.push(
                    slot_views[layer % ring.slots].address(),
                    host_view.address() + (layer * layer_bytes) as u64,
                    layer_bytes,
                );
                batch
            })
            .collect();
        let slot_count = ring.slots;
        let owner = Arc::new(Self {
            copies: SlotCopies::new(
                device,
                slot_count,
                copies,
                HostStorage {
                    _host: host,
                    _slots: slots,
                },
            )?,
            ring: Mutex::new(ring),
        });

        let bind = Interpreter::import(&format!("{HOST_WEIGHTS}.bind"))?;
        // The views' wrappings and the call take one handoff.
        Interpreter::execute(|| -> Result<()> {
            let (before, after) = host_hooks(&owner);
            let mut arguments = vec![
                model.handle().to_any(),
                Any::from(tvm_ffi::String::from(path)),
                before,
                after,
                Any::from(filled),
                host_view.handle()?.to_any(),
            ];
            for slot in &slot_views {
                arguments.push(slot.handle()?.to_any());
            }
            bind.call_values(&arguments).map(drop)
        })?;
        tracing::info!(
            "streaming {layers} layers of {path:?} from host memory through {slot_count} device slots of {layer_bytes} bytes"
        );
        Ok(owner)
    }
}

/// The hooks of a host-weight stack as TVM-FFI functions closed over its
/// owner: `before(layer, stream)` and `after(layer, stream)`, on the calling
/// CUDA stream handle.
fn host_hooks(owner: &Arc<HostWeights>) -> (Any, Any) {
    type Hook = fn(&HostWeights, usize, &Stream) -> Result<()>;
    let function = |call: Hook| {
        let owner = Arc::clone(owner);
        Any::from(tvm_ffi::Function::from_packed(
            move |args: &[AnyView<'_>]| {
                let [layer, stream] = args else {
                    return Err(ffi_error(&Error::Invalid(
                        "a layer hook takes a layer and a stream".into(),
                    )));
                };
                // SAFETY: the hook names the stream its layer computes on,
                // which the rank's lanes own until the weights close.
                let stream =
                    unsafe { crate::stream::external(owner.copies.device, count(stream)?) }
                        .map_err(|error| ffi_error(&error))?;
                call(&owner, count(layer)?, &stream)
                    .map(|()| Any::default())
                    .map_err(|error| ffi_error(&error))
            },
        ))
    };
    (function(HostWeights::before), function(HostWeights::after))
}

#[cfg(test)]
mod tests {
    use super::Ring;

    /// Drive one pass over `layers` in order, as a forward reaches them,
    /// and return the copies each first use enqueued.
    fn pass(ring: &mut Ring, layers: impl IntoIterator<Item = usize>) -> Vec<Vec<usize>> {
        layers
            .into_iter()
            .map(|layer| ring.acquire(layer).unwrap().unwrap_or_default())
            .collect()
    }

    #[test]
    fn the_ring_is_the_fewest_slots_of_at_least_four_dividing_the_stack() {
        assert_eq!(Ring::new(10).unwrap().slots, 5);
        assert_eq!(Ring::new(50).unwrap().slots, 5);
        assert_eq!(Ring::new(48).unwrap().slots, 4);
        assert_eq!(Ring::new(7).unwrap().slots, 7);
        assert!(Ring::new(3).is_err());
    }

    #[test]
    fn a_first_pass_copies_each_layer_once_ahead_of_its_use() {
        let mut ring = Ring::new(10).unwrap();
        let copies = pass(&mut ring, 0..10);
        // The first layer and the three after it, then one more layer per
        // step: two slots hold live layers.
        assert_eq!(copies[0], vec![0, 1, 2, 3]);
        for (layer, copied) in copies.iter().enumerate().skip(1) {
            assert_eq!(copied, &vec![(layer + 3) % 10], "layer {layer}");
        }
        let mut copied: Vec<usize> = copies.concat();
        copied.truncate(10);
        copied.sort_unstable();
        assert_eq!(copied, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn a_layer_already_live_needs_no_copy_or_wait() {
        let mut ring = Ring::new(8).unwrap();
        assert!(ring.acquire(0).unwrap().is_some());
        assert_eq!(ring.acquire(0).unwrap(), None);
        assert!(ring.acquire(1).unwrap().is_some());
        // The previous layer stays live while the next one starts.
        assert_eq!(ring.acquire(0).unwrap(), None);
    }

    #[test]
    fn no_copy_targets_the_slot_of_a_live_layer() {
        let mut ring = Ring::new(10).unwrap();
        for _ in 0..3 {
            for layer in 0..10 {
                let copies = ring.acquire(layer).unwrap().unwrap_or_default();
                // The layer before stays live while this one starts.
                assert!(ring.live.len() <= 2 && ring.live.last() == Some(&layer));
                for copied in copies.iter().filter(|&&copied| copied != layer) {
                    assert!(
                        ring.live.iter().all(|live| live % 5 != copied % 5),
                        "layer {layer} copied {copied} over a live slot"
                    );
                }
            }
        }
    }

    #[test]
    fn an_interrupted_pass_restarts_and_recopies_what_it_lacks() {
        let mut ring = Ring::new(10).unwrap();
        pass(&mut ring, 0..6);
        // A new pass starts at layer 0, whose slot now holds layer 5.
        let copies = ring.acquire(0).unwrap().unwrap();
        assert_eq!(copies[0], 0);
        assert_eq!(ring.live, vec![0]);
        // The remaining layers of the new pass are each in their slot when
        // first used.
        for layer in 1..10 {
            ring.acquire(layer).unwrap();
            assert_eq!(ring.loaded[layer % 5], Some(layer));
        }
    }
}
