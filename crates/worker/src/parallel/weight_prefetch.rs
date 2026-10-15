//! Local double-buffered expert-weight reads from immutable peer storage.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, PoisonError};

use indexmap::IndexMap;
use uniserve_cuda::{CopyBatch, DeviceGuard, DeviceId, Event, EventFlags, SourceOrder, Stream};

use std::sync::Arc;

use tvm_ffi::{Any, AnyView};

use super::world::World;
use crate::config::{Device, WorkerLaunch};
use crate::cuda_graph::GraphBreaks;
use crate::ffi::{Function, Handle, Interpreter};
use crate::stream::Streams;
use crate::worker::LoadedModel;
use crate::{Error, Result};

#[derive(Default)]
struct Call {
    started: bool,
    queued: Option<usize>,
    last: Option<usize>,
}

struct State<T> {
    backing: Option<T>,
    calls: Vec<Call>,
    closed: bool,
}

/// Prefetch the next expert layer into alternating remote-weight slots.
///
/// The numerical backend supplies immutable source mappings and destination
/// views. Calls serialize on their execution stream. External events protect
/// both copy-before-read and read-before-overwrite across CUDA graph segments.
pub struct WeightPrefetch<T> {
    device: DeviceId,
    layers: HashMap<usize, usize>,
    copies: Vec<CopyBatch>,
    stream: Stream,
    ready: [Event; 2],
    consumed: [Event; 2],
    state: Mutex<State<T>>,
}

impl<T> WeightPrefetch<T> {
    /// Copy tuples are `(peer, destination, source, bytes)`, grouped by layer.
    /// `backing` retains every allocation used by those addresses.
    pub fn new(
        device: i32,
        layers: Vec<usize>,
        copies: Vec<Vec<(usize, usize, usize, usize)>>,
        backing: T,
    ) -> Result<Self> {
        if layers.len() != copies.len() || layers.is_empty() {
            return Err(Error::Invalid(
                "each expert layer needs one copy plan".into(),
            ));
        }

        let external = || {
            Event::new(EventFlags {
                external: true,
                ..EventFlags::default()
            })
        };

        Ok(Self {
            device: DeviceId(device),
            layers: layers
                .into_iter()
                .enumerate()
                .map(|(i, layer)| (layer, i))
                .collect(),
            copies: copies.into_iter().map(plan).collect(),
            stream: Stream::new(DeviceId(device))?,
            ready: std::array::from_fn(|_| external()),
            consumed: std::array::from_fn(|_| external()),
            state: Mutex::new(State {
                backing: Some(backing),
                calls: Vec::new(),
                closed: false,
            }),
        })
    }

    fn state(&self) -> MutexGuard<'_, State<T>> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The device the weights and their copy stream live on.
    pub fn device(&self) -> DeviceId {
        self.device
    }

    pub fn contains(&self, module: usize) -> bool {
        self.layers.contains_key(&module)
    }

    fn index(&self, module: usize) -> Result<usize> {
        self.layers
            .get(&module)
            .copied()
            .ok_or_else(|| Error::Invalid("module has no prefetched expert weights".into()))
    }

    pub fn begin(&self) -> Result<()> {
        let mut state = self.state();
        if state.closed {
            return Err(Error::State("DWDP weight owner is closed"));
        }
        state.calls.push(Call::default());
        Ok(())
    }

    /// Join a speculative tail copy even if the call omitted its last layer.
    pub fn end(&self, compute: &Stream) -> Result<()> {
        let last = self
            .state()
            .calls
            .pop()
            .ok_or(Error::State("DWDP expert call requires an active context"))?
            .last;
        if let Some(last) = last {
            compute.wait_event(&self.ready[last % 2])?;
        }
        Ok(())
    }

    /// `submit` either copies now or records a copy between graph segments.
    /// It runs outside the state lock and may call back into this owner.
    /// Callback errors retain their backend type in the inner result.
    pub fn before<E>(
        &self,
        module: usize,
        compute: &Stream,
        mut submit: impl FnMut(usize) -> std::result::Result<(), E>,
    ) -> Result<std::result::Result<(), E>> {
        let index = self.index(module)?;
        let (started, queued) = {
            let mut state = self.state();
            let call = state
                .calls
                .last_mut()
                .ok_or(Error::State("DWDP expert call requires an active context"))?;
            let started = std::mem::replace(&mut call.started, true);
            (started, call.queued)
        };

        // A default-stream handle resolves against the current context.
        let _device = DeviceGuard::new(self.device)?;
        if !started {
            for event in &self.consumed {
                event.record(compute)?;
            }
        }

        if queued != Some(index)
            && let Err(error) = self.prefetch(index, &mut submit)?
        {
            return Ok(Err(error));
        }

        compute.wait_event(&self.ready[index % 2])?;

        let next = (index + 1 < self.copies.len()).then_some(index + 1);
        self.state()
            .calls
            .last_mut()
            .ok_or(Error::State("DWDP expert call requires an active context"))?
            .queued = next;
        if let Some(next) = next {
            return self.prefetch(next, &mut submit);
        }
        Ok(Ok(()))
    }

    fn prefetch<E>(
        &self,
        index: usize,
        submit: &mut impl FnMut(usize) -> std::result::Result<(), E>,
    ) -> Result<std::result::Result<(), E>> {
        let mut state = self.state();
        let call = state
            .calls
            .last_mut()
            .ok_or(Error::State("DWDP expert call requires an active context"))?;
        call.last = Some(index);
        drop(state);
        Ok(submit(index))
    }

    pub fn after(&self, module: usize, compute: &Stream) -> Result<()> {
        let index = self.index(module)?;
        self.consumed[index % 2].record(compute)?;
        Ok(())
    }

    /// Replay a recorded peer read, preserving the same two-slot dependencies
    /// as eager execution. No host synchronization is performed.
    pub fn copy(&self, index: usize) -> Result<()> {
        if self.state().closed {
            return Err(Error::State("DWDP weight owner is closed"));
        }

        let copy = self
            .copies
            .get(index)
            .ok_or_else(|| Error::Invalid("expert layer is outside the copy plan".into()))?;
        self.stream.wait_event(&self.consumed[index % 2])?;
        // SAFETY: `backing` retains every source mapping and destination view
        // until close drains this stream.
        unsafe { copy.submit(&self.stream) }?;
        self.ready[index % 2].record(&self.stream)?;
        Ok(())
    }

    /// The caller retires numerical readers first. Drain the copy stream and
    /// return backing for destruction outside owner locks and binding borrows.
    pub fn close(&self) -> Result<Option<T>> {
        let mut state = self.state();
        if state.backing.is_none() {
            return Ok(None);
        }
        if !state.calls.is_empty() {
            return Err(Error::State("cannot close active weight prefetch"));
        }
        state.closed = true;
        drop(state);

        self.stream.synchronize()?;
        Ok(self.state().backing.take())
    }

    pub fn visit<R>(&self, visit: impl FnOnce(&T) -> R) -> Option<R> {
        self.state().backing.as_ref().map(visit)
    }
}

impl<T> Drop for WeightPrefetch<T> {
    #[expect(
        clippy::mem_forget,
        reason = "unknown copy completion must retain peer mappings"
    )]
    fn drop(&mut self) {
        if self.close().is_err() {
            let state = self.state.get_mut().unwrap_or_else(PoisonError::into_inner);
            std::mem::forget(state.backing.take());
        }
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
pub struct ShardField {
    /// The field's name, unique within its layer (`up_gate.values`).
    pub name: String,
    /// Device address and bytes of this rank's shard.
    pub address: u64,
    pub bytes: usize,
}

/// Storage that must outlive every copy and every reader of the composite
/// weight views: published shards, reusable remote slots and the composite
/// mappings.
pub struct WeightStorage {
    published: Vec<crate::transfer::PeerStorage>,
    slots: Vec<uniserve_cuda::DeviceAllocation>,
    composites: Vec<std::sync::Arc<crate::storage::memory::Memory>>,
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
pub struct ExpertWeights {
    prefetch: WeightPrefetch<WeightStorage>,
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
    pub fn build(
        group: &super::host::HostGroup,
        device: i32,
        node: &str,
        layers: &[Vec<ShardField>],
        stream: &Stream,
    ) -> Result<(Self, Vec<Vec<crate::storage::memory::StorageView>>)> {
        let mut storage = WeightStorage {
            published: Vec::new(),
            slots: Vec::new(),
            composites: Vec::new(),
        };
        let built = Self::map(group, device, node, layers, stream, &mut storage);
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
                #[expect(
                    clippy::mem_forget,
                    reason = "peers may read published pages of a failed setup"
                )]
                std::mem::forget(storage);
                Err(error)
            }
        }
    }

    #[expect(clippy::type_complexity, reason = "one setup's complete result")]
    fn map(
        group: &super::host::HostGroup,
        device: i32,
        node: &str,
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
        use crate::storage::memory::{Memory, StorageView};

        let (rank, size) = (group.rank(), group.size());
        let page = DeviceAllocation::granularity(DeviceId(device))?;
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
                    crate::transfer::PeerStorage::allocate(device, capacity, node, group)?;
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
                            storage.slots.push(DeviceAllocation::exportable(
                                DeviceId(device),
                                length,
                                &[],
                            )?);
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
                let composite = Memory::mapped(PeerMapping::alias(&pieces)?);
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
    pub fn device(&self) -> DeviceId {
        self.prefetch.device()
    }

    /// Open one invocation.
    pub fn begin(&self) -> Result<()> {
        self.prefetch.begin()
    }

    /// Close one invocation on `compute`, joining a speculative tail copy.
    pub fn end(&self, compute: &Stream) -> Result<()> {
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
    pub fn before(
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
        })?
    }

    /// Release layer `layer`'s slots after its last reader on `compute`.
    pub fn after(&self, layer: usize, compute: &Stream) -> Result<()> {
        self.prefetch.after(layer, compute)
    }

    /// Submit one recorded copy (graph replay between segments).
    pub fn copy(&self, index: usize) -> Result<()> {
        self.prefetch.copy(index)
    }

    /// Drain the copy stream and release the storage after every reader and
    /// graph retired.
    pub fn close(&self) -> Result<()> {
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
        node: &str,
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
            [] => streams.preparation(DeviceId(device)),
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
        let (weights, views) = ExpertWeights::build(union_group, device, node, &layers, &stream)?;
        drop(shards);
        let bind = Interpreter::import(&format!("{MODULE}.bind_weights"))?;
        for (layer, (fields, views)) in layers.iter().zip(views).enumerate() {
            let mut arguments = vec![
                model.handle().to_any(),
                Any::from(layer as i64),
                Any::from(crate::ffi::strings(fields.iter().map(|field| &field.name))),
            ];
            for view in views {
                arguments.push(Any::from(view.to_tensor()?));
            }
            bind.call_values(&arguments)?;
        }
        tracing::info!(
            target: "uniserve_worker.launch",
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
    #[allow(clippy::mem_forget)]
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        if std::mem::replace(&mut self.closed, true) {
            return Ok(());
        }
        if aborted {
            std::mem::forget(Arc::clone(&self.weights));
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
