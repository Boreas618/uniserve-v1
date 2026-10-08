//! Local double-buffered expert-weight reads from immutable peer storage.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, PoisonError};

use indexmap::IndexMap;
use uniserve_cuda::{CopyBatch, DeviceGuard, DeviceId, Event, EventFlags, SourceOrder, Stream};

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

    /// Borrows the caller's compute stream for one operation.
    fn compute(&self, handle: usize) -> Result<Stream> {
        // SAFETY: expert calls submit on a live compute stream of this device,
        // which the caller keeps alive through the call.
        Ok(unsafe { Stream::borrowed(self.device, handle) }?)
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
    pub fn end(&self, compute: usize) -> Result<()> {
        let last = self
            .state()
            .calls
            .pop()
            .ok_or(Error::State("DWDP expert call requires an active context"))?
            .last;
        if let Some(last) = last {
            self.compute(compute)?.wait_event(&self.ready[last % 2])?;
        }
        Ok(())
    }

    /// `submit` either copies now or records a copy between graph segments.
    /// It runs outside the state lock and may call back into this owner.
    /// Callback errors retain their backend type in the inner result.
    pub fn before<E>(
        &self,
        module: usize,
        compute: usize,
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
        let compute = self.compute(compute)?;
        if !started {
            for event in &self.consumed {
                event.record(&compute)?;
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

    pub fn after(&self, module: usize, compute: usize) -> Result<()> {
        let index = self.index(module)?;
        self.consumed[index % 2].record(&self.compute(compute)?)?;
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

    /// Open one invocation.
    pub fn begin(&self) -> Result<()> {
        self.prefetch.begin()
    }

    /// Close one invocation on `compute`, joining a speculative tail copy.
    pub fn end(&self, compute: usize) -> Result<()> {
        self.prefetch.end(compute)
    }

    /// Make layer `layer` readable on `compute` and prefetch the next one.
    ///
    /// Eagerly, the copies are submitted now and the result is empty. On a
    /// capturing stream, copies cannot be recorded: the result lists the
    /// copies the caller submits between graph segments, in order.
    pub fn before(&self, layer: usize, compute: usize) -> Result<Vec<usize>> {
        // SAFETY: the caller's live compute stream on this device.
        let stream = unsafe { Stream::borrowed(self.prefetch.device, compute) }?;
        let capturing = stream.capture_status()? != uniserve_cuda::CaptureStatus::None;
        let mut recorded = Vec::new();
        self.prefetch.before(layer, compute, |index| {
            if capturing {
                recorded.push(index);
                Ok(())
            } else {
                self.prefetch.copy(index)
            }
        })??;
        Ok(recorded)
    }

    /// Release layer `layer`'s slots after its last reader on `compute`.
    pub fn after(&self, layer: usize, compute: usize) -> Result<()> {
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
