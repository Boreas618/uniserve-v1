//! The rank's storage owners and their part in request lifecycles.
//!
//! Startup allocates them on the capacity plan: the device event pool, the
//! product store (relay lanes and the buffer pool), the product transports
//! and, for a text model, the prefix cache (unit planes, block tables and import
//! workspaces). Capability runtimes borrow them to execute calls; lifecycle
//! commands release their request storage through [`RequestStorage`].

use std::collections::HashSet;
use std::sync::Arc;

use uniserve_cuda::Stream;
use uniserve_ipc::CallId;
use uniserve_ipc::{BufferId, CallKind, ForwardMode, RequestKey, Wake};

use super::cache_width;
use crate::Result;
use crate::capacity::plan::Plan;
use crate::config::{WorkerConfig, WorkerLaunch};
use crate::executor::lifecycle::{AdmittedSlot, LaneJobs, RequestStorage};
use crate::storage::{BufferPool, PrefixCache, ProductStore};
use crate::stream::{CUDAStream, EventPool, Streams};
use crate::transfer::{TransferCapacity, TransportKind, TransportOptions, Transports};

/// The call kind whose lane owns the stream slot releases are ordered on:
/// token calls, which read the rows a release clears.
const RELEASE: CallKind = CallKind::Forward(ForwardMode::Decode);

/// The storage owners of a rank.
pub(crate) struct WorkerStorage {
    pub events: Arc<EventPool>,
    pub tensors: ProductStore,
    /// Shared with the cache import copy tasks that read through them.
    pub transports: Arc<Transports>,
    pub prefix_cache: Option<PrefixCache>,
    /// Stream that orders slot releases after the batches reading them: the
    /// stream serving token calls, shared with its lane.
    release_stream: Option<Stream>,
    /// Finished requests whose guidance prefix slots (image trajectories)
    /// clear with their own slots, once their readers retired.
    finished_prefixes: HashSet<RequestKey>,
    /// Those of them whose retirement is ready.
    retired_prefixes: HashSet<RequestKey>,
}

impl WorkerStorage {
    pub(crate) fn new(
        events: Arc<EventPool>,
        tensors: ProductStore,
        transports: Arc<Transports>,
        prefix_cache: Option<PrefixCache>,
        release_stream: Option<Stream>,
    ) -> Self {
        Self {
            events,
            tensors,
            transports,
            prefix_cache,
            release_stream,
            finished_prefixes: HashSet::new(),
            retired_prefixes: HashSet::new(),
        }
    }

    /// Release the products (and cache exports) of finished calls.
    pub(crate) fn release_calls(&mut self, calls: &[(RequestKey, CallId)]) -> Result<()> {
        if calls.is_empty() {
            return Ok(());
        }
        self.tensors.release_calls(calls);
        if let Some(prefix_cache) = &mut self.prefix_cache {
            prefix_cache.release_calls(calls);
        }
        Ok(())
    }
}

impl RequestStorage for WorkerStorage {
    fn admit_slots(&mut self, _admitted: &[AdmittedSlot<'_>], _jobs: &LaneJobs) -> Result<()> {
        // Block tables are installed per batch; a new slot has no state.
        Ok(())
    }

    fn release_buffers(&mut self, buffers: &HashSet<BufferId>, jobs: &LaneJobs) -> Result<()> {
        self.tensors.release_buffers(buffers, jobs.any().on(None))?;
        if let Some(prefix_cache) = &mut self.prefix_cache {
            prefix_cache.release_buffers(buffers)?;
        }
        Ok(())
    }

    fn release_requests(
        &mut self,
        requests: &HashSet<RequestKey>,
        retained: &HashSet<BufferId>,
        jobs: &LaneJobs,
    ) -> Result<()> {
        self.tensors
            .release_requests(requests, retained, jobs.any().on(None))?;
        if let Some(prefix_cache) = &mut self.prefix_cache {
            prefix_cache.release_requests(requests, retained)?;
            self.finished_prefixes.extend(requests.iter().copied());
        }
        Ok(())
    }

    fn retirement_ready(
        &mut self,
        buffers: &HashSet<BufferId>,
        requests: &HashSet<RequestKey>,
        retained: &HashSet<BufferId>,
    ) -> Result<bool> {
        let ready = self.tensors.retirement_ready(buffers, requests, retained)?
            && self.prefix_cache.as_mut().is_none_or(|prefix_cache| {
                prefix_cache.retirement_ready(buffers, requests, retained)
            });
        if ready {
            // Their prefix slots clear with the requests' own slots.
            for request in requests {
                if self.finished_prefixes.remove(request) {
                    self.retired_prefixes.insert(*request);
                }
            }
        }
        Ok(ready)
    }

    fn retire_slots(&mut self, slots: &[usize], jobs: &LaneJobs) -> Result<()> {
        if let Some(prefix_cache) = &mut self.prefix_cache {
            // The rows are cleared by the lane that owns the release
            // stream, behind the batches it already performed.
            let release = match &self.release_stream {
                // A rank that does not list the kind has one lane for
                // every kind, which owns the stream.
                Some(stream) => jobs
                    .job(RELEASE)
                    .unwrap_or_else(|_| jobs.any())
                    .on(Some(stream)),
                None => jobs.any().on(None),
            };
            let slots: Vec<u32> = slots.iter().map(|&slot| slot as u32).collect();
            prefix_cache.release_slots(&slots, release)?;
            for request in std::mem::take(&mut self.retired_prefixes) {
                prefix_cache.release_request_prefixes(request, release)?;
                // A later epoch of the request id starts without the
                // retired epoch's resident transfers and accepted bases.
                prefix_cache.drop_request(request);
            }
        }
        Ok(())
    }

    fn reap(&mut self) -> Result<()> {
        // Work deferred on other threads gets its job from this one.
        self.events.submit_deferred()?;
        self.transports.reap()?;
        if let Some(prefix_cache) = &self.prefix_cache {
            prefix_cache.reap_imports()?;
        }
        Ok(())
    }

    fn awaiting_acknowledgment(&self) -> bool {
        self.transports.awaiting_acknowledgment()
    }

    fn set_wake(&self, wake: Option<Wake>) {
        self.events.set_wake(wake.clone());
        self.transports.set_wake(wake.clone());
        if let Some(prefix_cache) = &self.prefix_cache {
            prefix_cache.set_wake(wake);
        }
    }

    fn close(&mut self, aborted: bool) -> Result<()> {
        if aborted {
            // Device work may still address the pool and the staging.
            crate::leak(self.prefix_cache.take());
            return Ok(());
        }
        // Revoke every published product, stop the imports reading through
        // the transports, then close the transports, which retires the
        // exports they still register (the prefix cache requires its exports
        // retired before it closes).
        let mut result = self.tensors.revoke_exports();
        if let Some(prefix_cache) = &mut self.prefix_cache {
            result = result.and(prefix_cache.stop_imports());
        }
        result = result.and(self.transports.close());
        if let Some(prefix_cache) = &mut self.prefix_cache {
            result = result.and(prefix_cache.close());
        }
        self.tensors.close();
        result.and(self.events.close())
    }
}

/// Allocate the storage owners on the plan: device events, the product
/// transports and the prefix cache of a text model, whose planes are reset on
/// the device's preparation stream.
pub(super) fn allocate_storage(
    launch: &WorkerLaunch,
    config: &WorkerConfig,
    plan: &Plan,
    streams: &Streams,
    memory: &mut dyn crate::storage::ExclusiveMemory,
) -> Result<WorkerStorage> {
    let events = EventPool::new();
    let kinds = launch
        .data_plane
        .backends
        .iter()
        .map(|name| TransportKind::parse(name))
        .collect::<Result<Vec<_>>>()?;
    let capacity = Arc::new(TransferCapacity::new(
        plan.arena.transfer_bytes,
        plan.arena.transfer_tickets,
    )?);
    let options = TransportOptions {
        source: plan.info.endpoint.clone(),
        acknowledgment_slot: launch.ipc.acknowledgment_slot as usize,
        host_slots: launch
            .ipc
            .host_slots
            .iter()
            .map(|&slot| slot as usize)
            .collect(),
        cross_host_consumers: launch.ipc.products_cross_hosts,
        vmm_pool_bytes: plan.arena.export_pool_bytes,
    };
    let exports = launch
        .data_plane
        .exports
        .iter()
        .map(|name| TransportKind::parse(name))
        .collect::<Result<Vec<_>>>()?;
    let transports = Arc::new(
        Transports::new(&kinds, capacity, Arc::clone(&events), &options)?.with_exports(&exports)?,
    );

    // Products: scheduler-placed buffers in the buffer pool and scalar relay
    // lanes, one per request slot and unresolved call, on the primary
    // device where every binding computes.
    let devices = vec![config.device.memory()];
    let buffer_pool =
        BufferPool::allocate(plan.info.buffer_pool_bytes.max(1), &devices, &mut *memory)?;
    let tensors = ProductStore::new(
        plan.arena.device_product_bytes.max(1) as usize,
        plan.info.encoder_entry_bytes.max(1) as usize,
        devices,
        plan.info.request_slots as usize,
        plan.info.max_unresolved_calls as usize + 1,
        buffer_pool,
        Arc::clone(&events),
    )?;

    let device = config.device.cuda();
    let preparation = device.and_then(|cuda| streams.preparation(cuda));
    let prefix_cache = match (&plan.cache, &plan.info.prefix_cache) {
        (Some(cache), Some(info)) => Some(PrefixCache::new(
            Arc::clone(cache),
            info.num_units,
            plan.info.request_slots,
            cache_width(&cache.group_shapes()?, config.max_model_len),
            plan.info.max_unresolved_calls as usize,
            device,
            preparation.map(CUDAStream::stream).transpose()?,
            memory,
        )?),
        _ => None,
    };
    if let Some(stream) = preparation {
        stream.synchronize()?;
    }
    // Slot releases order after the token batches that read them.
    let release_stream = match device {
        Some(cuda) => streams
            .stream_for(CallKind::Forward(ForwardMode::Decode), cuda)
            .map(|stream| stream.stream().cloned())
            .transpose()?,
        None => None,
    };
    Ok(WorkerStorage::new(
        events,
        tensors,
        transports,
        prefix_cache,
        release_stream,
    ))
}
