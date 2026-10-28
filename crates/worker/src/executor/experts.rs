//! This rank's member of an expert union: what the worker keeps beside the
//! executor's union (`uniserve.executor.experts.ExpertUnion`).
//!
//! Ranks of an expert union execute every expert step together. The
//! executor agrees on each step, runs its microbatches and joins other
//! ranks' steps; the worker keeps what the executor's calls borrow and the
//! pacing of joins outside the rank's own steps:
//!
//! - the union's host group, without a deadline, which the executor reaches
//!   through host functions (`crate::parallel::host::bootstrap`) for its
//!   agreement and for its transports' bootstrap;
//! - one stream per microbatch: the device's primary stream, then forks of it,
//!   which the rank's streams close after the executor;
//! - split MegaMoE's peer-mapped storage, one per microbatch, which every
//!   member's exchange addresses until the executor closed the exchanges.
//!
//! The member is built at the same point of every rank's startup, after
//! loading and before capacity planning, so the exchanges' device
//! workspaces are resident when the rank measures its free storage:
//! the executor configures its agreement (`expert_storage_bytes`, which
//! also states the peer storage its exchanges need), the worker maps the
//! storage, and the executor creates its exchanges over it
//! (`connect_experts`). Both are collective over the union.

use std::sync::{Arc, Mutex};

use cuda::Stream;

use crate::config::{Device, WorkerLaunch};
use crate::executor::python::PythonExecutor;
use crate::ffi::Backing;
use crate::parallel::host::{HostGroup, bootstrap};
use crate::parallel::world::World;
use crate::storage::allocation::StorageView;
use crate::stream::Streams;
use crate::transfer::PeerStorage;
use crate::{Error, Result, lock};

/// This rank's member of an expert union.
pub(crate) struct ExpertMember {
    /// The union's host group, which the executor's host functions hold
    /// too.
    _host: Arc<Mutex<HostGroup>>,
    /// Each microbatch's stream: the device's primary stream first.
    streams: Vec<Stream>,
    /// Each microbatch's peer-mapped storage, when the exchanges address
    /// one; released once the executor closed them.
    storages: Vec<PeerStorage>,
}

/// What building the member borrows from the rank.
pub(crate) struct ExpertStart<'a> {
    pub(crate) launch: &'a WorkerLaunch,
    pub(crate) world: &'a World,
    pub(crate) executor: &'a PythonExecutor,
    pub(crate) streams: &'a mut Streams,
    /// The host's name, which identifies descriptor-shared peer storage.
    pub(crate) host: &'a str,
}

impl ExpertMember {
    /// Build this rank's member, collectively over the union. `None`
    /// without a union that exchanges tokens.
    pub(crate) fn start(start: ExpertStart<'_>) -> Result<Option<Self>> {
        let Some(union) = &start.launch.expert_parallel else {
            return Ok(None);
        };
        // Distributed expert weights have no union member.
        if union.prefetches_weights() {
            return Ok(None);
        }
        let config = &start.launch.execution;
        let Device::Cuda(device) = start.launch.device() else {
            return Err(Error::Unsupported(
                "expert parallelism requires a CUDA device".into(),
            ));
        };

        // The union's members, sharing the startup connections. Expert
        // steps wait on peers without a deadline: a peer may still compile
        // kernels before its first announcement, and an idle union agrees
        // indefinitely. A failed peer closes its connections, which fails
        // the waiting collective.
        let union_group = start
            .world
            .union
            .as_ref()
            .ok_or(Error::State("an expert union requires its host group"))?;
        let members: Vec<usize> = (0..union_group.size()).collect();
        let group = union_group.subgroup(&members)?;
        group.set_timeout(None)?;
        let host = Arc::new(Mutex::new(group));

        // Microbatch 0 runs on the primary stream, the others on forks of it.
        let primary = start
            .streams
            .stream_for(device)
            .ok_or(Error::State("expert parallelism requires a CUDA stream"))?
            .stream()?
            .clone();
        let mut streams = vec![primary];
        for _ in 1..config.expert_microbatches.max(1) {
            streams.push(start.streams.fork_microbatch()?.stream()?.clone());
        }

        let bytes = start.executor.expert_storage_bytes(bootstrap(&host))?;
        let mut storages = Vec::new();
        if bytes > 0 {
            for _ in &streams {
                storages.push(PeerStorage::allocate(
                    device,
                    bytes,
                    start.host,
                    &lock(&host),
                )?);
            }
        }
        // Each microbatch's stream and, with peer storage, every member's
        // address of its share.
        let views = storages
            .iter()
            .map(|storage| {
                StorageView::whole(
                    Arc::clone(storage.local()) as Arc<dyn Backing>,
                    crate::ffi::DType::I8,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let size = lock(&host).size();
        let microbatches: Vec<serde_json::Value> = streams
            .iter()
            .enumerate()
            .map(|(index, stream)| {
                let pointers: Vec<i64> = storages
                    .get(index)
                    .map(|storage| {
                        let base = storage.peers().address();
                        (0..size)
                            .map(|member| (base + (member * storage.bytes_each()) as u64) as i64)
                            .collect()
                    })
                    .unwrap_or_default();
                serde_json::json!({
                    "stream": stream.raw() as i64,
                    "device": device.0,
                    "pointers": pointers,
                })
            })
            .collect();
        let storage = serde_json::json!({ "microbatches": microbatches });
        let mut storage_views: Vec<Option<StorageView>> = views.into_iter().map(Some).collect();
        storage_views.resize(streams.len(), None);
        start
            .executor
            .connect_experts(storage.to_string(), storage_views)?;
        tracing::info!(
            "expert union member with {} microbatches, {} bytes of peer storage each",
            streams.len(),
            bytes
        );
        Ok(Some(Self {
            _host: host,
            streams,
            storages,
        }))
    }

    /// Each microbatch's stream, the device's primary stream first.
    pub(crate) fn streams(&self) -> &[Stream] {
        &self.streams
    }

    /// Release the peer storage once the executor closed its exchanges,
    /// whose closing barrier every member passed. An aborted close keeps
    /// it until the process exits: peers may still address it.
    pub(crate) fn close(&mut self, aborted: bool) {
        let storages = std::mem::take(&mut self.storages);
        if aborted {
            crate::leak(storages);
        }
    }
}

impl Drop for ExpertMember {
    fn drop(&mut self) {
        // A member dropped without a close belongs to a failed rank: peers
        // may still map its storage.
        self.close(true);
    }
}
