//! Launching worker groups: the ranks of a group started by the daemons of
//! their nodes, each from the launch its `WorkerProcessArgs` derive, and the
//! group formed from those ranks once they registered and reported their
//! startup capabilities. `spawn_all` starts every group's ranks before any
//! group waits for its own.

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use anyhow::Context;

use super::WorkerGroup;
use super::relaunch::Relaunch;
use super::routing::refuse_shared_storage_off_engine_host;
use crate::worker::WorkerProcessArgs;
use crate::worker::claims::{Claims, StorageClaims};
use crate::worker::process::WorkerSockets;
use crate::worker::processes::{ProcessClient, Processes, Reservation};
use crate::worker::registry::Registry;
use crate::worker::{IncarnationRegistry, PendingWorker, WorkerProcess};
use ipc::StorageLaunch;

/// One worker group's spawned ranks and the launch they register for.
pub(crate) struct LaunchedWorkers {
    /// The registry the ranks register with.
    registry: Registry,
    ranks: Vec<PendingWorker>,
    /// The daemons that started the ranks, which stop them when the launch
    /// fails.
    processes: Processes,
    worker_id: String,
}

impl Drop for LaunchedWorkers {
    /// A launch that did not complete stops its ranks on every node: a rank
    /// that never registered has no channel whose close would end it.
    fn drop(&mut self) {
        if !self.ranks.is_empty()
            && let Ok(mut processes) = crate::worker::processes::lock(&self.processes)
        {
            let _ = processes.stop(&self.worker_id);
        }
    }
}

impl WorkerProcessArgs {
    /// Has the daemons start every rank of one group without waiting for it
    /// to register.
    ///
    /// Spawning is separated from adoption so several groups start their
    /// processes concurrently and then wait for all of their registrations
    /// together.
    ///
    /// `claims` names the Storage Server and claim each worker's node holds
    /// for the deployment. `expert_store` is the reservation of the socket
    /// the group's first rank serves its expert-parallel world's store on,
    /// when that rank is the world's rank 0.
    pub(super) fn spawn_workers(
        &self,
        cancel: Option<Arc<AtomicBool>>,
        registry: &Registry,
        processes: &Processes,
        claims: &StorageClaims,
        mut expert_store: Option<Reservation>,
    ) -> anyhow::Result<LaunchedWorkers> {
        let cancel = cancel.unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        crate::WorkerGroupConfig::validate_members(&self.ranks, &self.components)?;
        anyhow::ensure!(
            self.components.is_empty() != self.role.serves_requests(),
            "worker role disagrees with request component placement"
        );
        if self.role == crate::WorkerGroupRole::Experts {
            anyhow::ensure!(
                self.expert_parallel.as_ref().is_some_and(|placement| {
                    placement.attention_ranks > 0 && placement.rank >= placement.attention_ranks
                }),
                "expert workers must belong to a disaggregated expert union"
            );
        }

        // A group of several ranks rendezvouses at one TCP address for its
        // entire lifetime. The first rank serves the rendezvous store, so the
        // daemon of that rank's node binds the store's socket and holds it
        // until that rank inherits it, so the port is never free while the
        // rank starts. The address is loopback where every rank runs on that
        // node and the node's routable address otherwise.
        let spans_nodes = |host: &str| self.ranks.iter().any(|rank| rank.host != host);
        let rendezvous = if self.ranks.len() > 1 && self.expert_parallel.is_none() {
            let first = &self.ranks[0].host;
            Some(crate::worker::processes::lock(processes)?.reserve(
                first,
                &self.worker_id,
                spans_nodes(first),
            )?)
        } else {
            None
        };

        // Every rank connects to the engine and registers as a member of
        // this launch: through the Unix socket from the engine's node, and
        // through the TCP port, at the address other nodes reach the engine's
        // node at, from any other. Beginning the launch ends the group's
        // earlier one, whose ranks the engine refuses from here on.
        let reachable = if self.ranks.iter().any(|rank| rank.host != self.host) {
            crate::worker::processes::lock(processes)?.address(&self.host)?
        } else {
            None
        };
        let (generation, local_address, remote_address) = {
            let mut registry = crate::worker::registry::lock(registry)?;
            (
                registry.begin_launch(&self.worker_id),
                registry.address(None)?,
                reachable
                    .map(|host| registry.address(Some(host)))
                    .transpose()?,
            )
        };
        let mut launched = LaunchedWorkers {
            registry: Arc::clone(registry),
            ranks: Vec::with_capacity(self.ranks.len()),
            processes: Arc::clone(processes),
            worker_id: self.worker_id.clone(),
        };
        for rank in 0..self.ranks.len() {
            // Only the first rank serves the stores; the others are their
            // clients and inherit nothing.
            let sockets = WorkerSockets {
                group: rendezvous
                    .as_ref()
                    .filter(|_| rank == 0)
                    .map(|reserved| (reserved.id, reserved.descriptor)),
                experts: if rank == 0 {
                    expert_store
                        .take()
                        .map(|reserved| (reserved.id, reserved.descriptor))
                } else {
                    None
                },
            };
            let address = match (&remote_address, self.ranks[rank].host == self.host) {
                (Some(address), false) => address.as_str(),
                _ => local_address.as_str(),
            };
            launched.ranks.push(PendingWorker::spawn(
                self,
                rank as u32,
                StorageLaunch {
                    standby: self.standby,
                    standby_memory: self.standby_memory,
                    ..claims.launch(&self.ranks[rank].host)?
                },
                rendezvous
                    .as_ref()
                    .map(|reserved| reserved.address.as_str()),
                sockets,
                processes,
                Arc::clone(&cancel),
                address,
                generation,
            )?);
        }
        Ok(launched)
    }

    /// Takes the connection each rank registered on and answers it with the
    /// rank's launch descriptor.
    pub(super) fn adopt_workers(
        &self,
        mut launched: LaunchedWorkers,
    ) -> anyhow::Result<Vec<WorkerProcess>> {
        let expected = launched.ranks.len();
        let registry = Arc::clone(&launched.registry);
        let ranks = &mut launched.ranks;
        // A rank that exits before registering reaches the engine as its
        // daemon's exit report, which fails the launch by name rather than
        // consuming the registration deadline.
        let alive = || {
            ranks
                .iter_mut()
                .try_for_each(|pending| pending.check_alive())
        };
        let channels =
            crate::worker::registry::collect(&registry, &self.worker_id, expected, alive)?;

        // `collect` orders the connections by rank, as `ranks` is.
        std::mem::take(&mut launched.ranks)
            .into_iter()
            .zip(channels)
            .map(|(pending, channel)| pending.adopt(channel))
            .collect::<anyhow::Result<Vec<_>>>()
    }
}

/// Starts every group of `arguments` and waits until each of their workers
/// is ready: for a deployment's first launch, and for the relaunch of a
/// failure domain, whose groups form their communicators again together.
///
/// Groups that share an expert world meet at a new rendezvous, which the
/// world's rank 0 serves on a socket its node's daemon reserves, routable
/// when some worker runs on another node, and which that rank inherits, so
/// the port is held from the reservation until it serves on it. `arguments`
/// take the rendezvous's address. Every group's processes start before any
/// group waits for registrations, so their interpreters and library imports
/// overlap, and they are awaited together, so the loss of any one ends the
/// wait of all. Returns each group's workers, in argument order.
///
/// `startup_cancel` is the startup cancellation every launched rank shares
/// until the launch is ready: the teardown of any rank sets it, and so may
/// the caller from another thread, which ends the launch at its next
/// liveness check and stops its ranks.
pub(super) fn launch_all(
    arguments: &mut [WorkerProcessArgs],
    registry: &Registry,
    processes: &Processes,
    claims: &StorageClaims,
    startup_cancel: Arc<AtomicBool>,
) -> anyhow::Result<Vec<Vec<WorkerProcess>>> {
    let mut expert_store = None;
    if arguments.iter().any(|args| args.expert_parallel.is_some()) {
        // The world's rank 0 is the first rank of the group whose placement
        // starts at union rank 0.
        let first = arguments
            .iter()
            .find(|args| {
                args.expert_parallel
                    .as_ref()
                    .is_some_and(|placement| placement.rank == 0)
            })
            .context("no worker group holds the expert-parallel world's rank 0")?;
        let host = first.ranks[0].host.clone();
        let routable = arguments
            .iter()
            .any(|args| args.ranks.iter().any(|rank| rank.host != host));
        let reservation = crate::worker::processes::lock(processes)?.reserve(
            &host,
            &first.worker_id,
            routable,
        )?;
        for args in arguments.iter_mut() {
            let placement = args.expert_parallel.as_mut().with_context(|| {
                format!(
                    "worker {} does not join the deployment's expert-parallel world",
                    args.worker_id
                )
            })?;
            placement.address = Some(reservation.address.clone());
        }
        expert_store = Some(reservation);
    }

    let launched = arguments
        .iter()
        .map(|args| {
            let store = match &args.expert_parallel {
                Some(placement) if placement.rank == 0 => expert_store.take(),
                _ => None,
            };
            args.spawn_workers(
                Some(Arc::clone(&startup_cancel)),
                registry,
                processes,
                claims,
                store,
            )
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut groups = arguments
        .iter()
        .zip(launched)
        .map(|(args, ranks)| args.adopt_workers(ranks))
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut starting: Vec<&mut WorkerProcess> = groups.iter_mut().flatten().collect();
    crate::worker::process::finish_startup(&mut starting)?;
    // Detach cancellation only after every group is ready. A later startup
    // failure must still terminate already prepared peers.
    for worker in groups.iter_mut().flatten() {
        worker.check_worker("deployment readiness")?;
    }
    for worker in groups.iter_mut().flatten() {
        worker.set_startup_cancel(None);
    }
    Ok(groups)
}

impl WorkerGroup {
    /// Launches every configured worker and exposes the group after capability agreement.
    pub fn spawn(process_args: WorkerProcessArgs) -> anyhow::Result<Self> {
        Self::spawn_all(vec![process_args])?
            .pop()
            .context("the worker group launch produced no group")
    }

    /// Launches the configured static rank groups and waits for loaded capabilities.
    ///
    /// The engine connects to the leader daemon at the first group's
    /// `daemon` address, which starts every rank on its node. Groups are
    /// returned in argument order. A failure of any group fails the whole
    /// launch; ranks already spawned are shut down as the values that own
    /// them drop.
    pub fn spawn_all(mut arguments: Vec<WorkerProcessArgs>) -> anyhow::Result<Vec<Self>> {
        let first = arguments
            .first()
            .context("a launch places a worker group")?;
        let processes = ProcessClient::connect(&first.daemon, &first.deployment)?;
        // One listener serves every group of the deployment. Its TCP port is
        // routable only when some rank runs on another node.
        let remote = arguments
            .iter()
            .any(|args| args.ranks.iter().any(|rank| rank.host != args.host));
        let registry = crate::worker::registry::WorkerRegistry::bind(remote)?;
        // Every node's Storage Server holds the deployment's claim before a
        // worker starts there, so the workers find their retained memory
        // and the server shoots down processes of a previous claim first.
        let claims = Claims::claim(&arguments, &processes)?;
        let groups = launch_all(
            &mut arguments,
            &registry,
            &processes,
            &claims,
            Arc::new(AtomicBool::new(false)),
        )?;

        // Groups that share an expert world fail and are replaced together;
        // every other group is its own failure domain.
        let wakes = (0..groups.len())
            .map(|_| crate::handle::WakeSignal::new())
            .collect::<std::io::Result<Vec<_>>>()?;
        let domain = |wakes: &[crate::handle::WakeSignal]| {
            Arc::new(Relaunch::new(
                Arc::clone(&registry),
                Arc::clone(&processes),
                Arc::clone(&claims),
                wakes.iter().map(crate::handle::WakeSignal::waker).collect(),
            ))
        };
        let domains: Vec<(Arc<Relaunch>, usize)> =
            if arguments.iter().any(|args| args.expert_parallel.is_some()) {
                let shared = domain(&wakes);
                (0..wakes.len())
                    .map(|index| (Arc::clone(&shared), index))
                    .collect()
            } else {
                wakes
                    .iter()
                    .map(|wake| (domain(std::slice::from_ref(wake)), 0))
                    .collect()
            };

        let mut workers = Vec::with_capacity(groups.len());
        for (((ranks, args), relaunch), failure_wake) in
            groups.into_iter().zip(arguments).zip(domains).zip(wakes)
        {
            workers.push(Self::from_ranks(
                args,
                ranks,
                Arc::clone(&processes),
                Arc::clone(&registry),
                Arc::clone(&claims),
                relaunch,
                failure_wake,
            )?);
        }
        // Groups that keep a standby start it once every group serves.
        for worker in &mut workers {
            worker.start_standby();
        }
        Ok(workers)
    }

    /// Checks that the ranks' startup reports describe one group and builds
    /// it, replaced with `relaunch` and woken by `failure_wake`.
    ///
    /// Each rank validated the report it built; the group registers each
    /// rank's incarnation as the rank it launched. The group takes rank 0's report:
    /// the ranks derive every shared capacity (block size, request slots,
    /// cache units) collectively before they report, so the reports differ
    /// only in the fields reconciled here. The muxer must run on the engine's
    /// host, and the ranks' cache regions must cover every layer's KV heads
    /// without a gap. Once all checks pass, no rank's startup may be
    /// cancelled and every rank this process started must still be running.
    fn from_ranks(
        process_args: WorkerProcessArgs,
        mut workers: Vec<WorkerProcess>,
        processes: Processes,
        registry: Registry,
        claims: StorageClaims,
        relaunch: (Arc<Relaunch>, usize),
        failure_wake: crate::handle::WakeSignal,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(!workers.is_empty(), "need >= 1 worker");
        let n = workers.len();

        // Rank 0's report becomes the group's.
        let mut info = workers[0].info().clone();

        // A rank's product storage follows the components it holds, which an
        // asymmetric placement makes rank-specific: a rank that executes or
        // reads a component backs the whole allocation of its products, and
        // a rank that does neither backs nothing. The executor places media
        // buffers rank by rank from each rank's own report
        // (`ExecutorInfo::rank_buffer_bytes`). The runtime's shared pool is
        // one layout every rank holds, so the group reports the smallest
        // nonzero pool, which no rank that holds products falls short of.
        info.buffer_pool_bytes = workers
            .iter()
            .map(|worker| worker.info().buffer_pool_bytes)
            .filter(|bytes| *bytes > 0)
            .min()
            .unwrap_or(0);

        // A rank's transfer entries follow the products it imports, which
        // its components set; the group holds what its narrowest rank does.
        info.transfer_tickets = workers
            .iter()
            .map(|worker| worker.info().transfer_tickets)
            .min()
            .unwrap_or(info.transfer_tickets);

        // A rank states the shapes the engine warms, and the image
        // denoiser's image-unconditional prefix, only for what it holds: the
        // image denoiser's ranks its image shapes and that prefix, a diffusion
        // LM's ranks its step rows, every other rank none. The group's are
        // those its stating ranks agree on.
        let mut image_warmups = Vec::new();
        let mut captured_images = None;
        let mut captured_latents = None;
        let mut image_capacity = None;
        let mut diffusion_lm_step_rows = 0;
        let mut image_unconditional = None;
        for (rank, worker) in workers.iter().enumerate() {
            let stated = worker.info();
            if let Some(tokens) = &stated.image_unconditional_token_ids {
                anyhow::ensure!(
                    image_unconditional
                        .as_ref()
                        .is_none_or(|known| known == tokens),
                    "physical rank {rank} states another image-unconditional prefix than another denoiser rank"
                );
                image_unconditional = Some(tokens.clone());
            }
            if !stated.image_warmups.is_empty() {
                anyhow::ensure!(
                    image_warmups.is_empty() || image_warmups == stated.image_warmups,
                    "physical rank {rank} states other image warmup shapes than another denoiser rank"
                );
                image_warmups.clone_from(&stated.image_warmups);
            }
            if let Some(shapes) = &stated.captured_images {
                anyhow::ensure!(
                    captured_images.as_ref().is_none_or(|known| known == shapes),
                    "physical rank {rank} states other captured image steps than another denoiser rank"
                );
                captured_images = Some(shapes.clone());
            }
            if let Some(sizes) = &stated.captured_latents {
                anyhow::ensure!(
                    captured_latents.as_ref().is_none_or(|known| known == sizes),
                    "physical rank {rank} states other captured latent image sizes than another \
                     latent encoder rank"
                );
                captured_latents = Some(sizes.clone());
            }
            if let Some(capacity) = stated.image_capacity {
                anyhow::ensure!(
                    image_capacity.is_none_or(|known| known == capacity),
                    "physical rank {rank} states another image capacity than another image rank"
                );
                image_capacity = Some(capacity);
            }
            if stated.diffusion_lm_step_rows > 0 {
                anyhow::ensure!(
                    diffusion_lm_step_rows == 0
                        || diffusion_lm_step_rows == stated.diffusion_lm_step_rows,
                    "physical rank {rank} states other diffusion LM step rows than another rank"
                );
                diffusion_lm_step_rows = stated.diffusion_lm_step_rows;
            }
        }
        info.image_warmups = image_warmups;
        info.captured_images = captured_images;
        info.captured_latents = captured_latents;
        info.image_capacity = image_capacity;
        info.diffusion_lm_step_rows = diffusion_lm_step_rows;
        info.image_unconditional_token_ids = image_unconditional;

        refuse_shared_storage_off_engine_host(
            &process_args.host,
            &process_args.ranks,
            &process_args.components,
            &info.media_components,
        )?;

        // Every logical cache layer, numbered from zero without a gap, must
        // belong to one group on every rank that stores it, and in each group
        // every layer must have its KV heads covered from head 0 to
        // `total_kv_heads` by the ranks storing it. Overlapping regions are
        // allowed; a gap is not. The group reports the union of its ranks'
        // layers, which bounds exports.
        if let Some(cache) = &mut info.prefix_cache {
            let regions: Vec<_> = workers
                .iter()
                .filter_map(|worker| worker.info().prefix_cache.as_ref())
                .collect();
            // Every rank describes the same cache groups in one order; the
            // coverage below reads group `index` of each.
            anyhow::ensure!(
                regions
                    .iter()
                    .all(|region| region.groups.len() == cache.groups.len()),
                "physical ranks report different cache groups"
            );
            let mut owners = BTreeMap::new();
            for (index, group) in cache.groups.iter_mut().enumerate() {
                let mut layers: BTreeMap<u32, Vec<(u32, u32)>> = BTreeMap::new();
                for region in &regions {
                    let local = &region.groups[index];
                    for layer in &local.layer_ids {
                        layers.entry(*layer).or_default().push((
                            local.kv_head_offset,
                            local.kv_head_offset + local.num_kv_heads,
                        ));
                    }
                }
                for (layer, mut heads) in layers {
                    anyhow::ensure!(
                        *owners.entry(layer).or_insert(index) == index,
                        "worker cache layer {layer} belongs to two cache groups"
                    );
                    heads.sort_unstable();
                    let mut covered = 0;
                    for (start, end) in heads {
                        anyhow::ensure!(
                            start <= covered,
                            "worker cache regions leave a logical head gap"
                        );
                        covered = covered.max(end);
                    }
                    anyhow::ensure!(
                        covered == group.total_kv_heads,
                        "worker cache regions do not cover the heads of layer {layer}"
                    );
                }
                group.layer_ids = owners
                    .iter()
                    .filter(|(_, owner)| **owner == index)
                    .map(|(layer, _)| *layer)
                    .collect();
            }
            anyhow::ensure!(
                owners.keys().copied().eq(0..owners.len() as u32),
                "worker cache regions do not cover every logical cache layer"
            );
        }

        // Keep startup cancellation attached until every deployment group
        // is ready; expert peers may still be preparing their resources.
        for worker in &mut workers {
            worker.check_worker("WorkerGroup readiness")?;
        }

        // The group admits the batches it launched its ranks to hold.
        let depth = process_args.queue_depth.max(1);
        let buffers = (0..n).map(|_| VecDeque::new()).collect();
        // Each process serves the rank the group launched it as.
        let incarnations = IncarnationRegistry::default();
        incarnations.assign(
            &crate::WorkerGroupId(process_args.worker_id.clone()),
            workers.iter().map(|worker| &worker.info().endpoint),
        );
        Ok(Self {
            processes,
            registry,
            claims,
            workers,
            buffers,
            depth,
            last_batch_id: None,
            last_progress: Instant::now(),
            pending_batches: BTreeMap::new(),
            media_routing: info.media_components.clone(),
            incarnations,
            info,
            process_args,
            resident_requests: HashSet::new(),
            readiness_changed: false,
            closed: false,
            failure: None,
            relaunch,
            failure_wake,
            awaiting: None,
            standby: None,
        })
    }
}
