//! Launching executor groups: the ranks of a group started by the daemons of
//! their nodes, each from the launch its `ExecutorProcessArgs` derive, and the
//! group formed from those ranks once they registered and reported their
//! startup capabilities. `spawn_all` starts every group's ranks before any
//! group waits for its own.

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use anyhow::Context;

use super::ExecutorGroup;
use super::relaunch::Relaunch;
use super::routing::refuse_shared_storage_off_engine_host;
use crate::executors::ExecutorProcessArgs;
use crate::executors::claims::{Claims, StorageClaims};
use crate::executors::process::ExecutorSockets;
use crate::executors::processes::{ProcessClient, Processes, Reservation};
use crate::executors::registry::Registry;
use crate::executors::{IncarnationRegistry, PendingWorker, ExecutorProcess};
use ipc::StorageLaunch;

/// One executor group's spawned ranks and the launch they register for.
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
            && let Ok(mut processes) = crate::executors::processes::lock(&self.processes)
        {
            let _ = processes.stop(&self.worker_id);
        }
    }
}

impl ExecutorProcessArgs {
    /// Has the daemons start every rank of one group without waiting for it
    /// to register.
    ///
    /// Spawning is separated from adoption so several groups start their
    /// processes concurrently and then wait for all of their registrations
    /// together.
    ///
    /// `claims` names the Storage Server and claim each executor's node holds
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
            "executor role disagrees with request component placement"
        );
        if self.role == crate::WorkerGroupRole::Experts {
            anyhow::ensure!(
                self.expert_parallel.as_ref().is_some_and(|placement| {
                    placement.attention_ranks > 0 && placement.rank >= placement.attention_ranks
                }),
                "expert executors must belong to a disaggregated expert union"
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
            Some(crate::executors::processes::lock(processes)?.reserve(
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
            crate::executors::processes::lock(processes)?.address(&self.host)?
        } else {
            None
        };
        let (generation, local_address, remote_address) = {
            let mut registry = crate::executors::registry::lock(registry)?;
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
            let sockets = ExecutorSockets {
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
                    standby_storage: self.standby_storage,
                    world: self.world,
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
    ) -> anyhow::Result<Vec<ExecutorProcess>> {
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
            crate::executors::registry::collect(&registry, &self.worker_id, expected, alive)?;

        // `collect` orders the connections by rank, as `ranks` is.
        std::mem::take(&mut launched.ranks)
            .into_iter()
            .zip(channels)
            .map(|(pending, channel)| pending.adopt(channel))
            .collect::<anyhow::Result<Vec<_>>>()
    }
}

/// Starts every group of `arguments` and waits until each of their executors
/// is ready: for a deployment's first launch, and for the relaunch of a
/// failure domain, whose groups form their communicators again together.
///
/// Groups that share an expert world meet at a new rendezvous, which the
/// world's rank 0 serves on a socket its node's daemon reserves, routable
/// when some executor runs on another node, and which that rank inherits, so
/// the port is held from the reservation until it serves on it. `arguments`
/// take the rendezvous's address. Every group's processes start before any
/// group waits for registrations, so their interpreters and library imports
/// overlap, and they are awaited together, so the loss of any one ends the
/// wait of all. Returns each group's executors, in argument order.
///
/// `startup_cancel` is the startup cancellation every launched rank shares
/// until the launch is ready: the teardown of any rank sets it, and so may
/// the caller from another thread, which ends the launch at its next
/// liveness check and stops its ranks.
pub(super) fn launch_all(
    arguments: &mut [ExecutorProcessArgs],
    registry: &Registry,
    processes: &Processes,
    claims: &StorageClaims,
    startup_cancel: Arc<AtomicBool>,
) -> anyhow::Result<Vec<Vec<ExecutorProcess>>> {
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
            .context("no executor group holds the expert-parallel world's rank 0")?;
        let host = first.ranks[0].host.clone();
        let routable = arguments
            .iter()
            .any(|args| args.ranks.iter().any(|rank| rank.host != host));
        let reservation = crate::executors::processes::lock(processes)?.reserve(
            &host,
            &first.worker_id,
            routable,
        )?;
        for args in arguments.iter_mut() {
            let placement = args.expert_parallel.as_mut().with_context(|| {
                format!(
                    "executor {} does not join the deployment's expert-parallel world",
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

    let mut starting: Vec<&mut ExecutorProcess> = groups.iter_mut().flatten().collect();
    crate::executors::process::finish_startup(&mut starting)?;
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

impl ExecutorGroup {
    /// Launches every configured executor and exposes the group after capability agreement.
    pub fn spawn(process_args: ExecutorProcessArgs) -> anyhow::Result<Self> {
        Self::spawn_all(vec![process_args])?
            .pop()
            .context("the executor group launch produced no group")
    }

    /// Launches the configured static rank groups and waits for loaded capabilities.
    ///
    /// The engine connects to the leader daemon at the first group's
    /// `daemon` address, which starts every rank on its node. Groups are
    /// returned in argument order. A failure of any group fails the whole
    /// launch; ranks already spawned are shut down as the values that own
    /// them drop.
    pub fn spawn_all(mut arguments: Vec<ExecutorProcessArgs>) -> anyhow::Result<Vec<Self>> {
        let first = arguments
            .first()
            .context("a launch places an executor group")?;
        // A spawner is not a main process: it states no engine number, and no program-layer command
        // is ever addressed to this connection.
        let processes = ProcessClient::connect(&first.daemon, &first.deployment, 0)?;
        // One listener serves every group of the deployment. Its TCP port is
        // routable only when some rank runs on another node.
        let remote = arguments
            .iter()
            .any(|args| args.ranks.iter().any(|rank| rank.host != args.host));
        let registry = crate::executors::registry::ExecutorRegistry::bind(remote)?;
        // Every node's Storage Server holds the deployment's claim before a
        // executor starts there, so the executors find their retained storage
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
    /// rank's incarnation as the rank it launched. The group's report is
    /// `inventory::pool_info`: rank 0's, with every value the ranks state
    /// jointly reconciled, which is the same derivation a deployment served in
    /// process gets from its executor. The muxer must run on the engine's host,
    /// and the ranks' cache regions must cover every layer's KV heads without a
    /// gap (`pool_info` refuses a gap). Once all checks pass, no rank's startup may be
    /// cancelled and every rank this process started must still be running.
    fn from_ranks(
        process_args: ExecutorProcessArgs,
        mut workers: Vec<ExecutorProcess>,
        processes: Processes,
        registry: Registry,
        claims: StorageClaims,
        relaunch: (Arc<Relaunch>, usize),
        failure_wake: crate::handle::WakeSignal,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(!workers.is_empty(), "need >= 1 executor");
        let n = workers.len();

        // The pool's agreed report is one derivation for both ways of serving
        // a deployment: this path reads it from the ranks' startup reports and
        // the in-process path from what the executor publishes (`ExecutorInfo::from_inventory`).
        let ranks = workers
            .iter()
            .map(|worker| worker.info().clone())
            .collect::<Vec<_>>();
        let info = crate::executor::inventory::pool_info(&ranks)?;

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
        // is ready; expert peers may still be preparing their resources.
        for worker in &mut workers {
            worker.check_worker("ExecutorGroup readiness")?;
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
            host_cache: None,
        })
    }
}
