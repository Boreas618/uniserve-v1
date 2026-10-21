//! Launching worker groups: the ranks of a group started here or through their
//! host's launcher, each from the launch its `WorkerProcessArgs` derive, and
//! the group formed from those ranks once they registered and reported their
//! startup capabilities. `spawn_all` starts every group's ranks before any
//! group waits for its own.

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use anyhow::Context;

use super::WorkerGroup;
use super::domain::FailureDomain;
use super::routing::refuse_shared_storage_off_head;
use crate::worker::WorkerProcessArgs;
use crate::worker::registry::Registry;
use crate::worker::storage::{Claims, StorageClaims};
use crate::worker::{IncarnationRegistry, PendingWorker, WorkerProcess};
use uniserve_ipc::StorageLaunch;

/// Every rank of one group, with the launchers that started the remote ones.
type LaunchedGroup = (
    Vec<WorkerProcess>,
    Option<crate::worker::launcher::Launchers>,
);

/// One worker group's spawned ranks and the launch they register for.
pub(crate) struct LaunchedWorkers {
    /// The registry the ranks register with.
    registry: Registry,
    ranks: Vec<PendingWorker>,
    /// Retained so this head's launchers stay connected; closing a
    /// launcher connection terminates the ranks it started.
    launchers: Option<crate::worker::launcher::Launchers>,
}

impl WorkerProcessArgs {
    /// Starts every rank of one group without waiting for it to register.
    ///
    /// Spawning is separated from adoption so several groups start their
    /// processes concurrently and then wait for all of their registrations
    /// together.
    ///
    /// `claims` names the Storage Server and claim each worker's host
    /// holds for the deployment. `expert_store` is the socket the group's
    /// first rank serves its expert-parallel world's store on, when that
    /// rank is the world's rank 0.
    pub(super) fn spawn_workers(
        &self,
        cancel: Option<Arc<AtomicBool>>,
        registry: &Registry,
        launchers: Option<crate::worker::launcher::Launchers>,
        claims: &StorageClaims,
        mut expert_store: Option<std::net::TcpListener>,
    ) -> anyhow::Result<LaunchedWorkers> {
        let cancel = cancel.unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        crate::WorkerGroupConfig::validate_members(&self.ranks, &self.components)?;
        anyhow::ensure!(
            self.components.is_empty() == (self.role == crate::WorkerGroupRole::Experts),
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
        // This process spawns exactly the ranks placed on its own host. A rank
        // placed elsewhere is started by that host's launcher, which presented
        // to the deployment's registry before any rank is sent to it.
        let places_remotely = self.ranks.iter().any(|rank| rank.host != self.host);
        let launchers = match launchers {
            Some(registry) if places_remotely => Some(registry),
            None if places_remotely => anyhow::bail!(
                "worker {} places a rank on another host but no launcher registry was bound",
                self.worker_id
            ),
            _ => None,
        };

        // A group of several ranks rendezvouses at one TCP address for its
        // entire lifetime. It is loopback where every rank runs here; where
        // some rank runs elsewhere it names a routable host, because a
        // loopback address reaches only the host that binds it. The first
        // rank serves the rendezvous store, so the store is placed on that
        // rank's host: this one, or the host whose launcher reserved a port
        // for it. Whichever process spawns the first rank holds the bound
        // socket until that rank inherits it, so the port is never free while
        // the rank starts.
        let reachable = match launchers.as_ref() {
            Some(registry) => Some(crate::worker::launcher::lock(registry)?.reachable_host()?),
            None => None,
        };
        let mut rendezvous = if self.ranks.len() > 1 && self.expert_parallel.is_none() {
            let first = &self.ranks[0].host;
            Some(match launchers.as_ref() {
                Some(registry) if *first != self.host => crate::worker::launcher::lock(registry)?
                    .rendezvous_on(first, &self.worker_id)?,
                _ => crate::worker::rendezvous::reserve_rendezvous(reachable)?,
            })
        } else {
            None
        };

        // Every rank connects to the head and registers as a member of this
        // launch: through the Unix socket from the head's host, and through
        // the TCP port, at an address its host routes to, from any other.
        // Beginning the launch ends the group's earlier one, whose ranks the
        // head refuses from here on.
        let (generation, local_address, remote_address) = {
            let mut head = crate::worker::registry::lock(registry)?;
            (
                head.begin_launch(&self.worker_id),
                head.address(None)?,
                reachable.map(|host| head.address(Some(host))).transpose()?,
            )
        };
        let mut ranks = Vec::with_capacity(self.ranks.len());
        for rank in 0..self.ranks.len() {
            // A rank placed elsewhere is delivered to the launcher that owns
            // its host; a rank placed here is spawned by this process.
            let mut remote = match (&launchers, self.ranks[rank].host == self.host) {
                (Some(registry), false) => Some(crate::worker::launcher::RemoteHost {
                    registry: registry.as_ref(),
                    host: &self.ranks[rank].host,
                }),
                _ => None,
            };
            // Only the first rank serves the store; the others are its clients
            // and bind nothing, so they receive only its address.
            let store_listener = match rendezvous.as_mut() {
                Some(rendezvous) if rank == 0 => rendezvous.listener.take(),
                _ => None,
            };
            let sockets = crate::worker::process::WorkerSockets {
                group: store_listener,
                experts: if rank == 0 { expert_store.take() } else { None },
            };
            // A first rank on another host inherits the store socket from
            // its launcher, at the number the launcher reported.
            let store_fd = match rendezvous.as_ref() {
                Some(rendezvous) if rank == 0 => rendezvous.listen_fd,
                _ => None,
            };
            let address = match (&remote, &remote_address) {
                (Some(_), Some(address)) => address.as_str(),
                _ => local_address.as_str(),
            };
            ranks.push(PendingWorker::spawn(
                self,
                rank as u32,
                StorageLaunch {
                    standby: self.standby,
                    ..claims.launch(&self.ranks[rank].host)?
                },
                rendezvous
                    .as_ref()
                    .map(|rendezvous| rendezvous.address.as_str()),
                sockets,
                store_fd,
                remote.as_mut(),
                Arc::clone(&cancel),
                address,
                generation,
            )?);
        }

        Ok(LaunchedWorkers {
            registry: Arc::clone(registry),
            ranks,
            launchers,
        })
    }

    /// Takes the connection each rank registered on and answers it with the
    /// rank's launch descriptor.
    pub(super) fn adopt_workers(&self, launched: LaunchedWorkers) -> anyhow::Result<LaunchedGroup> {
        let LaunchedWorkers {
            registry,
            mut ranks,
            mut launchers,
        } = launched;
        let expected = ranks.len();
        let alive = || {
            // A rank this process spawned is checked here; a rank another
            // host started reaches the head as an exit report from the
            // launcher that owns it, which fails the launch by name rather
            // than consuming the registration deadline.
            if let Some(exits) = launchers
                .as_ref()
                .map(crate::worker::launcher::lock)
                .transpose()?
                .map(|mut hosts| hosts.drain_exits(&self.worker_id))
                && let Some((host, exit)) = exits.into_iter().next()
            {
                anyhow::bail!(
                    "rank {} on host {host} exited with {} before registering",
                    exit.rank,
                    exit.status
                );
            }
            ranks
                .iter_mut()
                .try_for_each(|pending| pending.check_alive())
        };
        let channels =
            crate::worker::registry::collect(&registry, &self.worker_id, expected, alive)?;

        // `collect` orders the connections by rank, as `ranks` is.
        ranks
            .into_iter()
            .zip(channels)
            .map(|(pending, channel)| pending.adopt(channel))
            .collect::<anyhow::Result<Vec<_>>>()
            .map(|ranks| (ranks, launchers.take()))
    }
}

/// Starts every group of `arguments` and waits until each of their workers
/// is ready: for a deployment's first launch, and for the relaunch of a
/// failure domain, whose groups form their communicators again together.
///
/// Groups that share an expert world meet at a new rendezvous, which the
/// world's rank 0 serves on a socket the head binds here, routable when some
/// worker runs on another host, and which that rank inherits, so the port is
/// held from the reservation until it serves on it. `arguments` take the
/// rendezvous's address. Every group's processes start before any group
/// waits for registrations, so their interpreters and library imports
/// overlap, and they are awaited together, so the loss of any one ends the
/// wait of all. Returns each group's workers and launchers, in argument
/// order.
///
/// `startup_cancel` is the startup cancellation every launched rank shares
/// until the launch is ready: the teardown of any rank sets it, and so may
/// the caller from another thread, which ends the launch at its next
/// liveness check and terminates its ranks.
pub(super) fn launch_all(
    arguments: &mut [WorkerProcessArgs],
    registry: &Registry,
    launchers: Option<crate::worker::launcher::Launchers>,
    claims: &StorageClaims,
    startup_cancel: Arc<AtomicBool>,
) -> anyhow::Result<Vec<LaunchedGroup>> {
    let mut expert_store = None;
    if arguments.iter().any(|args| args.expert_parallel.is_some()) {
        let reachable = match launchers.as_ref() {
            Some(registry) => Some(crate::worker::launcher::lock(registry)?.reachable_host()?),
            None => None,
        };
        let rendezvous = crate::worker::rendezvous::reserve_rendezvous(reachable)?;
        for args in arguments.iter_mut() {
            let placement = args.expert_parallel.as_mut().with_context(|| {
                format!(
                    "worker {} does not join the deployment's expert-parallel world",
                    args.worker_id
                )
            })?;
            placement.address = Some(rendezvous.address.clone());
            if placement.rank == 0 {
                anyhow::ensure!(
                    args.ranks[0].host == args.host,
                    "expert-parallel rank 0 ({}) must run on the head's host",
                    args.worker_id
                );
            }
        }
        expert_store = rendezvous.listener;
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
                launchers.clone(),
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

    let mut starting: Vec<&mut WorkerProcess> = groups
        .iter_mut()
        .flat_map(|(ranks, _)| ranks.iter_mut())
        .collect();
    crate::worker::process::finish_startup(&mut starting)?;
    // Detach cancellation only after every group is ready. A later startup
    // failure must still terminate already prepared peers.
    for worker in groups.iter_mut().flat_map(|(ranks, _)| ranks.iter_mut()) {
        worker.check_worker("deployment readiness")?;
    }
    for worker in groups.iter_mut().flat_map(|(ranks, _)| ranks.iter_mut()) {
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
    /// Groups are returned in argument order. A failure of any group fails the
    /// whole launch; ranks already spawned are shut down as the values that
    /// own them, and the launcher registry, drop.
    pub fn spawn_all(mut arguments: Vec<WorkerProcessArgs>) -> anyhow::Result<Vec<Self>> {
        // One launcher per host serves every group of the deployment, so the
        // registry is bound and its hosts awaited once, before any group
        // sends a rank to them.
        // One listener serves every group of the deployment and the launchers
        // of its other hosts. Its TCP port is routable only when some rank
        // runs on another host.
        let remote = arguments
            .iter()
            .any(|args| args.ranks.iter().any(|rank| rank.host != args.host));
        let registry = crate::worker::registry::WorkerRegistry::bind(remote)?;
        let launchers = match arguments.first() {
            Some(first) => crate::worker::launcher::LauncherRegistry::for_hosts(
                &registry,
                &first.host,
                arguments.iter().map(|args| args.ranks.as_slice()),
                first.launcher_timeout,
            )?,
            None => None,
        };
        // Every host's Storage Server holds the deployment's claim before a
        // worker starts there, so the workers find their retained memory
        // and the server shoots down processes of a previous head first.
        let claims = Claims::claim(&arguments, launchers.as_ref())?;
        let groups = launch_all(
            &mut arguments,
            &registry,
            launchers.clone(),
            &claims,
            Arc::new(AtomicBool::new(false)),
        )?;

        // Groups that share an expert world fail and are replaced together;
        // every other group is its own failure domain.
        let wakes = (0..groups.len())
            .map(|_| crate::handle::WakeSignal::new())
            .collect::<std::io::Result<Vec<_>>>()?;
        let domain = |wakes: &[crate::handle::WakeSignal]| {
            Arc::new(FailureDomain::new(
                Arc::clone(&registry),
                launchers.clone(),
                Arc::clone(&claims),
                wakes.iter().map(crate::handle::WakeSignal::waker).collect(),
            ))
        };
        let domains: Vec<(Arc<FailureDomain>, usize)> =
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
        for ((((ranks, launchers), args), failure_domain), failure_wake) in
            groups.into_iter().zip(arguments).zip(domains).zip(wakes)
        {
            workers.push(Self::from_ranks(
                args,
                ranks,
                launchers,
                Arc::clone(&registry),
                Arc::clone(&claims),
                failure_domain,
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
    /// it, replaced with `failure_domain` and woken by `failure_wake`.
    ///
    /// Each rank validated the report it built; the group registers each
    /// rank's incarnation as the rank it launched. The group takes rank 0's report:
    /// the ranks derive every shared capacity (block size, request slots,
    /// cache units) collectively before they report, so the reports differ
    /// only in the fields reconciled here. The muxer must run on the head's
    /// host, and the ranks' cache regions must cover every layer's KV heads
    /// without a gap. Once all checks pass, no rank's startup may be
    /// cancelled and every rank this process started must still be running.
    fn from_ranks(
        process_args: WorkerProcessArgs,
        mut workers: Vec<WorkerProcess>,
        launchers: Option<crate::worker::launcher::Launchers>,
        registry: Registry,
        claims: StorageClaims,
        failure_domain: (Arc<FailureDomain>, usize),
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
        info.diffusion_lm_step_rows = diffusion_lm_step_rows;
        info.image_unconditional_token_ids = image_unconditional;

        refuse_shared_storage_off_head(
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
            launchers,
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
            failure_domain,
            failure_wake,
            awaiting: None,
            standby: None,
        })
    }
}
