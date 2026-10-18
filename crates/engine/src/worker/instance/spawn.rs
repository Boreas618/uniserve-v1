//! Launching worker groups: the ranks of a group started here or through their
//! host's launcher, each from the launch its `WorkerProcessArgs` derive, and
//! the group formed from those ranks once they registered and reported their
//! startup capabilities. `spawn_all` starts every group's ranks before any
//! group waits for its own.

use std::collections::{BTreeMap, HashSet, VecDeque};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

use anyhow::Context;

use super::routing::refuse_shared_storage_off_head;
use super::{FailureDomain, WorkerGroup};
use crate::worker::WorkerProcessArgs;
use crate::worker::registry::Registry;
use crate::worker::{PendingRank, RankProcess};

/// Every rank of one group, with the launchers that started the remote ones.
type LaunchedGroup = (Vec<RankProcess>, Option<crate::worker::launcher::Launchers>);

/// One worker group's spawned ranks and the launch they register for.
pub(crate) struct LaunchedRanks {
    /// The registry the ranks register with.
    registry: Registry,
    ranks: Vec<PendingRank>,
    /// Retained so this instance's launchers stay connected; closing a
    /// launcher connection terminates the ranks it started.
    launchers: Option<crate::worker::launcher::Launchers>,
}

impl WorkerProcessArgs {
    /// Launches and connects every rank in one physical worker group.
    ///
    /// `registry` is the deployment's listener, which the ranks connect to.
    /// `launchers` is the deployment's registry, which owns the launcher of
    /// every host this process does not run on; a group placing a rank on
    /// such a host requires it.
    ///
    /// A replica of an expert-parallel world is not relaunched: its peers
    /// hold that world's communicators and exchange tokens with it at every
    /// expert layer, so a replacement rank cannot join them alone.
    pub(super) fn launch(
        &self,
        cancel: Option<Arc<AtomicBool>>,
        registry: &Registry,
        launchers: Option<crate::worker::launcher::Launchers>,
    ) -> anyhow::Result<LaunchedGroup> {
        anyhow::ensure!(
            self.expert_parallel.is_none(),
            "worker {} shares its experts with other replicas and cannot rejoin their \
             expert-parallel world",
            self.worker_id
        );
        self.adopt_ranks(self.spawn_ranks(cancel, registry, launchers, None)?)
    }

    /// Starts every rank of one group without waiting for it to register.
    ///
    /// Spawning is separated from adoption so several groups start their
    /// processes concurrently and then wait for all of their registrations
    /// together.
    ///
    /// `expert_store` is the socket the group's first rank serves its
    /// expert-parallel world's store on, when that rank is the world's rank 0.
    fn spawn_ranks(
        &self,
        cancel: Option<Arc<AtomicBool>>,
        registry: &Registry,
        launchers: Option<crate::worker::launcher::Launchers>,
        mut expert_store: Option<std::net::TcpListener>,
    ) -> anyhow::Result<LaunchedRanks> {
        let cancel = cancel.unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
        crate::WorkerConfig::validate_members(&self.ranks, &self.components)?;
        anyhow::ensure!(
            self.components.is_empty() == (self.role == crate::WorkerRole::Experts),
            "worker role disagrees with request component placement"
        );
        if self.role == crate::WorkerRole::Experts {
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
        let places_remotely = self.ranks.iter().any(|rank| rank.node != self.host);
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
            let first = &self.ranks[0].node;
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
            let mut remote = match (&launchers, self.ranks[rank].node == self.host) {
                (Some(registry), false) => Some(crate::worker::launcher::RemoteHost {
                    registry: registry.as_ref(),
                    host: &self.ranks[rank].node,
                }),
                _ => None,
            };
            // Only the first rank serves the store; the others are its clients
            // and bind nothing, so they receive only its address.
            let store_listener = match rendezvous.as_mut() {
                Some(rendezvous) if rank == 0 => rendezvous.listener.take(),
                _ => None,
            };
            let sockets = crate::worker::process::RankSockets {
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
            ranks.push(PendingRank::spawn_rank(
                self,
                rank as u32,
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

        Ok(LaunchedRanks {
            registry: Arc::clone(registry),
            ranks,
            launchers,
        })
    }

    /// Takes the connection each rank registered on and answers it with the
    /// rank's launch descriptor.
    fn adopt_ranks(&self, launched: LaunchedRanks) -> anyhow::Result<LaunchedGroup> {
        let LaunchedRanks {
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
            crate::worker::registry::lock(&registry)?.collect(&self.worker_id, expected, alive)?;

        // `collect` orders the connections by rank, as `ranks` is.
        ranks
            .into_iter()
            .zip(channels)
            .map(|(pending, channel)| pending.adopt(channel))
            .collect::<anyhow::Result<Vec<_>>>()
            .map(|ranks| (ranks, launchers.take()))
    }
}

impl WorkerGroup {
    /// Launches every configured rank and exposes the instance after capability agreement.
    pub fn spawn(process_args: WorkerProcessArgs) -> anyhow::Result<Self> {
        Self::spawn_all(vec![process_args])?
            .pop()
            .context("WorkerGroup launch produced no instance")
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
            .any(|args| args.ranks.iter().any(|rank| rank.node != args.host));
        let registry = crate::worker::registry::RankRegistry::bind(remote)?;
        let launchers = match arguments.first() {
            Some(first) => crate::worker::launcher::LauncherRegistry::for_hosts(
                &registry,
                &first.host,
                arguments.iter().map(|args| args.ranks.as_slice()),
                first.launcher_timeout,
            )?,
            None => None,
        };

        // Replicas sharing their experts form one expert-parallel world at a
        // TCP store its rank 0 serves. The head binds the store's socket here,
        // routable when some rank runs on another host, and the world's rank
        // 0, which must run on this host, inherits it, so the port is held
        // from the reservation until that rank serves on it.
        let mut expert_store = None;
        if arguments.iter().any(|args| args.expert_parallel.is_some()) {
            let reachable = match launchers.as_ref() {
                Some(registry) => Some(crate::worker::launcher::lock(registry)?.reachable_host()?),
                None => None,
            };
            let rendezvous = crate::worker::rendezvous::reserve_rendezvous(reachable)?;
            for args in &mut arguments {
                let placement = args.expert_parallel.as_mut().with_context(|| {
                    format!(
                        "worker {} does not join the deployment's expert-parallel world",
                        args.worker_id
                    )
                })?;
                placement.address = Some(rendezvous.address.clone());
                if placement.rank == 0 {
                    anyhow::ensure!(
                        args.ranks[0].node == args.host,
                        "expert-parallel rank 0 ({}) must run on the head's host",
                        args.worker_id
                    );
                }
            }
            expert_store = rendezvous.listener;
        }

        // Every group's processes start before any group waits for
        // registrations, so their interpreter and library import overlap.
        let startup_cancel = Arc::new(AtomicBool::new(false));
        let launched = arguments
            .iter()
            .map(|args| {
                let store = match &args.expert_parallel {
                    Some(placement) if placement.rank == 0 => expert_store.take(),
                    _ => None,
                };
                args.spawn_ranks(
                    Some(Arc::clone(&startup_cancel)),
                    &registry,
                    launchers.clone(),
                    store,
                )
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let mut groups = arguments
            .iter()
            .zip(launched)
            .map(|(args, ranks)| args.adopt_ranks(ranks))
            .collect::<anyhow::Result<Vec<_>>>()?;

        // The ranks of every group are awaited together, so the loss of any
        // one of them ends the wait of all.
        let mut starting: Vec<&mut RankProcess> = groups
            .iter_mut()
            .flat_map(|(ranks, _)| ranks.iter_mut())
            .collect();
        crate::worker::process::finish_startup(&mut starting)?;

        let mut workers = Vec::with_capacity(groups.len());
        for ((ranks, launchers), args) in groups.into_iter().zip(arguments) {
            workers.push(Self::from_ranks(
                args,
                ranks,
                launchers,
                Arc::clone(&registry),
            )?);
        }
        if workers
            .iter()
            .any(|worker| worker.process_args.expert_parallel.is_some())
        {
            let wakes = (0..workers.len())
                .map(|_| crate::handle::WakeSignal::new())
                .collect::<std::io::Result<Vec<_>>>()?;
            let domain = Arc::new(FailureDomain {
                cause: OnceLock::new(),
                wakers: wakes.iter().map(crate::handle::WakeSignal::waker).collect(),
            });
            for (worker, wake) in workers.iter_mut().zip(wakes) {
                worker.failure_domain = Some(Arc::clone(&domain));
                worker.failure_wake = Some(wake);
            }
        }
        // Detach cancellation only after every group is ready. A later
        // startup failure must still terminate already prepared peers.
        for group in &mut workers {
            for rank in &mut group.workers {
                rank.check_worker("deployment readiness")?;
            }
        }
        for group in &mut workers {
            for rank in &mut group.workers {
                rank.set_startup_cancel(None);
            }
        }
        Ok(workers)
    }

    /// Checks that the ranks' startup reports describe one group and builds it.
    ///
    /// Every rank must report valid info naming its launched rank, the group's
    /// world size and worker identity. The group takes rank 0's report: the
    /// ranks derive every shared capacity (page size, request slots, cache
    /// units) collectively before they report, so the reports differ only in
    /// the fields reconciled here. The muxer must run on the head's host, and
    /// the ranks' cache regions must cover every layer's KV heads without a
    /// gap. Once all checks pass, no rank's startup may be cancelled and every
    /// rank this process started must still be running.
    fn from_ranks(
        process_args: WorkerProcessArgs,
        mut workers: Vec<RankProcess>,
        launchers: Option<crate::worker::launcher::Launchers>,
        registry: Registry,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(!workers.is_empty(), "need >= 1 worker");
        let n = workers.len();
        let world_size =
            u32::try_from(n).context("process world size exceeds the IPC representation")?;

        // Rank 0's report becomes the group's.
        let mut info = workers[0].info().clone();

        // A rank's product storage follows the components it holds, which an
        // asymmetric placement makes rank-specific: a rank that imports a
        // component's product reserves the whole logical allocation for it,
        // and a rank that neither produces nor consumes it reserves nothing.
        // The group reports the smallest, because the engine places work
        // against one pool per group and must not place more than the
        // smallest rank can hold.
        info.buffer_pool_bytes = workers
            .iter()
            .map(|worker| worker.info().buffer_pool_bytes)
            .min()
            .unwrap_or(info.buffer_pool_bytes);

        for (rank, worker) in workers.iter().enumerate() {
            let rank_info = worker.info();
            rank_info
                .validate()
                .with_context(|| format!("physical rank {rank} reported invalid worker info"))?;
            anyhow::ensure!(
                rank_info.endpoint.rank == rank as u32 && rank_info.world_size == world_size,
                "physical rank {rank} reported topology ({}/{}) for launched topology ({rank}/{world_size})",
                rank_info.endpoint.rank,
                rank_info.world_size,
            );
            anyhow::ensure!(
                rank_info.endpoint.worker_id == process_args.worker_id,
                "physical rank {rank} reported another WorkerGroup identity"
            );
        }

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
        // layers, which bounds exports, and the largest unit, because
        // the scheduler reserves units shared by every stage.
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
            cache.unit_bytes = regions
                .iter()
                .map(|region| region.unit_bytes)
                .max()
                .unwrap_or(cache.unit_bytes);
        }

        // Keep startup cancellation attached until every deployment group
        // is ready; expert peers may still be preparing their resources.
        for worker in &mut workers {
            worker.check_worker("WorkerGroup readiness")?;
        }

        let depth = info.queue_depth.max(1) as usize;
        let buffers = (0..n).map(|_| VecDeque::new()).collect();
        Ok(Self {
            launchers,
            registry,
            workers,
            buffers,
            depth,
            last_batch_id: None,
            last_progress: Instant::now(),
            pending_batches: BTreeMap::new(),
            media_routing: info.media_components.clone(),
            info,
            process_args,
            resident_requests: HashSet::new(),
            readiness_changed: false,
            closed: false,
            failure: None,
            failure_domain: None,
            failure_wake: None,
        })
    }
}
