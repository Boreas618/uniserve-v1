//! The deployment's claims at the Storage Servers of its nodes.
//!
//! Before any worker starts, the engine has the daemon of every node of the
//! deployment claim the deployment at that node's Storage Server
//! (`ProcessClient::claim`), naming the scopes the engine's world keeps
//! there (none on a node that runs none of its workers). The daemon holds
//! the claim for as long as it runs, so the claim outlives the engine. Each
//! worker starts with its node's socket and claim token ([`StorageLaunch`]),
//! by which the server tells it belongs to the deployment's current claim,
//! and the other nodes' servers, from which it may receive another world's
//! weights. The engine ends another world's memory on every node by
//! claiming its own scopes again with that world retired
//! ([`Claims::retire`]): the predecessor its commit supersedes, or the
//! successor a rollback discards, wherever they ran, so a world change that
//! moves the deployment to other nodes ends the memory the previous world
//! left on the nodes it leaves.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::Context;
use ipc::StorageLaunch;

use crate::worker::WorkerProcessArgs;
use crate::worker::processes::Processes;

/// The deployment's claims, shared by every group of the deployment.
pub(crate) type StorageClaims = Arc<Claims>;

pub(crate) struct Claims {
    /// The storage launch of each node's workers, by host.
    launches: BTreeMap<String, StorageLaunch>,
    /// Where other nodes reach each node's server, by host: empty for a
    /// server no other node reaches.
    endpoints: BTreeMap<String, String>,
    /// What each node's claim states: the scopes the world keeps there.
    scopes: BTreeMap<String, Vec<(String, u32)>>,
    deployment: String,
    world: u64,
    server_arguments: Vec<String>,
    processes: Processes,
}

impl Claims {
    /// Storage Server TCP endpoints of the deployment's nodes, by host.
    pub(crate) fn endpoints(&self) -> &BTreeMap<String, String> {
        &self.endpoints
    }
    /// Claims `arguments`' deployment at the server of every node of the
    /// deployment, keeping exactly the scopes their groups run there.
    ///
    /// Fails when a server cannot be started or reached, or refuses the
    /// claim because another head of the deployment is connected.
    pub(crate) fn claim(
        arguments: &[WorkerProcessArgs],
        processes: &Processes,
    ) -> anyhow::Result<StorageClaims> {
        let Some(first) = arguments.first() else {
            return Ok(Arc::new(Self {
                launches: BTreeMap::new(),
                endpoints: BTreeMap::new(),
                scopes: BTreeMap::new(),
                deployment: String::new(),
                world: 0,
                server_arguments: Vec::new(),
                processes: Arc::clone(processes),
            }));
        };
        let deployment = &first.deployment;
        let hosts = crate::worker::processes::lock(processes)?.hosts();
        // The leader's node runs the engine, whose workers reach their
        // server through its socket.
        let engine_host = hosts.first().cloned();
        let mut scopes: BTreeMap<String, Vec<(String, u32)>> =
            hosts.into_iter().map(|host| (host, Vec::new())).collect();
        for args in arguments {
            anyhow::ensure!(
                args.deployment == *deployment,
                "the groups of one launch belong to one deployment"
            );
            for (rank, placed) in args.ranks.iter().enumerate() {
                scopes
                    .entry(placed.host.clone())
                    .or_default()
                    .push((args.worker_id.clone(), rank as u32));
            }
        }

        // A server the deployment starts runs its Mooncake connector.
        let server_arguments = first
            .kv_storage
            .as_ref()
            .map(storage::connector::Settings::arguments)
            .unwrap_or_default();
        let mut launches = BTreeMap::new();
        let mut endpoints = BTreeMap::new();
        let mut claimed = BTreeMap::new();
        for (host, scopes) in scopes {
            claimed.insert(host.clone(), scopes.clone());
            let launch = crate::worker::processes::lock(processes)?
                .claim(
                    &host,
                    deployment,
                    first.world,
                    scopes,
                    // A world started again at a rollback retires the world on
                    // trial first, whose memory it needs.
                    crate::worker::processes::Retirement {
                        worlds: first.retire.clone(),
                        discard_pages: false,
                    },
                    server_arguments.clone(),
                )
                .with_context(|| format!("claiming deployment {deployment} on host {host}"))?;
            endpoints.insert(host.clone(), launch.tcp.clone().unwrap_or_default());
            // The engine decides whether its workers use the host KV cache,
            // and reaches its own node's server through its socket.
            let tcp = launch
                .tcp
                .clone()
                .filter(|_| Some(&host) != engine_host.as_ref());
            launches.insert(
                host,
                StorageLaunch {
                    host_kv_cache: first.host_kv_cache,
                    tcp,
                    ..launch
                },
            );
        }
        tracing::info!(
            deployment,
            world = first.world,
            hosts = launches.len(),
            "claimed the deployment's storage"
        );
        Ok(Arc::new(Self {
            launches,
            endpoints,
            scopes: claimed,
            deployment: deployment.clone(),
            world: first.world,
            server_arguments,
            processes: Arc::clone(processes),
        }))
    }

    /// Publishes this committed world's immutable pages on every claimed node.
    pub(crate) fn publish(&self) -> anyhow::Result<()> {
        for host in self.scopes.keys() {
            crate::worker::processes::lock(&self.processes)?.publish_worlds(
                host,
                &self.deployment,
                &[self.world],
            )?;
        }
        Ok(())
    }

    /// Ends the memory of every world of `worlds` on each node of the
    /// deployment, claiming the engine's own scopes there again with them
    /// retired: a tensor no other world retains is released. With
    /// `discard_pages` (the engine's world computes other numerics) each
    /// node's page pools discard the pages the engine's world did not
    /// publish, which it never hits.
    ///
    /// # Errors
    ///
    /// Fails when a node's daemon or Storage Server refuses the claim.
    pub(crate) fn retire(&self, worlds: &[u64], discard_pages: bool) -> anyhow::Result<()> {
        if worlds.is_empty() {
            return Ok(());
        }
        for (host, scopes) in &self.scopes {
            crate::worker::processes::lock(&self.processes)?
                .claim(
                    host,
                    &self.deployment,
                    self.world,
                    scopes.clone(),
                    crate::worker::processes::Retirement {
                        worlds: worlds.to_vec(),
                        discard_pages,
                    },
                    self.server_arguments.clone(),
                )
                .with_context(|| {
                    format!(
                        "retiring worlds {worlds:?} of deployment {} on host {host}",
                        self.deployment
                    )
                })?;
        }
        tracing::info!(
            deployment = %self.deployment,
            world = self.world,
            ?worlds,
            discard_pages,
            "retired worlds on every node of the deployment"
        );
        Ok(())
    }

    /// Keeps the weights alone of every world of `worlds` on each node of
    /// the deployment: a previous world kept as its weights, whose processes
    /// exited, keeps nothing else, so each GPU's ledger leaves the rest to
    /// this world. With `to_host` the weights then move to host memory,
    /// which leaves this world their device memory too; a node that cannot
    /// move them keeps them on its GPUs.
    ///
    /// Returns each node's report as JSON: `{"host", "released",
    /// "moved_bytes", "refused"}`, `refused` saying why a node kept the
    /// weights on its GPUs when `to_host` asked otherwise.
    ///
    /// # Errors
    ///
    /// Fails when a node's daemon or Storage Server refuses.
    pub(crate) fn keep_weights(
        &self,
        worlds: &[u64],
        to_host: bool,
    ) -> anyhow::Result<Vec<String>> {
        if worlds.is_empty() {
            return Ok(Vec::new());
        }
        let mut reports = Vec::with_capacity(self.scopes.len());
        for host in self.scopes.keys() {
            let kept = crate::worker::processes::lock(&self.processes)?.keep_weights(
                host,
                &self.deployment,
                worlds,
                to_host,
            )?;
            if !kept.refused.is_empty() {
                tracing::warn!(
                    host,
                    refused = %kept.refused,
                    "a node keeps the weights on its GPUs"
                );
            }
            reports.push(
                serde_json::json!({
                    "host": host,
                    "released": kept.released,
                    "moved_bytes": kept.moved_bytes,
                    "refused": (!kept.refused.is_empty()).then_some(kept.refused),
                })
                .to_string(),
            );
        }
        tracing::info!(
            deployment = %self.deployment,
            ?worlds,
            to_host,
            "worlds keep their weights alone on every node of the deployment"
        );
        Ok(reports)
    }

    /// Halves, or restores when not `halve`, the byte budgets of the links
    /// the deployment's transfers use on every node of the deployment as
    /// the leader knows it now, and returns their hosts; a node that holds
    /// no claim of the deployment has none to throttle.
    ///
    /// # Errors
    ///
    /// Fails when a node's daemon cannot reach its Transfer Server.
    pub(crate) fn throttle(&self, halve: bool) -> anyhow::Result<Vec<String>> {
        // The copies of a world change run on the Transfer Servers of the
        // nodes its successor runs on, which may have joined after this
        // engine started.
        let mut processes = crate::worker::processes::lock(&self.processes)?;
        let hosts = processes.current_hosts()?;
        for host in &hosts {
            processes.throttle(host, &self.deployment, halve)?;
        }
        Ok(hosts)
    }

    /// The memory ledgers of every node of the deployment: for each host, the
    /// JSON of its Storage Server's ledgers of the deployment and what it
    /// admits of new host memory, read by the node's daemon over its claim
    /// (no worker is asked, so serving goes on).
    ///
    /// # Errors
    ///
    /// Fails when a node's daemon cannot read them.
    pub(crate) fn ledgers(&self) -> anyhow::Result<Vec<(String, String)>> {
        let mut ledgers = Vec::with_capacity(self.scopes.len());
        for host in self.scopes.keys() {
            let ledger =
                crate::worker::processes::lock(&self.processes)?.ledger(host, &self.deployment)?;
            ledgers.push((host.clone(), ledger));
        }
        Ok(ledgers)
    }

    /// The storage launch of the workers on `host`, which names the
    /// servers of the other nodes that other nodes reach.
    pub(crate) fn launch(&self, host: &str) -> anyhow::Result<StorageLaunch> {
        let launch = self
            .launches
            .get(host)
            .with_context(|| format!("the deployment holds no storage claim on host {host}"))?;
        let peers = self
            .endpoints
            .iter()
            .filter(|(other, endpoint)| other.as_str() != host && !endpoint.is_empty())
            .map(|(_, endpoint)| endpoint.clone())
            .collect();
        Ok(StorageLaunch {
            peers,
            ..launch.clone()
        })
    }
}
