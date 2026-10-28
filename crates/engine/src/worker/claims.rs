//! The deployment's claims at the Storage Servers of its nodes.
//!
//! Before any worker starts, the engine has the daemon of every node the
//! deployment places a worker on claim the deployment at that node's Storage
//! Server (`ProcessClient::claim`), naming the scopes the engine's world
//! keeps there. The daemon holds the claim for as long as it runs, so the
//! claim outlives the engine. Each worker starts with its node's socket and
//! claim token ([`StorageLaunch`]), by which the server tells it belongs to
//! the deployment's current claim. The engine ends another world's memory
//! on its nodes by claiming its own scopes again with that world retired
//! ([`Claims::retire`]): the predecessor its commit supersedes, or the
//! successor a rollback discards.

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
    /// What each node's claim states: the scopes the world keeps there.
    scopes: BTreeMap<String, Vec<(String, u32)>>,
    deployment: String,
    world: u64,
    server_arguments: Vec<String>,
    processes: Processes,
}

impl Claims {
    /// Claims `arguments`' deployment at the server of every node their
    /// groups place a worker on, keeping exactly the scopes they place
    /// there.
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
                scopes: BTreeMap::new(),
                deployment: String::new(),
                world: 0,
                server_arguments: Vec::new(),
                processes: Arc::clone(processes),
            }));
        };
        let deployment = &first.deployment;
        let mut scopes: BTreeMap<&str, Vec<(String, u32)>> = BTreeMap::new();
        for args in arguments {
            anyhow::ensure!(
                args.deployment == *deployment,
                "the groups of one launch belong to one deployment"
            );
            for (rank, placed) in args.ranks.iter().enumerate() {
                scopes
                    .entry(placed.host.as_str())
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
        let mut claimed = BTreeMap::new();
        for (host, scopes) in scopes {
            claimed.insert(host.to_owned(), scopes.clone());
            let launch = crate::worker::processes::lock(processes)?
                .claim(
                    host,
                    deployment,
                    first.world,
                    scopes,
                    // A world started again at a rollback retires the world on
                    // trial first, whose memory it needs.
                    first.retire.clone(),
                    server_arguments.clone(),
                )
                .with_context(|| format!("claiming deployment {deployment} on host {host}"))?;
            // The engine decides whether its workers use the host KV cache.
            launches.insert(
                host.to_owned(),
                StorageLaunch {
                    host_kv_cache: first.host_kv_cache,
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
            scopes: claimed,
            deployment: deployment.clone(),
            world: first.world,
            server_arguments,
            processes: Arc::clone(processes),
        }))
    }

    /// Ends the memory of every world of `worlds` on each node of the
    /// engine's, claiming the engine's own scopes there again with them
    /// retired: a tensor no other world retains is released.
    ///
    /// # Errors
    ///
    /// Fails when a node's daemon or Storage Server refuses the claim.
    pub(crate) fn retire(&self, worlds: &[u64]) -> anyhow::Result<()> {
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
                    worlds.to_vec(),
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
            "retired worlds on every node of the engine"
        );
        Ok(())
    }

    /// Keeps the weights alone of every world of `worlds` on each node of
    /// the engine's: a previous world kept as its weights, whose processes
    /// exited, keeps nothing else, so each GPU's ledger leaves the rest to
    /// this world.
    ///
    /// # Errors
    ///
    /// Fails when a node's daemon or Storage Server refuses.
    pub(crate) fn keep_weights(&self, worlds: &[u64]) -> anyhow::Result<()> {
        if worlds.is_empty() {
            return Ok(());
        }
        for host in self.scopes.keys() {
            crate::worker::processes::lock(&self.processes)?.keep_weights(
                host,
                &self.deployment,
                worlds,
            )?;
        }
        tracing::info!(
            deployment = %self.deployment,
            ?worlds,
            "worlds keep their weights alone on every node of the engine"
        );
        Ok(())
    }

    /// The memory ledgers of every node of the engine's: for each host, the
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

    /// The storage launch of the workers on `host`.
    pub(crate) fn launch(&self, host: &str) -> anyhow::Result<StorageLaunch> {
        self.launches
            .get(host)
            .cloned()
            .with_context(|| format!("the deployment holds no storage claim on host {host}"))
    }
}
