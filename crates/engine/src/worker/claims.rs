//! The deployment's claims at the Storage Servers of its nodes.
//!
//! Before any worker starts, the engine has the daemon of every node the
//! deployment places a worker on claim the deployment at that node's Storage
//! Server (`ProcessClient::claim`), naming the scopes the deployment keeps
//! there. The daemon holds the claim for as long as it runs, so the claim
//! outlives the engine. Each worker starts with its node's socket and claim
//! token ([`StorageLaunch`]), by which the server tells it belongs to the
//! deployment's current claim.

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
        for (host, scopes) in scopes {
            let launch = crate::worker::processes::lock(processes)?
                .claim(host, deployment, scopes, server_arguments.clone())
                .with_context(|| format!("claiming deployment {deployment} on host {host}"))?;
            // The engine decides whether its workers use the host KV cache.
            launches.insert(
                host.to_owned(),
                StorageLaunch {
                    host_kv_cache: first.host_kv_cache,
                    host_kv_cache_bytes: first.host_kv_cache_bytes,
                    ..launch
                },
            );
        }
        tracing::info!(
            deployment,
            hosts = launches.len(),
            "claimed the deployment's storage"
        );
        Ok(Arc::new(Self { launches }))
    }

    /// The storage launch of the workers on `host`.
    pub(crate) fn launch(&self, host: &str) -> anyhow::Result<StorageLaunch> {
        self.launches
            .get(host)
            .cloned()
            .with_context(|| format!("the deployment holds no storage claim on host {host}"))
    }
}
