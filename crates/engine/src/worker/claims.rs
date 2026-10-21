//! The deployment's claims at the Storage Servers of its hosts.
//!
//! Before any worker starts, the head claims the deployment at the Storage
//! Server of every host the deployment places a worker on: its own host's,
//! which it starts when the host has none, and through each other host's
//! launcher that host's. A claim names the scopes the deployment keeps there
//! and holds for as long as the claiming connection does, so it ends with
//! the head, or with the launcher that holds it for the head. Each worker
//! starts with its host's socket and claim token ([`StorageLaunch`]), by
//! which the server tells it belongs to the current head.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Context;
use ipc::StorageLaunch;
use ipc::launcher::StorageClaim;
use storage::{Client, Role, Scope};

use crate::worker::WorkerProcessArgs;
use crate::worker::launcher::Launchers;

/// Longest a Storage Server the head starts takes to listen.
const START_TIMEOUT: Duration = Duration::from_secs(30);

/// The deployment's claims, shared by every group of the deployment.
pub(crate) type StorageClaims = Arc<Claims>;

pub(crate) struct Claims {
    /// The connection that holds the claim at this host's server, when the
    /// deployment places a worker here.
    _local: Option<Mutex<Client>>,
    /// The storage launch of each host's workers.
    launches: BTreeMap<String, StorageLaunch>,
}

impl Claims {
    /// Claim `arguments`' deployment at the server of every host their
    /// groups place a worker on, keeping exactly the scopes they place
    /// there. `launchers` reaches the hosts other than this one.
    ///
    /// Fails when a server cannot be started or reached, or refuses the
    /// claim because the deployment's previous head is still connected.
    pub(crate) fn claim(
        arguments: &[WorkerProcessArgs],
        launchers: Option<&Launchers>,
    ) -> anyhow::Result<StorageClaims> {
        let Some(first) = arguments.first() else {
            return Ok(Arc::new(Self {
                _local: None,
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
            .storage_connector
            .as_ref()
            .map(storage::connector::Settings::arguments)
            .unwrap_or_default();
        let mut local = None;
        let mut launches = BTreeMap::new();
        for (host, scopes) in scopes {
            let launch = if host == first.host {
                let socket = storage::socket::default_socket();
                storage::socket::ensure(
                    &first.executable,
                    &socket,
                    START_TIMEOUT,
                    &server_arguments,
                )
                .context("starting this host's storage server")?;
                let mut client = Client::connect(
                    &socket,
                    Role::Head {
                        deployment: deployment.clone(),
                    },
                )
                .with_context(|| {
                    format!("connecting to the storage server at {}", socket.display())
                })?;
                let claim = client
                    .claim(
                        scopes
                            .into_iter()
                            .map(|(group, rank)| Scope {
                                deployment: deployment.clone(),
                                group,
                                rank,
                            })
                            .collect(),
                    )
                    .with_context(|| format!("claiming deployment {deployment}"))?;
                local = Some(Mutex::new(client));
                StorageLaunch {
                    socket: socket.to_string_lossy().into_owned(),
                    deployment: deployment.clone(),
                    claim: claim.0,
                    page_tier: first.page_tier,
                    tcp: None,
                    standby: false,
                }
            } else {
                let registry = launchers.with_context(|| {
                    format!("host {host} places workers but no launcher registry was bound")
                })?;
                crate::worker::launcher::lock(registry)?.claim_on(
                    host,
                    StorageClaim {
                        deployment: deployment.clone(),
                        scopes,
                        arguments: server_arguments.clone(),
                    },
                )?
            };
            // The head decides whether its workers use the host page tier.
            launches.insert(
                host.to_owned(),
                StorageLaunch {
                    page_tier: first.page_tier,
                    ..launch
                },
            );
        }
        tracing::info!(
            deployment,
            hosts = launches.len(),
            "claimed the deployment's storage"
        );
        Ok(Arc::new(Self {
            _local: local,
            launches,
        }))
    }

    /// The storage launch of the workers on `host`.
    pub(crate) fn launch(&self, host: &str) -> anyhow::Result<StorageLaunch> {
        self.launches
            .get(host)
            .cloned()
            .with_context(|| format!("the deployment holds no storage claim on host {host}"))
    }
}
