//! The deployment's claim at this node's Storage Server.
//!
//! An engine asks every node's daemon to claim the deployment before a worker
//! starts there (`Claim`): the daemon starts the node's Storage Server when
//! it has none, claims the scopes the engine's epoch places on the node, and
//! holds the claim's client (`storage::ClaimClient`) for as long as it runs.
//! The claim therefore outlives every engine and every epoch: an engine that
//! succeeds another claims the scopes of its own epoch under the same claim,
//! and its workers hold the same token as the running epoch's, so neither
//! epoch's storage is touched by the other's claim. A claim names the epochs
//! whose storage on the node ends (`retire`): the epoch a commit superseded,
//! or the epoch a rollback discarded.
//!
//! A previous epoch kept in host storage on trial (`KeepWeights` with
//! `to_host`) has its weights moved there by the server's Transfer Server
//! (`Request::Offload`). Every claim of an epoch returns that epoch's weights
//! to the GPUs they left before it gets its response (`Request::Restore`), after
//! the epochs it retires released their storage: an epoch started again at a
//! rollback claims, and its workers then map its weights where they were.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::os::fd::OwnedFd;
use std::time::Duration;

use proto::process::{Claim, Claimed, WeightsKept};

/// Longest a Storage Server this daemon starts takes to serve.
const START_TIMEOUT: Duration = Duration::from_secs(30);

/// The claims this node's daemon holds, by deployment.
pub(crate) struct ClaimHolder {
    /// The `uniserve` executable a Storage Server this daemon starts runs as.
    executable: std::path::PathBuf,
    /// Whether processes on other nodes reach this node's server (their
    /// engine's page requests, their Transfer Servers' reads of its
    /// storage): the server it starts then also listens on TCP.
    remote_pages: bool,
    /// This node's address other nodes reach, which names that listener.
    address: Option<std::net::IpAddr>,
    /// The settings of the node's Storage Server the deployment states
    /// (`storage_config` and its Mooncake connector, `kv_cache_config.mooncake`),
    /// which a claim that states none of its own takes ([`Self::settings`]).
    settings: storage::Settings,
    /// Each deployment's claim, with the `host:port` other nodes reach the
    /// server at (empty when they cannot), as its engines are told.
    claims: HashMap<String, (storage::ClaimClient, String)>,
}

/// A node's claims as they cross to the daemon's successor in a restart in
/// place: each claim's state and the endpoint its engines were told. The
/// claiming connections' descriptors travel beside it, in the same order.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct Handoff {
    executable: std::path::PathBuf,
    remote_pages: bool,
    settings: storage::Settings,
    claims: Vec<(storage::ClaimHandoff, String)>,
}

impl ClaimHolder {
    pub(crate) fn new(
        executable: std::path::PathBuf,
        remote_pages: bool,
        address: Option<std::net::IpAddr>,
        settings: storage::Settings,
    ) -> Self {
        Self {
            executable,
            remote_pages,
            address,
            settings,
            claims: HashMap::new(),
        }
    }

    /// The settings of the node's Storage Server the deployment states, which the leader gives
    /// every claim that states none, so that the Storage Server a node starts for it serves with
    /// them.
    pub(crate) fn settings(&self) -> &storage::Settings {
        &self.settings
    }

    /// Claims `claim`'s deployment for its epoch at this node's Storage
    /// Server, starting the server when the node has none; an epoch that
    /// claimed the same scopes and retires nothing keeps the claim held.
    /// Either way the epoch's weights an offload left in host storage
    /// return to their GPUs first.
    ///
    /// # Errors
    ///
    /// Fails when the server cannot be started or reached, or refuses the
    /// claim because another head of the deployment holds it.
    pub(crate) fn claim(&mut self, claim: &Claim) -> anyhow::Result<Claimed> {
        use anyhow::Context as _;

        let socket =
            storage::default_socket().context("finding this node's Storage Server socket")?;
        let scopes: Vec<storage::Scope> = claim
            .scopes
            .iter()
            .map(|scope| storage::Scope {
                deployment: claim.deployment.clone(),
                group: scope.group.clone(),
                rank: scope.rank,
            })
            .collect();

        let (client, tcp) = match self.claims.entry(claim.deployment.clone()) {
            Entry::Occupied(held) => held.into_mut(),
            Entry::Vacant(vacant) => {
                let settings = match &claim.storage {
                    Some(stated) => storage::Settings::try_from(stated)
                        .map_err(anyhow::Error::msg)
                        .context("reading the claim's Storage Server settings")?,
                    None => self.settings.clone(),
                };
                let document = settings
                    .document()
                    .context("writing the Storage Server's settings")?;
                // The server serves with the settings the deployment states: one this daemon
                // starts reads them, and one that serves already states its own, which must be
                // these.
                let record: storage::Readiness = service::host::start(
                    &self.executable,
                    storage::SERVICE,
                    &socket,
                    &["--settings".to_owned(), "-".to_owned()],
                    Some(document.as_bytes()),
                    START_TIMEOUT,
                )
                .context("starting this node's storage server")?;
                storage::settings::Mismatch::check(&record, &settings)?;

                let client = storage::ClaimClient::connect(&socket, &claim.deployment)
                    .with_context(|| {
                        format!("connecting to the storage server at {}", socket.display())
                    })?;

                // Other nodes reach the server at this node's address, on the port it bound.
                let tcp = match (record.endpoints.peer, self.address) {
                    (Some(peer), Some(address)) if self.remote_pages => {
                        peer.with_host(address).to_string()
                    }
                    _ => String::new(),
                };
                vacant.insert((client, tcp))
            }
        };

        let token = client
            .claim(claim.epoch, scopes, claim.retire.clone())
            .with_context(|| format!("claiming deployment {}", claim.deployment))?;
        restore(client, claim.epoch)?;
        tracing::info!(
            deployment = %claim.deployment,
            epoch = claim.epoch,
            scopes = claim.scopes.len(),
            retired = ?claim.retire,
            "claimed the deployment's storage for an epoch"
        );

        Ok(Claimed {
            socket: socket.to_string_lossy().into_owned(),
            claim: token.0.to_vec(),
            tcp: tcp.clone(),
        })
    }

    /// The claim this node holds of `deployment`.
    fn held(&mut self, deployment: &str) -> anyhow::Result<&mut storage::ClaimClient> {
        use anyhow::Context as _;

        self.claims
            .get_mut(deployment)
            .map(|(client, _)| client)
            .with_context(|| format!("this node holds no claim of deployment {deployment}"))
    }

    /// Publishes committed epochs through this node's deployment claim.
    pub(crate) fn publish_epochs(
        &mut self,
        deployment: &str,
        epochs: &[u64],
    ) -> anyhow::Result<()> {
        self.held(deployment)?.publish(epochs.to_vec())?;
        Ok(())
    }

    /// Has this node's Storage Server keep the weights alone of `epochs` of
    /// `deployment` (`Request::KeepWeights`), over the claim this daemon
    /// holds, and with `to_host` move them to host storage
    /// (`Request::Offload`). Returns how many storages it released, the
    /// bytes it moved, and why it kept the weights on the GPUs when the
    /// server refused or failed the move: the weights then stay where they
    /// are, as a previous epoch kept as its weights keeps them.
    ///
    /// # Errors
    ///
    /// Fails when this daemon holds no claim of the deployment, or the
    /// server cannot be reached or refuses to keep the weights alone.
    pub(crate) fn keep_weights(
        &mut self,
        deployment: &str,
        epochs: &[u64],
        to_host: bool,
        cache_keys: &[Vec<u8>],
    ) -> anyhow::Result<WeightsKept> {
        use anyhow::Context as _;

        let client = self.held(deployment)?;
        let cache_keys = cache_keys
            .iter()
            .map(|key| {
                key.as_slice()
                    .try_into()
                    .map(storage::Key)
                    .map_err(|_| anyhow::anyhow!("a retained cache key has 32 bytes"))
            })
            .collect::<anyhow::Result<_>>()?;
        let released = client
            .keep_weights(epochs.to_vec(), cache_keys)
            .with_context(|| {
                format!("retaining weights and rollback cache of epochs {epochs:?}")
            })?;
        tracing::info!(
            deployment,
            ?epochs,
            released,
            "epochs keep their weights alone"
        );

        let mut kept = WeightsKept {
            released,
            ..WeightsKept::default()
        };
        if !to_host {
            return Ok(kept);
        }
        for &epoch in epochs {
            match client.offload(epoch) {
                Ok((storages, bytes)) => {
                    kept.moved_bytes += bytes;
                    tracing::info!(
                        deployment,
                        epoch,
                        storages,
                        bytes,
                        "an epoch's weights moved to host storage"
                    );
                }
                // An epoch that cannot is kept as its weights on the GPUs.
                Err(error) => {
                    tracing::warn!(deployment, epoch, %error, "an epoch's weights stay on the GPUs");
                    kept.refused = format!("epoch {epoch}: {error}");
                    break;
                }
            }
        }
        Ok(kept)
    }

    /// Halves, or restores when not `halve`, the byte budgets of the links
    /// `deployment`'s tasks use at this node's Transfer Server, over the
    /// claim this daemon holds: a halving takes every link a task of the
    /// deployment copies over now, a restoration every link a halving
    /// reduced (`Request::Throttle` with no links).
    ///
    /// # Errors
    ///
    /// Fails when the server cannot be reached or runs no Transfer Server.
    pub(crate) fn throttle(&mut self, deployment: &str, halve: bool) -> anyhow::Result<()> {
        use anyhow::Context as _;

        // A node that holds no claim of the deployment runs none of its
        // tasks: no epoch of it has claimed the node yet.
        let Some((client, _)) = self.claims.get_mut(deployment) else {
            tracing::debug!(
                deployment,
                "no claim of the deployment: nothing to throttle"
            );
            return Ok(());
        };
        let throttle = if halve {
            storage::protocol::Throttle::Halve
        } else {
            storage::protocol::Throttle::Restore
        };
        client
            .throttle(throttle)
            .with_context(|| format!("throttling the tasks of deployment {deployment}"))?;
        tracing::info!(
            deployment,
            ?throttle,
            "changed the budgets of the tasks' links"
        );
        Ok(())
    }

    /// The quotas this node's Storage Server keeps for `deployment`, one
    /// per GPU of the node it uses, what the host admits of new host
    /// storage, and what the deployment's tasks copied on the host by
    /// priority, as JSON (`{"quotas": [...], "host_storage": {...},
    /// "tasks": [...]}`), read over the claim this daemon holds.
    ///
    /// # Errors
    ///
    /// Fails when this daemon holds no claim of the deployment, or the
    /// server cannot be reached.
    pub(crate) fn usage(&mut self, deployment: &str) -> anyhow::Result<String> {
        use anyhow::Context as _;

        let usage = self
            .held(deployment)?
            .usage()
            .with_context(|| format!("reading the quotas of deployment {deployment}"))?;
        Ok(serde_json::json!({
            "quotas": usage.quotas,
            "host_storage": usage.host,
            "tasks": usage.tasks,
        })
        .to_string())
    }

    /// Hands every claim to this daemon's successor ([`ClaimHolder::adopt`]):
    /// their state, and each claiming connection's descriptor in the second
    /// value, in the same order. This daemon holds no claim afterwards.
    ///
    /// # Errors
    ///
    /// Fails when a claiming connection has unread messages.
    pub(crate) fn handoff(&mut self) -> anyhow::Result<(Handoff, Vec<OwnedFd>)> {
        let mut claims = Vec::with_capacity(self.claims.len());
        let mut descriptors = Vec::with_capacity(self.claims.len());
        for (_, (client, tcp)) in self.claims.drain() {
            let (claim, descriptor) = client.handoff()?;
            claims.push((claim, tcp));
            descriptors.push(descriptor);
        }

        let handoff = Handoff {
            executable: self.executable.clone(),
            remote_pages: self.remote_pages,
            settings: self.settings.clone(),
            claims,
        };
        Ok((handoff, descriptors))
    }

    /// Goes on with the claims a predecessor handed over, with their
    /// connections' descriptors in its order; `address` is this node's.
    pub(crate) fn adopt(
        address: Option<std::net::IpAddr>,
        handoff: Handoff,
        descriptors: Vec<OwnedFd>,
    ) -> storage::Result<Self> {
        let mut holder = Self::new(
            handoff.executable,
            handoff.remote_pages,
            address,
            handoff.settings,
        );
        for ((claim, tcp), descriptor) in handoff.claims.into_iter().zip(descriptors) {
            let deployment = claim.deployment.clone();
            let client = storage::ClaimClient::resume(claim, descriptor)?;
            holder.claims.insert(deployment, (client, tcp));
        }
        Ok(holder)
    }
}

/// Returns `epoch`'s weights that an offload left in host storage to the
/// GPUs they left (`Request::Restore`), which responds at once when there
/// are none.
fn restore(client: &mut storage::ClaimClient, epoch: u64) -> anyhow::Result<()> {
    use anyhow::Context as _;

    let (storages, bytes) = client
        .restore(epoch)
        .with_context(|| format!("returning the weights of epoch {epoch} to its GPUs"))?;
    if storages > 0 {
        tracing::info!(
            deployment = client.deployment(),
            epoch,
            storages,
            bytes,
            "an epoch's weights returned from host storage"
        );
    }
    Ok(())
}
