//! The deployment's claim at this node's Storage Server.
//!
//! An engine asks every node's daemon to claim the deployment before a worker
//! starts there (`Claim`): the daemon starts the node's Storage Server when
//! it has none, claims the scopes the engine's epoch places on the node, and
//! holds the claiming connection for as long as it runs. The claim therefore
//! outlives every engine and every epoch: an engine that succeeds another
//! claims the scopes of its own epoch under the same claim, and its workers
//! hold the same token as the running epoch's, so neither epoch's storage is
//! touched by the other's claim. A claim names the epochs whose storage on
//! the node ends (`retire`): the epoch a commit superseded, or the epoch a
//! rollback discarded.
//!
//! A previous epoch kept in host storage on trial (`KeepWeights` with
//! `to_host`) has its weights moved there by the server's Transfer Server
//! (`Request::Offload`). Every claim of an epoch returns that epoch's weights
//! to the GPUs they left before it is answered (`Request::Restore`), after
//! the epochs it retires released their storage: an epoch started again at a
//! rollback claims, and its workers then map its weights where they were.

use std::collections::{BTreeMap, HashMap};
use std::os::fd::OwnedFd;
use std::time::Duration;

use proto::process::{Claim, Claimed, WeightsKept};

/// Longest a Storage Server this daemon starts takes to listen.
const START_TIMEOUT: Duration = Duration::from_secs(30);

/// One deployment's claim on this node.
struct Held {
    /// The connection that holds the claim; the claim ends with it.
    client: storage::Client,
    /// The scopes each epoch claimed, sorted.
    epochs: BTreeMap<u64, Vec<(String, u32)>>,
    claimed: Claimed,
}

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
    held: HashMap<String, Held>,
}

impl ClaimHolder {
    pub(crate) fn new(
        executable: std::path::PathBuf,
        remote_pages: bool,
        address: Option<std::net::IpAddr>,
    ) -> Self {
        Self {
            executable,
            remote_pages,
            address,
            held: HashMap::new(),
        }
    }

    /// Claims `claim`'s deployment for its epoch at this node's Storage
    /// Server, starting the server when the node has none and restarting it
    /// when it speaks an earlier protocol, or answers the claim held when the
    /// epoch claimed the same scopes and retires nothing.
    ///
    /// # Errors
    ///
    /// Fails when the server cannot be started, restarted or reached, or
    /// refuses the claim because another head of the deployment holds it.
    pub(crate) fn claim(&mut self, claim: &Claim) -> anyhow::Result<Claimed> {
        use anyhow::Context as _;

        let mut scopes: Vec<(String, u32)> = claim
            .scopes
            .iter()
            .map(|scope| (scope.group.clone(), scope.rank))
            .collect();
        scopes.sort();
        let as_scopes = |deployment: &str| {
            scopes
                .iter()
                .map(|(group, rank)| storage::Scope {
                    deployment: deployment.to_owned(),
                    group: group.clone(),
                    rank: *rank,
                })
                .collect::<Vec<_>>()
        };
        if let Some(held) = self.held.get_mut(&claim.deployment) {
            if claim.retire.is_empty() && held.epochs.get(&claim.epoch) == Some(&scopes) {
                restore(&mut held.client, &claim.deployment, claim.epoch)?;
                return Ok(held.claimed.clone());
            }
            held.client
                .claim(
                    claim.epoch,
                    as_scopes(&claim.deployment),
                    claim.retire.clone(),
                )
                .with_context(|| format!("claiming deployment {}", claim.deployment))?;
            for epoch in &claim.retire {
                held.epochs.remove(epoch);
            }
            restore(&mut held.client, &claim.deployment, claim.epoch)?;
            held.epochs.insert(claim.epoch, scopes);
            tracing::info!(
                deployment = %claim.deployment,
                epoch = claim.epoch,
                retired = ?claim.retire,
                "claimed the deployment's storage for an epoch"
            );
            return Ok(held.claimed.clone());
        }

        let socket = storage::default_socket();
        let mut arguments = claim.server_arguments.clone();
        if self.remote_pages {
            arguments.extend(["--tcp".to_owned(), "0.0.0.0:0".to_owned()]);
        }
        service::ensure(&self.executable, "storage-server", &socket, &arguments, START_TIMEOUT)
            .context("starting this node's storage server")?;
        let mut client = storage::Client::connect(
            &socket,
            storage::Role::Head {
                deployment: claim.deployment.clone(),
                // The claim holds every epoch; it reads and publishes no
                // page.
                epoch: 0,
            },
        )
        .with_context(|| format!("connecting to the storage server at {}", socket.display()))?;
        let token = client
            .claim(
                claim.epoch,
                as_scopes(&claim.deployment),
                claim.retire.clone(),
            )
            .with_context(|| format!("claiming deployment {}", claim.deployment))?;
        restore(&mut client, &claim.deployment, claim.epoch)?;
        // A server another process started may accept no other node.
        let tcp = match client
            .call(&storage::protocol::Request::Endpoint, &[])?
            .reply
        {
            storage::protocol::Reply::Endpoint { tcp: Some(bound) } if self.remote_pages => bound
                .rsplit_once(':')
                .and_then(|(_, port)| port.parse::<u16>().ok())
                .zip(self.address)
                .map(|(port, address)| std::net::SocketAddr::new(address, port).to_string())
                .unwrap_or_default(),
            _ => String::new(),
        };
        let claimed = Claimed {
            socket: socket.to_string_lossy().into_owned(),
            claim: token.0.to_vec(),
            tcp,
        };
        tracing::info!(deployment = %claim.deployment, epoch = claim.epoch, scopes = scopes.len(), "claimed the deployment's storage");
        self.held.insert(
            claim.deployment.clone(),
            Held {
                client,
                epochs: BTreeMap::from([(claim.epoch, scopes)]),
                claimed: claimed.clone(),
            },
        );
        Ok(claimed)
    }

    /// Publishes committed epochs through this node's deployment claim.
    pub(crate) fn publish_epochs(
        &mut self,
        deployment: &str,
        epochs: &[u64],
    ) -> anyhow::Result<()> {
        use anyhow::Context as _;

        self.held
            .get_mut(deployment)
            .with_context(|| format!("this node holds no claim of deployment {deployment}"))?
            .client
            .publish_epochs(epochs.to_vec())?;
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

        let held = self
            .held
            .get_mut(deployment)
            .with_context(|| format!("this node holds no claim of deployment {deployment}"))?;
        let released = held
            .client
            .keep_weights(
                epochs.to_vec(),
                cache_keys
                    .iter()
                    .map(|key| {
                        key.as_slice()
                            .try_into()
                            .map(storage::Key)
                            .map_err(|_| anyhow::anyhow!("a retained cache key has 32 bytes"))
                    })
                    .collect::<anyhow::Result<_>>()?,
            )
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
            match held.client.move_weights(epoch, true) {
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
    /// `deployment`'s transfers use at this node's Transfer Server, over the
    /// claim this daemon holds: a halving takes every link a task of the
    /// deployment copies over now, a restoration every link a halving
    /// reduced (`Request::Throttle` with no links).
    ///
    /// # Errors
    ///
    /// Fails when this daemon holds no claim of the deployment, or the
    /// server cannot be reached or runs no Transfer Server.
    pub(crate) fn throttle(&mut self, deployment: &str, halve: bool) -> anyhow::Result<()> {
        use anyhow::Context as _;

        // A node that holds no claim of the deployment runs none of its
        // transfers: no epoch of it has claimed the node yet.
        let Some(held) = self.held.get_mut(deployment) else {
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
        held.client
            .throttle(Vec::new(), throttle)
            .with_context(|| format!("throttling the transfers of deployment {deployment}"))?;
        tracing::info!(
            deployment,
            ?throttle,
            "changed the budgets of the transfers' links"
        );
        Ok(())
    }

    /// The storage ledgers this node's Storage Server keeps for
    /// `deployment`, one per GPU of the node it uses, what the host admits
    /// of new host storage, and what the deployment's transfer tasks copied
    /// on the host by priority, as JSON (`{"ledgers": [...],
    /// "host_storage": {...}, "transfers": [...]}`), read over the claim
    /// this daemon holds.
    ///
    /// # Errors
    ///
    /// Fails when this daemon holds no claim of the deployment, or the
    /// server cannot be reached.
    pub(crate) fn ledger(&mut self, deployment: &str) -> anyhow::Result<String> {
        use anyhow::Context as _;

        let held = self
            .held
            .get_mut(deployment)
            .with_context(|| format!("this node holds no claim of deployment {deployment}"))?;
        let read = held
            .client
            .ledger(Vec::new())
            .with_context(|| format!("reading the storage ledgers of deployment {deployment}"))?;
        Ok(serde_json::json!({
            "ledgers": read.ledgers,
            "host_storage": read.host,
            "transfers": read.transfers,
        })
        .to_string())
    }
}

/// Returns `epoch`'s weights that an offload left in host storage to the
/// GPUs they left (`Request::Restore`), which answers at once when there
/// are none.
fn restore(client: &mut storage::Client, deployment: &str, epoch: u64) -> anyhow::Result<()> {
    use anyhow::Context as _;

    let (storages, bytes) = client
        .move_weights(epoch, false)
        .with_context(|| format!("returning the weights of epoch {epoch} to its GPUs"))?;
    if storages > 0 {
        tracing::info!(
            deployment,
            epoch,
            storages,
            bytes,
            "an epoch's weights returned from host storage"
        );
    }
    Ok(())
}

/// A node's claims as a daemon hands them to its successor in a hot
/// restart: each claiming connection crosses as a descriptor, so the
/// Storage Server sees the same head throughout.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct HandedClaims {
    executable: std::path::PathBuf,
    remote_pages: bool,
    claims: Vec<HandedClaim>,
}

/// One handed-over claim.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct HandedClaim {
    deployment: String,
    /// The scopes of a predecessor that knew no epochs, which epoch 0 holds.
    #[serde(default)]
    scopes: Vec<(String, u32)>,
    #[serde(default)]
    epochs: BTreeMap<u64, Vec<(String, u32)>>,
    socket: String,
    claim: Vec<u8>,
    tcp: String,
    /// The connection's next request identity.
    next: u64,
}

impl ClaimHolder {
    /// Hands every claim to this daemon's successor ([`ClaimHolder::adopt`]):
    /// their state, and each claiming connection's descriptor in
    /// `descriptors`, in the same order. This daemon holds no claim
    /// afterwards.
    ///
    /// # Errors
    ///
    /// Fails when a claiming connection has unread messages.
    pub(crate) fn hand_over(&mut self) -> anyhow::Result<(HandedClaims, Vec<OwnedFd>)> {
        let mut claims = Vec::with_capacity(self.held.len());
        let mut descriptors = Vec::with_capacity(self.held.len());
        for (deployment, held) in self.held.drain() {
            let (descriptor, next) = held.client.hand_over()?;
            descriptors.push(descriptor);
            claims.push(HandedClaim {
                deployment,
                scopes: Vec::new(),
                epochs: held.epochs,
                socket: held.claimed.socket,
                claim: held.claimed.claim,
                tcp: held.claimed.tcp,
                next,
            });
        }
        Ok((
            HandedClaims {
                executable: self.executable.clone(),
                remote_pages: self.remote_pages,
                claims,
            },
            descriptors,
        ))
    }

    /// Goes on with the claims a predecessor handed over, with their
    /// connections' descriptors in its order; `address` is this node's.
    pub(crate) fn adopt(
        address: Option<std::net::IpAddr>,
        handed: HandedClaims,
        descriptors: Vec<OwnedFd>,
    ) -> Self {
        let mut holder = Self::new(handed.executable, handed.remote_pages, address);
        for (claim, descriptor) in handed.claims.into_iter().zip(descriptors) {
            holder.held.insert(
                claim.deployment,
                Held {
                    client: storage::Client::resume(descriptor, claim.next),
                    epochs: if claim.scopes.is_empty() {
                        claim.epochs
                    } else {
                        BTreeMap::from([(0, claim.scopes)])
                    },
                    claimed: Claimed {
                        socket: claim.socket,
                        claim: claim.claim,
                        tcp: claim.tcp,
                    },
                },
            );
        }
        holder
    }
}
