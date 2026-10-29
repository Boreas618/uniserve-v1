//! The deployment's claim at this node's Storage Server.
//!
//! An engine asks every node's daemon to claim the deployment before a worker
//! starts there (`Claim`): the daemon starts the node's Storage Server when
//! it has none, claims the scopes the engine's world places on the node, and
//! holds the claiming connection for as long as it runs. The claim therefore
//! outlives every engine and every world: an engine that succeeds another
//! claims the scopes of its own world under the same claim, and its workers
//! hold the same token as the running world's, so neither world's storage is
//! touched by the other's claim. A claim names the worlds whose storage on
//! the node ends (`retire`): the world a commit superseded, or the world a
//! rollback discarded.
//!
//! A previous world kept in host storage on trial (`KeepWeights` with
//! `to_host`) has its weights moved there by the server's Transfer Server
//! (`Request::Offload`). Every claim of a world returns that world's weights
//! to the GPUs they left before it is answered (`Request::Restore`), after
//! the worlds it retires released their storage: a world started again at a
//! rollback claims, and its workers then map its weights where they were.
//!
//! A server of an earlier protocol than this build's is restarted with this
//! daemon's executable before the deployment is claimed, or taken over by a
//! hot restart ([`restart_earlier`]): the successor goes on with every
//! connection and every allocation it holds.

use std::collections::{BTreeMap, HashMap};
use std::os::fd::OwnedFd;
use std::time::Duration;

use proto::process::{Claim, Claimed, WeightsKept};

/// Longest a Storage Server this daemon starts takes to listen.
const START_TIMEOUT: Duration = Duration::from_secs(30);

/// Longest the successor of a Storage Server restart takes to serve.
const RESTART_TIMEOUT: Duration = Duration::from_secs(120);

/// How often a restart checks whether the successor serves.
const RESTART_POLL: Duration = Duration::from_millis(50);

/// One deployment's claim on this node.
struct Held {
    /// The connection that holds the claim; the claim ends with it.
    client: storage::Client,
    /// The scopes each world claimed, sorted.
    worlds: BTreeMap<u64, Vec<(String, u32)>>,
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

    /// Claims `claim`'s deployment for its world at this node's Storage
    /// Server, starting the server when the node has none and restarting it
    /// when it speaks an earlier protocol, or answers the claim held when the
    /// world claimed the same scopes and retires nothing.
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
            if claim.retire.is_empty() && held.worlds.get(&claim.world) == Some(&scopes) {
                restore(&mut held.client, &claim.deployment, claim.world)?;
                return Ok(held.claimed.clone());
            }
            held.client
                .claim(
                    claim.world,
                    as_scopes(&claim.deployment),
                    claim.retire.clone(),
                )
                .with_context(|| format!("claiming deployment {}", claim.deployment))?;
            for world in &claim.retire {
                held.worlds.remove(world);
            }
            restore(&mut held.client, &claim.deployment, claim.world)?;
            held.worlds.insert(claim.world, scopes);
            tracing::info!(
                deployment = %claim.deployment,
                world = claim.world,
                retired = ?claim.retire,
                "claimed the deployment's storage for a world"
            );
            return Ok(held.claimed.clone());
        }

        let socket = storage::socket::default_socket();
        let mut arguments = claim.server_arguments.clone();
        if self.remote_pages {
            arguments.extend(["--tcp".to_owned(), "0.0.0.0:0".to_owned()]);
        }
        storage::socket::ensure(&self.executable, &socket, START_TIMEOUT, &arguments)
            .context("starting this node's storage server")?;
        restart_earlier(&self.executable, &socket)?;
        let mut client = storage::Client::connect(
            &socket,
            storage::Role::Head {
                deployment: claim.deployment.clone(),
                // The claim holds every world; it reads and publishes no
                // page.
                world: 0,
            },
        )
        .with_context(|| format!("connecting to the storage server at {}", socket.display()))?;
        let token = client
            .claim(
                claim.world,
                as_scopes(&claim.deployment),
                claim.retire.clone(),
            )
            .with_context(|| format!("claiming deployment {}", claim.deployment))?;
        restore(&mut client, &claim.deployment, claim.world)?;
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
        tracing::info!(deployment = %claim.deployment, world = claim.world, scopes = scopes.len(), "claimed the deployment's storage");
        self.held.insert(
            claim.deployment.clone(),
            Held {
                client,
                worlds: BTreeMap::from([(claim.world, scopes)]),
                claimed: claimed.clone(),
            },
        );
        Ok(claimed)
    }

    /// Publishes committed worlds through this node's deployment claim.
    pub(crate) fn publish_worlds(
        &mut self,
        deployment: &str,
        worlds: &[u64],
    ) -> anyhow::Result<()> {
        use anyhow::Context as _;

        self.held
            .get_mut(deployment)
            .with_context(|| format!("this node holds no claim of deployment {deployment}"))?
            .client
            .publish_worlds(worlds.to_vec())?;
        Ok(())
    }

    /// Has this node's Storage Server keep the weights alone of `worlds` of
    /// `deployment` (`Request::KeepWeights`), over the claim this daemon
    /// holds, and with `to_host` move them to host storage
    /// (`Request::Offload`). Returns how many storages it released, the
    /// bytes it moved, and why it kept the weights on the GPUs when the
    /// server refused or failed the move: the weights then stay where they
    /// are, as a previous world kept as its weights keeps them.
    ///
    /// # Errors
    ///
    /// Fails when this daemon holds no claim of the deployment, or the
    /// server cannot be reached or refuses to keep the weights alone.
    pub(crate) fn keep_weights(
        &mut self,
        deployment: &str,
        worlds: &[u64],
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
                worlds.to_vec(),
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
                format!("retaining weights and rollback cache of worlds {worlds:?}")
            })?;
        tracing::info!(
            deployment,
            ?worlds,
            released,
            "worlds keep their weights alone"
        );
        let mut kept = WeightsKept {
            released,
            ..WeightsKept::default()
        };
        if !to_host {
            return Ok(kept);
        }
        for &world in worlds {
            match held.client.move_weights(world, true) {
                Ok((storages, bytes)) => {
                    kept.moved_bytes += bytes;
                    tracing::info!(
                        deployment,
                        world,
                        storages,
                        bytes,
                        "a world's weights moved to host storage"
                    );
                }
                // A world that cannot is kept as its weights on the GPUs.
                Err(error) => {
                    tracing::warn!(deployment, world, %error, "a world's weights stay on the GPUs");
                    kept.refused = format!("world {world}: {error}");
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
        // transfers: no world of it has claimed the node yet.
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
    /// `deployment`, one per GPU of the node it uses, and what the host
    /// admits of new host storage, as JSON (`{"ledgers": [...],
    /// "host_storage": {...}}`), read over the claim this daemon holds.
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
        let (ledgers, host) = held
            .client
            .ledger(Vec::new())
            .with_context(|| format!("reading the storage ledgers of deployment {deployment}"))?;
        Ok(serde_json::json!({ "ledgers": ledgers, "host_storage": host }).to_string())
    }
}

/// Restarts the server at `socket` with `executable` (this build's) when it
/// speaks an earlier protocol than this build, which needs this build's,
/// and waits until the successor serves: it goes on with every connection
/// and allocation, so no deployment of the host notices. A daemon of this
/// build calls it before it claims a deployment, and before it takes a
/// deployment over from a daemon of an earlier build, so that every update
/// replaces the Storage Server before the daemon and the daemon before the
/// world (`specs/online-operations.md` §4.4).
pub(crate) fn restart_earlier(
    executable: &std::path::Path,
    socket: &std::path::Path,
) -> anyhow::Result<()> {
    use anyhow::Context as _;

    let mut operator = storage::Client::connect_operator(socket)
        .with_context(|| format!("connecting to the storage server at {}", socket.display()))?;
    if operator.version() >= storage::protocol::VERSION {
        return Ok(());
    }
    let earlier = operator.version();
    let pid = operator
        .restart(Some(executable.to_string_lossy().into_owned()))
        .context("restarting a storage server of an earlier protocol")?;
    tracing::info!(
        earlier,
        version = storage::protocol::VERSION,
        pid,
        "restarting the storage server, which speaks an earlier protocol"
    );
    drop(operator);
    let deadline = std::time::Instant::now() + RESTART_TIMEOUT;
    loop {
        if let Ok(client) = storage::Client::connect_operator(socket)
            && client.version() >= storage::protocol::VERSION
        {
            return Ok(());
        }
        anyhow::ensure!(
            std::time::Instant::now() < deadline,
            "the storage server did not serve protocol {} within {RESTART_TIMEOUT:?} of its \
                 restart",
            storage::protocol::VERSION
        );
        std::thread::sleep(RESTART_POLL);
    }
}

/// Returns `world`'s weights that an offload left in host storage to the
/// GPUs they left (`Request::Restore`), which answers at once when there
/// are none.
fn restore(client: &mut storage::Client, deployment: &str, world: u64) -> anyhow::Result<()> {
    use anyhow::Context as _;

    let (storages, bytes) = client
        .move_weights(world, false)
        .with_context(|| format!("returning the weights of world {world} to its GPUs"))?;
    if storages > 0 {
        tracing::info!(
            deployment,
            world,
            storages,
            bytes,
            "a world's weights returned from host storage"
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
    /// The scopes of a predecessor that knew no worlds, which world 0 holds.
    #[serde(default)]
    scopes: Vec<(String, u32)>,
    #[serde(default)]
    worlds: BTreeMap<u64, Vec<(String, u32)>>,
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
                worlds: held.worlds,
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
                    worlds: if claim.scopes.is_empty() {
                        claim.worlds
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
