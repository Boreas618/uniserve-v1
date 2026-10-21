//! The deployment's claim at this node's Storage Server.
//!
//! An engine asks every node's daemon to claim the deployment before a worker
//! starts there (`Claim`): the daemon starts the node's Storage Server when
//! it has none, claims the scopes the deployment places on the node, and
//! holds the claiming connection for as long as it runs. The claim therefore
//! outlives every engine: the engine of a hot update, whose workers stand by
//! under the running engine's claim, asks for the same scopes and is answered
//! with the claim held. A claim of other scopes is a new placement: it
//! replaces the claim, and the server releases the scopes no longer listed
//! and shoots down the workers of the previous claim.

use std::collections::HashMap;
use std::time::Duration;

use proto::process::{Claim, Claimed};

/// Longest a Storage Server this daemon starts takes to listen.
const START_TIMEOUT: Duration = Duration::from_secs(30);

/// One deployment's claim on this node.
struct Held {
    /// The connection that holds the claim; the claim ends with it.
    _client: storage::Client,
    /// The scopes claimed, sorted.
    scopes: Vec<(String, u32)>,
    claimed: Claimed,
}

/// The claims this node's daemon holds, by deployment.
pub(crate) struct ClaimHolder {
    /// The `uniserve` executable a Storage Server this daemon starts runs as.
    executable: std::path::PathBuf,
    /// Whether processes on other nodes send this node's server page
    /// requests: the server it starts then also listens on TCP.
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

    /// Claims `claim`'s deployment at this node's Storage Server, starting
    /// the server when the node has none, or answers the claim held for the
    /// same scopes.
    ///
    /// # Errors
    ///
    /// Fails when the server cannot be started or reached, or refuses the
    /// claim because another head of the deployment holds it.
    pub(crate) fn claim(&mut self, claim: &Claim) -> anyhow::Result<Claimed> {
        use anyhow::Context as _;

        let mut scopes: Vec<(String, u32)> = claim
            .scopes
            .iter()
            .map(|scope| (scope.group.clone(), scope.rank))
            .collect();
        scopes.sort();
        if let Some(held) = self.held.get(&claim.deployment)
            && held.scopes == scopes
        {
            return Ok(held.claimed.clone());
        }

        let socket = storage::socket::default_socket();
        let mut arguments = claim.server_arguments.clone();
        if self.remote_pages {
            arguments.extend(["--tcp".to_owned(), "0.0.0.0:0".to_owned()]);
        }
        storage::socket::ensure(&self.executable, &socket, START_TIMEOUT, &arguments)
            .context("starting this node's storage server")?;
        // A previous claim of the deployment ends before the new one is made,
        // so the server sees no other head connected.
        self.held.remove(&claim.deployment);
        let mut client = storage::Client::connect(
            &socket,
            storage::Role::Head {
                deployment: claim.deployment.clone(),
            },
        )
        .with_context(|| format!("connecting to the storage server at {}", socket.display()))?;
        let token = client
            .claim(
                scopes
                    .iter()
                    .map(|(group, rank)| storage::Scope {
                        deployment: claim.deployment.clone(),
                        group: group.clone(),
                        rank: *rank,
                    })
                    .collect(),
            )
            .with_context(|| format!("claiming deployment {}", claim.deployment))?;
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
        tracing::info!(deployment = %claim.deployment, scopes = scopes.len(), "claimed the deployment's storage");
        self.held.insert(
            claim.deployment.clone(),
            Held {
                _client: client,
                scopes,
                claimed: claimed.clone(),
            },
        );
        Ok(claimed)
    }
}
