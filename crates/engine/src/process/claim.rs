//! The node's Storage Server claim: the daemon starts the node's server and mints the claim, and
//! each rank asks for it over the process interface before it maps any storage.

use std::time::Duration;

use anyhow::{Result, bail};
use proto::config::EngineArgs;

use super::launch::ProcessInterface;

/// Asks this node's daemon for the node's Storage Server.
///
/// The socket and the claim are runtime facts of the node: the daemon mints the claim when it starts
/// the server, so a launch file cannot carry either. The answer states the socket the server listens
/// on, the token this rank presents, and where another host reaches its pages.
pub(super) fn claim(
    process: &mut ProcessInterface,
    launch: &EngineArgs,
    host: &str,
    host_kv_cache: bool,
    ranks: &[u32],
) -> Result<ipc::StorageLaunch> {
    let request = proto::process::Request {
        id: 1,
        kind: Some(proto::process::request::Kind::Claim(
            proto::process::Claim {
                host: host.to_owned(),
                deployment: launch.engine_config.deployment.clone(),
                scopes: ranks
                    .iter()
                    .map(|rank| proto::process::Scope {
                        group: launch.group.clone(),
                        rank: *rank,
                    })
                    .collect(),
                // A server the daemon starts for this claim takes the deployment's connector settings;
                // the main process has none of its own to state.
                server_arguments: Vec::new(),
                epoch: launch.epoch,
                retire: Vec::new(),
            },
        )),
    };
    let reply = process.request(request, CLAIM_TIMEOUT)?;
    if !reply.error.is_empty() {
        bail!("the daemon refused this rank's claim: {}", reply.error);
    }
    let Some(proto::process::reply::Kind::Claimed(claimed)) = reply.kind else {
        bail!("the daemon answered a claim with another reply");
    };
    Ok(ipc::StorageLaunch {
        socket: claimed.socket,
        deployment: launch.engine_config.deployment.clone(),
        claim: claim_token(&claimed.claim)?,
        // The host KV cache is the deployment's decision, from the build config: a rank whose
        // deployment keeps a host tier maps a page pool per cache group, and one that does not
        // maps nothing. Stating `false` for a deployment that asked for it would size the epoch's
        // host KV cache with no rank holding a page of it.
        host_kv_cache,
        tcp: (!claimed.tcp.is_empty()).then_some(claimed.tcp),
        // The deployment's *storage* peers - where another node's Storage Server is reached for the
        // weights a rank receives from it - are empty because nothing this process is told states
        // them: the claim's answer carries this node's own server only. A deployment of several
        // hosts needs them before a cross-host weight transfer can be planned, and until then this
        // rank maps what its own node holds.
        peers: Vec::new(),
        epoch: launch.epoch,
        // The standby role is the epoch layer's: a standby maps its scope's storage
        // read-only until it is promoted, and claiming it writable would give the scope two
        // writers. The launch file states the role, so the process never has to infer it.
        standby: launch.standby(),
        standby_storage: ipc::StandbyStorage::Auto,
    })
}

/// The 32 hex digits of a claim token, as the Storage Server takes it.
fn claim_token(bytes: &[u8]) -> Result<[u8; 16]> {
    bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("a claim token is 16 bytes, not {}", bytes.len()))
}

/// How long the daemon may take to answer this rank's storage claim: starting a Storage
/// Server for the node is the slow part, and it is the daemon's to do.
const CLAIM_TIMEOUT: Duration = Duration::from_secs(60);
