//! The buffer pools' conversations with other hosts' Storage Servers: the
//! peers a prefetch reads host copies from, and the hosts of the ranks of
//! another host that a transfer or a crossing writes into.
//!
//! Every call to another host blocks until it answers, so each conversation
//! runs on a helper thread of its own, never on the server's main loop, and
//! reports to the main loop through notices ([`Notifier`]). Each connection
//! fails a call once [`REMOTE_TIMEOUT`] passes without an answer.
//!
//! - **Peers.** A prefetch from the peers asks each peer in turn to hold
//!   the keys no earlier peer holds ([`Request::PeerHold`]); the holds of
//!   one connection end together, so each read has connections of its own.
//!   The main loop then has this host's Transfer Server read the held
//!   copies, and once that task ended the helper releases the holds
//!   ([`Request::PeerRelease`]): a peer that answers `Gone`, or that cannot
//!   be reached any more, held nothing throughout, and what was read from
//!   it is not cached (hazard H2 of `specs/buffer-pool.md`).
//! - **Ranks of another host.** A transfer into a rank of another host, or
//!   a crossing into a main process there, runs on that host's Transfer
//!   Server: the helper submits it there as a
//!   head of the deployment, its sources on this host named as regions of
//!   another host ([`Region::Remote`]) at the address the connection leaves
//!   this host from, and waits for its end, which it reports. The main
//!   loop's stop cancels it there. The pages of a put or a prefetch that lie
//!   in such a rank's part of a host region are moved by that host's
//!   connector: the helper asks it there as a head of the deployment, one
//!   task per part ([`Request::Pages`]), and reports each part's results.
//!
//! [`Request::PeerHold`]: storage::protocol::Request::PeerHold
//! [`Request::PeerRelease`]: storage::protocol::Request::PeerRelease
//! [`Request::Pages`]: storage::protocol::Request::Pages

use std::io;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use storage::protocol::{HostCopy, Key, Priority, Region, RegionPart, Role, Span, TaskStatus};
use storage::{Client, Error, Result};

use super::Pages;
use super::polling::Notifier;
use crate::Server;

/// How long connecting to another host's Storage Server, or a call on that
/// connection, may take: a peer that does not answer within it is skipped,
/// and a transfer there fails.
pub(super) const REMOTE_TIMEOUT: Duration = Duration::from_secs(2);

/// How often a helper looks for the end of a transfer on another host.
const POLL: Duration = Duration::from_millis(1);

/// What a read from the peers found: per key of the task, in order, the
/// connection that holds its host copy and the copy; per connection, its
/// peer's endpoint and the parts of the peer's host regions.
pub(super) type Held = (
    Vec<Option<(usize, HostCopy)>>,
    Vec<(String, Vec<RegionPart>)>,
);

/// Reads the host copies of `keys` in `deployment`'s buffer pools at the
/// peers at `endpoints`, for the pool task `number`: holds them, notices
/// the main loop of what is held ([`Server::peers_held`]), and once
/// `release` closes or receives, releases the holds and notices whether
/// each connection's stood throughout ([`Server::peers_released`]).
///
/// # Errors
///
/// Fails when the helper thread cannot be started.
pub(super) fn read_from_peers(
    notifier: Notifier,
    number: u64,
    deployment: String,
    endpoints: Vec<String>,
    keys: Vec<Key>,
    release: mpsc::Receiver<()>,
) -> io::Result<()> {
    std::thread::Builder::new()
        .name("peer-read".into())
        .spawn(move || {
            let mut sources: Vec<Option<(usize, HostCopy)>> = vec![None; keys.len()];
            let mut connections: Vec<(Client, String, Vec<RegionPart>)> = Vec::new();
            for endpoint in &endpoints {
                // Each key is read from the first peer that holds it.
                let wanted: Vec<Key> = keys
                    .iter()
                    .zip(&sources)
                    .filter(|(_, source)| source.is_none())
                    .map(|(key, _)| *key)
                    .collect();
                if wanted.is_empty() {
                    break;
                }
                let held = Client::connect_peer(endpoint, REMOTE_TIMEOUT).and_then(|mut client| {
                    let (copies, parts) = client.peer_hold(deployment.clone(), wanted)?;
                    Ok((client, copies, parts))
                });
                let (client, copies, parts) = match held {
                    Ok(held) => held,
                    Err(error) => {
                        tracing::warn!(endpoint, %error, "a peer's host copies cannot be held");
                        continue;
                    }
                };
                // A connection that holds nothing is let go with nothing to release.
                if copies.is_empty() {
                    continue;
                }
                let index = connections.len();
                for copy in copies {
                    let at = keys.iter().position(|key| *key == copy.key);
                    if let Some(at) = at
                        && sources[at].is_none()
                    {
                        sources[at] = Some((index, copy));
                    }
                }
                connections.push((client, endpoint.clone(), parts));
            }

            let peers = connections
                .iter()
                .map(|(_, endpoint, parts)| (endpoint.clone(), parts.clone()))
                .collect();
            notifier.send(Box::new(move |server: &mut Server| {
                server.peers_held(number, (sources, peers));
            }));

            // The holds stand until the main loop has read what it reads.
            let _ = release.recv();
            let valid = connections
                .into_iter()
                .map(|(mut client, endpoint, _)| {
                    client
                        .peer_release(deployment.clone())
                        .unwrap_or_else(|error| {
                            tracing::warn!(endpoint, %error, "a peer was lost during a read");
                            false
                        })
                })
                .collect();
            notifier.send(Box::new(move |server: &mut Server| {
                server.peers_released(number, valid);
            }));
        })?;
    Ok(())
}

/// A transfer of the pool task `number` that runs on the Transfer Server of
/// the host whose Storage Server listens at `endpoint`, submitted there as
/// a head of `deployment`'s `epoch`. Each span's source is that host's own
/// storage or another host's region, or, when flagged, storage of this
/// host, whose server listens on `port`. Notices the main loop of its end
/// ([`Server::end_pool_task`]); a message on `cancel`, or its closing,
/// cancels it, as does `limit` passing.
///
/// # Errors
///
/// Fails when the helper thread cannot be started.
pub(super) fn transfer_at(
    notifier: Notifier,
    number: u64,
    endpoint: String,
    session: (String, u64),
    spans: Vec<(Span, bool)>,
    port: Option<u16>,
    (cancel, limit): (mpsc::Receiver<()>, Duration),
) -> io::Result<()> {
    std::thread::Builder::new()
        .name("remote-transfer".into())
        .spawn(move || {
            let ended = transfer(&endpoint, session, spans, port, &cancel, limit);
            let ok = ended.unwrap_or_else(|error| {
                tracing::warn!(endpoint, %error, "a transfer into a rank of another host failed");
                false
            });
            notifier.send(Box::new(move |server: &mut Server| {
                server.end_pool_task(number, ok);
            }));
        })?;
    Ok(())
}

/// Runs a transfer at another host's Storage Server ([`transfer_at`]).
/// Returns whether it copied every span. Once cancelled, its end is waited
/// for `limit` again, after which nothing more is heard of that host.
fn transfer(
    endpoint: &str,
    (deployment, epoch): (String, u64),
    spans: Vec<(Span, bool)>,
    port: Option<u16>,
    cancel: &mpsc::Receiver<()>,
    limit: Duration,
) -> Result<bool> {
    let role = Role::Head {
        deployment,
        epoch,
        generation: 0,
    };
    let mut client = Client::connect_remote(endpoint, role, REMOTE_TIMEOUT)?;

    // This host as the other one reaches it: the address the connection
    // leaves from, and the port this server listens on.
    let here = if spans.iter().any(|(_, here)| *here) {
        let port = port.ok_or_else(|| {
            Error::Protocol("this host's Storage Server accepts no other host".into())
        })?;
        let socket = std::net::TcpStream::from(client.as_fd().try_clone_to_owned()?);
        Some(std::net::SocketAddr::new(socket.local_addr()?.ip(), port).to_string())
    } else {
        None
    };
    let spans = spans
        .into_iter()
        .map(|(mut span, from_here)| {
            if let (true, Some(here)) = (from_here, &here) {
                span.source = Region::Remote {
                    endpoint: here.clone(),
                    region: Box::new(span.source),
                };
            }
            span
        })
        .collect();
    let task = client.copy(spans, Priority::Request, 0)?;

    let mut deadline = Instant::now() + limit;
    let mut cancelled = false;
    loop {
        if let Some((_, status)) = client.poll_task(task)? {
            return Ok(status == TaskStatus::Complete);
        }
        let stop = !matches!(cancel.try_recv(), Err(mpsc::TryRecvError::Empty));
        if !cancelled && (stop || Instant::now() >= deadline) {
            client.cancel(vec![task], None)?;
            cancelled = true;
            deadline = Instant::now() + limit;
        } else if cancelled && Instant::now() >= deadline {
            return Err(Error::Protocol(format!(
                "{endpoint} reported no end of a cancelled transfer"
            )));
        }
        std::thread::sleep(POLL);
    }
}

/// The pages of a put (`upload`) or a prefetch of the pool task `number` of
/// `session` that lie in parts of a host region on the host whose Storage
/// Server listens at `endpoint`, each part named by its region and index:
/// that host's connector moves them, asked there as a head of the
/// deployment's epoch, one task per part. Notices the main loop of each
/// part's results ([`Server::pool_pages_moved`]). A part that could not be
/// asked moved nothing, as did one whose end is not heard within `limit`
/// once `stop` received or closed: the task was stopped, and nothing more
/// is heard of that host.
///
/// # Errors
///
/// Fails when the helper thread cannot be started.
pub(super) fn pages_at(
    notifier: Notifier,
    number: u64,
    endpoint: String,
    (session, upload): ((String, u64), bool),
    parts: Vec<(String, u32, Pages)>,
    (stop, limit): (mpsc::Receiver<()>, Duration),
) -> io::Result<()> {
    std::thread::Builder::new()
        .name("remote-pages".into())
        .spawn(move || {
            let results = pages_there(&endpoint, session, upload, &parts, (&stop, limit))
                .unwrap_or_else(|error| {
                    tracing::warn!(endpoint, %error, "pages of a rank of another host failed");
                    Vec::new()
                });
            notifier.send(Box::new(move |server: &mut Server| {
                for (index, (.., (_, keys))) in parts.iter().enumerate() {
                    let moved = results.get(index).cloned().flatten().unwrap_or_default();
                    server.pool_pages_moved(number, keys, &moved);
                }
            }));
        })?;
    Ok(())
}

/// Has the connector of another host's Storage Server move pages
/// ([`pages_at`]). Returns per part the results its connector reported,
/// `None` for one it refused or did not report on: once the task was
/// stopped and `limit` passed, or once the connection failed.
///
/// # Errors
///
/// When that host's server cannot be reached.
fn pages_there(
    endpoint: &str,
    (deployment, epoch): (String, u64),
    upload: bool,
    parts: &[(String, u32, Pages)],
    (stop, limit): (&mpsc::Receiver<()>, Duration),
) -> Result<Vec<Option<Vec<bool>>>> {
    let role = Role::Head {
        deployment,
        epoch,
        generation: 0,
    };
    let mut client = Client::connect_remote(endpoint, role, REMOTE_TIMEOUT)?;
    let mut tasks = Vec::with_capacity(parts.len());
    for (region, part, (pages, _)) in parts {
        let asked = client.pages(region.clone(), *part, pages.clone(), upload);
        tasks.push(asked.map_err(|error| {
            tracing::warn!(endpoint, region, part, %error, "another host's connector refused pages");
        }));
    }

    let mut results: Vec<Option<Vec<bool>>> = vec![None; parts.len()];
    let mut deadline = None;
    loop {
        for (task, result) in tasks.iter().zip(&mut results) {
            let Ok(task) = task else {
                continue;
            };
            if result.is_none() {
                match client.poll_pages(*task) {
                    Ok(polled) => *result = polled,
                    Err(error) => {
                        tracing::warn!(endpoint, %error, "a host moving pages was lost");
                        return Ok(results);
                    }
                }
            }
        }
        let waiting = tasks
            .iter()
            .zip(&results)
            .any(|(task, result)| task.is_ok() && result.is_none());
        if !waiting {
            return Ok(results);
        }

        if deadline.is_none() && !matches!(stop.try_recv(), Err(mpsc::TryRecvError::Empty)) {
            deadline = Some(Instant::now() + limit);
        }
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            tracing::warn!(
                endpoint,
                "another host reported no end of a stopped task's pages"
            );
            return Ok(results);
        }
        std::thread::sleep(POLL);
    }
}
