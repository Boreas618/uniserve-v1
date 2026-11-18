//! The buffer pools' conversations with other hosts' Storage Servers: the
//! peers a prefetch reads host copies from, and the hosts of the ranks of
//! another host that a transfer or a crossing writes into.
//!
//! Every call to another host blocks until it answers, so each conversation
//! runs on a helper thread of its own, never on the server's main loop, and
//! reports to the main loop through notices ([`Notifier`]). Each connection
//! fails a call once the protocol's bound of its request passes without an
//! answer (`storage::protocol::Request::remote_deadline`). A helper that waits
//! for another host's task blocks on that connection and on the pipe its main
//! loop stops it through (`Stop`), until its task's end, its stop or its
//! deadline.
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
use std::os::fd::{AsFd as _, OwnedFd};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use storage::client::Waited;
use storage::protocol::{HostCopy, Key, Priority, Region, RegionPart, Role, Span, TaskStatus};
use storage::{Client, Error, Result};

use super::Pages;
use super::polling::Notifier;
use crate::Server;

/// Which end of a span that runs on another host is storage of this host, which that host
/// reaches as a region of another host ([`Region::Remote`]) at the address the connection leaves
/// this host from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Here {
    /// Neither: both lie on the running host or on a host the span names.
    Neither,
    /// The source, which the running host READs.
    Source,
    /// The destination, which the running host WRITEs.
    Destination,
}

/// How the main loop stops a helper that waits for another host's task: the write end of a pipe
/// whose read end the helper waits on. Stopping writes to it, and dropping it closes it, which
/// stops the helper alike: a helper whose pool task is gone stops too.
pub(super) struct Stop(OwnedFd);

impl Stop {
    /// A stop and the read end its helper waits on.
    ///
    /// # Errors
    ///
    /// Fails when the pipe cannot be made.
    pub(super) fn new() -> io::Result<(Self, OwnedFd)> {
        let (reader, writer) = rustix::pipe::pipe_with(rustix::pipe::PipeFlags::CLOEXEC)?;
        Ok((Self(writer), reader))
    }

    /// Stops the helper.
    pub(super) fn stop(&self) {
        // A pipe whose buffer is full holds bytes already, which stop the helper as well.
        let _ = rustix::io::write(&self.0, &[1]);
    }
}

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
                let held = Client::connect_peer(endpoint).and_then(|mut client| {
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
/// a head of `deployment`'s `epoch`. Each span's ends are that host's own
/// storage or another host's region, or, as its [`Here`] states, storage of
/// this host, whose server listens on `port`. Notices the main loop of its end
/// ([`Server::end_pool_task`]); its stop (`cancel`, [`Stop`]) cancels it, as does `limit`
/// passing.
///
/// # Errors
///
/// Fails when the helper thread cannot be started.
pub(super) fn transfer_at(
    notifier: Notifier,
    number: u64,
    endpoint: String,
    session: (String, u64),
    spans: Vec<(Span, Here)>,
    port: Option<u16>,
    (cancel, limit): (OwnedFd, Duration),
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
    spans: Vec<(Span, Here)>,
    port: Option<u16>,
    cancel: &OwnedFd,
    limit: Duration,
) -> Result<bool> {
    let role = Role::Head {
        deployment,
        epoch,
        generation: 0,
    };
    let mut client = Client::connect_remote(endpoint, role)?;

    // This host as the other one reaches it: the address the connection
    // leaves from, and the port this server listens on.
    let here = if spans.iter().any(|(_, here)| *here != Here::Neither) {
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
        .map(|(mut span, end)| {
            let remote = |region: Region, here: &String| Region::Remote {
                endpoint: here.clone(),
                region: Box::new(region),
            };
            match (end, &here) {
                (Here::Source, Some(here)) => span.source = remote(span.source, here),
                (Here::Destination, Some(here)) => {
                    span.destination = remote(span.destination, here);
                }
                _ => {}
            }
            span
        })
        .collect();
    let task = client.copy(spans, Priority::Request, 0)?;

    // The task runs until its end, its stop or `limit`; a cancelled one's end is waited for
    // `limit` again.
    let ended = match client.wait_task(task, Some(cancel.as_fd()), Some(Instant::now() + limit))? {
        Waited::Done(ended) => Some(ended),
        Waited::Interrupted | Waited::TimedOut => None,
    };
    let (_, status) = match ended {
        Some(ended) => ended,
        None => {
            client.cancel(vec![task], None)?;
            match client.wait_task(task, None, Some(Instant::now() + limit))? {
                Waited::Done(ended) => ended,
                Waited::Interrupted | Waited::TimedOut => {
                    return Err(Error::Protocol(format!(
                        "{endpoint} reported no end of a cancelled transfer"
                    )));
                }
            }
        }
    };
    Ok(status == TaskStatus::Complete)
}

/// The pages of a put (`upload`) or a prefetch of the pool task `number` of
/// `session` that lie in parts of a host region on the host whose Storage
/// Server listens at `endpoint`, each part named by its region and index:
/// that host's connector moves them, asked there as a head of the
/// deployment's epoch, one task per part. Notices the main loop of each
/// part's results ([`Server::pool_pages_moved`]). A part that could not be
/// asked moved nothing, as did one whose end is not heard within `limit`
/// once stopped ([`Stop`]): the task was stopped, and nothing more is heard
/// of that host.
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
    (stop, limit): (OwnedFd, Duration),
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
    (stop, limit): (&OwnedFd, Duration),
) -> Result<Vec<Option<Vec<bool>>>> {
    let role = Role::Head {
        deployment,
        epoch,
        generation: 0,
    };
    let mut client = Client::connect_remote(endpoint, role)?;
    let mut tasks = Vec::with_capacity(parts.len());
    for (region, part, (pages, _)) in parts {
        let asked = client.pages(region.clone(), *part, pages.clone(), upload);
        tasks.push(asked.map_err(|error| {
            tracing::warn!(endpoint, region, part, %error, "another host's connector refused pages");
        }));
    }

    let mut results: Vec<Option<Vec<bool>>> = vec![None; parts.len()];
    let mut waiting: Vec<u64> = tasks
        .iter()
        .filter_map(|task| task.as_ref().ok())
        .copied()
        .collect();
    // Until stopped, a part's results are waited for as long as its task runs; once stopped, for
    // `limit` more.
    let mut stopped: Option<Instant> = None;
    while !waiting.is_empty() {
        let deadline = stopped.map(|at| at + limit);
        let interrupt = stopped.is_none().then(|| stop.as_fd());
        match client.wait_pages(&waiting, interrupt, deadline) {
            Ok(Waited::Done((task, moved))) => {
                waiting.retain(|waited| *waited != task);
                let at = tasks
                    .iter()
                    .position(|asked| asked.as_ref().is_ok_and(|asked| *asked == task));
                if let Some(at) = at {
                    results[at] = Some(moved);
                }
            }
            Ok(Waited::Interrupted) => stopped = Some(Instant::now()),
            Ok(Waited::TimedOut) => {
                tracing::warn!(
                    endpoint,
                    "another host reported no end of a stopped task's pages"
                );
                return Ok(results);
            }
            Err(error) => {
                tracing::warn!(endpoint, %error, "a host moving pages was lost");
                return Ok(results);
            }
        }
    }
    Ok(results)
}
