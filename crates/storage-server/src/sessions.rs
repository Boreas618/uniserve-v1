//! The server's side of the buffer pools: the sessions of engine epochs
//! (`specs/buffer-pool.md` §11, §12.4, §14.1).
//!
//! The pools and their queue pairs live on a thread of their own
//! ([`polling`]); the main loop answers what travels on the socket and runs
//! the pools' tasks on its children.
//!
//! - **Sessions.** An engine's main process, a head of its epoch, opens the
//!   epoch's session ([`Request::Open`]) and gets one queue pair per
//!   scheduler. The session speaks the version of the interface its `Open`
//!   states, one of those the server serves (`storage::runtime::OLDEST` to
//!   `VERSION`), for its life, across restarts in place; the pool refuses
//!   any other. The engine closes it ([`Request::Close`]) once its workers
//!   stopped.
//!   When its connection closes first, the session closes only once the
//!   engine's processes have exited and no GPU lists them (A7), the
//!   evidence a shoot-down gathers, never on the connection's close alone:
//!   the processes are the main process and every worker of the epoch the
//!   server knows.
//! - **Where units lie.** A rank states where its part of a device region's
//!   units lies ([`Request::Place`]); the head states it for a rank of
//!   another host, whose tensors that host's Storage Server retains
//!   ([`Request::PlaceRemote`]). A part is named by the rank's index in its
//!   worker group: the ranks of one index, in every group and replica of a
//!   deployment, hold the same share of every unit, so a task matches the
//!   ranks of two memories by it. A host region is one memfd per part,
//!   which the server creates on the host's huge pages when it has them and
//!   admits with the host's storage part by part, as each rank of this host
//!   takes its part ([`Request::HostPart`]): a region whose ranks lie on
//!   several hosts holds on each host only that host's parts, so `Open`
//!   admits nothing of it. A rank of another host has its part on its own
//!   host, which the head's [`Request::PlaceRemote`] for that rank names
//!   (`specs/buffer-pool.md` §15). The part of one index on each host is a
//!   host part of its own, which holds the host copies written there: a
//!   rank states the host its part lies on as it would a memory, and the
//!   head states it for a rank of another host, and a task reaches a host
//!   part only where such a place states it. With them a task's runs turn
//!   into the Transfer Server's spans and the connector's pages.
//! - **Segments.** The head grants a device region's memory space each
//!   segment the engine's ranks created ([`Request::Grant`]), and withdraws
//!   one before they release it ([`Request::Withdraw`]); `Open`'s answer
//!   states each region's segment table.
//! - **Tasks.** A transfer between worker groups goes to the Transfer
//!   Server as one task of spans, field by field and rank by rank, from each
//!   rank of the memory the content is in to the rank of the same index of
//!   the target memory; a region no rank placed (slot rows, product
//!   buffers) is memory of the ranks' own processes, whose copies move no
//!   bytes. A crossing (`Replicate`) is such a task too, from the run its
//!   block's writer wrote in one main process's product buffer to the
//!   block's run in another's. The spans into a rank of another host run on
//!   that host's Transfer Server, and a source on another host is read from
//!   there ([`hosts`]). A prefetch from the remote store and a put to it go
//!   to the connector as the pages of each part of the host region, a block
//!   of `n` units being `n` pages under keys derived from its key
//!   ([`unit_key`]): a prefetch's into the host parts it fills, a put's from
//!   a host part of each index that holds the copy; the pages of a part on
//!   another host go to that host's connector ([`Request::Pages`]). A
//!   prefetch from the peers holds the keys at the
//!   peers the session names, has the Transfer Server of each part's host
//!   read each held copy from the peer's part of the same index, wherever
//!   that lies, and keeps what was read only from a peer whose holds stood
//!   until the read ended ([`hosts`]). A task that cannot reach every part
//!   of a unit moves none of it. A child's
//!   report ends the task; a child's loss ends its tasks as failed once its
//!   process has exited, so an end is reported only when nothing touches
//!   the blocks any more (A5).
//! - **Peers.** Another host's Storage Server holds the host copies it
//!   reads ([`Request::PeerHold`]) until it releases them
//!   ([`Request::PeerRelease`]) or its connection closes.

mod hosts;
pub(crate) mod polling;

use std::collections::{BTreeMap, BTreeSet};
use std::os::fd::OwnedFd;
use std::path::PathBuf;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use storage::protocol::{
    Allocation, ErrorKind, Event, HostCopy, Key, Location, PageShape, Priority, Region, RegionPart,
    Response, Role, Segment, Span, StorageRecord, UnitRun,
};
use storage::runtime::{Open, RequestId, Source, Tier};

use crate::buffer_pool::{ClientId as PoolSession, Encoding, QueueId};
use crate::kv_storage::{Owner, Task};
use crate::{
    Attachment, Client, ClientId, Handling, Holder, Server, StorageEntry, StorageId, invalid,
    resource,
};
use polling::{Copied, HostPart, NamedRun};

#[cfg(doc)]
use storage::protocol::Request;

/// How long a transfer between worker groups may take unless the server's
/// options say otherwise (`--transfer-deadline-s`).
pub(crate) const TRANSFER_DEADLINE: Duration = Duration::from_secs(60);

/// How long a put to the remote store may take unless the server's options
/// say otherwise (`--put-deadline-s`).
pub(crate) const PUT_DEADLINE: Duration = Duration::from_secs(60);

/// The tokens a block brought from the remote store holds: the store keeps
/// no count, and it keeps whole blocks.
const BLOCK_TOKENS: u16 = 64;

/// A task of a pool: the session's queue pair and the request that started it.
type PoolTask = (QueueId, RequestId);

/// Where a rank states the units of a device region lie: its deployment, the
/// region, the device memory and the part (the rank's index in its group).
type Place = (String, String, String, u32);

/// Where a rank's units lie: the endpoint of the Storage Server of its host
/// when it is another host, at which its runs' tensors are retained, and
/// the runs.
type Placed = (Option<String>, Vec<UnitRun>);

/// A session: its deployment and its epoch.
type Session = (String, u64);

/// An engine's session: its deployment, its epoch and the launch generation of the engine that
/// opened it. An epoch has a session per engine of it on the host: the serving engine's, its
/// standby's, a lost engine's beside its relaunch's.
type Engine = (String, u64, u64);

/// The pages of a part of a host region a connector task moves, with the
/// copy of the pool's task each page is of.
type Pages = (Vec<(Key, u32)>, Vec<usize>);

/// A host region a session stated: the limit of its numbering, its bytes of
/// a unit, summed over the parts, and the bytes of one part.
type HostRegion = (u32, u64, u64);

/// Where a part of a host region lies: its index, the endpoint of its host's
/// Storage Server (none for this host) and its bytes of a unit.
type PartHost = (u32, Option<String>, u64);

/// Where the parts of a host region lie ([`Server::part_hosts`]): how many
/// parts a unit has, and each part known.
type PartHosts = (u32, Vec<PartHost>);

/// The server's sessions and what it keeps for the buffer pools.
pub(crate) struct Sessions {
    pub(crate) thread: polling::Thread,
    /// The open sessions by engine, each with the client of the engine's
    /// main process that opened it, while it is connected, and its number in
    /// the pool.
    open: BTreeMap<Engine, (ClientId, PoolSession)>,
    /// The sessions whose opener's connection closed, each with its number
    /// in the pool and the engine's processes the server waits for: each
    /// closes once all of them have exited and no GPU lists them.
    gone: BTreeMap<Engine, (PoolSession, Vec<Arc<OwnedFd>>)>,
    /// The host regions the sessions stated, by deployment and name.
    hosts: BTreeMap<(String, String), HostRegion>,
    /// The parts of the host regions, by deployment, region and part (the
    /// index in their groups of the ranks that share it).
    parts: BTreeMap<(String, String, u32), Part>,
    /// The pool number the connector maps the next part under. It passes
    /// every part's: a successor numbers its parts above those it takes
    /// over, and its connector, which starts afresh, maps them anew.
    next_pool: u64,
    /// Where the units of the device regions lie, by deployment, region,
    /// device memory and part, as each rank stated it.
    places: BTreeMap<Place, Placed>,
    /// The peers each session names, which a prefetch from the peers reads.
    peers: BTreeMap<Session, Vec<String>>,
    /// The parts of each engine's ranks of other hosts, each with the
    /// endpoint of its host's Storage Server, as the head of the engine's
    /// session stated where those ranks' units lie
    /// ([`Request::PlaceRemote`]): a rank's part of a host region lies on its
    /// own host. They are the engine's, as its ranks are, and go with its
    /// session: another engine of the epoch (its standby) has ranks of its
    /// own.
    remote: BTreeMap<Engine, BTreeSet<(u32, String)>>,
    /// The pools' tasks that run on a child, by the number the server gave
    /// each.
    running: BTreeMap<u64, Running>,
    next: u64,
    /// The tasks of a child whose connection closed, with the child's
    /// process: each ends once that process has exited.
    lost: Vec<(u32, u64)>,
    /// The epochs on trial whose update was rolled back while a session of
    /// theirs was open: the pool rolls each back once its last session
    /// closes.
    rolled: BTreeSet<(String, u64)>,
    /// How long a transfer between worker groups may take. A helper that
    /// runs one on another host cancels it after that long too, and then
    /// waits that long again for its end.
    transfer_deadline: Duration,
    /// How long a put to the remote store may take: how long a helper whose
    /// pages another host's connector moves waits for their end once the
    /// task was stopped.
    put_deadline: Duration,
}

/// One rank's part of a host region: the storage that holds it, its bytes
/// of a unit, and the pool number the connector maps it under.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct Part {
    storage: StorageId,
    bytes: u64,
    pool: u64,
}

/// A pool's task that runs on a child or on another host: its session and
/// the pool's task; the keys of its copies in their order, each with
/// whether every page of it moved so far; and its parts that have not
/// reported, the connector's tasks or the transfers here and on other
/// hosts, with whether every part that reported ended well.
struct Running {
    session: Session,
    task: PoolTask,
    keys: Vec<(Key, bool)>,
    waiting: usize,
    ok: bool,
    /// The task was told to stop: what it has not started, it does not.
    stopped: bool,
    /// What stops its transfers on other hosts' Transfer Servers and their connectors' pages.
    cancels: Vec<hosts::Stop>,
    /// A read from the peers: what has the helper release its holds once
    /// dropped, and per copy the peer connection it is read from with the
    /// tokens its block holds, for the leading copies being read.
    release: Option<mpsc::Sender<()>>,
    reads: Vec<(usize, u16)>,
    /// For a read from the peers, the host parts each copy fills (none for a
    /// copy of groups that state no host parts).
    parts: Vec<Vec<HostPart>>,
}

impl Running {
    fn new(session: Session, task: PoolTask, keys: Vec<(Key, bool)>) -> Self {
        Self {
            session,
            task,
            keys,
            waiting: 0,
            ok: true,
            stopped: false,
            cancels: Vec::new(),
            release: None,
            reads: Vec::new(),
            parts: Vec::new(),
        }
    }
}

/// What the sessions are as a restart in place hands them over: the
/// pools' image with its format, and the number of queue pairs, whose memfds
/// travel apart, and what the main loop keeps. No task runs then.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct SessionsHandoff {
    #[serde(with = "serde_bytes")]
    image: Vec<u8>,
    /// The format of the image ([`crate::buffer_pool::FORMAT`]); a
    /// predecessor that states none wrote format 1.
    #[serde(default = "first_format")]
    pub(crate) format: u32,
    pub(crate) queues: usize,
    open: Vec<(Engine, (ClientId, PoolSession))>,
    /// The sessions waiting for their engine's processes, each process by
    /// the index of its pidfd in the handoff.
    gone: Vec<(Engine, (PoolSession, Vec<usize>))>,
    hosts: Vec<((String, String), HostRegion)>,
    parts: Vec<((String, String, u32), Part)>,
    places: Vec<(Place, Placed)>,
    peers: Vec<(Session, Vec<String>)>,
    remote: Vec<(Engine, BTreeSet<(u32, String)>)>,
    next: u64,
    rolled: Vec<(String, u64)>,
}

/// The format of an image whose handoff states none: that of the builds
/// before images stated their format.
fn first_format() -> u32 {
    1
}

/// The limits of a transfer and of a put to the remote store, in
/// nanoseconds, as the buffer pool keeps them.
fn limits(transfer: Duration, put: Duration) -> (u64, u64) {
    let nanoseconds = |deadline: Duration| u64::try_from(deadline.as_nanos()).unwrap_or(u64::MAX);
    (nanoseconds(transfer), nanoseconds(put))
}

impl Sessions {
    /// No session, and the buffer pool's thread started.
    pub(crate) fn new() -> std::io::Result<Self> {
        let limits = limits(TRANSFER_DEADLINE, PUT_DEADLINE);
        Self::with(polling::Thread::start(limits, None)?)
    }

    /// How long a transfer between worker groups and a put to the remote
    /// store may take from now on.
    pub(crate) fn set_deadlines(&mut self, transfer: Duration, put: Duration) {
        self.transfer_deadline = transfer;
        self.put_deadline = put;
        let limits = limits(transfer, put);
        self.thread.send(move |pools| pools.set_limits(limits));
    }

    /// From now on, records the trace of every buffer pool's inputs in the
    /// directory `trace`, and checks every pool's tables after each input
    /// when `check` is set ([`polling::Pools::watch`]).
    pub(crate) fn watch_pools(&self, trace: Option<PathBuf>, check: bool) {
        self.thread.send(move |pools| pools.watch(trace, check));
    }

    fn with(thread: polling::Thread) -> std::io::Result<Self> {
        Ok(Self {
            thread,
            open: BTreeMap::new(),
            gone: BTreeMap::new(),
            hosts: BTreeMap::new(),
            parts: BTreeMap::new(),
            next_pool: 1,
            places: BTreeMap::new(),
            peers: BTreeMap::new(),
            remote: BTreeMap::new(),
            running: BTreeMap::new(),
            next: 1,
            lost: Vec::new(),
            rolled: BTreeSet::new(),
            transfer_deadline: TRANSFER_DEADLINE,
            put_deadline: PUT_DEADLINE,
        })
    }

    /// Whether no pool task runs on a child and the buffer pool's thread
    /// is frozen with no task outstanding: a restart may hand it over.
    pub(crate) fn quiet(&self) -> bool {
        self.running.is_empty()
            && self.lost.is_empty()
            && self.thread.call(|pools| pools.quiet()).unwrap_or(true)
    }

    /// The sessions for a successor, with the queue pairs' memfds; the
    /// waiting sessions' pidfds are named by `pidfd`.
    pub(crate) fn hand_off(
        &mut self,
        mut pidfd: impl FnMut(&Arc<OwnedFd>) -> usize,
    ) -> std::io::Result<(SessionsHandoff, Vec<OwnedFd>)> {
        let (image, descriptors) = self
            .thread
            .call(|pools| pools.hand_off())
            .ok_or_else(|| std::io::Error::other("the buffer pool's thread ended"))??;
        let handoff = SessionsHandoff {
            image,
            format: crate::buffer_pool::FORMAT,
            queues: descriptors.len(),
            open: std::mem::take(&mut self.open).into_iter().collect(),
            gone: std::mem::take(&mut self.gone)
                .into_iter()
                .map(|(engine, (pool, pidfds))| {
                    (engine, (pool, pidfds.iter().map(&mut pidfd).collect()))
                })
                .collect(),
            hosts: std::mem::take(&mut self.hosts).into_iter().collect(),
            parts: std::mem::take(&mut self.parts).into_iter().collect(),
            places: std::mem::take(&mut self.places).into_iter().collect(),
            peers: std::mem::take(&mut self.peers).into_iter().collect(),
            remote: std::mem::take(&mut self.remote).into_iter().collect(),
            next: self.next,
            rolled: std::mem::take(&mut self.rolled).into_iter().collect(),
        };
        Ok((handoff, descriptors))
    }

    /// The sessions a predecessor handed over, the buffer pool's thread
    /// going on with its pools and queue pairs; `pidfd` gives the pidfd of
    /// an index.
    pub(crate) fn resume(
        handoff: SessionsHandoff,
        queues: Vec<OwnedFd>,
        pidfd: impl Fn(usize) -> std::io::Result<Arc<OwnedFd>>,
    ) -> std::io::Result<Self> {
        let image = Some((handoff.image, handoff.format, queues));
        let limits = limits(TRANSFER_DEADLINE, PUT_DEADLINE);
        let thread = polling::Thread::start(limits, image)?;
        let mut sessions = Self::with(thread)?;
        sessions.open = handoff.open.into_iter().collect();
        for (engine, (pool, indices)) in handoff.gone {
            let pidfds = indices
                .into_iter()
                .map(&pidfd)
                .collect::<std::io::Result<_>>()?;
            sessions.gone.insert(engine, (pool, pidfds));
        }
        sessions.hosts = handoff.hosts.into_iter().collect();
        sessions.parts = handoff.parts.into_iter().collect();
        sessions.next_pool = sessions
            .parts
            .values()
            .map(|part| part.pool + 1)
            .max()
            .unwrap_or(1);
        sessions.places = handoff.places.into_iter().collect();
        sessions.peers = handoff.peers.into_iter().collect();
        sessions.remote = handoff.remote.into_iter().collect();
        sessions.next = handoff.next;
        sessions.rolled = handoff.rolled.into_iter().collect();
        Ok(sessions)
    }
}

/// The key of unit `unit` of a block of `key` in the remote store: the
/// block's key for its first unit, and a key derived from it for each other
/// one, so that every unit is one object of the store.
fn unit_key(key: &Key, unit: u32) -> Key {
    if unit == 0 {
        return *key;
    }
    let mut digest = storage::key::Digest::new(b"uniserve block unit");
    digest.bytes(&key.0);
    digest.number(u64::from(unit));
    digest.key()
}

/// What a part of a host region is to the connector: a pool of pages of one
/// unit's bytes of the part each, whose layout is the region's name, so
/// that a region of one name on every host stores its blocks under the same
/// objects.
fn part_shape(region: &str, rank: u32, bytes: u64) -> PageShape {
    let mut layout = storage::key::Digest::new(b"uniserve host region");
    layout.text(region);
    PageShape {
        layout: layout.key(),
        kv_head_offset: rank,
        kv_head_count: 1,
        page_bytes: bytes,
    }
}

/// The spans that copy `count` units from the runs of one rank (`from`),
/// from unit `firsts.0` on, into the runs of another (`to`), from unit
/// `firsts.1` on: per field the destination holds, one span per stretch of
/// units one run of each holds. A transfer reads and writes the same units
/// in two memories; a crossing into another region writes units of their
/// own.
fn spans(
    from: &[UnitRun],
    to: &[UnitRun],
    (source_first, first): (u32, u32),
    count: u32,
) -> Result<Vec<Span>, String> {
    let fields: BTreeSet<&str> = to.iter().map(|run| run.field.as_str()).collect();
    let holding = |runs: &'_ [UnitRun], field: &str, unit: u32| {
        runs.iter()
            .find(|run| run.field == field && run.first <= unit && unit < run.first + run.count)
            .cloned()
    };
    let mut spans = Vec::new();
    for field in fields {
        let mut done = 0;
        while done < count {
            let (read, unit) = (source_first + done, first + done);
            let (Some(source), Some(destination)) =
                (holding(from, field, read), holding(to, field, unit))
            else {
                return Err(format!(
                    "unit {read}'s {field}, or unit {unit}'s it fills, lies in no run of its rank"
                ));
            };
            if source.bytes != destination.bytes {
                return Err(format!(
                    "the ranks hold {field} at {} and {} bytes a unit",
                    source.bytes, destination.bytes
                ));
            }
            let stretch = (count - done)
                .min(source.first + source.count - read)
                .min(destination.first + destination.count - unit);
            spans.push(Span {
                source: Region::Tensor {
                    key: source.key,
                    at: Some(source.location),
                },
                source_offset: source.offset + u64::from(read - source.first) * source.bytes,
                destination: Region::Tensor {
                    key: destination.key,
                    at: Some(destination.location),
                },
                destination_offset: destination.offset
                    + u64::from(unit - destination.first) * destination.bytes,
                bytes: u64::from(stretch) * source.bytes,
                rows: 1,
                source_pitch: 0,
                destination_pitch: 0,
            });
            done += stretch;
        }
    }
    Ok(spans)
}

/// Keeps of the parts `hosts` states one of each part index, this host's
/// where it holds one: the parts a read of a copy all of them hold takes.
fn one_of_each(hosts: &mut Vec<PartHost>) {
    hosts.sort_by_key(|(part, at, _)| (*part, at.is_some()));
    hosts.dedup_by_key(|(part, ..)| *part);
}

/// The spans that read a host copy a peer holds (`copy`, held at the peer
/// `peer`, whose parts of its host regions are `held`) into the units `write`
/// names of a host region whose parts lie as `destinations` states
/// ([`Server::part_hosts`]): per part, from the peer's part of the same
/// index and bytes, on the peer's host or on the one the peer names. Each
/// span comes with the host it is copied on, the destination's (`None` for
/// this host, which listens at `here`), and whether its source is storage
/// of this host, which another host reads. `None` when the copy cannot be
/// read whole: it holds other units than the write, a part has no host or
/// no counterpart at the peer.
fn read_spans(
    copy: &HostCopy,
    (peer, held): (&str, &[RegionPart]),
    (region, first, count): &NamedRun,
    (parts, destinations): &PartHosts,
    here: Option<&str>,
) -> Option<Vec<(Option<String>, Span, bool)>> {
    let whole = (0..*parts).all(|part| destinations.iter().any(|(at, ..)| *at == part));
    if copy.count != *count || *parts == 0 || !whole {
        return None;
    }

    let mut spans = Vec::with_capacity(destinations.len());
    for (part, host, bytes) in destinations {
        let source = held.iter().find(|source| {
            source.region == copy.region && source.rank == *part && source.bytes == *bytes
        })?;
        let source_host = source.endpoint.as_deref().unwrap_or(peer);
        let source_part = Region::Part {
            region: copy.region.clone(),
            rank: *part,
        };
        // A source on the destination's host is that host's own storage; one
        // on this host that another host reads is flagged for it to name.
        let on_this_host = here == Some(source_host);
        let (source, from_here) = match host {
            Some(endpoint) if endpoint == source_host => (source_part, false),
            Some(_) if on_this_host => (source_part, true),
            None if on_this_host => (source_part, false),
            _ => (
                Region::Remote {
                    endpoint: source_host.to_owned(),
                    region: Box::new(source_part),
                },
                false,
            ),
        };
        let span = Span {
            source,
            source_offset: u64::from(copy.first) * bytes,
            destination: Region::Part {
                region: region.clone(),
                rank: *part,
            },
            destination_offset: u64::from(*first) * bytes,
            bytes: u64::from(*count) * bytes,
            rows: 1,
            source_pitch: 0,
            destination_pitch: 0,
        };
        spans.push((host.clone(), span, from_here));
    }
    Some(spans)
}

impl Server {
    /// See [`Request::Open`], from the head `id` of `epoch` of `deployment`, the main process
    /// of the engine of launch `generation`.
    pub(crate) fn open_pool(
        &mut self,
        id: ClientId,
        (deployment, epoch, generation): (&str, u64, u64),
        open: &Open,
    ) -> Result<Handling, (ErrorKind, String)> {
        if self
            .clients
            .get(&id)
            .is_none_or(|client| client.pidfd.is_none())
        {
            return Err(invalid(
                "a session is opened on the host: its queue pairs travel as descriptors",
            ));
        }
        if open.epoch != epoch {
            return Err(invalid(&format!(
                "a head of epoch {epoch} opens no session of epoch {}",
                open.epoch
            )));
        }
        // One session per engine: another engine of the epoch (its standby, a relaunch beside a
        // lost engine) opens one of its own.
        let session = (deployment.to_owned(), epoch);
        let engine = (deployment.to_owned(), epoch, generation);
        if self.sessions.open.contains_key(&engine) || self.sessions.gone.contains_key(&engine) {
            return Err((
                ErrorKind::Held,
                format!(
                    "the session of engine {generation} of epoch {epoch} of {deployment} is \
                     open, or its engine's processes have not left yet"
                ),
            ));
        }

        // The host regions new to the deployment: the server creates and admits
        // their parts as the ranks of this host ask ([`Self::host_part`]).
        let new: Vec<_> = open
            .regions
            .iter()
            .filter(|region| region.tier == Tier::Host)
            .filter(|region| {
                !self
                    .sessions
                    .hosts
                    .contains_key(&(deployment.to_owned(), region.name.clone()))
            })
            .collect();

        let (named, layout) = (deployment.to_owned(), open.clone());
        let opened = self
            .sessions
            .thread
            .call(move |pools| pools.open(&named, &layout))
            .ok_or_else(|| resource(std::io::Error::other("the buffer pool's thread ended")))?;
        let (pool, opened, descriptors) = opened.map_err(|refusal| invalid(&refusal))?;
        for region in new {
            self.sessions.hosts.insert(
                (deployment.to_owned(), region.name.clone()),
                (region.limit, region.unit_bytes, region.part_bytes),
            );
        }
        self.sessions.peers.insert(session, open.peers.clone());
        self.sessions.open.insert(engine, (id, pool));
        Ok(Handling::Descriptors(Response::Opened(opened), descriptors))
    }

    /// The engine whose session the head `id` of `epoch` of `deployment`
    /// holds open, when it is on the host: what the requests a head makes
    /// about its session's memory need.
    fn session_head(
        &self,
        id: ClientId,
        deployment: &str,
        epoch: u64,
    ) -> Result<Engine, (ErrorKind, String)> {
        if self
            .clients
            .get(&id)
            .is_none_or(|client| client.pidfd.is_none())
        {
            return Err(invalid("a session's memory is stated on its host"));
        }
        let held = self
            .sessions
            .open
            .iter()
            .find(|((of, at, _), (head, _))| of == deployment && *at == epoch && *head == id);
        let Some((engine, _)) = held else {
            return Err(invalid(&format!(
                "no session of epoch {epoch} of {deployment} is open"
            )));
        };
        Ok(engine.clone())
    }

    /// What the end of `engine`'s session leaves of its epoch, the server
    /// holding it open no more: when it was its epoch's last session, the
    /// peers the epoch names go, and the return is whether the epoch's
    /// update was rolled back while it was open, which the pool then rolls
    /// back. An epoch's other sessions (its engine's standby) keep both.
    fn epoch_ends(&mut self, engine: &Engine) -> bool {
        let (deployment, epoch, _) = engine;
        let others = self
            .sessions
            .open
            .keys()
            .chain(self.sessions.gone.keys())
            .any(|(of, at, _)| of == deployment && at == epoch);
        if others {
            return false;
        }
        let session = (deployment.clone(), *epoch);
        self.sessions.peers.remove(&session);
        self.sessions.rolled.remove(&session)
    }

    /// See [`Request::Grant`], from the head `id` of `epoch` of `deployment`.
    pub(crate) fn grant(
        &mut self,
        id: ClientId,
        (deployment, epoch): (&str, u64),
        region: &str,
        segment: Segment,
        (first, units): (u32, u32),
    ) -> Result<Handling, (ErrorKind, String)> {
        self.session_head(id, deployment, epoch)?;
        let (named, region_name) = (deployment.to_owned(), region.to_owned());
        self.sessions
            .thread
            .call(move |pools| pools.grant(&named, &region_name, segment, first, units))
            .ok_or_else(|| resource(std::io::Error::other("the buffer pool's thread ended")))?
            .map_err(|refusal| invalid(&format!("{region}: {refusal}")))?;
        // A region's memory is hundreds of segments of one granule a column each, granted one by
        // one: each is a debug line, and the head states what it granted at once.
        tracing::debug!(
            deployment,
            region,
            epoch = segment.epoch,
            ordinal = segment.ordinal,
            first,
            units,
            "a segment joined a memory space"
        );
        Ok(Handling::Response(Response::Granted, Vec::new()))
    }

    /// See [`Request::Withdraw`], from the head `id` of `epoch` of
    /// `deployment`.
    pub(crate) fn withdraw(
        &mut self,
        id: ClientId,
        (deployment, epoch): (&str, u64),
        region: &str,
        segment: Segment,
    ) -> Result<Handling, (ErrorKind, String)> {
        self.session_head(id, deployment, epoch)?;
        let (named, region_name) = (deployment.to_owned(), region.to_owned());
        self.sessions
            .thread
            .call(move |pools| pools.withdraw(&named, &region_name, segment))
            .ok_or_else(|| resource(std::io::Error::other("the buffer pool's thread ended")))?
            .map_err(|refusal| invalid(&format!("{region}: {refusal}")))?;
        tracing::debug!(
            deployment,
            region,
            epoch = segment.epoch,
            ordinal = segment.ordinal,
            "a segment left a memory space"
        );
        Ok(Handling::Response(Response::Withdrawn, Vec::new()))
    }

    /// See [`Request::Close`], from the head of `epoch` of `deployment`
    /// that opened the session of its engine of launch `generation`.
    pub(crate) fn close_pool(
        &mut self,
        (deployment, epoch, generation): (&str, u64, u64),
    ) -> Result<Handling, (ErrorKind, String)> {
        let engine = (deployment.to_owned(), epoch, generation);
        let pool = match self.sessions.open.remove(&engine) {
            Some((_, pool)) => Some(pool),
            None => self.sessions.gone.remove(&engine).map(|(pool, _)| pool),
        };
        let Some(pool) = pool else {
            return Err(invalid(&format!(
                "no session of engine {generation} of epoch {epoch} of {deployment} is open"
            )));
        };
        self.sessions.remote.remove(&engine);
        let rolled = self.epoch_ends(&engine);
        let named = deployment.to_owned();
        self.sessions
            .thread
            .call(move |pools| {
                pools.close(&named, pool);
                if rolled {
                    pools.roll_back(&named, epoch);
                }
            })
            .ok_or_else(|| resource(std::io::Error::other("the buffer pool's thread ended")))?;
        Ok(Handling::Response(Response::Closed, Vec::new()))
    }

    /// See [`Request::Place`], from the worker `holder`. For a host region
    /// the rank states the host its part lies on, as the layout's host parts
    /// name it (`storage::runtime::Group::parts`), in place of a memory and
    /// with no run: the part it took at this server ([`Self::host_part`]),
    /// which a task reaches as that host's part of its index.
    pub(crate) fn place(
        &mut self,
        holder: &Holder,
        (region, memory, part): (&str, &str, u32),
        runs: &[UnitRun],
    ) -> Result<Handling, (ErrorKind, String)> {
        if !self.current(holder) {
            return Err(crate::stale(holder));
        }
        if let Some(run) = runs.iter().find(|run| run.count == 0 || run.bytes == 0) {
            return Err(invalid(&format!(
                "a run of {} holds no unit or no byte",
                run.field
            )));
        }
        let deployment = holder.scope.deployment.clone();
        let hosted = self
            .sessions
            .hosts
            .contains_key(&(deployment.clone(), region.to_owned()));
        let taken = (deployment.clone(), region.to_owned(), part);
        let held = self.sessions.parts.contains_key(&taken);
        if hosted && (memory.is_empty() || !runs.is_empty() || !held) {
            return Err(invalid(&format!(
                "a place of the host region {region} names the host of a part this host holds, \
                 and no run"
            )));
        }
        let place = (deployment, region.to_owned(), memory.to_owned(), part);
        self.sessions.places.insert(place, (None, runs.to_vec()));
        Ok(Handling::Response(Response::Placed, Vec::new()))
    }

    /// See [`Request::PlaceRemote`], from the head `id` of `epoch` of
    /// `deployment`: part `part` of `memory` lies on the host whose Storage
    /// Server listens at `endpoint`, and so does that rank's part of every
    /// host region, which its host's server created for it. For a host
    /// region `memory` names that host, as a rank there stated it
    /// ([`Self::place`]), with no run.
    pub(crate) fn place_remote(
        &mut self,
        id: ClientId,
        (deployment, epoch): (&str, u64),
        (region, memory, part): (&str, &str, u32),
        endpoint: &str,
        runs: &[UnitRun],
    ) -> Result<Handling, (ErrorKind, String)> {
        let engine = self.session_head(id, deployment, epoch)?;
        if endpoint.is_empty() {
            return Err(invalid(
                "a rank of another host names the endpoint of its host's Storage Server",
            ));
        }
        if let Some(run) = runs.iter().find(|run| run.count == 0 || run.bytes == 0) {
            return Err(invalid(&format!(
                "a run of {} holds no unit or no byte",
                run.field
            )));
        }
        let hosted = self
            .sessions
            .hosts
            .contains_key(&(deployment.to_owned(), region.to_owned()));
        if hosted && (memory.is_empty() || !runs.is_empty()) {
            return Err(invalid(&format!(
                "a place of the host region {region} names the host of its part, and no run"
            )));
        }
        let place = (
            deployment.to_owned(),
            region.to_owned(),
            memory.to_owned(),
            part,
        );
        let placed = (Some(endpoint.to_owned()), runs.to_vec());
        self.sessions.places.insert(place, placed);
        self.sessions
            .remote
            .entry(engine)
            .or_default()
            .insert((part, endpoint.to_owned()));
        Ok(Handling::Response(Response::Placed, Vec::new()))
    }

    /// See [`Request::HostPart`], from the worker `holder`.
    pub(crate) fn host_part(
        &mut self,
        holder: &Holder,
        (region, part): (&str, u32),
        bytes: u64,
        numa_node: Option<u32>,
    ) -> Result<Handling, (ErrorKind, String)> {
        if !self.current(holder) {
            return Err(crate::stale(holder));
        }
        let deployment = holder.scope.deployment.clone();
        let Some(&(limit, unit_bytes, _)) = self
            .sessions
            .hosts
            .get(&(deployment.clone(), region.to_owned()))
        else {
            return Err(invalid(&format!(
                "{region} is no host region a session of {deployment} stated"
            )));
        };
        let key = (deployment.clone(), region.to_owned(), part);
        if let Some(held) = self.sessions.parts.get(&key) {
            if held.bytes != bytes {
                return Err(invalid(&format!(
                    "part {part} of {region} holds {} bytes of a unit, not {bytes}",
                    held.bytes
                )));
            }
            return Ok(Handling::Response(
                Response::HostPart { units: limit },
                vec![Attachment::Storage(held.storage)],
            ));
        }
        let others: u64 = self
            .sessions
            .parts
            .iter()
            .filter(|((of, name, _), _)| *of == deployment && name == region)
            .map(|(_, part)| part.bytes)
            .sum();
        if bytes == 0 || others + bytes > unit_bytes {
            return Err(invalid(&format!(
                "{region}'s parts would hold {} of its {unit_bytes} bytes of a unit",
                others + bytes
            )));
        }

        // The part is admitted whole, the region's numbering of its bytes of
        // a unit: the other parts lie on their ranks' hosts, or are admitted as
        // their ranks here take them.
        let total = u64::from(limit) * bytes;
        self.admit_host(total)?;
        let descriptor = crate::memory::shared_memory("uniserve-host-region", total, numa_node)
            .map_err(resource)?;
        let storage = self.next_storage;
        self.next_storage += 1;
        self.storages.insert(
            storage,
            StorageEntry {
                record: StorageRecord {
                    location: Location::Host { numa_node },
                    bytes: total,
                    segment: None,
                    fabric: None,
                },
                _descriptor: descriptor,
                tensors: 0,
                moved_from: None,
                moving: false,
                part: true,
            },
        );
        let pool = self.sessions.next_pool;
        self.sessions.next_pool += 1;
        self.sessions.parts.insert(
            key,
            Part {
                storage,
                bytes,
                pool,
            },
        );
        self.offer_parts();
        tracing::info!(
            deployment,
            region,
            part,
            bytes = total,
            "created a part of a host region"
        );
        Ok(Handling::Response(
            Response::HostPart { units: limit },
            vec![Attachment::Storage(storage)],
        ))
    }

    /// Gives the connector every part of a host region it does not map yet.
    pub(crate) fn offer_parts(&mut self) {
        let Some(connector) = self.kv_storage.client else {
            return;
        };
        for ((deployment, region, rank), part) in &self.sessions.parts {
            if !self.kv_storage.offered.insert(part.pool) {
                continue;
            }
            let limit = self
                .sessions
                .hosts
                .get(&(deployment.clone(), region.clone()))
                .map_or(0, |(limit, ..)| *limit);
            self.events.push((
                connector,
                Event::Pool {
                    pool: part.pool,
                    shape: part_shape(region, *rank, part.bytes),
                    slots: u64::from(limit),
                },
                vec![Attachment::Storage(part.storage)],
            ));
        }
    }

    /// The storage of rank `rank`'s part of the host region `region` of
    /// `deployment`, with its bytes.
    pub(crate) fn part_storage(
        &self,
        deployment: &str,
        region: &str,
        rank: u32,
    ) -> Option<(StorageId, u64)> {
        let part = self
            .sessions
            .parts
            .get(&(deployment.to_owned(), region.to_owned(), rank))?;
        let storage = self.storages.get(&part.storage)?;
        Some((part.storage, storage.record.bytes))
    }

    /// Where the parts of the host region `region` of `deployment` lie, as
    /// far as this server knows, and how many parts a unit has. Each is
    /// stated by part (the index in their groups of the ranks that share it)
    /// with its bytes of a unit: this host's own parts, at no endpoint, and
    /// each other part on the host of every rank of that index on another
    /// host, at its Storage Server's endpoint, as the head of the engine
    /// whose session is `session` in the pool stated it (of every engine of
    /// the deployment when `None`).
    fn part_hosts(
        &self,
        deployment: &str,
        session: Option<PoolSession>,
        region: &str,
    ) -> PartHosts {
        let Some(&(_, unit_bytes, part_bytes)) = self
            .sessions
            .hosts
            .get(&(deployment.to_owned(), region.to_owned()))
        else {
            return (0, Vec::new());
        };
        let count = u32::try_from(unit_bytes / part_bytes.max(1)).unwrap_or(u32::MAX);

        let mut hosts: Vec<(u32, Option<String>, u64)> = self
            .sessions
            .parts
            .iter()
            .filter(|((of, name, _), _)| of == deployment && name == region)
            .map(|((_, _, part), held)| (*part, None, held.bytes))
            .collect();
        let engines: Vec<&Engine> = self
            .sessions
            .open
            .iter()
            .map(|(engine, (_, pool))| (engine, *pool))
            .chain(
                self.sessions
                    .gone
                    .iter()
                    .map(|(engine, (pool, _))| (engine, *pool)),
            )
            .filter(|((of, ..), pool)| of == deployment && session.is_none_or(|of| of == *pool))
            .map(|(engine, _)| engine)
            .collect();
        let elsewhere: BTreeSet<(u32, String)> = engines
            .into_iter()
            .filter_map(|engine| self.sessions.remote.get(engine))
            .flat_map(|parts| parts.iter().cloned())
            .filter(|(part, _)| *part < count && !hosts.iter().any(|(held, ..)| held == part))
            .collect();
        hosts.extend(
            elsewhere
                .into_iter()
                .map(|(part, endpoint)| (part, Some(endpoint), part_bytes)),
        );
        (count, hosts)
    }

    /// Where the host parts `named` of the host region `region` of
    /// `deployment` lie, stated as [`Self::part_hosts`] states parts: how
    /// many parts a unit has, and each named part a place states, by its
    /// index, with the endpoint of its host's Storage Server (none for this
    /// host) and its bytes of a unit. A host part lies on this host when a
    /// rank of it took the part and stated its host ([`Self::place`]), and on
    /// another when the head stated it for a rank there
    /// ([`Self::place_remote`]); a part no place states is left out. Without
    /// named parts (groups that state none), the parts of the region where
    /// the ranks of the engine whose session is `session` lie.
    fn parts_at(
        &self,
        deployment: &str,
        session: Option<PoolSession>,
        region: &str,
        named: &[HostPart],
    ) -> PartHosts {
        if named.is_empty() {
            return self.part_hosts(deployment, session, region);
        }
        let Some(&(_, unit_bytes, part_bytes)) = self
            .sessions
            .hosts
            .get(&(deployment.to_owned(), region.to_owned()))
        else {
            return (0, Vec::new());
        };
        let count = u32::try_from(unit_bytes / part_bytes.max(1)).unwrap_or(u32::MAX);

        let placed = |(host, part): &HostPart| {
            let place = (
                deployment.to_owned(),
                region.to_owned(),
                host.clone(),
                *part,
            );
            let (at, _) = self.sessions.places.get(&place)?;
            let bytes = match at {
                None => {
                    let held = (deployment.to_owned(), region.to_owned(), *part);
                    self.sessions.parts.get(&held)?.bytes
                }
                Some(_) => part_bytes,
            };
            Some((*part, at.clone(), bytes))
        };
        let parts = named
            .iter()
            .filter(|(_, part)| *part < count)
            .filter_map(placed)
            .collect();
        (count, parts)
    }

    /// See [`Request::Pages`], from the head `id` of `deployment`, which
    /// runs a task of its buffer pool on another host.
    pub(crate) fn pages_for(
        &mut self,
        id: ClientId,
        deployment: &str,
        (region, part): (&str, u32),
        pages: &[(Key, u32)],
        upload: bool,
    ) -> Result<Handling, (ErrorKind, String)> {
        if self.kv_storage.client.is_none() {
            return Err((
                ErrorKind::Unavailable,
                "no connector is connected".to_owned(),
            ));
        }
        let held = self
            .sessions
            .parts
            .get(&(deployment.to_owned(), region.to_owned(), part));
        let Some(pool) = held.map(|held| held.pool) else {
            return Err(invalid(&format!(
                "this host holds no part {part} of a host region {region} of {deployment}"
            )));
        };
        let limit = self
            .sessions
            .hosts
            .get(&(deployment.to_owned(), region.to_owned()))
            .map_or(0, |(limit, ..)| *limit);
        if let Some((_, unit)) = pages.iter().find(|(_, unit)| *unit >= limit) {
            return Err(invalid(&format!(
                "unit {unit} lies past the {limit} units of {region}"
            )));
        }

        self.offer_parts();
        let task = self.assign(Task {
            owner: Owner::Client(id),
            pool,
            pages: pages.to_vec(),
            upload,
        });
        Ok(Handling::Response(Response::Submitted { task }, Vec::new()))
    }

    /// See [`Request::PeerHold`], from the peer `id`. The peer reads every
    /// copy of a region from the same parts, one of each part index: those
    /// the region's first held copy lies in, this host's where it holds one,
    /// or for a copy of groups that state no host parts every part of the
    /// region this server knows. A later copy that does not lie in all of
    /// them is left out of the answer, as a key without a copy is: the peer
    /// reads a host copy only from host parts that hold it.
    pub(crate) fn peer_hold(
        &mut self,
        id: ClientId,
        deployment: &str,
        keys: &[Key],
    ) -> Result<Handling, (ErrorKind, String)> {
        let (named, held) = (deployment.to_owned(), keys.to_vec());
        let held = self
            .sessions
            .thread
            .call(move |pools| pools.hold(&named, id, &held))
            .unwrap_or_default();

        // Per region, whether its copies lie in named host parts, and the
        // parts the peer reads them from.
        let mut read: BTreeMap<String, (bool, Vec<PartHost>)> = BTreeMap::new();
        let mut copies = Vec::new();
        for (copy, holding) in held {
            let (_, lies) = self.parts_at(deployment, None, &copy.region, &holding);
            let named = !holding.is_empty();
            let (stated, parts) = read.entry(copy.region.clone()).or_insert_with(|| {
                let mut parts = lies.clone();
                if named {
                    one_of_each(&mut parts);
                }
                (named, parts)
            });
            if *stated == named && parts.iter().all(|part| lies.contains(part)) {
                copies.push(copy);
            }
        }
        let parts = read
            .into_iter()
            .flat_map(|(region, (_, parts))| {
                parts
                    .into_iter()
                    .map(move |(rank, endpoint, bytes)| RegionPart {
                        region: region.clone(),
                        rank,
                        bytes,
                        endpoint,
                    })
            })
            .collect();
        Ok(Handling::Response(
            Response::PeerHeld { copies, parts },
            Vec::new(),
        ))
    }

    /// See [`Request::PeerRelease`], from the peer `id`.
    pub(crate) fn peer_release(
        &mut self,
        id: ClientId,
        deployment: &str,
    ) -> Result<Handling, (ErrorKind, String)> {
        let named = deployment.to_owned();
        let held = self
            .sessions
            .thread
            .call(move |pools| pools.release_peer(&named, id))
            .unwrap_or(false);
        let response = if held {
            Response::Valid
        } else {
            Response::Gone
        };
        Ok(Handling::Response(response, Vec::new()))
    }

    /// A client's connection closed: a peer's holds end, and the session
    /// an engine's main process opened waits for the engine's processes to
    /// leave.
    pub(crate) fn sessions_client_gone(&mut self, id: ClientId, client: &Client) {
        match &client.role {
            Some(Role::Peer) => self.sessions.thread.send(move |pools| pools.peer_gone(id)),
            Some(Role::Head {
                deployment,
                epoch,
                generation,
            }) => {
                let engine = (deployment.clone(), *epoch, *generation);
                if let Some(&(head, pool)) = self.sessions.open.get(&engine)
                    && head == id
                {
                    self.follow_engine(engine, pool, client);
                }
            }
            _ => {}
        }
    }

    /// Follows the processes of `engine`, whose main process's connection
    /// closed and whose session is `pool` in its deployment's pool: that
    /// process, every worker of the engine (of its epoch and its launch
    /// generation) connected, and every process of the engine that holds
    /// access to a tensor of the epoch. Each is out once it has exited and no
    /// GPU the epoch's storage lies on lists it. Another engine of the epoch,
    /// its standby, is not followed: it serves on.
    fn follow_engine(&mut self, engine: Engine, pool: PoolSession, opener: &Client) {
        let (deployment, epoch, generation) = &engine;
        let mut processes: Vec<(i32, Arc<OwnedFd>)> = Vec::new();
        if let Some(pidfd) = &opener.pidfd {
            processes.push((opener.pid, Arc::clone(pidfd)));
        }
        for client in self.clients.values() {
            if let (
                Some(Role::Worker {
                    scope,
                    epoch: of,
                    generation: launched,
                    ..
                }),
                Some(pidfd),
            ) = (&client.role, &client.pidfd)
                && scope.deployment == *deployment
                && of == epoch
                && launched == generation
            {
                processes.push((client.pid, Arc::clone(pidfd)));
            }
        }
        let mut gpus = BTreeSet::new();
        for tensor in self.tensors.values() {
            let of_engine = |holder: &Holder| {
                holder.scope.deployment == *deployment
                    && holder.epoch == *epoch
                    && holder.generation == *generation
            };
            let holders = tensor.writer.iter().chain(&tensor.standby);
            processes.extend(
                holders
                    .filter(|holder| of_engine(holder))
                    .map(|holder| (holder.pid, Arc::clone(&holder.pidfd))),
            );
            let retained = tensor.retainers.iter().any(|retainer| {
                retainer.scope.deployment == *deployment && retainer.epoch == *epoch
            });
            if retained
                && let Some(storage) = self.storages.get(&tensor.storage)
                && let Location::Device { gpu } = storage.record.location
            {
                gpus.insert(gpu);
            }
        }
        let mut followed: Vec<(i32, Arc<OwnedFd>)> = Vec::new();
        for (pid, pidfd) in processes {
            if !followed.iter().any(|(_, known)| Arc::ptr_eq(known, &pidfd)) {
                gpus.extend(self.gpus_held_by(&pidfd));
                followed.push((pid, pidfd));
            }
        }
        // Each process is out once no GPU of the epoch's storage lists it.
        for (pid, pidfd) in &followed {
            let gpus = gpus.iter().copied().collect();
            self.shootdowns.follow(*pid, Arc::clone(pidfd), gpus);
        }
        let followed: Vec<Arc<OwnedFd>> = followed.into_iter().map(|(_, pidfd)| pidfd).collect();
        tracing::info!(
            deployment,
            epoch,
            generation,
            processes = followed.len(),
            "an engine's main process left; its session closes once its processes are out"
        );
        self.sessions.open.remove(&engine);
        self.sessions.gone.insert(engine, (pool, followed));
    }

    /// Closes every session whose engine's processes are all out.
    pub(crate) fn settle_sessions(&mut self) {
        let out: Vec<Engine> = self
            .sessions
            .gone
            .iter()
            .filter(|(_, (_, processes))| {
                processes
                    .iter()
                    .all(|pidfd| self.shootdowns.finished(pidfd))
            })
            .map(|(engine, _)| engine.clone())
            .collect();
        for engine in out {
            let Some((pool, _)) = self.sessions.gone.remove(&engine) else {
                continue;
            };
            self.sessions.remote.remove(&engine);
            let rolled = self.epoch_ends(&engine);
            let (deployment, epoch, generation) = engine;
            tracing::info!(
                deployment,
                epoch,
                generation,
                "an engine's processes are out; its session closes"
            );
            self.sessions.thread.send(move |pools| {
                pools.close(&deployment, pool);
                if rolled {
                    pools.roll_back(&deployment, epoch);
                }
            });
        }
    }

    /// The head published `epoch` of `deployment`: its update is committed,
    /// and what it made in the buffer pool is every epoch's.
    pub(crate) fn update_committed(&mut self, deployment: &str, epoch: u64) {
        let named = deployment.to_owned();
        self.sessions
            .thread
            .send(move |pools| pools.commit_update(&named, epoch));
    }

    /// The head retired `epoch` of `deployment` on trial: its update is
    /// rolled back, and what it made in the buffer pool is evicted once its
    /// last session is closed.
    pub(crate) fn update_rolled_back(&mut self, deployment: &str, epoch: u64) {
        let session = (deployment.to_owned(), epoch);
        let held = self
            .sessions
            .open
            .keys()
            .chain(self.sessions.gone.keys())
            .any(|(of, at, _)| of == deployment && *at == epoch);
        if held {
            self.sessions.rolled.insert(session);
            return;
        }
        let named = deployment.to_owned();
        self.sessions
            .thread
            .send(move |pools| pools.roll_back(&named, epoch));
    }

    /// Runs the notices of the buffer pool's thread.
    pub(crate) fn take_notices(&mut self) {
        for notice in self.sessions.thread.notices() {
            notice(self);
        }
    }

    /// A pool's task starts: a transfer goes to the Transfer Servers of the
    /// hosts its destinations lie on, a prefetch from the remote store and
    /// a put to the connector, a prefetch from the peers to the peers its
    /// session names; whatever cannot run ends at once as failed.
    pub(crate) fn start_pool_task(
        &mut self,
        session: Session,
        task: PoolTask,
        source: Option<Source>,
        copies: Vec<Copied>,
    ) {
        let number = self.sessions.next;
        self.sessions.next += 1;
        let keys: Vec<(Key, bool)> = copies
            .iter()
            .filter_map(|(key, ..)| key.map(|key| (key, true)))
            .collect();
        let deployment = session.0.clone();
        let running = Running::new(session.clone(), task, keys);
        self.sessions.running.insert(number, running);
        let started = if copies.iter().all(|(key, ..)| key.is_none()) {
            self.transfer_blocks(&session, number, &copies)
        } else if copies.iter().all(|(_, reads, ..)| reads.is_empty()) {
            match source {
                Some(Source::Remote) => self.move_pages(&session, number, &copies, false),
                Some(Source::Peers) | None => self.read_from_peers(&session, number, &copies),
            }
        } else {
            self.move_pages(&session, number, &copies, true)
        };
        match started {
            Ok(true) => {}
            // Nothing to move: the task has ended.
            Ok(false) => self.end_pool_task(number, true),
            Err(reason) => {
                tracing::warn!(deployment, %reason, "a buffer pool task cannot run");
                self.end_pool_task(number, false);
            }
        }
    }

    /// The ranks of a deployment's memory that placed a region's units,
    /// by rank index, with where their units lie.
    fn ranks(&self, deployment: &str, region: &str, memory: &str) -> BTreeMap<u32, &Placed> {
        self.sessions
            .places
            .iter()
            .filter(|((of, name, at, _), _)| of == deployment && name == region && at == memory)
            .map(|((_, _, _, rank), placed)| (*rank, placed))
            .collect()
    }

    /// A transfer's or a crossing's spans as one task of the Transfer Server
    /// of each host a destination lies on: this host's, or another's through
    /// its Storage Server ([`hosts::transfer_at`]). A run a copy writes is
    /// filled from the same run in a memory that holds it (a transfer, and a
    /// crossing of a region whose units exist in every memory that maps it),
    /// and otherwise from the copy's one read run: a crossing reads the run
    /// its writer wrote in one main process's product buffer and writes the
    /// block's run in another's, which lie in different regions. The ranks of
    /// the two memories are matched by their part. A source on another host
    /// than its destination's is read from there. Returns whether the task
    /// moves bytes.
    fn transfer_blocks(
        &mut self,
        session: &Session,
        number: u64,
        copies: &[Copied],
    ) -> Result<bool, String> {
        let deployment = session.0.as_str();
        // By the host of the destinations, `None` for this one: the spans,
        // each with whether its source is storage of this host that
        // another host reads.
        let mut tasks: BTreeMap<Option<String>, Vec<(Span, bool)>> = BTreeMap::new();
        for (_, reads, writes, (from, _), (to, _)) in copies {
            let Some(to) = to.first() else {
                // The slot row: memory of the ranks' own processes.
                continue;
            };
            for write in writes {
                let (region, first, count) = write;
                let read = match (reads.iter().find(|read| *read == write), reads.as_slice()) {
                    (Some(read), _) | (None, [read]) => read,
                    _ => return Err(format!("no run the copy reads fills {region}'s")),
                };
                let (source_region, source_first, source_count) = read;
                if source_count != count {
                    return Err(format!(
                        "{count} units of {region} would hold {source_count} of {source_region}"
                    ));
                }

                // The content is read from a memory that holds it and whose
                // ranks placed the region's units.
                let destinations = self.ranks(deployment, region, to);
                let source = from
                    .iter()
                    .map(|memory| (memory, self.ranks(deployment, source_region, memory)))
                    .find(|(_, sources)| !sources.is_empty());
                let Some((from, sources)) = source else {
                    // A run of the ranks' own processes that no rank placed
                    // in either memory moves no bytes; a run filled from
                    // another region's (a crossing into a product buffer)
                    // moves its bytes, so both are placed.
                    if destinations.is_empty() && read == write {
                        continue;
                    }
                    return Err(format!(
                        "no rank of a memory that holds {source_region}'s units placed them"
                    ));
                };
                if destinations.is_empty() {
                    return Err(format!("no rank of {to} placed the units of {region}"));
                }
                for (rank, (at, destination)) in destinations {
                    let (source_at, source) = sources.get(&rank).ok_or_else(|| {
                        format!("rank {rank} of {from} placed no unit of {source_region}")
                    })?;
                    let firsts = (*source_first, *first);
                    for mut span in spans(source, destination, firsts, *count)? {
                        let here = match (source_at, at) {
                            (source_at, at) if source_at == at => false,
                            (Some(endpoint), _) => {
                                span.source = Region::Remote {
                                    endpoint: endpoint.clone(),
                                    region: Box::new(span.source),
                                };
                                false
                            }
                            (None, _) => true,
                        };
                        tasks.entry(at.clone()).or_default().push((span, here));
                    }
                }
            }
        }
        if tasks.is_empty() {
            return Ok(false);
        }
        self.spread(session, number, tasks)?;
        Ok(true)
    }

    /// Runs the spans of the pool task `number` of `session` as one task of
    /// the Transfer Server of each host their destinations lie on (`tasks`,
    /// by host, `None` for this one, each span flagged when its source is
    /// storage of this host): this host's at once, another's through its
    /// Storage Server ([`hosts::transfer_at`]). The pool task ends with the
    /// last of them; one that cannot start has ended as failed.
    ///
    /// # Errors
    ///
    /// This host's part refused, when nothing was started.
    fn spread(
        &mut self,
        session: &Session,
        number: u64,
        mut tasks: BTreeMap<Option<String>, Vec<(Span, bool)>>,
    ) -> Result<(), String> {
        let deployment = session.0.as_str();
        let local = tasks.remove(&None);
        let parts = tasks.len() + usize::from(local.is_some());
        if let Some(spans) = local {
            let spans: Vec<Span> = spans.into_iter().map(|(span, _)| span).collect();
            let owner = crate::transfers::Owner::Pool { number };
            self.submit_for(owner, deployment, &spans, Priority::Request, 0)
                .map_err(|(_, message)| message)?;
        }
        if let Some(running) = self.sessions.running.get_mut(&number) {
            running.waiting = parts;
        }
        let port = self
            .tcp
            .as_ref()
            .and_then(|(_, address)| address.rsplit_once(':'))
            .and_then(|(_, port)| port.parse().ok());
        let mut failed = 0;
        for (endpoint, spans) in tasks {
            let Some(endpoint) = endpoint else {
                continue;
            };
            let started = hosts::Stop::new().and_then(|(cancel, cancelled)| {
                hosts::transfer_at(
                    self.sessions.thread.notifier(),
                    number,
                    endpoint,
                    session.clone(),
                    spans,
                    port,
                    (cancelled, self.sessions.transfer_deadline),
                )
                .map(|()| cancel)
            });
            match started {
                Ok(cancel) => {
                    if let Some(running) = self.sessions.running.get_mut(&number) {
                        running.cancels.push(cancel);
                    }
                }
                Err(error) => {
                    tracing::warn!(deployment, %error, "a transfer on another host cannot start");
                    failed += 1;
                }
            }
        }
        for _ in 0..failed {
            self.end_pool_task(number, false);
        }
        Ok(())
    }

    /// A prefetch from the peers the session names: a helper holds the
    /// keys there ([`hosts::read_from_peers`]), then
    /// [`Self::peers_held`] reads what they hold. Returns whether there is
    /// a peer to ask.
    fn read_from_peers(
        &mut self,
        session: &Session,
        number: u64,
        copies: &[Copied],
    ) -> Result<bool, String> {
        let endpoints = self
            .sessions
            .peers
            .get(session)
            .cloned()
            .unwrap_or_default();
        if endpoints.is_empty() {
            return Ok(false);
        }
        let keys = copies.iter().filter_map(|(key, ..)| *key).collect();
        let (release, released) = mpsc::channel();
        let notifier = self.sessions.thread.notifier();
        hosts::read_from_peers(
            notifier,
            number,
            session.0.clone(),
            endpoints,
            keys,
            released,
        )
        .map_err(|error| format!("a read from the peers cannot start: {error}"))?;
        if let Some(running) = self.sessions.running.get_mut(&number) {
            running.release = Some(release);
            running.parts = copies
                .iter()
                .map(|(.., (_, parts))| parts.clone())
                .collect();
        }
        Ok(true)
    }

    /// The peers hold what they hold of a read's keys: the buffer pool
    /// chooses the block each leading copy held is read into by the encoding
    /// of the peer's copy ([`crate::buffer_pool::BufferPool::peer_copies`]),
    /// and the Transfer Server of each part's host reads the copies into
    /// them, in each host part the copy fills (in every part of the region,
    /// this host's and those of the session's ranks of other hosts, for
    /// groups that state no host parts), from the peer's part of the same
    /// index, wherever that lies, which must hold the same bytes of a unit,
    /// and the copy the same units ([`read_spans`]). A copy with a host part
    /// to fill that no place states is not read. With nothing to read, the
    /// holds are released at once.
    pub(crate) fn peers_held(&mut self, number: u64, (sources, peers): hosts::Held) {
        let Some(running) = self.sessions.running.get(&number) else {
            return;
        };
        let session = running.session.clone();
        let task = running.task;
        // The engine whose session submitted the task: where its ranks' parts lie.
        let pool = task.0.0;
        let parts = if running.stopped {
            Vec::new()
        } else {
            running.parts.clone()
        };
        let here = self.tcp.as_ref().map(|(_, address)| address.as_str());

        // The block each copy is read into follows the encoding of the peer's
        // copy, which the buffer pool weighs against its class.
        let encodings: Vec<Option<Encoding>> = sources
            .iter()
            .take(parts.len())
            .map(|source| {
                source
                    .as_ref()
                    .map(|(_, copy)| (copy.tensors, copy.encoding))
            })
            .collect();
        let writes = if encodings.is_empty() {
            Vec::new()
        } else {
            let deployment = session.0.clone();
            self.sessions
                .thread
                .call(move |pools| pools.peer_copies(&deployment, task, &encodings))
                .unwrap_or_default()
        };

        // By the host of the parts they write, `None` for this one: the spans,
        // each with whether its source is storage of this host.
        let mut tasks: BTreeMap<Option<String>, Vec<(Span, bool)>> = BTreeMap::new();
        let mut reads = Vec::new();
        for (index, (write, named)) in writes.iter().zip(&parts).enumerate() {
            let Some(Some((connection, copy))) = sources.get(index) else {
                break;
            };
            let Some((peer, held)) = peers.get(*connection) else {
                break;
            };
            let destinations = self.parts_at(&session.0, Some(pool), &write.0, named);
            if destinations.1.len() < named.len() {
                break;
            }
            let Some(spans) = read_spans(copy, (peer, held), write, &destinations, here) else {
                break;
            };
            for (host, span, from_here) in spans {
                tasks.entry(host).or_default().push((span, from_here));
            }
            reads.push((*connection, copy.tokens));
        }

        if let Some(running) = self.sessions.running.get_mut(&number) {
            running.reads = reads;
        }
        let held = sources.iter().take(parts.len()).flatten().count();
        if tasks.is_empty() && held > 0 {
            tracing::info!(
                deployment = session.0,
                held,
                "the peers hold no copy of the prefetch's leading keys that its class reads"
            );
        }
        let started = if tasks.is_empty() {
            Err(None)
        } else {
            self.spread(&session, number, tasks).map_err(Some)
        };
        if let Err(failure) = started {
            if let Some(message) = failure {
                tracing::warn!(deployment = session.0, %message, "a peer's host copies cannot be read");
            }
            if let Some(running) = self.sessions.running.get_mut(&number) {
                running.ok = false;
                running.release = None;
            }
        }
    }

    /// A read from the peers released its holds: `valid` says, per peer
    /// connection, whether its holds stood until then. The copies read are
    /// cached, in order, up to the first read from a peer whose holds did
    /// not stand, or not read whole; the task then ends.
    pub(crate) fn peers_released(&mut self, number: u64, valid: Vec<bool>) {
        let Some(running) = self.sessions.running.remove(&number) else {
            return;
        };
        let stood = |connection: &usize| valid.get(*connection).copied().unwrap_or(false);
        let done: Vec<u16> = if running.ok {
            running
                .reads
                .iter()
                .map_while(|(connection, tokens)| stood(connection).then_some(*tokens))
                .collect()
        } else {
            Vec::new()
        };
        let whole = done.len() == running.keys.len();
        let ((deployment, _), task) = (running.session, running.task);
        self.sessions.thread.send(move |pools| {
            for tokens in done {
                pools.done(&deployment, task, tokens);
            }
            if !whole {
                pools.ended(&deployment, task, false);
            }
        });
    }

    /// A prefetch's or a put's pages as one connector task per part of the
    /// host region: this host's connector moves the pages of its own parts,
    /// and the connector of the host of each part of a rank of another host
    /// those of that part, asked there by a helper ([`hosts::pages_at`]). A
    /// prefetch fills the host parts its copy names, a put reads each part
    /// index from one host part that holds the copy, this host's where it
    /// can ([`Self::parts_at`]); a copy of groups that state no host parts
    /// moves every part of the region where the session's ranks lie.
    /// Returns whether there is a page to move.
    ///
    /// # Errors
    ///
    /// No connector here, a host part a prefetch fills that no place
    /// states, or a part index of a copy's region that lies on no host this
    /// server knows: no copy would move whole.
    fn move_pages(
        &mut self,
        session: &Session,
        number: u64,
        copies: &[Copied],
        upload: bool,
    ) -> Result<bool, String> {
        if self.kv_storage.client.is_none() {
            return Err("no connector is connected".into());
        }
        let deployment = session.0.as_str();
        // The engine whose session submitted the task: where its ranks' parts lie.
        let engine = self
            .sessions
            .running
            .get(&number)
            .map(|running| running.task.0.0);
        // Per part of this host, its pool and its pages with the copy each is
        // of; per other host, the same per region and part.
        let mut pages: BTreeMap<u64, Pages> = BTreeMap::new();
        let mut elsewhere: BTreeMap<String, BTreeMap<(String, u32), Pages>> = BTreeMap::new();
        for (index, (key, _, writes, (_, holding), (_, filled))) in copies.iter().enumerate() {
            let (Some(key), Some((region, first, count))) = (key, writes.first()) else {
                continue;
            };
            let named = if upload { holding } else { filled };
            let (parts, mut hosts) = self.parts_at(deployment, engine, region, named);
            if parts == 0 {
                return Err(format!("{region} is no host region of {deployment}"));
            }
            if !upload && hosts.len() < named.len() {
                return Err(format!(
                    "a part of {region} the copy fills is placed nowhere"
                ));
            }
            if upload && !named.is_empty() {
                one_of_each(&mut hosts);
            }
            if let Some(part) = (0..parts).find(|part| hosts.iter().all(|(at, ..)| at != part)) {
                return Err(format!(
                    "part {part} of {region} lies on no host this server knows"
                ));
            }
            for (part, host, _) in hosts {
                let (keys, of) = match host {
                    None => {
                        let held = (deployment.to_owned(), region.clone(), part);
                        let Some(pool) = self.sessions.parts.get(&held).map(|held| held.pool)
                        else {
                            continue;
                        };
                        pages.entry(pool).or_default()
                    }
                    Some(endpoint) => elsewhere
                        .entry(endpoint)
                        .or_default()
                        .entry((region.clone(), part))
                        .or_default(),
                };
                for unit in 0..*count {
                    keys.push((unit_key(key, unit), first + unit));
                    of.push(index);
                }
            }
        }
        if pages.is_empty() && elsewhere.is_empty() {
            return Ok(false);
        }

        self.offer_parts();
        let parts = pages.len() + elsewhere.values().map(BTreeMap::len).sum::<usize>();
        if let Some(running) = self.sessions.running.get_mut(&number) {
            running.waiting = parts;
        }
        for (pool, (pages, keys)) in pages {
            self.assign(Task {
                owner: Owner::Pool { number, keys },
                pool,
                pages,
                upload,
            });
        }
        for (endpoint, parts) in elsewhere {
            let parts: Vec<(String, u32, Pages)> = parts
                .into_iter()
                .map(|((region, part), pages)| (region, part, pages))
                .collect();
            let keys: Vec<Vec<usize>> = parts.iter().map(|(.., (_, of))| of.clone()).collect();
            let started = hosts::Stop::new().and_then(|(stop, stopped)| {
                hosts::pages_at(
                    self.sessions.thread.notifier(),
                    number,
                    endpoint,
                    (session.clone(), upload),
                    parts,
                    (stopped, self.sessions.put_deadline),
                )
                .map(|()| stop)
            });
            match started {
                Ok(stop) => {
                    if let Some(running) = self.sessions.running.get_mut(&number) {
                        running.cancels.push(stop);
                    }
                }
                Err(error) => {
                    tracing::warn!(deployment, %error, "pages on another host cannot be moved");
                    for keys in keys {
                        self.pool_pages_moved(number, &keys, &[]);
                    }
                }
            }
        }
        Ok(true)
    }

    /// The connector moved the pages of a pool task's part, `results` one
    /// per page, each of the copy `keys` names.
    pub(crate) fn pool_pages_moved(&mut self, number: u64, keys: &[usize], results: &[bool]) {
        let Some(running) = self.sessions.running.get_mut(&number) else {
            return;
        };
        for (page, copy) in keys.iter().enumerate() {
            if !results.get(page).copied().unwrap_or(false)
                && let Some((_, moved)) = running.keys.get_mut(*copy)
            {
                *moved = false;
            }
        }
        running.waiting = running.waiting.saturating_sub(1);
        if running.waiting > 0 {
            return;
        }
        let Some(running) = self.sessions.running.remove(&number) else {
            return;
        };
        // The keys done, in order, up to the first that did not move whole.
        let done = running.keys.iter().take_while(|(_, moved)| *moved).count();
        let whole = done == running.keys.len();
        let ((deployment, _), task) = (running.session, running.task);
        self.sessions.thread.send(move |pools| {
            for _ in 0..done {
                pools.done(&deployment, task, BLOCK_TOKENS);
            }
            if !whole {
                pools.ended(&deployment, task, false);
            }
        });
    }

    /// A part of a pool's task ended on a child or on another host, which
    /// touches its blocks no more: the task ends with its last part, as
    /// having copied everything when every part did. A read from the peers
    /// releases its holds first ([`Self::peers_released`]).
    pub(crate) fn end_pool_task(&mut self, number: u64, ok: bool) {
        let Some(running) = self.sessions.running.get_mut(&number) else {
            return;
        };
        running.ok &= ok;
        running.waiting = running.waiting.saturating_sub(1);
        if running.waiting > 0 {
            return;
        }
        if running.release.is_some() {
            // The helper releases the holds once the sender is gone.
            running.release = None;
            return;
        }
        let Some(running) = self.sessions.running.remove(&number) else {
            return;
        };
        let ((deployment, _), task, ok) = (running.session, running.task, running.ok);
        self.sessions
            .thread
            .send(move |pools| pools.ended(&deployment, task, ok));
    }

    /// The child process `pid` that ran a pool's task lost its connection:
    /// the task ends as failed once that process has exited.
    pub(crate) fn pool_task_lost(&mut self, number: u64, pid: Option<u32>) {
        match pid {
            Some(pid) => self.sessions.lost.push((pid, number)),
            None => self.end_pool_task(number, false),
        }
    }

    /// Ends the lost tasks whose child process has exited: the pools'
    /// tasks, and the connector tasks of heads of other hosts, whose pages
    /// all failed.
    pub(crate) fn settle_lost(&mut self) {
        let running = [self.transfers.process.pid(), self.kv_storage.process.pid()];
        let (still, out): (Vec<_>, Vec<_>) = std::mem::take(&mut self.sessions.lost)
            .into_iter()
            .partition(|(pid, _)| running.contains(&Some(*pid)));
        self.sessions.lost = still;
        for (_, number) in out {
            self.end_pool_task(number, false);
        }

        let connector = self.kv_storage.process.pid();
        let (still, out): (Vec<_>, Vec<_>) = std::mem::take(&mut self.kv_storage.lost)
            .into_iter()
            .partition(|(pid, ..)| connector == Some(*pid));
        self.kv_storage.lost = still;
        for (_, client, task, pages) in out {
            let results = vec![false; pages];
            let event = Event::PagesMoved { task, results };
            self.events.push((client, event, Vec::new()));
        }
    }

    /// A pool's task is past its deadline or its session closed: its
    /// transfers here and on other hosts are cancelled, and what it has not
    /// started it does not; the connector's runs to its end, and another
    /// host's connector's is waited for a put's deadline more.
    pub(crate) fn stop_pool_task(&mut self, deployment: &str, task: PoolTask) {
        let found = self
            .sessions
            .running
            .iter_mut()
            .find(|(_, running)| running.session.0 == deployment && running.task == task);
        let Some((number, running)) = found else {
            return;
        };
        running.stopped = true;
        for cancel in &running.cancels {
            cancel.stop();
        }
        let number = *number;
        self.cancel_pool_transfer(number);
    }

    /// The deployment is no longer served: its pool goes with every session,
    /// and so do its host regions' parts and the places of its units.
    pub(crate) fn release_pools(&mut self, deployment: &str) {
        let named = deployment.to_owned();
        self.sessions
            .thread
            .send(move |pools| pools.release(&named));
        self.sessions.open.retain(|(of, ..), _| of != deployment);
        self.sessions.gone.retain(|(of, ..), _| of != deployment);
        self.sessions.hosts.retain(|(of, _), _| of != deployment);
        self.sessions.places.retain(|(of, ..), _| of != deployment);
        self.sessions.peers.retain(|(of, _), _| of != deployment);
        self.sessions.remote.retain(|(of, ..), _| of != deployment);
        let parts: Vec<Part> = self
            .sessions
            .parts
            .iter()
            .filter(|((of, ..), _)| of == deployment)
            .map(|(_, part)| *part)
            .collect();
        self.sessions.parts.retain(|(of, ..), _| of != deployment);
        for part in parts {
            self.storages.remove(&part.storage);
            self.allocation_released(Allocation::Storage {
                storage: part.storage,
            });
            if self.kv_storage.offered.remove(&part.pool)
                && let Some(connector) = self.kv_storage.client
            {
                self.events.push((
                    connector,
                    Event::PoolReleased { pool: part.pool },
                    Vec::new(),
                ));
            }
        }
    }
}
