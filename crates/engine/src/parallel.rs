//! Cross-rank coordination off the device: the host collectives of a rank group.
//!
//! A [`HostGroup`] connects every member of a rank group to every other member once, when the
//! group forms, and runs small host-side collectives over those connections: byte gathers,
//! fixed-width record gathers, byte broadcasts and integer reductions. It carries startup
//! agreement (block sizes, cache capacity, slot fits), the comparison of the ranks' buckets,
//! warm-up broadcasts, the bootstrap bytes of device communicators (NCCL unique ids) and an expert
//! union's agreement on each step. Device collectives never use it.
//!
//! **Connections.** Every pair of members holds one connection of the workspace's framed kind
//! (`service::Connection`). Two members that are threads of one process are connected by a socket
//! pair the process creates ([`HostGroup::local`]): nothing outside the process can reach it, and
//! no listener exists for it. Two members of different processes, which a group whose ranks span
//! several hosts has, are connected over TCP ([`HostGroup::world`]).
//!
//! **Forming a group across processes.** The process that holds member 0 serves the group's
//! rendezvous on the listener its daemon reserved before the process existed, so the endpoint
//! every process was given is live from the start. Every other process binds a listener of its
//! own before it announces it, connects to the rendezvous once and states the members it holds
//! and that listener ([`Formation::Join`]). Once every member is stated, the rendezvous answers
//! every process with the endpoint of each member's process ([`Formation::Table`]). Each member
//! then connects once to every lower member another process holds, opening the connection with
//! the two members it joins ([`Formation::Pair`]), and each process accepts the connections of
//! the higher members on its listener. A process connects on a thread of its own while it
//! accepts, so no two processes wait on each other's backlog. Every frame states the build and
//! the epoch: a frame of another build or epoch fails the formation by name. A connection whose
//! first frame is no frame of the formation is closed, and it holds up nothing: the formation
//! reads every connection as its bytes arrive.
//!
//! **Waits.** The formation waits for processes that may not exist yet, so it is bounded by the
//! epoch's join bound (`worker_config.join_timeout`). Once formed, a collective waits for the
//! other members without a bound of its own: a member whose process exits closes its
//! connections, and a TCP connection fails once the peer's host stops answering
//! (`service::Liveness`), so the collective fails with the connection.
//!
//! **Collective ordering.** Every member must call the same collectives in the same order, and one
//! member runs one collective at a time. A collective queues its sends on every connection and
//! reads the others' values while the sends drain, so members that send to each other at once
//! never wait on one another, whatever the size of the values.

use std::net::TcpListener;
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::net::UnixStream;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use service::{Connection, Endpoint, Listener, lock};

use crate::worker::{Error, Result};

/// One member's connections to every other member of a group.
pub(crate) struct HostGroup {
    /// This member's index in the group.
    rank: usize,
    /// How many members the group has.
    size: usize,
    /// The connection to each member by index; `None` at this member's own index. One collective
    /// holds them at a time.
    peers: Mutex<Vec<Option<Connection>>>,
}

/// Where the members of a group whose ranks span several processes meet, and what admits a
/// process there.
pub(crate) struct Rendezvous {
    /// The socket the daemon reserved for the rendezvous, which the process that holds member 0
    /// inherited and serves; `None` in every other process.
    pub(crate) listener: Option<TcpListener>,
    /// The rendezvous's endpoint, as every process of the group was given it (`host:port`).
    pub(crate) endpoint: String,
    /// The epoch the group's ranks serve, by the engine number of its scheduler node: a frame of
    /// another epoch is refused.
    pub(crate) epoch: u64,
    /// How long the processes wait for one another to form the group
    /// (`worker_config.join_timeout`).
    pub(crate) timeout: Duration,
}

impl HostGroup {
    /// Every member of a group of `size` members that are all threads of this process, in member
    /// order: each pair of members is connected by a socket pair. A group of one member has no
    /// connection: a rank that serves alone still agrees its block size and slot count over a
    /// group, so capacity fitting takes one shape for every deployment size.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a group of no member, and a transport error when a socket pair
    /// cannot be made.
    pub(crate) fn local(size: usize) -> Result<Vec<Self>> {
        if size == 0 {
            return Err(Error::Invalid(
                "a host group has at least one member".into(),
            ));
        }

        Ok(socket_pairs(size)?
            .into_iter()
            .enumerate()
            .map(|(rank, peers)| Self {
                rank,
                size,
                peers: Mutex::new(peers),
            })
            .collect())
    }

    /// The members `held` of a group of `size` members whose others are held by other processes,
    /// in the order of `held`, formed at `rendezvous` (module documentation). Two members this
    /// process holds are connected by a socket pair.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] when `held` names a member outside the group or twice, or the
    /// rendezvous's listener is held by another process than member 0's; a transport error when
    /// the rendezvous cannot be reached, a frame of another build or epoch arrives, the processes
    /// state the members inconsistently, or the group does not form within the rendezvous's
    /// timeout.
    pub(crate) fn world(rendezvous: Rendezvous, held: &[usize], size: usize) -> Result<Vec<Self>> {
        let mut seen = vec![false; size];
        for &member in held {
            if member >= size || std::mem::replace(&mut seen[member], true) {
                return Err(Error::Invalid(format!(
                    "this process holds member {member} of a host group of {size} twice or \
                     outside it"
                )));
            }
        }
        if seen.first().copied() != Some(rendezvous.listener.is_some()) {
            return Err(Error::Invalid(
                "the process that holds the host group's first member serves its rendezvous, \
                 and only it"
                    .into(),
            ));
        }

        let deadline = Instant::now() + rendezvous.timeout;
        let identity = Identity {
            build: proto::BUILD.to_owned(),
            epoch: rendezvous.epoch,
        };
        let (listener, table) = match rendezvous.listener {
            Some(listener) => {
                let listener = Listener::from_tcp(listener).map_err(|error| {
                    Error::transport_source("the host group's rendezvous", error)
                })?;
                let table = serve(
                    &listener,
                    &rendezvous.endpoint,
                    &identity,
                    held,
                    size,
                    deadline,
                )?;
                (listener, table)
            }
            None => join(&rendezvous.endpoint, &identity, held, size, deadline)?,
        };

        // Each member's connections: socket pairs to the members held here, then the connections
        // dialled to and accepted from the members of other processes.
        let mut groups: Vec<Vec<Option<Connection>>> = Vec::with_capacity(held.len());
        for locals in socket_pairs(held.len())? {
            let mut peers: Vec<Option<Connection>> = (0..size).map(|_| None).collect();
            for (place, connection) in locals.into_iter().enumerate() {
                peers[held[place]] = connection;
            }
            groups.push(peers);
        }

        let (dialled, accepted) = std::thread::scope(|scope| {
            let dialling = scope.spawn(|| dial(&table, &identity, held));
            let accepted = accept_pairs(&listener, &identity, held, size, deadline);
            let dialled = dialling
                .join()
                .map_err(|_| Error::Invariant("the host group's dialling panicked".into()))?;
            Ok::<_, Error>((dialled?, accepted?))
        })?;
        for (member, peer, connection) in dialled.into_iter().chain(accepted) {
            let place = held
                .iter()
                .position(|&held| held == member)
                .ok_or_else(|| {
                    Error::Invariant(format!("host group member {member} is not held here"))
                })?;
            groups[place][peer] = Some(connection);
        }

        Ok(groups
            .into_iter()
            .zip(held)
            .map(|(peers, &rank)| Self {
                rank,
                size,
                peers: Mutex::new(peers),
            })
            .collect())
    }

    /// This member's index.
    pub(crate) fn rank(&self) -> usize {
        self.rank
    }

    /// How many members the group has.
    pub(crate) fn size(&self) -> usize {
        self.size
    }

    /// Gathers `value` from every member and returns the values in member order.
    ///
    /// # Errors
    ///
    /// A transport error when a peer is lost.
    pub(crate) fn all_gather_bytes(&self, value: &[u8]) -> Result<Vec<Vec<u8>>> {
        let mut values = vec![Vec::new(); self.size()];
        values[self.rank] = value.to_vec();
        self.exchange(Some(value), None, |member, received| {
            values[member] = received;
            Ok(())
        })?;
        Ok(values)
    }

    /// Gathers one fixed-width record from every member into `records`, member-major.
    ///
    /// Every member passes a record of the same width.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] when `records` is not one record per member, and a transport error when
    /// a peer is lost or sends a record of another width.
    pub(crate) fn all_gather_fixed(&self, local: &[u8], records: &mut [u8]) -> Result<()> {
        let width = local.len();
        if records.len() != width * self.size() {
            return Err(Error::Invalid(format!(
                "a fixed host gather of {width}-byte records fills one record per member, not \
                 {} bytes",
                records.len()
            )));
        }

        records[self.rank * width..(self.rank + 1) * width].copy_from_slice(local);
        self.exchange(Some(local), None, |member, received| {
            if received.len() != width {
                return Err(Error::transport(format!(
                    "host group member {member} sent a record of another width"
                )));
            }
            records[member * width..(member + 1) * width].copy_from_slice(&received);
            Ok(())
        })
    }

    /// Broadcasts member `root`'s bytes and returns them on every member. Only `root` supplies
    /// `value`.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] when `root` is outside the group, when `root` supplies no value or when
    /// another member supplies one, and a transport error when the connection to `root` is lost.
    pub(crate) fn broadcast_bytes(&self, value: Option<&[u8]>, root: usize) -> Result<Vec<u8>> {
        if root >= self.size() {
            return Err(Error::Invalid(
                "broadcast root is outside the host group".into(),
            ));
        }
        if root == self.rank {
            let value = value
                .ok_or_else(|| Error::Invalid("the broadcast root must supply its value".into()))?;
            self.exchange(Some(value), Some(&[]), |_, _| Ok(()))?;
            return Ok(value.to_vec());
        }
        if value.is_some() {
            return Err(Error::Invalid(
                "only the broadcast root supplies a value".into(),
            ));
        }

        let mut broadcast = Vec::new();
        self.exchange(None, Some(&[root]), |_, received| {
            broadcast = received;
            Ok(())
        })?;
        Ok(broadcast)
    }

    /// Reduces every member's integers to their element-wise minimum, which every member returns.
    ///
    /// # Errors
    ///
    /// [`Error::Invariant`] when the members pass different counts of values, and a transport
    /// error when a peer is lost.
    pub(crate) fn all_reduce_min(&self, values: &[i64]) -> Result<Vec<i64>> {
        let encoded: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let mut result = values.to_vec();
        for record in self.all_gather_bytes(&encoded)? {
            let other = decode_integers(&record)?;
            if other.len() != result.len() {
                return Err(Error::Invariant(format!(
                    "host group members reduce {} and {} values",
                    other.len(),
                    result.len()
                )));
            }
            for (value, other) in result.iter_mut().zip(other) {
                *value = (*value).min(other);
            }
        }
        Ok(result)
    }

    /// The core of every collective: sends `value`, when given, to every other member and takes
    /// one value from each member of `sources` (every other member when `None`) with `take`, all
    /// at once. Every send is queued and written as its socket takes it while the values of the
    /// others are read, so the members' sends never wait on each other's reads.
    fn exchange(
        &self,
        value: Option<&[u8]>,
        sources: Option<&[usize]>,
        mut take: impl FnMut(usize, Vec<u8>) -> Result<()>,
    ) -> Result<()> {
        let mut peers = lock(&self.peers);
        if let Some(value) = value {
            let frame = Value(value.to_vec());
            for (member, peer) in peers.iter_mut().enumerate() {
                if let Some(connection) = peer {
                    connection
                        .send(&frame)
                        .map_err(|error| lost(member, error))?;
                }
            }
        }

        let mut waiting: Vec<usize> = match sources {
            Some(sources) => sources.to_vec(),
            None => (0..peers.len())
                .filter(|&member| member != self.rank)
                .collect(),
        };
        loop {
            // Every value that arrived whole is taken.
            let mut index = 0;
            while index < waiting.len() {
                let member = waiting[index];
                let connection = peers[member].as_mut().ok_or_else(|| {
                    Error::Invariant(format!("host group has no connection to {member}"))
                })?;
                match connection.try_recv::<Value>() {
                    Ok(Some(Value(received))) => {
                        take(member, received)?;
                        waiting.swap_remove(index);
                    }
                    Ok(None) => index += 1,
                    Err(error) => return Err(lost(member, error)),
                }
            }

            // Every send writes what its socket takes now.
            let mut writing = false;
            for (member, peer) in peers.iter_mut().enumerate() {
                if let Some(connection) = peer
                    && connection.wants_write()
                {
                    writing |= !connection
                        .try_flush()
                        .map_err(|error| lost(member, error))?;
                }
            }
            if waiting.is_empty() && !writing {
                return Ok(());
            }

            // Sleep until a value can be read or a send can be written.
            let mut descriptors: Vec<libc::pollfd> = peers
                .iter()
                .enumerate()
                .filter_map(|(member, peer)| {
                    let connection = peer.as_ref()?;
                    let mut events = 0;
                    if waiting.contains(&member) {
                        events |= libc::POLLIN;
                    }
                    if connection.wants_write() {
                        events |= libc::POLLOUT;
                    }
                    (events != 0).then(|| pollfd(connection.as_raw_fd(), events))
                })
                .collect();
            wait(&mut descriptors, None)?;
        }
    }
}

/// One member's value in a collective: a frame whose payload is the value.
struct Value(Vec<u8>);

impl service::Message for Value {
    fn encode(&self) -> std::io::Result<Vec<u8>> {
        Ok(self.0.clone())
    }

    fn decode(payload: &[u8]) -> std::io::Result<Self> {
        Ok(Self(payload.to_vec()))
    }
}

/// What every frame of a group's formation states: the build its process runs, since the
/// formation's frames and the collectives' values are this build's, and the epoch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Identity {
    /// The process's build (`proto::BUILD`).
    build: String,
    /// The engine number of the epoch's scheduler node.
    epoch: u64,
}

/// A frame of a group's formation across processes, in MessagePack (module documentation).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
enum Formation {
    /// A process's statement at the rendezvous: the members it holds, and the endpoint its
    /// listener accepts their peers' connections at, bound before it is stated.
    Join {
        identity: Identity,
        members: Vec<u32>,
        endpoint: String,
    },
    /// The rendezvous's answer to every process: the endpoint of each member's process, by
    /// member.
    Table { endpoints: Vec<String> },
    /// The first frame of a connection between members of two processes: member `from` dialled
    /// member `to`, the lower of the two.
    Pair {
        identity: Identity,
        from: u32,
        to: u32,
    },
}

impl service::Message for Formation {
    /// A table states one endpoint per member; a longer frame is no frame of a formation.
    const MAX_BYTES: usize = 1 << 20;

    fn encode(&self) -> std::io::Result<Vec<u8>> {
        rmp_serde::to_vec(self).map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("encoding a host group's formation: {error}"),
            )
        })
    }

    fn decode(payload: &[u8]) -> std::io::Result<Self> {
        rmp_serde::from_slice(payload).map_err(|error| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("decoding a host group's formation: {error}"),
            )
        })
    }
}

/// Refuses `identity` when it is not `ours`: a process of another build or epoch.
fn same_epoch(identity: &Identity, ours: &Identity) -> Result<()> {
    if identity == ours {
        return Ok(());
    }
    Err(Error::transport(format!(
        "a process of build {} and epoch {} reached the host group of build {} and epoch {}",
        identity.build, identity.epoch, ours.build, ours.epoch
    )))
}

/// Serves the rendezvous on `listener`, as the process holding the members `held` (member 0
/// among them) at `endpoint`: accepts every other process's statement until every member of the
/// group's `size` is stated once, then answers each with the table, which it returns.
fn serve(
    listener: &Listener,
    endpoint: &str,
    identity: &Identity,
    held: &[usize],
    size: usize,
    deadline: Instant,
) -> Result<Vec<String>> {
    let mut endpoints: Vec<Option<String>> = (0..size).map(|_| None).collect();
    for &member in held {
        endpoints[member] = Some(endpoint.to_owned());
    }
    let mut stated = held.len();
    let mut joined: Vec<Connection> = Vec::new();

    gather(
        listener,
        deadline,
        "the processes of the host group stating their members",
        stated == size,
        |connection, frame| {
            let Formation::Join {
                identity: theirs,
                members,
                endpoint,
            } = frame
            else {
                return Err(Error::transport(
                    "a process opened the host group's rendezvous with no statement of members",
                ));
            };
            same_epoch(&theirs, identity)?;
            for member in members {
                let slot = usize::try_from(member)
                    .ok()
                    .and_then(|member| endpoints.get_mut(member))
                    .ok_or_else(|| {
                        Error::transport(format!(
                            "a process states member {member} of a host group of {size}"
                        ))
                    })?;
                if slot.replace(endpoint.clone()).is_some() {
                    return Err(Error::transport(format!(
                        "two processes state host group member {member}"
                    )));
                }
                stated += 1;
            }
            joined.push(connection);
            Ok(stated == size)
        },
    )?;

    let table: Vec<String> = endpoints.into_iter().flatten().collect();
    let answer = Formation::Table {
        endpoints: table.clone(),
    };
    for mut connection in joined {
        connection
            .deliver(&answer)
            .map_err(|error| Error::transport_source("answering a host group's process", error))?;
    }
    Ok(table)
}

/// Joins the rendezvous at `endpoint` as the process holding the members `held`: binds a listener
/// on the interface the rendezvous is reached through, states it with the members, and returns it
/// with the table the rendezvous answers.
fn join(
    endpoint: &str,
    identity: &Identity,
    held: &[usize],
    size: usize,
    deadline: Instant,
) -> Result<(Listener, Vec<String>)> {
    let rendezvous: Endpoint = endpoint.parse().map_err(|error| {
        Error::transport_source(
            format!("the host group's rendezvous {endpoint}"),
            std::io::Error::from(error),
        )
    })?;
    let mut connection = Connection::connect(&rendezvous).map_err(|error| {
        Error::transport_source(
            format!("reaching the host group's rendezvous at {endpoint}"),
            error,
        )
    })?;
    let interface = connection
        .local_addr()
        .ok_or_else(|| Error::transport("the host group's rendezvous is reached over no TCP"))?
        .ip();
    let listener = Listener::bind(&Endpoint::tcp(std::net::SocketAddr::new(interface, 0)))
        .map_err(|error| Error::transport_source("the host group's listener", error))?;

    let members = held
        .iter()
        .map(|&member| {
            u32::try_from(member)
                .map_err(|_| Error::Invalid("a host group member beyond u32".into()))
        })
        .collect::<Result<Vec<u32>>>()?;
    connection
        .send(&Formation::Join {
            identity: identity.clone(),
            members,
            endpoint: listener.endpoint().to_string(),
        })
        .map_err(|error| Error::transport_source("stating this process's members", error))?;

    let left = deadline.saturating_duration_since(Instant::now());
    let answer = connection
        .recv::<Formation>(left)
        .map_err(|error| Error::transport_source("awaiting the host group's table", error))?
        .ok_or_else(|| {
            Error::transport(format!(
                "the host group's rendezvous at {endpoint} answered with no table in time"
            ))
        })?;
    let Formation::Table { endpoints } = answer else {
        return Err(Error::transport(
            "the host group's rendezvous answered with no table",
        ));
    };
    if endpoints.len() != size {
        return Err(Error::transport(format!(
            "the host group's table names {} members of {size}",
            endpoints.len()
        )));
    }
    Ok((listener, endpoints))
}

/// Connects each member of `held` to every lower member of another process, at that process's
/// endpoint in `table`, and opens each connection with its pair: `(member, peer, connection)`,
/// the member held here first.
fn dial(
    table: &[String],
    identity: &Identity,
    held: &[usize],
) -> Result<Vec<(usize, usize, Connection)>> {
    let mut dialled = Vec::new();
    for &from in held {
        for to in (0..from).filter(|to| !held.contains(to)) {
            let endpoint: Endpoint = table[to].parse().map_err(|error| {
                Error::transport_source(
                    format!("host group member {to}'s endpoint {}", table[to]),
                    std::io::Error::from(error),
                )
            })?;
            let mut connection = Connection::connect(&endpoint).map_err(|error| {
                Error::transport_source(format!("connecting to host group member {to}"), error)
            })?;
            connection
                .send(&Formation::Pair {
                    identity: identity.clone(),
                    from: from as u32,
                    to: to as u32,
                })
                .map_err(|error| {
                    Error::transport_source(format!("joining host group member {to}"), error)
                })?;
            dialled.push((from, to, connection));
        }
    }
    Ok(dialled)
}

/// Accepts on `listener`, for each member of `held`, the connection of every higher member of a
/// group of `size` that another process holds: `(member, peer, connection)`, the member held here
/// first, as [`dial`] returns them.
fn accept_pairs(
    listener: &Listener,
    identity: &Identity,
    held: &[usize],
    size: usize,
    deadline: Instant,
) -> Result<Vec<(usize, usize, Connection)>> {
    let expected: usize = held
        .iter()
        .map(|&to| (to + 1..size).filter(|from| !held.contains(from)).count())
        .sum();
    let mut accepted: Vec<(usize, usize, Connection)> = Vec::with_capacity(expected);

    gather(
        listener,
        deadline,
        "the host group's members connecting",
        expected == 0,
        |connection, frame| {
            let Formation::Pair {
                identity: theirs,
                from,
                to,
            } = frame
            else {
                return Err(Error::transport(
                    "a member's connection opened with no statement of its pair",
                ));
            };
            same_epoch(&theirs, identity)?;
            let (from, to) = (from as usize, to as usize);
            if !held.contains(&to) || held.contains(&from) || from <= to || from >= size {
                return Err(Error::transport(format!(
                    "host group member {from} connected to member {to}, which is not its pair \
                     here"
                )));
            }
            if accepted
                .iter()
                .any(|(member, peer, _)| (*member, *peer) == (to, from))
            {
                return Err(Error::transport(format!(
                    "host group member {from} connected to member {to} twice"
                )));
            }
            accepted.push((to, from, connection));
            Ok(accepted.len() == expected)
        },
    )?;
    Ok(accepted)
}

/// Accepts connections on `listener` and hands each connection's first frame to `take` as it
/// arrives, until `take` says the formation has what it waits for (at once when `complete`); a
/// connection whose first frame is no frame of the formation is closed and the rest go on. `what`
/// names the wait in the error of a formation that `deadline` ends.
fn gather(
    listener: &Listener,
    deadline: Instant,
    what: &str,
    complete: bool,
    mut take: impl FnMut(Connection, Formation) -> Result<bool>,
) -> Result<()> {
    if complete {
        return Ok(());
    }

    let mut pending: Vec<Connection> = Vec::new();
    loop {
        while let Some(connection) = listener
            .try_accept()
            .map_err(|error| Error::transport_source("accepting a host group connection", error))?
        {
            pending.push(connection);
        }

        let mut index = 0;
        while index < pending.len() {
            match pending[index].try_recv::<Formation>() {
                Ok(Some(frame)) => {
                    if take(pending.swap_remove(index), frame)? {
                        return Ok(());
                    }
                }
                Ok(None) => index += 1,
                Err(error) => {
                    tracing::warn!(%error, "a connection to the host group sent no frame of it");
                    pending.swap_remove(index);
                }
            }
        }

        if Instant::now() >= deadline {
            return Err(Error::transport(format!("{what} did not end in time")));
        }
        let mut descriptors: Vec<libc::pollfd> = std::iter::once(listener.as_raw_fd())
            .chain(pending.iter().map(AsRawFd::as_raw_fd))
            .map(|fd| pollfd(fd, libc::POLLIN))
            .collect();
        wait(&mut descriptors, Some(deadline))?;
    }
}

/// The connections of `size` members of this process to one another, by member and peer: each pair
/// of members is connected by a socket pair, of which each member holds one end.
fn socket_pairs(size: usize) -> Result<Vec<Vec<Option<Connection>>>> {
    let end = |stream: UnixStream| {
        Connection::new(stream)
            .map_err(|error| Error::transport_source("a host group's socket pair", error))
    };

    let mut peers: Vec<Vec<Option<Connection>>> = (0..size)
        .map(|_| (0..size).map(|_| None).collect())
        .collect();
    let pairs = (0..size).flat_map(|lower| (lower + 1..size).map(move |higher| (lower, higher)));
    for (lower, higher) in pairs {
        let (low, high) = UnixStream::pair()
            .map_err(|error| Error::transport_source("a host group's socket pair", error))?;
        peers[lower][higher] = Some(end(low)?);
        peers[higher][lower] = Some(end(high)?);
    }
    Ok(peers)
}

/// The failure of a collective whose connection to `member` failed with `error`.
fn lost(member: usize, error: std::io::Error) -> Error {
    Error::transport_source(format!("host group member {member}"), error)
}

fn decode_integers(bytes: &[u8]) -> Result<Vec<i64>> {
    if !bytes.len().is_multiple_of(8) {
        return Err(Error::transport("host group integer record is malformed"));
    }
    Ok(bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|word| i64::from_le_bytes(*word))
        .collect())
}

/// A `pollfd` of `fd` for `events`.
fn pollfd(fd: RawFd, events: libc::c_short) -> libc::pollfd {
    libc::pollfd {
        fd,
        events,
        revents: 0,
    }
}

/// Blocks until one of `descriptors` is ready or `deadline` passes (never for `None`), keeping the
/// deadline across signals.
fn wait(descriptors: &mut [libc::pollfd], deadline: Option<Instant>) -> Result<()> {
    loop {
        // Rounded up to whole milliseconds, so the poll does not return before the deadline.
        let milliseconds = deadline.map_or(-1, |deadline| {
            let left = deadline.saturating_duration_since(Instant::now());
            let rounded =
                left.as_millis() + u128::from(!left.subsec_nanos().is_multiple_of(1_000_000));
            libc::c_int::try_from(rounded).unwrap_or(libc::c_int::MAX)
        });
        // SAFETY: poll borrows this live array only for the call.
        let result = unsafe {
            libc::poll(
                descriptors.as_mut_ptr(),
                descriptors.len() as libc::nfds_t,
                milliseconds,
            )
        };
        if result >= 0 {
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        if error.kind() != std::io::ErrorKind::Interrupted {
            return Err(Error::transport_source("waiting on a host group", error));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read as _, Write as _};
    use std::net::TcpStream;

    use super::*;

    /// Runs `body` on every member of a group of `size` members of this process, one thread each.
    fn group<T: Send>(size: usize, body: impl Fn(HostGroup) -> T + Sync) -> Vec<T> {
        let members = HostGroup::local(size).unwrap();
        std::thread::scope(|scope| {
            let handles: Vec<_> = members
                .into_iter()
                .map(|member| {
                    let body = &body;
                    scope.spawn(move || body(member))
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect()
        })
    }

    /// The rendezvous of epoch 7 at `endpoint`, which a test's processes form their group at
    /// within a bound only a defect reaches; `listener` for the process that holds member 0.
    fn rendezvous(listener: Option<TcpListener>, endpoint: &str) -> Rendezvous {
        Rendezvous {
            listener,
            endpoint: endpoint.to_owned(),
            epoch: 7,
            timeout: Duration::from_secs(30),
        }
    }

    /// Forms a group of `size` members spread over processes at the rendezvous `listener` serves,
    /// each process holding the members of one entry of `processes` (the first holds member 0),
    /// one thread per process, and runs `body` on every member, one thread each; returns the
    /// results by member.
    fn world<T: Send>(
        listener: TcpListener,
        processes: &[&[usize]],
        size: usize,
        body: impl Fn(HostGroup) -> T + Sync,
    ) -> Vec<T> {
        let endpoint = listener.local_addr().unwrap().to_string();
        let mut listener = Some(listener);
        let mut results: Vec<(usize, T)> = std::thread::scope(|scope| {
            let handles: Vec<_> = processes
                .iter()
                .map(|held| {
                    let rendezvous = rendezvous(listener.take(), &endpoint);
                    let body = &body;
                    scope.spawn(move || {
                        let members = HostGroup::world(rendezvous, held, size).unwrap();
                        std::thread::scope(|threads| {
                            let handles: Vec<_> = members
                                .into_iter()
                                .map(|member| threads.spawn(move || (member.rank(), body(member))))
                                .collect();
                            handles
                                .into_iter()
                                .map(|handle| handle.join().unwrap())
                                .collect::<Vec<_>>()
                        })
                    })
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|handle| handle.join().unwrap())
                .collect()
        });
        results.sort_by_key(|(member, _)| *member);
        results.into_iter().map(|(_, result)| result).collect()
    }

    #[test]
    fn members_gather_every_value_in_member_order() {
        let results = group(4, |group| {
            group.all_gather_bytes(&[group.rank() as u8; 3]).unwrap()
        });
        for values in results {
            assert_eq!(
                values,
                (0..4u8).map(|rank| vec![rank; 3]).collect::<Vec<_>>()
            );
        }
    }

    /// Values far beyond any socket buffer gather as small ones do: every member sends to every
    /// other at once, and none waits for another to read first.
    #[test]
    fn large_values_gather_without_deadlock() {
        let results = group(3, |group| {
            let value = vec![group.rank() as u8; 16 << 20];
            group.all_gather_bytes(&value).unwrap()
        });
        for values in results {
            for (rank, value) in values.iter().enumerate() {
                assert_eq!(value.len(), 16 << 20);
                assert!(value.iter().all(|&byte| byte == rank as u8));
            }
        }
    }

    #[test]
    fn fixed_records_gather_in_member_order() {
        let results = group(3, |group| {
            let mut records = [0_u8; 3 * 24];
            for round in 0..3_u8 {
                group
                    .all_gather_fixed(&[group.rank() as u8 + 10 * round; 24], &mut records)
                    .unwrap();
            }
            records
        });
        for records in results {
            for (member, record) in records.chunks(24).enumerate() {
                assert!(record.iter().all(|&byte| byte == member as u8 + 20));
            }
        }
    }

    #[test]
    fn broadcast_delivers_the_roots_value() {
        let results = group(3, |group| {
            let value = (group.rank() == 1).then_some(&b"store:1234"[..]);
            group.broadcast_bytes(value, 1).unwrap()
        });
        assert!(results.iter().all(|value| value == b"store:1234"));
    }

    #[test]
    fn reductions_agree_on_every_member() {
        let results = group(3, |group| {
            let values = [10 - group.rank() as i64, group.rank() as i64 * 7];
            group.all_reduce_min(&values).unwrap()
        });
        for minimum in results {
            assert_eq!(minimum, vec![8, 0]);
        }
    }

    #[test]
    fn a_single_member_group_needs_no_connection() {
        let group = HostGroup::local(1).unwrap().remove(0);
        assert_eq!(group.all_reduce_min(&[3]).unwrap(), vec![3]);
        assert_eq!(group.broadcast_bytes(Some(b"x"), 0).unwrap(), b"x");
    }

    /// A member that goes away fails the others' next collective instead of leaving them waiting.
    #[test]
    fn a_lost_member_fails_the_collective() {
        let results = group(3, |group| {
            if group.rank() == 1 {
                drop(group);
                return None;
            }
            Some(group.all_gather_bytes(b"value").is_err())
        });
        assert_eq!(results, [Some(true), None, Some(true)]);
    }

    /// Members spread over three processes, each holding members that are not consecutive, form
    /// one group whose collectives reach every member in member order.
    #[test]
    fn members_of_several_processes_form_one_group() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let results = world(listener, &[&[0, 4], &[1, 2], &[3, 5]], 6, |group| {
            let gathered = group.all_gather_bytes(&[group.rank() as u8]).unwrap();
            let root = (group.rank() == 5).then_some(&b"id"[..]);
            let broadcast = group.broadcast_bytes(root, 5).unwrap();
            (gathered, broadcast)
        });
        assert_eq!(results.len(), 6);
        for (gathered, broadcast) in results {
            assert_eq!(
                gathered,
                (0..6u8).map(|rank| vec![rank]).collect::<Vec<_>>()
            );
            assert_eq!(broadcast, b"id");
        }
    }

    /// Another program reaching the rendezvous (a client probing the port, or one that connects
    /// and stays silent) neither stops nor joins the group: the members form the group, and the
    /// client's connection is closed without an answer.
    #[test]
    fn a_stranger_on_the_rendezvous_does_not_stop_the_group() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap().to_string();
        let mut stranger = TcpStream::connect(&endpoint).unwrap();
        stranger
            .write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
            .unwrap();
        let silent = TcpStream::connect(&endpoint).unwrap();

        let results = world(listener, &[&[0], &[1], &[2]], 3, |group| {
            group.all_gather_bytes(&[group.rank() as u8]).unwrap()
        });
        for values in results {
            assert_eq!(values, vec![vec![0], vec![1], vec![2]]);
        }

        let mut answer = Vec::new();
        assert_eq!(stranger.read_to_end(&mut answer).unwrap_or(0), 0);
        drop(silent);
    }

    /// A process of another epoch at the rendezvous fails the formation by name.
    #[test]
    fn a_process_of_another_epoch_fails_the_formation() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap().to_string();
        let mut other = rendezvous(None, &endpoint);
        other.epoch = 8;

        let (served, joined) = std::thread::scope(|scope| {
            let joining = scope.spawn(|| HostGroup::world(other, &[1], 2).map(drop));
            let served = HostGroup::world(rendezvous(Some(listener), &endpoint), &[0], 2);
            (served.map(drop), joining.join().unwrap())
        });
        let refused = served.unwrap_err().to_string();
        assert!(refused.contains("epoch 8"), "{refused}");
        assert!(joined.is_err());
    }

    /// A process that never arrives fails the formation once the rendezvous's bound passes.
    #[test]
    fn a_process_that_never_arrives_fails_the_formation_in_time() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = listener.local_addr().unwrap().to_string();
        let mut alone = rendezvous(Some(listener), &endpoint);
        alone.timeout = Duration::from_millis(50);

        let refused = HostGroup::world(alone, &[0], 2).map(drop).unwrap_err();
        assert!(refused.to_string().contains("in time"), "{refused}");
    }
}
