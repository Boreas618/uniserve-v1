//! Host collectives over a TCP full mesh.
//!
//! A [`HostGroup`] connects every member of a rank group to every other
//! member once, at startup, and runs small host-side collectives over those
//! connections: barriers, byte gathers and broadcasts, and integer
//! reductions. It carries startup agreement (block sizes, cache capacity, slot
//! fits), plan comparison, warmup broadcasts and the bootstrap bytes of
//! device communicators (NCCL unique ids, torch store addresses). Device
//! collectives never use it.
//!
//! Rendezvous: the group's first member holds a listening socket its
//! daemon bound before the process existed (inherited as a descriptor), so
//! the address every member was given is live from the start. Every other
//! member binds an ephemeral listener, connects to the first member and
//! reports its member index and listener address. The first member then
//! sends every member the address table, and each member connects to every
//! member with a lower index above zero. One round forms the full mesh.
//!
//! Collective ordering: every member must call the same collectives in the
//! same order. A subgroup shares its parent's connections, so the caller
//! must not run a parent and a subgroup collective concurrently.

use std::io::{Read, Write};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::os::fd::{FromRawFd, OwnedFd, RawFd};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::{Error, Result};

/// How long startup waits for every member to join the mesh: torch's
/// default collective timeout.
pub(crate) const STARTUP_TIMEOUT: Duration = Duration::from_secs(600);

/// Messages at most this long are written before reading without a helper
/// thread: the kernel's socket buffers absorb them, so concurrent sends from
/// every member cannot deadlock.
const INLINE_SEND_BYTES: usize = 32 << 10;

/// The widest record [`HostGroup::all_gather_fixed`] carries.
pub(crate) const FIXED_RECORD_BYTES: usize = 56;

const CONNECT_INTERVAL: Duration = Duration::from_millis(20);

/// Where a group's members meet.
pub struct Rendezvous {
    /// The address every member was given.
    address: String,
    /// The first member's inherited listener.
    listener: Option<TcpListener>,
}

impl Rendezvous {
    /// A rendezvous at `address` (`host:port`), joined by connecting.
    pub fn connect(address: impl Into<String>) -> Self {
        Self {
            address: address.into(),
            listener: None,
        }
    }

    /// The first member's rendezvous: it accepts on `listener`.
    pub fn serve(address: impl Into<String>, listener: TcpListener) -> Self {
        Self {
            address: address.into(),
            listener: Some(listener),
        }
    }

    /// Adopt a listening socket inherited across exec.
    ///
    /// # Safety
    /// `fd` is an open listening TCP socket that nothing else in this process
    /// owns; the returned value closes it.
    pub(crate) unsafe fn inherited(address: impl Into<String>, fd: RawFd) -> Result<Self> {
        // SAFETY: forwarded from the caller.
        let owned = unsafe { OwnedFd::from_raw_fd(fd) };
        // The daemon cleared close-on-exec only for this process's exec;
        // nothing this rank starts may inherit the socket.
        rustix::io::fcntl_setfd(&owned, rustix::io::FdFlags::CLOEXEC)
            .map_err(|error| Error::transport_source("rendezvous socket", error))?;
        Ok(Self::serve(address, TcpListener::from(owned)))
    }
}

/// One member's connections to every other member of a group.
pub struct HostGroup {
    /// This member's index in `ranks`.
    rank: usize,
    /// Members' global ranks (process-world or union ranks), in member order.
    ranks: Vec<usize>,
    /// Connection to each member by index; `None` for this member.
    peers: Vec<Option<Arc<TcpStream>>>,
}

impl HostGroup {
    /// A group of one member, which needs no connections.
    pub(crate) fn single(rank: usize) -> Self {
        Self {
            rank: 0,
            ranks: vec![rank],
            peers: vec![None],
        }
    }

    /// Join the group of `size` members as member `rank` and form the mesh.
    ///
    /// Member 0 must hold the rendezvous listener. Fails when a member does
    /// not join within `timeout` or reports an inconsistent identity.
    pub fn connect(
        rendezvous: Rendezvous,
        rank: usize,
        size: usize,
        timeout: Duration,
    ) -> Result<Self> {
        if size == 0 || rank >= size {
            return Err(Error::Invalid(
                "host group rank must satisfy 0 <= rank < size".into(),
            ));
        }
        if size == 1 {
            return Ok(Self::single(rank));
        }
        if (rank == 0) != rendezvous.listener.is_some() {
            return Err(Error::Invalid(
                "exactly the first host-group member serves the rendezvous".into(),
            ));
        }

        let deadline = Instant::now() + timeout;
        let mut peers: Vec<Option<Arc<TcpStream>>> = vec![None; size];
        if let Some(listener) = rendezvous.listener {
            // Member 0: accept every other member, then publish the table.
            let mut table = vec![String::new(); size];
            let mut joined = Vec::with_capacity(size - 1);
            listener
                .set_nonblocking(true)
                .map_err(|error| Error::transport_source("rendezvous listener", error))?;
            while joined.len() < size - 1 {
                let stream = accept(&listener, deadline)?;
                let hello = read_frame(&stream)?;
                let (member, address) = parse_hello(&hello, size)?;
                if member == 0 || peers[member].is_some() {
                    return Err(Error::transport(format!(
                        "host group member {member} joined twice or claims the first index"
                    )));
                }
                table[member] = address;
                let stream = Arc::new(stream);
                peers[member] = Some(Arc::clone(&stream));
                joined.push(stream);
            }
            let encoded = table.join("\n").into_bytes();
            for stream in &joined {
                write_frame(stream, &encoded)?;
            }
        } else {
            // Other members: report a listener, receive the table, then
            // connect to every lower member above zero and accept the rest.
            let first = connect(&rendezvous.address, deadline)?;
            let interface = first
                .local_addr()
                .map_err(|error| Error::transport_source("host group connection", error))?
                .ip();
            let listener = TcpListener::bind(SocketAddr::new(interface, 0))
                .map_err(|error| Error::transport_source("host group listener", error))?;
            let port = listener
                .local_addr()
                .map_err(|error| Error::transport_source("host group listener", error))?
                .port();
            write_frame(
                &first,
                format!("{rank}\n{}", SocketAddr::new(interface, port)).as_bytes(),
            )?;
            let table = String::from_utf8(read_frame(&first)?)
                .map_err(|_| Error::transport("host group address table is not text"))?;
            let table: Vec<&str> = table.split('\n').collect();
            if table.len() != size {
                return Err(Error::transport(
                    "host group address table has the wrong size",
                ));
            }
            peers[0] = Some(Arc::new(first));

            for (member, address) in table.iter().enumerate().take(rank).skip(1) {
                let stream = connect(address, deadline)?;
                write_frame(&stream, rank.to_string().as_bytes())?;
                peers[member] = Some(Arc::new(stream));
            }
            listener
                .set_nonblocking(true)
                .map_err(|error| Error::transport_source("host group listener", error))?;
            for _ in rank + 1..size {
                let stream = accept(&listener, deadline)?;
                let member = std::str::from_utf8(&read_frame(&stream)?)
                    .ok()
                    .and_then(|text| text.parse::<usize>().ok())
                    .filter(|&member| member > rank && member < size)
                    .ok_or_else(|| Error::transport("host group peer sent no member index"))?;
                if peers[member].is_some() {
                    return Err(Error::transport(format!(
                        "host group member {member} connected twice"
                    )));
                }
                peers[member] = Some(Arc::new(stream));
            }
        }

        let group = Self {
            rank,
            ranks: (0..size).collect(),
            peers,
        };
        // No read may wait forever on a lost peer during startup.
        group.set_timeout(Some(timeout))?;
        Ok(group)
    }

    /// This member's index.
    pub(crate) fn rank(&self) -> usize {
        self.rank
    }

    pub(crate) fn size(&self) -> usize {
        self.ranks.len()
    }

    /// Members' global ranks, in member order.
    pub(crate) fn ranks(&self) -> &[usize] {
        &self.ranks
    }

    /// Bound every subsequent read and write; `None` waits indefinitely.
    pub(crate) fn set_timeout(&self, timeout: Option<Duration>) -> Result<()> {
        for stream in self.peers.iter().flatten() {
            stream
                .set_read_timeout(timeout)
                .and_then(|()| stream.set_write_timeout(timeout))
                .map_err(|error| Error::transport_source("host group connection", error))?;
        }
        Ok(())
    }

    /// The members at indices `members` of this group, sharing its
    /// connections. Every member of this group must call it, and only the
    /// selected members use the result.
    pub(crate) fn subgroup(&self, members: &[usize]) -> Result<Self> {
        let rank = members
            .iter()
            .position(|&member| member == self.rank)
            .ok_or_else(|| Error::Invalid("a host subgroup must contain its creator".into()))?;
        if members.iter().any(|&member| member >= self.size()) {
            return Err(Error::Invalid(
                "host subgroup members are outside the group".into(),
            ));
        }
        Ok(Self {
            rank,
            ranks: members.iter().map(|&member| self.ranks[member]).collect(),
            peers: members
                .iter()
                .map(|&member| self.peers[member].clone())
                .collect(),
        })
    }

    /// Every member's bytes, in member order.
    pub(crate) fn all_gather_bytes(&self, value: &[u8]) -> Result<Vec<Vec<u8>>> {
        let peers: Vec<(usize, &TcpStream)> = self
            .peers
            .iter()
            .enumerate()
            .filter_map(|(member, stream)| stream.as_deref().map(|stream| (member, stream)))
            .collect();
        let receive = || -> Result<Vec<Vec<u8>>> {
            let mut values = vec![Vec::new(); self.size()];
            values[self.rank] = value.to_vec();
            for &(member, stream) in &peers {
                values[member] = read_frame(stream)?;
            }
            Ok(values)
        };

        if value.len() <= INLINE_SEND_BYTES {
            for &(_, stream) in &peers {
                write_frame(stream, value)?;
            }
            return receive();
        }
        // Larger values are sent while receiving, so members writing to each
        // other with full socket buffers still make progress.
        std::thread::scope(|scope| {
            let sender = scope.spawn(|| -> Result<()> {
                for &(_, stream) in &peers {
                    write_frame(stream, value)?;
                }
                Ok(())
            });
            let received = receive();
            let sent = sender
                .join()
                .map_err(|_| Error::Invariant("host group sender panicked".into()))?;
            sent.and(received)
        })
    }

    /// Gather one fixed-width record from every member into `records`,
    /// member-major, without allocating.
    ///
    /// Every member passes a record of the same width, at most
    /// [`FIXED_RECORD_BYTES`]. Each record travels to each peer as one write
    /// of its length prefix and payload, so a small record such as an expert
    /// step agreement occupies one TCP segment per peer.
    pub(crate) fn all_gather_fixed(&self, local: &[u8], records: &mut [u8]) -> Result<()> {
        let width = local.len();
        if width > FIXED_RECORD_BYTES || records.len() != width * self.size() {
            return Err(Error::Invalid(format!(
                "a fixed host gather carries at most {FIXED_RECORD_BYTES} bytes per member \
                 into one record per member"
            )));
        }
        let mut frame = [0_u8; 8 + FIXED_RECORD_BYTES];
        frame[..8].copy_from_slice(&(width as u64).to_le_bytes());
        frame[8..8 + width].copy_from_slice(local);
        let frame = &frame[..8 + width];
        // Records are far below the socket buffers, so every member writes
        // to all peers before reading without risking a deadlock.
        for mut stream in self.peers.iter().flatten().map(|stream| &**stream) {
            stream
                .write_all(frame)
                .map_err(|error| Error::transport_source("host group send", error))?;
        }
        records[self.rank * width..(self.rank + 1) * width].copy_from_slice(local);
        for (member, stream) in self.peers.iter().enumerate() {
            let Some(stream) = stream.as_deref() else {
                continue;
            };
            let mut reader = stream;
            let mut length = [0; 8];
            reader
                .read_exact(&mut length)
                .map_err(|error| Error::transport_source("host group receive", error))?;
            if u64::from_le_bytes(length) != width as u64 {
                return Err(Error::transport(format!(
                    "host group member {member} sent a record of another width"
                )));
            }
            reader
                .read_exact(&mut records[member * width..(member + 1) * width])
                .map_err(|error| Error::transport_source("host group receive", error))?;
        }
        Ok(())
    }

    /// `root`'s bytes on every member. Only `root` supplies `value`.
    pub(crate) fn broadcast_bytes(&self, value: Option<&[u8]>, root: usize) -> Result<Vec<u8>> {
        if root >= self.size() {
            return Err(Error::Invalid(
                "broadcast root is outside the host group".into(),
            ));
        }
        if root == self.rank {
            let value = value
                .ok_or_else(|| Error::Invalid("the broadcast root must supply its value".into()))?;
            for stream in self.peers.iter().flatten() {
                write_frame(stream, value)?;
            }
            return Ok(value.to_vec());
        }
        if value.is_some() {
            return Err(Error::Invalid(
                "only the broadcast root supplies a value".into(),
            ));
        }
        read_frame(self.peer(root)?)
    }

    /// Wait until every member reaches the barrier.
    pub(crate) fn barrier(&self) -> Result<()> {
        self.all_gather_bytes(&[]).map(drop)
    }

    /// Element-wise minimum of every member's integers.
    pub(crate) fn all_reduce_min(&self, values: &[i64]) -> Result<Vec<i64>> {
        self.all_reduce(values, i64::min)
    }

    /// Split into groups of members with equal `color`, ordered by
    /// `(key, member)`. Every member calls it.
    pub(crate) fn split(&self, color: i64, key: i64) -> Result<Self> {
        let encoded: Vec<u8> = [color, key]
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let mut members: Vec<(i64, usize)> = Vec::new();
        for (member, record) in self.all_gather_bytes(&encoded)?.iter().enumerate() {
            let [other_color, other_key] = decode_integers(record)?[..] else {
                return Err(Error::transport("host group split record is malformed"));
            };
            if other_color == color {
                members.push((other_key, member));
            }
        }
        members.sort_unstable();
        let members: Vec<usize> = members.into_iter().map(|(_, member)| member).collect();
        self.subgroup(&members)
    }

    fn all_reduce(&self, values: &[i64], reduce: impl Fn(i64, i64) -> i64) -> Result<Vec<i64>> {
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
                *value = reduce(*value, other);
            }
        }
        Ok(result)
    }

    fn peer(&self, member: usize) -> Result<&TcpStream> {
        self.peers
            .get(member)
            .and_then(Option::as_deref)
            .ok_or_else(|| Error::Invariant(format!("host group has no connection to {member}")))
    }
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

fn parse_hello(hello: &[u8], size: usize) -> Result<(usize, String)> {
    let text =
        std::str::from_utf8(hello).map_err(|_| Error::transport("host group hello is not text"))?;
    let (member, address) = text
        .split_once('\n')
        .ok_or_else(|| Error::transport("host group hello is malformed"))?;
    let member = member
        .parse::<usize>()
        .ok()
        .filter(|&member| member < size)
        .ok_or_else(|| Error::transport("host group hello names no member"))?;
    Ok((member, address.to_owned()))
}

/// Connect, retrying refused connections until `deadline`.
fn connect(address: &str, deadline: Instant) -> Result<TcpStream> {
    loop {
        let attempt = address
            .to_socket_addrs()
            .map_err(|error| Error::transport_source(format!("resolving {address}"), error))?
            .find_map(|address| {
                let remaining = deadline.saturating_duration_since(Instant::now());
                TcpStream::connect_timeout(&address, remaining.max(CONNECT_INTERVAL)).ok()
            });
        if let Some(stream) = attempt {
            stream.set_nodelay(true).map_err(|error| {
                Error::transport_source(format!("connecting to {address}"), error)
            })?;
            return Ok(stream);
        }
        if Instant::now() >= deadline {
            return Err(Error::transport(format!(
                "host group rendezvous at {address} did not accept before the deadline"
            )));
        }
        std::thread::sleep(CONNECT_INTERVAL);
    }
}

/// Accept one connection from a non-blocking listener before `deadline`.
fn accept(listener: &TcpListener, deadline: Instant) -> Result<TcpStream> {
    loop {
        match listener.accept() {
            Ok((stream, _)) => {
                stream
                    .set_nonblocking(false)
                    .and_then(|()| stream.set_nodelay(true))
                    .and_then(|()| {
                        stream.set_read_timeout(Some(
                            deadline
                                .saturating_duration_since(Instant::now())
                                .max(CONNECT_INTERVAL),
                        ))
                    })
                    .map_err(|error| Error::transport_source("host group connection", error))?;
                return Ok(stream);
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    return Err(Error::transport(
                        "host group members did not join before the deadline",
                    ));
                }
                std::thread::sleep(CONNECT_INTERVAL);
            }
            Err(error) => return Err(Error::transport_source("host group rendezvous", error)),
        }
    }
}

/// Write one length-prefixed frame.
fn write_frame(mut stream: &TcpStream, value: &[u8]) -> Result<()> {
    let length = (value.len() as u64).to_le_bytes();
    stream
        .write_all(&length)
        .and_then(|()| stream.write_all(value))
        .map_err(|error| Error::transport_source("host group send", error))
}

/// Read one length-prefixed frame.
fn read_frame(stream: &TcpStream) -> Result<Vec<u8>> {
    let mut reader = stream;
    let mut length = [0; 8];
    reader
        .read_exact(&mut length)
        .map_err(|error| Error::transport_source("host group receive", error))?;
    let length = usize::try_from(u64::from_le_bytes(length))
        .map_err(|_| Error::transport("host group frame is too large"))?;
    let mut value = vec![0; length];
    reader
        .read_exact(&mut value)
        .map_err(|error| Error::transport_source("host group receive", error))?;
    Ok(value)
}

/// Bind a listener on the interface a rendezvous address names, for a
/// service whose address members learn through the host group (for
/// example a torch store). Returns the listener and the address to publish.
pub(crate) fn bind_beside(rendezvous: &str) -> Result<(TcpListener, String)> {
    let (host, _) = rendezvous
        .rsplit_once(':')
        .ok_or_else(|| Error::Invalid("rendezvous address must be host:port".into()))?;
    let loopback = host
        .parse::<IpAddr>()
        .is_ok_and(|address| address.is_loopback());
    let interface = if loopback {
        IpAddr::V4(Ipv4Addr::LOCALHOST)
    } else {
        IpAddr::V4(Ipv4Addr::UNSPECIFIED)
    };
    let listener = TcpListener::bind(SocketAddr::new(interface, 0))
        .map_err(|error| Error::transport_source("store listener", error))?;
    let port = listener
        .local_addr()
        .map_err(|error| Error::transport_source("store listener", error))?
        .port();
    Ok((listener, format!("{host}:{port}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Form a loopback group of `size` members, one thread each, and run
    /// `body` on every member.
    fn group<T: Send>(size: usize, body: impl Fn(HostGroup) -> T + Sync) -> Vec<T> {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let mut listener = Some(listener);
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..size)
                .map(|rank| {
                    let rendezvous = match listener.take().filter(|_| rank == 0) {
                        Some(listener) => Rendezvous::serve(address.clone(), listener),
                        None => Rendezvous::connect(address.clone()),
                    };
                    let body = &body;
                    scope.spawn(move || {
                        body(
                            HostGroup::connect(rendezvous, rank, size, Duration::from_secs(30))
                                .unwrap(),
                        )
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect()
        })
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

    #[test]
    fn large_values_gather_without_deadlock() {
        let results = group(3, |group| {
            let value = vec![group.rank() as u8; 4 << 20];
            group.all_gather_bytes(&value).unwrap()
        });
        for values in results {
            for (rank, value) in values.iter().enumerate() {
                assert_eq!(value.len(), 4 << 20);
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
    fn subgroups_and_splits_share_the_parent_connections() {
        let results = group(4, |group| {
            group.barrier().unwrap();
            let odd = group
                .split(group.rank() as i64 % 2, -(group.rank() as i64))
                .unwrap();
            let gathered = odd.all_gather_bytes(&[group.rank() as u8]).unwrap();
            (odd.rank(), odd.ranks().to_vec(), gathered)
        });
        // Keys order members descending by rank within each color.
        assert_eq!(results[0], (1, vec![2, 0], vec![vec![2], vec![0]]));
        assert_eq!(results[1], (1, vec![3, 1], vec![vec![3], vec![1]]));
        assert_eq!(results[2], (0, vec![2, 0], vec![vec![2], vec![0]]));
        assert_eq!(results[3], (0, vec![3, 1], vec![vec![3], vec![1]]));
    }

    #[test]
    fn a_single_member_group_needs_no_rendezvous() {
        let group =
            HostGroup::connect(Rendezvous::connect("unused:0"), 0, 1, Duration::ZERO).unwrap();
        assert_eq!(group.all_reduce_min(&[3]).unwrap(), vec![3]);
        assert_eq!(group.broadcast_bytes(Some(b"x"), 0).unwrap(), b"x");
    }

    #[test]
    fn a_lost_member_fails_the_collective() {
        let results = group(2, |group| {
            if group.rank() == 1 {
                drop(group);
                return None;
            }
            Some(group.all_gather_bytes(b"value").is_err())
        });
        assert_eq!(results[0], Some(true));
    }
}
