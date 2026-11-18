//! The path between hosts outside one NVLink fabric: READs this host's
//! Transfer Server posts, from storage another host's Transfer Server
//! registered with its own agent, and WRITEs into such storage from this
//! host's (a handoff's push), over the transport each span names
//! ([`Region::Remote`]): NIXL, or Mooncake Transfer Engine.
//!
//! The process has one NIXL agent with the UCX backend, over the host's
//! usable RDMA ports with the transports the server's settings state, created
//! at its first use ([`Network::agent`]), so a host whose tasks never cross
//! hosts never loads NIXL; and one Transfer Engine, which names its segment by
//! this host's name and a port of its own, created at its first use too
//! ([`Network::engine`]). Either serves both ends of a READ:
//!
//! - **Source.** The server has the Transfer Server register allocations a
//!   peer asked it to expose ([`Event::Register`]): host storage with the
//!   allocation ([`Mapping::nixl_host_region`]), device storage in its GPU's
//!   session, which the exposure holds until the peer ends it
//!   ([`Event::Deregister`]); the report carries each range and the agent's
//!   metadata taken after the registrations ([`register`]).
//! - **Destination.** A task's move from another host names that host's
//!   Storage Server by its endpoint; so does a move into another host, which
//!   a WRITE copies, read from this host's storage. Before the task's pieces are queued,
//!   [`Network::pieces`] asks that server, over this process's connection to
//!   it as a [`Role::Peer`], to expose the task's regions there
//!   ([`Request::Expose`]), loads the metadata of the agent that registered
//!   them, checks each span against its region's range and registers each
//!   destination with this agent; network copy threads then post the READs
//!   ([`super::threads`]). Once the task ended, its exposures end
//!   ([`Network::unexpose`]), so that the other host releases what it holds
//!   for them, GPU contexts among them.
//!
//! A source whose creator recorded its device allocation's fabric handle
//! ([`Registration::fabric`]) is copied over the NVLink fabric instead,
//! whenever the handle imports on the destination's GPU: the destination's
//! copy engines read it as they read another host's region a task names by
//! fabric handle. A handle that does not import (the GPUs share no fabric,
//! or the handle names no allocation) leaves the source to its READ, from
//! the registration the exposure made.
//!
//! An agent's name is unique to its process, so a peer whose Transfer
//! Server restarted responds with another agent: its metadata is loaded and
//! the previous agent's forgotten, and the new one registers the regions
//! afresh, since its server gives it every allocation again. A Transfer
//! Engine's segment is opened afresh at each exposure, so its description
//! states what the engine registered for it. Regions are exposed for each
//! task, not remembered between tasks: a region's allocation, and its
//! registration, may change between two tasks.
//!
//! A source whose span takes the Transfer Engine is the engine's to carry: no
//! fabric handle replaces its READ.
//!
//! The connection to another host's Storage Server stays open between tasks.
//! A server that ended or restarted since closed it, which a read of the
//! connection shows: the next task finds it closed and connects afresh. A
//! request on a connection that fails fails its task.
//!
//! [`Event::Register`]: ::storage::protocol::Event::Register
//! [`Event::Deregister`]: ::storage::protocol::Event::Deregister
//! [`Request::Expose`]: ::storage::protocol::Request::Expose
//! [`Role::Peer`]: ::storage::protocol::Role::Peer

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use ::storage::Client;
use ::storage::protocol::{FabricExport, Link, LinkEnd, Region, Registration, TaskFailure};
use ::storage::settings::Transports;
use platform::DeviceId;
use scheduling::kv::Transport;

use super::calibration::RemotePath;
use super::contexts::Hold;
use super::mooncake;
use super::nixl;
use super::storage::{Fabric, Mapping};
use super::threads::{End, Piece, Runner, ThreadKey};

/// The process's NIXL agent and its connections to other hosts' Storage
/// Servers.
pub(super) struct Network {
    /// The transports the agent takes.
    transports: Transports,
    /// The agent, or why it could not start, once something needed it.
    agent: OnceLock<Result<nixl::Agent, String>>,
    /// The Transfer Engine, or why it could not start, once something needed it.
    engine: OnceLock<Result<mooncake::Engine, String>>,
    /// Other hosts' Storage Servers by endpoint, each locked by the helper
    /// thread that exposes a task's regions there.
    hosts: Mutex<HashMap<String, Arc<Mutex<Host>>>>,
}

/// Another host's Storage Server, as this process reaches it.
#[derive(Default)]
struct Host {
    /// The connection, while it works.
    client: Option<Client>,
    /// The agent of that host's Transfer Server whose metadata was loaded
    /// last.
    peer: Option<String>,
}

/// A region of another host's storage that its Transfer Server registered,
/// which READs read and WRITEs write.
pub(super) struct Source {
    /// The host's Storage Server, which names it in errors.
    pub(super) endpoint: String,
    /// The link end the region is: the host and its GPU or host storage.
    pub(super) end: LinkEnd,
    /// How this process's transfers reach the region.
    pub(super) reach: Reach,
    /// The bytes from the region's first byte to the end of its allocation,
    /// as the other host registered them.
    pub(super) len: u64,
    /// The device allocation's fabric handle, when its creator recorded
    /// one.
    pub(super) fabric: Option<FabricExport>,
}

/// How this process's transfers reach a region of another host, by the
/// transport its span takes.
pub(super) enum Reach {
    /// The NIXL agent that registered it, and the range as this agent's
    /// transfers name it.
    Nixl { peer: String, region: nixl::Region },
    /// The segment of the Transfer Engine that registered it, as this
    /// process's engine opened it, and the region's first byte in the
    /// process that registered it.
    Mooncake {
        segment: mooncake::SegmentId,
        address: u64,
    },
}

/// Which way a move between this host and another host goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Direction {
    /// From the other host's region into this host's allocation: a READ.
    Read,
    /// From this host's allocation into the other host's region: a WRITE.
    Write,
}

/// A task's move between another host's region and an allocation of this host's, before that
/// host exposed its region; `hold` holds the GPU of a device allocation of this host.
pub(super) struct Inbound {
    pub(super) endpoint: String,
    pub(super) deployment: String,
    pub(super) direction: Direction,
    /// The transport the move crosses between the hosts by.
    pub(super) transport: Transport,
    /// The other host's region and the offset of the move in it.
    pub(super) region: Region,
    pub(super) remote_offset: u64,
    /// This host's allocation and the offset of the move in it.
    pub(super) local: Arc<Mapping>,
    pub(super) hold: Option<Hold>,
    pub(super) local_offset: u64,
    /// The bytes of each row, the rows, and their pitches in the other host's region and in
    /// this host's allocation, as the span states them.
    pub(super) bytes: u64,
    pub(super) rows: u64,
    pub(super) remote_pitch: u64,
    pub(super) local_pitch: u64,
}

/// An exposure of another host a task READs, which it ends when it ends.
pub(super) struct Lease {
    endpoint: String,
    exposure: u64,
}

/// A task's pieces from other hosts, each on its copy thread, and the exposures
/// it ends when it ends.
pub(super) struct Sources {
    pub(super) pieces: Vec<(ThreadKey, Piece)>,
    pub(super) leases: Vec<Lease>,
}

/// An allocation exposed to another host, with the hold on its GPU for
/// device storage.
pub(super) type Exposed = (Arc<Mapping>, Option<Hold>);

/// Why a task's sources on other hosts cannot be read, and the message that
/// tells it.
pub(super) type Failed = (TaskFailure, String);

/// A failure of this host's part of a READ: a registration, a GPU context, the
/// agent, a span that does not fit.
fn copy(message: String) -> Failed {
    (TaskFailure::Copy, message)
}

impl Network {
    /// No agent yet; the agent takes `transports` once it starts.
    pub(super) fn new(transports: Transports) -> Self {
        Self {
            transports,
            agent: OnceLock::new(),
            engine: OnceLock::new(),
            hosts: Mutex::new(HashMap::new()),
        }
    }

    /// The process's Transfer Engine, started at the first call: other hosts' engines reach it
    /// at this host's name (as the Mooncake connector names its host) and the port it picks; an
    /// error says why it cannot start, for every task that needs it.
    pub(super) fn engine(&self) -> Result<&mooncake::Engine, String> {
        self.engine
            .get_or_init(|| {
                let host = crate::host_name().map_err(|error| {
                    format!("naming this host for the Transfer Engine: {error}")
                })?;
                let engine = mooncake::Topology::of_host()
                    .and_then(|topology| mooncake::Engine::new(&host, &topology))
                    .map_err(|error| format!("starting the Transfer Engine: {error}"))?;
                tracing::info!(segment = engine.name(), "started the Transfer Engine");
                Ok(engine)
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    /// Registers `mapped`, an allocation of this host, with the agent of `transport`, as the
    /// end of a READ or a WRITE: device storage in its GPU's session, which `hold` holds, host
    /// storage with the allocation. A registration is made once, at the first call.
    pub(super) fn register_local(
        &self,
        transport: Transport,
        mapped: &Mapping,
        hold: Option<&Hold>,
    ) -> Result<(), String> {
        match transport {
            Transport::Nixl => {
                let agent = self.agent()?;
                match hold {
                    Some(hold) => hold.session()?.nixl_region(agent, mapped).map(drop),
                    None => mapped.nixl_host_region(agent).map(drop),
                }
            }
            Transport::Mooncake => {
                let engine = self.engine()?;
                match hold {
                    Some(hold) => hold.session()?.mooncake_address(engine, mapped).map(drop),
                    None => mapped.mooncake_host_address(engine).map(drop),
                }
            }
        }
    }

    /// The process's NIXL agent, started at the first call; an error says
    /// why NIXL or UCX cannot start, for every task that needs them.
    pub(super) fn agent(&self) -> Result<&nixl::Agent, String> {
        self.agent
            .get_or_init(|| {
                let name = agent_name().map_err(|error| format!("naming a NIXL agent: {error}"))?;
                let backend = nixl::Backend::of_host(self.transports)
                    .map_err(|error| format!("starting a NIXL agent: {error}"))?;
                let agent = nixl::Agent::new(&name, &backend)
                    .map_err(|error| format!("starting a NIXL agent: {error}"))?;
                tracing::info!(agent = name, "started the NIXL agent");
                Ok(agent)
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    /// The pieces of task `task`'s moves from other hosts, each on the copy thread
    /// of its link from the source host's storage, and the exposures the task
    /// ends when it ends: every host exposes the task's regions there in one
    /// request per deployment, and every destination is registered with this
    /// agent, so a task whose sources or destinations cannot be read or
    /// written fails before its first READ, ending the exposures it got.
    /// Blocks on the other hosts' responses and on the destinations' GPU
    /// contexts.
    pub(super) fn pieces(&self, task: u64, moves: Vec<Inbound>) -> Result<Sources, Failed> {
        let mut leases = Vec::new();
        let pieces = self.expose_moves(task, moves, &mut leases);
        match pieces {
            Ok(pieces) => Ok(Sources { pieces, leases }),
            Err(failed) => {
                self.unexpose(leases);
                Err(failed)
            }
        }
    }

    /// [`Network::pieces`], adding each exposure it gets to `leases`.
    fn expose_moves(
        &self,
        task: u64,
        moves: Vec<Inbound>,
        leases: &mut Vec<Lease>,
    ) -> Result<Vec<(ThreadKey, Piece)>, Failed> {
        type Wanted = (String, String, Transport);
        let mut wanted: HashMap<Wanted, Vec<Region>> = HashMap::new();
        for copy in &moves {
            let regions = wanted
                .entry((
                    copy.endpoint.clone(),
                    copy.deployment.clone(),
                    copy.transport,
                ))
                .or_default();
            if !regions.contains(&copy.region) {
                regions.push(copy.region.clone());
            }
        }

        let mut sources: HashMap<(String, String, Transport, Region), Arc<Source>> = HashMap::new();
        for ((endpoint, deployment, transport), regions) in wanted {
            let (peer, exposure, ranges) =
                self.expose(&endpoint, &deployment, regions.clone(), transport)?;
            leases.push(Lease {
                endpoint: endpoint.clone(),
                exposure,
            });
            if ranges.len() != regions.len() {
                return Err(copy(format!(
                    "{endpoint} exposed {} of {} regions",
                    ranges.len(),
                    regions.len()
                )));
            }
            for (region, range) in regions.into_iter().zip(ranges) {
                let reach = match &peer {
                    Peer::Nixl(peer) => Reach::Nixl {
                        peer: peer.clone(),
                        region: source_region(&range).map_err(copy)?,
                    },
                    Peer::Mooncake(segment) => Reach::Mooncake {
                        segment: *segment,
                        address: range.address,
                    },
                };
                let source = Source {
                    end: LinkEnd::Remote {
                        endpoint: endpoint.clone(),
                        gpu: match range.end {
                            LinkEnd::Device { gpu } => Some(gpu),
                            _ => None,
                        },
                    },
                    reach,
                    len: range.bytes,
                    endpoint: endpoint.clone(),
                    fabric: range.fabric,
                };
                sources.insert(
                    (endpoint.clone(), deployment.clone(), transport, region),
                    Arc::new(source),
                );
            }
        }

        let mut pieces = Vec::with_capacity(moves.len());
        // Each fabric handle is imported once per task and GPU.
        let mut imports: HashMap<([u8; 64], DeviceId), Arc<Fabric>> = HashMap::new();
        for copy in moves {
            let transport = copy.transport;
            let Some(source) =
                sources.get(&(copy.endpoint, copy.deployment, transport, copy.region))
            else {
                return Err(self::copy("a region was not exposed".into()));
            };
            // Another host checked the region; the span is checked here,
            // against the range it registered from the region's first byte:
            // from its first row's start to its last row's end.
            let within = (copy.rows - 1)
                .checked_mul(copy.remote_pitch)
                .and_then(|start| start.checked_add(copy.remote_offset))
                .and_then(|start| start.checked_add(copy.bytes))
                .is_some_and(|end| end <= source.len);
            if !within {
                return Err(self::copy(format!(
                    "a span's end of {} rows of {} bytes at offset {} and pitch {} reaches \
                     past the {} bytes of its region at {}",
                    copy.rows,
                    copy.bytes,
                    copy.remote_offset,
                    copy.remote_pitch,
                    source.len,
                    source.endpoint
                )));
            }
            let local = End::Given {
                mapped: Arc::clone(&copy.local),
                hold: copy.hold.clone(),
            };
            if copy.direction == Direction::Write {
                // A WRITE posts from this host's allocation into the other host's region.
                self.register_local(transport, &copy.local, copy.hold.as_ref())
                    .map_err(self::copy)?;
                let link = Link {
                    source: copy.local.end.clone(),
                    destination: source.end.clone(),
                };
                pieces.push((
                    ThreadKey {
                        link,
                        runner: Runner::Network,
                        path: Some(remote_path(transport)),
                    },
                    Piece {
                        task,
                        source: local,
                        source_offset: copy.local_offset,
                        destination: End::Remote(Arc::clone(source)),
                        destination_offset: copy.remote_offset,
                        bytes: copy.bytes,
                        rows: copy.rows,
                        source_pitch: copy.local_pitch,
                        destination_pitch: copy.remote_pitch,
                    },
                ));
                continue;
            }
            let read = Piece {
                task,
                source: End::Remote(Arc::clone(source)),
                source_offset: copy.remote_offset,
                destination: local,
                destination_offset: copy.local_offset,
                bytes: copy.bytes,
                rows: copy.rows,
                source_pitch: copy.remote_pitch,
                destination_pitch: copy.local_pitch,
            };
            let End::Given { mapped, hold } = &read.destination else {
                unreachable!("a READ's destination is this host's allocation");
            };
            // A span over the Transfer Engine is the engine's to carry.
            let fabric = source.fabric.filter(|_| transport == Transport::Nixl);
            if let (Some(fabric), Some(hold)) = (fabric, hold) {
                match over_fabric(source, fabric, mapped, hold, &mut imports) {
                    Ok((thread, import)) => {
                        let source_offset = fabric
                            .offset
                            .checked_add(read.source_offset)
                            .ok_or_else(|| self::copy("a span's source offset overflows".into()))?;
                        pieces.push((
                            thread,
                            Piece {
                                source: import,
                                source_offset,
                                ..read
                            },
                        ));
                        continue;
                    }
                    Err(message) => tracing::debug!(
                        message,
                        endpoint = source.endpoint,
                        "a source's fabric handle did not import: READ it"
                    ),
                }
            }
            self.register_local(transport, mapped, hold.as_ref())
                .map_err(self::copy)?;
            let link = Link {
                source: source.end.clone(),
                destination: mapped.end.clone(),
            };
            pieces.push((
                ThreadKey {
                    link,
                    runner: Runner::Network,
                    path: Some(remote_path(transport)),
                },
                read,
            ));
        }
        Ok(pieces)
    }

    /// End `leases`, the exposures of other hosts a task READ from. A host
    /// that cannot be reached ended them already: its server ends a peer's
    /// exposures when the peer's connection closes.
    pub(super) fn unexpose(&self, leases: Vec<Lease>) {
        for lease in leases {
            let host = self.host(&lease.endpoint);
            let mut host = ::service::lock(&host);
            let Some(client) = &mut host.client else {
                continue;
            };
            if let Err(error) = client.unexpose(lease.exposure) {
                tracing::warn!(%error, endpoint = lease.endpoint, "ending an exposure failed");
                host.client = None;
            }
        }
    }

    /// The host whose Storage Server listens at `endpoint`.
    fn host(&self, endpoint: &str) -> Arc<Mutex<Host>> {
        Arc::clone(
            ::service::lock(&self.hosts)
                .entry(endpoint.to_owned())
                .or_default(),
        )
    }

    /// Have the Storage Server at `endpoint` expose `regions` of
    /// `deployment` over `transport`, and load what reaches the agent that
    /// registered them: returns that agent as this process reaches it, the
    /// exposure and each region's range. A server that cannot be reached
    /// fails it as [`TaskFailure::Unreachable`], one that refuses as
    /// [`TaskFailure::Refused`] with its error's kind.
    fn expose(
        &self,
        endpoint: &str,
        deployment: &str,
        regions: Vec<Region>,
        transport: Transport,
    ) -> Result<(Peer, u64, Vec<Registration>), Failed> {
        let host = self.host(endpoint);
        let mut host = ::service::lock(&host);
        tracing::debug!(
            endpoint,
            deployment,
            regions = regions.len(),
            "asking another host to expose regions"
        );
        let (exposure, metadata, ranges) = host
            .expose(endpoint, deployment, regions, transport)
            .map_err(|error| {
                let failure = match &error {
                    ::storage::Error::Server { kind, .. } => TaskFailure::Refused { kind: *kind },
                    ::storage::Error::Io(_) | ::storage::Error::Protocol(_) => {
                        TaskFailure::Unreachable
                    }
                    _ => TaskFailure::Copy,
                };
                let message =
                    format!("{endpoint} did not expose the regions of {deployment}: {error}");
                (failure, message)
            })?;

        if transport == Transport::Mooncake {
            // The metadata is the engine's segment name, opened afresh so that its description
            // states the regions just registered.
            let name = String::from_utf8(metadata)
                .map_err(|_| copy(format!("{endpoint} named its Transfer Engine in no UTF-8")))?;
            let engine = self.engine().map_err(copy)?;
            let segment = engine.open(&name).map_err(|error| {
                copy(format!(
                    "opening the Transfer Engine segment {name} of {endpoint}: {error}"
                ))
            })?;
            tracing::debug!(
                endpoint,
                exposure,
                segment = name,
                regions = ranges.len(),
                "another host exposed regions"
            );
            return Ok((Peer::Mooncake(segment), exposure, ranges));
        }
        let agent = self.agent().map_err(copy)?;
        let peer = agent.load_peer(&metadata).map_err(|error| {
            copy(format!(
                "loading the metadata of the NIXL agent of {endpoint}: {error}"
            ))
        })?;
        if let Some(previous) = host.peer.replace(peer.clone())
            && previous != peer
        {
            // The host's Transfer Server restarted, and its previous agent
            // ended with it.
            if let Err(error) = agent.forget_peer(&previous) {
                tracing::warn!(%error, peer = previous, "forgetting an ended NIXL agent failed");
            }
        }
        tracing::debug!(
            endpoint,
            exposure,
            peer,
            regions = ranges.len(),
            "another host exposed regions"
        );
        Ok((Peer::Nixl(peer), exposure, ranges))
    }
}

/// The agent of another host that registered an exposure's regions, as this process reaches
/// it: a NIXL agent by its name, a Transfer Engine by its segment.
enum Peer {
    Nixl(String),
    Mooncake(mooncake::SegmentId),
}

impl Host {
    /// One exposure on the connection to the server at `endpoint`: the open
    /// connection while the server keeps it, a new one otherwise.
    fn expose(
        &mut self,
        endpoint: &str,
        deployment: &str,
        regions: Vec<Region>,
        transport: Transport,
    ) -> ::storage::Result<(u64, Vec<u8>, Vec<Registration>)> {
        // A server that ended or restarted since the last task closed the
        // connection: reading what it sent shows the end.
        if let Some(client) = &mut self.client
            && client.pump().is_err()
        {
            self.client = None;
        }
        let client = match &mut self.client {
            Some(client) => client,
            None => self.client.insert(Client::connect_peer(endpoint)?),
        };
        let responded = client.expose(deployment.to_owned(), regions, transport);
        // The server's response, an exposure or a refusal, leaves the
        // connection usable; a failed connection is not used again.
        if let Err(::storage::Error::Io(_) | ::storage::Error::Protocol(_)) = &responded {
            self.client = None;
        }
        responded
    }
}

/// The copy thread and the source end of a move from `source`, which its host
/// exposed with the fabric handle `fabric`, into `destination`, copied over
/// the NVLink fabric by the copy engines of the destination's GPU, which
/// `hold` holds; the move's source offset counts from the exposed range's
/// first byte, `fabric.offset` bytes into the imported allocation. The
/// handle is imported there now (once per task and GPU, through
/// `imports`), so that a handle that does not import fails here and leaves
/// the move to its READ.
fn over_fabric(
    source: &Source,
    fabric: FabricExport,
    destination: &Mapping,
    hold: &Hold,
    imports: &mut HashMap<([u8; 64], DeviceId), Arc<Fabric>>,
) -> Result<(ThreadKey, End), String> {
    let LinkEnd::Remote { gpu: Some(gpu), .. } = &source.end else {
        return Err("a fabric handle names device storage".into());
    };
    let device = destination.device()?;
    // The destination's GPU context, which the import lives in.
    hold.session()?;
    let import = Arc::clone(
        imports
            .entry((fabric.handle.0, device))
            .or_insert_with(|| Arc::new(Fabric::new(fabric.handle.0, fabric.bytes, device))),
    );
    import.address()?;
    Ok((
        ThreadKey {
            link: Link {
                source: LinkEnd::Device { gpu: *gpu },
                destination: destination.end.clone(),
            },
            runner: Runner::Device(device),
            path: Some(RemotePath::Fabric),
        },
        End::Fabric {
            import,
            _hold: hold.clone(),
        },
    ))
}

/// Register `allocations` with `network`'s agent of `transport` for other hosts' READs and
/// WRITEs, each with the hold on its GPU for device storage: each one's range, in order, and the
/// agent's metadata taken after the registrations, which carries them (a Transfer Engine's
/// segment name, whose description the other side fetches when it opens it).
pub(super) fn register(
    network: &Network,
    allocations: &[Exposed],
    transport: Transport,
) -> Result<(Vec<u8>, Vec<Registration>), String> {
    if transport == Transport::Mooncake {
        let engine = network.engine()?;
        let ranges = allocations
            .iter()
            .map(|(mapped, hold)| {
                let address = match hold {
                    Some(hold) => hold.session()?.mooncake_address(engine, mapped)?,
                    None => mapped.mooncake_host_address(engine)?,
                };
                Ok(Registration {
                    end: mapped.end.clone(),
                    // The engine names a registration by its address alone.
                    key: 0,
                    address,
                    bytes: mapped.bytes,
                    fabric: None,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        return Ok((engine.name().as_bytes().to_vec(), ranges));
    }
    let agent = network.agent()?;
    let ranges = allocations
        .iter()
        .map(|(mapped, hold)| {
            let region = match hold {
                Some(hold) => hold.session()?.nixl_region(agent, mapped)?,
                None => mapped.nixl_host_region(agent)?,
            };
            Ok(Registration {
                end: mapped.end.clone(),
                key: region.key,
                address: region.address,
                bytes: region.len as u64,
                // The server states the fabric handles its records know.
                fabric: None,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let metadata = agent
        .metadata()
        .map_err(|error| format!("exporting the NIXL agent's metadata: {error}"))?;
    Ok((metadata, ranges))
}

/// The calibrated path of copies with another host over `transport`.
fn remote_path(transport: Transport) -> RemotePath {
    match transport {
        Transport::Nixl => RemotePath::Read,
        Transport::Mooncake => RemotePath::Mooncake,
    }
}

/// A range another host's agent registered, as this agent's READs name it.
fn source_region(range: &Registration) -> Result<nixl::Region, String> {
    let segment = match range.end {
        LinkEnd::Host => nixl::Segment::Host,
        LinkEnd::Device { .. } => nixl::Segment::Device,
        LinkEnd::Remote { .. } => {
            return Err("another host exposed storage it does not hold".into());
        }
    };
    Ok(nixl::Region {
        segment,
        address: range.address,
        key: range.key,
        len: usize::try_from(range.bytes)
            .map_err(|_| "an exposed region exceeds the address space".to_owned())?,
    })
}

/// A name no other agent takes, on any host or across restarts: 128 bits from the kernel's
/// random source.
pub(super) fn agent_name() -> std::io::Result<String> {
    let mut identity = [0_u8; 16];
    let filled = rustix::rand::getrandom(&mut identity, rustix::rand::GetRandomFlags::empty())?;
    if filled != identity.len() {
        return Err(std::io::Error::other(
            "the kernel gave fewer random bytes than asked",
        ));
    }
    let hex: String = identity.iter().map(|byte| format!("{byte:02x}")).collect();
    Ok(format!("uniserve-transfer-{hex}"))
}
