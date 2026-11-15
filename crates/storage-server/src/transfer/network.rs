//! The path between hosts outside one NVLink fabric: NIXL READs this host's
//! Transfer Server posts, from storage another host's Transfer Server
//! registered with its own agent.
//!
//! The process has one NIXL agent with the UCX backend, created at its first
//! use ([`Network::agent`]), so a host whose tasks never cross hosts never
//! loads NIXL. The agent serves both ends of a READ:
//!
//! - **Source.** The server has the Transfer Server register allocations a
//!   peer asked it to expose ([`Event::Register`]): host storage with the
//!   allocation ([`Mapping::nixl_host_region`]), device storage in its GPU's
//!   session, which the exposure holds until the peer ends it
//!   ([`Event::Deregister`]); the report carries each range and the agent's
//!   metadata taken after the registrations ([`register`]).
//! - **Destination.** A task's move from another host names that host's
//!   Storage Server by its endpoint. Before the task's pieces are queued,
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
//! afresh, since its server gives it every allocation again. Regions are
//! exposed for each task, not remembered between tasks: a region's
//! allocation, and its registration, may change between two tasks.
//!
//! [`Event::Register`]: ::storage::protocol::Event::Register
//! [`Event::Deregister`]: ::storage::protocol::Event::Deregister
//! [`Request::Expose`]: ::storage::protocol::Request::Expose
//! [`Role::Peer`]: ::storage::protocol::Role::Peer

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use ::storage::Client;
use ::storage::protocol::{FabricExport, Link, LinkEnd, Region, Registration, TaskFailure};
use platform::DeviceId;

use super::contexts::Hold;
use super::nixl;
use super::storage::{Fabric, Mapping};
use super::threads::{End, Piece, Runner, ThreadKey};

/// The process's NIXL agent and its connections to other hosts' Storage
/// Servers.
pub(super) struct Network {
    /// The agent, or why it could not start, once something needed it.
    agent: OnceLock<Result<nixl::Agent, String>>,
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
/// which READs read.
pub(super) struct Source {
    /// The host's Storage Server, which names it in errors.
    pub(super) endpoint: String,
    /// The link end the region is: the host and its GPU or host storage.
    pub(super) end: LinkEnd,
    /// The agent that registered it.
    pub(super) peer: String,
    /// The range from the region's first byte to the end of its
    /// allocation, as that agent registered it.
    pub(super) region: nixl::Region,
    /// The device allocation's fabric handle, when its creator recorded
    /// one.
    pub(super) fabric: Option<FabricExport>,
}

/// A task's move from another host's storage into an allocation of this
/// host's, before that host exposed its source; `hold` holds the GPU of a
/// device destination.
pub(super) struct Inbound {
    pub(super) endpoint: String,
    pub(super) deployment: String,
    pub(super) region: Region,
    pub(super) source_offset: u64,
    pub(super) destination: Arc<Mapping>,
    pub(super) hold: Option<Hold>,
    pub(super) destination_offset: u64,
    /// The bytes of each row, the rows, and their pitches on each side, as
    /// the span states them.
    pub(super) bytes: u64,
    pub(super) rows: u64,
    pub(super) source_pitch: u64,
    pub(super) destination_pitch: u64,
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
    pub(super) fn new() -> Self {
        Self {
            agent: OnceLock::new(),
            hosts: Mutex::new(HashMap::new()),
        }
    }

    /// The process's NIXL agent, started at the first call; an error says
    /// why NIXL or UCX cannot start, for every task that needs them.
    pub(super) fn agent(&self) -> Result<&nixl::Agent, String> {
        self.agent
            .get_or_init(|| {
                let name = agent_name().map_err(|error| format!("naming a NIXL agent: {error}"))?;
                let agent = nixl::Agent::new(&name)
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
        let mut wanted: HashMap<(String, String), Vec<Region>> = HashMap::new();
        for copy in &moves {
            let regions = wanted
                .entry((copy.endpoint.clone(), copy.deployment.clone()))
                .or_default();
            if !regions.contains(&copy.region) {
                regions.push(copy.region.clone());
            }
        }

        let mut sources: HashMap<(String, String, Region), Arc<Source>> = HashMap::new();
        for ((endpoint, deployment), regions) in wanted {
            let (peer, exposure, ranges) = self.expose(&endpoint, &deployment, regions.clone())?;
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
                let source = Source {
                    end: LinkEnd::Remote {
                        endpoint: endpoint.clone(),
                        gpu: match range.end {
                            LinkEnd::Device { gpu } => Some(gpu),
                            _ => None,
                        },
                    },
                    region: source_region(&range).map_err(copy)?,
                    endpoint: endpoint.clone(),
                    peer: peer.clone(),
                    fabric: range.fabric,
                };
                sources.insert(
                    (endpoint.clone(), deployment.clone(), region),
                    Arc::new(source),
                );
            }
        }

        let agent = self.agent().map_err(copy)?;
        let mut pieces = Vec::with_capacity(moves.len());
        // Each fabric handle is imported once per task and GPU.
        let mut imports: HashMap<([u8; 64], DeviceId), Arc<Fabric>> = HashMap::new();
        for copy in moves {
            let Some(source) = sources.get(&(copy.endpoint, copy.deployment, copy.region)) else {
                return Err(self::copy("a region was not exposed".into()));
            };
            // Another host checked the region; the span is checked here,
            // against the range it registered from the region's first byte:
            // from its first row's start to its last row's end.
            let within = (copy.rows - 1)
                .checked_mul(copy.source_pitch)
                .and_then(|start| start.checked_add(copy.source_offset))
                .and_then(|start| start.checked_add(copy.bytes))
                .is_some_and(|end| end <= source.region.len as u64);
            if !within {
                return Err(self::copy(format!(
                    "a span's source of {} rows of {} bytes at offset {} and pitch {} reaches \
                     past the {} bytes of its region at {}",
                    copy.rows,
                    copy.bytes,
                    copy.source_offset,
                    copy.source_pitch,
                    source.region.len,
                    source.endpoint
                )));
            }
            let read = Piece {
                task,
                source: End::Remote(Arc::clone(source)),
                source_offset: copy.source_offset,
                destination: End::Given {
                    mapped: copy.destination,
                    hold: copy.hold,
                },
                destination_offset: copy.destination_offset,
                bytes: copy.bytes,
                rows: copy.rows,
                source_pitch: copy.source_pitch,
                destination_pitch: copy.destination_pitch,
            };
            let End::Given { mapped, hold } = &read.destination else {
                unreachable!("a READ's destination is this host's allocation");
            };
            if let (Some(fabric), Some(hold)) = (source.fabric, hold) {
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
            match hold {
                Some(hold) => hold
                    .session()
                    .and_then(|session| session.nixl_region(agent, mapped)),
                None => mapped.nixl_host_region(agent),
            }
            .map_err(self::copy)?;
            let link = Link {
                source: source.end.clone(),
                destination: mapped.end.clone(),
            };
            pieces.push((
                ThreadKey {
                    link,
                    runner: Runner::Network,
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
    /// `deployment`, and load the metadata of the agent that registered
    /// them: returns that agent's name, the exposure and each region's
    /// range. A server that cannot be reached fails it as
    /// [`TaskFailure::Unreachable`], one that refuses as
    /// [`TaskFailure::Refused`] with its error's kind.
    fn expose(
        &self,
        endpoint: &str,
        deployment: &str,
        regions: Vec<Region>,
    ) -> Result<(String, u64, Vec<Registration>), Failed> {
        let host = self.host(endpoint);
        let mut host = ::service::lock(&host);
        tracing::debug!(
            endpoint,
            deployment,
            regions = regions.len(),
            "asking another host to expose regions"
        );
        let (exposure, metadata, ranges) =
            host.expose(endpoint, deployment, regions)
                .map_err(|error| {
                    let failure = match &error {
                        ::storage::Error::Server { kind, .. } => {
                            TaskFailure::Refused { kind: *kind }
                        }
                        ::storage::Error::Io(_) | ::storage::Error::Protocol(_) => {
                            TaskFailure::Unreachable
                        }
                        _ => TaskFailure::Copy,
                    };
                    let message =
                        format!("{endpoint} did not expose the regions of {deployment}: {error}");
                    (failure, message)
                })?;

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
        Ok((peer, exposure, ranges))
    }
}

impl Host {
    /// One exposure on the connection to the server at `endpoint`, opened
    /// when there is none. A connection that fails is opened afresh once:
    /// the server may have ended and started again since the last task.
    fn expose(
        &mut self,
        endpoint: &str,
        deployment: &str,
        regions: Vec<Region>,
    ) -> ::storage::Result<(u64, Vec<u8>, Vec<Registration>)> {
        let mut retry = self.client.is_some();
        loop {
            let client = match &mut self.client {
                Some(client) => client,
                None => self.client.insert(Client::connect_peer(endpoint)?),
            };
            match client.expose(deployment.to_owned(), regions.clone()) {
                Err(error @ (::storage::Error::Io(_) | ::storage::Error::Protocol(_))) => {
                    self.client = None;
                    if !retry {
                        return Err(error);
                    }
                    retry = false;
                }
                // The server's response, an exposure or a refusal, leaves the
                // connection usable.
                responded => return responded,
            }
        }
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
        },
        End::Fabric {
            import,
            _hold: hold.clone(),
        },
    ))
}

/// Register `allocations` with `network`'s agent for other hosts' READs,
/// each with the hold on its GPU for device storage: each one's range, in
/// order, and the agent's metadata taken after the registrations, which
/// carries them.
pub(super) fn register(
    network: &Network,
    allocations: &[Exposed],
) -> Result<(Vec<u8>, Vec<Registration>), String> {
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
fn agent_name() -> std::io::Result<String> {
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
