//! The path between hosts outside one NVLink fabric: NIXL READs this host's
//! Transfer Server posts, from memory another host's Transfer Server
//! registered with its own agent.
//!
//! The process has one NIXL agent with the UCX backend, created at its first
//! use ([`Network::agent`]), so a host whose tasks never cross hosts never
//! loads NIXL. The agent serves both ends of a READ:
//!
//! - **Source.** The server has the Transfer Server register allocations a
//!   peer asked it to expose ([`Event::Register`]): each is registered once
//!   ([`Mapped::nixl_region`]) and stays registered, with its mapping, until
//!   the server takes the allocation back; the report carries each range and
//!   the agent's metadata taken after the registrations ([`register`]).
//! - **Destination.** A task's move from another host names that host's
//!   Storage Server by its endpoint. Before the task's pieces are queued,
//!   [`Network::pieces`] asks that server, over this process's connection to
//!   it as a [`Role::Peer`], to expose the task's regions there
//!   ([`Request::Expose`]), loads the metadata of the agent that registered
//!   them, checks each span against its region's range and registers each
//!   destination with this agent; network lanes then post the READs
//!   ([`super::lanes`]).
//!
//! An agent's name is unique to its process, so a peer whose Transfer
//! Server restarted answers with another agent: its metadata is loaded and
//! the previous agent's forgotten, and the new one registers the regions
//! afresh, since its server gives it every allocation again. Regions are
//! exposed for each task, not remembered between tasks: a region's
//! allocation, and its registration, may change between two tasks.
//!
//! [`Event::Register`]: storage::protocol::Event::Register
//! [`Request::Expose`]: storage::protocol::Request::Expose
//! [`Role::Peer`]: storage::protocol::Role::Peer

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use storage::Client;
use storage::protocol::{Link, Memory, Region, Registered};

use super::lanes::{End, LaneKey, Piece, Runner};
use super::memory::Mapped;

/// How long another host's Storage Server may take to accept the connection
/// and to answer an exposure: far longer than registering the largest
/// region takes there (pinning a host page pool of tens of GiB takes
/// seconds), so that only a host that stopped answering reaches it.
const PEER_TIMEOUT: Duration = Duration::from_secs(120);

/// The process's NIXL agent and its connections to other hosts' Storage
/// Servers.
pub(super) struct Network {
    /// NIXL's C API library, when the loader's search path does not find it.
    library: Option<PathBuf>,
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

/// A region of another host's memory that its Transfer Server registered,
/// which READs read.
pub(super) struct Source {
    /// The host's Storage Server, which names it in errors.
    pub(super) endpoint: String,
    /// The link end the region is: the host and its GPU or host memory.
    pub(super) memory: Memory,
    /// The agent that registered it.
    pub(super) peer: String,
    /// The range from the region's first byte to the end of its
    /// allocation, as that agent registered it.
    pub(super) region: nixl::Region,
}

/// A task's move from another host's memory into an allocation of this
/// host's, before that host exposed its source.
pub(super) struct Inbound {
    pub(super) endpoint: String,
    pub(super) deployment: String,
    pub(super) region: Region,
    pub(super) source_offset: u64,
    pub(super) destination: Arc<Mapped>,
    pub(super) destination_offset: u64,
    pub(super) bytes: u64,
}

impl Network {
    pub(super) fn new(library: Option<PathBuf>) -> Self {
        Self {
            library,
            agent: OnceLock::new(),
            hosts: Mutex::new(HashMap::new()),
        }
    }

    /// The process's NIXL agent, started at the first call; an error says
    /// why NIXL or UCX cannot start, for every task that needs them.
    pub(super) fn agent(&self) -> Result<&nixl::Agent, String> {
        self.agent
            .get_or_init(|| {
                if let Some(library) = &self.library {
                    nixl::load_library(library).map_err(|error| error.to_string())?;
                }
                let name = agent_name();
                let agent = nixl::Agent::new(&name)
                    .map_err(|error| format!("starting a NIXL agent: {error}"))?;
                tracing::info!(agent = name, "started the NIXL agent");
                Ok(agent)
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    /// The pieces of task `task`'s moves from other hosts, each on the lane
    /// of its link from the source host's memory: every host exposes the
    /// task's regions there in one request per deployment, and every
    /// destination is registered with this agent, so a task whose sources
    /// or destinations cannot be read or written fails before its first
    /// READ. Blocks on the other hosts' answers.
    pub(super) fn pieces(
        &self,
        task: u64,
        moves: Vec<Inbound>,
    ) -> Result<Vec<(LaneKey, Piece)>, String> {
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
            let (peer, ranges) = self.expose(&endpoint, &deployment, regions.clone())?;
            if ranges.len() != regions.len() {
                return Err(format!(
                    "{endpoint} exposed {} of {} regions",
                    ranges.len(),
                    regions.len()
                ));
            }
            for (region, range) in regions.into_iter().zip(ranges) {
                let source = Source {
                    memory: Memory::Remote {
                        endpoint: endpoint.clone(),
                        gpu: match range.memory {
                            Memory::Device { gpu } => Some(gpu),
                            _ => None,
                        },
                    },
                    region: source_region(&range)?,
                    endpoint: endpoint.clone(),
                    peer: peer.clone(),
                };
                sources.insert(
                    (endpoint.clone(), deployment.clone(), region),
                    Arc::new(source),
                );
            }
        }

        let agent = self.agent()?;
        let mut pieces = Vec::with_capacity(moves.len());
        for copy in moves {
            let Some(source) = sources.get(&(copy.endpoint, copy.deployment, copy.region)) else {
                return Err("a region was not exposed".into());
            };
            // Another host checked the region; the span is checked here,
            // against the range it registered from the region's first byte.
            let within = copy
                .source_offset
                .checked_add(copy.bytes)
                .is_some_and(|end| end <= source.region.len as u64);
            if !within {
                return Err(format!(
                    "a span's source of {} bytes at offset {} reaches past the {} bytes of its \
                     region at {}",
                    copy.bytes, copy.source_offset, source.region.len, source.endpoint
                ));
            }
            copy.destination.nixl_region(agent)?;
            let link = Link {
                source: source.memory.clone(),
                destination: copy.destination.memory.clone(),
            };
            pieces.push((
                LaneKey {
                    link,
                    runner: Runner::Network,
                },
                Piece {
                    task,
                    source: End::Remote(Arc::clone(source)),
                    source_offset: copy.source_offset,
                    destination: End::Given(copy.destination),
                    destination_offset: copy.destination_offset,
                    bytes: copy.bytes,
                },
            ));
        }
        Ok(pieces)
    }

    /// Have the Storage Server at `endpoint` expose `regions` of
    /// `deployment`, and load the metadata of the agent that registered
    /// them: returns that agent's name and each region's range.
    fn expose(
        &self,
        endpoint: &str,
        deployment: &str,
        regions: Vec<Region>,
    ) -> Result<(String, Vec<Registered>), String> {
        let host = Arc::clone(
            self.hosts
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .entry(endpoint.to_owned())
                .or_default(),
        );
        let mut host = host.lock().unwrap_or_else(PoisonError::into_inner);
        let (metadata, ranges) = host
            .expose(endpoint, deployment, regions)
            .map_err(|error| {
                format!("{endpoint} did not expose the regions of {deployment}: {error}")
            })?;

        let agent = self.agent()?;
        let peer = agent.load_peer(&metadata).map_err(|error| {
            format!("loading the metadata of the NIXL agent of {endpoint}: {error}")
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
        Ok((peer, ranges))
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
    ) -> storage::Result<(Vec<u8>, Vec<Registered>)> {
        let mut retry = self.client.is_some();
        loop {
            let client = match &mut self.client {
                Some(client) => client,
                None => self
                    .client
                    .insert(Client::connect_peer(endpoint, PEER_TIMEOUT)?),
            };
            match client.expose(deployment.to_owned(), regions.clone()) {
                Err(error @ (storage::Error::Io(_) | storage::Error::Protocol(_))) => {
                    self.client = None;
                    if !retry {
                        return Err(error);
                    }
                    retry = false;
                }
                // The server's answer, an exposure or a refusal, leaves the
                // connection usable.
                answered => return answered,
            }
        }
    }
}

/// Register `allocations` with `network`'s agent for other hosts' READs:
/// each one's range, in order, and the agent's metadata taken after the
/// registrations, which carries them.
pub(super) fn register(
    network: &Network,
    allocations: &[Arc<Mapped>],
) -> Result<(Vec<u8>, Vec<Registered>), String> {
    let agent = network.agent()?;
    let ranges = allocations
        .iter()
        .map(|mapped| {
            let region = mapped.nixl_region(agent)?;
            Ok(Registered {
                memory: mapped.memory.clone(),
                key: region.key,
                address: region.address,
                bytes: region.len as u64,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let metadata = agent
        .metadata()
        .map_err(|error| format!("exporting the NIXL agent's metadata: {error}"))?;
    Ok((metadata, ranges))
}

/// A range another host's agent registered, as this agent's READs name it.
fn source_region(range: &Registered) -> Result<nixl::Region, String> {
    let memory = match range.memory {
        Memory::Host => nixl::Memory::Host,
        Memory::Device { .. } => nixl::Memory::Device,
        Memory::Remote { .. } => {
            return Err("another host exposed memory it does not hold".into());
        }
    };
    Ok(nixl::Region {
        memory,
        address: range.address,
        key: range.key,
        len: usize::try_from(range.bytes)
            .map_err(|_| "an exposed region exceeds the address space".to_owned())?,
    })
}

/// A name no other agent takes, on any host: this host's name, the process,
/// and when the agent started (a process id is reused after its process
/// ends).
fn agent_name() -> String {
    let started = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_nanos());
    format!(
        "uniserve-transfer-{}-{}-{started:x}",
        host_name(),
        std::process::id()
    )
}

/// This host's name, or `host` when it has none readable.
fn host_name() -> String {
    let mut name = [0_u8; 256];
    // SAFETY: gethostname writes at most `name.len()` bytes into `name`.
    if unsafe { libc::gethostname(name.as_mut_ptr().cast(), name.len()) } != 0 {
        return "host".into();
    }
    let length = name.iter().position(|byte| *byte == 0).unwrap_or(0);
    String::from_utf8_lossy(&name[..length]).into_owned()
}
