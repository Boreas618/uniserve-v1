//! The messages a Storage Server and its clients exchange.
//!
//! A connection opens with the client's [`Hello`], which states the protocol
//! version and the client's [`Role`], and the server's [`Welcome`]. Every
//! later client message is a [`Request`] with an identity of the client's
//! choosing; the server responds to requests in order on their connection,
//! each with one [`Response`] that repeats the identity, and sends
//! asynchronous completions as [`Event`]s on the same connection. A request
//! the server refuses is responded with [`Response::Error`], the protocol's
//! one error: an [`ErrorKind`] and a message.
//!
//! The protocol has one version per release ([`VERSION`]). The client states
//! its version at `Hello`, and the server serves a client of its own version
//! alone: a client of any other is refused with [`Refusal::Version`], which
//! names both, before it sends a request. The `Hello` and the `Welcome` are
//! the one part of the protocol that no release changes, so that a peer of
//! any version reads its refusal. Every refusal of a `Hello` is typed
//! ([`Refusal`]).
//!
//! The encoding is MessagePack with named fields, and every field is
//! required.
//!
//! The server stores bytes with DLPack metadata and never interprets them:
//! clients compute every [`Key`] and define what the bytes mean. The
//! protocol holds six objects, and each request acts on one of them.
//!
//! **Claim.** A daemon's hold on a deployment's storage on its host
//! ([`Claim`], [`Request::Claim`]). The claiming connection is the
//! deployment's head; the claim lasts while it is connected, and its token
//! is what the deployment's workers state in their [`Role`]. Storage belongs
//! to epochs: an epoch is the set of processes that executes one
//! configuration of a deployment, and a deployment runs at most two at once
//! while it updates. Each claim is made by one [`Claimant`] of its epoch (an
//! engine of the epoch, or the head on the epoch's behalf) and names the
//! scopes that claimant places on the host and the epochs it retires: the
//! server retains an epoch's storage of every scope some claimant of the
//! epoch names, so one claimant's claim never ends another's retention. What
//! an epoch publishes stays private to it until the head publishes the epoch
//! ([`Request::Publish`]). The head also
//! keeps a previous epoch as its weights alone ([`Request::KeepWeights`]) and
//! moves them to host storage and back ([`Request::Offload`],
//! [`Request::Restore`]).
//!
//! **Scope.** Who may map or adopt storage ([`Scope`]): one rank of one
//! worker group of one deployment. Each tensor is retained by the
//! `(scope, epoch)` pairs that use it ([`Retainer`]): a worker's requests
//! retain and release under its own epoch alone, so an epoch that starts
//! beside another neither takes nor releases the other's storage.
//!
//! **Tensor.** Read-only retained storage with a key
//! ([`TensorRecord`] with [`Access::ReadOnly`] over a [`StorageRecord`]):
//! weights, shared by every process that maps them. Published with
//! [`Request::Put`], mapped with [`Request::Get`] ([`Mode::Read`]), kept with
//! [`Request::Retain`].
//!
//! **Storage.** Exclusive storage with one writer
//! ([`TensorRecord`] with [`Access::Exclusive`]): cache segments, buffer
//! pools, latent buffers. Its creator writes it; a successor of the same
//! scope adopts it ([`Request::Get`] with [`Mode::Write`], once the previous
//! writer is out) or maps it in standby and is promoted
//! ([`Mode::Standby`], [`Request::Promote`]); a writer gives it up with
//! [`Request::Release`].
//!
//! **Quota.** A GPU's limit for the deployment and the reservations against
//! it ([`Quota`]): the limit is `⌊total × fraction⌋` of the whole GPU for the
//! fraction the deployment's workers configure (`mem_fraction_static`). The
//! server reserves what it derives from the device storage it retains (the
//! weights, each KV cache segment, the other exclusive storages), each
//! allocation once whatever number of epochs map it; the processes reserve
//! what they hold themselves ([`Request::Quota`]): a worker's graph pool,
//! process storage and staging, and the Transfer Server's staging and its
//! CUDA context while it holds one on the GPU. The server refuses a
//! reservation or a new device storage that would take a GPU's usage past
//! its limit, so the two epochs of a deployment and the host's Transfer
//! Server stay within one quota on a GPU they share ([`Request::Usage`]).
//!
//! **Task.** A copy of [`Span`]s between [`Region`]s, run by the Transfer
//! Server ([`Role::Transfer`]); the server copies no byte itself. A client
//! submits a task with a [`Priority`] ([`Request::Submit`]), the server
//! resolves each region to one of its allocations and forwards the task, and
//! relays the Transfer Server's progress and the task's end to the client as
//! events ([`Event::TaskProgress`], [`Event::TaskEnded`]). The submitter
//! keeps a task's sources unchanged and its destinations unread until the
//! task ends; the Transfer Server reports the end only after it saw every
//! copy complete on the host. Tasks take their turns by priority on each
//! [`Link`], within the link's byte budget, which a head halves and restores
//! ([`Request::Throttle`]).
//!
//! **Session.** An engine's session with the buffer pool of its
//! deployment on the host (`crate::runtime`): the engine's main process, a
//! [`Role::Head`] of its epoch that states its launch generation, opens it
//! with [`Request::Open`] and gets one queue pair per scheduler, on which
//! every request of the buffer pool travels in shared memory; it ends it
//! with [`Request::Close`] once its workers have stopped, and the server
//! closes it when the engine's processes are gone. An epoch has a session
//! per engine of it on the host: the serving engine's and its standby's,
//! or a lost engine's and its relaunch's. Each rank states where its part of a device region's
//! units lies ([`Request::Place`]), or the head states it for a rank of
//! another host ([`Request::PlaceRemote`]), and each rank takes its part of
//! a host region, which the server creates ([`Request::HostPart`]). The
//! head grants a device region's memory space each segment its ranks
//! created ([`Request::Grant`]) and withdraws one before they release it
//! ([`Request::Withdraw`]). The server reads the host copies of other hosts
//! that serve the deployment for a prefetch from the peers; another host's
//! Storage Server holds the host copies it reads ([`Request::PeerHold`],
//! [`Request::PeerRelease`]). A rank of another host has its part of a host
//! region on its own host, so a put or a prefetch of the session's that
//! touches it runs on that host: its Transfer Server copies into the part
//! ([`Request::Submit`]) and its connector moves the part's pages
//! ([`Request::Pages`]), each asked by this server as a head of the
//! deployment.
//!
//! A span's source may lie on another host ([`Region::Remote`]): the
//! destination host's Transfer Server READs it through NIXL. It asks the
//! source host's server, over that server's TCP endpoint as a
//! [`Role::Peer`], to expose the regions ([`Request::Expose`]); that server
//! has its own Transfer Server register their allocations with its NIXL
//! agent ([`Event::Register`], [`Request::Registered`]) and responds with the
//! agent's metadata and each region's address ([`Response::Exposed`]). The
//! exposure lasts until the reading Transfer Server ends it, once its task
//! ended ([`Request::Unexpose`]), or its connection closes: the source's
//! Transfer Server then lets the registrations go ([`Event::Deregister`]),
//! and with them the GPU contexts they held. A peer connection states no
//! credential: the TCP endpoint is unauthenticated, and a host relies on the
//! isolation of the network it listens on to keep other processes from
//! reading the host tensors and host copies it serves.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::runtime::{Open, Opened};

/// The protocol version this build's clients state and its Storage Server
/// speaks: the one version either accepts. Each process that talks to the
/// server checks it at its own `Hello`, independently of the other
/// interfaces it speaks.
pub const VERSION: u32 = 27;

/// How long a server of another host may take to answer a request it answers from what it holds
/// (a hello, a peer's hold and release, a task's submission or cancellation, a connector's pages):
/// a host whose server does not answer within it is taken as unresponsive, so a prefetch skips
/// the peer and a transfer there fails rather than holding its task. Connecting to the host is
/// bounded alike. A server of this host has no such bound: its end closes the connection.
pub const ANSWER: Duration = Duration::from_secs(2);

/// How long a server of another host may take to answer an exposure ([`Request::Expose`]),
/// which it answers once its Transfer Server registered the regions with its NIXL agent: pinning
/// a host region's part of tens of GiB takes seconds there, and the bound lies far above it, so
/// that only a host that stopped responding reaches it.
pub const EXPOSURE: Duration = Duration::from_secs(120);

/// A content address: the 32-byte digest of everything that determines a
/// tensor's bytes, computed by the client.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Key(pub [u8; 32]);

impl Key {
    /// The key's lowercase hexadecimal form.
    pub fn hex(&self) -> String {
        self.0.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}

impl std::fmt::Debug for Key {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Key({})", self.hex())
    }
}

/// The owner of a tensor's lifetime: one worker of one worker group of one
/// deployment.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Scope {
    /// The served model name the deployment's head serves under.
    pub deployment: String,
    /// The worker group.
    pub group: String,
    /// The worker's rank within its group.
    pub rank: u32,
}

impl std::fmt::Display for Scope {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{}/{}/{}",
            self.deployment, self.group, self.rank
        )
    }
}

/// The token a deployment's claim carries: the server issues one when a head
/// first claims the deployment and keeps it for every later claim of that
/// head, of whatever epoch, and the head passes it to its workers, which
/// state it in their [`Role`]. A worker that states another token belongs to
/// a previous head of the deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Claim(pub [u8; 16]);

/// Who makes a claim of an epoch ([`Request::Claim`]): the server retains an
/// epoch's storage of each scope that some claimant of the epoch names, so a
/// claimant's claim ends no other claimant's retention, and a claimant ends
/// its own by claiming again without the scope (or the epoch's retirement
/// ends every claimant's).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Claimant {
    /// The head itself, on behalf of the epoch as a whole: an update's claims,
    /// which name the scopes of every engine the epoch keeps on the host.
    Head,
    /// One engine process of the epoch, by the number its head started it
    /// under: each independent replica, a relaunch of a lost engine and a
    /// standby claim the scopes they place.
    Engine(u64),
}

/// What a client is to the server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// A head of the deployment: the holder of its claim on this host (the
    /// host's daemon), or an engine of the deployment, which sends the page
    /// requests.
    ///
    /// An engine states its `epoch`: the pages it commits are that epoch's,
    /// and it reads its own epoch's pages and those every epoch shares.
    /// Uncommitted epochs keep pages and tensors private until publication;
    /// retiring such an epoch discards its pages.
    ///
    /// An engine's main process also states its launch `generation`, the
    /// one its workers state ([`Role::Worker`]), which names its session
    /// among the sessions of its epoch; zero for a head that opens none.
    Head {
        deployment: String,
        epoch: u64,
        generation: u64,
    },
    /// A worker of `scope` in `epoch`, started by the launch `generation` of
    /// its group, for the head that holds `claim`.
    Worker {
        scope: Scope,
        generation: u64,
        claim: Claim,
        /// The epoch the worker belongs to: the head's identity of the
        /// processes that execute one configuration of the deployment.
        epoch: u64,
    },
    /// The Mooncake connector the server runs.
    Connector,
    /// The Transfer Server the server runs, which copies bytes between the
    /// storage the server holds. Every quota of a GPU it uses reserves its CUDA
    /// context there, as it measured it ([`Request::Measured`]), and any
    /// staging it reserves ([`Request::Quota`]).
    Transfer,
    /// A process of another host, connected to the server's TCP endpoint:
    /// that host's Transfer Server, which READs storage of this host through
    /// NIXL for the tasks of its own host and asks the server to expose
    /// regions ([`Request::Expose`]), or that host's Storage Server, which
    /// holds the host copies of a buffer pool it reads
    /// ([`Request::PeerHold`]). A peer states no credential: the
    /// endpoint is unauthenticated and relies on network isolation.
    Peer,
    /// An operator's client, which lists what the server holds and retires
    /// deployments that are no longer served.
    Operator,
}

/// The first message on a connection: the client's [`VERSION`] and its role.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hello {
    pub version: u32,
    pub role: Role,
}

/// The server's response to [`Hello`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Welcome {
    /// The server serves the client in this build's [`VERSION`].
    Accepted,
    /// The server does not serve this client, for the reason stated.
    Refused { refusal: Refusal },
}

/// Why a server refuses a client's [`Hello`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
#[serde(rename_all = "snake_case")]
pub enum Refusal {
    /// The client states another version of the protocol than the server's:
    /// `offered` is the client's, `spoken` the server's, the only one it
    /// serves.
    #[error("the client speaks version {offered} of the storage protocol, the server {spoken}")]
    Version { offered: u32, spoken: u32 },
    /// The connection said hello before.
    #[error("the connection already said hello")]
    Repeated,
    /// A connection from another host states a role other than a head's or a
    /// peer's.
    #[error("a connection from another host is a head's or a peer's")]
    Remote,
    /// A connector says hello while the server has one.
    #[error("the server has a connector")]
    Connector,
    /// A Transfer Server says hello that the server does not take, for the
    /// reason given.
    #[error("the server does not take this Transfer Server: {reason}")]
    Transfer { reason: String },
}

/// Where a storage's physical storage is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Location {
    /// A CUDA VMM allocation on the GPU with this UUID, held through a POSIX
    /// file descriptor handle.
    Device { gpu: [u8; 16] },
    /// A host memfd, bound to a NUMA node when the client stated one.
    Host { numa_node: Option<u32> },
}

/// One physical allocation. Its descriptor travels with the message that
/// names it, in the order the message lists its storages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StorageRecord {
    pub location: Location,
    /// The allocation's size, a multiple of the device's allocation
    /// granularity for device storage.
    pub bytes: u64,
    /// The KV cache segment the storage is part of, which its GPU's quota
    /// reserves as that segment ([`Item::CacheSegment`]); `None` for any other
    /// storage.
    pub segment: Option<Segment>,
    /// The fabric handle the device allocation's creator exported, when it
    /// holds the allocation and its GPU belongs to an NVLink fabric
    /// (multi-node NVLink): another host of the fabric imports the handle
    /// to copy the storage over NVLink while the creator holds it. `None`
    /// for host storage, for an allocation without one, and for every
    /// storage of a client that states none.
    pub fabric: Option<FabricHandle>,
}

/// A device allocation's 64-byte fabric handle (`CU_MEM_HANDLE_TYPE_FABRIC`),
/// which only the process that created the allocation exports, and which
/// imports only while that process holds the allocation.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FabricHandle(pub [u8; 64]);

impl std::fmt::Debug for FabricHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "FabricHandle(")?;
        for byte in &self.0[..8] {
            write!(formatter, "{byte:02x}")?;
        }
        write!(formatter, "..)")
    }
}

impl Serialize for FabricHandle {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // As a sequence of bytes, the encoding of `Region::Fabric`'s handle.
        self.0.as_slice().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for FabricHandle {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes = Vec::<u8>::deserialize(deserializer)?;
        let handle = <[u8; 64]>::try_from(bytes.as_slice())
            .map_err(|_| serde::de::Error::invalid_length(bytes.len(), &"64 bytes"))?;
        Ok(Self(handle))
    }
}

/// One segment of a KV cache's value planes: the epoch that created its
/// storage and its ordinal there. A segment is one storage per layer and
/// plane, which a quota reserves together.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Segment {
    pub epoch: u64,
    pub ordinal: u32,
}

/// What one entry of a GPU's quota is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Item {
    /// Read-only device storages the server retains: an epoch's weights.
    Weights,
    /// The device storages of one KV cache segment.
    CacheSegment,
    /// Exclusive device storages the server retains beside the KV cache's
    /// value planes: the cache's flags and scales, the buffer pool, the
    /// latent buffers.
    Exclusive,
    /// A worker's CUDA graph storage pool: the storage its captured graphs
    /// address. Reserved by the worker.
    GraphPool,
    /// What a worker process holds on the device beside the storages the
    /// server retains and its graph pool: its CUDA context, communicator
    /// buffers, PyTorch's allocations, Rust's arenas and tables, the
    /// product storage it has yet to make resident, and the device storage
    /// a profiling session's collectors hold while it runs. Reserved by the
    /// worker.
    Process,
    /// Device storage a process holds to stage transfers. Reserved by the
    /// process.
    Staging,
    /// The device storage of the Transfer Server's CUDA context, at the
    /// bytes it measured when it started ([`Request::Measured`]). Reserved by
    /// the Transfer Server before it retains the context on the GPU, which
    /// it holds only while it has work there, and ended when it released
    /// it.
    TransferContext,
}

impl Item {
    /// Whether a process reserves the item itself ([`Request::Quota`]), rather
    /// than the server deriving it from the storages it retains.
    pub fn of_process(self) -> bool {
        matches!(
            self,
            Self::GraphPool | Self::Process | Self::Staging | Self::TransferContext
        )
    }

    /// The item's name, as listings print it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Weights => "weights",
            Self::CacheSegment => "cache_segment",
            Self::Exclusive => "exclusive",
            Self::GraphPool => "graph_pool",
            Self::Process => "process",
            Self::Staging => "staging",
            Self::TransferContext => "transfer_context",
        }
    }
}

/// One deployment's quota of one GPU: everything the deployment's
/// epochs hold on it, and the Transfer Server's items there, beside which
/// every deployment that uses the GPU holds its storage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Quota {
    /// The GPU's UUID.
    pub gpu: [u8; 16],
    /// The deployment; empty for a GPU only the Transfer Server reserves.
    pub deployment: String,
    /// The GPU's total storage, as the processes that use it state it; 0
    /// until one does.
    pub total: u64,
    /// `⌊total × fraction⌋` for the smallest fraction an epoch of the
    /// deployment states for the GPU; `None` until one states one.
    pub limit: Option<u64>,
    pub reservations: Vec<Reservation>,
    /// The bandwidth of copies from host storage to the GPU, in bytes per
    /// second, as the host's Transfer Server measured it when it started;
    /// `None` while none did. What returns from host storage to the GPU
    /// takes its bytes over this time.
    pub host_link_bytes_per_second: Option<u64>,
    /// The device storage the Transfer Server's CUDA context takes on the
    /// GPU while it holds one, as it measured it when it started; 0 when no
    /// Transfer Server measured the GPU. The quota reserves it as an entry
    /// ([`Item::TransferContext`]) only while the Transfer Server holds the
    /// context, from its first work on the GPU to its last, so a division of the GPU that
    /// leaves room for transfers counts it whether or not it is reserved.
    pub transfer_context: u64,
}

/// One entry of a [`Quota`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reservation {
    pub item: Item,
    /// The worker scope that holds it; `None` for the Transfer Server's
    /// items and for storages several scopes hold.
    pub scope: Option<Scope>,
    /// The epochs that hold it: one for an epoch's own, two for the storages
    /// a successor epoch maps of its predecessor, which are reserved once;
    /// none for the Transfer Server's items.
    pub epochs: Vec<u64>,
    /// The segment of an [`Item::CacheSegment`].
    pub segment: Option<Segment>,
    pub bytes: u64,
}

impl Quota {
    /// The bytes every reservation takes.
    pub fn usage(&self) -> u64 {
        self.bytes(|_| true)
    }

    /// The bytes of the entries `select` picks.
    pub fn bytes(&self, select: impl Fn(&Reservation) -> bool) -> u64 {
        self.reservations
            .iter()
            .filter(|reservation| select(reservation))
            .map(|reservation| reservation.bytes)
            .sum()
    }

    /// Every entry, one per line, for a refusal or a log.
    pub fn describe(&self) -> String {
        let limit = self.limit.map_or_else(
            || "no limit".to_owned(),
            |limit| format!("a limit of {limit}"),
        );
        let holder = if self.deployment.is_empty() {
            "the Transfer Server"
        } else {
            self.deployment.as_str()
        };
        let mut lines = vec![format!(
            "GPU {} of {holder}: {} bytes reserved of {limit}",
            gpu_name(&self.gpu),
            self.usage(),
        )];
        for reservation in &self.reservations {
            lines.push(format!("  {}", reservation.describe()));
        }
        lines.join("\n")
    }
}

impl Reservation {
    /// The entry on one line: its item, holder and bytes.
    pub fn describe(&self) -> String {
        let epochs: Vec<String> = self.epochs.iter().map(u64::to_string).collect();
        let holder = match (&self.scope, epochs.is_empty()) {
            (None, true) => "the Transfer Server".to_owned(),
            (Some(scope), _) => format!("{scope} of epoch {}", epochs.join(" and ")),
            (None, false) => format!("epoch {}", epochs.join(" and ")),
        };
        let segment = self
            .segment
            .map(|segment| format!(" {}/{}", segment.epoch, segment.ordinal))
            .unwrap_or_default();
        format!(
            "{}{segment} of {holder}: {} bytes",
            self.item.name(),
            self.bytes
        )
    }
}

/// A GPU's UUID as `nvidia-smi` prints it: `GPU-` and its bytes in groups
/// of 8, 4, 4, 4 and 12 hexadecimal digits.
pub fn gpu_name(gpu: &[u8; 16]) -> String {
    let hex: String = gpu.iter().map(|byte| format!("{byte:02x}")).collect();
    format!(
        "GPU-{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// What a host's Storage Server admits of new host storage (host regions,
/// staging, weights moved to the host): the host's available storage less
/// the server's reserve and what its retained storage has yet to allocate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostStorage {
    /// The host's `MemAvailable`.
    pub available: u64,
    /// What the server leaves to the host's other processes.
    pub reserve: u64,
    /// Unallocated bytes of host regions and retained host storages.
    pub unallocated: u64,
}

impl HostStorage {
    /// The bytes of new host storage the server admits now.
    pub fn room(&self) -> u64 {
        self.available
            .saturating_sub(self.reserve)
            .saturating_sub(self.unallocated)
    }
}

/// How a tensor may be accessed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Access {
    /// Immutable after `put`; any number of readers.
    ReadOnly,
    /// One writer at a time.
    Exclusive,
}

/// One tensor: a strided window of one storage, with DLPack metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorRecord {
    pub key: Key,
    /// A readable name, for listing and diagnostics.
    pub name: String,
    /// Index of the storage, in the message that carries this record.
    pub storage: u32,
    /// Byte offset of the first element in the storage.
    pub offset: u64,
    /// DLPack data type name, such as `bfloat16` or `float8_e4m3fn`.
    pub dtype: String,
    pub shape: Vec<u64>,
    /// Strides in elements.
    pub strides: Vec<u64>,
    pub access: Access,
}

/// What the pages of a pool the connector maps are (a rank's part of a
/// buffer pool's host region, [`Event::Pool`]), and what the source pages of
/// a converted load are: their byte layout, the kv heads a page holds and
/// its bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PageShape {
    /// The digest of a page's byte layout, computed by the worker: the page
    /// encoding version, the cache dtype, every cache group's block size,
    /// layers and heads, and the group the pool holds. The model's weights
    /// and code are not part of it: the pages of an engine and of its
    /// standby with other code share their pools, and their keys differ,
    /// since the content the worker reports (the model fingerprint with
    /// this layout) seeds them.
    pub layout: Key,
    /// The first kv head the pages hold.
    pub kv_head_offset: u32,
    /// The number of kv heads the pages hold.
    pub kv_head_count: u32,
    /// The bytes of one page.
    pub page_bytes: u64,
}

/// How a `get` takes the tensors it finds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// Map read-only; exclusive tensors are refused.
    Read,
    /// Become the writer of the exclusive tensors, once their previous
    /// writer is out; read-only tensors are taken for reading.
    Write,
    /// Map exclusive tensors without becoming their writer, to be promoted
    /// later.
    Standby,
    /// Map exclusive tensors writable without becoming their writer: the
    /// storage of a buffer pool region, which every session that names the
    /// region maps writable at once, and whose blocks the buffer pool, not
    /// a writer, keeps apart. No writer is recorded and none is shot down.
    Region,
}

/// Storage a transfer copies from or into, as its client names it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Region {
    /// The storage of the retained tensor `key`, which a scope of the
    /// caller's deployment holds. Offsets count from the tensor's first
    /// byte, and a copy may reach every byte of the storage from there (the
    /// other tensors packed in it included). Only an exclusive tensor is
    /// written: a read-only tensor is immutable. A key with copies at
    /// several locations names the copy at `at`; `None` names the key's
    /// only copy.
    Tensor { key: Key, at: Option<Location> },
    /// Rank `rank`'s part of the host region `region` of the caller's
    /// deployment's buffer pool ([`Request::HostPart`]); offsets count from
    /// unit 0, at `unit × bytes` for the part's bytes of a unit.
    Part { region: String, rank: u32 },
    /// A device allocation of another host that shares this host's NVLink
    /// fabric (multi-node NVLink): its 64-byte fabric handle, which the
    /// process that created the allocation exported, its GPU and size, and
    /// the byte offsets count from `offset`. The handle imports only while
    /// that process holds the allocation: CUDA exports no fabric handle
    /// from an allocation imported through its POSIX descriptor, which is
    /// how the Storage Server and its Transfer Server hold storage, and a
    /// fabric handle stops importing once its creator released the
    /// allocation. Copies import the handle into this host's Transfer
    /// Server.
    Fabric {
        handle: Vec<u8>,
        gpu: [u8; 16],
        bytes: u64,
        offset: u64,
    },
    /// Storage of another host that this host's Transfer Server READs
    /// through NIXL: `region`, a retained tensor or a host region's part of
    /// the caller's deployment, as a client of that host names it, at the
    /// Storage Server listening at `endpoint` (that server's
    /// [`Request::Status`], `peer`). Offsets count as they do in `region`. Only a
    /// span's source lies on another host, since the destination's host
    /// posts the READ. That host's server checks the region and its
    /// Transfer Server registers it once the task runs, so a region of
    /// another deployment, or a span past its end, fails the task instead
    /// of refusing it.
    Remote {
        endpoint: String,
        region: Box<Region>,
    },
}

/// One copy of a transfer: `rows` rows of `bytes` bytes, row `r` from
/// `source_offset + r * source_pitch` of `source` to `destination_offset +
/// r * destination_pitch` of `destination`. One row copies `bytes` bytes
/// and its pitches are unused; rows at a pitch, each at least `bytes`, copy
/// a strided region as one span (a column of a row-major matrix, a part of
/// every expert's rows), which the Transfer Server runs as pitched copies.
/// The spans of a task run in no fixed order, so no span writes bytes
/// another span of the task reads or writes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    pub source: Region,
    pub source_offset: u64,
    pub destination: Region,
    pub destination_offset: u64,
    pub bytes: u64,
    pub rows: u64,
    pub source_pitch: u64,
    pub destination_pitch: u64,
}

impl Span {
    /// The bytes the span copies: its rows' bytes.
    pub fn copied(&self) -> u64 {
        self.bytes.saturating_mul(self.rows)
    }
}

/// The order in which the tasks on one link take their turns, highest
/// first. A link runs a chunk of the highest priority that has one, and the
/// tasks of one priority in the order they were submitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Priority {
    /// What an epoch's promotion waits for: the data the in-flight requests
    /// wrote after the next epoch's store was warmed.
    Promotion,
    /// Warming the next epoch's store while the executing epoch runs: the
    /// pages the in-flight requests filled, copied ahead of the promotion.
    Warming,
    /// Weight moves and every other copy no promotion waits for.
    Background,
    /// What a request waits for at its admission: the pages a scheduler
    /// reads from another host's pools for a prompt. Such a load is short
    /// and bounded per request, so it ranks above a promotion, whose copies
    /// yield to it between chunks.
    Request,
}

impl Priority {
    /// The number of priorities.
    pub const COUNT: usize = 4;

    /// Every priority, highest first.
    pub const ORDER: [Priority; Self::COUNT] = [
        Priority::Request,
        Priority::Promotion,
        Priority::Warming,
        Priority::Background,
    ];

    /// The priority's position in [`Self::ORDER`]: 0 is the highest.
    pub fn rank(self) -> usize {
        match self {
            Priority::Request => 0,
            Priority::Promotion => 1,
            Priority::Warming => 2,
            Priority::Background => 3,
        }
    }
}

/// One end of a [`Link`]: host storage, the storage of one GPU, or storage of
/// another host.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkEnd {
    Host,
    /// The GPU with this UUID.
    Device {
        gpu: [u8; 16],
    },
    /// Storage of another host, which a READ reaches through the network
    /// adapters: the host whose Storage Server listens at `endpoint`, and
    /// there the GPU `gpu`, or host storage when `None`.
    Remote {
        endpoint: String,
        gpu: Option<[u8; 16]>,
    },
}

/// The path a copy takes, from the storage it reads to the storage it writes.
/// The Transfer Server measures each link within its host when it starts,
/// and a link from another host by its first copy long enough to time; it
/// cuts copies into chunks by the bandwidth, and keeps a byte budget per
/// link.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Link {
    pub source: LinkEnd,
    pub destination: LinkEnd,
}

/// A change of links' byte budgets, which the head of the executing epoch
/// makes when its scheduler steps slow down while transfers run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Throttle {
    /// Halve each link's budget, down to a floor; a restoration under way
    /// stops.
    Halve,
    /// Raise each link's budget back to its bandwidth, gradually.
    Restore,
}

/// How a transfer task ended.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    /// Every span was copied.
    Complete,
    /// The task was cancelled; its destinations hold any part of the spans.
    Cancelled,
    /// A span could not be copied, or the Transfer Server ended; its
    /// destinations hold any part of the spans.
    Failed { message: String },
}

/// One allocation of the server's, as the server names it to its Transfer
/// Server, or an allocation of another host the Transfer Server imports.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Allocation {
    /// The storage the server numbers `storage`.
    Storage { storage: u64 },
    /// A device allocation of another host of the NVLink fabric, `bytes`
    /// long, on the GPU `gpu`, by its fabric handle.
    Fabric {
        handle: Vec<u8>,
        gpu: [u8; 16],
        bytes: u64,
    },
    /// The region `region` of `deployment` at the Storage Server of another
    /// host listening at `endpoint`, as a [`Region::Remote`] names it: the
    /// Transfer Server has that server expose it ([`Request::Expose`]) and
    /// READs it.
    Remote {
        endpoint: String,
        deployment: String,
        region: Box<Region>,
    },
}

/// One copy the server gives its Transfer Server: a [`Span`] with its
/// regions resolved to allocations, and offsets from their first byte (from
/// the region's first byte, for another host's region), with the span's
/// rows and pitches.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Move {
    pub source: Allocation,
    pub source_offset: u64,
    pub destination: Allocation,
    pub destination_offset: u64,
    pub bytes: u64,
    pub rows: u64,
    pub source_pitch: u64,
    pub destination_pitch: u64,
}

/// A link's bandwidth as the Transfer Server measured it: when it started,
/// or by the first copy long enough to time, for a link from another host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinkBandwidth {
    pub link: Link,
    pub bytes_per_second: u64,
}

/// A range of storage a Transfer Server registered with its NIXL agent, as a
/// peer's READ names it: NIXL finds the registration in the agent's
/// metadata by the range's key, address and length.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Registration {
    /// Host storage, or the GPU with this UUID, of the registering host.
    pub end: LinkEnd,
    /// The key the agent registered the range under, which no other
    /// registration of the agent takes, so that a peer that loaded the
    /// metadata of a registration since undone at the same address does not
    /// mistake it for this one.
    pub key: u64,
    /// The address of the range's first byte in the registering process.
    pub address: u64,
    /// The range's length in bytes.
    pub bytes: u64,
    /// The fabric handle of the device allocation the range lies in, when
    /// its creator recorded one ([`StorageRecord::fabric`]): a reader whose
    /// GPU imports it copies the range over the NVLink fabric rather than
    /// READ it.
    pub fabric: Option<FabricExport>,
}

/// A device allocation's fabric handle, as an exposure names it: the
/// allocation's length, which an import maps whole, and the offset of the
/// exposed range's first byte in it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FabricExport {
    pub handle: FabricHandle,
    pub bytes: u64,
    pub offset: u64,
}

/// Where one run of a buffer pool region's units lies in one rank, for one
/// field of a unit: for each unit `u` in `first..first + count`, `bytes`
/// bytes at `offset + (u - first) * bytes` of the retained exclusive tensor
/// `key` at `location`. This is the content of a rank's cache plane run
/// (`engine::memory::cache::info::CachePlaneRun`), with the GPU stated as the location.
///
/// A unit is not one byte range: its value planes, flags, scales and state
/// planes are fields of their own. The runs of one field hold that field's
/// bytes of every unit once, and two ranks of one layout name the same
/// fields at the same bytes per unit, so a unit copies between them field
/// by field, from the run of one holding the unit to the run of the other.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnitRun {
    pub field: String,
    pub key: Key,
    pub location: Location,
    pub first: u32,
    pub count: u32,
    pub offset: u64,
    pub bytes: u64,
}

/// A host copy another host's Storage Server holds for its reads
/// ([`Request::PeerHold`]): the key, the tokens its block holds, the units
/// of the host region `region` that hold it, `count` from `first`, and the
/// encoding it is written in, as the class that wrote it states it
/// (`crate::runtime::Class::tensors` and `Class::encoding`). Each part of
/// the region ([`RegionPart`]) holds its bytes of those units. The reader
/// reads a copy only into a block of a class of the same encoding, or of a
/// class that converts from it (`crate::runtime::Class::source`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostCopy {
    pub key: Key,
    pub tokens: u16,
    pub region: String,
    pub first: u32,
    pub count: u32,
    pub tensors: [u8; 32],
    pub encoding: [u8; 32],
}

/// One rank's part of a host region: unit `u` lies at byte `u × bytes` of
/// the part, which a transfer names as [`Region::Part`]. The part lies on
/// the host whose Storage Server listens at `endpoint` when that is another
/// host than the answering server's: the part of a rank of another host of
/// the deployment's engine, which lies on that rank's host.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionPart {
    pub region: String,
    pub rank: u32,
    pub bytes: u64,
    pub endpoint: Option<String>,
}

/// The device storage the Transfer Server's CUDA context takes on one GPU,
/// as NVML attributes it to the process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextBytes {
    pub gpu: [u8; 16],
    pub bytes: u64,
}

/// A client's request. Head requests come from a [`Role::Head`] client and
/// worker requests from a [`Role::Worker`] client; `List` from any. The
/// connector's, the Transfer Server's and a peer's requests say which they
/// are.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Request {
    /// Make the caller the head of its deployment, keeping the claim it holds
    /// already, and state the scopes `claimant` places on the host for
    /// `epoch`, replacing those it stated before: the epoch's retention of a
    /// scope ends once no claimant of the epoch names it. Every epoch of
    /// `retire` ends on the host: its retention of every tensor ends, and a
    /// tensor no epoch retains is released. Refused while a previous head of
    /// the deployment is connected. Every process of the deployment that
    /// belongs to no current worker is shot down. Every other deployment
    /// whose head is not connected is retired: a host serves the deployments
    /// whose heads run on it.
    Claim {
        claimant: Claimant,
        scopes: Vec<Scope>,
        epoch: u64,
        retire: Vec<u64>,
    },
    /// Publish all listed committed epochs together. Claimed epochs remain
    /// private until this request, including during preparation and trial.
    /// Every listed epoch must be claimed by the head's deployment.
    Publish { epochs: Vec<u64> },
    /// Release every tensor the caller's scope retains in the caller's epoch
    /// that `keys` does not list; the copies of listed keys at the locations
    /// `at` lists are retained (every copy when `at` is empty).
    Retain { keys: Vec<Key>, at: Vec<Location> },
    /// Epochs in the head's deployment retain their weights and the exact
    /// `cache_keys` holding completed rollback copies. Their retention of
    /// every other exclusive tensor ends. A tensor no other epoch retains
    /// is released, while segments mapped by the successor remain retained
    /// by that successor. What their processes reserved on the GPUs' quotas
    /// ends, whether a process ended in order or was stopped, and nothing is
    /// kept for a process of theirs until one of them reserves again (a
    /// rollback's relaunch). Call after the predecessor's processes exit.
    /// Responded with `Retained`.
    KeepWeights {
        epochs: Vec<u64>,
        cache_keys: Vec<Key>,
    },
    /// From a head: epoch `epoch` of its deployment keeps its weights in
    /// host storage (`update_config.previous_epoch = "host"`, its processes
    /// exited). Every device copy of read-only tensors that this epoch
    /// alone retains moves to host storage: the server creates host storage of
    /// each storage's size, admitted under the host rule as one (refused,
    /// moving nothing, when the host cannot hold it all), has its Transfer
    /// Server copy each storage there, and then the tensors' copies lie in
    /// host storage and the device storages are released: each GPU's quota
    /// leaves their storage to the other epoch. Responded with `Moved` once
    /// every copy ended; refused while no Transfer Server runs.
    Offload { epoch: u64 },
    /// From a head: epoch `epoch`'s weights that `Offload` moved to host
    /// storage return to the GPUs they came from, before its processes start
    /// again: its Transfer Server creates device storage of each storage's
    /// size on its GPU, admitted against the GPU's quota (refused, moving
    /// nothing, past the quota), and copies each storage there; then the
    /// tensors' copies lie on those GPUs again and the host copies are
    /// released. Responded with `Moved` once every copy ended.
    Restore { epoch: u64 },
    /// The copies of `keys` that exist at the locations `at` lists (at
    /// every location when `at` is empty), with their storages'
    /// descriptors, and the keys with no such copy.
    Get {
        keys: Vec<Key>,
        mode: Mode,
        at: Vec<Location>,
    },
    /// Allocate an exclusive byte tensor in host storage after host
    /// admission. Returns its retained storage through `Got`, with write
    /// ownership belonging to this worker and lifetime to its epoch.
    AllocateHost { key: Key, name: String, bytes: u64 },
    /// Publish tensors whose bytes are written, over the storages this
    /// message carries. Read-only tensors become readable; exclusive ones
    /// keep the caller as their writer. A key names what a tensor's bytes
    /// are, and its storage's record where they lie: a key has at most
    /// one copy at each location. A read-only tensor already retained under
    /// the same key at the same location holds the same bytes: the put
    /// adds the caller's scope to it and the server keeps the retained one.
    Put {
        tensors: Vec<TensorRecord>,
        storages: Vec<StorageRecord>,
    },
    /// Make the caller the writer of exclusive tensors it holds in standby,
    /// once their previous writer is out.
    Promote { keys: Vec<Key> },
    /// Give up write access: the caller has drained its streams and
    /// unmapped or downgraded its mappings.
    Release { keys: Vec<Key> },
    /// Move the copies at the locations `at` lists (every copy when `at`
    /// is empty) of read-only tensors to new keys, each `(old, new)`: the
    /// caller rewrites their bytes in place, as an in-place weight update
    /// does, so the old keys no longer name them. Refused, moving none,
    /// unless every old key has such a copy, each a read-only tensor that
    /// the caller's deployment alone holds, and no new key has a copy at
    /// its location.
    Rekey {
        moves: Vec<(Key, Key)>,
        at: Vec<Location>,
    },
    /// Every tensor, storage and claim the server holds.
    List,
    /// Release every tensor and buffer pool of `deployment` and forget its
    /// claim: the deployment is no longer served. Refused while its head is
    /// connected.
    Retire { deployment: String },
    /// Pass the server to a successor server of `executable` (this
    /// server's own when `None`), which goes on with every connection, the
    /// listening sockets, the tensors' storage and the buffer pools once no
    /// request, shoot-down, connector task or transfer is in flight (a hot
    /// restart);
    /// the server then exits. Responded with the successor's process id
    /// once it started.
    Restart { executable: Option<String> },
    /// The server's readiness record: its endpoints (its socket, and the TCP endpoint other
    /// hosts connect to when it accepts them) and the settings it serves with. Responded with
    /// [`Response::Status`].
    Status,
    /// Stop the server, as SIGTERM does: it responds with [`Response::Stopping`], stops serving,
    /// ends its children and closes every connection, the operator's last, so the operator's
    /// client sees its connection close once the server ended. An operator's request.
    Stop,
    /// The connector ran task `task` of an [`Event::Upload`] or an
    /// [`Event::Download`]: one result per page of the task.
    Finished { task: u64, results: Vec<bool> },
    /// Reserve the caller's own items on GPU `gpu` (of total storage `total`):
    /// each item of `reservations` takes its bytes, 0 releasing it; an item
    /// left out keeps its reservation. A worker reserves [`Item::GraphPool`],
    /// [`Item::Process`] and [`Item::Staging`] against its deployment's quota
    /// under its scope and epoch, and states `limit`, `⌊total × fraction⌋`
    /// for its configured fraction. The processes of one scope and epoch (a
    /// rank, and the relaunch or standby that replaces it) serve in turn, so
    /// the larger of their reservations of an item counts. The Transfer
    /// Server reserves [`Item::Staging`] and [`Item::TransferContext`] against
    /// every quota of the GPU, stating no limit: its context before it
    /// retains it, which a quota without room refuses. A process that ends in
    /// order releases its reservations first; a worker's process that is lost
    /// leaves them reserved for its relaunch. Refused, reserving nothing,
    /// when it would take a GPU's usage past its limit; responded with the
    /// quotas of the GPU the caller reserves against ([`Response::Usage`]).
    Quota {
        gpu: [u8; 16],
        total: u64,
        limit: Option<u64>,
        reservations: Vec<(Item, u64)>,
    },
    /// The quotas of `gpus` (of every GPU when empty) and what the host
    /// admits of new host storage: of the caller's deployment for a worker
    /// and a head, of every deployment for the Transfer Server and an
    /// operator.
    Usage { gpus: Vec<[u8; 16]> },
    /// From a head or a worker: copy `spans` as one task of `update`, which
    /// takes its turn at `priority` on every link it uses. Responded with the
    /// task's number at once; [`Event::TaskProgress`] follows as it
    /// copies and [`Event::TaskEnded`] when it ends. Refused, copying
    /// nothing, when no Transfer Server runs, when a region is not storage
    /// of the caller's deployment, or when a span reaches past its region
    /// or writes a read-only tensor; a region of another host is checked
    /// by that host once the task runs ([`Region::Remote`]). A task ends
    /// with its submitter's connection, cancelled.
    Submit {
        spans: Vec<Span>,
        priority: Priority,
        /// The update (epoch change) the task serves, a number its
        /// deployment's head chooses; the tasks of an update that aborts
        /// are cancelled together.
        update: u64,
    },
    /// Hold the transfer contexts of these local GPUs across successive
    /// tasks. Contexts remain lazy and quota-accounted. The session ends
    /// explicitly or when its owning connection closes.
    OpenSession { gpus: Vec<[u8; 16]> },
    /// End a session of this connection. Submitted tasks keep their own
    /// holds until physical completion; this releases only the session.
    CloseSession { session: u64 },
    /// From a head or a worker: cancel `tasks` and, with `update`, every
    /// task of that update, of the caller's deployment. Responded at once;
    /// each task's [`Event::TaskEnded`] follows once the Transfer Server
    /// copies none of its bytes any more.
    Cancel {
        tasks: Vec<u64>,
        update: Option<u64>,
    },
    /// From a head: halve or restore the byte budget of each of `links`.
    /// With no links, a halving takes every link a task of the head's
    /// deployment copies over now (a deployment changes one epoch at a
    /// time, so they are its update's links), and a restoration every link
    /// whose budget a halving reduced.
    Throttle {
        links: Vec<Link>,
        throttle: Throttle,
    },
    /// From the Transfer Server: each link's bandwidth and the device
    /// storage its CUDA context takes on each GPU, once after it connected,
    /// for every link within its host, and again for a link from another
    /// host once a copy timed it. The server keeps the latest report of
    /// each link and each GPU.
    Measured {
        links: Vec<LinkBandwidth>,
        contexts: Vec<ContextBytes>,
    },
    /// From the Transfer Server: task `task` copied `bytes` so far, and
    /// ended as `end` when given.
    Copied {
        task: u64,
        bytes: u64,
        status: Option<TaskStatus>,
    },
    /// From another host's Transfer Server ([`Role::Peer`]): have this
    /// host's Transfer Server register `regions`, retained tensors or page
    /// pools of `deployment` on this host, with its NIXL agent, for that
    /// Transfer Server's READs. Responded with [`Response::Exposed`] once they
    /// are registered; refused when a region is not storage of the
    /// deployment, when no Transfer Server is connected, or when it cannot
    /// register them. The regions stay registered for the exposure until
    /// the peer ends it ([`Request::Unexpose`]) or its connection closes,
    /// the server takes their allocations back, or its Transfer Server
    /// ends. Whoever reads them keeps them retained and unchanged until the
    /// read ends: the submitter of the task that reads them.
    Expose {
        deployment: String,
        regions: Vec<Region>,
    },
    /// From another host's Transfer Server: the reads of the peer's exposure
    /// `exposure` ended. This host's Transfer Server lets its registrations
    /// go once no other exposure holds them. An exposure the server does
    /// not hold for the caller (it ended with its Transfer Server, or the
    /// caller's previous connection) gets its response alike.
    Unexpose { exposure: u64 },
    /// From the Transfer Server: the allocations of the server's
    /// [`Event::Register`] `request` are registered with its NIXL agent:
    /// each allocation's range, in the order the event listed them, and
    /// the agent's `metadata` taken after the registrations; or `failure`
    /// says why they are not.
    Registered {
        request: u64,
        metadata: Vec<u8>,
        ranges: Vec<Registration>,
        failure: Option<String>,
    },
    /// From the Transfer Server: the device storage of the server's
    /// [`Event::Allocate`] `request`, `bytes` long, whose POSIX descriptor
    /// travels with the request; or `failure` says why there is none.
    Allocated {
        request: u64,
        bytes: u64,
        failure: Option<String>,
    },
    /// From an engine's main process, a [`Role::Head`] of the epoch the
    /// layout states, connected on the host: open the engine's session with
    /// the buffer pool of its deployment (`specs/buffer-pool.md` §11), one of
    /// its epoch's sessions, named by the head's launch generation.
    /// Responded with [`Response::Opened`], with the descriptor of each
    /// scheduler's queue pair, in the order of [`Open::schedulers`]; the
    /// server's end of each is served from then on. A region another
    /// session of the deployment stated under the same name is that
    /// region, and groups that state one device memory read and write the
    /// same memory. Refused, opening nothing, when the server does not
    /// serve the interface version the layout states, when the engine's
    /// session is open (or its processes have not left since it lost its
    /// connection), or when the layout does not fit the buffer
    /// pool: a region stated again with another tier, space, unit size or
    /// numbering, or more than eight device memories over the deployment's
    /// sessions. A host region's storage is admitted part by part, as each
    /// rank of this host takes its part ([`Request::HostPart`]), not here.
    Open(Open),
    /// From the engine's main process that opened its session: its
    /// workers have stopped, and the session closes as the session of an
    /// engine that is gone (`specs/buffer-pool.md` §12.4). Responded with
    /// [`Response::Closed`].
    Close,
    /// From a worker: where its part of the units of the device region
    /// `region` of its deployment's buffer pool lies, in the device memory
    /// `memory` its group addresses (the memory a group of the layout
    /// states), field by field. `part` is the rank's index in its worker
    /// group, which holds the same share of every unit in every group of
    /// the layout. It replaces what was stated before for the region, memory
    /// and part. The server turns a task's runs into copies with it,
    /// matching the ranks of two memories by their part. For a host region,
    /// `memory` names the host the part the rank took ([`Request::HostPart`])
    /// lies on, as the layout's host parts name it
    /// (`runtime::Group::parts`), with no run: a task reaches a host part
    /// only where a place states it. Responded with [`Response::Placed`].
    Place {
        region: String,
        memory: String,
        part: u32,
        runs: Vec<UnitRun>,
    },
    /// From the head of an epoch whose session is open, on the host: where
    /// the part `part` of the units of the device region `region` lies in
    /// the device memory `memory` for its rank of another host (the rank
    /// whose index in its worker group is `part`, [`Request::Place`]), as
    /// that rank stated it to its engine. The runs name tensors retained at
    /// that host's Storage Server, listening at `endpoint`, whose Transfer
    /// Server reaches them: a task writing them runs there, and one reading
    /// them here READs them from there. It replaces what was stated before
    /// for the region, memory and part. For a host region, `memory` names
    /// the host, as the rank's [`Request::Place`] there does, with no run.
    /// Responded with [`Response::Placed`].
    PlaceRemote {
        region: String,
        memory: String,
        part: u32,
        endpoint: String,
        runs: Vec<UnitRun>,
    },
    /// From the head of an epoch whose session is open, on the host: the
    /// engine's ranks created the storages of `segment` of the device
    /// region `region`, units `first` to `first + units` of its memory
    /// space, on every rank that maps the region. The units the region
    /// may allocate among them join the space's free ranges, from which a
    /// resize grows its regions, and the segment joins the space's table,
    /// which [`Response::Opened`] states to every later session. Responded
    /// with [`Response::Granted`]; refused, changing nothing, when the
    /// region is no device region of the deployment's buffer pool, the
    /// units lie past its numbering, or the segment or a unit of it is in
    /// the table, the free ranges or a region already.
    Grant {
        region: String,
        segment: Segment,
        first: u32,
        units: u32,
    },
    /// From the head of an epoch whose session is open, on the host: the
    /// engine's ranks are to release the storages of `segment` of the
    /// memory space of the device region `region`. Its units leave the
    /// space's free ranges and the segment leaves its table, so no region
    /// grows into them again until a grant. Responded with
    /// [`Response::Withdrawn`], after which the ranks may unmap and release
    /// the storages; refused, changing nothing, while a unit of it is in a
    /// region (a resize shrinks the regions first), or when the table holds
    /// no such segment.
    Withdraw { region: String, segment: Segment },
    /// From a worker: part `part` of the host region `region` of its
    /// deployment's buffer pool, `bytes` of every unit, whose memfd the
    /// server creates at the first request (on `numa_node` when given) and
    /// holds until the deployment is released. `part` is the rank's index in
    /// its worker group: the ranks of one index, in every group and replica
    /// of the deployment, hold the same share of every unit and share the
    /// part. The part is admitted whole, the region's numbering of `bytes`
    /// each, when it is created: a region whose ranks lie on several hosts
    /// holds on each only that host's parts. Responded with
    /// [`Response::HostPart`], whose memfd travels with it; refused when the
    /// region is not a host region an open session stated, when the parts'
    /// bytes of a unit exceed the region's unit size, or when the host's
    /// storage cannot hold a new part.
    HostPart {
        region: String,
        part: u32,
        bytes: u64,
        numa_node: Option<u32>,
    },
    /// From another host's Storage Server ([`Role::Peer`]): hold the host
    /// copies of `keys` in `deployment`'s buffer pool that every epoch may
    /// use, for that host to READ, until its [`Request::PeerRelease`] or
    /// its connection closes. A held copy is a holder of its block like
    /// any other, so it is not evicted. Responded with
    /// [`Response::PeerHeld`], naming the copies held; keys without such a
    /// copy are left out.
    PeerHold { deployment: String, keys: Vec<Key> },
    /// From another host's Storage Server: its reads of the copies it holds
    /// in `deployment`'s buffer pool ended, and every hold ends. Responded
    /// with [`Response::Valid`] when the holds stood until now, so what it
    /// read is what the copies held, and with [`Response::Gone`] when this
    /// connection holds nothing there; the reading host commits what it
    /// read only on `Valid`.
    PeerRelease { deployment: String },
    /// From a head of the deployment, which is another host's Storage
    /// Server running a task of its buffer pool that touches the part of a
    /// rank of this host (`specs/buffer-pool.md` §15): have this host's
    /// connector write `pages` of part `part` of the host region `region` to
    /// the KV storage (`upload`), or read them from it into their units, each
    /// page a key and the unit that holds it. Responded with
    /// [`Response::Submitted`] and the task's number at once;
    /// [`Event::PagesMoved`] follows once the connector ran the task, or once
    /// a connector lost meanwhile has exited. Refused when this host holds no
    /// such part, when a unit lies past the region's numbering, or when no
    /// connector is connected.
    Pages {
        region: String,
        part: u32,
        pages: Vec<(Key, u32)>,
        upload: bool,
    },
}

/// Why a request failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    /// The request does not fit the caller's role or names what does not
    /// exist.
    Invalid,
    /// The deployment's previous head is still connected.
    Claimed,
    /// A tensor's writer is a live worker of the current head.
    Held,
    /// A previous writer did not leave the GPU after its shoot-down; the GPU
    /// needs a reset.
    ShootDown,
    /// The server failed to hold or pass a descriptor, or its Transfer
    /// Server to register storage with its NIXL agent.
    Resource,
    /// No Transfer Server is connected: it has not started yet, or it
    /// ended and the server starts it again.
    Unavailable,
    /// A reservation or a new device storage would take a GPU's quota past
    /// its quota.
    Quota,
}

impl Request {
    /// How long a server of another host may take to answer this request: [`EXPOSURE`] for an
    /// exposure, [`ANSWER`] for every other request a client of another host makes.
    pub fn remote_deadline(&self) -> Duration {
        match self {
            Self::Expose { .. } => EXPOSURE,
            _ => ANSWER,
        }
    }
}

/// The server's response to one request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Response {
    Published,
    Claimed {
        claim: Claim,
    },
    Retained {
        released: u64,
    },
    /// What a [`Request::Get`] or a [`Request::AllocateHost`] found; the
    /// storages' descriptors travel with the response, in order.
    Tensors(Tensors),
    Put,
    Promoted,
    Released,
    Rekeyed,
    Listing(Listing),
    Retired {
        released: u64,
    },
    /// The successor of a [`Request::Restart`] started.
    Restarting {
        pid: u32,
    },
    /// What a [`Request::Status`] asks.
    Status(Box<crate::settings::Readiness>),
    /// The server stops ([`Request::Stop`]).
    Stopping,
    Finished,
    /// What a [`Request::Quota`] or a [`Request::Usage`] names.
    Usage(Usage),
    /// The task a [`Request::Submit`] or a [`Request::Pages`] submitted.
    Submitted {
        task: u64,
    },
    SessionOpened {
        session: u64,
    },
    SessionClosed,
    /// How many tasks a [`Request::Cancel`] cancelled.
    Cancelled {
        tasks: u64,
    },
    Throttled,
    /// The server took a report of its Transfer Server's
    /// ([`Request::Measured`], [`Request::Copied`], [`Request::Registered`]).
    Noted,
    /// What an `Offload` or a `Restore` moved: its storages and their bytes.
    Moved {
        storages: u64,
        bytes: u64,
    },
    /// The regions of a [`Request::Expose`], registered with the NIXL agent
    /// of this host's Transfer Server for the exposure `exposure`: the
    /// agent's `metadata`, which a READ's agent loads to reach them, and
    /// each region's range, in order, from the region's first byte to the
    /// end of its allocation.
    Exposed {
        exposure: u64,
        metadata: Vec<u8>,
        regions: Vec<Registration>,
    },
    /// A [`Request::Unexpose`] was taken.
    Unexposed,
    /// The session a [`Request::Open`] opened. The descriptor of each
    /// scheduler's queue pair travels with it, in the order of
    /// [`Open::schedulers`].
    Opened(Opened),
    Closed,
    Placed,
    Granted,
    Withdrawn,
    /// The rank's part of a host region, `units` units of the bytes the
    /// request stated: unit `u` at byte `u × bytes`. Its memfd travels with
    /// the response.
    HostPart {
        units: u32,
    },
    /// The host copies a [`Request::PeerHold`] holds, and every part of
    /// the host regions they lie in that this server knows: its own host's,
    /// and those of the ranks of other hosts its sessions' heads stated. A
    /// region's copies are read from the same parts, one of each index; a
    /// held copy that does not lie in all of them is left out.
    PeerHeld {
        copies: Vec<HostCopy>,
        parts: Vec<RegionPart>,
    },
    /// The holds a [`Request::PeerRelease`] ended stood until now.
    Valid,
    /// The connection held nothing to release.
    Gone,
    Error {
        kind: ErrorKind,
        message: String,
    },
}

/// The tensors a [`Request::Get`] found, over `storages`, which they name by
/// index, and the keys it did not find.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tensors {
    pub tensors: Vec<TensorRecord>,
    pub storages: Vec<StorageRecord>,
    pub missing: Vec<Key>,
}

/// The quotas a [`Request::Quota`] or a [`Request::Usage`] names, what the
/// host admits of new host storage, and, for a [`Request::Usage`], what the
/// tasks of the asking deployment copied, by priority.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub quotas: Vec<Quota>,
    pub host: HostStorage,
    pub tasks: Vec<TaskTotal>,
}

/// What a server holds.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Listing {
    pub tensors: Vec<TensorListing>,
    /// Bytes of device storages per GPU (by UUID) and per scope that holds a
    /// tensor in them; a storage shared by scopes counts for each.
    pub device_bytes: Vec<ScopeBytes>,
    /// Bytes of host storages, per scope.
    pub host_bytes: Vec<ScopeBytes>,
    /// The deployments claimed, and whether their head is connected.
    pub deployments: Vec<DeploymentListing>,
    /// Whether the Mooncake connector is connected.
    pub connector: bool,
    /// The Transfer Server, while it is connected.
    pub transfer: Option<TransferListing>,
    /// Every deployment's quota of every GPU the server reserves.
    pub quotas: Vec<Quota>,
    /// The buffer pool of every deployment that opened a session.
    pub buffer_pools: Vec<BufferPoolListing>,
}

/// One deployment's buffer pool in a [`Listing`]: what it holds and what
/// waits (`specs/buffer-pool.md` §8).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufferPoolListing {
    pub deployment: String,
    /// Per region and class, the blocks observed in each state; free units
    /// are in no block.
    pub blocks: Vec<BlockCount>,
    /// Every worker group of an open session, with what waits on it.
    pub groups: Vec<GroupListing>,
    /// Every request whose copies are in flight.
    pub copies: Vec<CopyListing>,
    /// The checks of the pool's tables after an input that found them
    /// inconsistent, while the server checks them
    /// (the server's `buffer_pool_check` setting).
    pub check_failures: u64,
}

/// The blocks of one region and class observed in one state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockCount {
    pub region: String,
    /// The class, as the sessions number it; `None` for slot rows.
    pub class: Option<u8>,
    pub state: BlockState,
    pub blocks: u32,
}

/// What a block is observed as, from who holds it: an entry of the cache
/// (shared), an owner, a waiting commit or a copy writing it (private), or
/// only deferred releases and ended copies (retired). The first that
/// applies is the state, in the order declared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockState {
    Shared,
    Private,
    Retired,
}

/// One worker group of a session: the `completed` word as the buffer pool
/// last read it, and the stamps its deferred work waits for, in order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupListing {
    pub epoch: u64,
    pub group: u8,
    pub completed: u64,
    pub waiting: Vec<(u64, Waiting)>,
}

/// What deferred work does when its stamp is reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Waiting {
    Release,
    Commit,
    Transfer,
    Finish,
    /// A crossing (`storage::runtime::Request::Replicate`).
    Replicate,
}

/// The copies one request started, while they are in flight: an
/// worker's copies, whose `Finish` they wait for, or a task's, which ends
/// by its deadline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CopyListing {
    /// The epoch of the session, or `None` once its session closed.
    pub epoch: Option<u64>,
    pub scheduler: u16,
    pub request: u64,
    pub copies: u32,
    /// The task's deadline in nanoseconds of the host's monotonic clock;
    /// `None` for a worker's copies.
    pub deadline: Option<u64>,
}

/// The Transfer Server in a [`Listing`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferListing {
    /// Its process id.
    pub pid: i32,
    /// The links it measured when it started; empty until it reported them.
    pub links: Vec<LinkBandwidth>,
    /// The device storage its CUDA context takes on each GPU it uses.
    pub contexts: Vec<ContextBytes>,
    /// The tasks it runs.
    pub tasks: Vec<TaskListing>,
}

/// One transfer task in a [`Listing`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskListing {
    pub task: u64,
    pub deployment: String,
    pub update: u64,
    pub priority: Priority,
    /// The bytes of all its spans.
    pub bytes: u64,
    /// The bytes copied so far.
    pub copied: u64,
    /// Whether it is being cancelled.
    pub cancelled: bool,
}

/// What the transfer tasks of one deployment at one priority copied on a
/// host since its Storage Server started (a restart in place passes the totals
/// over): the tasks the deployment's heads and workers submitted and the
/// weight moves the server made for it, ended or running.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskTotal {
    pub priority: Priority,
    /// The bytes of every span of the tasks submitted.
    pub bytes: u64,
    /// The bytes of them copied so far: those of every task that ended,
    /// cancelled or failed ones included, and the progress the running
    /// ones reported.
    pub copied: u64,
}

/// One retained tensor in a [`Listing`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorListing {
    pub key: Key,
    pub name: String,
    pub scopes: Vec<Scope>,
    /// The `(scope, epoch)` pairs that retain the tensor.
    pub retainers: Vec<Retainer>,
    pub location: Location,
    /// Bytes of the tensor's storage.
    pub storage_bytes: u64,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub access: Access,
    /// The process id of the tensor's writer, for an exclusive tensor that
    /// has one.
    pub writer: Option<i32>,
}

/// Storage bytes one scope of one epoch holds on one location.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopeBytes {
    /// The GPU's UUID for device bytes; `None` for host bytes.
    pub gpu: Option<[u8; 16]>,
    pub scope: Scope,
    /// The epoch whose scope holds the bytes.
    pub epoch: u64,
    pub bytes: u64,
}

/// One use that keeps a tensor: a scope of one epoch.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Retainer {
    pub scope: Scope,
    pub epoch: u64,
}

/// One claimed deployment in a [`Listing`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeploymentListing {
    pub deployment: String,
    pub head_connected: bool,
}

/// A message the server sends outside the responses to requests: the
/// completion of a transfer, the connector's work and the Transfer Server's.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Event {
    /// To a transfer's submitter: task `task` copied `bytes` of its spans
    /// so far.
    TaskProgress { task: u64, bytes: u64 },
    /// To a transfer's submitter: task `task` ended as `end`, having copied
    /// `bytes`. From then on the Transfer Server reads none of its sources
    /// and writes none of its destinations.
    TaskEnded {
        task: u64,
        bytes: u64,
        status: TaskStatus,
    },
    /// To the Transfer Server: an allocation of `bytes` in `storage`, whose
    /// descriptor travels with the event (a CUDA VMM allocation's POSIX
    /// handle, or a memfd), before the first task that copies it. It maps
    /// the allocation for writing when `writable`: a storage that holds an
    /// exclusive tensor, or a rank's part of a host region.
    Allocation {
        allocation: Allocation,
        end: LinkEnd,
        bytes: u64,
        writable: bool,
    },
    /// To the Transfer Server: the server released the allocation; it
    /// unmaps it once no task copies it.
    AllocationReleased { allocation: Allocation },
    /// To the Transfer Server: copy `moves` as task `task` at `priority`,
    /// and report with [`Request::Copied`].
    Task {
        task: u64,
        priority: Priority,
        moves: Vec<Move>,
    },
    /// Retain local GPU contexts between tasks of an operation. A session
    /// with no GPUs ends its holds; task holds remain independent.
    Session { session: u64, gpus: Vec<[u8; 16]> },
    /// To the Transfer Server: cancel task `task`.
    Cancel { task: u64 },
    /// To the Transfer Server: change the budgets of `links` and of every
    /// link a task of `tasks` copies over; a restoration with neither
    /// restores every link a halving reduced.
    Throttle {
        links: Vec<Link>,
        throttle: Throttle,
        tasks: Vec<u64>,
    },
    /// To the Transfer Server: register `allocations`, each given before,
    /// with its NIXL agent for another host's READs, and report with
    /// [`Request::Registered`].
    Register {
        request: u64,
        allocations: Vec<Allocation>,
    },
    /// To the Transfer Server: create device storage of at least `bytes` on
    /// GPU `gpu`, retainable through a POSIX descriptor and addressed by
    /// every GPU that can, send its descriptor with
    /// [`Request::Allocated`], and keep no handle of it: the server holds
    /// the storage from then on, which an epoch's restored weights take.
    Allocate {
        request: u64,
        gpu: [u8; 16],
        bytes: u64,
    },
    /// To the Transfer Server: the exposure its registrations of `request`
    /// served ended; they go once no other exposure holds them.
    Deregister { request: u64 },
    /// To the connector: a pool of pages, a rank's part of a buffer pool's
    /// host region, whose memfd travels with the event.
    Pool {
        pool: u64,
        shape: PageShape,
        slots: u64,
    },
    /// To the connector: the pool is released; the connector has no task on
    /// it, and unmaps it.
    PoolReleased { pool: u64 },
    /// To the connector: write the pages in these slots to the KV storage.
    Upload {
        task: u64,
        pool: u64,
        pages: Vec<(Key, u64)>,
    },
    /// To the connector: read these pages from the KV storage into their
    /// slots.
    Download {
        task: u64,
        pool: u64,
        pages: Vec<(Key, u64)>,
    },
    /// To the head that asked for task `task` ([`Request::Pages`]): one
    /// result per page, in the order the request named them; a page whose
    /// result is false did not move. From then on the connector touches
    /// none of the task's units.
    PagesMoved { task: u64, results: Vec<bool> },
}

/// A message from a client.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClientMessage {
    Hello(Hello),
    Request { id: u64, request: Request },
}

/// A message from the server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServerMessage {
    Welcome(Welcome),
    Response { id: u64, response: Response },
    Event(Event),
}

/// The payload of a frame of the Storage Server's protocol: `message` in MessagePack with named
/// fields. Each such frame (`service::frame`) carries the descriptors of the storage its
/// message names.
///
/// # Errors
///
/// Fails with `InvalidInput` when the message does not encode.
pub fn encode<T: Serialize>(message: &T) -> std::io::Result<Vec<u8>> {
    rmp_serde::to_vec_named(message)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))
}

/// The message a frame's payload of the Storage Server's protocol holds ([`encode`]).
///
/// # Errors
///
/// Fails with `InvalidData` when the payload is no encoding of a `T`.
pub fn decode<T: serde::de::DeserializeOwned>(payload: &[u8]) -> std::io::Result<T> {
    rmp_serde::from_slice(payload)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))
}

impl service::Message for ClientMessage {
    fn encode(&self) -> std::io::Result<Vec<u8>> {
        encode(self)
    }

    fn decode(payload: &[u8]) -> std::io::Result<Self> {
        decode(payload)
    }
}

impl service::Message for ServerMessage {
    fn encode(&self) -> std::io::Result<Vec<u8>> {
        encode(self)
    }

    fn decode(payload: &[u8]) -> std::io::Result<Self> {
        decode(payload)
    }
}
