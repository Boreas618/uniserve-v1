//! The messages a Storage Server and its clients exchange.
//!
//! A connection opens with the client's [`Hello`], which states the protocol
//! version and the client's [`Role`], and the server's [`Welcome`]. Every
//! later client message is a [`Request`] with an identity of the client's
//! choosing; the server answers requests in order on their connection, each
//! with one [`Reply`] that repeats the identity, and sends asynchronous
//! completions as [`Event`]s on the same connection.
//!
//! The protocol is versioned independently of the build ([`VERSION`]),
//! because a Storage Server outlives builds of the head and the workers. The
//! encoding is MessagePack with named fields, so a field added in a later
//! version of a message is ignored by an older peer, and a field a client of
//! an earlier version leaves out takes its default. A server serves clients
//! of its version and of every earlier one from [`MIN_VERSION`] on, and
//! refuses a client of a later version at `Hello`: a component that needs a
//! later protocol than the running server speaks restarts the server with
//! its own build first, which goes on with every connection.
//!
//! Storage belongs to worlds: a world is the set of processes that executes
//! one configuration of a deployment (an engine and its workers, relaunched
//! or not), and a deployment runs at most two at once while it updates. Each
//! tensor is retained by the `(scope, world)` pairs that use it: a worker's
//! requests retain and release under its own world alone, so a world that
//! starts beside another neither takes nor releases the other's storage,
//! and a world's storage outlives its processes until the head retires the
//! world.
//!
//! The server stores bytes with DLPack metadata and never interprets them:
//! clients compute every [`Key`] and define what the bytes mean.
//!
//! The server keeps one storage ledger per GPU for each deployment that uses
//! it ([`GpuLedger`]), so that the two worlds of a deployment and the host's
//! Transfer Server stay within one quota on a GPU they share, rather than
//! each sizing itself against what its own process holds. The quota is
//! `⌊total × fraction⌋` of the whole GPU for the fraction the deployment's
//! workers configure (`mem_fraction_static`). The ledger books what the
//! server derives from the storages it retains (the weights, each KV cache
//! segment, the other exclusive storages), each storage once whatever
//! number of worlds map it, and what the processes book themselves
//! ([`Request::Book`]): a worker's graph pool, process storage and staging,
//! and the Transfer Server's staging and its CUDA context while it holds
//! one on the GPU. The server refuses a booking or a new device storage
//! that would take a ledger past its quota.
//!
//! Besides tensors, the server keeps host page pools (L2): one memfd of
//! equal slots per deployment, NUMA node and [`PageShape`], shared by the
//! deployment's workers whose pages have that shape, of the size the
//! worker asks for. A deployment's pools are released with its tensors.
//! The server refuses a pool the host's available storage cannot hold.
//! The server indexes committed pages by key
//! and tracks each slot's state; workers copy pages in and out of the slots
//! on their own streams, at the head's direction. Behind the pools, the
//! server's Mooncake connector ([`Role::Connector`]) keeps committed pages
//! in the KV storage (L3) and reads them back for prefetches.
//!
//! The server copies no byte itself. Its Transfer Server ([`Role::Transfer`])
//! copies between the storage the server holds for heads and workers: a
//! client submits a task of [`Span`]s with a [`Priority`]
//! ([`Request::Transfer`]), the server resolves each [`Region`] to one of its
//! allocations and forwards the task, and relays the Transfer Server's
//! progress and the task's end to the client as events
//! ([`Event::TransferProgress`], [`Event::Transferred`]). The submitter
//! keeps a task's sources unchanged and its destinations unread until the
//! task ends; the Transfer Server reports the end only after it saw every
//! copy complete on the host.
//!
//! A span's source may lie on another host ([`Region::Remote`]): the
//! destination host's Transfer Server READs it through NIXL. It asks the
//! source host's server, over that server's TCP endpoint as a
//! [`Role::Peer`], to expose the regions ([`Request::Expose`]); that server
//! has its own Transfer Server register their allocations with its NIXL
//! agent ([`Event::Register`], [`Request::Registered`]) and answers with the
//! agent's metadata and each region's address ([`Reply::Exposed`]). The
//! exposure lasts until the reading Transfer Server ends it, once its task
//! ended ([`Request::Unexpose`]), or its connection closes: the source's
//! Transfer Server then lets the registrations go ([`Event::Deregister`]),
//! and with them the GPU contexts they held.

use serde::{Deserialize, Serialize};

/// Wire contract negotiated when a Storage Server client connects. Committed
/// pages stay reusable by content key until ordinary eviction; only a
/// world's uncommitted publications are discarded when it retires.
pub const VERSION: u32 = 17;

/// The most bytes a [`Request::Read`] answers with: a weight manifest of the
/// largest models the library serves fits many times over, and a reply of
/// this size holds the server's loop for a few milliseconds on a host's
/// network.
pub const READ_BYTES: u64 = 64 << 20;

/// The earliest client version the server serves: a client of version 4
/// knows no worlds, and its requests act on world 0; a client of version 4
/// or 5 states no world as a head, which reads and publishes world 0's
/// pages; a client before version 7 makes no transfer request and reads no
/// transfer event; a client before version 8 names no storage of another
/// host; a client before version 9 books nothing in the ledgers and marks
/// no storage as a KV cache segment; a client before version 10 names no
/// location of a key's copy, and its requests take every copy.
pub const MIN_VERSION: u32 = 4;

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
/// head, of whatever world, and the head hands it to its workers, which
/// state it in their [`Role`]. A worker that states another token belongs to
/// a previous head of the deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Claim(pub [u8; 16]);

/// What a client is to the server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// A head of the deployment: the holder of its claim on this host (the
    /// host's daemon), or an engine of the deployment, which sends the page
    /// requests.
    ///
    /// An engine states its `world`: the pages it commits are that world's,
    /// and it reads its own world's pages and those every world shares.
    /// Uncommitted worlds keep pages and tensors private until publication;
    /// retiring such a world discards its pages.
    Head {
        deployment: String,
        #[serde(default)]
        world: u64,
    },
    /// A worker of `scope` in `world`, started by the launch `generation` of
    /// its group, for the head that holds `claim`.
    Worker {
        scope: Scope,
        generation: u64,
        claim: Claim,
        /// The world the worker belongs to: the head's identity of the
        /// processes that execute one configuration of the deployment.
        #[serde(default)]
        world: u64,
    },
    /// The Mooncake connector the server runs.
    Connector,
    /// The Transfer Server the server runs, which copies bytes between the
    /// storage the server holds. Every ledger of a GPU it uses books its CUDA
    /// context there, as it measured it ([`Request::Measured`]), and any
    /// staging it books ([`Request::Book`]).
    Transfer,
    /// Another host's Transfer Server, which READs storage of this host
    /// through NIXL for the tasks of its own host: it asks the server to
    /// expose regions ([`Request::Expose`]). It connects to the server's
    /// TCP endpoint.
    Peer,
    /// An operator's client, which lists what the server holds and retires
    /// deployments that are no longer served.
    Operator,
}

/// The first message on a connection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hello {
    pub version: u32,
    pub role: Role,
}

/// The server's answer to [`Hello`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Welcome {
    Accepted {
        version: u32,
    },
    /// The server does not serve this client, for the reason given.
    Refused {
        message: String,
    },
}

/// Where a storage's physical storage is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Placement {
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
    pub placement: Placement,
    /// The allocation's size, a multiple of the device's allocation
    /// granularity for device storage.
    pub bytes: u64,
    /// The KV cache segment the storage is part of, which its GPU's ledger
    /// books as that segment ([`Item::CacheSegment`]); `None` for any other
    /// storage, and for every storage of a client of protocol 5.
    #[serde(default)]
    pub segment: Option<Segment>,
    /// The fabric handle the device allocation's creator exported, when it
    /// holds the allocation and its GPU belongs to an NVLink fabric
    /// (multi-node NVLink): another host of the fabric imports the handle
    /// to copy the storage over NVLink while the creator holds it. `None`
    /// for host storage, for an allocation without one, and for every
    /// storage of a client that states none.
    #[serde(default)]
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

/// One segment of a KV cache's value planes: the world that created its
/// storage and its ordinal there. A segment is one storage per layer and
/// plane, which a ledger books together.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Segment {
    pub world: u64,
    pub ordinal: u32,
}

/// What one entry of a GPU's storage ledger is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Item {
    /// Read-only device storages the server retains: a world's weights.
    Weights,
    /// The device storages of one KV cache segment.
    CacheSegment,
    /// Exclusive device storages the server retains beside the KV cache's
    /// value planes: the cache's flags and scales, the buffer pool, the
    /// latent banks.
    Exclusive,
    /// A worker's CUDA graph storage pool: the storage its captured graphs
    /// address. Booked by the worker.
    GraphPool,
    /// What a worker process holds on the device beside the storages the
    /// server retains and its graph pool: its CUDA context, communicator
    /// buffers, PyTorch's allocations, Rust's arenas and tables, the
    /// product storage it has yet to make resident, and the device storage
    /// a profiling session's collectors hold while it runs. Booked by the
    /// worker.
    Process,
    /// Device storage a process holds to stage transfers. Booked by the
    /// process.
    Staging,
    /// The device storage of the Transfer Server's CUDA context, at the
    /// bytes it measured when it started ([`Request::Measured`]). Booked by
    /// the Transfer Server before it retains the context on the GPU, which
    /// it holds only while it has work there, and ended when it released
    /// it.
    TransferContext,
}

impl Item {
    /// Whether a process books the item itself ([`Request::Book`]), rather
    /// than the server deriving it from the storages it retains.
    pub fn booked(self) -> bool {
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

/// One deployment's storage ledger of one GPU: everything the deployment's
/// worlds hold on it, and the Transfer Server's items there, beside which
/// every deployment that uses the GPU holds its storage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuLedger {
    /// The GPU's UUID.
    pub gpu: [u8; 16],
    /// The deployment; empty for a GPU only the Transfer Server books.
    pub deployment: String,
    /// The GPU's total storage, as the processes that use it state it; 0
    /// until one does.
    pub total: u64,
    /// `⌊total × fraction⌋` for the smallest fraction a world of the
    /// deployment states for the GPU; `None` until one states one.
    pub quota: Option<u64>,
    pub entries: Vec<LedgerEntry>,
    /// The bandwidth of copies from host storage to the GPU, in bytes per
    /// second, as the host's Transfer Server measured it when it started;
    /// `None` while none did. What returns from host storage to the GPU
    /// takes its bytes over this time.
    #[serde(default)]
    pub host_link_bytes_per_second: Option<u64>,
    /// The device storage the Transfer Server's CUDA context takes on the
    /// GPU while it holds one, as it measured it when it started; 0 when no
    /// Transfer Server measured the GPU. The ledger books it as an entry
    /// ([`Item::TransferContext`]) only while the Transfer Server holds the
    /// context, from its first work on the GPU to its last, so a plan that
    /// leaves room for transfers counts it whether or not it is booked.
    #[serde(default)]
    pub transfer_context: u64,
}

/// One entry of a [`GpuLedger`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerEntry {
    pub item: Item,
    /// The worker scope that holds it; `None` for the Transfer Server's
    /// items and for storages several scopes hold.
    pub scope: Option<Scope>,
    /// The worlds that hold it: one for a world's own, two for the storages
    /// a successor world maps of its predecessor, which are booked once;
    /// none for the Transfer Server's items.
    pub worlds: Vec<u64>,
    /// The segment of an [`Item::CacheSegment`].
    pub segment: Option<Segment>,
    pub bytes: u64,
}

impl GpuLedger {
    /// The bytes every entry books.
    pub fn used(&self) -> u64 {
        self.bytes(|_| true)
    }

    /// The bytes of the entries `select` picks.
    pub fn bytes(&self, select: impl Fn(&LedgerEntry) -> bool) -> u64 {
        self.entries
            .iter()
            .filter(|entry| select(entry))
            .map(|entry| entry.bytes)
            .sum()
    }

    /// Every entry, one per line, for a refusal or a log.
    pub fn describe(&self) -> String {
        let quota = self.quota.map_or_else(
            || "no quota".to_owned(),
            |quota| format!("a quota of {quota}"),
        );
        let holder = if self.deployment.is_empty() {
            "the Transfer Server"
        } else {
            self.deployment.as_str()
        };
        let mut lines = vec![format!(
            "GPU {} of {holder}: {} bytes booked of {quota}",
            gpu_name(&self.gpu),
            self.used(),
        )];
        for entry in &self.entries {
            lines.push(format!("  {}", entry.describe()));
        }
        lines.join("\n")
    }
}

impl LedgerEntry {
    /// The entry on one line: its item, holder and bytes.
    pub fn describe(&self) -> String {
        let worlds: Vec<String> = self.worlds.iter().map(u64::to_string).collect();
        let holder = match (&self.scope, worlds.is_empty()) {
            (None, true) => "the Transfer Server".to_owned(),
            (Some(scope), _) => format!("{scope} of world {}", worlds.join(" and ")),
            (None, false) => format!("world {}", worlds.join(" and ")),
        };
        let segment = self
            .segment
            .map(|segment| format!(" {}/{}", segment.world, segment.ordinal))
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

/// What a host's Storage Server admits of new host storage (page pools,
/// staging, weights moved to the host): the host's available storage less
/// the server's reserve and what its retained storage has yet to allocate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostStorage {
    /// The host's `MemAvailable`.
    pub available: u64,
    /// What the server leaves to the host's other processes.
    pub reserve: u64,
    /// Unallocated bytes of page pools and retained host storages.
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

/// What the pages of one page pool are: every worker of a deployment on a
/// host's NUMA node whose pages have this shape shares the pool, whatever
/// its worker group, scheduler or engine, so an identical page is stored
/// once.
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

/// Where a `lookup` found one page key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PageLevel {
    /// Committed in the pool, the host KV cache (L2).
    Host,
    /// Not in the pool, but in the KV storage (L3), which a prefetch reads
    /// it from.
    Storage,
    /// Neither.
    Absent,
}

/// The pages of one sliding window of attention, which a `reserve` names:
/// a window missing one page is useless, so the pool evicts the window
/// whole. Its pages are the reservation's keys, those the pool reserves and
/// those it holds already.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageWindow {
    /// The page the window follows in another pool of the server: the last
    /// full-attention page before the window's boundary. A reuse at the
    /// boundary needs it, so the window is evicted with it.
    pub anchor: Option<PageAnchor>,
}

/// A page of a pool: the anchor of a [`PageWindow`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageAnchor {
    pub pool: u64,
    pub key: Key,
}

/// What a `reserve` made of one page key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PageSlot {
    /// The slot reserved for the page, which its writer writes and the
    /// head then commits or aborts.
    Reserved { slot: u64 },
    /// The page is committed already, or another writer has it reserved.
    Present,
    /// Every slot is reserved or pinned; the page is not stored.
    Full,
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
    Tensor {
        key: Key,
        #[serde(default)]
        at: Option<Placement>,
    },
    /// A live page pool of the caller's deployment; offsets count from slot
    /// 0, at `slot × page_bytes`.
    Pool { pool: u64 },
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
    /// through NIXL: `region`, a retained tensor or a page pool of the
    /// caller's deployment, as a client of that host names it, at the
    /// Storage Server listening at `endpoint` (that server's
    /// [`Request::Endpoint`]). Offsets count as they do in `region`. Only a
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
    /// What a world's promotion waits for: the data the in-flight requests
    /// wrote after the next world's store was warmed.
    Promotion,
    /// Warming the next world's store while the executing world runs: the
    /// pages the in-flight requests filled, copied ahead of the promotion.
    Warming,
    /// Weight moves and every other copy no promotion waits for.
    Background,
}

impl Priority {
    /// Every priority, highest first.
    pub const ORDER: [Priority; 3] = [Priority::Promotion, Priority::Warming, Priority::Background];

    /// The priority's position in [`Self::ORDER`]: 0 is the highest.
    pub fn rank(self) -> usize {
        match self {
            Priority::Promotion => 0,
            Priority::Warming => 1,
            Priority::Background => 2,
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

/// A change of links' byte budgets, which the head of the executing world
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
pub enum TransferEnd {
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
    /// The page pool `pool`.
    Pool { pool: u64 },
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
pub struct Registered {
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
    #[serde(default)]
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
    /// already, and state the scopes `world` places on the host: the world's
    /// retention of every other scope ends. Every world of `retire` ends on
    /// the host: its retention of every tensor ends, and a tensor no world
    /// retains is released. Refused while a previous head of the deployment
    /// is connected. Every process of the deployment that belongs to no
    /// current worker is shot down. Every other deployment whose head is not
    /// connected is retired: a host serves the deployments whose heads run
    /// on it.
    Claim {
        scopes: Vec<Scope>,
        #[serde(default)]
        world: u64,
        #[serde(default)]
        retire: Vec<u64>,
    },
    /// Publish all listed committed worlds together. Claimed worlds remain
    /// private until this request, including during preparation and trial.
    /// Every listed world must be claimed by the head's deployment.
    PublishWorlds { worlds: Vec<u64> },
    /// Release every tensor the caller's scope retains in the caller's world
    /// that `keys` does not list; the copies of listed keys at the locations
    /// `at` lists are retained (every copy when `at` is empty).
    Retain {
        keys: Vec<Key>,
        #[serde(default)]
        at: Vec<Placement>,
    },
    /// Worlds in the head's deployment retain their weights and the exact
    /// `cache_keys` holding completed rollback copies. Their retention of
    /// every other exclusive tensor ends. A tensor no other world retains
    /// is released, while segments mapped by the successor remain retained
    /// by that successor. Call after the predecessor's processes exit.
    /// Answered with `Retained`.
    KeepWeights {
        worlds: Vec<u64>,
        cache_keys: Vec<Key>,
    },
    /// From a head: world `world` of its deployment keeps its weights in
    /// host storage (`update_config.previous_world = "host"`, its processes
    /// exited). Every device copy of read-only tensors that this world
    /// alone retains moves to host storage: the server creates host storage of
    /// each storage's size, admitted under the host rule as one (refused,
    /// moving nothing, when the host cannot hold it all), has its Transfer
    /// Server copy each storage there, and then the tensors' copies lie in
    /// host storage and the device storages are released: each GPU's ledger
    /// leaves their storage to the other world. Answered with `Moved` once
    /// every copy ended; refused while no Transfer Server runs.
    Offload { world: u64 },
    /// From a head: world `world`'s weights that `Offload` moved to host
    /// storage return to the GPUs they came from, before its processes start
    /// again: its Transfer Server creates device storage of each storage's
    /// size on its GPU, admitted against the GPU's ledger (refused, moving
    /// nothing, past the quota), and copies each storage there; then the
    /// tensors' copies lie on those GPUs again and the host copies are
    /// released. Answered with `Moved` once every copy ended.
    Restore { world: u64 },
    /// The copies of `keys` that exist at the locations `at` lists (at
    /// every location when `at` is empty), with their storages'
    /// descriptors, and the keys with no such copy.
    Get {
        keys: Vec<Key>,
        mode: Mode,
        #[serde(default)]
        at: Vec<Placement>,
    },
    /// Allocate an exclusive byte tensor in host storage after host
    /// admission. Returns its retained storage through `Got`, with write
    /// ownership belonging to this worker and lifetime to its world.
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
        #[serde(default)]
        at: Vec<Placement>,
    },
    /// Every tensor, storage and claim the server holds.
    List,
    /// From a process of another host ([`Role::Peer`]): the bytes of the
    /// copy of `key` in host storage that a scope of `deployment` retains
    /// read-only, a one-dimensional tensor of `uint8` elements (such as a
    /// weight manifest) of at most [`READ_BYTES`] bytes. Answered with
    /// [`Reply::Read`]; refused when the deployment retains no such tensor,
    /// or a longer one.
    Read { deployment: String, key: Key },
    /// Release every tensor and page pool of `deployment` and forget its
    /// claim: the deployment is no longer served. Refused while its head is
    /// connected.
    Retire { deployment: String },
    /// Hand the server to a successor server of `executable` (this
    /// server's own when `None`), which goes on with every connection, the
    /// listening sockets, the tensors' storage and the page pools once no
    /// request, shoot-down, connector task or transfer is in flight (a hot
    /// restart);
    /// the server then exits. Answered with the successor's process id
    /// once it started.
    Restart { executable: Option<String> },
    /// A worker's page pool for pages of `shape` on `numa_node`: the pool
    /// of the worker's deployment of that shape, whose memfd travels with
    /// the reply. When absent it is created with `bytes` (the server's
    /// default when `None`), refused when the host's available storage,
    /// less the server's reserve and what its pools have yet to allocate,
    /// cannot hold it.
    PagePool {
        shape: PageShape,
        numa_node: Option<u32>,
        #[serde(default)]
        bytes: Option<u64>,
    },
    /// Where each page of `keys` is: committed in the pool (L2), in the KV
    /// storage (L3), or absent. The pages the pool holds become its most
    /// recently used.
    Lookup { pool: u64, keys: Vec<Key> },
    /// A slot for each key the pool holds no page of, to be written by the
    /// worker of `writer`. With `window`, the keys are the pages of one
    /// sliding window, which the pool evicts whole.
    Reserve {
        pool: u64,
        writer: Scope,
        keys: Vec<Key>,
        window: Option<PageWindow>,
    },
    /// The reserved pages are written: they become readable.
    Commit { pool: u64, keys: Vec<Key> },
    /// The reserved pages were not written: their slots become free.
    Abort { pool: u64, keys: Vec<Key> },
    /// The slots of the committed pages of `keys`, which stay readable and
    /// are not evicted until the head unpins them, for the worker of
    /// `reader` to read; `None` for a key the pool holds no page of.
    Pin {
        pool: u64,
        reader: Scope,
        keys: Vec<Key>,
    },
    /// The worker of `reader` read the pinned pages.
    Unpin {
        pool: u64,
        reader: Scope,
        keys: Vec<Key>,
    },
    /// Where the server accepts connections from other hosts.
    Endpoint,
    /// Read the pages of `keys` the pool does not hold from the KV storage
    /// into slots reserved for them. Answered at once; the event
    /// [`Event::Prefetched`] with the same `tag` follows when the reads
    /// end.
    Prefetch { pool: u64, keys: Vec<Key>, tag: u64 },
    /// The token count each rank of a world holds in the host KV cache of
    /// this host, beside the world's other ranks here: the least of
    /// `tokens`, the count the world asks for, and the host's budget over
    /// the bytes one token takes on all of those ranks (`token_bytes`, one
    /// entry per rank). The budget is `host_bytes`, the deployment's host KV
    /// cache on each host, when stated; otherwise four fifths of the host's
    /// available storage (bounded by the server's storage cgroup limit, and
    /// counting the deployment's page pools no world maps, which the server
    /// releases for a new pool) less what the page pools have yet to
    /// allocate and the server's reserve. Answered with [`Reply::HostCache`].
    HostCache {
        tokens: u64,
        token_bytes: Vec<u64>,
        host_bytes: Option<u64>,
    },
    /// The connector ran task `task` of an [`Event::Exists`],
    /// [`Event::Upload`] or [`Event::Download`]: one result per page of the
    /// task.
    Finished { task: u64, results: Vec<bool> },
    /// Book the caller's own items on GPU `gpu` (of total storage `total`):
    /// each item of `items` takes its bytes, 0 ending it; an item left out
    /// keeps its booking. A worker books [`Item::GraphPool`],
    /// [`Item::Process`] and [`Item::Staging`] in its deployment's ledger
    /// under its scope and world, and states `quota`, `⌊total × fraction⌋`
    /// for its configured fraction. The processes of one scope and world (a
    /// rank, and the relaunch or standby that replaces it) serve in turn, so
    /// the larger of their bookings of an item counts. The Transfer Server
    /// books [`Item::Staging`] and [`Item::TransferContext`] in every ledger
    /// of the GPU, stating no quota: its context before it retains it, which
    /// a ledger without room refuses. A booking ends with the connection of
    /// the process that made it, as that process's storage ends with it. Refused, booking nothing, when it
    /// would take a ledger past its quota; answered with the ledgers of the
    /// GPU the caller books in.
    Book {
        gpu: [u8; 16],
        total: u64,
        quota: Option<u64>,
        items: Vec<(Item, u64)>,
    },
    /// The ledgers of `gpus` (of every GPU when empty) and what the host
    /// admits of new host storage: of the caller's deployment for a worker
    /// and a head, of every deployment for the Transfer Server and an
    /// operator.
    Ledger { gpus: Vec<[u8; 16]> },
    /// From a head or a worker: copy `spans` as one task of `update`, which
    /// takes its turn at `priority` on every link it uses. Answered with the
    /// task's number at once; [`Event::TransferProgress`] follows as it
    /// copies and [`Event::Transferred`] when it ends. Refused, copying
    /// nothing, when no Transfer Server runs, when a region is not storage
    /// of the caller's deployment, or when a span reaches past its region
    /// or writes a read-only tensor; a region of another host is checked
    /// by that host once the task runs ([`Region::Remote`]). A task ends
    /// with its submitter's connection, cancelled.
    Transfer {
        spans: Vec<Span>,
        priority: Priority,
        /// The update (world change) the task serves, a number its
        /// deployment's head chooses; the tasks of an update that aborts
        /// are cancelled together.
        update: u64,
    },
    /// Hold the transfer contexts of these local GPUs across successive
    /// tasks. Contexts remain lazy and quota-accounted. The session ends
    /// explicitly or when its owning connection closes.
    OpenTransfer { gpus: Vec<[u8; 16]> },
    /// End a session of this connection. Submitted tasks keep their own
    /// holds until physical completion; this releases only the session.
    CloseTransfer { session: u64 },
    /// From a head or a worker: cancel `tasks` and, with `update`, every
    /// task of that update, of the caller's deployment. Answered at once;
    /// each task's [`Event::Transferred`] follows once the Transfer Server
    /// copies none of its bytes any more.
    Cancel {
        tasks: Vec<u64>,
        update: Option<u64>,
    },
    /// From a head: halve or restore the byte budget of each of `links`.
    /// With no links, a halving takes every link a task of the head's
    /// deployment copies over now (a deployment changes one world at a
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
        end: Option<TransferEnd>,
    },
    /// From another host's Transfer Server ([`Role::Peer`]): have this
    /// host's Transfer Server register `regions`, retained tensors or page
    /// pools of `deployment` on this host, with its NIXL agent, for that
    /// Transfer Server's READs. Answered with [`Reply::Exposed`] once they
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
    /// caller's previous connection) is answered alike.
    Unexpose { exposure: u64 },
    /// From the Transfer Server: the allocations of the server's
    /// [`Event::Register`] `request` are registered with its NIXL agent:
    /// each allocation's range, in the order the event listed them, and
    /// the agent's `metadata` taken after the registrations; or `failure`
    /// says why they are not.
    Registered {
        request: u64,
        metadata: Vec<u8>,
        ranges: Vec<Registered>,
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
    /// A booking or a new device storage would take a GPU's ledger past
    /// its quota.
    Quota,
}

/// The server's answer to one request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reply {
    WorldsPublished,
    Claimed {
        claim: Claim,
    },
    Retained {
        released: u64,
    },
    /// The tensors found, over `storages`, whose descriptors travel with the
    /// reply in order, and the keys not found.
    Got {
        tensors: Vec<TensorRecord>,
        storages: Vec<StorageRecord>,
        missing: Vec<Key>,
    },
    Put,
    Promoted,
    Released,
    Rekeyed,
    Listing(Listing),
    /// The bytes a [`Request::Read`] asked for.
    Read {
        #[serde(with = "serde_bytes")]
        bytes: Vec<u8>,
    },
    Retired {
        released: u64,
    },
    /// The successor of a [`Request::Restart`] started.
    Restarting {
        pid: u32,
    },
    /// The page pool, whose memfd travels with the reply: `slots` slots of
    /// `page_bytes` bytes, slot `i` at byte `i * page_bytes`.
    PagePool {
        pool: u64,
        slots: u64,
        page_bytes: u64,
    },
    /// One level per key of the lookup.
    Lookup {
        levels: Vec<PageLevel>,
    },
    Reserved {
        slots: Vec<PageSlot>,
    },
    Committed,
    Aborted,
    Pinned {
        slots: Vec<Option<u64>>,
    },
    Unpinned,
    /// The `host:port` other hosts connect to; `None` when the server
    /// accepts local connections only.
    Endpoint {
        tcp: Option<String>,
    },
    Prefetching,
    /// The token count of a [`Request::HostCache`].
    HostCache {
        tokens: u64,
    },
    Finished,
    /// The ledgers a `Book` or a `Ledger` names, and what the host admits
    /// of new host storage.
    Ledger {
        ledgers: Vec<GpuLedger>,
        host: HostStorage,
    },
    /// The task a [`Request::Transfer`] submitted.
    Transferring {
        task: u64,
    },
    TransferOpened {
        session: u64,
    },
    TransferClosed,
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
        regions: Vec<Registered>,
    },
    /// A [`Request::Unexpose`] was taken.
    Unexposed,
    Error {
        kind: ErrorKind,
        message: String,
    },
}

/// What a server holds.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Listing {
    pub tensors: Vec<ListedTensor>,
    /// Bytes of device storages per GPU (by UUID) and per scope that holds a
    /// tensor in them; a storage shared by scopes counts for each.
    pub device_bytes: Vec<ScopeBytes>,
    /// Bytes of host storages, per scope.
    pub host_bytes: Vec<ScopeBytes>,
    /// The deployments claimed, and whether their head is connected.
    pub deployments: Vec<ListedDeployment>,
    /// The page pools.
    pub page_pools: Vec<ListedPool>,
    /// Whether the Mooncake connector is connected.
    #[serde(default)]
    pub connector: bool,
    /// The Transfer Server, while it is connected.
    #[serde(default)]
    pub transfer: Option<ListedTransfer>,
    /// Every deployment's ledger of every GPU the server books.
    #[serde(default)]
    pub ledgers: Vec<GpuLedger>,
}

/// The Transfer Server in a [`Listing`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ListedTransfer {
    /// Its process id.
    pub pid: i32,
    /// The links it measured when it started; empty until it reported them.
    pub links: Vec<LinkBandwidth>,
    /// The device storage its CUDA context takes on each GPU it uses.
    pub contexts: Vec<ContextBytes>,
    /// The tasks it runs.
    pub tasks: Vec<ListedTask>,
}

/// One transfer task in a [`Listing`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ListedTask {
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

/// One page pool in a [`Listing`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ListedPool {
    pub pool: u64,
    /// The deployment the pool belongs to.
    #[serde(default)]
    pub deployment: String,
    pub shape: PageShape,
    pub numa_node: Option<u32>,
    pub slots: u64,
    /// Slots holding committed pages, pinned ones included.
    pub ready: u64,
    pub reserved: u64,
    pub pinned: u64,
    /// Slots the connector is reading pages into.
    pub fetching: u64,
    /// Committed pages the connector is writing to the KV storage.
    pub uploading: u64,
}

/// One retained tensor in a [`Listing`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ListedTensor {
    pub key: Key,
    pub name: String,
    pub scopes: Vec<Scope>,
    /// The `(scope, world)` pairs that retain the tensor.
    #[serde(default)]
    pub retainers: Vec<Retainer>,
    pub placement: Placement,
    /// Bytes of the tensor's storage.
    pub storage_bytes: u64,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub access: Access,
    /// The process id of the tensor's writer, for an exclusive tensor that
    /// has one.
    pub writer: Option<i32>,
}

/// Storage bytes one scope of one world holds on one placement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopeBytes {
    /// The GPU's UUID for device bytes; `None` for host bytes.
    pub gpu: Option<[u8; 16]>,
    pub scope: Scope,
    /// The world whose scope holds the bytes.
    #[serde(default)]
    pub world: u64,
    pub bytes: u64,
}

/// One use that keeps a tensor: a scope of one world.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Retainer {
    pub scope: Scope,
    pub world: u64,
}

/// One claimed deployment in a [`Listing`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ListedDeployment {
    pub deployment: String,
    pub head_connected: bool,
}

/// A message the server sends outside the answers to requests: the
/// completion of a head's prefetch and of a transfer, the connector's work
/// and the Transfer Server's.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Event {
    /// To a transfer's submitter: task `task` copied `bytes` of its spans
    /// so far.
    TransferProgress { task: u64, bytes: u64 },
    /// To a transfer's submitter: task `task` ended as `end`, having copied
    /// `bytes`. From then on the Transfer Server reads none of its sources
    /// and writes none of its destinations.
    Transferred {
        task: u64,
        bytes: u64,
        end: TransferEnd,
    },
    /// To the Transfer Server: an allocation of `bytes` in `storage`, whose
    /// descriptor travels with the event (a CUDA VMM allocation's POSIX
    /// handle, or a memfd), before the first task that copies it. It maps
    /// the allocation for writing when `writable`: a page pool, or a
    /// storage that holds an exclusive tensor.
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
    Transfer {
        task: u64,
        priority: Priority,
        moves: Vec<Move>,
    },
    /// Retain local GPU contexts between tasks of an operation. A session
    /// with no GPUs ends its holds; task holds remain independent.
    TransferSession { session: u64, gpus: Vec<[u8; 16]> },
    /// To the Transfer Server: cancel task `task`.
    Cancel { task: u64 },
    /// To the Transfer Server: change the budgets of `links` and of every
    /// link a task of `tasks` copies over; a restoration with neither
    /// restores every link a halving reduced.
    Throttle {
        links: Vec<Link>,
        throttle: Throttle,
        #[serde(default)]
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
    /// the storage from then on, which a world's restored weights take.
    Allocate {
        request: u64,
        gpu: [u8; 16],
        bytes: u64,
    },
    /// To the Transfer Server: the exposure its registrations of `request`
    /// served ended; they go once no other exposure holds them.
    Deregister { request: u64 },
    /// To a head: the prefetch `tag` ended. `results[i]` is whether the pool
    /// holds the page of the prefetch's key `i`.
    Prefetched { tag: u64, results: Vec<bool> },
    /// To the connector: a page pool, whose memfd travels with the event.
    Pool {
        pool: u64,
        shape: PageShape,
        slots: u64,
    },
    /// To the connector: the page pool is released; the connector has no
    /// task on it, and unmaps it.
    PoolReleased { pool: u64 },
    /// To the connector: which pages of `keys` the KV storage holds.
    Exists {
        task: u64,
        pool: u64,
        keys: Vec<Key>,
    },
    /// To the connector: write the committed pages in these slots to the
    /// KV storage.
    Upload {
        task: u64,
        pool: u64,
        pages: Vec<(Key, u64)>,
    },
    /// To the connector: read these pages from the KV storage into
    /// their slots.
    Download {
        task: u64,
        pool: u64,
        pages: Vec<(Key, u64)>,
    },
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
    Reply { id: u64, reply: Reply },
    Event(Event),
}
