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
//! version of a message is ignored by an older peer; a change an older peer
//! cannot read takes a new version, which the server refuses at `Hello`.
//!
//! The server stores bytes with DLPack metadata and never interprets them:
//! clients compute every [`Key`] and define what the bytes mean.
//!
//! Besides tensors, the server keeps host page pools (L2): one memfd of
//! equal slots per NUMA node and [`PageShape`], shared by every worker
//! whose pages have that shape. The server indexes committed pages by key
//! and tracks each slot's state; workers copy pages in and out of the slots
//! on their own streams, at the head's direction. Behind the pools, the
//! server's Mooncake connector ([`Role::Connector`]) keeps committed pages
//! in the distributed tier (L3) and reads them back for prefetches.

use serde::{Deserialize, Serialize};

/// The protocol version this build speaks.
pub const VERSION: u32 = 2;

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

/// The token a deployment's claim carries: the server issues a new one with
/// every `claim`, and the head hands it to its workers, which state it in
/// their [`Role`]. A worker that states another token belongs to a previous
/// head of the deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Claim(pub [u8; 16]);

/// What a client is to the server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// The holder of a deployment's claim on this host: the host's daemon,
    /// for the head of the deployment.
    Head { deployment: String },
    /// A worker of `scope`, started by the launch `generation` of its group,
    /// for the head that holds `claim`.
    Worker {
        scope: Scope,
        generation: u64,
        claim: Claim,
    },
    /// The Mooncake connector the server runs.
    Connector,
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

/// Where a storage's physical memory is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    /// granularity for device memory.
    pub bytes: u64,
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

/// What the pages of one page pool are: every worker of a host's NUMA node
/// whose pages have this shape shares the pool, whatever its deployment,
/// worker group or engine core, so an identical page is stored once.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PageShape {
    /// The digest of what determines a page's encoding, computed by the
    /// worker: the model fingerprint, the page encoding version, the cache
    /// dtype, the block size and the cache group list.
    pub content: Key,
    /// The first kv head the pages hold.
    pub kv_head_offset: u32,
    /// The number of kv heads the pages hold.
    pub kv_head_count: u32,
    /// The bytes of one page.
    pub page_bytes: u64,
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

/// A client's request. Head requests come from a [`Role::Head`] client and
/// worker requests from a [`Role::Worker`] client; `List` from any.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Request {
    /// Make the caller the head of its deployment and release every scope of
    /// the deployment it does not list. Refused while a previous head of the
    /// deployment is connected. Every process of the deployment that
    /// belongs to no current worker is shot down.
    Claim { scopes: Vec<Scope> },
    /// Release every tensor of the caller's scope that `keys` does not list.
    Retain { keys: Vec<Key> },
    /// The tensors of `keys` that exist, with their storages' descriptors,
    /// and the keys that do not.
    Get { keys: Vec<Key>, mode: Mode },
    /// Publish tensors whose bytes are written, over the storages this
    /// message carries. Read-only tensors become readable; exclusive ones
    /// keep the caller as their writer. A read-only tensor already retained
    /// under the same key holds the same bytes: the put adds the caller's
    /// scope to it and the server keeps the retained one.
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
    /// Every tensor, storage and claim the server holds.
    List,
    /// Release every tensor of `deployment` and forget its claim: the
    /// deployment is no longer served. Refused while its head is connected.
    Retire { deployment: String },
    /// A worker's page pool for pages of `shape` on `numa_node`: the pool
    /// of that shape, created when absent, whose memfd travels with the
    /// reply.
    PagePool {
        shape: PageShape,
        numa_node: Option<u32>,
    },
    /// The lengths of the leading runs of `keys` committed in the pool
    /// (L2), and committed in the pool or in the distributed tier (L3).
    Lookup { pool: u64, keys: Vec<Key> },
    /// A slot for each key the pool holds no page of, to be written by the
    /// worker of `writer`.
    Reserve {
        pool: u64,
        writer: Scope,
        keys: Vec<Key>,
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
    /// Read the pages of `keys` the pool does not hold from the distributed
    /// tier into slots reserved for them. Answered at once; the event
    /// [`Event::Prefetched`] with the same `tag` follows when the reads
    /// end.
    Prefetch { pool: u64, keys: Vec<Key>, tag: u64 },
    /// The connector ran task `task` of an [`Event::Exists`],
    /// [`Event::Upload`] or [`Event::Download`]: one result per page of the
    /// task.
    Finished { task: u64, results: Vec<bool> },
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
    /// The server failed to hold or pass a descriptor.
    Resource,
}

/// The server's answer to one request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reply {
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
    Listing(Listing),
    Retired {
        released: u64,
    },
    /// The page pool, whose memfd travels with the reply: `slots` slots of
    /// `page_bytes` bytes, slot `i` at byte `i * page_bytes`.
    PagePool {
        pool: u64,
        slots: u64,
        page_bytes: u64,
    },
    Lookup {
        l2: u32,
        l3: u32,
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
    Finished,
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
}

/// One page pool in a [`Listing`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ListedPool {
    pub pool: u64,
    pub shape: PageShape,
    pub numa_node: Option<u32>,
    pub slots: u64,
    /// Slots holding committed pages, pinned ones included.
    pub ready: u64,
    pub reserved: u64,
    pub pinned: u64,
    /// Slots the connector is reading pages into.
    pub fetching: u64,
    /// Committed pages the connector is writing to the distributed tier.
    pub uploading: u64,
}

/// One retained tensor in a [`Listing`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ListedTensor {
    pub key: Key,
    pub name: String,
    pub scopes: Vec<Scope>,
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

/// Storage bytes one scope holds on one placement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScopeBytes {
    /// The GPU's UUID for device bytes; `None` for host bytes.
    pub gpu: Option<[u8; 16]>,
    pub scope: Scope,
    pub bytes: u64,
}

/// One claimed deployment in a [`Listing`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ListedDeployment {
    pub deployment: String,
    pub head_connected: bool,
}

/// A message the server sends outside the answers to requests: the
/// completion of a head's prefetch, and the connector's work.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Event {
    /// To a head: the prefetch `tag` ended. `results[i]` is whether the pool
    /// holds the page of the prefetch's key `i`.
    Prefetched { tag: u64, results: Vec<bool> },
    /// To the connector: a page pool, whose memfd travels with the event.
    Pool {
        pool: u64,
        shape: PageShape,
        slots: u64,
    },
    /// To the connector: which pages of `keys` the distributed tier holds.
    Exists {
        task: u64,
        pool: u64,
        keys: Vec<Key>,
    },
    /// To the connector: write the committed pages in these slots to the
    /// distributed tier.
    Upload {
        task: u64,
        pool: u64,
        pages: Vec<(Key, u64)>,
    },
    /// To the connector: read these pages from the distributed tier into
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
