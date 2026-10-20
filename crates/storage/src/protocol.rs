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

use serde::{Deserialize, Serialize};

/// The protocol version this build speaks.
pub const VERSION: u32 = 1;

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
    /// The head of a deployment on this host: the head itself, or on another
    /// host the head's launcher there.
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
    /// keep the caller as their writer.
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

/// An asynchronous completion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Event {}

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
