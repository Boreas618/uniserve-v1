//! The rank's storage layer through its public surface: a model's load over the worker's weight
//! stores, through `nn::WeightLoader`, and a rank's connection to a Storage Server in the
//! test's process.
//!
//! The host tests load a small checkpoint over the meta store and connect ranks to the server.
//! The device tests load the checkpoint over the weight store of a GPU: publishing a weight
//! creates a device allocation and copies into it on a stream, and mapping one imports the
//! allocation's descriptor into the device's address space, so they need a GPU.

// The helpers below unwrap, as tests do (`clippy.toml`).
#![cfg(test)]

use std::os::fd::{AsFd, OwnedFd};
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use nn::{Checkpoint, DType, RetainedWeights, Shape, Tensor, WeightKey, WeightLoader};
use platform::DeviceId;
use storage::{Claim, Client, Role, Scope};
use storage_server::Server;
use uniserve_tensor::DeviceLocation;
use worker::memory::client::StorageClient;
use worker::memory::reload::{MetaWeights, Reload, paired};
use worker::memory::weights::WeightStore;

/// The deployment, worker group and epoch the device tests' rank belongs to.
const DEPLOYMENT: &str = "model";
const GROUP: &str = "text";
const EPOCH: u64 = 1;

/// The elements of the fixture's norm: a weight far smaller than a device's allocation
/// granularity, which a device store packs.
const NORM: usize = 8;

/// The columns of the fixture's embedding.
const COLUMNS: usize = 8;

/// The norm's values in a checkpoint whose values start at `first`: BF16 has every integer of
/// this range, so the bytes a load keeps are the bytes the fixture wrote.
fn norm_bytes(first: f32) -> Vec<u8> {
    let values: Vec<f32> = (0..NORM).map(|index| first + index as f32).collect();
    ops::values::encode(DType::BF16, &values).unwrap()
}

/// The embedding's values, `rows` rows of [`COLUMNS`] FP32 elements counting up from `first`.
fn embed_bytes(rows: usize, first: f32) -> Vec<u8> {
    (0..rows * COLUMNS)
        .flat_map(|index| (first + index as f32).to_le_bytes())
        .collect()
}

/// Writes a checkpoint of one shard under `directory` and opens it: `norm.weight`, [`NORM`]
/// BF16 elements, and `embed.weight`, `rows` rows of [`COLUMNS`] FP32 elements, with values
/// counting up from `first`.
///
/// The shard is composed from the safetensors grammar (an eight-byte length, the JSON header
/// padded to eight bytes, then the payload), so the fixture states the format without a writer.
fn checkpoint(directory: &Path, rows: usize, first: f32) -> Checkpoint {
    let (norm, embed) = (norm_bytes(first), embed_bytes(rows, first));

    let mut header = format!(
        r#"{{"norm.weight":{{"dtype":"BF16","shape":[{NORM}],"data_offsets":[0,{}]}},"embed.weight":{{"dtype":"F32","shape":[{rows},{COLUMNS}],"data_offsets":[{},{}]}}}}"#,
        norm.len(),
        norm.len(),
        norm.len() + embed.len()
    );
    let padded = header.len().div_ceil(8) * 8;
    header.push_str(&" ".repeat(padded - header.len()));

    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&norm);
    file.extend_from_slice(&embed);
    std::fs::write(directory.join("model.safetensors"), file).unwrap();

    Checkpoint::open(directory).unwrap()
}

/// Loads the fixture's two weights from `checkpoint` through `store`, as a model's modules do:
/// the norm, then the embedding.
fn load(checkpoint: &Checkpoint, store: &dyn RetainedWeights, rows: usize) -> (Tensor, Tensor) {
    let loader = WeightLoader::new(checkpoint, store, None, None);
    let root = loader.root();

    let norm = root.pp("norm").get(NORM, "weight").unwrap();
    let embed = root.pp("embed").get((rows, COLUMNS), "weight").unwrap();
    (norm, embed)
}

/// The checkpoint tensors each weight of `order` was read from.
fn names(order: &[(WeightKey, bool)]) -> Vec<&str> {
    order
        .iter()
        .flat_map(|(key, _)| key.tensors.iter().map(String::as_str))
        .collect()
}

/// A loader over the meta store runs the model's own load over the shard's header: every
/// weight is a tensor of the element type and extents a device load gives, with no data, and
/// the store states the weights the load reached, in its order, with the table the model
/// writes after its load marked as one.
#[test]
fn a_load_over_the_meta_store_states_its_weights_in_order_and_holds_no_data() {
    const ROWS: usize = 4;
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(directory.path(), ROWS, 1.0);
    let meta = DeviceLocation::Backend {
        kind: "meta",
        ordinal: 0,
    };

    let store = MetaWeights::new();
    let (norm, embed) = load(&checkpoint, &store, ROWS);
    let table = WeightLoader::new(&checkpoint, &store, None, None)
        .writable_constant(
            "frequencies",
            DType::F32,
            &Shape::from(2),
            &[0_u8; 8],
            "worker::tests",
        )
        .unwrap();

    assert_eq!(
        (norm.dims(), norm.dtype()),
        ([NORM].as_slice(), DType::BF16)
    );
    assert_eq!(
        (embed.dims(), embed.dtype()),
        ([ROWS, COLUMNS].as_slice(), DType::F32)
    );
    assert_eq!((table.dims(), table.dtype()), ([2].as_slice(), DType::F32));
    for tensor in [&norm, &embed, &table] {
        assert_eq!(tensor.device().location(), meta);
    }

    let order = store.into_order();
    assert_eq!(
        names(&order),
        ["norm.weight", "embed.weight", "frequencies"]
    );
    assert_eq!(
        order.iter().map(|(_, table)| *table).collect::<Vec<_>>(),
        [false, false, true]
    );
}

/// A Storage Server on a socket of its own, in the test's process, stopped when dropped.
struct StorageServer {
    socket: PathBuf,
    stop: Arc<AtomicBool>,
    /// The write end of the pipe whose readability wakes the server's loop.
    wake: OwnedFd,
    thread: Option<std::thread::JoinHandle<()>>,
    _directory: tempfile::TempDir,
}

impl StorageServer {
    fn start() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("storage.sock");
        let listener = UnixListener::bind(&socket).unwrap();
        let (reader, wake) = rustix::pipe::pipe_with(
            rustix::pipe::PipeFlags::NONBLOCK | rustix::pipe::PipeFlags::CLOEXEC,
        )
        .unwrap();

        let stop = Arc::new(AtomicBool::new(false));
        let stopped = Arc::clone(&stop);
        let thread = std::thread::spawn(move || {
            let mut server = Server::new(listener, 1 << 20, 0, None, None).unwrap();
            server
                .run(reader.as_fd(), &|| stopped.load(Ordering::Acquire))
                .unwrap();
        });

        Self {
            socket,
            stop,
            wake,
            thread: Some(thread),
            _directory: directory,
        }
    }
}

impl Drop for StorageServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        rustix::io::write(&self.wake, b"x").unwrap();

        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}

/// The scope of rank 0 of the tests' worker group.
fn scope() -> Scope {
    Scope {
        deployment: DEPLOYMENT.into(),
        group: GROUP.into(),
        rank: 0,
    }
}

/// The head of the tests' deployment at `server`, which claims its epochs.
fn head(server: &StorageServer) -> Client {
    Client::connect(
        &server.socket,
        Role::Head {
            deployment: DEPLOYMENT.into(),
            epoch: EPOCH,
        },
    )
    .unwrap()
}

/// What a rank of `epoch` is launched with to reach `server` under `claim`.
fn launch(server: &StorageServer, claim: Claim, epoch: u64, standby: bool) -> ipc::StorageLaunch {
    ipc::StorageLaunch {
        socket: server.socket.to_string_lossy().into_owned(),
        deployment: DEPLOYMENT.into(),
        claim: claim.0,
        host_kv_cache: false,
        tcp: None,
        peers: Vec::new(),
        epoch,
        standby,
        standby_storage: ipc::StandbyStorage::default(),
    }
}

/// Claims the deployment at `server` as its head and connects rank 0 of its worker group. The
/// head's connection is returned with the rank's client, because the claim lasts while the
/// head is connected.
fn rank(server: &StorageServer) -> (Client, Arc<StorageClient>) {
    let mut head = head(server);
    let claim = head.claim(EPOCH, vec![scope()], Vec::new()).unwrap();

    let client =
        StorageClient::connect(&launch(server, claim, EPOCH, false), GROUP, 0, 1, None).unwrap();
    (head, client)
}

/// A rank connects to its host's Storage Server under the claim its deployment's head holds,
/// and makes requests as its scope. It keys the exclusive storages it takes by its own epoch;
/// a standby keys them by the epoch of the worker it stands by for, whose storages it maps.
#[test]
fn a_rank_keys_its_exclusive_storages_by_its_epoch_and_a_standby_by_its_predecessor_s() {
    let server = StorageServer::start();
    let mut head = head(&server);
    let claim = head.claim(EPOCH, vec![scope()], Vec::new()).unwrap();

    let serving =
        StorageClient::connect(&launch(&server, claim, EPOCH, false), GROUP, 0, 1, None).unwrap();
    assert_eq!(serving.scope(), &scope());
    assert_eq!((serving.epoch(), serving.storage_epoch()), (EPOCH, EPOCH));

    // The head adds the next epoch under the claim it holds; the rank of that epoch that
    // succeeds the serving one keys its own storages, and its standby the serving rank's.
    let next = EPOCH + 1;
    let claim = head.claim(next, vec![scope()], Vec::new()).unwrap();
    let successor = StorageClient::connect(
        &launch(&server, claim, next, false),
        GROUP,
        0,
        2,
        Some(EPOCH),
    )
    .unwrap();
    let standby = StorageClient::connect(
        &launch(&server, claim, next, true),
        GROUP,
        0,
        2,
        Some(EPOCH),
    )
    .unwrap();
    assert_eq!((successor.epoch(), successor.storage_epoch()), (next, next));
    assert_eq!((standby.epoch(), standby.storage_epoch()), (next, EPOCH));

    // A request on the connection reaches the server as the rank's: the deployment it lists
    // is the one the rank's scope names.
    let listing = serving
        .call("listing the storage server", Client::list)
        .unwrap();
    let deployments: Vec<&str> = listing
        .deployments
        .iter()
        .map(|entry| entry.deployment.as_str())
        .collect();
    assert_eq!(deployments, [DEPLOYMENT]);
}

/// The bytes of `tensor`, a weight on `device`, copied to the host.
fn device_bytes(tensor: &Tensor, device: DeviceId) -> Vec<u8> {
    let operand = backend::operand(tensor).unwrap();
    let length = operand.byte_len().unwrap() as usize;
    let mut bytes = vec![0_u8; length];
    let stream = platform::Stream::new(device).unwrap();

    // SAFETY: the weight is mapped for as long as its store lives, which the caller holds, and
    // the vector holds `length` writable bytes that are read only after the stream completed
    // the copy.
    unsafe {
        platform::memcpy_async(
            bytes.as_mut_ptr() as u64,
            operand.ptr.address(),
            length,
            &stream,
        )
        .unwrap();
    }
    stream.synchronize().unwrap();
    bytes
}

/// The rows of the device tests' embedding: 4 MiB of FP32, above any device's allocation
/// granularity, so the weight takes an allocation of its own while the norm is packed.
const DEVICE_ROWS: usize = (4 << 20) / (COLUMNS * 4);

/// A loader over the weight store publishes a checkpoint's weights as tensors on the device,
/// and a store that loads the checkpoint again maps what was published instead of converting
/// it: the same keys, shapes, element types and bytes, with nothing published a second time.
#[test]
#[ignore = "requires a GPU"]
fn a_loader_publishes_a_checkpoint_and_a_second_load_maps_what_was_published() {
    let device = DeviceId(0);
    let cuda = DeviceLocation::Backend {
        kind: "cuda",
        ordinal: 0,
    };
    let server = StorageServer::start();
    let (_head, client) = rank(&server);
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(directory.path(), DEVICE_ROWS, 1.0);

    let first = WeightStore::new(Arc::clone(&client), device).unwrap();
    let (norm, embed) = load(&checkpoint, &first, DEVICE_ROWS);
    let published = first.finish().unwrap();

    assert_eq!(names(&published), ["norm.weight", "embed.weight"]);
    assert_eq!((first.published(), first.from_store()), (2, 0));
    assert_eq!(
        (norm.dims(), norm.dtype()),
        ([NORM].as_slice(), DType::BF16)
    );
    assert_eq!(
        (embed.dims(), embed.dtype()),
        ([DEVICE_ROWS, COLUMNS].as_slice(), DType::F32)
    );
    for tensor in [&norm, &embed] {
        assert_eq!(tensor.device().location(), cuda);
    }

    let second = WeightStore::new(Arc::clone(&client), device).unwrap();
    let (norm, embed) = load(&checkpoint, &second, DEVICE_ROWS);
    let mapped = second.finish().unwrap();

    assert_eq!(mapped, published);
    assert_eq!((second.published(), second.from_store()), (0, 2));
    assert_eq!(
        (norm.dims(), norm.dtype()),
        ([NORM].as_slice(), DType::BF16)
    );
    assert_eq!(
        (embed.dims(), embed.dtype()),
        ([DEVICE_ROWS, COLUMNS].as_slice(), DType::F32)
    );
    assert_eq!(device_bytes(&norm, device), norm_bytes(1.0));
    assert_eq!(device_bytes(&embed, device), embed_bytes(DEVICE_ROWS, 1.0));
}

/// An update in place loads another checkpoint of the model beside the weights that serve and
/// puts each changed weight at the address of the one it replaces: the tensors the first load
/// returned, which a rank's modules and graphs hold, read the other checkpoint's bytes.
#[test]
#[ignore = "requires a GPU"]
fn an_update_in_place_puts_the_changed_weights_at_the_serving_addresses() {
    let device = DeviceId(0);
    let server = StorageServer::start();
    let (_head, client) = rank(&server);
    let (served, updated) = (tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap());
    let serving_checkpoint = checkpoint(served.path(), DEVICE_ROWS, 1.0);
    let target_checkpoint = checkpoint(updated.path(), DEVICE_ROWS, 100.0);

    let store = WeightStore::new(client, device).unwrap();
    let (norm, embed) = load(&serving_checkpoint, &store, DEVICE_ROWS);
    let serving = store.finish().unwrap();

    // The target's load is stated first, with no storage, and compared with the one that
    // serves: both weights are read from another file, so both are replaced.
    let stated = MetaWeights::new();
    load(&target_checkpoint, &stated, DEVICE_ROWS);
    let target = stated.into_order();
    let pairs = paired(&serving, &target, false).unwrap();
    assert_eq!(pairs.len(), 2);

    let reload = Reload::new(&store, &serving).unwrap();
    load(&target_checkpoint, &reload, DEVICE_ROWS);
    assert_eq!(reload.finish().unwrap(), target);

    assert_eq!(store.rebind(&pairs).unwrap(), 2);
    assert_eq!(device_bytes(&norm, device), norm_bytes(100.0));
    assert_eq!(
        device_bytes(&embed, device),
        embed_bytes(DEVICE_ROWS, 100.0)
    );
}
