//! The graph pool of a device through its public surface: the runners' regions mapped at fixed
//! addresses, their reservation in the GPU's quota at a Storage Server in the test's process,
//! and suspension and restoration at the same addresses.
//!
//! Every case maps device memory, so every case needs a GPU.

// The helpers below unwrap, as tests do (`clippy.toml`).
#![cfg(test)]

use std::os::fd::{AsFd, OwnedFd};
use std::os::unix::net::UnixListener;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use engine::memory::client::{StorageClient, quota_limit};
use engine::memory::graph_pool::GraphPool;
use engine::runner::RunnerId;
use platform::{DeviceId, DeviceRange, PinnedBuffer, PoolKind, Stream};
use storage::{Client, Item, Role, Scope};
use storage_server::Server;

const DEPLOYMENT: &str = "model";
const GROUP: &str = "text";
const EPOCH: u64 = 1;

const MIB: u64 = 1 << 20;

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
            let mut server = Server::new(listener, 0, None, None).unwrap();
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

/// Claims the deployment at `server` as its head and connects rank 0 of its worker group. The
/// head's connection is returned with the rank's client, because the claim lasts while the
/// head is connected.
fn rank(server: &StorageServer) -> (Client, Arc<StorageClient>) {
    let scope = Scope {
        deployment: DEPLOYMENT.into(),
        group: GROUP.into(),
        rank: 0,
    };
    let mut head = Client::connect(
        &server.socket,
        Role::Head {
            deployment: DEPLOYMENT.into(),
            epoch: EPOCH,
            generation: 0,
        },
    )
    .unwrap();
    let claim = head.claim(EPOCH, vec![scope], Vec::new()).unwrap();

    let launch = engine::memory::client::StorageLaunch {
        socket: server.socket.to_string_lossy().into_owned(),
        deployment: DEPLOYMENT.into(),
        claim: claim.0,
        tcp: None,
        peers: Vec::new(),
        epoch: EPOCH,
        standby: false,
        standby_storage: proto::request::StandbyStorage::default(),
    };
    let client = StorageClient::connect_rank(&launch, GROUP, 0, 1).unwrap();
    (head, client)
}

/// The bytes the rank's pools reserve in the quota of `device`.
fn reserved(client: &StorageClient, device: DeviceId) -> u64 {
    let (quota, _) = client.usage(device).unwrap();
    quota.bytes(|reservation| reservation.item == Item::GraphPool)
}

/// The bytes the report of `pool` states for the pool named `name`.
fn reported(pool: &GraphPool, name: &str) -> u64 {
    pool.usage()
        .pools
        .iter()
        .find(|entry| entry.pool == name)
        .map(|entry| entry.mapped)
        .unwrap()
}

/// Each runner's region is mapped when the pool opens, holds at least the runner's bytes,
/// overlaps no other region, and is reserved in the GPU's quota with the scratch the pool
/// maps afterwards.
#[test]
#[ignore = "requires a GPU"]
fn the_runners_regions_are_mapped_apart_and_reserved_in_the_quota() {
    let device = DeviceId(0);
    let server = StorageServer::start();
    let (_head, client) = rank(&server);

    let sizes = [(RunnerId(0), 5 * MIB), (RunnerId(1), 3 * MIB)];
    let mut pool =
        GraphPool::open(Arc::clone(&client), device, 0.5, &sizes, 4 * MIB, 4 * MIB).unwrap();

    let first = pool.region(RunnerId(0)).unwrap();
    let second = pool.region(RunnerId(1)).unwrap();
    assert!(first.bytes() as u64 >= 5 * MIB && second.bytes() as u64 >= 3 * MIB);

    let (start, end) = (
        first.ptr().address(),
        first.ptr().address() + first.bytes() as u64,
    );
    let other = second.ptr().address();
    assert!(other >= end || other + second.bytes() as u64 <= start);

    // A region's address does not change while the pool lives.
    assert_eq!(pool.region(RunnerId(0)).unwrap(), first);
    assert!(pool.region(RunnerId(2)).is_err());

    let graph = (first.bytes() + second.bytes()) as u64;
    assert_eq!(pool.mapped(), graph);
    assert_eq!(reserved(&client, device), graph);

    // The shadow region of a runner comes from the scratch pool and joins the reservation.
    let shadow = pool.shadow(MIB).unwrap();
    assert!(shadow.capacity() >= MIB);
    assert_eq!(reported(&pool, "graph"), graph);
    assert_eq!(reported(&pool, "scratch"), shadow.capacity());
    assert_eq!(reserved(&client, device), graph + shadow.capacity());

    // A state owner's live buffers come from the resident pool through the pool's allocator
    // and join the reservation too.
    let live = pool
        .allocator()
        .allocate(PoolKind::Resident, MIB as usize)
        .unwrap();
    let resident = DeviceRange::from(&live).bytes() as u64;
    assert!(resident >= MIB);
    assert_eq!(reported(&pool, "resident"), resident);
    assert_eq!(
        reserved(&client, device),
        graph + shadow.capacity() + resident
    );

    let total = device.memory_info().unwrap().total;
    assert_eq!(pool.usage().budget, quota_limit(total, 0.5));
    assert_eq!(pool.usage().device, 0);
}

/// A pool whose regions the quota cannot hold is refused when it opens, and reserves nothing.
#[test]
#[ignore = "requires a GPU"]
fn regions_past_the_quota_are_refused_and_reserve_nothing() {
    let device = DeviceId(0);
    let server = StorageServer::start();
    let (_head, client) = rank(&server);

    // A hundredth of the device is the limit; the second region alone is a fiftieth.
    let total = device.memory_info().unwrap().total;
    let sizes = [(RunnerId(0), 2 * MIB), (RunnerId(1), total / 50)];
    let refused = GraphPool::open(Arc::clone(&client), device, 0.01, &sizes, 0, 0);
    assert!(refused.is_err());

    // The first region was mapped and reserved before the second was refused; its storage
    // goes back through the process's reclaimer, which unmaps it and then books the pool's
    // reservation down to nothing. A flush waits for that release.
    platform::reclaim::flush().unwrap();
    assert_eq!(reserved(&client, device), 0);
}

/// A suspended pool maps nothing and reserves nothing; restored, its regions are at the
/// addresses they had and hold the bytes they held.
#[test]
#[ignore = "requires a GPU"]
fn a_suspended_pool_is_restored_at_the_same_addresses_with_its_bytes() {
    let device = DeviceId(0);
    let server = StorageServer::start();
    let (_head, client) = rank(&server);

    let sizes = [(RunnerId(0), 4 * MIB)];
    let mut pool = GraphPool::open(Arc::clone(&client), device, 0.5, &sizes, 0, 0).unwrap();
    let region = pool.region(RunnerId(0)).unwrap();
    let mapped = pool.mapped();

    // The region's first bytes, written from page-locked host memory and read back after the
    // restore.
    let stream = Stream::new(device).unwrap();
    let written: Vec<u8> = (0..=255).collect();
    let mut host = PinnedBuffer::new(device, written.len()).unwrap();
    // SAFETY: the buffer is this test's alone, and no copy is in flight.
    unsafe { host.as_mut_slice() }.copy_from_slice(&written);
    // SAFETY: the host buffer and the region are mapped and at least as long as the copy,
    // and the stream is waited for before either is touched again.
    unsafe {
        platform::memcpy_async(
            region.ptr().address(),
            host.address(),
            written.len(),
            &stream,
        )
    }
    .unwrap();
    stream.synchronize().unwrap();

    pool.suspend().unwrap();
    assert_eq!(pool.mapped(), 0);
    assert_eq!(reserved(&client, device), 0);

    pool.restore().unwrap();
    assert_eq!(pool.region(RunnerId(0)).unwrap(), region);
    assert_eq!(pool.mapped(), mapped);
    assert_eq!(reserved(&client, device), mapped);

    // SAFETY: as above; the copy has ended, so the buffer is this test's alone again.
    unsafe { host.as_mut_slice() }.fill(0);
    // SAFETY: as the copy to the device.
    unsafe {
        platform::memcpy_async(
            host.address(),
            region.ptr().address(),
            written.len(),
            &stream,
        )
    }
    .unwrap();
    stream.synchronize().unwrap();
    // SAFETY: the copy has ended.
    assert_eq!(unsafe { host.as_slice() }, written);
}
