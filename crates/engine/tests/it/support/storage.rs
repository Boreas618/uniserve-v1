//! A Storage Server on a socket of its own, in the test's process, and the
//! host page pools simulated video ranks map there: a denoising rank's
//! latent pool and an encoder rank's encoder pool.

use std::os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use storage::protocol::{PageShape, Reply, Request};
use storage::{Claim, Client, Key, Role, Scope};
use storage_server::Server;

/// The deployment the fixture's pools belong to, which a scheduler's
/// trajectory store names as its head.
pub(crate) const DEPLOYMENT: &str = "model";

/// The host the fixture's ranks run on, which a scheduler's trajectory
/// store is told is its own.
pub(crate) const HOST: &str = "local";

/// A running Storage Server, stopped when dropped.
pub(crate) struct StorageServer {
    socket: PathBuf,
    stop: Arc<AtomicBool>,
    wake: OwnedFd,
    thread: Option<std::thread::JoinHandle<()>>,
    /// The worker connections that created the pools: the server keeps a
    /// pool's pages for its deployment while a worker of it is connected.
    _workers: Vec<Client>,
    _directory: tempfile::TempDir,
}

impl StorageServer {
    /// Starts a server whose page pools hold `pool_bytes` unless a worker
    /// asks for another size.
    pub(crate) fn start(pool_bytes: u64) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("storage.sock");
        let (listener, lock) = storage::socket::bind(&socket).unwrap().unwrap();
        let mut ends = [0; 2];
        // SAFETY: pipe2 fills the two descriptors it creates.
        let created = unsafe { libc::pipe2(ends.as_mut_ptr(), libc::O_NONBLOCK) };
        assert_eq!(created, 0);
        // SAFETY: two new descriptors this fixture owns.
        let (reader, writer) =
            unsafe { (OwnedFd::from_raw_fd(ends[0]), OwnedFd::from_raw_fd(ends[1])) };
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = Arc::clone(&stop);
        let thread = std::thread::spawn(move || {
            let _lock = lock;
            let mut server = Server::new(listener, pool_bytes, 0, None, None).unwrap();
            server
                .run(reader.as_fd(), &|| stopped.load(Ordering::Acquire))
                .unwrap();
        });
        Self {
            socket,
            stop,
            wake: writer,
            thread: Some(thread),
            _workers: Vec::new(),
            _directory: directory,
        }
    }

    pub(crate) fn socket(&self) -> &Path {
        &self.socket
    }

    /// A page pool of pages of `page_bytes` bytes, as rank 0 of worker group
    /// `group` maps it for its latent pages, holding `slots` pages.
    pub(crate) fn latent_pool(&mut self, group: &str, page_bytes: u64, slots: u64) -> u64 {
        self.page_pool(group, Key([7; 32]), page_bytes, slots)
    }

    /// A page pool of pages of `page_bytes` bytes, as rank 0 of worker group
    /// `group` maps it for its encoder outputs, holding `slots` pages.
    pub(crate) fn encoder_pool(&mut self, group: &str, page_bytes: u64, slots: u64) -> u64 {
        self.page_pool(group, Key([8; 32]), page_bytes, slots)
    }

    /// A page pool of pages laid out as `layout` says, which rank 0 of
    /// worker group `group` maps.
    fn page_pool(&mut self, group: &str, layout: Key, page_bytes: u64, slots: u64) -> u64 {
        let mut worker = Client::connect(
            &self.socket,
            Role::Worker {
                scope: Scope {
                    deployment: DEPLOYMENT.into(),
                    group: group.into(),
                    rank: 0,
                },
                generation: 1,
                claim: Claim([0; 16]),
                world: 0,
            },
        )
        .unwrap();
        let shape = PageShape {
            layout,
            kv_head_offset: 0,
            kv_head_count: 0,
            page_bytes,
        };
        let request = Request::PagePool {
            shape,
            numa_node: None,
            bytes: Some(slots * page_bytes),
        };
        let Reply::PagePool { pool, .. } = worker.call(&request, &[]).unwrap().reply else {
            panic!("the latent pool was refused");
        };
        self._workers.push(worker);
        pool
    }
}

impl Drop for StorageServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        // SAFETY: a write to a pipe this fixture owns.
        unsafe { libc::write(self.wake.as_raw_fd(), b"x".as_ptr().cast(), 1) };
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}
