//! The Mooncake connector: the distributed host KV cache (L3) behind the
//! buffer pools' host regions.
//!
//! The Storage Server runs the connector as a child process
//! (`uniserve storage-server connector`), so that a fault in Mooncake ends
//! the connector and never the server; the server starts it again. The
//! connector connects to the server as [`Role::Connector`], maps every pool
//! of pages the server sends it (a rank's part of a host region), registers
//! each with Mooncake (`register_buffer`), and runs the server's tasks:
//!
//! - [`Event::Upload`]: pages from their slots into Mooncake
//!   (`batch_put_from`);
//! - [`Event::Download`]: pages from Mooncake into their slots
//!   (`batch_get_into`).
//!
//! Each task runs in calls of at most [`BATCH_PAGES`] pages, and a page
//! counts only when its call succeeded. The connector never removes
//! objects: Mooncake's master evicts them.
//!
//! A page's object key is `uniserve_{hex(page key)}_{hex(pool
//! layout)}_{kv head offset}-{kv head count}`, one object per page per rank
//! and representation: a page key names the logical content, which two
//! cache dtypes store differently. It names no deployment or host, so
//! identical content is shared across both.

use std::collections::HashMap;
use std::ffi::c_void;
use std::io;
use std::os::fd::{AsFd, OwnedFd};
use std::path::Path;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use mooncake_store::MooncakeStore;
use rustix::event::{PollFd, PollFlags, poll};

use ::storage::client::Client;
use ::storage::connector::Settings;
use ::storage::protocol::{Event, Key, PageShape, Request, Response, Role};

/// The most pages one Mooncake call carries.
pub const BATCH_PAGES: usize = 128;

use crate::Error;

/// The bytes Mooncake's client keeps for transfers that do not come from a
/// registered buffer. Every page moves between a registered pool and
/// Mooncake, so this only needs to exist.
const LOCAL_BUFFER_BYTES: u64 = 16 << 20;

/// One pool of pages, mapped into the connector and registered with Mooncake.
#[derive(Clone, Copy)]
struct Mapping {
    /// The address of slot 0.
    base: usize,
    slots: u64,
    shape: PageShape,
}

impl Mapping {
    /// The object key of `key`'s page in this pool.
    fn object(&self, key: &Key) -> String {
        format!(
            "uniserve_{}_{}_{}-{}",
            key.hex(),
            self.shape.layout.hex(),
            self.shape.kv_head_offset,
            self.shape.kv_head_count
        )
    }

    /// The address of `slot`, which the server named.
    fn slot(&self, slot: u64) -> Option<*mut c_void> {
        (slot < self.slots)
            .then(|| (self.base + (slot * self.shape.page_bytes) as usize) as *mut c_void)
    }
}

/// A task for the worker thread.
enum Job {
    Upload {
        task: u64,
        pool: u64,
        pages: Vec<(Key, u64)>,
    },
    Download {
        task: u64,
        pool: u64,
        pages: Vec<(Key, u64)>,
    },
}

/// The connector's state shared with its worker threads.
struct Shared {
    store: MooncakeStore,
    pools: Mutex<HashMap<u64, Mapping>>,
    /// Results of finished tasks, waiting for the main thread to report
    /// them.
    finished: Mutex<Vec<(u64, Vec<bool>)>>,
    /// The write end of the main thread's wake pipe.
    wake: OwnedFd,
}

/// Connect to Mooncake with `settings` and to the Storage Server on
/// `socket`, and run the server's tasks until the server closes the
/// connection.
pub fn run(socket: &Path, settings: &Settings) -> Result<(), Error> {
    let hostname = settings
        .hostname
        .as_deref()
        .ok_or_else(|| Error::Settings("the connector needs this host's name".into()))?;
    if let Some(library) = &settings.library {
        mooncake_store::load_library(library)?;
    }
    let store = MooncakeStore::new()?;
    store.setup(
        hostname,
        &settings.metadata,
        settings.segment_bytes,
        LOCAL_BUFFER_BYTES,
        &settings.protocol,
        &settings.device,
        &settings.master,
    )?;
    tracing::info!(master = %settings.master, "connected to Mooncake");

    let mut client = Client::connect(socket, Role::Connector)?;
    let (wake_reader, wake_writer) = rustix::pipe::pipe_with(
        rustix::pipe::PipeFlags::CLOEXEC | rustix::pipe::PipeFlags::NONBLOCK,
    )
    .map_err(io::Error::from)?;
    let shared = Arc::new(Shared {
        store,
        pools: Mutex::new(HashMap::new()),
        finished: Mutex::new(Vec::new()),
        wake: wake_writer,
    });
    let transfers = worker(&shared, "uniserve-l3-transfers")?;

    let mut reports: Vec<u64> = Vec::new();
    loop {
        {
            let connection = client.as_fd();
            let mut fds = [
                PollFd::new(&connection, PollFlags::IN),
                PollFd::new(&wake_reader, PollFlags::IN),
            ];
            match poll(&mut fds, None) {
                Ok(_) | Err(rustix::io::Errno::INTR) => {}
                Err(error) => return Err(io::Error::from(error).into()),
            }
        }
        drain(&wake_reader);
        match client.pump() {
            Ok(()) => {}
            // The server closes the connection when it stops, which ends
            // the connector.
            Err(::storage::Error::Io(error)) if error.kind() == io::ErrorKind::UnexpectedEof => {
                tracing::info!("the storage server closed the connection");
                return Ok(());
            }
            Err(error) => return Err(error.into()),
        }
        while let Some((event, descriptors)) = client.next_event() {
            match event {
                Event::Pool { pool, shape, slots } => {
                    let descriptor = descriptors.into_iter().next().ok_or_else(|| {
                        Error::Settings(format!("pool {pool} came without its storage"))
                    })?;
                    let mapped = map(&shared.store, descriptor, shape, slots)?;
                    lock(&shared.pools).insert(pool, mapped);
                    tracing::info!(
                        pool,
                        slots,
                        page_bytes = shape.page_bytes,
                        "registered a pool of pages"
                    );
                }
                Event::PoolReleased { pool } => {
                    // The server releases a pool once the connector has no
                    // task on it, so no worker thread touches its storage.
                    if let Some(mapped) = lock(&shared.pools).remove(&pool) {
                        unmap(&shared.store, mapped);
                        tracing::info!(pool, "unmapped a released pool of pages");
                    }
                }
                Event::Upload { task, pool, pages } => {
                    let _ = transfers.send(Job::Upload { task, pool, pages });
                }
                Event::Download { task, pool, pages } => {
                    let _ = transfers.send(Job::Download { task, pool, pages });
                }
                // The events of heads, workers and the Transfer Server.
                Event::TaskProgress { .. }
                | Event::TaskEnded { .. }
                | Event::Allocation { .. }
                | Event::AllocationReleased { .. }
                | Event::Task { .. }
                | Event::Cancel { .. }
                | Event::Throttle { .. }
                | Event::Register { .. }
                | Event::Deregister { .. }
                | Event::Allocate { .. }
                | Event::Session { .. } => {}
            }
        }
        let finished = std::mem::take(&mut *lock(&shared.finished));
        for (task, results) in finished {
            reports.push(client.submit(&Request::Finished { task, results }, &[])?);
        }
        // The server acknowledges each report; an error names a task it no
        // longer holds, which changes nothing here.
        reports.retain(|id| match client.take(*id) {
            Some((response, _)) => {
                if let Response::Error { message, .. } = response {
                    tracing::warn!(message, "the storage server refused a task's results");
                }
                false
            }
            None => true,
        });
    }
}

/// Start a thread that runs the jobs sent to it in order.
fn worker(shared: &Arc<Shared>, name: &str) -> io::Result<mpsc::Sender<Job>> {
    let (sender, receiver) = mpsc::channel::<Job>();
    let shared = Arc::clone(shared);
    std::thread::Builder::new()
        .name(name.to_owned())
        .spawn(move || {
            for job in receiver {
                let (task, results) = shared.execute(job);
                lock(&shared.finished).push((task, results));
                let _ = rustix::io::write(&shared.wake, b"x");
            }
        })?;
    Ok(sender)
}

/// The value behind `mutex`. A thread that panicked while holding it left
/// it whole: every holder only inserts or takes whole entries.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

impl Shared {
    /// Run one job; returns its task and one result per page.
    fn execute(&self, job: Job) -> (u64, Vec<bool>) {
        match job {
            Job::Upload { task, pool, pages } => (task, self.transfer(pool, &pages, true)),
            Job::Download { task, pool, pages } => (task, self.transfer(pool, &pages, false)),
        }
    }

    fn pool(&self, pool: u64) -> Option<Mapping> {
        lock(&self.pools).get(&pool).copied()
    }

    /// Move `pages` between their slots and Mooncake: into Mooncake when
    /// `upload`, else out of it. Returns whether each page moved whole.
    fn transfer(&self, pool: u64, pages: &[(Key, u64)], upload: bool) -> Vec<bool> {
        let Some(mapped) = self.pool(pool) else {
            return vec![false; pages.len()];
        };
        let page_bytes = mapped.shape.page_bytes as usize;
        let mut moved = Vec::with_capacity(pages.len());
        for chunk in pages.chunks(BATCH_PAGES) {
            let mut objects = Vec::with_capacity(chunk.len());
            let mut buffers = Vec::with_capacity(chunk.len());
            let mut valid = Vec::with_capacity(chunk.len());
            for (key, slot) in chunk {
                match mapped.slot(*slot) {
                    Some(buffer) => {
                        objects.push(mapped.object(key));
                        buffers.push(buffer);
                        valid.push(true);
                    }
                    None => valid.push(false),
                }
            }
            let names: Vec<&str> = objects.iter().map(String::as_str).collect();
            let sizes = vec![page_bytes; names.len()];
            // Every buffer is a whole slot of a registered pool mapping,
            // which the server releases only once the connector has no task
            // on it (`Event::PoolReleased`), and the buffer pool's task that
            // names a slot holds its block until the connector reports the
            // task.
            let results: Vec<bool> = if upload {
                // SAFETY: see above; Mooncake only reads the slots.
                match unsafe { self.store.batch_put_from(&names, &buffers, &sizes, None) } {
                    Ok(codes) => codes.into_iter().map(|code| code == 0).collect(),
                    Err(error) => {
                        tracing::warn!(%error, "a Mooncake upload failed");
                        vec![false; names.len()]
                    }
                }
            } else {
                // SAFETY: see above; Mooncake writes at most `page_bytes`
                // into each slot.
                match unsafe { self.store.batch_get_into(&names, &buffers, &sizes) } {
                    Ok(read) => read
                        .into_iter()
                        .map(|bytes| bytes == page_bytes as i64)
                        .collect(),
                    Err(error) => {
                        tracing::warn!(%error, "a Mooncake download failed");
                        vec![false; names.len()]
                    }
                }
            };
            let mut results = results.into_iter();
            moved.extend(
                valid
                    .into_iter()
                    .map(|valid| valid && results.next().unwrap_or(false)),
            );
        }
        moved
    }
}

/// Map a pool's memfd and register the mapping with Mooncake. The
/// mapping lives until the server releases the pool ([`unmap`]).
fn map(
    store: &MooncakeStore,
    descriptor: OwnedFd,
    shape: PageShape,
    slots: u64,
) -> io::Result<Mapping> {
    let bytes = (slots * shape.page_bytes) as usize;
    // SAFETY: a fresh shared mapping of the whole memfd, which the server
    // sized to `slots * page_bytes`.
    let base = unsafe {
        rustix::mm::mmap(
            std::ptr::null_mut(),
            bytes,
            rustix::mm::ProtFlags::READ | rustix::mm::ProtFlags::WRITE,
            rustix::mm::MapFlags::SHARED,
            descriptor.as_fd(),
            0,
        )
    }?;
    // SAFETY: the mapping above, which `unmap` unregisters before it unmaps
    // it.
    unsafe { store.register_buffer(base, bytes) }.map_err(|error| {
        io::Error::other(format!(
            "registering a pool of pages with Mooncake: {error}"
        ))
    })?;
    Ok(Mapping {
        base: base as usize,
        slots,
        shape,
    })
}

/// Unregister a released pool's mapping from Mooncake and unmap it.
fn unmap(store: &MooncakeStore, mapped: Mapping) {
    let base = mapped.base as *mut c_void;
    // SAFETY: a mapping `map` registered; no task reads or writes it.
    if let Err(error) = unsafe { store.unregister_buffer(base) } {
        tracing::warn!(%error, "unregistering a released pool from Mooncake failed");
    }
    let bytes = (mapped.slots * mapped.shape.page_bytes) as usize;
    // SAFETY: the mapping `map` made of the whole pool, which nothing uses.
    let unmapped = unsafe { rustix::mm::munmap(base, bytes) };
    if let Err(error) = unmapped {
        tracing::warn!(%error, "unmapping a released pool failed");
    }
}

/// Empty the wake pipe.
fn drain(reader: &OwnedFd) {
    let mut buffer = [0_u8; 64];
    while matches!(rustix::io::read(reader, &mut buffer), Ok(read) if read > 0) {}
}
