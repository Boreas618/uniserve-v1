//! A Mooncake Transfer Engine, its registrations and the transfers it posts.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::sync::Arc;
use std::time::Instant;

use super::capi::{self, Api, BatchId, Handle, SegmentId};
use super::library;
use super::{Error, Result};

/// The metadata service of every engine these bindings create: peer-to-peer handshakes, in
/// which an engine serves its own segment's description on the port its name states and no
/// metadata server takes part.
const P2P_HANDSHAKE: &CStr = c"P2PHANDSHAKE";

/// The location every registration states: the engine finds the storage's device (a GPU's, or
/// host storage's NUMA node) from its address, and the network adapters nearest it.
const ANY_LOCATION: &CStr = c"*";

/// One process's Transfer Engine.
///
/// The engine is created with peer-to-peer handshakes ([`P2P_HANDSHAKE`]): it listens on a port
/// of its own for other engines' handshakes, which fetch the description of its segment (the
/// storage it registered, remotely accessible) and set its RDMA connections up, and its name
/// ([`Engine::name`], `host:port`) is the segment name another engine opens it by. It discovers
/// the host's topology and installs the transports the host has (RDMA over every usable adapter,
/// TCP without one) when it starts.
///
/// The engine is synchronized within: its clones, registrations and batches call it from any
/// thread at once.
#[derive(Clone)]
pub(crate) struct Engine {
    shared: Arc<Shared>,
}

/// The engine's handle, shared by its clones, registrations and batches, which keep it alive
/// until they drop.
pub(crate) struct Shared {
    api: &'static Api,
    handle: Handle,
    name: String,
}

// SAFETY: the Transfer Engine's calls lock its own state: Mooncake's own clients call one engine
// from many threads at once.
unsafe impl Send for Shared {}
// SAFETY: as for `Send`.
unsafe impl Sync for Shared {}

/// `Ok` for a call that returned 0, the failure of `call` otherwise.
fn check(call: &'static str, code: c_int) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(Error::TransferEngine { call, code })
    }
}

impl Engine {
    /// Creates an engine that other hosts reach at `host` (an address or a host name of this
    /// host), on a port it picks. Loads the Transfer Engine's library from the loader's default
    /// search path on first use.
    ///
    /// # Errors
    ///
    /// Fails when the library cannot be loaded, or the engine cannot start: no port to listen
    /// on, or a transport that does not install.
    pub(crate) fn new(host: &str) -> Result<Self> {
        let api = library::api()?;
        let c_host = CString::new(host).map_err(|_| {
            Error::TransferEngineArgument("a host's name holds no NUL byte".to_owned())
        })?;

        // SAFETY: the strings outlive the call, which copies them; it returns a new engine, or
        // null when the engine did not start.
        let handle = unsafe {
            (api.create)(
                P2P_HANDSHAKE.as_ptr(),
                c_host.as_ptr(),
                c_host.as_ptr(),
                0,
                1,
            )
        };
        if handle.is_null() {
            return Err(Error::TransferEngine {
                call: "creating an engine",
                code: -1,
            });
        }

        // The name an engine of another host opens this one's segment by: `host:port`, the port
        // the engine picked for its handshakes.
        let mut buffer = [0 as c_char; 512];
        // SAFETY: a live engine and a buffer of the length stated, which the call writes a
        // NUL-terminated string into.
        let named = check("reading the engine's name", unsafe {
            (api.local_name)(handle, buffer.as_mut_ptr(), buffer.len())
        });
        let name = named.and_then(|()| {
            // SAFETY: the call wrote a NUL-terminated string within the buffer.
            let name = unsafe { CStr::from_ptr(buffer.as_ptr()) };
            name.to_str().map(str::to_owned).map_err(|_| {
                Error::TransferEngineArgument("the engine's name is not UTF-8".to_owned())
            })
        });
        let name = match name {
            Ok(name) if !name.is_empty() => name,
            Ok(_) | Err(_) => {
                // SAFETY: the engine created above, which nothing else holds.
                unsafe { (api.destroy)(handle) };
                return Err(Error::TransferEngineArgument(
                    "the engine states no name to open its segment by".to_owned(),
                ));
            }
        };

        Ok(Self {
            shared: Arc::new(Shared { api, handle, name }),
        })
    }

    /// The name of the engine's segment, `host:port`, which another engine opens it by
    /// ([`Engine::open`]).
    pub(crate) fn name(&self) -> &str {
        &self.shared.name
    }

    /// Registers `len` bytes from `address` (a host pointer or a CUDA device address of this
    /// process) with every transport of the engine, remotely accessible: this engine's
    /// transfers read and write it, and engines that open this one's segment from now on read
    /// and write it. A range is registered once: a range that overlaps a registration is
    /// refused.
    ///
    /// # Safety
    ///
    /// Until the registration drops, the range stays mapped writable at its address in this
    /// process, and no Rust reference to its bytes is held while a transfer may write them: its
    /// bytes are read and written outside Rust, by this engine's transfers and by other
    /// engines'.
    ///
    /// # Errors
    ///
    /// Fails for an empty range, an overlapping one, or one a transport cannot register.
    pub(crate) unsafe fn register(&self, address: u64, len: usize) -> Result<Registration> {
        if len == 0 {
            return Err(Error::TransferEngineArgument(
                "a registered range is not empty".to_owned(),
            ));
        }
        // SAFETY: a live engine; the caller keeps the range mapped while it is registered. A
        // device address is a unified address, which the engine takes as a pointer value.
        check("registering storage", unsafe {
            (self.shared.api.register)(
                self.shared.handle,
                address as *mut c_void,
                len,
                ANY_LOCATION.as_ptr(),
                1,
            )
        })?;
        Ok(Registration {
            shared: Arc::clone(&self.shared),
            address,
            len,
        })
    }

    /// The segment of the engine named `name`, with the storage it registered up to now: its
    /// description is fetched anew from that engine at every call, so a range it registered
    /// since the last call is reachable.
    ///
    /// # Errors
    ///
    /// Fails when that engine cannot be reached or does not answer the handshake.
    pub(crate) fn open(&self, name: &str) -> Result<SegmentId> {
        let c_name = CString::new(name).map_err(|_| {
            Error::TransferEngineArgument("a segment's name holds no NUL byte".to_owned())
        })?;
        // SAFETY: a live engine and a string the call copies.
        let segment =
            unsafe { (self.shared.api.open_uncached)(self.shared.handle, c_name.as_ptr()) };
        if segment < 0 {
            return Err(Error::TransferEngine {
                call: "opening another engine's segment",
                code: segment,
            });
        }
        Ok(segment)
    }

    /// Posts one READ of every span's remote range, in `segment`'s registered storage, into its
    /// local range, in storage this engine registered. The caller keeps every destination
    /// mapped and unread until the batch ended.
    ///
    /// # Errors
    ///
    /// Fails for no span or an empty one, or when the engine refuses the batch.
    pub(crate) fn read(&self, segment: SegmentId, spans: &[Span]) -> Result<Batch> {
        self.post(segment, spans, capi::OPCODE_READ)
    }

    /// Posts one WRITE of every span's local range into its remote range: [`Engine::read`] the
    /// other way, which a handoff's push takes. The caller keeps every source unchanged until
    /// the batch ended, and the other process reads no destination before then.
    ///
    /// # Errors
    ///
    /// As [`Engine::read`].
    pub(crate) fn write(&self, segment: SegmentId, spans: &[Span]) -> Result<Batch> {
        self.post(segment, spans, capi::OPCODE_WRITE)
    }

    fn post(&self, segment: SegmentId, spans: &[Span], opcode: c_int) -> Result<Batch> {
        if spans.is_empty() || spans.iter().any(|span| span.len == 0) {
            return Err(Error::TransferEngineArgument(
                "a transfer has at least one span, none of them empty".to_owned(),
            ));
        }
        let mut requests: Vec<capi::Request> = spans
            .iter()
            .map(|span| capi::Request {
                opcode,
                source: span.local as *mut c_void,
                target_id: segment,
                target_offset: span.remote,
                length: span.len,
            })
            .collect();

        let api = self.shared.api;
        // SAFETY: a live engine; the batch is freed by the `Batch` that owns it.
        let id = unsafe { (api.allocate_batch)(self.shared.handle, requests.len()) };
        if id == capi::INVALID_BATCH {
            return Err(Error::TransferEngine {
                call: "allocating a batch",
                code: -1,
            });
        }
        let mut batch = Batch {
            shared: Arc::clone(&self.shared),
            id,
            ended: vec![false; requests.len()],
            submitted: false,
        };
        // SAFETY: a live engine, a batch it allocated for this many requests, and requests the
        // call copies; their addresses lie in registered storage the caller keeps mapped.
        check("submitting a transfer", unsafe {
            (api.submit)(
                self.shared.handle,
                id,
                requests.as_mut_ptr(),
                requests.len(),
            )
        })?;
        batch.submitted = true;
        Ok(batch)
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        // SAFETY: the engine this structure created; every registration and batch holds the
        // structure, so none outlives it.
        unsafe { (self.api.destroy)(self.handle) };
    }
}

/// One contiguous piece of a transfer: `len` bytes at `local`, an address of storage this
/// engine registered, and at `remote`, an address of storage the other engine registered, in
/// the process that registered it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Span {
    pub(crate) local: u64,
    pub(crate) remote: u64,
    pub(crate) len: u64,
}

/// A range registered with an engine. Dropping it unregisters the range; it keeps the engine
/// alive until then.
pub(crate) struct Registration {
    shared: Arc<Shared>,
    address: u64,
    len: usize,
}

impl Registration {
    /// The registered range's first byte.
    pub(crate) fn address(&self) -> u64 {
        self.address
    }
}

impl std::fmt::Debug for Registration {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Registration")
            .field("address", &self.address)
            .field("len", &self.len)
            .finish()
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        // SAFETY: a live engine and a range it registered.
        let code = unsafe {
            (self.shared.api.unregister)(self.shared.handle, self.address as *mut c_void)
        };
        if code != 0 {
            tracing::warn!(
                code,
                name = capi::code_name(code),
                "unregistering storage from the Transfer Engine failed"
            );
        }
    }
}

/// A posted batch of transfers, polled on the host until every transfer ended.
///
/// The engine frees a batch only once each of its transfers ended, completed or failed, so
/// dropping a batch waits for that: no transfer of it writes a destination after the drop.
pub(crate) struct Batch {
    shared: Arc<Shared>,
    id: BatchId,
    /// Per transfer of the batch, whether it ended.
    ended: Vec<bool>,
    /// Whether the batch's transfers were submitted: a batch whose submission failed holds
    /// none, and is freed at once.
    submitted: bool,
}

impl Batch {
    /// Whether every transfer of the batch completed.
    ///
    /// # Errors
    ///
    /// Fails when a transfer failed, timed out or was canceled (the other engine or the network
    /// failed); the destination bytes are then unspecified.
    pub(crate) fn poll(&mut self) -> Result<bool> {
        let api = self.shared.api;
        for (index, ended) in self.ended.iter_mut().enumerate() {
            if *ended {
                continue;
            }
            let mut status = capi::Status::default();
            // SAFETY: a live engine, a batch it allocated, and a transfer of the batch.
            check("reading a transfer's status", unsafe {
                (api.status)(self.shared.handle, self.id, index, &mut status)
            })?;
            match status.status {
                capi::STATUS_COMPLETED => *ended = true,
                capi::STATUS_WAITING | capi::STATUS_PENDING => return Ok(false),
                failed => {
                    *ended = true;
                    return Err(Error::TransferStatus {
                        status: failed,
                        transferred: status.transferred_bytes,
                    });
                }
            }
        }
        Ok(true)
    }

    /// Waits until every transfer of the batch completed, as the host observes it: until
    /// `expected`, when the batch's bytes are due at its path's rate, and then until its status
    /// reports it complete.
    ///
    /// The C API offers no blocking wait and no descriptor that becomes readable when a batch
    /// completes: its status call ([`Batch::poll`]) is the only observation, so the wait polls
    /// it, yielding the processor between calls, from the moment the bytes are due.
    ///
    /// # Errors
    ///
    /// As [`Batch::poll`].
    pub(crate) fn wait(mut self, expected: Instant) -> Result<()> {
        std::thread::sleep(expected.saturating_duration_since(Instant::now()));
        while !self.poll()? {
            std::thread::yield_now();
        }
        Ok(())
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        let api = self.shared.api;
        loop {
            // SAFETY: a live engine and a batch it allocated.
            let code = unsafe { (api.free_batch)(self.shared.handle, self.id) };
            if code != capi::ERR_BATCH_BUSY || !self.submitted {
                return;
            }
            // A transfer still runs: its status, read until it ends, is what marks it ended for
            // the engine's free. A failure ends the transfer as well.
            let _ = self.poll();
            std::thread::yield_now();
        }
    }
}
