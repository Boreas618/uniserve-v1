//! A NIXL agent with the UCX backend, and the READs it posts.

use std::collections::HashSet;
use std::ffi::{CStr, CString, c_void};
use std::ptr;
use std::sync::{Arc, Mutex, MutexGuard};

use ::storage::settings::Transports;

use super::capi::{self, Api, Handle};
use super::check;
use super::regions::{Access, Region, Registration, Span};
use super::{Error, Result};

/// The UCX backend's plugin name: the backend NIXL reaches other hosts' RDMA ports with.
const UCX: &CStr = c"UCX";

/// How an agent's UCX backend reaches other hosts: the transports it takes and the RDMA ports
/// it may open ([`crate::transfer::ports::usable`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Backend {
    pub(crate) transports: Transports,
    /// The usable RDMA ports, as UCX names them (`mlx5_0:1`); at least one.
    pub(crate) ports: Vec<String>,
}

impl Backend {
    /// The backend over this host's usable RDMA ports with `transports`.
    ///
    /// # Errors
    ///
    /// Fails when the kernel states no RDMA device or no usable port.
    pub(crate) fn of_host(transports: Transports) -> Result<Self> {
        let class = crate::transfer::ports::CLASS;
        let ports =
            crate::transfer::ports::usable(std::path::Path::new(class)).map_err(|error| {
                Error::NixlArgument(format!("reading the host's RDMA ports in {class}: {error}"))
            })?;
        if ports.is_empty() {
            return Err(Error::NixlArgument(
                "the host has no usable RDMA port (an active link that holds an address)"
                    .to_owned(),
            ));
        }
        Ok(Self { transports, ports })
    }

    /// The UCX settings the backend's context takes (NIXL's `engine_config`, comma-separated UCX
    /// settings without their `UCX_` prefix), over UCX's defaults:
    ///
    /// - `NET_DEVICES` names the usable ports alone, since UCX fails to start when it cannot open
    ///   a device it found.
    /// - `TLS=rc,cuda_copy` for [`Transports::Rdma`]: RDMA between processes and CUDA copies
    ///   within one.
    /// - `REG_NONBLOCK_MEM_TYPES=` pins host storage when it is registered. By default UCX
    ///   registers it for on-demand paging, whose page faults on the first READs hold a link
    ///   between hosts to a fraction of its rate, and which defers the refusal of a registration
    ///   the mapping cannot grant to a READ that never completes.
    /// - `RCACHE_ENABLE=n` keeps every registration out of UCX's registration cache. The cache
    ///   keeps a released registration, and learns that its storage was unmapped only through
    ///   UCX's storage hooks, which the Transfer Server's direct system calls bypass: a later
    ///   registration at the same address would then get the stale one, and READs into it would
    ///   land in storage the process no longer maps. Each [`Registration`] registers once and
    ///   deregisters before its storage is unmapped, so the cache saves nothing.
    fn engine_config(&self) -> Result<CString> {
        let mut settings = vec![
            "RCACHE_ENABLE=n".to_owned(),
            "REG_NONBLOCK_MEM_TYPES=".to_owned(),
            format!("NET_DEVICES={}", self.ports.join(",")),
        ];
        if self.transports == Transports::Rdma {
            settings.push("TLS=rc,cuda_copy".to_owned());
        }
        CString::new(settings.join(","))
            .map_err(|_| Error::NixlArgument("a port's name holds no NUL byte".to_owned()))
    }
}

/// One process's NIXL agent with a UCX backend.
///
/// The agent's metadata ([`Agent::metadata`]) carries its UCX worker
/// address and the remote keys of the storage registered when it is taken;
/// a peer that loaded it ([`Agent::load_peer`]) can READ that storage. The
/// metadata travels over the caller's own channel: the agent runs no
/// listener.
///
/// A progress thread drives the UCX worker, so connection setup and
/// transfers advance while the process makes no NIXL call, as a source
/// whose storage a peer reads does. Clones share the agent; every call on it
/// is serialized, since the agent is created without its own locking.
#[derive(Clone)]
pub(crate) struct Agent {
    shared: Arc<Shared>,
}

/// The agent's handles, shared by its clones, registrations and transfers,
/// which keep it alive until they drop.
pub(crate) struct Shared {
    api: &'static Api,
    name: String,
    state: Mutex<State>,
}

/// The handles every call takes the lock of.
struct State {
    agent: Handle,
    backend: Handle,
    /// The peers whose metadata is loaded, forgotten before the agent ends.
    peers: HashSet<String>,
}

// SAFETY: the handles are owned by this structure alone and used only under
// its mutex; NIXL's agent accepts calls from any thread one at a time.
unsafe impl Send for State {}

impl Agent {
    /// Creates an agent called `name`, which names it to its peers and must
    /// differ from theirs, with a UCX backend as `backend` states. Loads NIXL's C API library
    /// from the loader's default search path on first use.
    ///
    /// The agent takes NIXL's default configuration with a progress thread
    /// (`nixl_capi_create_agent`): no listener, since metadata travels over the caller's
    /// channel, and no synchronization of its own, since every call on it is serialized here.
    ///
    /// # Errors
    ///
    /// Fails when the library cannot be loaded, the UCX plugin is missing,
    /// or UCX cannot start.
    pub(crate) fn new(name: &str, backend: &Backend) -> Result<Self> {
        let api = super::library::api()?;
        let c_name = CString::new(name)
            .map_err(|_| Error::NixlArgument("an agent's name holds no NUL byte".to_owned()))?;
        let engine_config = backend.engine_config()?;

        let mut agent = ptr::null_mut();
        // SAFETY: the name outlives the call, which writes the new agent's handle on success.
        check("creating an agent", unsafe {
            (api.create_agent)(c_name.as_ptr(), &mut agent)
        })?;

        // The plugin's default parameters (one worker, and peer error handling, under which a
        // peer's failure fails the transfers with it rather than this process), with the
        // backend's UCX settings.
        let created = Self::create_backend(api, agent, &engine_config);
        let backend = match created {
            Ok(backend) => backend,
            Err(error) => {
                // SAFETY: the agent created above, which nothing else holds.
                unsafe { (api.destroy_agent)(agent) };
                return Err(error);
            }
        };

        Ok(Self {
            shared: Arc::new(Shared {
                api,
                name: name.to_owned(),
                state: Mutex::new(State {
                    agent,
                    backend,
                    peers: HashSet::new(),
                }),
            }),
        })
    }

    /// Creates `agent`'s UCX backend with the plugin's default parameters
    /// and the UCX settings `engine_config` ([`Backend::engine_config`]).
    fn create_backend(api: &Api, agent: Handle, engine_config: &CStr) -> Result<Handle> {
        let mut storages = ptr::null_mut();
        let mut parameters = ptr::null_mut();
        // SAFETY: a live agent; the call writes two lists this function
        // destroys below.
        check("reading the UCX plugin's parameters", unsafe {
            (api.get_plugin_params)(agent, UCX.as_ptr(), &mut storages, &mut parameters)
        })?;
        // SAFETY: the parameters read above and two NUL-terminated strings,
        // which the call copies.
        let configured = check("configuring the UCX backend", unsafe {
            (api.params_add)(
                parameters,
                c"engine_config".as_ptr(),
                engine_config.as_ptr(),
            )
        });
        let mut backend = ptr::null_mut();
        let created = configured.and_then(|()| {
            // SAFETY: a live agent and the parameters read above.
            check("creating the UCX backend", unsafe {
                (api.create_backend)(agent, UCX.as_ptr(), parameters, &mut backend)
            })
        });
        // SAFETY: the lists the plugin-parameter call created, used no more.
        unsafe {
            (api.destroy_mem_list)(storages);
            (api.destroy_params)(parameters);
        }
        created.map(|()| backend)
    }

    /// Registers `region` with the UCX backend for `access`, so that this
    /// agent's READs can write it (with [`Access::ReadWrite`]) and peers
    /// that load metadata taken from now on can read it.
    ///
    /// # Safety
    ///
    /// Until the registration drops, the region stays mapped at its address
    /// in this process, writable when `access` is [`Access::ReadWrite`],
    /// and no Rust reference to its bytes is held while a transfer may
    /// write them. Its bytes are read and written outside Rust: by READs
    /// this agent posts into it and by peers that loaded this agent's
    /// metadata.
    ///
    /// # Errors
    ///
    /// Fails for an empty region or when UCX cannot register the storage.
    pub(crate) unsafe fn register(&self, region: Region, access: Access) -> Result<Registration> {
        if region.len == 0 {
            return Err(Error::NixlArgument(
                "a registered region is not empty".to_owned(),
            ));
        }
        let metadata = access.metadata();
        self.shared
            .with_registration_list(region, metadata, |api, agent, list| {
                // SAFETY: a live agent and a list describing `region`, which the
                // caller keeps mapped while registered.
                check("registering storage", unsafe {
                    (api.register_mem)(agent, list, ptr::null_mut())
                })
            })?;
        Ok(Registration {
            agent: Arc::clone(&self.shared),
            region,
            access,
        })
    }

    /// This agent's metadata, for peers to load: its UCX worker address and
    /// the storage registered now. A region registered later is readable by
    /// a peer once it loads metadata taken after the registration.
    ///
    /// # Errors
    ///
    /// Fails when NIXL cannot serialize the metadata.
    pub(crate) fn metadata(&self) -> Result<Vec<u8>> {
        let state = self.shared.lock();
        let mut data: *mut c_void = ptr::null_mut();
        let mut len = 0;
        // SAFETY: a live agent; on success the call writes a buffer of
        // `len` bytes the caller owns and frees with `free`.
        check("exporting agent metadata", unsafe {
            (self.shared.api.get_local_md)(state.agent, &mut data, &mut len)
        })?;
        // SAFETY: the buffer written above, `len` bytes long, freed once
        // copied.
        let metadata = unsafe { std::slice::from_raw_parts(data.cast::<u8>(), len) }.to_vec();
        // SAFETY: the C API allocated the buffer with `malloc`.
        unsafe { capi::free(data) };
        Ok(metadata)
    }

    /// Loads a peer's [`Agent::metadata`] and returns the peer's name, which
    /// names it to [`Agent::read`]. Loading a later export of the same peer
    /// adds the storage it registered since.
    ///
    /// # Errors
    ///
    /// Fails for metadata NIXL cannot parse or whose backend this agent
    /// lacks.
    pub(crate) fn load_peer(&self, metadata: &[u8]) -> Result<String> {
        if metadata.is_empty() {
            return Err(Error::NixlArgument("peer metadata is not empty".to_owned()));
        }
        let mut state = self.shared.lock();
        let mut name: *mut std::ffi::c_char = ptr::null_mut();
        // SAFETY: a live agent and `metadata.len()` readable bytes; on
        // success the call writes a NUL-terminated name the caller frees.
        check("loading peer metadata", unsafe {
            (self.shared.api.load_remote_md)(
                state.agent,
                metadata.as_ptr().cast(),
                metadata.len(),
                &mut name,
            )
        })?;
        // SAFETY: the name written above, NUL-terminated, freed once copied.
        let peer = unsafe { CStr::from_ptr(name) }
            .to_string_lossy()
            .into_owned();
        // SAFETY: the C API allocated the name with `strdup`.
        unsafe { capi::free(name.cast()) };
        state.peers.insert(peer.clone());
        Ok(peer)
    }

    /// Forgets `peer`'s metadata and its connection: a peer whose process
    /// ended is not read again, and a later agent of that name is loaded
    /// afresh.
    ///
    /// # Errors
    ///
    /// Fails when no metadata of `peer` is loaded.
    pub(crate) fn forget_peer(&self, peer: &str) -> Result<()> {
        let c_peer = CString::new(peer)
            .map_err(|_| Error::NixlArgument("a peer's name holds no NUL byte".to_owned()))?;
        let mut state = self.shared.lock();
        // SAFETY: a live agent and a NUL-terminated name.
        check("forgetting a peer", unsafe {
            (self.shared.api.invalidate_remote_md)(state.agent, c_peer.as_ptr())
        })?;
        state.peers.remove(peer);
        Ok(())
    }

    /// Posts one READ of every span's source, in storage `peer` registered,
    /// into its destination, in storage this agent registered. All sources
    /// lie in one kind of storage (host or device), and so do all
    /// destinations; NIXL keeps one kind per side of a transfer.
    ///
    /// The READ is one-sided: the peer's process takes no part beyond its
    /// progress thread. The caller keeps every source unchanged and every
    /// destination unread until the transfer completes. The transfer does
    /// not pass through a CUDA stream: its completion, observed on the host
    /// ([`Transfer::poll`]), is the only ordering it offers.
    ///
    /// # Errors
    ///
    /// Fails for an empty span list, spans of unequal or zero length or of
    /// mixed memory kinds, and spans NIXL does not find within storage this
    /// agent or `peer` registered under the span's key.
    pub(crate) fn read(&self, peer: &str, spans: &[Span]) -> Result<Transfer> {
        self.post(peer, spans, Op::Read)
    }

    /// Posts one WRITE of every span's source, in storage this agent registered, into its
    /// destination, in storage `peer` registered: [`Agent::read`] the other way, which a
    /// handoff's push takes. The caller keeps every source unchanged until the transfer
    /// completes, and the peer's process reads no destination before then.
    ///
    /// # Errors
    ///
    /// As [`Agent::read`].
    pub(crate) fn write(&self, peer: &str, spans: &[Span]) -> Result<Transfer> {
        self.post(peer, spans, Op::Write)
    }

    /// Posts one transfer of `spans` with `peer`: a READ of its storage into this agent's, or
    /// a WRITE of this agent's into its.
    fn post(&self, peer: &str, spans: &[Span], op: Op) -> Result<Transfer> {
        let Some(first) = spans.first() else {
            return Err(Error::NixlArgument(
                "a READ has at least one span".to_owned(),
            ));
        };
        let source_kind = first.source.segment;
        let destination_kind = first.destination.segment;
        for span in spans {
            let (source, destination) = (span.source, span.destination);
            if source.len != destination.len || source.len == 0 {
                return Err(Error::NixlArgument(format!(
                    "a span reads {} bytes into {}: the lengths are equal and nonzero",
                    source.len, destination.len
                )));
            }
            if source.segment != source_kind || destination.segment != destination_kind {
                return Err(Error::NixlArgument(
                    "the spans of one READ read one kind of storage into one kind".to_owned(),
                ));
            }
        }
        let c_peer = CString::new(peer)
            .map_err(|_| Error::NixlArgument("a peer's name holds no NUL byte".to_owned()))?;

        // NIXL's local list is this agent's storage, its remote list the peer's: a READ's
        // destinations and sources, a WRITE's sources and destinations.
        let api = self.shared.api;
        let destinations = Descriptors::new(api, destination_kind.kind())?;
        let sources = Descriptors::new(api, source_kind.kind())?;
        for span in spans {
            destinations.add(span.destination)?;
            sources.add(span.source)?;
        }
        let (local, remote, code, what) = match op {
            Op::Read => (&destinations, &sources, capi::XFER_OP_READ, "a READ"),
            Op::Write => (&sources, &destinations, capi::XFER_OP_WRITE, "a WRITE"),
        };

        let state = self.shared.lock();
        let mut request = ptr::null_mut();
        // SAFETY: a live agent, two live lists and a NUL-terminated name;
        // the request copies the descriptors, so the lists may go after.
        check(what, unsafe {
            (api.create_xfer_req)(
                state.agent,
                code,
                local.list,
                remote.list,
                c_peer.as_ptr(),
                &mut request,
                ptr::null_mut(),
            )
        })?;
        // From here the request is released when the transfer drops.
        let mut transfer = Transfer {
            shared: Arc::clone(&self.shared),
            request,
            complete: false,
        };
        // SAFETY: a live agent and the request created above.
        let posted = unsafe { (api.post_xfer_req)(state.agent, request, ptr::null_mut()) };
        drop(state);
        match posted {
            capi::SUCCESS => transfer.complete = true,
            capi::IN_PROG => {}
            status => {
                return Err(Error::Nixl {
                    call: match op {
                        Op::Read => "posting a READ",
                        Op::Write => "posting a WRITE",
                    },
                    status,
                });
            }
        }
        Ok(transfer)
    }
}

/// Which way a transfer with a peer goes.
#[derive(Clone, Copy)]
enum Op {
    Read,
    Write,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Agent")
            .field("name", &self.shared.name)
            .finish()
    }
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        ::service::lock(&self.state)
    }

    /// Runs `call` with the agent's handle and a registration list holding
    /// one descriptor of `region` with `metadata`, under the agent's lock.
    fn with_registration_list(
        &self,
        region: Region,
        metadata: &[u8],
        call: impl FnOnce(&Api, Handle, Handle) -> Result<()>,
    ) -> Result<()> {
        let api = self.api;
        let mut list = ptr::null_mut();
        // SAFETY: the call writes a new list this function destroys below.
        check("creating a registration list", unsafe {
            (api.create_reg_dlist)(region.segment.kind(), &mut list)
        })?;
        // SAFETY: the list created above; a device address is a unified
        // address, which NIXL and UCX take as a pointer value and never
        // dereference on the host; the metadata is copied by the call.
        let result = check("describing storage", unsafe {
            (api.reg_dlist_add_desc)(
                list,
                region.address as usize,
                region.len,
                region.key,
                metadata.as_ptr().cast(),
                metadata.len(),
            )
        })
        .and_then(|()| {
            let state = self.lock();
            call(api, state.agent, list)
        });
        // SAFETY: the list created above, used no more.
        unsafe { (api.destroy_reg_dlist)(list) };
        result
    }

    /// Deregisters `region`, which a [`Registration`] registered. A failure
    /// leaves the registration to the agent's end.
    pub(crate) fn deregister(&self, region: Region) {
        let _ = self.with_registration_list(region, &[], |api, agent, list| {
            // SAFETY: a live agent and a list describing a registered region.
            check("deregistering storage", unsafe {
                (api.deregister_mem)(agent, list, ptr::null_mut())
            })
        });
    }
}

impl Drop for Shared {
    #[expect(
        clippy::expect_used,
        reason = "a poisoned lock is unreachable under `panic = \"abort\"`"
    )]
    fn drop(&mut self) {
        let api = self.api;
        let state = self
            .state
            .get_mut()
            .expect("a thread panicked while it held the agent's state");
        for peer in state.peers.drain() {
            if let Ok(c_peer) = CString::new(peer) {
                // SAFETY: a live agent and a NUL-terminated name.
                unsafe { (api.invalidate_remote_md)(state.agent, c_peer.as_ptr()) };
            }
        }
        // SAFETY: the backend and the agent this structure created; no
        // registration or transfer holds them any more, since each keeps
        // this structure alive.
        unsafe {
            (api.destroy_backend)(state.backend);
            (api.destroy_agent)(state.agent);
        }
    }
}

/// A descriptor list of one kind of storage for a READ, destroyed on drop.
struct Descriptors {
    api: &'static Api,
    list: Handle,
}

impl Descriptors {
    fn new(api: &'static Api, kind: capi::MemType) -> Result<Self> {
        let mut list = ptr::null_mut();
        // SAFETY: the call writes a new list this structure destroys.
        check("creating a descriptor list", unsafe {
            (api.create_xfer_dlist)(kind, &mut list)
        })?;
        Ok(Self { api, list })
    }

    fn add(&self, region: Region) -> Result<()> {
        // SAFETY: a live list; a device address is a unified address, which
        // NIXL and UCX take as a pointer value.
        check("describing a span", unsafe {
            (self.api.xfer_dlist_add_desc)(
                self.list,
                region.address as usize,
                region.len,
                region.key,
            )
        })
    }
}

impl Drop for Descriptors {
    fn drop(&mut self) {
        // SAFETY: the list this structure created, used no more.
        unsafe { (self.api.destroy_xfer_dlist)(self.list) };
    }
}

/// A posted transfer, polled on the host until it completes.
///
/// Dropping a transfer that has not completed asks the backend to abort it,
/// which an RDMA operation already posted may outlive: the transfer can
/// still write its destinations after the drop. Whoever drops an incomplete
/// transfer keeps the destinations mapped and treats their bytes as
/// unspecified.
pub(crate) struct Transfer {
    shared: Arc<Shared>,
    request: Handle,
    complete: bool,
}

// SAFETY: the request is used only under the agent's lock.
unsafe impl Send for Transfer {}

impl Transfer {
    /// Whether every byte of the transfer has arrived. Each call drives the
    /// backend's progress.
    ///
    /// # Errors
    ///
    /// Fails when the transfer failed, for example because the peer
    /// disconnected; the destination bytes are then unspecified.
    pub(crate) fn poll(&mut self) -> Result<bool> {
        if !self.complete {
            let state = self.shared.lock();
            // SAFETY: a live agent and the request it created.
            let status = unsafe { (self.shared.api.get_xfer_status)(state.agent, self.request) };
            match status {
                capi::SUCCESS => self.complete = true,
                capi::IN_PROG => {}
                status => {
                    return Err(Error::Nixl {
                        call: "reading a transfer's status",
                        status,
                    });
                }
            }
        }
        Ok(self.complete)
    }

    /// Waits until every byte of the transfer has arrived, as the host observes it: until
    /// `expected`, when the transfer's bytes are due at its path's rate, and then until its
    /// status reports it complete.
    ///
    /// NIXL 1.5.1's C API offers no blocking wait for a transfer and no descriptor that becomes
    /// readable when one completes: its status call ([`Transfer::poll`]) is the only observation,
    /// so the wait polls it, yielding the processor between calls, from the moment the bytes are
    /// due.
    ///
    /// # Errors
    ///
    /// As [`Transfer::poll`].
    pub(crate) fn wait(mut self, expected: std::time::Instant) -> Result<()> {
        std::thread::sleep(expected.saturating_duration_since(std::time::Instant::now()));
        while !self.poll()? {
            std::thread::yield_now();
        }
        Ok(())
    }
}

impl Drop for Transfer {
    fn drop(&mut self) {
        let state = self.shared.lock();
        // SAFETY: a live agent and the request it created; releasing an
        // incomplete request aborts it, and destroying it frees the handle.
        unsafe {
            (self.shared.api.release_xfer_req)(state.agent, self.request);
            (self.shared.api.destroy_xfer_req)(self.request);
        }
    }
}
