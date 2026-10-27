//! A NIXL agent with the UCX backend, and the READs it posts.

use nixl_sys::{AgentConfig, ThreadSync, XferDescList, XferOp, XferRequest};

use crate::library;
use crate::memory::{Described, Region, Registration, Span};
use crate::{Error, Result};

/// The UCX backend's plugin name.
const UCX: &str = "UCX";

/// One process's NIXL agent with a UCX backend.
///
/// The agent's metadata ([`Agent::metadata`]) carries its UCX worker
/// address and the remote keys of the memory registered when it is taken;
/// a peer that loaded it ([`Agent::load_peer`]) can READ that memory. The
/// metadata travels over the caller's own channel: the agent runs no
/// listener.
///
/// A progress thread drives the UCX worker, so connection setup and
/// transfers advance while the process makes no NIXL call, as a source
/// whose memory a peer reads does.
#[derive(Clone, Debug)]
pub struct Agent {
    agent: nixl_sys::Agent,
}

impl Agent {
    /// Creates an agent called `name`, which names it to its peers and must
    /// differ from theirs, with a UCX backend as NIXL configures one by
    /// default. The `UCX_*` environment variables apply: `UCX_NET_DEVICES`
    /// limits UCX to the host's usable RDMA ports, since UCX fails to start
    /// when it cannot open a device it found. Loads NIXL's C API library
    /// from the loader's search path unless [`crate::load_library`] loaded
    /// it.
    ///
    /// # Errors
    ///
    /// Fails when the library cannot be loaded, the UCX plugin is missing,
    /// or UCX cannot start.
    pub fn new(name: &str) -> Result<Self> {
        library::ensure_loaded()?;

        let config = AgentConfig {
            enable_prog_thread: true,
            enable_listen_thread: false,
            // NIXL's own default: the agent locks its state against the
            // progress thread and concurrent callers.
            thread_sync: ThreadSync::Default,
            ..AgentConfig::default()
        };
        let agent = nixl_sys::Agent::new_configured(name, &config)
            .map_err(Error::nixl("creating an agent"))?;

        // The plugin's default parameters: every device UCX finds, one
        // worker, and peer error handling, under which a peer's failure
        // fails the transfers with it rather than this process.
        let (_, parameters) = agent
            .get_plugin_params(UCX)
            .map_err(Error::nixl("reading the UCX plugin's parameters"))?;
        agent
            .create_backend(UCX, &parameters)
            .map_err(Error::nixl("creating the UCX backend"))?;

        Ok(Self { agent })
    }

    pub fn name(&self) -> String {
        self.agent.name()
    }

    /// Registers `region` with the UCX backend, so that this agent's READs
    /// can write it and peers that load metadata taken from now on can read
    /// it.
    ///
    /// # Safety
    ///
    /// Until the registration drops, the region stays mapped at its address
    /// in this process, and no Rust reference to its bytes is held while a
    /// transfer may write them. Its bytes are read and written outside Rust:
    /// by READs this agent posts into it and by peers that loaded this
    /// agent's metadata.
    ///
    /// # Errors
    ///
    /// Fails for an empty region or when UCX cannot register the memory.
    pub unsafe fn register(&self, region: Region) -> Result<Registration> {
        if region.len == 0 {
            return Err(Error::Invalid(
                "a registered region is not empty".to_owned(),
            ));
        }
        let handle = self
            .agent
            .register_memory(&Described(region), None)
            .map_err(Error::nixl("registering memory"))?;
        Ok(Registration::new(region, handle))
    }

    /// This agent's metadata, for peers to load: its UCX worker address and
    /// the memory registered now. A region registered later is readable by
    /// a peer once it loads metadata taken after the registration.
    ///
    /// # Errors
    ///
    /// Fails when NIXL cannot serialize the metadata.
    pub fn metadata(&self) -> Result<Vec<u8>> {
        self.agent
            .get_local_md()
            .map_err(Error::nixl("exporting agent metadata"))
    }

    /// Loads a peer's [`Agent::metadata`] and returns the peer's name, which
    /// names it to [`Agent::read`]. Loading a later export of the same peer
    /// adds the memory it registered since.
    ///
    /// # Errors
    ///
    /// Fails for metadata NIXL cannot parse or whose backend this agent
    /// lacks.
    pub fn load_peer(&self, metadata: &[u8]) -> Result<String> {
        self.agent
            .load_remote_md(metadata)
            .map_err(Error::nixl("loading peer metadata"))
    }

    /// Forgets `peer`'s metadata and its connection: a peer whose process
    /// ended is not read again, and a later agent of that name is loaded
    /// afresh.
    ///
    /// # Errors
    ///
    /// Fails when no metadata of `peer` is loaded.
    pub fn forget_peer(&self, peer: &str) -> Result<()> {
        self.agent
            .invalidate_remote_md(peer)
            .map_err(Error::nixl("forgetting a peer"))
    }

    /// Posts one READ of every span's source, in memory `peer` registered,
    /// into its destination, in memory this agent registered. All sources
    /// lie in one kind of memory (host or device), and so do all
    /// destinations; NIXL keeps one kind per side of a transfer.
    ///
    /// The READ is one-sided: the peer's process takes no part beyond its
    /// progress thread. The caller keeps every source unchanged and every
    /// destination unread until the transfer completes. The transfer does
    /// not pass through a CUDA stream: its completion, observed on the host
    /// ([`Transfer::wait`]), is the only ordering it offers.
    ///
    /// # Errors
    ///
    /// Fails for an empty span list, spans of unequal or zero length or of
    /// mixed memory kinds, and spans NIXL does not find within memory this
    /// agent or `peer` registered under the span's key.
    pub fn read(&self, peer: &str, spans: &[Span]) -> Result<Transfer> {
        let Some(first) = spans.first() else {
            return Err(Error::Invalid("a READ has at least one span".to_owned()));
        };
        let source_kind = first.source.memory.kind();
        let destination_kind = first.destination.memory.kind();

        let mut sources =
            XferDescList::new(source_kind).map_err(Error::nixl("creating a descriptor list"))?;
        let mut destinations = XferDescList::new(destination_kind)
            .map_err(Error::nixl("creating a descriptor list"))?;
        for span in spans {
            let (source, destination) = (span.source, span.destination);
            if source.len != destination.len || source.len == 0 {
                return Err(Error::Invalid(format!(
                    "a span reads {} bytes into {}: the lengths are equal and nonzero",
                    source.len, destination.len
                )));
            }
            if source.memory.kind() != source_kind || destination.memory.kind() != destination_kind
            {
                return Err(Error::Invalid(
                    "the spans of one READ read one kind of memory into one kind".to_owned(),
                ));
            }
            sources.add_desc(source.address as usize, source.len, source.key);
            destinations.add_desc(
                destination.address as usize,
                destination.len,
                destination.key,
            );
        }

        // NIXL's local list is this agent's destinations; its remote list,
        // the peer's sources.
        let request = self
            .agent
            .create_xfer_req(XferOp::Read, &destinations, &sources, peer, None)
            .map_err(Error::nixl("creating a READ"))?;
        let in_progress = self
            .agent
            .post_xfer_req(&request, None)
            .map_err(Error::nixl("posting a READ"))?;

        Ok(Transfer {
            agent: self.agent.clone(),
            request,
            complete: !in_progress,
        })
    }
}

/// A posted transfer, polled on the host until it completes.
///
/// Dropping a transfer that has not completed asks the backend to abort it,
/// which an RDMA operation already posted may outlive: the transfer can
/// still write its destinations after the drop. Whoever drops an incomplete
/// transfer keeps the destinations mapped and treats their bytes as
/// unspecified.
pub struct Transfer {
    agent: nixl_sys::Agent,
    request: XferRequest,
    complete: bool,
}

impl Transfer {
    /// Whether every byte of the transfer has arrived. Each call drives the
    /// backend's progress.
    ///
    /// # Errors
    ///
    /// Fails when the transfer failed, for example because the peer
    /// disconnected; the destination bytes are then unspecified.
    pub fn poll(&mut self) -> Result<bool> {
        if !self.complete {
            self.complete = self
                .agent
                .get_xfer_status(&self.request)
                .map_err(Error::nixl("reading a transfer's status"))?
                .is_success();
        }
        Ok(self.complete)
    }

    /// Spins on the calling thread until the transfer completes; the
    /// destination bytes are then readable, by the host or by device work
    /// launched after the return.
    ///
    /// # Errors
    ///
    /// Fails as [`Transfer::poll`] does.
    pub fn wait(mut self) -> Result<()> {
        while !self.poll()? {
            std::thread::yield_now();
        }
        Ok(())
    }
}
