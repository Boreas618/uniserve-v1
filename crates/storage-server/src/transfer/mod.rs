//! The Transfer Server: the Storage Server's child process that copies
//! bytes between the memory the server holds.
//!
//! The server runs it as `uniserve storage transfer` and starts it again
//! when it ends, so a fault in a copy or a CUDA context ends this process
//! and never the server, which creates no CUDA context of its own. It knows
//! the allocations the server gives it (retained storages: CUDA VMM
//! allocations through their POSIX handles and host memfds; and page
//! pools) and copies between them: it knows nothing of models,
//! configurations, requests or byte encodings.
//!
//! When it starts it measures the bandwidth of every link (each GPU's own
//! memory, each pair of GPUs, each GPU to and from host memory, and host
//! memory to itself) and the device memory its context takes on each GPU
//! ([`devices`]), connects as [`Role::Transfer`] and reports both
//! ([`Request::Measured`]). It then runs the server's tasks
//! ([`Event::Transfer`]): every move goes to the lane of its link and of
//! what runs its copies ([`lanes`]), by path:
//!
//! | Source → destination | Runs on |
//! | --- | --- |
//! | one GPU's memory → the same GPU | that GPU's copy engines |
//! | GPU → another GPU of the host | the destination's copy engines, reading over the GPUs' link |
//! | host → GPU, GPU → host | the GPU's copy engines, with the host memory registered |
//! | another host's GPU of the NVLink fabric ↔ this host | the local GPU's copy engines (the first GPU's for host memory), with the fabric handle imported |
//! | host → host | the CPU |
//! | another host's GPU or host memory → this host | NIXL READs this host's agent posts, through the network adapters ([`network`]) |
//!
//! No compute kernel runs: copies use copy engines or the network adapters
//! only, so the world executing beside them keeps its compute units. A
//! same-GPU read-only mapping instead of a copy is the worker's own (it
//! maps the storage from the server). A link from another host has a lane
//! of its own, measured by its first copies long enough to time and then
//! reported ([`Request::Measured`]), with the chunks, priorities and
//! budget of every lane.
//!
//! The same agent serves other hosts' READs of this host's memory: the
//! server has it register the allocations a peer asks to expose
//! ([`Event::Register`]), and it reports the ranges and its metadata
//! ([`Request::Registered`]). What waits on another host or on such a
//! registration runs on a helper thread, so the main thread keeps taking
//! the server's events and the lanes' reports meanwhile.
//!
//! It reports each task's progress and end ([`Request::Copied`]) only
//! after the host saw the copies complete (an event its lane queried, or a
//! READ it polled), so no CUDA event crosses a process. It ends when the
//! server closes its connection.

mod devices;
mod lanes;
mod memory;
mod network;

use std::collections::HashMap;
use std::io;
use std::os::fd::OwnedFd;
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};
use std::time::Duration;

use rustix::event::{PollFd, PollFlags, poll};
use storage::client::Client;
use storage::protocol::{
    Allocation, Event, Link, Memory, Move, Priority, Registered, Reply, Request, Role, TransferEnd,
};

use devices::Gpus;
use lanes::{End, LaneKey, Lanes, Piece, Runner};
use memory::{Fabric, Mapped};
use network::{Inbound, Network};

/// How the Transfer Server copies.
#[derive(Debug, Clone)]
pub struct Options {
    /// How long one chunk of a copy lasts at its link's bandwidth: far
    /// below one scheduler step of the world that executes beside the
    /// copies, so a copy engine turns to that world's copies, and to
    /// higher-priority tasks, between chunks.
    pub chunk: Duration,
    /// NIXL's C API library (`libnixl_capi.so`) of the release `nixl-sys`
    /// pins, which the agent of the path between hosts loads; the loader's
    /// search path finds it when `None`.
    pub nixl_library: Option<PathBuf>,
}

/// Why the Transfer Server stopped.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Cuda(#[from] cuda::Error),
    #[error(transparent)]
    Storage(#[from] storage::client::Error),
    #[error(transparent)]
    Io(#[from] io::Error),
}

/// Measure this host's links, connect to the Storage Server on `socket`
/// and run its tasks until it closes the connection.
///
/// # Errors
///
/// Fails when a GPU cannot be measured (it has no memory for the context
/// or the measurement) or the server cannot be reached.
pub fn run(socket: &Path, options: &Options) -> Result<(), Error> {
    let gpus = Gpus::discover()?;
    let (links, contexts) = devices::measure(&gpus)?;
    let bandwidths = links
        .iter()
        .map(|measured| (measured.link.clone(), measured.bytes_per_second as f64))
        .collect();

    let mut client = Client::connect(socket, Role::Transfer)?;
    let (wake_reader, wake_writer) = rustix::pipe::pipe_with(
        rustix::pipe::PipeFlags::CLOEXEC | rustix::pipe::PipeFlags::NONBLOCK,
    )
    .map_err(io::Error::from)?;
    let network = Arc::new(Network::new(options.nixl_library.clone()));
    let (sender, finished) = mpsc::channel();
    let mut server = Served {
        gpus,
        allocations: HashMap::new(),
        helpers: Helpers {
            sender,
            wake: Arc::new(wake_writer.try_clone()?),
        },
        lanes: Lanes::new(wake_writer, options.chunk, bandwidths, Arc::clone(&network)),
        network,
        finished,
        exposing: HashMap::new(),
        replies: Vec::new(),
    };
    server.send(&mut client, &Request::Measured { links, contexts })?;
    tracing::info!("the Transfer Server serves");

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
            // the Transfer Server.
            Err(storage::client::Error::Io(error))
                if error.kind() == io::ErrorKind::UnexpectedEof =>
            {
                tracing::info!("the storage server closed the connection");
                return Ok(());
            }
            Err(error) => return Err(error.into()),
        }
        while let Some((event, descriptors)) = client.next_event() {
            server.handle(event, descriptors);
        }
        while let Ok(done) = server.finished.try_recv() {
            if let Some(report) = server.finish(done) {
                server.send(&mut client, &report)?;
            }
        }
        for (task, bytes, end) in server.lanes.reports() {
            server.send(&mut client, &Request::Copied { task, bytes, end })?;
        }
        let links = server.lanes.measured();
        if !links.is_empty() {
            let measured = Request::Measured {
                links,
                contexts: Vec::new(),
            };
            server.send(&mut client, &measured)?;
        }
        server.collect(&mut client);
    }
}

/// What a helper thread finished, for the main thread.
enum Done {
    /// The pieces of task `task`, its moves from other hosts among them
    /// once those hosts exposed their sources, or why it cannot run.
    Sources {
        task: u64,
        priority: Priority,
        pieces: Result<Vec<(LaneKey, Piece)>, String>,
    },
    /// The registrations of the server's `request`: the agent's metadata
    /// and each allocation's range, or why they failed.
    Registered {
        request: u64,
        registered: Result<(Vec<u8>, Vec<Registered>), String>,
    },
}

/// Runs work that waits on another host, or on registering memory, off
/// the main thread, and wakes the main thread with its outcome.
struct Helpers {
    sender: mpsc::Sender<Done>,
    /// The write end of the main thread's wake pipe.
    wake: Arc<OwnedFd>,
}

impl Helpers {
    fn spawn(&self, work: impl FnOnce() -> Done + Send + 'static) {
        let sender = self.sender.clone();
        let wake = Arc::clone(&self.wake);
        let spawned = std::thread::Builder::new()
            .name("uniserve-transfer-helper".into())
            .spawn(move || {
                // The main thread ends only with the process, so its
                // receiver outlives every helper.
                let _ = sender.send(work());
                let _ = rustix::io::write(&*wake, b"x");
            });
        if let Err(error) = spawned {
            lanes::fatal("starting a helper thread", &error);
        }
    }
}

/// The Transfer Server's state on its main thread.
struct Served {
    gpus: Gpus,
    /// The allocations the server gave, until it takes them back.
    allocations: HashMap<Allocation, Arc<Mapped>>,
    lanes: Lanes,
    /// The NIXL agent of the path between hosts, which the lanes and the
    /// helpers share.
    network: Arc<Network>,
    helpers: Helpers,
    /// What the helpers finished.
    finished: mpsc::Receiver<Done>,
    /// The tasks whose sources other hosts are exposing, and whether the
    /// server cancelled each meanwhile.
    exposing: HashMap<u64, bool>,
    /// Requests sent and not answered yet.
    replies: Vec<u64>,
}

impl Served {
    fn send(&mut self, client: &mut Client, request: &Request) -> Result<(), Error> {
        self.replies.push(client.submit(request, &[])?);
        Ok(())
    }

    /// Take the server's answers; an error names a task it no longer holds,
    /// which changes nothing here.
    fn collect(&mut self, client: &mut Client) {
        self.replies.retain(|id| match client.take(*id) {
            Some(answer) => {
                if let Reply::Error { message, .. } = answer.reply {
                    tracing::warn!(message, "the storage server refused a report");
                }
                false
            }
            None => true,
        });
    }

    /// Take one of the server's events.
    fn handle(&mut self, event: Event, descriptors: Vec<OwnedFd>) {
        match event {
            Event::Allocation {
                allocation,
                memory,
                bytes,
                writable,
            } => {
                let Some(descriptor) = descriptors.into_iter().next() else {
                    tracing::warn!(?allocation, "an allocation came without its descriptor");
                    return;
                };
                let mapped = Mapped::new(memory, bytes, writable, descriptor, &self.gpus);
                self.allocations.insert(allocation, Arc::new(mapped));
            }
            Event::AllocationReleased { allocation } => {
                // A task that copies it holds it until its copies complete.
                // Its NIXL registration ends with its mapping, so another
                // host's READ of it fails from then on.
                self.allocations.remove(&allocation);
            }
            Event::Transfer {
                task,
                priority,
                moves,
            } => self.transfer(task, priority, &moves),
            Event::Cancel { task } => match self.exposing.get_mut(&task) {
                // Its pieces are not queued yet: it ends, cancelled, once
                // the other hosts answered.
                Some(cancelled) => *cancelled = true,
                None => self.lanes.cancel(task),
            },
            Event::Throttle { links, throttle } => {
                tracing::info!(links = links.len(), ?throttle, "changing link budgets");
                self.lanes.throttle(&links, throttle);
            }
            Event::Register {
                request,
                allocations,
            } => self.register(request, &allocations),
            Event::TransferProgress { .. }
            | Event::Transferred { .. }
            | Event::Prefetched { .. }
            | Event::Pool { .. }
            | Event::PoolReleased { .. }
            | Event::Exists { .. }
            | Event::Upload { .. }
            | Event::Download { .. } => {}
        }
    }

    /// Queue task `task`'s moves on their lanes, or fail it whole when one
    /// cannot run here. A task with moves from other hosts queues once a
    /// helper had those hosts expose their sources.
    fn transfer(&mut self, task: u64, priority: Priority, moves: &[Move]) {
        let mut pieces = Vec::with_capacity(moves.len());
        let mut inbound = Vec::new();
        // Another host's allocation of the fabric is imported once per task
        // and GPU, whatever number of its moves read or write it.
        let mut imports = HashMap::new();
        for copy in moves {
            let taken = match &copy.source {
                Allocation::Remote {
                    endpoint,
                    deployment,
                    region,
                } => self.given(&copy.destination).map(|destination| {
                    inbound.push(Inbound {
                        endpoint: endpoint.clone(),
                        deployment: deployment.clone(),
                        region: (**region).clone(),
                        source_offset: copy.source_offset,
                        destination,
                        destination_offset: copy.destination_offset,
                        bytes: copy.bytes,
                    });
                }),
                _ => self
                    .piece(task, copy, &mut imports)
                    .map(|piece| pieces.push(piece)),
            };
            if let Err(message) = taken {
                self.lanes.end(task, TransferEnd::Failed { message });
                return;
            }
        }
        if inbound.is_empty() {
            self.lanes.submit(task, priority, pieces);
            return;
        }

        self.exposing.insert(task, false);
        let network = Arc::clone(&self.network);
        self.helpers.spawn(move || {
            let pieces = network.pieces(task, inbound).map(|remote| {
                pieces.extend(remote);
                pieces
            });
            Done::Sources {
                task,
                priority,
                pieces,
            }
        });
    }

    /// Register `allocations` with the NIXL agent for another host's READs,
    /// on a helper: registering pins the memory, which takes a while for a
    /// large page pool.
    fn register(&mut self, request: u64, allocations: &[Allocation]) {
        let mapped: Result<Vec<Arc<Mapped>>, String> = allocations
            .iter()
            .map(|allocation| self.given(allocation))
            .collect();
        let network = Arc::clone(&self.network);
        self.helpers.spawn(move || Done::Registered {
            request,
            registered: mapped.and_then(|mapped| network::register(&network, &mapped)),
        });
    }

    /// Take what a helper finished; returns the report it makes to the
    /// server.
    fn finish(&mut self, done: Done) -> Option<Request> {
        match done {
            Done::Sources {
                task,
                priority,
                pieces,
            } => {
                let cancelled = self.exposing.remove(&task).unwrap_or(false);
                match pieces {
                    _ if cancelled => self.lanes.end(task, TransferEnd::Cancelled),
                    Ok(pieces) => self.lanes.submit(task, priority, pieces),
                    Err(message) => {
                        tracing::warn!(task, message, "a task's sources on other hosts failed");
                        self.lanes.end(task, TransferEnd::Failed { message });
                    }
                }
                None
            }
            Done::Registered {
                request,
                registered,
            } => Some(match registered {
                Ok((metadata, ranges)) => Request::Registered {
                    request,
                    metadata,
                    ranges,
                    failure: None,
                },
                Err(message) => {
                    tracing::warn!(message, "registering memory for another host failed");
                    Request::Registered {
                        request,
                        metadata: Vec::new(),
                        ranges: Vec::new(),
                        failure: Some(message),
                    }
                }
            }),
        }
    }

    /// The lane and the piece of one move between allocations of this host
    /// and of the fabric; `imports` holds the task's imports of other
    /// hosts' allocations.
    fn piece(
        &self,
        task: u64,
        copy: &Move,
        imports: &mut Imports,
    ) -> Result<(LaneKey, Piece), String> {
        let source = self.side(&copy.source)?;
        let destination = self.side(&copy.destination)?;
        let link = Link {
            source: source.memory(),
            destination: destination.memory(),
        };
        let runner = self.runner(&source, &destination)?;
        let piece = Piece {
            task,
            source: source.end(runner, imports)?,
            source_offset: copy.source_offset,
            destination: destination.end(runner, imports)?,
            destination_offset: copy.destination_offset,
            bytes: copy.bytes,
        };
        Ok((LaneKey { link, runner }, piece))
    }

    /// An allocation the server gave.
    fn given(&self, allocation: &Allocation) -> Result<Arc<Mapped>, String> {
        match self.side(allocation)? {
            Side::Given(mapped) => Ok(mapped),
            Side::Fabric { .. } => {
                Err("another host's memory of the fabric is neither READ into nor exposed".into())
            }
        }
    }

    fn side(&self, allocation: &Allocation) -> Result<Side, String> {
        match allocation {
            Allocation::Storage { .. } | Allocation::Pool { .. } => self
                .allocations
                .get(allocation)
                .map(|mapped| Side::Given(Arc::clone(mapped)))
                .ok_or_else(|| format!("{allocation:?} was not given to the Transfer Server")),
            Allocation::Fabric { handle, gpu, bytes } => Ok(Side::Fabric {
                handle: handle
                    .as_slice()
                    .try_into()
                    .map_err(|_| "a fabric handle has 64 bytes".to_owned())?,
                gpu: *gpu,
                bytes: *bytes,
            }),
            Allocation::Remote { endpoint, .. } => Err(format!(
                "memory of {endpoint} is read by a READ into this host, never copied otherwise"
            )),
        }
    }

    /// What runs a copy from `source` to `destination`: the copy engines of
    /// the destination's GPU, else of the source's, else (between host
    /// memory and another host's GPU) of this host's first GPU; the CPU
    /// between host buffers. The runner's copy engines must address both.
    fn runner(&self, source: &Side, destination: &Side) -> Result<Runner, String> {
        let local = |side: &Side| match side {
            Side::Given(mapped) if matches!(mapped.memory, Memory::Device { .. }) => {
                mapped.device().map(Some)
            }
            _ => Ok(None),
        };
        let runner = match (local(destination)?, local(source)?) {
            (Some(device), _) | (None, Some(device)) => Runner::Device(device),
            (None, None) if source.is_host() && destination.is_host() => Runner::Host,
            (None, None) => Runner::Device(
                self.gpus
                    .first()
                    .ok_or("this host has no GPU to import another host's memory on")?
                    .device,
            ),
        };
        if let Runner::Device(device) = runner {
            for side in [source, destination] {
                if let Side::Given(mapped) = side
                    && matches!(mapped.memory, Memory::Device { .. })
                    && !mapped.readable_by(device)
                {
                    let reader = self.gpus.by_device(device).map_or_else(
                        || device.to_string(),
                        |gpu| memory_name(&Memory::Device { gpu: gpu.uuid }),
                    );
                    return Err(format!(
                        "{reader} cannot address the memory of {}",
                        memory_name(&mapped.memory)
                    ));
                }
            }
        }
        Ok(runner)
    }
}

/// One task's imports of other hosts' allocations, by fabric handle and
/// importing GPU.
type Imports = HashMap<([u8; 64], cuda::DeviceId), Arc<Fabric>>;

/// One side of a move, before the lane that copies it is known.
enum Side {
    Given(Arc<Mapped>),
    Fabric {
        handle: [u8; 64],
        gpu: [u8; 16],
        bytes: u64,
    },
}

impl Side {
    fn memory(&self) -> Memory {
        match self {
            Side::Given(mapped) => mapped.memory.clone(),
            Side::Fabric { gpu, .. } => Memory::Device { gpu: *gpu },
        }
    }

    fn is_host(&self) -> bool {
        self.memory() == Memory::Host
    }

    /// The piece's end on `runner`: another host's allocation is imported
    /// on the GPU that runs the copy, once per task (`imports`).
    fn end(&self, runner: Runner, imports: &mut Imports) -> Result<End, String> {
        match (self, runner) {
            (Side::Given(mapped), _) => Ok(End::Given(Arc::clone(mapped))),
            (Side::Fabric { handle, bytes, .. }, Runner::Device(device)) => {
                let import = imports
                    .entry((*handle, device))
                    .or_insert_with(|| Arc::new(Fabric::new(*handle, *bytes, device)));
                Ok(End::Fabric(Arc::clone(import)))
            }
            (Side::Fabric { .. }, Runner::Host | Runner::Network) => {
                Err("another host's GPU memory is copied by a GPU of this host".into())
            }
        }
    }
}

/// A link end's name, for the log and the listing: `host`, a GPU's UUID,
/// or another host's Storage Server endpoint with its GPU or `host`.
pub(crate) fn memory_name(memory: &Memory) -> String {
    match memory {
        Memory::Host => "host".into(),
        Memory::Device { gpu } => cuda::DeviceUuid(*gpu).to_string(),
        Memory::Remote { endpoint, gpu } => match gpu {
            Some(gpu) => format!("{endpoint} {}", cuda::DeviceUuid(*gpu)),
            None => format!("{endpoint} host"),
        },
    }
}

/// Empty the wake pipe.
fn drain(reader: &OwnedFd) {
    let mut buffer = [0_u8; 64];
    while matches!(rustix::io::read(reader, &mut buffer), Ok(read) if read > 0) {}
}
