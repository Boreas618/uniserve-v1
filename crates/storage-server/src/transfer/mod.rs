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
//! ([`devices`]), then releases its contexts, connects as
//! [`Role::Transfer`] and reports both ([`Request::Measured`]). It holds a
//! GPU's context again only while it has work on the GPU, booked in the
//! GPU's memory ledgers at the measured bytes meanwhile ([`contexts`]). It
//! then runs the server's tasks
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
//! ([`Request::Registered`]); the registrations, and the GPU contexts they
//! need, stay until the peer ends the exposure ([`Event::Deregister`]). A
//! task READing from another host ends its exposures there once it ended.
//! What waits on another host, on a GPU's context or on a registration runs
//! on a helper or lane thread, so the main thread keeps taking the server's
//! events and the lanes' reports meanwhile, and places the bookings the
//! other threads wait for.
//!
//! It also creates device memory the server holds from then on: a world's
//! weights returning from host memory need memory on their GPUs, and the
//! server creates no CUDA context ([`Event::Allocate`]). It creates the
//! allocation on a helper, retainable through a POSIX descriptor, sends the
//! descriptor ([`Request::Allocated`]) and drops its own handle: the
//! descriptor keeps the memory, and the server gives it back for the copy.
//!
//! It reports each task's progress and end ([`Request::Copied`]) only
//! after the host saw the copies complete (an event its lane queried, or a
//! READ it polled), so no CUDA event crosses a process. It ends when the
//! server closes its connection.

mod contexts;
mod devices;
mod lanes;
mod memory;
mod network;

use std::collections::HashMap;
use std::io;
use std::os::fd::{AsFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};
use std::time::Duration;

use cuda::DeviceId;
use rustix::event::{PollFd, PollFlags, poll};
use storage::client::Client;
use storage::protocol::{
    Allocation, ContextBytes, Event, Item, Link, Memory, Move, Priority, Registered, Reply,
    Request, Role, TransferEnd,
};

use contexts::{Booked, Booking, Bookings, Contexts, Hold};
use devices::Gpus;
use lanes::{End, LaneKey, Lanes, Piece, Runner};
use memory::{Fabric, Mapped};
use network::{Exposed, Inbound, Lease, Network, Sources};

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
    let (links, context_bytes) = devices::measure(&gpus)?;
    let bandwidths = links
        .iter()
        .map(|measured| (measured.link.clone(), measured.bytes_per_second as f64))
        .collect();
    let booked = booked(&gpus, &context_bytes)?;

    let mut client = Client::connect(socket, Role::Transfer)?;
    let (wake_reader, wake_writer) = rustix::pipe::pipe_with(
        rustix::pipe::PipeFlags::CLOEXEC | rustix::pipe::PipeFlags::NONBLOCK,
    )
    .map_err(io::Error::from)?;
    let wake = Arc::new(wake_writer.try_clone()?);
    let (booking_sender, bookings) = mpsc::channel();
    let contexts = Arc::new(Contexts::new(
        booked,
        Bookings {
            sender: booking_sender,
            wake: Arc::clone(&wake),
        },
    ));
    let network = Arc::new(Network::new(options.nixl_library.clone()));
    let (sender, finished) = mpsc::channel();
    let mut server = Served {
        gpus,
        contexts: Arc::clone(&contexts),
        allocations: HashMap::new(),
        helpers: Helpers { sender, wake },
        lanes: Lanes::new(
            wake_writer,
            options.chunk,
            bandwidths,
            Arc::clone(&network),
            contexts,
        ),
        network,
        finished,
        bookings,
        answers: HashMap::new(),
        exposing: HashMap::new(),
        exposures: HashMap::new(),
        leases: HashMap::new(),
        replies: Vec::new(),
    };
    let measured = Request::Measured {
        links,
        contexts: context_bytes,
    };
    server.send(&mut client, &measured)?;
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
            if let Some((report, descriptor)) = server.finish(done) {
                server.send_with(&mut client, &report, descriptor.as_ref())?;
            }
        }
        while let Ok(booking) = server.bookings.try_recv() {
            server.book(&mut client, booking)?;
        }
        for (task, bytes, end) in server.lanes.reports() {
            if end.is_some() {
                server.ended(task);
            }
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

/// Each GPU as its context is booked: the context's bytes as measured, and
/// the GPU's total memory as NVML reports it.
fn booked(gpus: &Gpus, measured: &[ContextBytes]) -> Result<Vec<Booked>, Error> {
    let identities: Vec<(DeviceId, [u8; 16])> = gpus.identities().collect();
    if identities.is_empty() {
        return Ok(Vec::new());
    }
    let nvml = cuda::Nvml::init()?;
    identities
        .into_iter()
        .map(|(device, uuid)| {
            Ok(Booked {
                device,
                uuid,
                total: nvml.memory_info(device)?.total,
                context: measured
                    .iter()
                    .find(|context| context.gpu == uuid)
                    .map_or(0, |context| context.bytes),
            })
        })
        .collect()
}

/// What a helper thread finished, for the main thread.
enum Done {
    /// The pieces of task `task`, its moves from other hosts among them
    /// once those hosts exposed their sources, with the exposures it ends
    /// when it ends; or why it cannot run.
    Sources {
        task: u64,
        priority: Priority,
        pieces: Result<Sources, String>,
    },
    /// The registrations of the server's `request`: the agent's metadata
    /// and each allocation's range, or why they failed; and the holds on
    /// the GPUs of the exposed device allocations, which the exposure
    /// keeps.
    Registered {
        request: u64,
        holds: Vec<Hold>,
        registered: Result<(Vec<u8>, Vec<Registered>), String>,
    },
    /// The device memory of the server's allocation `request`: its POSIX
    /// descriptor and its size, rounded up to the GPU's granularity; or
    /// why it could not be created.
    Allocated {
        request: u64,
        allocated: Result<(OwnedFd, u64), String>,
    },
}

/// Runs work that waits on another host, on a GPU's context or on
/// registering memory, off the main thread, and wakes the main thread with
/// its outcome.
struct Helpers {
    sender: mpsc::Sender<Done>,
    /// The write end of the main thread's wake pipe.
    wake: Arc<OwnedFd>,
}

impl Helpers {
    fn spawn(&self, work: impl FnOnce() -> Option<Done> + Send + 'static) {
        let sender = self.sender.clone();
        let wake = Arc::clone(&self.wake);
        let spawned = std::thread::Builder::new()
            .name("uniserve-transfer-helper".into())
            .spawn(move || {
                if let Some(done) = work() {
                    // The main thread ends only with the process, so its
                    // receiver outlives every helper.
                    let _ = sender.send(done);
                    let _ = rustix::io::write(&*wake, b"x");
                }
            });
        if let Err(error) = spawned {
            lanes::fatal("starting a helper thread", &error);
        }
    }
}

/// The holds one task takes, one per GPU, which its pieces share.
type TaskHolds = HashMap<DeviceId, Hold>;

/// The Transfer Server's state on its main thread.
struct Served {
    gpus: Gpus,
    /// The GPUs' contexts, held while work needs them.
    contexts: Arc<Contexts>,
    /// The allocations the server gave, until it takes them back.
    allocations: HashMap<Allocation, Arc<Mapped>>,
    lanes: Lanes,
    /// The NIXL agent of the path between hosts, which the lanes and the
    /// helpers share.
    network: Arc<Network>,
    helpers: Helpers,
    /// What the helpers finished.
    finished: mpsc::Receiver<Done>,
    /// The bookings of GPU contexts the main thread places in the server's
    /// ledgers.
    bookings: mpsc::Receiver<Booking>,
    /// Bookings placed whose answer a thread waits for, by request.
    answers: HashMap<u64, mpsc::SyncSender<Result<(), String>>>,
    /// The tasks whose sources other hosts are exposing, and whether the
    /// server cancelled each meanwhile.
    exposing: HashMap<u64, bool>,
    /// Exposures to other hosts, by the server's register request, with the
    /// holds on the GPUs of their device allocations.
    exposures: HashMap<u64, Vec<Hold>>,
    /// The exposures of other hosts each running task READs, which end
    /// with it.
    leases: HashMap<u64, Vec<Lease>>,
    /// Requests sent and not answered yet.
    replies: Vec<u64>,
}

impl Served {
    fn send(&mut self, client: &mut Client, request: &Request) -> Result<(), Error> {
        self.send_with(client, request, None)
    }

    /// Send `request` with `descriptor` travelling beside it.
    fn send_with(
        &mut self,
        client: &mut Client,
        request: &Request,
        descriptor: Option<&OwnedFd>,
    ) -> Result<(), Error> {
        let descriptors: Vec<_> = descriptor.iter().map(|fd| fd.as_fd()).collect();
        self.replies.push(client.submit(request, &descriptors)?);
        Ok(())
    }

    /// Place `booking` in the server's ledgers: a GPU's context, booked at
    /// its bytes before a thread retains it, or ended once it was released.
    fn book(&mut self, client: &mut Client, booking: Booking) -> Result<(), Error> {
        let request = Request::Book {
            gpu: booking.gpu,
            total: booking.total,
            quota: None,
            items: vec![(Item::TransferContext, booking.bytes)],
        };
        let id = client.submit(&request, &[])?;
        match booking.answer {
            Some(answer) => {
                self.answers.insert(id, answer);
            }
            None => self.replies.push(id),
        }
        Ok(())
    }

    /// Take the server's answers: a booking's answer goes to the thread that
    /// waits for it; an error to a report names a task the server no longer
    /// holds, which changes nothing here.
    fn collect(&mut self, client: &mut Client) {
        self.answers.retain(|id, answer| match client.take(*id) {
            Some(answered) => {
                let outcome = match answered.reply {
                    Reply::Error { message, .. } => Err(message),
                    _ => Ok(()),
                };
                let _ = answer.send(outcome);
                false
            }
            None => true,
        });
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
                let mapped = Mapped::new(
                    memory,
                    bytes,
                    writable,
                    descriptor,
                    &self.gpus,
                    Arc::clone(&self.contexts),
                );
                self.allocations.insert(allocation, Arc::new(mapped));
            }
            Event::AllocationReleased { allocation } => {
                // A task that copies it holds it until its copies complete;
                // then its mappings and registrations go, so another host's
                // READ of it fails from then on.
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
            Event::Throttle {
                links,
                throttle,
                tasks,
            } => {
                tracing::info!(
                    links = links.len(),
                    tasks = tasks.len(),
                    ?throttle,
                    "changing link budgets"
                );
                self.lanes.throttle(&links, &tasks, throttle);
            }
            Event::Register {
                request,
                allocations,
            } => self.register(request, &allocations),
            Event::Allocate {
                request,
                gpu,
                bytes,
            } => self.allocate(request, gpu, bytes),
            Event::Deregister { request } => {
                if let Some(holds) = self.exposures.remove(&request) {
                    // The last hold of a GPU releases its context, which
                    // waits for the driver: on a helper.
                    self.helpers.spawn(move || {
                        drop(holds);
                        None
                    });
                }
            }
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
        // and GPU, whatever number of its moves read or write it; a GPU is
        // held once per task, whatever number of its pieces need it.
        let mut imports = HashMap::new();
        let mut holds = TaskHolds::new();
        for copy in moves {
            let taken = match &copy.source {
                Allocation::Remote {
                    endpoint,
                    deployment,
                    region,
                } => self.given(&copy.destination).and_then(|destination| {
                    let hold = match destination.memory {
                        Memory::Device { .. } => {
                            Some(self.hold(&mut holds, destination.device()?)?)
                        }
                        _ => None,
                    };
                    inbound.push(Inbound {
                        endpoint: endpoint.clone(),
                        deployment: deployment.clone(),
                        region: (**region).clone(),
                        source_offset: copy.source_offset,
                        destination,
                        hold,
                        destination_offset: copy.destination_offset,
                        bytes: copy.bytes,
                    });
                    Ok(())
                }),
                _ => self
                    .piece(task, copy, &mut imports, &mut holds)
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
            let pieces = network.pieces(task, inbound).map(|mut sources| {
                pieces.append(&mut sources.pieces);
                Sources {
                    pieces,
                    leases: sources.leases,
                }
            });
            Some(Done::Sources {
                task,
                priority,
                pieces,
            })
        });
    }

    /// Task `task` ended: the exposures of other hosts it READ from end, on
    /// a helper, since each waits for its host's answer.
    fn ended(&mut self, task: u64) {
        if let Some(leases) = self.leases.remove(&task) {
            let network = Arc::clone(&self.network);
            self.helpers.spawn(move || {
                network.unexpose(leases);
                None
            });
        }
    }

    /// Register `allocations` with the NIXL agent for another host's READs,
    /// on a helper: registering device memory retains its GPU's context and
    /// pins the memory, which takes a while for a large page pool.
    fn register(&mut self, request: u64, allocations: &[Allocation]) {
        let mut holds = TaskHolds::new();
        let exposed: Result<Vec<Exposed>, String> = allocations
            .iter()
            .map(|allocation| {
                let mapped = self.given(allocation)?;
                let hold = match mapped.memory {
                    Memory::Device { .. } => Some(self.hold(&mut holds, mapped.device()?)?),
                    _ => None,
                };
                Ok((mapped, hold))
            })
            .collect();
        let network = Arc::clone(&self.network);
        self.helpers.spawn(move || {
            let registered = exposed
                .as_deref()
                .map_err(Clone::clone)
                .and_then(|exposed| network::register(&network, exposed));
            Some(Done::Registered {
                request,
                holds: holds.into_values().collect(),
                registered,
            })
        });
    }

    /// Create `bytes` of device memory on GPU `gpu` for the server's
    /// allocation `request`, on a helper: creating tens of gigabytes takes
    /// a while, and the lanes' reports go on meanwhile.
    fn allocate(&mut self, request: u64, gpu: [u8; 16], bytes: u64) {
        let held = self
            .gpus
            .by_uuid(gpu)
            .map(|found| (found.device, found.readers.clone()))
            .ok_or_else(|| {
                format!(
                    "GPU {} is not visible to the Transfer Server",
                    cuda::DeviceUuid(gpu)
                )
            })
            .and_then(|(device, readers)| Ok((self.contexts.hold(device)?, device, readers)));
        self.helpers.spawn(move || {
            // The allocation is made in the GPU's session, whose context the
            // ledgers book while it is held; the session closes once the
            // last hold ends, after the allocation's handle was released.
            let allocated = held.and_then(|(hold, device, readers)| {
                hold.session()?;
                device_memory(device, bytes, &readers)
            });
            Some(Done::Allocated { request, allocated })
        });
    }

    /// Take what a helper finished; returns the report it makes to the
    /// server, with the descriptor that travels beside it.
    fn finish(&mut self, done: Done) -> Option<(Request, Option<OwnedFd>)> {
        match done {
            Done::Sources {
                task,
                priority,
                pieces,
            } => {
                let cancelled = self.exposing.remove(&task).unwrap_or(false);
                match pieces {
                    Ok(sources) if cancelled => {
                        self.lanes.end(task, TransferEnd::Cancelled);
                        self.leases.insert(task, sources.leases);
                        self.ended(task);
                    }
                    Ok(sources) => {
                        self.leases.insert(task, sources.leases);
                        self.lanes.submit(task, priority, sources.pieces);
                    }
                    Err(_) if cancelled => self.lanes.end(task, TransferEnd::Cancelled),
                    Err(message) => {
                        tracing::warn!(task, message, "a task's sources on other hosts failed");
                        self.lanes.end(task, TransferEnd::Failed { message });
                    }
                }
                None
            }
            Done::Allocated { request, allocated } => Some(match allocated {
                Ok((descriptor, bytes)) => (
                    Request::Allocated {
                        request,
                        bytes,
                        failure: None,
                    },
                    Some(descriptor),
                ),
                Err(message) => {
                    tracing::warn!(message, "creating device memory for the server failed");
                    (
                        Request::Allocated {
                            request,
                            bytes: 0,
                            failure: Some(message),
                        },
                        None,
                    )
                }
            }),
            Done::Registered {
                request,
                holds,
                registered,
            } => Some((
                match registered {
                    Ok((metadata, ranges)) => {
                        self.exposures.insert(request, holds);
                        Request::Registered {
                            request,
                            metadata,
                            ranges,
                            failure: None,
                        }
                    }
                    Err(message) => {
                        tracing::warn!(message, "registering memory for another host failed");
                        self.helpers.spawn(move || {
                            drop(holds);
                            None
                        });
                        Request::Registered {
                            request,
                            metadata: Vec::new(),
                            ranges: Vec::new(),
                            failure: Some(message),
                        }
                    }
                },
                None,
            )),
        }
    }

    /// The task's hold on `device`, taken at its first use.
    fn hold(&self, holds: &mut TaskHolds, device: DeviceId) -> Result<Hold, String> {
        if let Some(hold) = holds.get(&device) {
            return Ok(hold.clone());
        }
        let hold = self.contexts.hold(device)?;
        holds.insert(device, hold.clone());
        Ok(hold)
    }

    /// The lane and the piece of one move between allocations of this host
    /// and of the fabric; `imports` holds the task's imports of other
    /// hosts' allocations, and `holds` its holds on the GPUs.
    fn piece(
        &self,
        task: u64,
        copy: &Move,
        imports: &mut Imports,
        holds: &mut TaskHolds,
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
            source: self.end(&source, runner, imports, holds)?,
            source_offset: copy.source_offset,
            destination: self.end(&destination, runner, imports, holds)?,
            destination_offset: copy.destination_offset,
            bytes: copy.bytes,
            rows: copy.rows,
            source_pitch: copy.source_pitch,
            destination_pitch: copy.destination_pitch,
        };
        Ok((LaneKey { link, runner }, piece))
    }

    /// A piece's end on `runner`, holding the GPU whose context reaches it:
    /// a device allocation's own GPU, the runner's for host memory a GPU
    /// copies and for another host's allocation of the fabric, which is
    /// imported there once per task (`imports`).
    fn end(
        &self,
        side: &Side,
        runner: Runner,
        imports: &mut Imports,
        holds: &mut TaskHolds,
    ) -> Result<End, String> {
        match (side, runner) {
            (Side::Given(mapped), _) => {
                let gpu = match (&mapped.memory, runner) {
                    (Memory::Device { .. }, _) => Some(mapped.device()?),
                    (Memory::Host, Runner::Device(device)) => Some(device),
                    _ => None,
                };
                let hold = gpu.map(|device| self.hold(holds, device)).transpose()?;
                Ok(End::Given {
                    mapped: Arc::clone(mapped),
                    hold,
                })
            }
            (Side::Fabric { handle, bytes, .. }, Runner::Device(device)) => {
                let import = imports
                    .entry((*handle, device))
                    .or_insert_with(|| Arc::new(Fabric::new(*handle, *bytes, device)));
                Ok(End::Fabric {
                    import: Arc::clone(import),
                    _hold: self.hold(holds, device)?,
                })
            }
            (Side::Fabric { .. }, Runner::Host | Runner::Network) => {
                Err("another host's GPU memory is copied by a GPU of this host".into())
            }
        }
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
type Imports = HashMap<([u8; 64], DeviceId), Arc<Fabric>>;

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

/// `bytes` of device memory on `device`, which every GPU of `readers` can
/// address, retainable through a POSIX descriptor: the descriptor and the
/// size, rounded up to the device's granularity. The allocation's handle
/// and mapping here are dropped; the descriptor alone keeps the memory.
fn device_memory(
    device: cuda::DeviceId,
    bytes: u64,
    readers: &[cuda::DeviceId],
) -> Result<(OwnedFd, u64), String> {
    let length = usize::try_from(bytes).map_err(|_| format!("{bytes} bytes overflow"))?;
    let allocation = cuda::DeviceAllocation::retainable(device, length, readers)
        .map_err(|error| format!("creating {bytes} bytes of device memory: {error}"))?;
    let descriptor = allocation
        .export_descriptor()
        .map_err(|error| format!("exporting device memory: {error}"))?
        .ok_or_else(|| "a retainable allocation exported no descriptor".to_owned())?;
    Ok((descriptor, allocation.len() as u64))
}

/// Empty the wake pipe.
fn drain(reader: &OwnedFd) {
    let mut buffer = [0_u8; 64];
    while matches!(rustix::io::read(reader, &mut buffer), Ok(read) if read > 0) {}
}
