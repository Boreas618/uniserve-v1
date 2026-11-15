//! Each GPU's CUDA context, held only while the Transfer Server has work on
//! the GPU, and reserved in the GPU's quotas only while held.
//!
//! A primary context takes device storage whether or not anything runs in it
//! (hundreds of MiB on a GB200), and the Transfer Server runs on every host
//! with a Storage Server, so it holds a context only for the work that needs
//! one, as an update's preparation does:
//!
//! - A [`Hold`] is a claim on one GPU, counted per GPU, and taking or
//!   dropping one waits for no CUDA call and no quota. Every piece of a
//!   task holds the GPUs it reads, writes or runs on; a GPU copy thread holds its
//!   GPU while its thread runs; an exposure to another host holds the GPU of
//!   each device allocation it exposes, until that host ends it.
//!   An explicit transfer session also holds its operation's GPUs between
//!   tasks, until its client closes the session or disconnects. Opening a
//!   session alone neither starts nor reserves a CUDA context.
//! - The first use of a GPU under its holds starts a [`Session`]
//!   ([`Slot::retain`]): the context is reserved in every quota of the
//!   GPU, at the bytes the Transfer Server measured it at when it started,
//!   and then the GPU's primary context is retained. A quota that has no
//!   room for it refuses the reservation, and the use fails, so a quota never
//!   passes its quota for the context. Everything the Transfer Server
//!   creates in the context lives in the session or under one of its holds:
//!   device mappings of the server's allocations, host allocations
//!   registered for the GPU's copy engines, NIXL registrations of device
//!   storage, and the copy threads' streams and events.
//! - When the last hold of a GPU ends, its session closes ([`Slot::release`]):
//!   the registrations, then the mappings, then the context are released,
//!   the process waits for the driver calls it deferred (`platform::reclaim`),
//!   so that the driver destroyed the context and NVML lists the process on
//!   the GPU no more, and then the reservation ends. The next hold that uses the
//!   GPU starts a new session.
//!
//! The main thread, which holds the connection to the server, places the
//! reservations ([`Reservations`]); a session starts on the thread that first uses
//! the GPU, never on the main thread, which waits for no reservation.
//!
//! An allocation the server takes back leaves every session once no piece
//! copies it ([`Contexts::forget`]), so that a GPU busy for long does not
//! keep released storage mapped.

use std::collections::HashMap;
use std::os::fd::OwnedFd;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock, PoisonError, mpsc};
use std::time::Instant;

use platform::{DeviceId, HostRegistration, PeerMapping, PrimaryContext, Protection};

use super::nixl;
use super::storage::{HostMap, Mapping};

/// A value made once, or why it could not be made.
type Once<T> = OnceLock<Result<T, String>>;

/// One GPU as its context is reserved: its device, UUID and total storage, and
/// the bytes its context takes, as the Transfer Server measured them when
/// it started.
pub(super) struct GpuContext {
    pub(super) device: DeviceId,
    pub(super) uuid: [u8; 16],
    pub(super) total: u64,
    pub(super) context: u64,
}

/// A change of a GPU's reservation the main thread places in the server's
/// quotas ([`Request::Quota`](::storage::protocol::Request::Quota) of
/// [`Item::TransferContext`](::storage::protocol::Item::TransferContext)).
pub(super) struct Reservation {
    pub(super) gpu: [u8; 16],
    pub(super) total: u64,
    /// The context's bytes; 0 ends the reservation.
    pub(super) bytes: u64,
    /// Where the server's response goes: a refusal's message, or nothing when
    /// it reserved. `None` for an ending, which is never refused.
    pub(super) response: Option<mpsc::SyncSender<Result<(), String>>>,
}

/// The way to the main thread for reservations: their queue and its wake pipe.
pub(super) struct Reservations {
    pub(super) sender: mpsc::Sender<Reservation>,
    pub(super) wake: Arc<OwnedFd>,
}

impl Reservations {
    fn send(&self, reservation: Reservation) -> Result<(), String> {
        self.sender
            .send(reservation)
            .map_err(|_| "the Transfer Server stops".to_owned())?;
        let _ = rustix::io::write(&*self.wake, b"x");
        Ok(())
    }
}

/// The GPUs' contexts, one slot per GPU the process sees.
pub(super) struct Contexts {
    slots: Vec<Arc<Slot>>,
}

impl Contexts {
    /// A slot for each of `gpus`, with no context held, reservation through
    /// `reservations`.
    pub(super) fn new(gpus: Vec<GpuContext>, reservations: Reservations) -> Self {
        let reservations = Arc::new(reservations);
        Self {
            slots: gpus
                .into_iter()
                .map(|gpu| {
                    Arc::new(Slot {
                        gpu,
                        reservations: Arc::clone(&reservations),
                        state: Mutex::new(State {
                            holders: 0,
                            session: None,
                            changing: false,
                        }),
                        changed: Condvar::new(),
                    })
                })
                .collect(),
        }
    }

    /// A hold on `device`; its context is retained once the hold uses it
    /// ([`Hold::session`]).
    pub(super) fn hold(&self, device: DeviceId) -> Result<Hold, String> {
        let slot = self
            .slots
            .iter()
            .find(|slot| slot.gpu.device == device)
            .ok_or_else(|| format!("{device} is not visible to the Transfer Server"))?;
        slot.lock().holders += 1;
        Ok(Hold(Arc::new(Guard {
            slot: Arc::clone(slot),
        })))
    }

    /// Drop what every live session made for the allocation `id`: the server
    /// took it back and no piece copies it any more.
    pub(super) fn forget(&self, id: u64) {
        for slot in &self.slots {
            let session = slot.lock().session.clone();
            if let Some(session) = session {
                session.forget(id);
            }
        }
    }
}

/// One GPU's holds and, while one of them used the GPU, its session.
pub(super) struct Slot {
    gpu: GpuContext,
    reservations: Arc<Reservations>,
    state: Mutex<State>,
    /// Signalled when a session started or closed.
    changed: Condvar,
}

struct State {
    holders: usize,
    session: Option<Arc<Session>>,
    /// Whether a session is starting or closing, outside the lock: its
    /// reservation waits for the server, and its release for the driver.
    changing: bool,
}

impl Slot {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The session, started when there is none. One session starts or
    /// closes at a time: a use while the previous session closes waits for
    /// the close, and then starts a new one.
    fn session(&self) -> Result<Arc<Session>, String> {
        let mut state = self.lock();
        loop {
            if let Some(session) = &state.session {
                return Ok(Arc::clone(session));
            }
            if !state.changing {
                break;
            }
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
        state.changing = true;
        drop(state);

        // The caller's hold keeps the slot's holders above 0 meanwhile, so
        // no close runs beside the start.
        let started = self.retain();
        let mut state = self.lock();
        state.changing = false;
        if let Ok(session) = &started {
            state.session = Some(Arc::clone(session));
        }
        drop(state);
        self.changed.notify_all();
        started
    }

    /// The moment the Transfer Server starts to hold the GPU's context: it
    /// reserves the context's bytes in every quota of the GPU, and once
    /// the server reserved them, retains the context. Its storage counts in the
    /// quotas from here until [`Slot::release`].
    fn retain(&self) -> Result<Arc<Session>, String> {
        let (response, responded) = mpsc::sync_channel(1);
        self.reservations.send(Reservation {
            gpu: self.gpu.uuid,
            total: self.gpu.total,
            bytes: self.gpu.context,
            response: Some(response),
        })?;
        responded
            .recv()
            .map_err(|_| "the Transfer Server stops".to_owned())?
            .map_err(|refusal| {
                format!(
                    "the quotas of {} have no room for the Transfer Server's CUDA \
                     context: {refusal}",
                    self.name()
                )
            })?;

        let started = Instant::now();
        let context = match PrimaryContext::retain(self.gpu.device) {
            Ok(context) => context,
            Err(error) => {
                self.release_reservation();
                return Err(format!(
                    "retaining the CUDA context of {}: {error}",
                    self.name()
                ));
            }
        };
        tracing::info!(
            gpu = %self.name(),
            milliseconds = started.elapsed().as_secs_f64() * 1e3,
            "retained the GPU's CUDA context for transfers"
        );
        Ok(Arc::new(Session {
            device: self.gpu.device,
            live: Mutex::new(Some(Live {
                nixl: HashMap::new(),
                host: HashMap::new(),
                mappings: HashMap::new(),
                _context: context,
            })),
        }))
    }

    /// The moment the Transfer Server stops holding the GPU's context: the
    /// session's registrations, mappings and context are released, the
    /// driver calls they deferred have run, so the context's device storage
    /// is free, and then its reservation ends.
    fn release(&self, session: &Session) {
        let started = Instant::now();
        session.close();
        if let Err(error) = platform::reclaim::flush() {
            tracing::warn!(%error, gpu = %self.name(), "releasing a GPU's mappings failed");
        }
        self.release_reservation();
        tracing::info!(
            gpu = %self.name(),
            milliseconds = started.elapsed().as_secs_f64() * 1e3,
            "released the GPU's CUDA context: no transfer uses the GPU"
        );
    }

    /// End the context's reservation; the main thread places it without waiting.
    fn release_reservation(&self) {
        let _ = self.reservations.send(Reservation {
            gpu: self.gpu.uuid,
            total: self.gpu.total,
            bytes: 0,
            response: None,
        });
    }

    fn name(&self) -> String {
        platform::DeviceUuid(self.gpu.uuid).to_string()
    }
}

/// A hold on one GPU's context. Clones share the hold, which ends with the
/// last.
#[derive(Clone)]
pub(super) struct Hold(Arc<Guard>);

struct Guard {
    slot: Arc<Slot>,
}

impl Drop for Guard {
    fn drop(&mut self) {
        let slot = &self.slot;
        let mut state = slot.lock();
        state.holders -= 1;
        if state.holders > 0 {
            return;
        }
        let Some(session) = state.session.take() else {
            return;
        };
        state.changing = true;
        drop(state);

        // Outside the lock, so that taking a hold never waits for a release.
        slot.release(&session);
        slot.lock().changing = false;
        slot.changed.notify_all();
    }
}

impl Hold {
    /// The GPU's session, started (its context reserved and retained) when the
    /// GPU has none. The caller uses it only while it holds this hold, and
    /// never on the main thread, which places the reservation.
    pub(super) fn session(&self) -> Result<Arc<Session>, String> {
        self.0.slot.session()
    }
}

/// One GPU's context while the Transfer Server holds it, and what lives in
/// it, by the id of the allocation it was made for.
pub(super) struct Session {
    device: DeviceId,
    /// `None` once the session closed.
    live: Mutex<Option<Live>>,
}

/// A session's objects. Fields drop in declaration order: a NIXL
/// registration before the mapping it registers, the mappings and host
/// registrations before the context they live in.
struct Live {
    nixl: HashMap<u64, Arc<Once<nixl::Registration>>>,
    host: HashMap<u64, Arc<Once<HostRange>>>,
    mappings: HashMap<u64, Arc<Once<PeerMapping>>>,
    _context: PrimaryContext,
}

/// A host allocation mapped for one GPU's copy engines: a mapping of its own
/// (a range can be registered once, in one context) and its registration in
/// the GPU's context, undone before the range is unmapped.
struct HostRange {
    registration: HostRegistration,
    _map: HostMap,
}

impl Session {
    /// The entry of `id` in one of the session's tables, made by `make` at
    /// its first use outside the table's lock: making a mapping takes a
    /// while, and other allocations' entries are used meanwhile.
    fn entry<T>(
        &self,
        table: impl FnOnce(&mut Live) -> &mut HashMap<u64, Arc<Once<T>>>,
        id: u64,
        make: impl FnOnce() -> Result<T, String>,
    ) -> Result<Arc<Once<T>>, String> {
        let entry = {
            let mut live = self.live.lock().unwrap_or_else(PoisonError::into_inner);
            let live = live.as_mut().ok_or("the GPU's transfer session closed")?;
            Arc::clone(table(live).entry(id).or_default())
        };
        entry.get_or_init(make);
        Ok(entry)
    }

    /// The device address of byte 0 of `mapped`, a device allocation on this
    /// GPU, mapped with access for every GPU that can address it.
    pub(super) fn device_address(&self, mapped: &Mapping) -> Result<u64, String> {
        let entry = self.entry(
            |live| &mut live.mappings,
            mapped.id(),
            || {
                let protection = if mapped.writable() {
                    Protection::ReadWrite
                } else {
                    Protection::ReadOnly
                };
                PeerMapping::import_retained(
                    self.device,
                    mapped.descriptor(),
                    mapped.length()?,
                    mapped.readers(),
                    protection,
                )
                .map_err(|error| format!("mapping a device allocation: {error}"))
            },
        )?;
        let made = entry.get().map(Result::as_ref);
        match made {
            Some(Ok(mapping)) => Ok(mapping.address()),
            Some(Err(message)) => Err(message.clone()),
            None => Err("a mapping was not made".into()),
        }
    }

    /// The address of byte 0 of `mapped`, host storage, in this GPU's own
    /// mapping of it, registered so that the GPU's copy engines reach it.
    ///
    /// A read-only allocation registers for reads alone where the device
    /// supports it. A device that does not (GB200 reports
    /// `ReadOnlyHostRegisterSupported` 0 and refuses the flag) registers its
    /// read-only mapping as a plain registration. Either way the Transfer
    /// Server only reads a read-only allocation: the server refuses it as a
    /// destination.
    pub(super) fn host_address(&self, mapped: &Mapping) -> Result<u64, String> {
        let entry = self.entry(
            |live| &mut live.host,
            mapped.id(),
            || {
                let length = mapped.length()?;
                let map = HostMap::new(mapped.descriptor(), length, mapped.writable())
                    .map_err(|error| format!("mapping host storage: {error}"))?;
                let pointer = map.address() as *mut u8;
                let read_only = !mapped.writable()
                    && HostRegistration::read_only_supported(self.device)
                        .map_err(|error| format!("querying host registration: {error}"))?;
                // SAFETY: `map` maps the whole range and is kept beside the
                // registration, which is undone first.
                let registered = unsafe {
                    if read_only {
                        HostRegistration::read_only(self.device, pointer, length)
                    } else {
                        HostRegistration::new(self.device, pointer, length)
                    }
                };
                let registration =
                    registered.map_err(|error| format!("registering host storage: {error}"))?;
                Ok(HostRange {
                    registration,
                    _map: map,
                })
            },
        )?;
        let made = entry.get().map(Result::as_ref);
        match made {
            Some(Ok(range)) => Ok(range.registration.address()),
            Some(Err(message)) => Err(message.clone()),
            None => Err("a host registration was not made".into()),
        }
    }

    /// `mapped`, a device allocation on this GPU, as NIXL names it,
    /// registered with `agent` at the first call: the range this agent's
    /// READs write into, or another host's READs read. It registers the
    /// session's mapping, under a key no other registration of the process
    /// takes ([`nixl::Region::key`]).
    pub(super) fn nixl_region(
        &self,
        agent: &nixl::Agent,
        mapped: &Mapping,
    ) -> Result<nixl::Region, String> {
        let address = self.device_address(mapped)?;
        let entry = self.entry(
            |live| &mut live.nixl,
            mapped.id(),
            || {
                let region = nixl::Region {
                    segment: nixl::Segment::Device,
                    address,
                    len: mapped.length()?,
                    key: super::storage::nixl_key(),
                };
                // SAFETY: the session keeps the mapping until after the
                // registration dropped (`Live`'s field order, and `forget`
                // drops them in the same order); the Transfer Server holds
                // no Rust reference to the bytes, which its copies reach
                // through raw addresses only.
                unsafe { agent.register(region, nixl::Access::ReadWrite) }
                    .map_err(|error| format!("registering storage with NIXL: {error}"))
            },
        )?;
        let made = entry.get().map(Result::as_ref);
        match made {
            Some(Ok(registration)) => Ok(registration.region()),
            Some(Err(message)) => Err(message.clone()),
            None => Err("a NIXL registration was not made".into()),
        }
    }

    /// Drop what the session made for the allocation `id`, registration
    /// first.
    fn forget(&self, id: u64) {
        let taken = {
            let mut live = self.live.lock().unwrap_or_else(PoisonError::into_inner);
            live.as_mut().map(|live| {
                (
                    live.nixl.remove(&id),
                    live.host.remove(&id),
                    live.mappings.remove(&id),
                )
            })
        };
        // A tuple drops its fields in order, outside the table's lock.
        drop(taken);
    }

    /// Release everything the session made, then the context.
    fn close(&self) {
        let live = self
            .live
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        drop(live);
    }
}
