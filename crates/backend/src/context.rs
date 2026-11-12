//! The execution context: which stream, kernels, allocator, input buffer, group, expert exchange
//! and profiler a device operation uses.
//!
//! Execution builds one context per runner and installs it on its thread around a model call
//! ([`Context::enter`]), as PyTorch's current stream and graph memory pool are the thread's. A
//! model computes on tensors and never sees the context; every device operation of this crate
//! takes the one installed on its thread, and refuses when there is none.
//!
//! A context is one of two kinds, and what it holds says which. With a stream and a generation it
//! launches: its operations issue kernels and copies on the stream, and while the stream is being
//! captured they become the graph's nodes and input copies. With neither it is a **meta
//! context**: its allocator hands out addresses nothing backs or only counts, its launches state
//! their keys, and its host-to-device copies write the input buffer and issue nothing. The
//! profile run and the forward of `prepare_inputs` run under meta contexts.
//!
//! One executor thread issues a device's work, so nothing here is contended. The state is behind
//! a lock all the same, because a runner is built on one thread and moved to the thread of its
//! device with its context.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use ops::moe::ExchangeTransport;
use ops::platform::{DeviceRange, Stream};
use ops::{Generation, Route};

use crate::allocator::{ALIGNMENT, Allocator};
use crate::breaks::HostWork;
use crate::collective::Group;
use crate::profiler::Profiler;
use crate::{Error, Result};

thread_local! {
    /// The context installed on this thread, or none.
    static INSTALLED: RefCell<Option<Arc<Context>>> = const { RefCell::new(None) };
}

/// The execution resources of one runner's forwards.
pub struct Context {
    /// The GPU ordinal and the stream of a context that launches; a meta context has neither.
    device: Option<(usize, Stream)>,
    /// Where a forward's tensors take their memory. Shared, because a storage keeps its
    /// allocator to return its block when the tensor drops, on whatever thread that is.
    allocator: Arc<Allocator>,
    /// What changes between and during forwards. Shared with the handle a thread holds while the
    /// context is installed on it ([`Context::enter`]).
    state: Arc<Mutex<State>>,
}

/// The part of a context that changes: what execution sets between forwards, and what a forward
/// leaves behind for execution to take.
pub(crate) struct State {
    /// The generation launches resolve their kernels from; a meta context has none.
    pub(crate) kernels: Option<Arc<Generation>>,
    /// Where launches are offered first, when execution routes some of them itself.
    pub(crate) route: Option<Arc<dyn Route + Send + Sync>>,
    /// The communicator collectives are issued through.
    pub(crate) group: Option<Arc<Group>>,
    /// This rank's index in the group and the group's member count, when the forward is
    /// partitioned.
    pub(crate) split: Option<(u64, u64)>,
    /// The transport the expert exchange of this context's forwards runs over: the package's
    /// handle for this device and expert group, borrowed from execution, which keeps it alive
    /// for as long as it is set.
    pub(crate) exchange: Option<ExchangeTransport>,
    /// What an eager forward calls before each exchange launch to meet the other ranks of the
    /// expert group on the host ([`Context::set_exchange`]).
    pub(crate) meet: Option<Arc<dyn Fn() -> Result<()> + Send + Sync>>,
    /// What launches the range markers of the modules a forward enters.
    pub(crate) profiler: Option<Arc<dyn Profiler>>,
    /// What takes the graph breaks of a forward execution captures ([`Context::set_breaks`]).
    pub(crate) breaks: Option<Arc<dyn Fn(HostWork) -> Result<()> + Send + Sync>>,

    /// The launch table being recorded, as `(operator, key, module path)` in launch order, or
    /// `None` while none is.
    pub(crate) launches: Option<Vec<(&'static str, String, String)>>,
    /// The paths of the profiler ranges the forward is inside, outermost first. A range adds its
    /// path only while a launch table is being recorded, the table being its one reader.
    pub(crate) paths: Vec<String>,

    /// The page-locked host memory lent as the input buffer: its address and size in bytes.
    pub(crate) input_memory: Option<(u64, u64)>,
    /// The first byte of the input buffer no host-to-device copy of this forward has taken.
    pub(crate) input_cursor: u64,
    /// The furthest `input_cursor` any forward reached.
    pub(crate) input_peak: u64,
    /// The input copies of this forward, as `(buffer offset, device address, bytes)`: the
    /// host-to-device copies made while the stream was being captured, or under a meta context.
    pub(crate) input_copies: Vec<(u64, u64, u64)>,
}

/// Keeps a context installed on the thread that entered it; dropping it returns the context that
/// was installed before.
///
/// Guards nest as scopes do: the one entered last is dropped first.
#[must_use = "the context is installed only until the guard drops"]
pub struct ContextGuard<'a> {
    /// What was installed when this guard's context was entered.
    previous: Option<Arc<Context>>,
    /// The guard borrows its context, and stays on the thread whose installed context it
    /// restores (a raw pointer is neither `Send` nor `Sync`).
    _installed: PhantomData<(&'a Context, *const ())>,
}

impl Context {
    /// A context that launches: `stream` and `kernels` on GPU `ordinal`, allocating in `region`.
    ///
    /// The region is device memory the caller keeps mapped for as long as any tensor of this
    /// context, or any graph captured under it, is in use.
    pub fn new(
        ordinal: usize,
        stream: Stream,
        kernels: Arc<Generation>,
        region: DeviceRange,
    ) -> Self {
        Self {
            device: Some((ordinal, stream)),
            allocator: Arc::new(allocator(region)),
            state: Arc::new(Mutex::new(State::new(Some(kernels)))),
        }
    }

    /// A meta context: it launches nothing.
    ///
    /// With a region its allocations receive the addresses a launching context over the same
    /// region gives; without one it only counts.
    pub fn meta(region: Option<DeviceRange>) -> Self {
        Self {
            device: None,
            allocator: Arc::new(region.map_or_else(Allocator::counting, allocator)),
            state: Arc::new(Mutex::new(State::new(None))),
        }
    }

    /// Installs the context on this thread until the guard drops; the previous one returns then.
    pub fn enter(&self) -> ContextGuard<'_> {
        // The thread holds its own handle on the context's resources instead of a pointer to the
        // caller's value, so a guard that is never dropped cannot leave the thread with a context
        // that no longer exists.
        let installed = Arc::new(Self {
            device: self.device.clone(),
            allocator: Arc::clone(&self.allocator),
            state: Arc::clone(&self.state),
        });
        let previous = INSTALLED.with(|slot| slot.replace(Some(installed)));

        ContextGuard {
            previous,
            _installed: PhantomData,
        }
    }

    /// Empties the allocator and the input-buffer cursor: the start of a forward.
    ///
    /// A block allocated before the reset is no longer the allocator's when its tensor drops, and
    /// the input copies an earlier forward recorded and nobody took are dropped with it.
    pub fn reset(&self) {
        self.allocator.reset();

        let mut state = self.state();
        state.input_cursor = 0;
        state.input_copies.clear();
    }

    /// The furthest byte of the region any forward reached.
    pub fn peak(&self) -> u64 {
        self.allocator.peak()
    }

    /// The most bytes any forward of this context allocated in total: the bytes of everything
    /// one forward writes into tensors of its own.
    pub fn allocated_peak(&self) -> u64 {
        self.allocator.allocated_peak()
    }

    /// The furthest byte of the input buffer any forward reached.
    pub fn input_peak(&self) -> u64 {
        self.state().input_peak
    }

    /// Where host-to-device copies write their bytes: page-locked memory of this replay's slot.
    ///
    /// A null `base` withdraws the buffer. The cursor is not moved: [`Context::reset`] starts the
    /// forward that fills the buffer.
    ///
    /// # Safety
    ///
    /// `base` addresses `bytes` bytes of page-locked host memory that stay valid, and that
    /// nothing but this context's forwards writes, until another buffer is set and every copy
    /// issued or recorded from this one has completed. While the buffer is set, the caller
    /// resets the context only once the copies issued from the buffer have completed: the
    /// forward after a reset writes where they read.
    pub unsafe fn set_input_buffer(&self, base: *mut u8, bytes: u64) {
        self.state().input_memory = (!base.is_null()).then_some((base as u64, bytes));
    }

    /// Starts the recording of a launch table: `(operator, key, module path)` per launch, in
    /// launch order. A table already being recorded is dropped.
    ///
    /// Execution starts and takes a table outside the forward it records, so a profiler range
    /// lies wholly inside or wholly outside the recording.
    pub fn record_launches(&self) {
        self.state().launches = Some(Vec::new());
    }

    /// Ends the recording and returns the launch table; empty when none was being recorded.
    pub fn take_launches(&self) -> Vec<(&'static str, String, String)> {
        self.state().launches.take().unwrap_or_default()
    }

    /// The input copies of the forward since the last reset, as
    /// `(buffer offset, device address, bytes)` in the order the forward made them: those a
    /// capture recorded, or those a meta forward states a capture of it records.
    pub fn take_input_copies(&self) -> Vec<(u64, u64, u64)> {
        std::mem::take(&mut self.state().input_copies)
    }

    /// The generation launches resolve from, from the next launch on: set at a switch.
    pub fn set_kernels(&self, kernels: Arc<Generation>) {
        self.state().kernels = Some(kernels);
    }

    /// Where launches are offered before the context launches them itself, or `None` to launch
    /// every call once with the generation's kernel.
    pub fn set_route(&self, route: Option<Arc<dyn Route + Send + Sync>>) {
        self.state().route = route;
    }

    /// The group of a partitioned forward: this rank is member `rank` of `ranks`, and its
    /// collectives are issued through `group`. `ranks == 0` states an unpartitioned forward.
    ///
    /// A meta context issues nothing, so it needs the split alone and takes `None` for the
    /// group.
    ///
    /// # Errors
    ///
    /// [`Error::Collective`] when `rank` is not below `ranks`, when `group` is not member `rank`
    /// of `ranks`, or when a context that launches is given a split without a group.
    pub fn set_group(&self, group: Option<Arc<Group>>, rank: u64, ranks: u64) -> Result<()> {
        let refused = |problem: String| Error::Collective {
            collective: "set_group",
            problem,
        };

        if ranks == 0 {
            if group.is_some() {
                return Err(refused("a group of no members".to_owned()));
            }
            let mut state = self.state();
            state.group = None;
            state.split = None;
            return Ok(());
        }

        if rank >= ranks {
            return Err(refused(format!(
                "rank {rank} is not a member of a group of {ranks}"
            )));
        }

        match &group {
            // The collectives size their buffers by `ranks` and NCCL writes by the communicator's
            // own count, so the two must be one number.
            Some(group) => {
                let (member, size) = group.place();
                if (member, size) != (rank, ranks) {
                    return Err(refused(format!(
                        "the communicator is member {member} of {size}, not {rank} of {ranks}"
                    )));
                }
            }
            None if !self.is_meta() => {
                return Err(refused(
                    "a context that launches issues its collectives through a communicator, \
                     and none was given"
                        .to_owned(),
                ));
            }
            None => {}
        }

        let mut state = self.state();
        state.group = group;
        state.split = Some((rank, ranks));
        Ok(())
    }

    /// The expert exchange of this context's forwards: `transport` is the package's transport for
    /// this device and expert group, which an exchange launch runs over
    /// ([`crate::collective::expert_exchange`]), and `None` withdraws it.
    ///
    /// `meet`, when given, is called on the forward's thread before each exchange launch of a
    /// context that launches, and returns once every rank of the expert group has called it. An
    /// exchange launch waits on its device for the same launch of every rank, and the first use
    /// of a kernel takes a lock of the whole process and waits for its device; the ranks of a
    /// host are threads of one process, so execution gives a warm-up forward a `meet` that
    /// keeps every rank out of the waiting launch until each has finished the first uses before
    /// it, and gives none once every kernel is ready.
    ///
    /// A meta context launches nothing and needs neither. The transport is borrowed: execution
    /// keeps it valid, as the handle's constructor requires, for as long as it is set here.
    pub fn set_exchange(
        &self,
        transport: Option<ExchangeTransport>,
        meet: Option<Arc<dyn Fn() -> Result<()> + Send + Sync>>,
    ) {
        let mut state = self.state();
        state.exchange = transport;
        state.meet = meet;
    }

    /// What launches the range markers of the modules a forward enters, or `None`.
    pub fn set_profiler(&self, profiler: Option<Arc<dyn Profiler>>) {
        self.state().profiler = profiler;
    }

    /// What takes the graph breaks of this context's forwards ([`crate::graph_break`]), or
    /// `None` to run each break's host work where the forward states it.
    ///
    /// Execution sets a taker around a capture: the taker ends the segment being captured,
    /// keeps the host work for the replays and begins the next segment. It is called on the
    /// forward's thread, between two launches of the forward.
    pub fn set_breaks(&self, taker: Option<Arc<dyn Fn(HostWork) -> Result<()> + Send + Sync>>) {
        self.state().breaks = taker;
    }

    /// Whether this is a meta context: one with no stream, which launches and copies nothing.
    pub(crate) fn is_meta(&self) -> bool {
        self.device.is_none()
    }

    /// The stream launches, copies and collectives are issued on; `None` under a meta context.
    pub(crate) fn stream(&self) -> Option<&Stream> {
        self.device.as_ref().map(|(_, stream)| stream)
    }

    /// The GPU a launching context's tensors are on; `None` under a meta context.
    pub(crate) fn ordinal(&self) -> Option<usize> {
        self.device.as_ref().map(|(ordinal, _)| *ordinal)
    }

    /// The allocator a new storage takes its block from and keeps, to return the block on drop.
    pub(crate) fn allocator(&self) -> &Arc<Allocator> {
        &self.allocator
    }

    /// The changing part of the context, for one short section.
    ///
    /// The lock is never held across a call that leaves this crate (a route, a profiler, a
    /// kernel launch, the CUDA boundary): such a call may come back to the context. A panic
    /// while it was held leaves fields that the next forward's reset and setters overwrite, so a
    /// poisoned lock gives the state as it stands.
    pub(crate) fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl State {
    /// The state of a new context serving from `kernels`.
    fn new(kernels: Option<Arc<Generation>>) -> Self {
        Self {
            kernels,
            route: None,
            group: None,
            split: None,
            exchange: None,
            meet: None,
            profiler: None,
            breaks: None,
            launches: None,
            paths: Vec::new(),
            input_memory: None,
            input_cursor: 0,
            input_peak: 0,
            input_copies: Vec::new(),
        }
    }
}

impl Drop for ContextGuard<'_> {
    fn drop(&mut self) {
        let previous = self.previous.take();

        // A guard dropped while the thread's own storage is being torn down has nothing left to
        // restore.
        let _ = INSTALLED.try_with(|slot| slot.replace(previous));
    }
}

/// The context installed on this thread, when one is.
pub(crate) fn installed() -> Option<Arc<Context>> {
    INSTALLED.with(|slot| slot.borrow().clone())
}

/// The context installed on this thread, which `operation` needs.
///
/// # Errors
///
/// [`Error::NoContext`], naming `operation`, when none is installed.
pub(crate) fn current(operation: &'static str) -> Result<Arc<Context>> {
    installed().ok_or(Error::NoContext { operation })
}

/// The allocator of `region`.
///
/// The region's first address is aligned up to [`ALIGNMENT`] and the same bytes are taken off its
/// size: the allocator hands out offsets at that granularity, so a region that starts at the
/// pool's own granularity would put every tensor of a forward at the same skew, and kernels
/// compiled for aligned tensor pointers fault on it.
fn allocator(region: DeviceRange) -> Allocator {
    let address = region.ptr().address();
    let base = address.next_multiple_of(ALIGNMENT);
    let capacity = (region.bytes() as u64).saturating_sub(base - address);

    Allocator::new(base, capacity)
}

#[cfg(test)]
mod tests {
    use ops::platform::{DeviceRange, PlatformDevicePtr};

    use super::Context;
    use crate::allocator::ALIGNMENT;
    use crate::collective::{finish, split};
    use crate::{Error, Result};

    /// Whether `result` is the refusal of an operation that found no context installed.
    fn uninstalled(result: Result<()>) -> bool {
        matches!(result, Err(Error::NoContext { .. }))
    }

    /// A context's split tells it from another's, so `split()` shows which one an operation
    /// finds installed.
    #[test]
    fn guards_nest_and_return_the_previous_context() {
        let outer = Context::meta(None);
        let inner = Context::meta(None);
        outer.set_group(None, 0, 2).unwrap();
        inner.set_group(None, 1, 4).unwrap();

        assert!(uninstalled(finish(0)));
        {
            let _outer = outer.enter();
            assert_eq!(split(), Some((0, 2)));
            {
                let _inner = inner.enter();
                assert_eq!(split(), Some((1, 4)));
            }
            assert_eq!(split(), Some((0, 2)));
        }
        assert!(uninstalled(finish(0)));
    }

    #[test]
    fn an_installed_context_sees_what_is_set_after_it_was_entered() {
        let context = Context::meta(None);
        let _guard = context.enter();
        assert_eq!(split(), None);

        context.set_group(None, 1, 4).unwrap();
        assert_eq!(split(), Some((1, 4)));
    }

    #[test]
    fn a_context_moves_to_the_thread_of_its_device() {
        fn shared<T: Send + Sync>() {}
        shared::<Context>();
    }

    #[test]
    fn a_skewed_region_is_aligned_before_anything_is_allocated() {
        let region = DeviceRange::new(PlatformDevicePtr::from_address(0x1_0008), 4096);
        let context = Context::meta(Some(region));

        let (address, _) = context.allocator().allocate(8).unwrap();
        assert_eq!(address, 0x1_0100);

        // The bytes below the aligned base are not the allocator's: 4096 - 0xF8 are left, which
        // hold fourteen whole blocks after the first and not one more.
        context.allocator().allocate(14 * ALIGNMENT).unwrap();
        assert!(context.allocator().allocate(1).is_err());
    }

    #[test]
    fn a_group_is_refused_when_the_rank_is_not_a_member() {
        let context = Context::meta(None);
        let refused = context.set_group(None, 2, 2).unwrap_err();
        assert!(matches!(
            refused,
            Error::Collective {
                collective: "set_group",
                ..
            }
        ));

        // No members: the forward is unpartitioned again.
        context.set_group(None, 0, 2).unwrap();
        context.set_group(None, 0, 0).unwrap();
        let _guard = context.enter();
        assert_eq!(split(), None);
    }
}
