//! Hardware counters per launch or per caller's range, collected in one
//! pass (CUPTI's range profiler, `cupti_range_profiler.h`).
//!
//! A [`RangeProfiler`] is enabled on one context with a single-pass counter
//! configuration ([`super::metrics::Counters::single_pass`]), CUPTI's user
//! ranges and user replay of that one pass, so no work is ever replayed.
//! CUPTI profiles the kernels launched between a range's push and its pop,
//! in the order of the device's work: a range pushed before a CUDA graph's
//! launch and popped after it measures the whole graph, though the pop
//! returns before the graph ran, whenever the graph was instantiated.
//! Between [`RangeProfiler::start`] and [`RangeProfiler::stop`] it collects:
//!
//! - [`Ranges::Launches`]: a range around each kernel launch and each CUDA
//!   graph launch on the context, by any thread of the process, named by
//!   the launch's index in the pass. The process's CUPTI subscription
//!   ([`super::subscriber`]) calls back as each launch's driver call is
//!   entered, which pushes the range, and as it returns, which pops it;
//!   PyTorch's runtime launches reach these driver calls too. One lock is
//!   held from the push to the pop, so no other launch of the process, from
//!   any thread, is issued inside a range, and a stop never meets an open
//!   range. On the device CUPTI counts a range's own kernels only, whatever
//!   other streams run beside them: on GB200 with `nvidia-cuda-cupti`
//!   13.4.92, a launch's range executes as many instructions while another
//!   stream's kernel runs as with nothing beside it.
//! - [`Ranges::User`]: the ranges the caller pushes and pops
//!   ([`RangeProfiler::push`], [`RangeProfiler::pop`]). A range is pushed on
//!   the host, so a CUDA graph's replay pushes none of the ranges its capture
//!   saw.
//!
//! The counter data image holds a bounded number of ranges;
//! [`RangeProfiler::stop`] decodes it into [`Range`] values and prepares it
//! for the next start, so a long collection stops and starts again before
//! the image fills.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard};

use super::metrics::{Availability, Counters, Profiler};
use super::subscriber::{self, API_ENTER, DRIVER_DOMAIN};
use super::{call, cupti, struct_size};
use crate::{Error, Result, lock};

/// What delimits the ranges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ranges {
    /// A range per kernel launch and per CUDA graph launch.
    Launches,
    /// The caller's pushed ranges.
    User,
}

/// `CUPTI_UserRange`: ranges are pushed and popped, here by the caller or
/// around each launch.
const USER_RANGE: u32 = 2;

/// `CUPTI_UserReplay`: the caller replays, which a one-pass configuration
/// never needs.
const USER_REPLAY: u32 = 3;

/// The driver calls that launch a kernel or a CUDA graph
/// (`CUpti_driver_api_trace_cbid`), each the range of a
/// [`Ranges::Launches`] profiler. Host functions launch no kernel.
const LAUNCHES: &[(u32, &str)] = &[
    (115, "cuLaunch"),
    (116, "cuLaunchGrid"),
    (117, "cuLaunchGridAsync"),
    (307, "cuLaunchKernel"),
    (442, "cuLaunchKernel_ptsz"),
    (477, "cuLaunchCooperativeKernel"),
    (478, "cuLaunchCooperativeKernel_ptsz"),
    (480, "cuLaunchCooperativeKernelMultiDevice"),
    (514, "cuGraphLaunch"),
    (515, "cuGraphLaunch_ptsz"),
    (652, "cuLaunchKernelEx"),
    (653, "cuLaunchKernelEx_ptsz"),
];

/// The started [`Ranges::Launches`] profiler, whose ranges the launch
/// callbacks push and pop.
struct Armed {
    /// The profiler object (`CUpti_RangeProfiler_Object`).
    object: usize,
    /// Its context (`CUcontext`): launches on other contexts take no range.
    context: usize,
    /// The launches ranged in this pass, which name the next range.
    launches: u64,
    /// The first push or pop CUPTI refused in this pass.
    failure: Option<Error>,
}

/// The started launch profiler, if any. Its lock is the launch lock: a
/// launch holds it from its range's push to its pop.
static ARMED: Mutex<Option<Armed>> = Mutex::new(None);

/// Whether [`ARMED`] holds a profiler, read by every launch callback before
/// it takes the lock.
static ARMED_FLAG: AtomicBool = AtomicBool::new(false);

thread_local! {
    /// The range this thread pushed as it entered a launch call: the launch
    /// lock it holds until the call returns, and the call's correlation id,
    /// which pairs the entry with its exit.
    static OPEN: RefCell<Option<(MutexGuard<'static, Option<Armed>>, u32)>> =
        const { RefCell::new(None) };
}

/// Push the range `name` on the profiler `object`.
fn push_range(object: *mut c_void, name: &CStr) -> Result<()> {
    let cupti = cupti()?;
    let mut params = PushParams {
        struct_size: struct_size!(PushParams, name),
        private: std::ptr::null_mut(),
        object,
        name: name.as_ptr(),
    };
    // SAFETY: the struct is CUPTI's parameter struct; CUPTI copies the name.
    unsafe {
        call(
            &cupti.range_push,
            "cuptiRangeProfilerPushRange",
            &mut params,
        )
    }
}

/// Pop the innermost range of the profiler `object`.
fn pop_range(object: *mut c_void) -> Result<()> {
    let cupti = cupti()?;
    let mut params = ObjectParams {
        struct_size: struct_size!(ObjectParams, object),
        private: std::ptr::null_mut(),
        object,
    };
    // SAFETY: the struct is CUPTI's parameter struct.
    unsafe { call(&cupti.range_pop, "cuptiRangeProfilerPopRange", &mut params) }
}

/// The dispatch's part: push a range as a launch's driver call on the armed
/// profiler's context is entered, and pop it as the call returns. Nothing
/// is ranged while no launch profiler is started, nor a launch the calling
/// thread makes inside another launch call. A push or pop CUPTI refuses is
/// kept for the profiler's next stop, which reports it.
///
/// The worker captures its CUDA graphs at warmup, before its world serves,
/// and a session starts the profiler only around served steps, so no ranged
/// launch is one a capture records.
///
/// # Safety
/// `data` is the `CUpti_CallbackData` of a driver API callback.
pub(super) unsafe fn on_launch(cbid: u32, data: *const c_void) {
    if !LAUNCHES.iter().any(|&(launch, _)| launch == cbid) {
        return;
    }
    let data = data.cast::<u8>();
    // SAFETY: `CUpti_CallbackData`: the site at 0, the call's context at 40,
    // its correlation id at 64.
    let (site, context, correlation) = unsafe {
        (
            data.cast::<i32>().read_unaligned(),
            data.add(40).cast::<usize>().read_unaligned(),
            data.add(64).cast::<u32>().read_unaligned(),
        )
    };
    OPEN.with(|open| {
        let mut open = open.borrow_mut();
        if site != API_ENTER {
            // The exit of the call whose entry pushed the open range.
            if open
                .as_ref()
                .is_some_and(|(_, pushed)| *pushed == correlation)
                && let Some((mut armed, _)) = open.take()
                && let Some(armed) = armed.as_mut()
                && let Err(error) = pop_range(armed.object as *mut c_void)
            {
                armed.failure.get_or_insert(error);
            }
            return;
        }
        if open.is_some() || !ARMED_FLAG.load(Ordering::Acquire) {
            return;
        }
        let mut guard = lock(&ARMED);
        let Some(armed) = guard.as_mut().filter(|armed| armed.context == context) else {
            return;
        };
        let name = CString::new(armed.launches.to_string()).unwrap_or_default();
        armed.launches += 1;
        match push_range(armed.object as *mut c_void, &name) {
            Ok(()) => *open = Some((guard, correlation)),
            Err(error) => {
                armed.failure.get_or_insert(error);
            }
        }
    });
}

#[repr(C)]
struct EnableParams {
    struct_size: usize,
    private: *mut c_void,
    context: *mut c_void,
    object: *mut c_void,
}

#[repr(C)]
struct ObjectParams {
    struct_size: usize,
    private: *mut c_void,
    object: *mut c_void,
}

#[repr(C)]
struct SetConfigParams {
    struct_size: usize,
    private: *mut c_void,
    object: *mut c_void,
    config_size: usize,
    config: *const u8,
    data_size: usize,
    data: *mut u8,
    range: u32,
    replay: u32,
    max_ranges_per_pass: usize,
    nesting_levels: u16,
    min_nesting_level: u16,
    pass_index: usize,
    target_nesting_level: u16,
}

#[repr(C)]
struct StopParams {
    struct_size: usize,
    private: *mut c_void,
    object: *mut c_void,
    pass_index: usize,
    target_nesting_level: usize,
    all_passes_submitted: u8,
}

#[repr(C)]
struct PushParams {
    struct_size: usize,
    private: *mut c_void,
    object: *mut c_void,
    name: *const c_char,
}

#[repr(C)]
struct DecodeParams {
    struct_size: usize,
    private: *mut c_void,
    object: *mut c_void,
    dropped: usize,
}

#[repr(C)]
struct DataSizeParams {
    struct_size: usize,
    private: *mut c_void,
    object: *mut c_void,
    names: *const *const c_char,
    count: usize,
    max_ranges: usize,
    max_tree_nodes: u32,
    size: usize,
}

#[repr(C)]
struct DataInitializeParams {
    struct_size: usize,
    private: *mut c_void,
    object: *mut c_void,
    size: usize,
    data: *mut u8,
}

#[repr(C)]
struct DataInfoParams {
    struct_size: usize,
    private: *mut c_void,
    data: *const u8,
    size: usize,
    ranges: usize,
}

#[repr(C)]
struct RangeInfoParams {
    struct_size: usize,
    private: *mut c_void,
    data: *const u8,
    size: usize,
    index: usize,
    delimiter: *const c_char,
    name: *const c_char,
}

/// One collected range: its name (its launch's index in the pass for
/// [`Ranges::Launches`]) and the metrics' values, in configuration order.
#[derive(Clone, Debug, PartialEq)]
pub struct Range {
    pub name: String,
    pub values: Vec<f64>,
}

/// What one decode yields: the ranges, and those dropped for lack of room in
/// the counter data image.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Collected {
    pub ranges: Vec<Range>,
    pub dropped: usize,
}

/// The range profiler of one context; disabled on drop.
pub struct RangeProfiler {
    object: *mut c_void,
    /// The context (`CUcontext`) it profiles.
    context: usize,
    counters: Counters,
    image: Vec<u8>,
    ranges: Ranges,
    max_ranges: usize,
    started: bool,
    /// Whether this profiler enabled the launch callbacks
    /// ([`Ranges::Launches`]), which its drop releases.
    callbacks: bool,
}

/// Enable the callbacks of every launch on the process's subscription.
fn enable_launch_callbacks() -> Result<()> {
    subscriber::subscribe()?;
    for (index, &(cbid, _)) in LAUNCHES.iter().enumerate() {
        if let Err(error) = subscriber::enable_callback(DRIVER_DOMAIN, cbid) {
            for &(enabled, _) in &LAUNCHES[..index] {
                subscriber::disable_callback(DRIVER_DOMAIN, enabled);
            }
            return Err(error);
        }
    }
    Ok(())
}

// SAFETY: the profiler object is CUPTI's, used only through this value;
// CUPTI accepts its calls from any thread.
unsafe impl Send for RangeProfiler {}

impl RangeProfiler {
    /// Enable the range profiler on `context` (a `CUcontext` of the device of
    /// ordinal `device`) for `metrics`, delimited as `ranges` says, holding up
    /// to `max_ranges` ranges between two stops. [`Ranges::Launches`] also
    /// enables the launch callbacks on the process's CUPTI subscription.
    ///
    /// # Errors
    ///
    /// Fails when CUPTI lacks the range profiler, a metric is unknown, the
    /// metrics need more than one pass, another profiler holds the
    /// context's counters, or launch ranges find no CUPTI subscription.
    pub fn enable(
        context: usize,
        device: usize,
        metrics: &[String],
        ranges: Ranges,
        max_ranges: usize,
    ) -> Result<Self> {
        let counters = Counters::configure(
            Profiler::Range,
            device,
            Availability::Context(context),
            metrics,
        )?
        .single_pass()?;
        let cupti = cupti()?;
        let mut enable = EnableParams {
            struct_size: struct_size!(EnableParams, object),
            private: std::ptr::null_mut(),
            context: context as *mut c_void,
            object: std::ptr::null_mut(),
        };
        // SAFETY: the struct is CUPTI's parameter struct; CUPTI writes the
        // object.
        unsafe { call(&cupti.range_enable, "cuptiRangeProfilerEnable", &mut enable)? };
        // From here the drop disables the profiler.
        let mut profiler = Self {
            object: enable.object,
            context,
            counters,
            image: Vec::new(),
            ranges,
            max_ranges: max_ranges.max(1),
            started: false,
            callbacks: false,
        };
        if ranges == Ranges::Launches {
            enable_launch_callbacks()?;
            profiler.callbacks = true;
        }
        let mut size = DataSizeParams {
            struct_size: struct_size!(DataSizeParams, size),
            private: std::ptr::null_mut(),
            object: profiler.object,
            names: profiler.counters.name_pointers().as_ptr(),
            count: profiler.counters.name_pointers().len(),
            max_ranges: profiler.max_ranges,
            max_tree_nodes: u32::try_from(profiler.max_ranges).unwrap_or(u32::MAX),
            size: 0,
        };
        // SAFETY: as above; the names outlive the call.
        unsafe {
            call(
                &cupti.range_data_size,
                "cuptiRangeProfilerGetCounterDataSize",
                &mut size,
            )?;
        }
        profiler.image = vec![0; size.size];
        profiler.prepare()?;
        Ok(profiler)
    }

    /// Initialize the counter data image and set the configuration, before
    /// each start.
    fn prepare(&mut self) -> Result<()> {
        let cupti = cupti()?;
        let mut initialize = DataInitializeParams {
            struct_size: struct_size!(DataInitializeParams, data),
            private: std::ptr::null_mut(),
            object: self.object,
            size: self.image.len(),
            data: self.image.as_mut_ptr(),
        };
        // SAFETY: the struct is CUPTI's parameter struct; the image has the
        // size CUPTI stated.
        unsafe {
            call(
                &cupti.range_data_initialize,
                "cuptiRangeProfilerCounterDataImageInitialize",
                &mut initialize,
            )?;
        }
        let config = self.counters.config();
        let mut set = SetConfigParams {
            struct_size: struct_size!(SetConfigParams, target_nesting_level),
            private: std::ptr::null_mut(),
            object: self.object,
            config_size: config.len(),
            config: config.as_ptr(),
            data_size: self.image.len(),
            data: self.image.as_mut_ptr(),
            range: USER_RANGE,
            replay: USER_REPLAY,
            max_ranges_per_pass: self.max_ranges,
            nesting_levels: 1,
            min_nesting_level: 1,
            pass_index: 0,
            target_nesting_level: 0,
        };
        // SAFETY: as above; the configuration and the image outlive the
        // profiler's use of them (this value owns both).
        unsafe {
            call(
                &cupti.range_set_config,
                "cuptiRangeProfilerSetConfig",
                &mut set,
            )
        }
    }

    /// The metric names, in the order of every range's values.
    pub fn metrics(&self) -> Vec<String> {
        self.counters.names()
    }

    /// Start collecting ranges; [`Ranges::Launches`] ranges every launch on
    /// the context from here, the first named `0`.
    ///
    /// # Errors
    ///
    /// Fails when CUPTI refuses the start, or another launch profiler of the
    /// process is started.
    pub fn start(&mut self) -> Result<()> {
        if self.started {
            return Ok(());
        }
        let cupti = cupti()?;
        // The launch lock is held through the start, so the first launch
        // ranged is one entered after it.
        let mut armed = (self.ranges == Ranges::Launches).then(|| lock(&ARMED));
        if armed.as_ref().is_some_and(|armed| armed.is_some()) {
            return Err(Error::InvalidArgument(
                "another range profiler of the process ranges its launches".into(),
            ));
        }
        let mut params = self.object_params();
        // SAFETY: the struct is CUPTI's parameter struct.
        unsafe { call(&cupti.range_start, "cuptiRangeProfilerStart", &mut params)? };
        if let Some(armed) = armed.as_mut() {
            **armed = Some(Armed {
                object: self.object as usize,
                context: self.context,
                launches: 0,
                failure: None,
            });
            ARMED_FLAG.store(true, Ordering::Release);
        }
        self.started = true;
        Ok(())
    }

    /// Push a user range named `name` ([`Ranges::User`]).
    pub fn push(&self, name: &str) -> Result<()> {
        let name = CString::new(name).map_err(|error| Error::InvalidArgument(error.to_string()))?;
        push_range(self.object, &name)
    }

    /// Pop the innermost user range ([`Ranges::User`]).
    pub fn pop(&self) -> Result<()> {
        pop_range(self.object)
    }

    /// Stop collecting, decode the ranges collected since the start and
    /// prepare the image for the next start.
    ///
    /// # Errors
    ///
    /// Fails when CUPTI cannot stop or decode, or refused a launch range's
    /// push or pop during the pass; the ranges are lost then.
    pub fn stop(&mut self) -> Result<Collected> {
        if !self.started {
            return Ok(Collected::default());
        }
        self.started = false;
        // Withdrawn under the launch lock: no launch range is open once it is
        // taken, and a launch entered afterwards pushes none.
        let failure = if self.ranges == Ranges::Launches {
            let mut armed = lock(&ARMED);
            ARMED_FLAG.store(false, Ordering::Release);
            armed.take().and_then(|armed| armed.failure)
        } else {
            None
        };
        let cupti = cupti()?;
        let mut stop = StopParams {
            struct_size: struct_size!(StopParams, all_passes_submitted),
            private: std::ptr::null_mut(),
            object: self.object,
            pass_index: 0,
            target_nesting_level: 0,
            all_passes_submitted: 0,
        };
        // SAFETY: the struct is CUPTI's parameter struct.
        unsafe { call(&cupti.range_stop, "cuptiRangeProfilerStop", &mut stop)? };
        if let Some(failure) = failure {
            return Err(failure);
        }
        let mut decode = DecodeParams {
            struct_size: struct_size!(DecodeParams, dropped),
            private: std::ptr::null_mut(),
            object: self.object,
            dropped: 0,
        };
        // SAFETY: as above.
        unsafe {
            call(
                &cupti.range_decode,
                "cuptiRangeProfilerDecodeData",
                &mut decode,
            )?;
        }
        let mut info = DataInfoParams {
            struct_size: struct_size!(DataInfoParams, ranges),
            private: std::ptr::null_mut(),
            data: self.image.as_ptr(),
            size: self.image.len(),
            ranges: 0,
        };
        // SAFETY: as above.
        unsafe {
            call(
                &cupti.range_data_info,
                "cuptiRangeProfilerGetCounterDataInfo",
                &mut info,
            )?;
        }
        let mut collected = Collected {
            ranges: Vec::with_capacity(info.ranges),
            dropped: decode.dropped,
        };
        for index in 0..info.ranges {
            let mut range = RangeInfoParams {
                struct_size: struct_size!(RangeInfoParams, name),
                private: std::ptr::null_mut(),
                data: self.image.as_ptr(),
                size: self.image.len(),
                index,
                delimiter: c"/".as_ptr(),
                name: std::ptr::null(),
            };
            // SAFETY: as above; CUPTI writes a name it owns until the next
            // call.
            unsafe {
                call(
                    &cupti.range_info,
                    "cuptiRangeProfilerCounterDataGetRangeInfo",
                    &mut range,
                )?;
            }
            // SAFETY: as above.
            let name = unsafe { super::text(range.name) }.unwrap_or_default();
            let values = self.counters.evaluate(&self.image, index)?;
            collected.ranges.push(Range { name, values });
        }
        self.prepare()?;
        Ok(collected)
    }

    fn object_params(&self) -> ObjectParams {
        ObjectParams {
            struct_size: struct_size!(ObjectParams, object),
            private: std::ptr::null_mut(),
            object: self.object,
        }
    }
}

impl Drop for RangeProfiler {
    fn drop(&mut self) {
        let _ = self.stop();
        if let Ok(cupti) = cupti() {
            let mut params = self.object_params();
            // SAFETY: the object is live until this call.
            let _ = unsafe {
                call(
                    &cupti.range_disable,
                    "cuptiRangeProfilerDisable",
                    &mut params,
                )
            };
        }
        if self.callbacks {
            for &(cbid, _) in LAUNCHES {
                subscriber::disable_callback(DRIVER_DOMAIN, cbid);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parameter structs state the sizes the CUPTI 13 headers'
    /// `CUPTI_PROFILER_STRUCT_SIZE` gives them.
    #[test]
    fn parameter_structs_state_the_sizes_of_the_cupti_headers() {
        assert_eq!(struct_size!(EnableParams, object), 32);
        assert_eq!(struct_size!(ObjectParams, object), 24);
        assert_eq!(struct_size!(SetConfigParams, target_nesting_level), 90);
        assert_eq!(struct_size!(StopParams, all_passes_submitted), 41);
        assert_eq!(struct_size!(PushParams, name), 32);
        assert_eq!(struct_size!(DecodeParams, dropped), 32);
        assert_eq!(struct_size!(DataSizeParams, size), 64);
        assert_eq!(struct_size!(DataInitializeParams, data), 40);
        assert_eq!(struct_size!(DataInfoParams, ranges), 40);
        assert_eq!(struct_size!(RangeInfoParams, name), 56);
    }

    /// The metrics of the launch tests: each range's device time and the
    /// warp instructions its kernels executed, which the spin kernel makes
    /// exact for a given launch.
    const METRICS: [&str; 2] = ["gpu__time_duration.sum", "smsp__inst_executed.sum"];

    fn names(collected: &Collected) -> Vec<&str> {
        collected
            .ranges
            .iter()
            .map(|range| range.name.as_str())
            .collect()
    }

    fn instructions(collected: &Collected) -> Vec<f64> {
        collected
            .ranges
            .iter()
            .map(|range| range.values[1])
            .collect()
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn launch_ranges_take_each_launch_and_each_graph_launch_whole() -> Result<()> {
        use crate::cupti::tests::{SERIAL, Spin, with_captured_graph};
        use crate::{DeviceId, GraphExec, PrimaryContext, Stream, lock};

        let _serial = lock(&SERIAL);
        let device = DeviceId(0);
        let context = PrimaryContext::retain(device)?;
        let stream = Stream::new(device)?;
        let spin = Spin::load(&stream)?;
        let metrics = METRICS.map(str::to_owned);

        // The graph is instantiated before the profiler is enabled, as a
        // worker instantiates its graphs at warmup, before any session.
        let (eager, graphed) = with_captured_graph(
            &stream,
            |stream| (0..2).try_for_each(|_| spin.launch(stream, 4, 100_000)),
            |source| {
                // SAFETY: with_captured_graph keeps the source live, and this
                // test is its sole owner and submitter.
                let graph = unsafe { GraphExec::instantiate(device, source)? };
                let mut launches =
                    RangeProfiler::enable(context.handle(), 0, &metrics, Ranges::Launches, 16)?;
                launches.start()?;
                for _ in 0..3 {
                    spin.launch(&stream, 4, 100_000)?;
                }
                stream.synchronize()?;
                let eager = launches.stop()?;

                // The next pass, after the stop prepared the image again,
                // names its launches from 0 again.
                launches.start()?;
                graph.launch(&stream)?;
                spin.launch(&stream, 4, 100_000)?;
                stream.synchronize()?;
                let graphed = launches.stop()?;
                Ok((eager, graphed))
            },
        )?;
        assert_eq!(names(&eager), ["0", "1", "2"], "{eager:?}");
        assert!(
            eager
                .ranges
                .iter()
                .all(|range| range.values.iter().all(|&value| value > 0.0)),
            "{eager:?}"
        );
        // The graph's launch is one range holding both its kernels.
        assert_eq!(names(&graphed), ["0", "1"], "{graphed:?}");
        let [graph, kernel] = instructions(&graphed)[..] else {
            panic!("two ranges, not {graphed:?}");
        };
        assert!(kernel > 0.0, "{graphed:?}");
        assert_eq!(graph, 2.0 * kernel, "{graphed:?}");
        Ok(())
    }

    /// A launch range holds its own launch's work only, while another
    /// stream's kernel runs beside it: each range executes as many
    /// instructions as the same launch with nothing beside it.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn a_launch_range_holds_only_its_launch_beside_another_stream() -> Result<()> {
        use crate::cupti::tests::{SERIAL, Spin};
        use crate::{DeviceId, PrimaryContext, Stream, lock};

        let _serial = lock(&SERIAL);
        let device = DeviceId(0);
        let context = PrimaryContext::retain(device)?;
        let long = Stream::new(device)?;
        let short = Stream::new(device)?;
        let spin = Spin::load(&long)?;
        let metrics = METRICS.map(str::to_owned);
        let mut launches =
            RangeProfiler::enable(context.handle(), 0, &metrics, Ranges::Launches, 16)?;

        // Each kernel alone.
        launches.start()?;
        spin.launch(&long, 4, 20_000_000)?;
        long.synchronize()?;
        spin.launch(&short, 4, 100_000)?;
        short.synchronize()?;
        let alone = launches.stop()?;

        // The short kernel launched while the long one runs.
        launches.start()?;
        spin.launch(&long, 4, 20_000_000)?;
        spin.launch(&short, 4, 100_000)?;
        long.synchronize()?;
        short.synchronize()?;
        let beside = launches.stop()?;

        assert_eq!(names(&alone), ["0", "1"], "{alone:?}");
        assert_eq!(names(&beside), ["0", "1"], "{beside:?}");
        assert!(instructions(&alone).iter().all(|&count| count > 0.0));
        assert_eq!(instructions(&beside), instructions(&alone), "{beside:?}");
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn user_ranges_measure_what_ran_in_them() -> Result<()> {
        use crate::cupti::tests::{SERIAL, Spin};
        use crate::{DeviceId, PrimaryContext, Stream, lock};

        let _serial = lock(&SERIAL);
        let device = DeviceId(0);
        let context = PrimaryContext::retain(device)?;
        let stream = Stream::new(device)?;
        let spin = Spin::load(&stream)?;
        let metrics = ["gpu__time_duration.sum".to_owned()];

        let mut user = RangeProfiler::enable(context.handle(), 0, &metrics, Ranges::User, 16)?;
        user.start()?;
        user.push("two")?;
        spin.launch(&stream, 4, 100_000)?;
        spin.launch(&stream, 4, 100_000)?;
        user.pop()?;
        stream.synchronize()?;
        let collected = user.stop()?;
        let [range] = &collected.ranges[..] else {
            panic!("one user range, not {collected:?}");
        };
        assert_eq!(range.name, "two");
        assert!(range.values[0] > 0.0);
        Ok(())
    }

    /// The crate's address reservations leave a pass the addresses it takes
    /// its device storage at, however much they reserve.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn a_pass_starts_beside_two_tib_of_reserved_addresses() -> Result<()> {
        use crate::cupti::tests::SERIAL;
        use crate::{DeviceId, PrimaryContext, VirtualPool, lock};

        let _serial = lock(&SERIAL);
        let device = DeviceId(0);
        let context = PrimaryContext::retain(device)?;
        let pools = (0..8)
            .map(|_| VirtualPool::reserve(device, 1 << 38))
            .collect::<Result<Vec<_>>>()?;
        let metrics = ["gpu__time_duration.sum".to_owned()];
        // 2048 ranges take 2.9 GB of device storage.
        let mut profiler =
            RangeProfiler::enable(context.handle(), 0, &metrics, Ranges::Launches, 2048)?;
        profiler.start()?;
        profiler.stop()?;
        drop(profiler);
        drop(pools);
        Ok(())
    }
}
