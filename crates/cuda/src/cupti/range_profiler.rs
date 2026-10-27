//! Hardware counters per kernel or per user range, collected in one pass
//! (CUPTI's range profiler, `cupti_range_profiler.h`).
//!
//! A [`RangeProfiler`] is enabled on one context with a single-pass counter
//! configuration ([`super::metrics::Counters::single_pass`]) and user replay
//! of that one pass, so no work is ever replayed. Between [`RangeProfiler::start`]
//! and [`RangeProfiler::stop`] it collects:
//!
//! - [`Ranges::Kernels`]: a range around each launch on the context, which
//!   CUPTI names by its index; a CUDA graph's launch is one range.
//! - [`Ranges::User`]: the ranges the caller pushes and pops
//!   ([`RangeProfiler::push`], [`RangeProfiler::pop`]), which CUPTI orders on
//!   the device with the work issued between them: a range pushed before a
//!   graph's launch and popped after it measures the whole graph, though the
//!   pop returns before the graph ran. A range is pushed on the host, so a
//!   CUDA graph's replay pushes none of the ranges its capture saw.
//!
//! The counter data image holds a bounded number of ranges;
//! [`RangeProfiler::stop`] decodes it into [`Range`] values and prepares it
//! for the next start, so a long collection stops and starts again before
//! the image fills.

use std::ffi::{CString, c_char, c_void};

use super::metrics::{Availability, Counters, Profiler};
use super::{call, cupti, struct_size};
use crate::{Error, Result};

/// How ranges are delimited (`CUpti_ProfilerRange`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ranges {
    /// A range per kernel or graph launch.
    Kernels = 1,
    /// The caller's pushed ranges.
    User = 2,
}

/// `CUPTI_UserReplay`: the caller replays, which a one-pass configuration
/// never needs.
const USER_REPLAY: u32 = 3;

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

/// One collected range: its name (its index among the kernel launches for
/// [`Ranges::Kernels`]) and the metrics' values, in configuration order.
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
    counters: Counters,
    image: Vec<u8>,
    ranges: Ranges,
    max_ranges: usize,
    started: bool,
}

// SAFETY: the profiler object is CUPTI's, used only through this value;
// CUPTI accepts its calls from any thread.
unsafe impl Send for RangeProfiler {}

impl RangeProfiler {
    /// Enable the range profiler on `context` (a `CUcontext` of the device of
    /// ordinal `device`) for `metrics`, delimited as `ranges` says, holding up
    /// to `max_ranges` ranges between two stops.
    ///
    /// # Errors
    ///
    /// Fails when CUPTI lacks the range profiler, a metric is unknown, the
    /// metrics need more than one pass, or another profiler holds the
    /// context's counters.
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
        let mut profiler = Self {
            object: enable.object,
            counters,
            image: Vec::new(),
            ranges,
            max_ranges: max_ranges.max(1),
            started: false,
        };
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
            range: self.ranges as u32,
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

    /// Start collecting ranges.
    pub fn start(&mut self) -> Result<()> {
        if self.started {
            return Ok(());
        }
        let cupti = cupti()?;
        let mut params = self.object_params();
        // SAFETY: the struct is CUPTI's parameter struct.
        unsafe { call(&cupti.range_start, "cuptiRangeProfilerStart", &mut params)? };
        self.started = true;
        Ok(())
    }

    /// Push a user range named `name`.
    pub fn push(&self, name: &str) -> Result<()> {
        let cupti = cupti()?;
        let name = CString::new(name).map_err(|error| Error::InvalidArgument(error.to_string()))?;
        let mut params = PushParams {
            struct_size: struct_size!(PushParams, name),
            private: std::ptr::null_mut(),
            object: self.object,
            name: name.as_ptr(),
        };
        // SAFETY: the struct is CUPTI's parameter struct; CUPTI copies the
        // name.
        unsafe {
            call(
                &cupti.range_push,
                "cuptiRangeProfilerPushRange",
                &mut params,
            )
        }
    }

    /// Pop the innermost user range.
    pub fn pop(&self) -> Result<()> {
        let cupti = cupti()?;
        let mut params = self.object_params();
        // SAFETY: the struct is CUPTI's parameter struct.
        unsafe { call(&cupti.range_pop, "cuptiRangeProfilerPopRange", &mut params) }
    }

    /// Stop collecting, decode the ranges collected since the start and
    /// prepare the image for the next start.
    ///
    /// # Errors
    ///
    /// Fails when CUPTI cannot stop or decode; the ranges are lost then.
    pub fn stop(&mut self) -> Result<Collected> {
        if !self.started {
            return Ok(Collected::default());
        }
        self.started = false;
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

    #[test]
    #[ignore = "requires a CUDA device"]
    fn kernel_ranges_and_user_ranges_measure_what_ran_in_them() -> Result<()> {
        use crate::cupti::tests::{SERIAL, Spin};
        use crate::{DeviceId, PrimaryContext, Stream, lock};

        let _serial = lock(&SERIAL);
        let device = DeviceId(0);
        let context = PrimaryContext::retain(device)?;
        let stream = Stream::new(device)?;
        let spin = Spin::load(&stream)?;
        let metrics = ["gpu__time_duration.sum".to_owned()];

        let mut kernels =
            RangeProfiler::enable(context.raw() as usize, 0, &metrics, Ranges::Kernels, 16)?;
        kernels.start()?;
        for _ in 0..3 {
            spin.launch(&stream, 4, 100_000)?;
        }
        stream.synchronize()?;
        let collected = kernels.stop()?;
        assert_eq!(collected.ranges.len(), 3, "{collected:?}");
        assert!(collected.ranges.iter().all(|range| range.values[0] > 0.0));
        drop(kernels);

        let mut user =
            RangeProfiler::enable(context.raw() as usize, 0, &metrics, Ranges::User, 16)?;
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
}
