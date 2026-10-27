//! Hardware counter configurations and their evaluation, shared by the range
//! profiler ([`super::range_profiler`]) and PM sampling
//! ([`super::pm_sampling`]).
//!
//! A [`Counters`] configuration names metrics in NVIDIA's performance
//! library's terms (the names Nsight Compute reports, such as
//! `gpu__time_duration.sum`), resolved for the device's chip and the counters
//! its context or device has available. CUPTI compiles them into a
//! configuration image and states how many passes the hardware needs to
//! collect them all. Online collection never replays work, so a
//! configuration is accepted only when one pass collects it
//! ([`Counters::single_pass`]). The values of a collected range or sample are
//! evaluated from the counter data image on the host.

use std::ffi::{CString, c_char, c_void};
use std::sync::OnceLock;

use super::{call, cupti, struct_size};
use crate::{Error, Result};

/// The profiler a configuration serves (`CUpti_ProfilerType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Profiler {
    /// Counters per kernel or per user range.
    Range = 0,
    /// Counters sampled at a fixed interval.
    PmSampling = 1,
}

#[repr(C)]
struct InitializeParams {
    struct_size: usize,
    private: *mut c_void,
}

#[repr(C)]
struct ChipNameParams {
    struct_size: usize,
    private: *mut c_void,
    device_index: usize,
    chip_name: *const c_char,
}

#[repr(C)]
pub(super) struct RangeAvailabilityParams {
    struct_size: usize,
    private: *mut c_void,
    context: *mut c_void,
    image_size: usize,
    image: *mut u8,
}

#[repr(C)]
pub(super) struct PmAvailabilityParams {
    struct_size: usize,
    private: *mut c_void,
    device_index: usize,
    image_size: usize,
    image: *mut u8,
}

#[repr(C)]
struct HostInitializeParams {
    struct_size: usize,
    private: *mut c_void,
    profiler: u32,
    chip_name: *const c_char,
    availability: *const u8,
    host: *mut c_void,
}

#[repr(C)]
struct HostDeinitializeParams {
    struct_size: usize,
    private: *mut c_void,
    host: *mut c_void,
}

#[repr(C)]
struct AddMetricsParams {
    struct_size: usize,
    private: *mut c_void,
    host: *mut c_void,
    names: *const *const c_char,
    count: usize,
}

#[repr(C)]
struct ConfigSizeParams {
    struct_size: usize,
    private: *mut c_void,
    host: *mut c_void,
    size: usize,
}

#[repr(C)]
struct ConfigParams {
    struct_size: usize,
    private: *mut c_void,
    host: *mut c_void,
    size: usize,
    image: *mut u8,
}

#[repr(C)]
struct PassesParams {
    struct_size: usize,
    private: *mut c_void,
    size: usize,
    image: *mut u8,
    passes: usize,
}

#[repr(C)]
struct EvaluateParams {
    struct_size: usize,
    private: *mut c_void,
    host: *mut c_void,
    data: *const u8,
    data_size: usize,
    range: usize,
    names: *const *const c_char,
    count: usize,
    values: *mut f64,
}

/// Initialize CUPTI's profiling, once per process.
pub(super) fn initialize() -> Result<()> {
    static INITIALIZED: OnceLock<Result<()>> = OnceLock::new();
    INITIALIZED
        .get_or_init(|| {
            let cupti = cupti()?;
            let mut params = InitializeParams {
                struct_size: struct_size!(InitializeParams, private),
                private: std::ptr::null_mut(),
            };
            // SAFETY: the struct is CUPTI's parameter struct.
            unsafe {
                call(
                    &cupti.profiler_initialize,
                    "cuptiProfilerInitialize",
                    &mut params,
                )
            }
        })
        .clone()
}

/// The chip name of the CUDA device of ordinal `device`, as the performance
/// library names it (`GB100` for a GB200's GPU).
pub fn chip_name(device: usize) -> Result<String> {
    initialize()?;
    let cupti = cupti()?;
    let mut params = ChipNameParams {
        struct_size: struct_size!(ChipNameParams, chip_name),
        private: std::ptr::null_mut(),
        device_index: device,
        chip_name: std::ptr::null(),
    };
    // SAFETY: the struct is CUPTI's parameter struct; CUPTI writes the name.
    unsafe { call(&cupti.chip_name, "cuptiDeviceGetChipName", &mut params)? };
    // SAFETY: CUPTI returns a static string.
    unsafe { super::text(params.chip_name) }
        .ok_or_else(|| Error::InvalidArgument("CUPTI named no chip".into()))
}

/// Where a configuration's counter availability comes from.
#[derive(Clone, Copy, Debug)]
pub enum Availability {
    /// The counters of a context (`CUcontext`), for the range profiler.
    Context(usize),
    /// The counters of the device of an ordinal, for PM sampling.
    Device(usize),
}

/// The counter availability image of `availability`.
fn availability_image(availability: Availability) -> Result<Vec<u8>> {
    let cupti = cupti()?;
    match availability {
        Availability::Context(context) => {
            let mut params = RangeAvailabilityParams {
                struct_size: struct_size!(RangeAvailabilityParams, image),
                private: std::ptr::null_mut(),
                context: context as *mut c_void,
                image_size: 0,
                image: std::ptr::null_mut(),
            };
            // SAFETY: the struct is CUPTI's parameter struct; a null image
            // asks for its size.
            unsafe {
                call(
                    &cupti.counter_availability,
                    "cuptiProfilerGetCounterAvailability",
                    &mut params,
                )?;
            }
            let mut image = vec![0; params.image_size];
            params.image = image.as_mut_ptr();
            // SAFETY: as above; the image holds the size CUPTI stated.
            unsafe {
                call(
                    &cupti.counter_availability,
                    "cuptiProfilerGetCounterAvailability",
                    &mut params,
                )?;
            }
            Ok(image)
        }
        Availability::Device(device) => {
            let mut params = PmAvailabilityParams {
                struct_size: struct_size!(PmAvailabilityParams, image),
                private: std::ptr::null_mut(),
                device_index: device,
                image_size: 0,
                image: std::ptr::null_mut(),
            };
            // SAFETY: as above.
            unsafe {
                call(
                    &cupti.pm_availability,
                    "cuptiPmSamplingGetCounterAvailability",
                    &mut params,
                )?;
            }
            let mut image = vec![0; params.image_size];
            params.image = image.as_mut_ptr();
            // SAFETY: as above.
            unsafe {
                call(
                    &cupti.pm_availability,
                    "cuptiPmSamplingGetCounterAvailability",
                    &mut params,
                )?;
            }
            Ok(image)
        }
    }
}

/// A compiled counter configuration: its metrics, its configuration image
/// and the host object that evaluates its counter data.
pub struct Counters {
    host: *mut c_void,
    metrics: Vec<CString>,
    pointers: Vec<*const c_char>,
    config: Vec<u8>,
    passes: usize,
}

// SAFETY: the host object is CUPTI's, used only through this value, whose
// methods CUPTI accepts from any thread.
unsafe impl Send for Counters {}

impl Counters {
    /// Compile `metrics` for `profiler` on the device of ordinal `device`,
    /// against the counters `availability` reports.
    ///
    /// # Errors
    ///
    /// Fails when CUPTI lacks the profiling API, a metric is unknown or
    /// unavailable, or the configuration does not compile.
    pub fn configure(
        profiler: Profiler,
        device: usize,
        availability: Availability,
        metrics: &[String],
    ) -> Result<Self> {
        if metrics.is_empty() {
            return Err(Error::InvalidArgument("no metric is named".into()));
        }
        initialize()?;
        let cupti = cupti()?;
        let chip = CString::new(chip_name(device)?)
            .map_err(|error| Error::InvalidArgument(error.to_string()))?;
        let image = availability_image(availability)?;
        let metrics = metrics
            .iter()
            .map(|name| {
                CString::new(name.as_str())
                    .map_err(|error| Error::InvalidArgument(error.to_string()))
            })
            .collect::<Result<Vec<_>>>()?;
        let pointers: Vec<*const c_char> = metrics.iter().map(|name| name.as_ptr()).collect();

        let mut init = HostInitializeParams {
            struct_size: struct_size!(HostInitializeParams, host),
            private: std::ptr::null_mut(),
            profiler: profiler as u32,
            chip_name: chip.as_ptr(),
            availability: image.as_ptr(),
            host: std::ptr::null_mut(),
        };
        // SAFETY: the struct is CUPTI's parameter struct; CUPTI copies the
        // chip name and availability image it reads.
        unsafe {
            call(
                &cupti.host_initialize,
                "cuptiProfilerHostInitialize",
                &mut init,
            )?
        };
        // From here the drop releases the host object.
        let mut counters = Self {
            host: init.host,
            metrics,
            pointers,
            config: Vec::new(),
            passes: 0,
        };

        let mut add = AddMetricsParams {
            struct_size: struct_size!(AddMetricsParams, count),
            private: std::ptr::null_mut(),
            host: counters.host,
            names: counters.pointers.as_ptr(),
            count: counters.pointers.len(),
        };
        // SAFETY: as above; the names outlive the call.
        unsafe {
            call(
                &cupti.host_add_metrics,
                "cuptiProfilerHostConfigAddMetrics",
                &mut add,
            )?;
        }
        let mut size = ConfigSizeParams {
            struct_size: struct_size!(ConfigSizeParams, size),
            private: std::ptr::null_mut(),
            host: counters.host,
            size: 0,
        };
        // SAFETY: as above.
        unsafe {
            call(
                &cupti.host_config_size,
                "cuptiProfilerHostGetConfigImageSize",
                &mut size,
            )?;
        }
        counters.config = vec![0; size.size];
        let mut config = ConfigParams {
            struct_size: struct_size!(ConfigParams, image),
            private: std::ptr::null_mut(),
            host: counters.host,
            size: counters.config.len(),
            image: counters.config.as_mut_ptr(),
        };
        // SAFETY: as above; the image holds the size CUPTI stated.
        unsafe {
            call(
                &cupti.host_config,
                "cuptiProfilerHostGetConfigImage",
                &mut config,
            )?;
        }
        let mut passes = PassesParams {
            struct_size: struct_size!(PassesParams, passes),
            private: std::ptr::null_mut(),
            size: counters.config.len(),
            image: counters.config.as_mut_ptr(),
            passes: 0,
        };
        // SAFETY: as above.
        unsafe {
            call(
                &cupti.host_passes,
                "cuptiProfilerHostGetNumOfPasses",
                &mut passes,
            )?;
        }
        counters.passes = passes.passes;
        Ok(counters)
    }

    /// How many passes the hardware needs to collect every metric.
    pub fn passes(&self) -> usize {
        self.passes
    }

    /// The configuration, refused unless one pass collects it: online
    /// collection never replays work.
    ///
    /// # Errors
    ///
    /// Fails, naming the passes, when the metrics need more than one.
    pub fn single_pass(self) -> Result<Self> {
        if self.passes == 1 {
            Ok(self)
        } else {
            Err(Error::InvalidArgument(format!(
                "the metrics {} need {} passes; online collection takes metrics one pass \
                 collects, without replay",
                self.names().join(", "),
                self.passes
            )))
        }
    }

    /// The configuration image.
    pub fn config(&self) -> &[u8] {
        &self.config
    }

    /// The metric names, in configuration order.
    pub fn names(&self) -> Vec<String> {
        self.metrics
            .iter()
            .map(|name| name.to_string_lossy().into_owned())
            .collect()
    }

    pub(super) fn name_pointers(&self) -> &[*const c_char] {
        &self.pointers
    }

    /// The metrics' values over range or sample `index` of the counter data
    /// `image`, in configuration order.
    pub fn evaluate(&self, image: &[u8], index: usize) -> Result<Vec<f64>> {
        let cupti = cupti()?;
        let mut values = vec![0.0; self.pointers.len()];
        let mut params = EvaluateParams {
            struct_size: struct_size!(EvaluateParams, values),
            private: std::ptr::null_mut(),
            host: self.host,
            data: image.as_ptr(),
            data_size: image.len(),
            range: index,
            names: self.pointers.as_ptr(),
            count: self.pointers.len(),
            values: values.as_mut_ptr(),
        };
        // SAFETY: the struct is CUPTI's parameter struct; the values hold a
        // value per metric.
        unsafe {
            call(
                &cupti.host_evaluate,
                "cuptiProfilerHostEvaluateToGpuValues",
                &mut params,
            )?;
        }
        Ok(values)
    }
}

impl Drop for Counters {
    fn drop(&mut self) {
        if let Ok(cupti) = cupti() {
            let mut params = HostDeinitializeParams {
                struct_size: struct_size!(HostDeinitializeParams, host),
                private: std::ptr::null_mut(),
                host: self.host,
            };
            // SAFETY: the host object is live until this call.
            let _ = unsafe {
                call(
                    &cupti.host_deinitialize,
                    "cuptiProfilerHostDeinitialize",
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
    /// `CUPTI_PROFILER_STRUCT_SIZE` gives them; CUPTI reads only the fields
    /// a struct's size covers.
    #[test]
    fn parameter_structs_state_the_sizes_of_the_cupti_headers() {
        assert_eq!(struct_size!(InitializeParams, private), 16);
        assert_eq!(struct_size!(ChipNameParams, chip_name), 32);
        assert_eq!(struct_size!(RangeAvailabilityParams, image), 40);
        assert_eq!(struct_size!(PmAvailabilityParams, image), 40);
        assert_eq!(struct_size!(HostInitializeParams, host), 48);
        assert_eq!(struct_size!(HostDeinitializeParams, host), 24);
        assert_eq!(struct_size!(AddMetricsParams, count), 40);
        assert_eq!(struct_size!(ConfigSizeParams, size), 32);
        assert_eq!(struct_size!(ConfigParams, image), 40);
        assert_eq!(struct_size!(PassesParams, passes), 40);
        assert_eq!(struct_size!(EvaluateParams, values), 72);
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn a_kernel_duration_is_one_pass_and_a_mixed_set_is_refused() -> Result<()> {
        let device = crate::DeviceId(0);
        let context = crate::PrimaryContext::retain(device)?;
        let one = Counters::configure(
            Profiler::Range,
            0,
            Availability::Context(context.raw() as usize),
            &["gpu__time_duration.sum".to_owned()],
        )?;
        assert_eq!(one.passes(), 1);
        assert!(one.single_pass().is_ok());

        let many = Counters::configure(
            Profiler::Range,
            0,
            Availability::Context(context.raw() as usize),
            &[
                "gpu__time_duration.sum".to_owned(),
                "sm__throughput.avg.pct_of_peak_sustained_elapsed".to_owned(),
                "dram__throughput.avg.pct_of_peak_sustained_elapsed".to_owned(),
            ],
        )?;
        assert!(many.passes() > 1);
        let refused = many
            .single_pass()
            .err()
            .expect("a multi-pass set is refused");
        assert!(refused.to_string().contains("passes"), "{refused}");
        Ok(())
    }
}
