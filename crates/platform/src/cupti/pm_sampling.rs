//! Hardware counters sampled at a fixed interval of device time, in one pass
//! (CUPTI's PM sampling, `cupti_pmsampling.h`).
//!
//! A [`PmSampler`] is enabled on one device with a counter configuration
//! ([`super::metrics::Profiler::PmSampling`]), a sampling interval in
//! nanoseconds of the GPU's fixed-frequency trigger and a hardware buffer the
//! device writes raw samples into. Between [`PmSampler::start`] and
//! [`PmSampler::stop`] the device samples whatever runs on it, without
//! regard to streams or kernels. [`PmSampler::decode`], which may run while
//! sampling, moves the samples completed so far from the hardware buffer
//! into the counter data image and evaluates them into [`Sample`]s, each
//! with its interval in CUPTI's clock ([`super::timestamp`]). The hardware
//! buffer must hold the samples taken between two decodes: on GB200 with
//! `nvidia-cuda-cupti` 13.0 a buffer that wraps loses samples and can fail
//! the next decode (`CUPTI_ERROR_UNKNOWN`), whatever its append mode.

use std::ffi::{c_char, c_void};

use super::metrics::{Availability, Counters, Profiler};
use super::{call, cupti, struct_size};
use crate::Result;

/// `CUPTI_PM_SAMPLING_TRIGGER_MODE_GPU_TIME_INTERVAL`: the interval is in
/// nanoseconds of a fixed-frequency source.
const TIME_INTERVAL: u32 = 1;

/// `CUPTI_PM_SAMPLING_HARDWARE_BUFFER_APPEND_MODE_KEEP_LATEST`: a full
/// hardware buffer overwrites its oldest records. Keeping the oldest
/// instead fails every decode once the buffer overflowed
/// (`CUPTI_ERROR_OUT_OF_MEMORY`), losing every later sample.
const KEEP_LATEST: u32 = 1;

/// `CUPTI_PM_SAMPLING_DECODE_STOP_REASON_COUNTER_DATA_FULL`: the decode
/// filled the counter data image before it decoded every record.
const COUNTER_DATA_FULL: u32 = 1;

#[repr(C)]
struct EnableParams {
    struct_size: usize,
    private: *mut c_void,
    device_index: usize,
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
    hardware_buffer: usize,
    interval: u64,
    trigger: u32,
    append: u32,
}

#[repr(C)]
struct DecodeParams {
    struct_size: usize,
    private: *mut c_void,
    object: *mut c_void,
    data: *mut u8,
    size: usize,
    stop_reason: u32,
    overflow: u8,
}

#[repr(C)]
struct DataSizeParams {
    struct_size: usize,
    private: *mut c_void,
    object: *mut c_void,
    names: *const *const c_char,
    count: usize,
    max_samples: u32,
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
    total: usize,
    populated: usize,
    completed: usize,
}

#[repr(C)]
struct SampleInfoParams {
    struct_size: usize,
    private: *mut c_void,
    object: *mut c_void,
    data: *const u8,
    size: usize,
    index: usize,
    start: u64,
    end: u64,
}

/// One sample: its interval and the metrics' values, in configuration order.
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
    pub start: u64,
    pub end: u64,
    pub values: Vec<f64>,
}

/// What one decode yields.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Decoded {
    pub samples: Vec<Sample>,
    /// Samples were lost since the last decode: the decode filled its
    /// counter data image, whose overflowing records CUPTI drops.
    pub overflow: bool,
}

/// PM sampling of one device; disabled on drop.
pub struct PmSampler {
    object: *mut c_void,
    counters: Counters,
    image: Vec<u8>,
    started: bool,
}

// SAFETY: the sampling object is CUPTI's, used only through this value;
// CUPTI accepts its calls from any thread.
unsafe impl Send for PmSampler {}

impl PmSampler {
    /// Enable PM sampling of `metrics` on the device of ordinal `device`,
    /// every `interval_ns` nanoseconds, the device writing raw samples into a
    /// hardware buffer of `hardware_buffer` bytes and each decode taking up
    /// to `max_samples` samples.
    ///
    /// # Errors
    ///
    /// Fails when CUPTI lacks PM sampling, a metric is unknown, or another
    /// profiler holds the device's counters.
    pub fn enable(
        device: usize,
        metrics: &[String],
        interval_ns: u64,
        hardware_buffer: usize,
        max_samples: u32,
    ) -> Result<Self> {
        let counters = Counters::configure(
            Profiler::PmSampling,
            device,
            Availability::Device(device),
            metrics,
        )?
        .single_pass()?;
        let cupti = cupti()?;
        let mut enable = EnableParams {
            struct_size: struct_size!(EnableParams, object),
            private: std::ptr::null_mut(),
            device_index: device,
            object: std::ptr::null_mut(),
        };
        // SAFETY: the struct is CUPTI's parameter struct; CUPTI writes the
        // object.
        unsafe { call(&cupti.pm_enable, "cuptiPmSamplingEnable", &mut enable)? };
        let mut sampler = Self {
            object: enable.object,
            counters,
            image: Vec::new(),
            started: false,
        };
        let config = sampler.counters.config();
        let mut set = SetConfigParams {
            struct_size: struct_size!(SetConfigParams, append),
            private: std::ptr::null_mut(),
            object: sampler.object,
            config_size: config.len(),
            config: config.as_ptr(),
            hardware_buffer,
            interval: interval_ns,
            trigger: TIME_INTERVAL,
            append: KEEP_LATEST,
        };
        // SAFETY: as above; the configuration outlives the sampler's use
        // (this value owns it).
        unsafe { call(&cupti.pm_set_config, "cuptiPmSamplingSetConfig", &mut set)? };
        let mut size = DataSizeParams {
            struct_size: struct_size!(DataSizeParams, size),
            private: std::ptr::null_mut(),
            object: sampler.object,
            names: sampler.counters.name_pointers().as_ptr(),
            count: sampler.counters.name_pointers().len(),
            max_samples: max_samples.max(1),
            size: 0,
        };
        // SAFETY: as above; the names outlive the call.
        unsafe {
            call(
                &cupti.pm_data_size,
                "cuptiPmSamplingGetCounterDataSize",
                &mut size,
            )?;
        }
        sampler.image = vec![0; size.size];
        sampler.initialize()?;
        Ok(sampler)
    }

    /// The bytes of counter data one sample of `metrics` takes on the device
    /// of ordinal `device`: what CUPTI states a counter data image of 1024
    /// samples takes, per sample. A sampler's hardware buffer is sized from
    /// it: on GB200 with `nvidia-cuda-cupti` 13.0 the default metrics take
    /// about 13.6 KB of image per sample and 8 KiB of hardware buffer.
    ///
    /// # Errors
    ///
    /// Fails as [`Self::enable`] does.
    pub fn sample_bytes(device: usize, metrics: &[String]) -> Result<usize> {
        const SAMPLES: u32 = 1024;
        let sampler = Self::enable(device, metrics, 1_000_000, 1 << 20, SAMPLES)?;
        Ok(sampler.image.len().div_ceil(SAMPLES as usize))
    }

    /// Initialize the counter data image for the samples to come.
    fn initialize(&mut self) -> Result<()> {
        let cupti = cupti()?;
        let mut params = DataInitializeParams {
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
                &cupti.pm_data_initialize,
                "cuptiPmSamplingCounterDataImageInitialize",
                &mut params,
            )
        }
    }

    /// The metric names, in the order of every sample's values.
    pub fn metrics(&self) -> Vec<String> {
        self.counters.names()
    }

    /// Start sampling.
    pub fn start(&mut self) -> Result<()> {
        if self.started {
            return Ok(());
        }
        let cupti = cupti()?;
        let mut params = self.object_params();
        // SAFETY: the struct is CUPTI's parameter struct.
        unsafe { call(&cupti.pm_start, "cuptiPmSamplingStart", &mut params)? };
        self.started = true;
        Ok(())
    }

    /// Stop sampling; the samples taken stay for the next decode.
    pub fn stop(&mut self) -> Result<()> {
        if !self.started {
            return Ok(());
        }
        self.started = false;
        let cupti = cupti()?;
        let mut params = self.object_params();
        // SAFETY: the struct is CUPTI's parameter struct.
        unsafe { call(&cupti.pm_stop, "cuptiPmSamplingStop", &mut params) }
    }

    /// Decode the samples completed since the last decode into the counter
    /// data image. A decode that fills the image loses the records past it,
    /// which it reports ([`Decoded::overflow`]): on GB200 with
    /// `nvidia-cuda-cupti` 13.0 the next decode finds none of them, whether
    /// sampling or stopped (100 ms of samples every 20 µs decoded as 256 in
    /// an image of 256 samples, and as 5014 in one of 8192), so the image
    /// must hold every sample one decode takes.
    pub fn decode(&mut self) -> Result<Decoded> {
        let mut decoded = Decoded::default();
        let full = self.decode_image(&mut decoded)?;
        decoded.overflow |= full;
        Ok(decoded)
    }

    /// Decode into the counter data image, move its completed samples into
    /// `decoded` and empty it; returns whether the image filled before every
    /// record was decoded.
    fn decode_image(&mut self, decoded: &mut Decoded) -> Result<bool> {
        let cupti = cupti()?;
        let mut decode = DecodeParams {
            struct_size: struct_size!(DecodeParams, overflow),
            private: std::ptr::null_mut(),
            object: self.object,
            data: self.image.as_mut_ptr(),
            size: self.image.len(),
            stop_reason: 0,
            overflow: 0,
        };
        // SAFETY: the struct is CUPTI's parameter struct; the image is the
        // initialized counter data image.
        unsafe { call(&cupti.pm_decode, "cuptiPmSamplingDecodeData", &mut decode)? };
        decoded.overflow |= decode.overflow != 0;
        let mut info = DataInfoParams {
            struct_size: struct_size!(DataInfoParams, completed),
            private: std::ptr::null_mut(),
            data: self.image.as_ptr(),
            size: self.image.len(),
            total: 0,
            populated: 0,
            completed: 0,
        };
        // SAFETY: as above.
        unsafe {
            call(
                &cupti.pm_data_info,
                "cuptiPmSamplingGetCounterDataInfo",
                &mut info,
            )?;
        }
        decoded.samples.reserve(info.completed);
        for index in 0..info.completed {
            let mut sample = SampleInfoParams {
                struct_size: struct_size!(SampleInfoParams, end),
                private: std::ptr::null_mut(),
                object: self.object,
                data: self.image.as_ptr(),
                size: self.image.len(),
                index,
                start: 0,
                end: 0,
            };
            // SAFETY: as above.
            unsafe {
                call(
                    &cupti.pm_sample_info,
                    "cuptiPmSamplingCounterDataGetSampleInfo",
                    &mut sample,
                )?;
            }
            let values = self.counters.evaluate(&self.image, index)?;
            decoded.samples.push(Sample {
                start: sample.start,
                end: sample.end,
                values,
            });
        }
        self.initialize()?;
        Ok(decode.stop_reason == COUNTER_DATA_FULL && info.completed > 0)
    }

    fn object_params(&self) -> ObjectParams {
        ObjectParams {
            struct_size: struct_size!(ObjectParams, object),
            private: std::ptr::null_mut(),
            object: self.object,
        }
    }
}

impl Drop for PmSampler {
    fn drop(&mut self) {
        let _ = self.stop();
        if let Ok(cupti) = cupti() {
            let mut params = self.object_params();
            // SAFETY: the object is live until this call.
            let _ = unsafe { call(&cupti.pm_disable, "cuptiPmSamplingDisable", &mut params) };
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
        assert_eq!(struct_size!(SetConfigParams, append), 64);
        assert_eq!(struct_size!(DecodeParams, overflow), 45);
        assert_eq!(struct_size!(DataSizeParams, size), 56);
        assert_eq!(struct_size!(DataInitializeParams, data), 40);
        assert_eq!(struct_size!(DataInfoParams, completed), 56);
        assert_eq!(struct_size!(SampleInfoParams, end), 64);
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn samples_cover_the_time_sampled_in_order() -> Result<()> {
        use crate::cupti::tests::{SERIAL, Spin};
        use crate::{DeviceId, Stream, lock};

        let _serial = lock(&SERIAL);
        let stream = Stream::new(DeviceId(0))?;
        let spin = Spin::load(&stream)?;
        let mut sampler = PmSampler::enable(
            0,
            &["sm__cycles_active.avg".to_owned()],
            100_000,
            64 << 20,
            4096,
        )?;
        sampler.start()?;
        let started = crate::cupti::timestamp()?;
        for _ in 0..8 {
            spin.launch(&stream, 132, 200_000)?;
        }
        stream.synchronize()?;
        sampler.stop()?;
        let decoded = sampler.decode()?;
        assert!(!decoded.overflow);
        assert!(
            decoded.samples.len() > 2,
            "{} samples",
            decoded.samples.len()
        );
        // After the first, whose start precedes sampling, every sample
        // follows the one before it, in CUPTI's clock.
        for pair in decoded.samples[1..].windows(2) {
            assert!(pair[0].end <= pair[1].start + 1, "{pair:?}");
        }
        assert!(decoded.samples[1].start + 1_000_000_000 > started);
        assert!(decoded.samples.iter().any(|sample| sample.values[0] > 0.0));
        Ok(())
    }

    /// A decode whose image fills takes the image's samples and reports
    /// that it lost the rest.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn a_decode_that_fills_its_image_reports_the_loss() -> Result<()> {
        use crate::cupti::tests::{SERIAL, Spin};
        use crate::{DeviceId, Stream, lock};

        let _serial = lock(&SERIAL);
        let stream = Stream::new(DeviceId(0))?;
        let spin = Spin::load(&stream)?;
        let mut sampler = PmSampler::enable(
            0,
            &["sm__cycles_active.avg".to_owned()],
            20_000,
            64 << 20,
            256,
        )?;
        sampler.start()?;
        // 100 ms of a busy device sampled every 20 µs: about 5000 samples,
        // past one image.
        let began = std::time::Instant::now();
        while began.elapsed() < std::time::Duration::from_millis(100) {
            spin.launch(&stream, 132, 20_000)?;
            stream.synchronize()?;
        }
        sampler.stop()?;
        let decoded = sampler.decode()?;
        assert_eq!(decoded.samples.len(), 256);
        assert!(decoded.overflow);
        Ok(())
    }
}
