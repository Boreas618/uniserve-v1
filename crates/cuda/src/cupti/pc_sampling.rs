//! Warp program counters and their stall reasons, sampled continuously
//! (CUPTI's PC sampling API, `cupti_pcsampling.h`).
//!
//! A [`PcSampler`] is enabled on one context in continuous collection mode,
//! which samples the warps of every kernel without serializing kernels, with
//! every stall reason the device reports. Between [`PcSampler::start`] and
//! [`PcSampler::stop`] the device samples a warp of each SM every
//! `2^(5 + period)` cycles. CUPTI aggregates the samples per program counter:
//! a [`Pc`] is a function, an offset in it and the samples of each stall
//! reason at that offset. On this CUPTI (13.0.85), stopping moves what was
//! collected into the sampler's buffer; [`PcSampler::take`] reads it, and
//! while sampling asks CUPTI for what it collected so far.

use std::ffi::{c_char, c_void};

use super::{call, cupti};
use crate::{Error, Result};

/// The bytes CUPTI writes a stall reason's name into
/// (`CUPTI_STALL_REASON_STRING_SIZE`).
const STALL_NAME: usize = 128;

/// `CUpti_PCSamplingConfigurationAttributeType`s.
const SAMPLING_PERIOD: u32 = 1;
const STALL_REASON: u32 = 2;
const HARDWARE_BUFFER_SIZE: u32 = 4;
const COLLECTION_MODE: u32 = 5;
const START_STOP_CONTROL: u32 = 6;
const OUTPUT_DATA_FORMAT: u32 = 7;
const SAMPLING_DATA_BUFFER: u32 = 8;

/// `CUPTI_PC_SAMPLING_COLLECTION_MODE_CONTINUOUS`.
const CONTINUOUS: u64 = 1;
/// `CUPTI_PC_SAMPLING_OUTPUT_DATA_FORMAT_PARSED`.
const PARSED: u64 = 1;

/// `CUpti_PCSamplingStallReason`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct StallSamples {
    reason: u32,
    samples: u32,
}

/// `CUpti_PCSamplingPCData` (64 bytes, 8-byte aligned).
#[repr(C)]
#[derive(Clone, Copy)]
struct PcData {
    size: usize,
    cubin_crc: u64,
    pc_offset: u64,
    function_index: u32,
    pad: u32,
    function_name: *const c_char,
    stall_count: usize,
    stalls: *mut StallSamples,
    correlation: u32,
}

/// `CUpti_PCSamplingData` (80 bytes).
#[repr(C)]
struct SamplingData {
    size: usize,
    collect: usize,
    total_samples: u64,
    dropped_samples: u64,
    total_pcs: usize,
    remaining_pcs: usize,
    range: u64,
    pcs: *mut PcData,
    non_user_samples: u64,
    hardware_buffer_full: u8,
}

/// `CUpti_PCSamplingConfigurationInfo`: its type, its status and its
/// union of data.
#[repr(C)]
struct ConfigurationInfo {
    kind: u32,
    status: i32,
    data: [u64; 3],
}

#[repr(C)]
struct ConfigurationParams {
    size: usize,
    private: *mut c_void,
    context: *mut c_void,
    count: usize,
    info: *mut ConfigurationInfo,
}

#[repr(C)]
struct ContextParams {
    size: usize,
    private: *mut c_void,
    context: *mut c_void,
}

#[repr(C)]
struct GetDataParams {
    size: usize,
    private: *mut c_void,
    context: *mut c_void,
    data: *mut c_void,
}

#[repr(C)]
struct StallCountParams {
    size: usize,
    private: *mut c_void,
    context: *mut c_void,
    count: *mut usize,
}

#[repr(C)]
struct StallReasonsParams {
    size: usize,
    private: *mut c_void,
    context: *mut c_void,
    count: usize,
    indices: *mut u32,
    names: *mut *mut c_char,
}

/// The samples of one program counter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pc {
    pub function: String,
    pub function_index: u32,
    pub cubin_crc: u64,
    pub offset: u64,
    /// The launch's correlation id; 0 where CUPTI does not attribute samples
    /// to launches (continuous mode on this CUPTI).
    pub correlation: u32,
    /// Samples by stall reason index ([`PcSampler::stall_reasons`]).
    pub stalls: Vec<(u32, u32)>,
}

/// What one take yields.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Taken {
    pub pcs: Vec<Pc>,
    pub total_samples: u64,
    pub dropped_samples: u64,
    /// Samples of kernels CUPTI does not attribute to the user (its own and
    /// the driver's).
    pub non_user_samples: u64,
    /// The hardware buffer filled: samples were lost.
    pub hardware_buffer_full: bool,
}

/// Continuous PC sampling of one context; disabled on drop.
pub struct PcSampler {
    context: *mut c_void,
    /// The buffer CUPTI fills, at a fixed address while sampling is enabled.
    data: Box<SamplingData>,
    pcs: Vec<PcData>,
    stalls: Vec<StallSamples>,
    reasons: Vec<(u32, String)>,
    started: bool,
}

// SAFETY: the buffers belong to this value; CUPTI accepts its calls from any
// thread.
unsafe impl Send for PcSampler {}

impl PcSampler {
    /// Enable continuous PC sampling on `context` (a `CUcontext`), each SM
    /// sampling a warp every `2^period` cycles (`period` from 5 to 31), into
    /// a hardware buffer of `hardware_buffer` bytes of device storage
    /// (CUPTI's default, 512 MiB, when `None`), each take holding up to
    /// `max_pcs` program counters.
    ///
    /// # Errors
    ///
    /// Fails when CUPTI lacks PC sampling or refuses the configuration.
    pub fn enable(
        context: usize,
        period: u32,
        max_pcs: usize,
        hardware_buffer: Option<usize>,
    ) -> Result<Self> {
        let cupti = cupti()?;
        let context = context as *mut c_void;
        let mut enable = ContextParams {
            size: size_of::<ContextParams>(),
            private: std::ptr::null_mut(),
            context,
        };
        // SAFETY: the struct is CUPTI's parameter struct.
        unsafe { call(&cupti.pc_enable, "cuptiPCSamplingEnable", &mut enable)? };

        let mut count = 0;
        let mut stall_count = StallCountParams {
            size: size_of::<StallCountParams>(),
            private: std::ptr::null_mut(),
            context,
            count: &mut count,
        };
        // SAFETY: as above; CUPTI writes the count.
        let counted = unsafe {
            call(
                &cupti.pc_stall_count,
                "cuptiPCSamplingGetNumStallReasons",
                &mut stall_count,
            )
        };
        let disable = || {
            let mut disable = ContextParams {
                size: size_of::<ContextParams>(),
                private: std::ptr::null_mut(),
                context,
            };
            // SAFETY: as above.
            let _ = unsafe { call(&cupti.pc_disable, "cuptiPCSamplingDisable", &mut disable) };
        };
        if let Err(error) = counted {
            disable();
            return Err(error);
        }
        let mut indices = vec![0_u32; count];
        let mut names = vec![[0 as c_char; STALL_NAME]; count];
        let mut name_pointers: Vec<*mut c_char> =
            names.iter_mut().map(|name| name.as_mut_ptr()).collect();
        let mut reasons = StallReasonsParams {
            size: size_of::<StallReasonsParams>(),
            private: std::ptr::null_mut(),
            context,
            count,
            indices: indices.as_mut_ptr(),
            names: name_pointers.as_mut_ptr(),
        };
        // SAFETY: as above; CUPTI writes an index and a name of at most
        // `STALL_NAME` bytes per reason.
        if let Err(error) = unsafe {
            call(
                &cupti.pc_stall_reasons,
                "cuptiPCSamplingGetStallReasons",
                &mut reasons,
            )
        } {
            disable();
            return Err(error);
        }
        let reasons: Vec<(u32, String)> = indices
            .iter()
            .zip(&names)
            .map(|(&index, name)| {
                // SAFETY: CUPTI wrote a NUL-terminated name.
                (
                    index,
                    unsafe { super::text(name.as_ptr()) }.unwrap_or_default(),
                )
            })
            .collect();

        let max_pcs = max_pcs.max(1);
        let mut sampler = Self {
            context,
            data: Box::new(SamplingData {
                size: size_of::<SamplingData>(),
                collect: max_pcs,
                total_samples: 0,
                dropped_samples: 0,
                total_pcs: 0,
                remaining_pcs: 0,
                range: 0,
                pcs: std::ptr::null_mut(),
                non_user_samples: 0,
                hardware_buffer_full: 0,
            }),
            pcs: Vec::new(),
            stalls: vec![StallSamples::default(); max_pcs * count],
            reasons,
            started: false,
        };
        let stall_base = sampler.stalls.as_mut_ptr();
        sampler.pcs = (0..max_pcs)
            .map(|pc| PcData {
                size: size_of::<PcData>(),
                cubin_crc: 0,
                pc_offset: 0,
                function_index: 0,
                pad: 0,
                function_name: std::ptr::null(),
                stall_count: 0,
                // SAFETY: each program counter gets its own `count` entries.
                stalls: unsafe { stall_base.add(pc * count) },
                correlation: 0,
            })
            .collect();
        sampler.data.pcs = sampler.pcs.as_mut_ptr();

        let mut all = indices.clone();
        let mut info = vec![
            ConfigurationInfo {
                kind: COLLECTION_MODE,
                status: 0,
                data: [CONTINUOUS, 0, 0],
            },
            ConfigurationInfo {
                kind: SAMPLING_PERIOD,
                status: 0,
                data: [u64::from(period), 0, 0],
            },
            ConfigurationInfo {
                kind: STALL_REASON,
                status: 0,
                data: [count as u64, all.as_mut_ptr() as u64, 0],
            },
            ConfigurationInfo {
                kind: START_STOP_CONTROL,
                status: 0,
                data: [1, 0, 0],
            },
            ConfigurationInfo {
                kind: OUTPUT_DATA_FORMAT,
                status: 0,
                data: [PARSED, 0, 0],
            },
            ConfigurationInfo {
                kind: SAMPLING_DATA_BUFFER,
                status: 0,
                data: [(&raw mut *sampler.data) as u64, 0, 0],
            },
        ];
        if let Some(bytes) = hardware_buffer {
            info.push(ConfigurationInfo {
                kind: HARDWARE_BUFFER_SIZE,
                status: 0,
                data: [bytes as u64, 0, 0],
            });
        }
        let mut configure = ConfigurationParams {
            size: size_of::<ConfigurationParams>(),
            private: std::ptr::null_mut(),
            context,
            count: info.len(),
            info: info.as_mut_ptr(),
        };
        // SAFETY: as above; the stall indices outlive the call and the
        // sampling buffer this value.
        unsafe {
            call(
                &cupti.pc_configure,
                "cuptiPCSamplingSetConfigurationAttribute",
                &mut configure,
            )?;
        }
        if let Some(refused) = info.iter().find(|info| info.status != 0) {
            return Err(Error::Cupti {
                call: "cuptiPCSamplingSetConfigurationAttribute",
                code: refused.status,
            });
        }
        Ok(sampler)
    }

    /// Sample every `2^period` cycles from the next start on; the sampler
    /// is stopped.
    ///
    /// # Errors
    ///
    /// Fails when CUPTI refuses the period.
    pub fn set_period(&mut self, period: u32) -> Result<()> {
        self.configure(SAMPLING_PERIOD, u64::from(period))
    }

    /// Use a hardware buffer of `bytes` from the next start on; the sampler
    /// is stopped.
    ///
    /// # Errors
    ///
    /// Fails when CUPTI refuses the size.
    pub fn set_hardware_buffer(&mut self, bytes: usize) -> Result<()> {
        self.configure(HARDWARE_BUFFER_SIZE, bytes as u64)
    }

    /// Set one configuration attribute of the enabled sampler.
    fn configure(&mut self, kind: u32, value: u64) -> Result<()> {
        let cupti = cupti()?;
        let mut info = [ConfigurationInfo {
            kind,
            status: 0,
            data: [value, 0, 0],
        }];
        let mut configure = ConfigurationParams {
            size: size_of::<ConfigurationParams>(),
            private: std::ptr::null_mut(),
            context: self.context,
            count: info.len(),
            info: info.as_mut_ptr(),
        };
        // SAFETY: the struct is CUPTI's parameter struct; PC sampling is
        // enabled on the context by this value.
        unsafe {
            call(
                &cupti.pc_configure,
                "cuptiPCSamplingSetConfigurationAttribute",
                &mut configure,
            )?;
        }
        match info[0].status {
            0 => Ok(()),
            code => Err(Error::Cupti {
                call: "cuptiPCSamplingSetConfigurationAttribute",
                code,
            }),
        }
    }

    /// The stall reasons the samples name: index and name.
    pub fn stall_reasons(&self) -> &[(u32, String)] {
        &self.reasons
    }

    fn context_params(&self) -> ContextParams {
        ContextParams {
            size: size_of::<ContextParams>(),
            private: std::ptr::null_mut(),
            context: self.context,
        }
    }

    /// Start sampling.
    pub fn start(&mut self) -> Result<()> {
        if self.started {
            return Ok(());
        }
        let cupti = cupti()?;
        let mut params = self.context_params();
        // SAFETY: the struct is CUPTI's parameter struct.
        unsafe { call(&cupti.pc_start, "cuptiPCSamplingStart", &mut params)? };
        self.started = true;
        Ok(())
    }

    /// Stop sampling; what was collected waits for [`PcSampler::take`].
    pub fn stop(&mut self) -> Result<()> {
        if !self.started {
            return Ok(());
        }
        self.started = false;
        let cupti = cupti()?;
        let mut params = self.context_params();
        // SAFETY: the struct is CUPTI's parameter struct.
        unsafe { call(&cupti.pc_stop, "cuptiPCSamplingStop", &mut params) }
    }

    /// Take what was collected: what the last stop moved into the buffer,
    /// then, while CUPTI reports more, what it holds.
    pub fn take(&mut self) -> Result<Taken> {
        let mut taken = Taken::default();
        self.drain(&mut taken);
        let cupti = cupti()?;
        loop {
            let mut params = GetDataParams {
                size: size_of::<GetDataParams>(),
                private: std::ptr::null_mut(),
                context: self.context,
                data: (&raw mut *self.data).cast(),
            };
            // SAFETY: the struct is CUPTI's parameter struct; the buffer
            // holds `collect` program counters.
            let got = unsafe { call(&cupti.pc_get_data, "cuptiPCSamplingGetData", &mut params) };
            if let Err(error) = got {
                // A stopped sampler has nothing pending past what the stop
                // moved into the buffer.
                if self.started && taken.pcs.is_empty() {
                    return Err(error);
                }
                break;
            }
            let more = self.data.remaining_pcs > 0;
            let read = self.drain(&mut taken);
            if !more || read == 0 {
                break;
            }
        }
        Ok(taken)
    }

    /// Move what the buffer holds into `taken` and empty it; returns the
    /// program counters read.
    fn drain(&mut self, taken: &mut Taken) -> usize {
        let count = self.data.total_pcs.min(self.pcs.len());
        taken.total_samples += self.data.total_samples;
        taken.dropped_samples += self.data.dropped_samples;
        taken.non_user_samples += self.data.non_user_samples;
        taken.hardware_buffer_full |= self.data.hardware_buffer_full != 0;
        for pc in &self.pcs[..count] {
            let stalls = (0..pc.stall_count.min(self.reasons.len()))
                .map(|index| {
                    // SAFETY: CUPTI wrote `stall_count` entries into this
                    // program counter's own entries.
                    let stall = unsafe { *pc.stalls.add(index) };
                    (stall.reason, stall.samples)
                })
                .collect();
            taken.pcs.push(Pc {
                // SAFETY: CUPTI's function names live as long as the module.
                function: unsafe { super::text(pc.function_name) }.unwrap_or_default(),
                function_index: pc.function_index,
                cubin_crc: pc.cubin_crc,
                offset: pc.pc_offset,
                correlation: pc.correlation,
                stalls,
            });
        }
        self.data.total_pcs = 0;
        self.data.total_samples = 0;
        self.data.dropped_samples = 0;
        self.data.non_user_samples = 0;
        self.data.hardware_buffer_full = 0;
        count
    }
}

impl Drop for PcSampler {
    fn drop(&mut self) {
        let _ = self.stop();
        if let Ok(cupti) = cupti() {
            let mut params = self.context_params();
            // SAFETY: sampling was enabled on the context by this value.
            let _ = unsafe { call(&cupti.pc_disable, "cuptiPCSamplingDisable", &mut params) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The structs have the sizes of the CUPTI 13 headers, which CUPTI reads
    /// from their `size` fields.
    #[test]
    fn structs_have_the_sizes_of_the_cupti_headers() {
        assert_eq!(size_of::<PcData>(), 64);
        assert_eq!(std::mem::offset_of!(PcData, correlation), 56);
        assert_eq!(size_of::<SamplingData>(), 80);
        assert_eq!(std::mem::offset_of!(SamplingData, hardware_buffer_full), 72);
        assert_eq!(size_of::<StallSamples>(), 8);
        assert_eq!(size_of::<ConfigurationInfo>(), 32);
        assert_eq!(size_of::<ConfigurationParams>(), 40);
        assert_eq!(size_of::<GetDataParams>(), 32);
        assert_eq!(size_of::<ContextParams>(), 24);
        assert_eq!(size_of::<StallCountParams>(), 32);
        assert_eq!(size_of::<StallReasonsParams>(), 48);
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn a_running_kernel_is_sampled_with_its_stall_reasons() -> Result<()> {
        use crate::cupti::tests::{SERIAL, Spin};
        use crate::{DeviceId, PrimaryContext, Stream, lock};

        let _serial = lock(&SERIAL);
        let device = DeviceId(0);
        let context = PrimaryContext::retain(device)?;
        let stream = Stream::new(device)?;
        let spin = Spin::load(&stream)?;
        let mut sampler = PcSampler::enable(context.raw() as usize, 10, 1000, None)?;
        assert!(!sampler.stall_reasons().is_empty());
        sampler.start()?;
        for _ in 0..4 {
            spin.launch(&stream, 132, 2_000_000)?;
        }
        stream.synchronize()?;
        sampler.stop()?;
        let taken = sampler.take()?;
        assert!(taken.total_samples > 0, "{taken:?}");
        let pc = taken
            .pcs
            .iter()
            .find(|pc| pc.function == "spin")
            .expect("the kernel's program counters are sampled");
        let known: Vec<u32> = sampler
            .stall_reasons()
            .iter()
            .map(|&(index, _)| index)
            .collect();
        assert!(pc.stalls.iter().all(|(reason, _)| known.contains(reason)));
        Ok(())
    }
}
