//! The instruction-set extensions of the host CPU that select a vector kernel at run time.
//!
//! Only the extensions a kernel of this module tests are detected. The x86_64 baseline has no
//! AVX, so its kernels test AVX2 and the conversions they use; Advanced SIMD (NEON) is part of
//! every aarch64 CPU, so only its optional FP16 and BF16 arithmetic is tested there. On any other
//! architecture the scalar kernels run.

use std::sync::OnceLock;

/// The extensions of the x86_64 host the vector kernels select by.
#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CpuFeatures {
    /// 256-bit integer and float vectors, which every `avx` kernel requires.
    pub(crate) avx2: bool,
    /// Conversions between F16 and F32 vectors, for the F16 kernels of a build that enables it.
    pub(crate) f16c: bool,
    /// AVX-512 BF16 conversions; with `avx512vl`, the 256-bit BF16 loads and stores.
    pub(crate) avx512bf16: bool,
    /// The 128-bit and 256-bit forms of AVX-512 instructions.
    pub(crate) avx512vl: bool,
}

/// The extensions of the aarch64 host the vector kernels select by.
#[cfg(target_arch = "aarch64")]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CpuFeatures {
    /// F16 vector arithmetic, for the F16 kernels of a build that enables it.
    pub(crate) fp16: bool,
    /// BF16 dot products and conversions, for the BF16 kernels of a build that enables it.
    pub(crate) bf16: bool,
}

/// The host CPU's extensions, detected once per process.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) fn get() -> CpuFeatures {
    static FEATURES: OnceLock<CpuFeatures> = OnceLock::new();
    *FEATURES.get_or_init(detect)
}

#[cfg(target_arch = "x86_64")]
fn detect() -> CpuFeatures {
    CpuFeatures {
        avx2: std::is_x86_feature_detected!("avx2"),
        f16c: std::is_x86_feature_detected!("f16c"),
        avx512bf16: std::is_x86_feature_detected!("avx512bf16"),
        avx512vl: std::is_x86_feature_detected!("avx512vl"),
    }
}

#[cfg(target_arch = "aarch64")]
fn detect() -> CpuFeatures {
    CpuFeatures {
        fp16: std::arch::is_aarch64_feature_detected!("fp16"),
        bf16: std::arch::is_aarch64_feature_detected!("bf16"),
    }
}
