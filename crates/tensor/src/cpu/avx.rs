use super::{Cpu, CpuBF16, CpuF16, features};
use core::arch::x86_64::*;
use half::{bf16, f16};
use std::mem::transmute;

pub struct CurrentCpu {}

/// Whether the host converts between BF16 and F32 in 256-bit vectors (AVX-512 BF16 with VL).
#[inline(always)]
fn avx512_bf16() -> bool {
    let features = features::get();
    features.avx512bf16 && features.avx512vl
}

const STEP: usize = 32;
const EPR: usize = 8;
const ARR: usize = STEP / EPR;

impl Cpu for CurrentCpu {
    type Unit = __m256;
    type Array = [__m256; ARR];

    const STEP: usize = STEP;
    const EPR: usize = EPR;
    const ARR: usize = ARR;

    unsafe fn zero() -> Self::Unit {
        // SAFETY: the caller runs this on a CPU with this module's instruction set (`Cpu`); the
        // instructions compute in registers.
        unsafe { _mm256_setzero_ps() }
    }

    unsafe fn zero_array() -> Self::Array {
        // SAFETY: the caller runs this on a CPU with this module's instruction set (`Cpu`); the
        // instructions compute in registers.
        unsafe { [Self::zero(); ARR] }
    }

    unsafe fn load(mem_addr: *const f32) -> Self::Unit {
        // SAFETY: the caller passes `mem_addr` valid for `EPR` reads, on a CPU with this module's
        // instruction set (`Cpu`).
        unsafe { _mm256_loadu_ps(mem_addr) }
    }

    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit {
        // SAFETY: the caller runs this on a CPU with this module's instruction set (`Cpu`); the
        // instructions compute in registers.
        unsafe { _mm256_add_ps(a, b) }
    }

    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit {
        // SAFETY: the caller runs this on a CPU with this module's instruction set (`Cpu`); the
        // instructions compute in registers.
        unsafe { _mm256_add_ps(_mm256_mul_ps(b, c), a) }
    }

    unsafe fn vec_reduce(mut x: Self::Array, y: *mut f32) {
        // SAFETY: the caller passes `y` valid for one write, on a CPU with this module's
        // instruction set (`Cpu`).
        unsafe {
            for i in 0..ARR / 2 {
                x[2 * i] = _mm256_add_ps(x[2 * i], x[2 * i + 1]);
            }
            for i in 0..ARR / 4 {
                x[4 * i] = _mm256_add_ps(x[4 * i], x[4 * i + 2]);
            }
            #[allow(clippy::reversed_empty_ranges)]
            for i in 0..ARR / 8 {
                x[8 * i] = _mm256_add_ps(x[8 * i], x[8 * i + 4]);
            }
            let t0 = _mm_add_ps(_mm256_castps256_ps128(x[0]), _mm256_extractf128_ps(x[0], 1));
            let t1 = _mm_hadd_ps(t0, t0);
            *y = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
        }
    }
}

pub struct CurrentCpuF16 {}
impl CpuF16 for CurrentCpuF16 {
    type Unit = __m256;
    type Array = [__m256; ARR];

    const STEP: usize = STEP;
    const EPR: usize = EPR;
    const ARR: usize = ARR;

    unsafe fn zero() -> Self::Unit {
        // SAFETY: the caller runs this on a CPU with this module's instruction set (`CpuF16`); the
        // instructions compute in registers.
        unsafe { _mm256_setzero_ps() }
    }

    unsafe fn zero_array() -> Self::Array {
        // SAFETY: the caller runs this on a CPU with this module's instruction set (`CpuF16`); the
        // instructions compute in registers.
        unsafe { [Self::zero(); ARR] }
    }

    unsafe fn from_f32(v: f32) -> Self::Unit {
        // SAFETY: the caller runs this on a CPU with this module's instruction set (`CpuF16`); the
        // instructions compute in registers.
        unsafe { _mm256_set1_ps(v) }
    }

    #[cfg(target_feature = "f16c")]
    unsafe fn load(mem_addr: *const f16) -> Self::Unit {
        // SAFETY: the caller passes `mem_addr` valid for `EPR` reads, on a CPU with this module's
        // instruction set (`CpuF16`).
        unsafe { _mm256_cvtph_ps(_mm_loadu_si128(mem_addr as *const __m128i)) }
    }

    #[cfg(not(target_feature = "f16c"))]
    unsafe fn load(mem_addr: *const f16) -> Self::Unit {
        // SAFETY: the caller passes `mem_addr` valid for `EPR` reads, on a CPU with this module's
        // instruction set (`CpuF16`).
        unsafe {
            let mut tmp = [0.0f32; 8];
            for i in 0..8 {
                tmp[i] = (*mem_addr.add(i)).to_f32();
            }
            _mm256_loadu_ps(tmp.as_ptr())
        }
    }

    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit {
        // SAFETY: the caller runs this on a CPU with this module's instruction set (`CpuF16`); the
        // instructions compute in registers.
        unsafe { _mm256_add_ps(a, b) }
    }

    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit {
        // SAFETY: the caller runs this on a CPU with this module's instruction set (`CpuF16`); the
        // instructions compute in registers.
        unsafe { _mm256_add_ps(_mm256_mul_ps(b, c), a) }
    }

    #[cfg(target_feature = "f16c")]
    unsafe fn vec_store(mem_addr: *mut f16, a: Self::Unit) {
        // SAFETY: the caller passes `mem_addr` valid for `EPR` writes, on a CPU with this module's
        // instruction set (`CpuF16`).
        unsafe { _mm_storeu_si128(mem_addr as *mut __m128i, _mm256_cvtps_ph(a, 0)) }
    }

    #[cfg(not(target_feature = "f16c"))]
    unsafe fn vec_store(mem_addr: *mut f16, a: Self::Unit) {
        // SAFETY: the caller passes `mem_addr` valid for `EPR` writes, on a CPU with this module's
        // instruction set (`CpuF16`).
        unsafe {
            let mut tmp = [0.0f32; 8];
            _mm256_storeu_ps(tmp.as_mut_ptr(), a);
            for i in 0..8 {
                *mem_addr.add(i) = f16::from_f32(tmp[i]);
            }
        }
    }

    unsafe fn vec_reduce(mut x: Self::Array, y: *mut f32) {
        // SAFETY: the caller passes `y` valid for one write, on a CPU with this module's
        // instruction set (`CpuF16`).
        unsafe {
            let mut offset = ARR >> 1;
            for i in 0..offset {
                x[i] = _mm256_add_ps(x[i], x[offset + i]);
            }
            offset >>= 1;
            for i in 0..offset {
                x[i] = _mm256_add_ps(x[i], x[offset + i]);
            }
            offset >>= 1;
            for i in 0..offset {
                x[i] = _mm256_add_ps(x[i], x[offset + i]);
            }
            let t0 = _mm_add_ps(_mm256_castps256_ps128(x[0]), _mm256_extractf128_ps(x[0], 1));
            let t1 = _mm_hadd_ps(t0, t0);
            *y = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
        }
    }
}

pub struct CurrentCpuBF16 {}
impl CpuBF16 for CurrentCpuBF16 {
    type Unit = __m256;
    type Array = [__m256; ARR];

    const STEP: usize = STEP;
    const EPR: usize = EPR;
    const ARR: usize = ARR;

    unsafe fn zero() -> Self::Unit {
        // SAFETY: the caller runs this on a CPU with this module's instruction set (`CpuBF16`); the
        // instructions compute in registers.
        unsafe { _mm256_setzero_ps() }
    }

    unsafe fn zero_array() -> Self::Array {
        // SAFETY: the caller runs this on a CPU with this module's instruction set (`CpuBF16`); the
        // instructions compute in registers.
        unsafe { [Self::zero(); ARR] }
    }

    unsafe fn from_f32(v: f32) -> Self::Unit {
        // SAFETY: the caller runs this on a CPU with this module's instruction set (`CpuBF16`); the
        // instructions compute in registers.
        unsafe { _mm256_set1_ps(v) }
    }

    unsafe fn load(mem_addr: *const bf16) -> Self::Unit {
        // SAFETY: the caller passes `mem_addr` valid for `EPR` reads, on a CPU with this module's
        // instruction set (`CpuBF16`).
        unsafe {
            if avx512_bf16() {
                _mm256_cvtpbh_ps(transmute::<__m128i, __m128bh>(_mm_loadu_si128(
                    mem_addr as *const __m128i,
                )))
            } else {
                let mut tmp = [0.0f32; 8];
                for i in 0..8 {
                    tmp[i] = (*mem_addr.add(i)).to_f32();
                }
                _mm256_loadu_ps(tmp.as_ptr())
            }
        }
    }

    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit {
        // SAFETY: the caller runs this on a CPU with this module's instruction set (`CpuBF16`); the
        // instructions compute in registers.
        unsafe { _mm256_add_ps(a, b) }
    }

    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit {
        // SAFETY: the caller runs this on a CPU with this module's instruction set (`CpuBF16`); the
        // instructions compute in registers.
        unsafe { _mm256_add_ps(_mm256_mul_ps(b, c), a) }
    }

    unsafe fn vec_store(mem_addr: *mut bf16, a: Self::Unit) {
        // SAFETY: the caller passes `mem_addr` valid for `EPR` writes, on a CPU with this module's
        // instruction set (`CpuBF16`).
        unsafe {
            if avx512_bf16() {
                _mm_storeu_si128(
                    mem_addr as *mut __m128i,
                    transmute::<__m128bh, __m128i>(_mm256_cvtneps_pbh(a)),
                )
            } else {
                let mut tmp = [0.0f32; 8];
                _mm256_storeu_ps(tmp.as_mut_ptr(), a);
                for i in 0..8 {
                    *mem_addr.add(i) = bf16::from_f32(tmp[i]);
                }
            }
        }
    }

    unsafe fn vec_reduce(mut x: Self::Array, y: *mut f32) {
        // SAFETY: the caller passes `y` valid for one write, on a CPU with this module's
        // instruction set (`CpuBF16`).
        unsafe {
            let mut offset = ARR >> 1;
            for i in 0..offset {
                x[i] = _mm256_add_ps(x[i], x[offset + i]);
            }
            offset >>= 1;
            for i in 0..offset {
                x[i] = _mm256_add_ps(x[i], x[offset + i]);
            }
            offset >>= 1;
            for i in 0..offset {
                x[i] = _mm256_add_ps(x[i], x[offset + i]);
            }
            let t0 = _mm_add_ps(_mm256_castps256_ps128(x[0]), _mm256_extractf128_ps(x[0], 1));
            let t1 = _mm_hadd_ps(t0, t0);
            *y = _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vec_dot_f32(a_row: *const f32, b_row: *const f32, c: *mut f32, k: usize) {
    // SAFETY: the caller passes `a_row` and `b_row` valid for `k` reads and `c` valid for one
    // write, on a CPU with this module's instruction set.
    unsafe {
        let mut sum = CurrentCpu::zero_array();
        let mut i = 0;
        while i + CurrentCpu::STEP <= k {
            for j in 0..CurrentCpu::ARR {
                sum[j] = CurrentCpu::vec_fma(
                    sum[j],
                    CurrentCpu::load(a_row.add(i + j * CurrentCpu::EPR)),
                    CurrentCpu::load(b_row.add(i + j * CurrentCpu::EPR)),
                );
            }
            i += CurrentCpu::STEP;
        }
        CurrentCpu::vec_reduce(sum, c);
        while i < k {
            *c += *a_row.add(i) * (*b_row.add(i));
            i += 1;
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vec_sum(row: *const f32, b: *mut f32, k: usize) {
    // SAFETY: the caller passes `row` valid for `k` reads and `b` valid for one write, on a CPU
    // with this module's instruction set.
    unsafe {
        let np = k & !(CurrentCpu::STEP - 1);
        let mut sum = CurrentCpu::zero_array();
        let mut x = CurrentCpu::zero_array();
        for i in (0..np).step_by(CurrentCpu::STEP) {
            for j in 0..CurrentCpu::ARR {
                x[j] = CurrentCpu::load(row.add(i + j * CurrentCpu::EPR));
                sum[j] = CurrentCpu::vec_add(sum[j], x[j]);
            }
        }
        CurrentCpu::vec_reduce(sum, b);
        for i in np..k {
            *b += *row.add(i)
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vec_dot_f16(a_row: *const f16, b_row: *const f16, c: *mut f32, k: usize) {
    // SAFETY: the caller passes `a_row` and `b_row` valid for `k` reads and `c` valid for one
    // write, on a CPU with this module's instruction set.
    unsafe {
        let mut sumf = 0.0f32;
        let mut sum = CurrentCpuF16::zero_array();
        let mut steps_since_flush = 0usize;
        let mut i = 0;
        while i + CurrentCpuF16::STEP <= k {
            for j in 0..CurrentCpuF16::ARR {
                sum[j] = CurrentCpuF16::vec_fma(
                    sum[j],
                    CurrentCpuF16::load(a_row.add(i + j * CurrentCpuF16::EPR)),
                    CurrentCpuF16::load(b_row.add(i + j * CurrentCpuF16::EPR)),
                );
            }
            steps_since_flush += 1;
            if steps_since_flush == CurrentCpuF16::FLUSH_INTERVAL {
                let mut partial = 0.0f32;
                CurrentCpuF16::vec_reduce(sum, &mut partial);
                sumf += partial;
                sum = CurrentCpuF16::zero_array();
                steps_since_flush = 0;
            }
            i += CurrentCpuF16::STEP;
        }
        let mut partial = 0.0f32;
        CurrentCpuF16::vec_reduce(sum, &mut partial);
        sumf += partial;
        while i < k {
            sumf += (*a_row.add(i)).to_f32() * (*b_row.add(i)).to_f32();
            i += 1;
        }
        *c = sumf;
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vec_dot_bf16(a_row: *const bf16, b_row: *const bf16, c: *mut f32, k: usize) {
    // SAFETY: the caller passes `a_row` and `b_row` valid for `k` reads and `c` valid for one
    // write, on a CPU with this module's instruction set.
    unsafe {
        let mut sum = CurrentCpuBF16::zero_array();
        let mut i = 0;
        while i + CurrentCpuBF16::STEP <= k {
            for j in 0..CurrentCpuBF16::ARR {
                sum[j] = CurrentCpuBF16::vec_fma(
                    sum[j],
                    CurrentCpuBF16::load(a_row.add(i + j * CurrentCpuBF16::EPR)),
                    CurrentCpuBF16::load(b_row.add(i + j * CurrentCpuBF16::EPR)),
                );
            }
            i += CurrentCpuBF16::STEP;
        }
        CurrentCpuBF16::vec_reduce(sum, c);
        while i < k {
            *c += (*a_row.add(i)).to_f32() * (*b_row.add(i)).to_f32();
            i += 1;
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vec_add_f16(a_row: *const f16, b_row: *const f16, c: *mut f16, k: usize) {
    // SAFETY: the caller passes `a_row` and `b_row` valid for `k` reads and `c` valid for `k`
    // writes, on a CPU with this module's instruction set.
    unsafe {
        let mut i = 0;
        while i + CurrentCpuF16::STEP <= k {
            for j in 0..CurrentCpuF16::ARR {
                CurrentCpuF16::vec_store(
                    c.add(i + j * CurrentCpuF16::EPR),
                    CurrentCpuF16::vec_add(
                        CurrentCpuF16::load(a_row.add(i + j * CurrentCpuF16::EPR)),
                        CurrentCpuF16::load(b_row.add(i + j * CurrentCpuF16::EPR)),
                    ),
                );
            }
            i += CurrentCpuF16::STEP;
        }
        for j in i..k {
            *c.add(j) = *a_row.add(j) + *b_row.add(j);
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vec_add_bf16(a_row: *const bf16, b_row: *const bf16, c: *mut bf16, k: usize) {
    // SAFETY: the caller passes `a_row` and `b_row` valid for `k` reads and `c` valid for `k`
    // writes, on a CPU with this module's instruction set.
    unsafe {
        let mut i = 0;
        while i + CurrentCpuBF16::STEP <= k {
            for j in 0..CurrentCpuBF16::ARR {
                CurrentCpuBF16::vec_store(
                    c.add(i + j * CurrentCpuBF16::EPR),
                    CurrentCpuBF16::vec_add(
                        CurrentCpuBF16::load(a_row.add(i + j * CurrentCpuBF16::EPR)),
                        CurrentCpuBF16::load(b_row.add(i + j * CurrentCpuBF16::EPR)),
                    ),
                );
            }
            i += CurrentCpuBF16::STEP;
        }
        for j in i..k {
            *c.add(j) = *a_row.add(j) + *b_row.add(j);
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vec_scalar_add_f16(scalar: f16, xs: *const f16, ys: *mut f16, k: usize) {
    // SAFETY: the caller passes `xs` valid for `k` reads and `ys` valid for `k` writes, on a CPU
    // with this module's instruction set.
    unsafe {
        let sv = CurrentCpuF16::from_f32(scalar.to_f32());
        let mut i = 0;
        while i + CurrentCpuF16::STEP <= k {
            for j in 0..CurrentCpuF16::ARR {
                CurrentCpuF16::vec_store(
                    ys.add(i + j * CurrentCpuF16::EPR),
                    CurrentCpuF16::vec_add(
                        CurrentCpuF16::load(xs.add(i + j * CurrentCpuF16::EPR)),
                        sv,
                    ),
                );
            }
            i += CurrentCpuF16::STEP;
        }
        for j in i..k {
            *ys.add(j) = *xs.add(j) + scalar;
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn vec_scalar_add_bf16(scalar: bf16, xs: *const bf16, ys: *mut bf16, k: usize) {
    // SAFETY: the caller passes `xs` valid for `k` reads and `ys` valid for `k` writes, on a CPU
    // with this module's instruction set.
    unsafe {
        let sv = CurrentCpuBF16::from_f32(scalar.to_f32());
        let mut i = 0;
        while i + CurrentCpuBF16::STEP <= k {
            for j in 0..CurrentCpuBF16::ARR {
                CurrentCpuBF16::vec_store(
                    ys.add(i + j * CurrentCpuBF16::EPR),
                    CurrentCpuBF16::vec_add(
                        CurrentCpuBF16::load(xs.add(i + j * CurrentCpuBF16::EPR)),
                        sv,
                    ),
                );
            }
            i += CurrentCpuBF16::STEP;
        }
        for j in i..k {
            *ys.add(j) = *xs.add(j) + scalar;
        }
    }
}
