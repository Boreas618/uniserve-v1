//! Traits and methods for CPU-backed Tensors
//!
//! The vector kernels exist for x86_64 (`avx`) and aarch64 (`neon`); every other architecture,
//! and an x86_64 CPU without AVX2, runs the scalar loops of this module.
#![allow(clippy::needless_range_loop)]
pub mod erf;
pub(crate) mod features;
pub mod kernels;

/// The vector instructions of one instruction set over `f32` lanes, in which the dot products
/// and sums of this module are written.
///
/// # Safety
///
/// Every method executes instructions of its implementation's instruction set, so the caller runs
/// it on a CPU that has them (`features::get`). `load` reads and, in [`CpuF16`] and
/// [`CpuBF16`], `vec_store` writes `EPR` consecutive elements at `mem_addr`, and `vec_reduce`
/// writes one value to `y`: the caller passes pointers valid for these accesses.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
trait Cpu {
    type Unit;
    type Array;
    const STEP: usize;
    const EPR: usize;
    const ARR: usize = Self::STEP / Self::EPR;

    unsafe fn zero() -> Self::Unit;
    unsafe fn zero_array() -> Self::Array;
    unsafe fn load(mem_addr: *const f32) -> Self::Unit;
    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit;
    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit;
    unsafe fn vec_reduce(x: Self::Array, y: *mut f32);
}

/// [`Cpu`] over F16 elements in memory, with the same safety requirements.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
trait CpuF16 {
    type Unit;
    type Array;
    const STEP: usize;
    const EPR: usize;
    const ARR: usize = Self::STEP / Self::EPR;
    // How many outer loop steps to accumulate in Self::Unit before flushing to f32.
    // Set to a finite value to bound rounding error (16 for example).
    const FLUSH_INTERVAL: usize = usize::MAX;

    unsafe fn zero() -> Self::Unit;
    unsafe fn zero_array() -> Self::Array;
    unsafe fn load(mem_addr: *const f16) -> Self::Unit;
    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit;
    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit;
    unsafe fn vec_reduce(x: Self::Array, y: *mut f32);
    unsafe fn from_f32(v: f32) -> Self::Unit;
    unsafe fn vec_store(mem_addr: *mut f16, a: Self::Unit);
}

/// [`Cpu`] over BF16 elements in memory, with the same safety requirements.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[cfg_attr(
    target_arch = "aarch64",
    expect(
        dead_code,
        reason = "the NEON kernels add and store BF16 by widening to F32 and call only the dot \
                  product's methods; the AVX kernels call every one"
    )
)]
trait CpuBF16 {
    type Unit;
    type Array;
    const STEP: usize;
    const EPR: usize;
    const ARR: usize = Self::STEP / Self::EPR;

    unsafe fn zero() -> Self::Unit;
    unsafe fn zero_array() -> Self::Array;
    unsafe fn load(mem_addr: *const bf16) -> Self::Unit;
    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit;
    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit;
    unsafe fn vec_reduce(x: Self::Array, y: *mut f32);
    unsafe fn from_f32(v: f32) -> Self::Unit;
    unsafe fn vec_store(mem_addr: *mut bf16, a: Self::Unit);
}

use half::{bf16, f16};

#[cfg(target_arch = "x86_64")]
pub mod avx;

#[cfg(target_arch = "aarch64")]
pub mod neon;

// The F32 kernels of `neon` use baseline instructions only, so an aarch64 build never runs the
// scalar F32 loops.
#[inline(always)]
#[cfg(not(target_arch = "aarch64"))]
unsafe fn scalar_vec_dot_f32(a_row: *const f32, b_row: *const f32, c: *mut f32, k: usize) {
    // SAFETY: the caller passes `a_row` and `b_row` valid for `k` reads and `c` valid for one
    // write.
    unsafe {
        for i in 0..k {
            *c += *a_row.add(i) * (*b_row.add(i));
        }
    }
}

#[inline(always)]
#[cfg(not(target_arch = "aarch64"))]
unsafe fn scalar_vec_sum(row: *const f32, b: *mut f32, k: usize) {
    // SAFETY: the caller passes `row` valid for `k` reads and `b` valid for one write.
    unsafe {
        *b = 0f32;
        for i in 0..k {
            *b += *row.add(i)
        }
    }
}

#[inline(always)]
unsafe fn scalar_vec_dot_f16(a_row: *const f16, b_row: *const f16, c: *mut f32, k: usize) {
    // SAFETY: the caller passes `a_row` and `b_row` valid for `k` reads and `c` valid for one
    // write.
    unsafe {
        let mut sum = 0.0;
        for i in 0..k {
            sum += (*a_row.add(i)).to_f32() * (*b_row.add(i)).to_f32();
        }
        *c = sum;
    }
}

#[inline(always)]
unsafe fn scalar_vec_dot_bf16(a_row: *const bf16, b_row: *const bf16, c: *mut f32, k: usize) {
    // SAFETY: the caller passes `a_row` and `b_row` valid for `k` reads and `c` valid for one
    // write.
    unsafe {
        let mut sum = 0.0;
        for i in 0..k {
            sum += (*a_row.add(i)).to_f32() * (*b_row.add(i)).to_f32();
        }
        *c = sum;
    }
}

#[inline(always)]
unsafe fn scalar_vec_add_f16(a_row: *const f16, b_row: *const f16, c: *mut f16, k: usize) {
    // SAFETY: the caller passes `a_row` and `b_row` valid for `k` reads and `c` valid for `k`
    // writes.
    unsafe {
        for i in 0..k {
            *c.add(i) = *a_row.add(i) + *b_row.add(i);
        }
    }
}

#[inline(always)]
unsafe fn scalar_vec_add_bf16(a_row: *const bf16, b_row: *const bf16, c: *mut bf16, k: usize) {
    // SAFETY: the caller passes `a_row` and `b_row` valid for `k` reads and `c` valid for `k`
    // writes.
    unsafe {
        for i in 0..k {
            *c.add(i) = *a_row.add(i) + *b_row.add(i);
        }
    }
}

#[inline(always)]
unsafe fn scalar_vec_scalar_add_f16(scalar: f16, xs: *const f16, ys: *mut f16, k: usize) {
    // SAFETY: the caller passes `xs` valid for `k` reads and `ys` valid for `k` writes.
    unsafe {
        for i in 0..k {
            *ys.add(i) = *xs.add(i) + scalar;
        }
    }
}

#[inline(always)]
unsafe fn scalar_vec_scalar_add_bf16(scalar: bf16, xs: *const bf16, ys: *mut bf16, k: usize) {
    // SAFETY: the caller passes `xs` valid for `k` reads and `ys` valid for `k` writes.
    unsafe {
        for i in 0..k {
            *ys.add(i) = *xs.add(i) + scalar;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn use_avx2() -> bool {
    features::get().avx2
}

/// Whether the F16 kernels of `avx` run: they need AVX2, and F16C when the build enables it.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn use_avx2_f16() -> bool {
    let features = features::get();
    features.avx2 && (!cfg!(target_feature = "f16c") || features.f16c)
}

/// Whether the F16 kernels of `neon` run: a build that enables FP16 arithmetic computes in it,
/// which the CPU must have, and any other build widens to F32 with baseline instructions.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn use_neon_f16() -> bool {
    !cfg!(target_feature = "fp16") || features::get().fp16
}

/// Whether the BF16 kernels of `neon` run: a build that enables BF16 arithmetic computes in it,
/// which the CPU must have, and any other build widens to F32 with baseline instructions.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn use_neon_bf16() -> bool {
    !cfg!(target_feature = "bf16") || features::get().bf16
}

#[inline(always)]
pub(crate) unsafe fn vec_dot_f32(a_row: *const f32, b_row: *const f32, c: *mut f32, k: usize) {
    // SAFETY: the caller passes `a_row` and `b_row` valid for `k` reads and `c` valid for one
    // write; each instruction set runs only where `features::get` reports it or the target's
    // baseline holds it.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        if use_avx2() {
            return avx::vec_dot_f32(a_row, b_row, c, k);
        }
        #[cfg(target_arch = "aarch64")]
        {
            neon::vec_dot_f32(a_row, b_row, c, k)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            scalar_vec_dot_f32(a_row, b_row, c, k)
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn vec_sum(row: *const f32, b: *mut f32, k: usize) {
    // SAFETY: the caller passes `row` valid for `k` reads and `b` valid for one write; each
    // instruction set runs only where `features::get` reports it or the target's baseline holds
    // it.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        if use_avx2() {
            return avx::vec_sum(row, b, k);
        }
        #[cfg(target_arch = "aarch64")]
        {
            neon::vec_sum(row, b, k)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            scalar_vec_sum(row, b, k)
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn vec_dot_f16(a_row: *const f16, b_row: *const f16, c: *mut f32, k: usize) {
    // SAFETY: the caller passes `a_row` and `b_row` valid for `k` reads and `c` valid for one
    // write; each instruction set runs only where `features::get` reports it.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        if use_avx2_f16() {
            return avx::vec_dot_f16(a_row, b_row, c, k);
        }
        #[cfg(target_arch = "aarch64")]
        if use_neon_f16() {
            return neon::vec_dot_f16(a_row, b_row, c, k);
        }
        scalar_vec_dot_f16(a_row, b_row, c, k)
    }
}

#[inline(always)]
pub(crate) unsafe fn vec_dot_bf16(a_row: *const bf16, b_row: *const bf16, c: *mut f32, k: usize) {
    // SAFETY: the caller passes `a_row` and `b_row` valid for `k` reads and `c` valid for one
    // write; each instruction set runs only where `features::get` reports it.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        if use_avx2() {
            return avx::vec_dot_bf16(a_row, b_row, c, k);
        }
        #[cfg(target_arch = "aarch64")]
        if use_neon_bf16() {
            return neon::vec_dot_bf16(a_row, b_row, c, k);
        }
        scalar_vec_dot_bf16(a_row, b_row, c, k)
    }
}

#[inline(always)]
pub(crate) unsafe fn vec_add_f16(a_row: *const f16, b_row: *const f16, c: *mut f16, k: usize) {
    // SAFETY: the caller passes `a_row` and `b_row` valid for `k` reads and `c` valid for `k`
    // writes; each instruction set runs only where `features::get` reports it.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        if use_avx2_f16() {
            return avx::vec_add_f16(a_row, b_row, c, k);
        }
        #[cfg(target_arch = "aarch64")]
        if use_neon_f16() {
            return neon::vec_add_f16(a_row, b_row, c, k);
        }
        scalar_vec_add_f16(a_row, b_row, c, k)
    }
}

#[inline(always)]
pub(crate) unsafe fn vec_add_bf16(a_row: *const bf16, b_row: *const bf16, c: *mut bf16, k: usize) {
    // SAFETY: the caller passes `a_row` and `b_row` valid for `k` reads and `c` valid for `k`
    // writes; each instruction set runs only where `features::get` reports it.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        if use_avx2() {
            return avx::vec_add_bf16(a_row, b_row, c, k);
        }
        #[cfg(target_arch = "aarch64")]
        if use_neon_bf16() {
            return neon::vec_add_bf16(a_row, b_row, c, k);
        }
        scalar_vec_add_bf16(a_row, b_row, c, k)
    }
}

#[inline(always)]
pub(crate) unsafe fn vec_scalar_add_f16(scalar: f16, xs: *const f16, ys: *mut f16, k: usize) {
    // SAFETY: the caller passes `xs` valid for `k` reads and `ys` valid for `k` writes; each
    // instruction set runs only where `features::get` reports it.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        if use_avx2_f16() {
            return avx::vec_scalar_add_f16(scalar, xs, ys, k);
        }
        #[cfg(target_arch = "aarch64")]
        if use_neon_f16() {
            return neon::vec_scalar_add_f16(scalar, xs, ys, k);
        }
        scalar_vec_scalar_add_f16(scalar, xs, ys, k)
    }
}

#[inline(always)]
pub(crate) unsafe fn vec_mul_bf16(a_row: *const bf16, b_row: *const bf16, c: *mut bf16, k: usize) {
    // SAFETY: the caller passes `a_row` and `b_row` valid for `k` reads and `c` valid for `k`
    // writes; each instruction set runs only where `features::get` reports it.
    unsafe {
        #[cfg(target_arch = "aarch64")]
        if use_neon_bf16() {
            return neon::vec_mul_bf16(a_row, b_row, c, k);
        }
        for j in 0..k {
            *c.add(j) = *a_row.add(j) * *b_row.add(j);
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn vec_scalar_add_bf16(scalar: bf16, xs: *const bf16, ys: *mut bf16, k: usize) {
    // SAFETY: the caller passes `xs` valid for `k` reads and `ys` valid for `k` writes; each
    // instruction set runs only where `features::get` reports it.
    unsafe {
        #[cfg(target_arch = "x86_64")]
        if use_avx2() {
            return avx::vec_scalar_add_bf16(scalar, xs, ys, k);
        }
        #[cfg(target_arch = "aarch64")]
        if use_neon_bf16() {
            return neon::vec_scalar_add_bf16(scalar, xs, ys, k);
        }
        scalar_vec_scalar_add_bf16(scalar, xs, ys, k)
    }
}
