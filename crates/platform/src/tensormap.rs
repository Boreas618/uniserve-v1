//! TMA tensor maps: `cuTensorMapEncodeTiled`.
//!
//! A tensor map describes a multidimensional tensor in device storage to a
//! kernel that moves it with the tensor memory accelerator: the data type,
//! the global dimensions and strides, the box the kernel moves per
//! instruction, the shared-memory swizzle, and how out-of-bounds reads are
//! filled. It is passed to a kernel by value as a `__grid_constant__`
//! parameter, so [`TensorMap::as_bytes`] is the kernel parameter's value.

use std::ffi::c_void;

use crate::driver::{self, TensorMap as RawTensorMap};
use crate::{Error, Result};

/// The largest tensor rank a tensor map describes.
const MAX_RANK: usize = 5;

/// A tensor map (`CUtensorMap`), 64-byte aligned as the driver requires, and
/// filled by [`TensorMap::encode_tiled`].
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug)]
pub struct TensorMap {
    raw: RawTensorMap,
}

// SAFETY: the tensor map is plain data: bytes the driver wrote and the kernel
// reads. No driver handle is involved.
unsafe impl Send for TensorMap {}
// SAFETY: as for Send above.
unsafe impl Sync for TensorMap {}

impl TensorMap {
    /// Encodes the tensor map `spec` describes.
    ///
    /// # Errors
    ///
    /// Fails when the rank is not 1 to 5, when a per-dimension array does not
    /// match the rank, or with the driver's status when it rejects the
    /// description (alignment, box size against the swizzle, and so on).
    pub fn encode_tiled(spec: &TensorMapSpec<'_>) -> Result<Self> {
        let rank = spec.global_dim.len();
        if rank == 0 || rank > MAX_RANK {
            return Err(Error::invalid(format!(
                "a tensor map has rank 1 to {MAX_RANK}, got {rank}"
            )));
        }
        // The driver takes one stride per dimension but the innermost, and one
        // box dimension and element stride per dimension.
        for (name, length, expected) in [
            ("global_strides", spec.global_strides.len(), rank - 1),
            ("box_dim", spec.box_dim.len(), rank),
            ("element_strides", spec.element_strides.len(), rank),
        ] {
            if length != expected {
                return Err(Error::invalid(format!(
                    "{name} has {length} entries, not {expected} for rank {rank}"
                )));
            }
        }
        if spec.global_address == 0 {
            return Err(Error::invalid("a tensor map addresses no storage"));
        }

        let mut raw = RawTensorMap { opaque: [0; 16] };
        let driver = driver::driver()?;
        driver.check(
            // SAFETY: every array holds one entry per dimension of the tensor,
            // and the driver writes the 128-byte descriptor to the local
            // `raw`, whose alignment `TensorMap` guarantees.
            unsafe {
                (driver.tensor_map_encode_tiled)(
                    &mut raw,
                    spec.data_type as i32,
                    rank as u32,
                    spec.global_address as *mut c_void,
                    spec.global_dim.as_ptr(),
                    // A rank-1 tensor has no outer strides; the driver reads
                    // none, and a null array says so.
                    if spec.global_strides.is_empty() {
                        std::ptr::null()
                    } else {
                        spec.global_strides.as_ptr()
                    },
                    spec.box_dim.as_ptr(),
                    spec.element_strides.as_ptr(),
                    spec.interleave as i32,
                    spec.swizzle as i32,
                    spec.l2_promotion as i32,
                    spec.oob_fill as i32,
                )
            },
            "cuTensorMapEncodeTiled",
        )?;
        Ok(Self { raw })
    }

    /// The descriptor's 128 bytes, as a kernel takes it by value.
    pub fn as_bytes(&self) -> &[u8; 128] {
        // SAFETY: `CUtensorMap` is 128 opaque bytes with no padding, so the
        // same memory is a byte array.
        unsafe { &*(self.raw.opaque.as_ptr().cast::<[u8; 128]>()) }
    }
}

/// What a tensor map describes: the tensor's layout in device storage, the
/// box a TMA instruction moves, and how the move is shaped.
///
/// Every per-dimension array has one entry per dimension, innermost first, as
/// `cuTensorMapEncodeTiled` takes them.
#[derive(Clone, Copy, Debug)]
pub struct TensorMapSpec<'a> {
    /// The element type the kernel reads or writes.
    pub data_type: TensorDataType,
    /// The tensor's base address in device storage; 16-byte aligned, 32-byte
    /// aligned for a 32-byte interleave.
    pub global_address: u64,
    /// The tensor's size in each dimension, in elements.
    pub global_dim: &'a [u64],
    /// The stride of each dimension but the innermost, in bytes, one entry per
    /// dimension after the first: the innermost dimension's stride is the
    /// element size and is not given.
    pub global_strides: &'a [u64],
    /// The box's size in each dimension, in elements.
    pub box_dim: &'a [u32],
    /// The step between elements the box takes in each dimension.
    pub element_strides: &'a [u32],
    /// The interleaved layout of the tensor in global memory.
    pub interleave: Interleave,
    /// The shared-memory swizzle the kernel reads the box with.
    pub swizzle: Swizzle,
    /// The L2 promotion applied to the tensor's accesses.
    pub l2_promotion: L2Promotion,
    /// The value an out-of-bounds read returns.
    pub oob_fill: OobFill,
}

/// `CUtensorMapDataType`: the element type of the described tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum TensorDataType {
    Uint8 = 0,
    Uint16 = 1,
    Uint32 = 2,
    Int32 = 3,
    Uint64 = 4,
    Int64 = 5,
    Float16 = 6,
    Float32 = 7,
    Float64 = 8,
    BFloat16 = 9,
    /// FP32 with denormals flushed to zero.
    Float32Ftz = 10,
    /// TF32, the 19-bit format the tensor cores take.
    TFloat32 = 11,
    /// TF32 with denormals flushed to zero.
    TFloat32Ftz = 12,
    /// Packed 4-bit unsigned values, 16 per 8 bytes.
    Uint4Align8 = 13,
    /// Packed 4-bit unsigned values, 16 per 16-byte slot.
    Uint4Align16 = 14,
    /// Packed 6-bit unsigned values, 16 per 16-byte slot.
    Uint6Align16 = 15,
}

/// `CUtensorMapInterleave`: the interleaved layout of the tensor in global
/// memory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(i32)]
pub enum Interleave {
    #[default]
    None = 0,
    B16 = 1,
    B32 = 2,
}

/// `CUtensorMapSwizzle`: how the shared-memory destination of a box is
/// swizzled to avoid bank conflicts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(i32)]
pub enum Swizzle {
    #[default]
    None = 0,
    /// Swizzle 16-byte chunks within a 32-byte span.
    B32 = 1,
    /// Swizzle 16-byte chunks within a 64-byte span.
    B64 = 2,
    /// Swizzle 16-byte chunks within a 128-byte span.
    B128 = 3,
    /// Swizzle 32-byte chunks within a 128-byte span.
    B128Atom32 = 4,
    /// Swizzle 32-byte chunks within a 128-byte span, swapping the lower and
    /// upper 8 bytes of every alternate row's 16-byte chunk.
    B128Atom32Flip8 = 5,
    /// Swizzle 64-byte chunks within a 128-byte span.
    B128Atom64 = 6,
}

/// `CUtensorMapL2promotion`: the L2 promotion applied to the tensor's
/// accesses.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(i32)]
pub enum L2Promotion {
    #[default]
    None = 0,
    B64 = 1,
    B128 = 2,
    B256 = 3,
}

/// `CUtensorMapFloatOOBfill`: what a TMA read outside the tensor returns.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(i32)]
pub enum OobFill {
    /// Zero in the element's format.
    #[default]
    None = 0,
    /// The special NaN that requests zero-fill in a following operation,
    /// which avoids a separate predicate; floating-point types only.
    NanRequestZeroFma = 1,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A description whose arrays do not match its rank is the caller's
    /// mistake, named before the driver sees it; the checks below therefore
    /// need no device.
    #[test]
    fn a_tensor_map_description_is_checked_against_its_rank() {
        let address = 1 << 20;
        let dim = [16u64, 16];
        let strides = [64u64];
        let boxes = [4u32, 4];
        let element = [1u32, 1];
        let valid = TensorMapSpec {
            data_type: TensorDataType::Float32,
            global_address: address,
            global_dim: &dim,
            global_strides: &strides,
            box_dim: &boxes,
            element_strides: &element,
            interleave: Interleave::None,
            swizzle: Swizzle::None,
            l2_promotion: L2Promotion::None,
            oob_fill: OobFill::None,
        };

        for spec in [
            TensorMapSpec {
                global_dim: &[16, 16, 16, 16, 16, 16],
                ..valid
            },
            TensorMapSpec {
                global_strides: &[64, 128],
                ..valid
            },
            TensorMapSpec {
                box_dim: &[4],
                ..valid
            },
            TensorMapSpec {
                element_strides: &[1, 1, 1],
                ..valid
            },
            TensorMapSpec {
                global_address: 0,
                ..valid
            },
        ] {
            assert!(
                matches!(
                    TensorMap::encode_tiled(&spec),
                    Err(Error::InvalidArgument(_))
                ),
                "{spec:?}"
            );
        }
    }
}
