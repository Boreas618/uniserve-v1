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
use crate::{DataType, Device, Error, Result};

/// The largest tensor rank a tensor map describes.
const MAX_RANK: usize = 5;
/// The smallest tensor rank this platform's driver encodes. The CUDA
/// documentation allows rank one, and the driver on this platform refuses
/// every rank-one description it was given — with no stride array, with a
/// valid one, and at three box and dimension sizes — while rank two encodes.
/// The API takes what the driver takes; a one-dimensional tensor is described
/// as rank two with a trailing dimension of one.
const MIN_RANK: usize = 2;

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
    /// Encodes the tensor map `spec` describes on `device`.
    ///
    /// The driver call needs the device's context current on the calling
    /// thread; this function enters it itself, as every other `Device`-based
    /// entry point of the crate does, so a caller never pushes a context.
    ///
    /// # Errors
    ///
    /// Fails when the rank is not 2 to 5 (the driver refuses a rank-one map;
    /// see [`TensorMapSpec::check`]), when a per-dimension array does not
    /// match the rank, or with the driver's status when it rejects the
    /// description (alignment, box size against the swizzle, and so on).
    pub fn encode_tiled(device: &Device, spec: &TensorMapSpec<'_>) -> Result<Self> {
        let rank = spec.check()?;

        let mut raw = RawTensorMap { opaque: [0; 16] };
        let driver = driver::driver()?;
        let _context = device.enter()?;
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

impl TensorMapSpec<'_> {
    /// Checks the description against the driver's requirements that can be
    /// checked without a device, and answers its rank.
    ///
    /// A rank-one description is refused: the CUDA documentation allows it,
    /// and this platform's driver encodes none of the rank-one maps it was
    /// given, while rank two encodes. A one-dimensional tensor is therefore
    /// described as rank two with a trailing dimension of one.
    ///
    /// `encode_tiled` calls it before the driver sees the description; a
    /// caller may call it earlier, at the point the description is built.
    pub fn check(&self) -> Result<usize> {
        let rank = self.global_dim.len();
        if !(MIN_RANK..=MAX_RANK).contains(&rank) {
            return Err(Error::invalid(format!(
                "a tensor map has rank {MIN_RANK} to {MAX_RANK}, got {rank}; \
                 the driver refuses a rank-one map, so a one-dimensional tensor is \
                 described as rank two with a trailing dimension of one"
            )));
        }
        // The driver takes one stride per dimension but the innermost, and one
        // box dimension and element stride per dimension.
        for (name, length, expected) in [
            ("global_strides", self.global_strides.len(), rank - 1),
            ("box_dim", self.box_dim.len(), rank),
            ("element_strides", self.element_strides.len(), rank),
        ] {
            if length != expected {
                return Err(Error::invalid(format!(
                    "{name} has {length} entries, not {expected} for rank {rank}"
                )));
            }
        }
        if self.global_address == 0 {
            return Err(Error::invalid("a tensor map addresses no storage"));
        }
        Ok(rank)
    }
}

/// `CUtensorMapDataType`: the element type of the described tensor.
///
/// This is not a second element type of the crate — [`DataType`] is the one a
/// buffer, kernel, collective or convolution is told its storage holds. A
/// tensor map's code set is wider than [`DataType`]'s: it names TF32, an FP32 that
/// flushes denormals, and packed 4- and 6-bit formats that no device buffer
/// has an element size for, which is why the driver's codes are spelled out
/// here. [`TensorDataType::from_dtype`] converts the types both name.
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

impl TensorDataType {
    /// The tensor-map code for `dtype`, when the driver names that element
    /// type; the signed narrow and packed formats of [`DataType`] have none.
    pub fn from_dtype(dtype: DataType) -> Option<Self> {
        Some(match dtype {
            DataType::U8 => Self::Uint8,
            DataType::I32 => Self::Int32,
            DataType::U64 => Self::Uint64,
            DataType::I64 => Self::Int64,
            DataType::F16 => Self::Float16,
            DataType::F32 => Self::Float32,
            DataType::F64 => Self::Float64,
            DataType::Bf16 => Self::BFloat16,
            DataType::U32 => Self::Uint32,
            DataType::I16 | DataType::I8 | DataType::F8E4M3 | DataType::F8E5M2 => return None,
        })
    }
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
            // The documentation allows a rank-one tensor map and the driver on
            // this platform encodes none, so the description is refused here.
            TensorMapSpec {
                global_dim: &[64],
                global_strides: &[],
                box_dim: &[4],
                element_strides: &[1],
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
                matches!(spec.check(), Err(Error::InvalidArgument(_))),
                "{spec:?}"
            );
        }
    }

    #[test]
    fn the_element_types_both_names_share_convert() {
        use crate::DataType;

        assert_eq!(
            TensorDataType::from_dtype(DataType::F32),
            Some(TensorDataType::Float32)
        );
        assert_eq!(
            TensorDataType::from_dtype(DataType::Bf16),
            Some(TensorDataType::BFloat16)
        );
        // No tensor-map code names a signed byte or an FP8 element.
        assert_eq!(TensorDataType::from_dtype(DataType::I8), None);
        assert_eq!(TensorDataType::from_dtype(DataType::F8E4M3), None);
    }
}
