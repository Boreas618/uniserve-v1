//! DLPack export of Rust-owned storage.
//!
//! The worker allocates runtime storage itself (device memory, pinned host
//! memory, shared mappings) and hands Python views of it. A view is exported
//! as a versioned DLPack tensor whose deleter releases one strong reference to
//! the owning allocation, so Python keeps the memory alive for as long as any
//! torch tensor derived from it exists, and the allocation is released only
//! after both sides have dropped their references.

use std::ffi::c_void;
use std::sync::Arc;

use tvm_ffi::tvm_ffi_sys::dlpack::{DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLTensor};
use tvm_ffi::{Any, Tensor};
use uniserve_cuda::DeviceId;

use crate::{Error, Result};

/// The kind of memory a view's bytes live in: host memory, page-locked
/// host memory, or the memory of one CUDA device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryKind {
    Host,
    /// Page-locked host memory, readable by CUDA copies.
    PinnedHost,
    Cuda(DeviceId),
}

impl MemoryKind {
    /// The CUDA device of device memory.
    pub(crate) fn cuda(self) -> Option<DeviceId> {
        match self {
            Self::Cuda(device) => Some(device),
            Self::Host | Self::PinnedHost => None,
        }
    }

    /// The device spelling of transfer locators (`cuda:<ordinal>`, or
    /// `cpu` for host memory of either kind), which consumers read back
    /// with [`Self::from_locator`].
    pub(crate) fn locator_name(self) -> String {
        match self {
            Self::Cuda(device) => device.to_string(),
            Self::Host | Self::PinnedHost => "cpu".to_owned(),
        }
    }

    /// The memory a transfer locator's device names: CUDA memory for
    /// `cuda:<ordinal>`, host memory otherwise.
    pub(crate) fn from_locator(name: &str) -> Self {
        name.strip_prefix("cuda:")
            .and_then(|ordinal| ordinal.parse().ok())
            .map_or(Self::Host, |ordinal| Self::Cuda(DeviceId(ordinal)))
    }

    fn to_dlpack(self) -> DLDevice {
        match self {
            Self::Host => DLDevice {
                device_type: DLDeviceType::kDLCPU,
                device_id: 0,
            },
            // Page-locked memory crosses as CPU memory: torch imports no
            // `kDLCUDAHost` tensor, and it recognizes the pages as pinned
            // from the pointer itself, so non-blocking copies stay
            // asynchronous.
            Self::PinnedHost => DLDevice {
                device_type: DLDeviceType::kDLCPU,
                device_id: 0,
            },
            Self::Cuda(device) => DLDevice {
                device_type: DLDeviceType::kDLCUDA,
                device_id: device.0,
            },
        }
    }
}

/// Element types of Rust-owned storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    U8,
    I8,
    I16,
    I32,
    I64,
    U32,
    F16,
    BF16,
    F32,
    F64,
    F8E4M3,
    F8E5M2,
}

impl DType {
    pub fn bytes(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 | Self::F8E4M3 | Self::F8E5M2 => 1,
            Self::I16 | Self::F16 | Self::BF16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }

    /// Parse a torch dtype name (`bfloat16`, `torch.int64`, ...).
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        Some(match name.trim_start_matches("torch.") {
            "bool" => Self::Bool,
            "uint8" => Self::U8,
            "int8" => Self::I8,
            "int16" => Self::I16,
            "int32" => Self::I32,
            "int64" => Self::I64,
            "uint32" => Self::U32,
            "float16" => Self::F16,
            "bfloat16" => Self::BF16,
            "float32" => Self::F32,
            "float64" => Self::F64,
            "float8_e4m3fn" => Self::F8E4M3,
            "float8_e5m2" => Self::F8E5M2,
            _ => return None,
        })
    }

    /// Parse a torch dtype name, failing for any other.
    pub(crate) fn parse(name: &str) -> Result<Self> {
        Self::from_name(name)
            .ok_or_else(|| Error::Invalid(format!("unknown element type {name:?}")))
    }

    /// The torch spelling of this dtype, which transfer locators and the
    /// Python interface use.
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::U8 => "uint8",
            Self::I8 => "int8",
            Self::I16 => "int16",
            Self::I32 => "int32",
            Self::I64 => "int64",
            Self::U32 => "uint32",
            Self::F16 => "float16",
            Self::BF16 => "bfloat16",
            Self::F32 => "float32",
            Self::F64 => "float64",
            Self::F8E4M3 => "float8_e4m3fn",
            Self::F8E5M2 => "float8_e5m2",
        }
    }

    fn to_dlpack(self) -> DLDataType {
        let (code, bits) = match self {
            Self::Bool => (DLDataTypeCode::kDLBool, 8),
            Self::U8 => (DLDataTypeCode::kDLUInt, 8),
            Self::U32 => (DLDataTypeCode::kDLUInt, 32),
            Self::I8 => (DLDataTypeCode::kDLInt, 8),
            Self::I16 => (DLDataTypeCode::kDLInt, 16),
            Self::I32 => (DLDataTypeCode::kDLInt, 32),
            Self::I64 => (DLDataTypeCode::kDLInt, 64),
            Self::F16 => (DLDataTypeCode::kDLFloat, 16),
            Self::BF16 => (DLDataTypeCode::kDLBfloat, 16),
            Self::F32 => (DLDataTypeCode::kDLFloat, 32),
            Self::F64 => (DLDataTypeCode::kDLFloat, 64),
            Self::F8E4M3 => (DLDataTypeCode::kDLFloat8_e4m3fn, 8),
            Self::F8E5M2 => (DLDataTypeCode::kDLFloat8_e5m2, 8),
        };
        DLDataType {
            code: code as u8,
            bits,
            lanes: 1,
        }
    }
}

/// The storage element type of a head-to-rank message's dtype; messages keep
/// their own serialized enum, converted here once.
impl From<uniserve_ipc::DType> for DType {
    fn from(dtype: uniserve_ipc::DType) -> Self {
        match dtype {
            uniserve_ipc::DType::U8 => Self::U8,
            uniserve_ipc::DType::I16 => Self::I16,
            uniserve_ipc::DType::I32 => Self::I32,
            uniserve_ipc::DType::I64 => Self::I64,
            uniserve_ipc::DType::F16 => Self::F16,
            uniserve_ipc::DType::BF16 => Self::BF16,
            uniserve_ipc::DType::F32 => Self::F32,
        }
    }
}

/// The message dtype of a storage element type, for the element types
/// messages carry.
impl TryFrom<DType> for uniserve_ipc::DType {
    type Error = Error;

    fn try_from(dtype: DType) -> Result<Self> {
        Ok(match dtype {
            DType::U8 => Self::U8,
            DType::I16 => Self::I16,
            DType::I32 => Self::I32,
            DType::I64 => Self::I64,
            DType::F16 => Self::F16,
            DType::BF16 => Self::BF16,
            DType::F32 => Self::F32,
            other => {
                return Err(Error::Unsupported(format!(
                    "{} has no message dtype",
                    other.name()
                )));
            }
        })
    }
}

/// Memory a view can borrow. Implementors keep the bytes valid and unmoved
/// for as long as the value is alive.
pub trait Backing: Send + Sync + 'static {
    fn device(&self) -> MemoryKind;
    /// Base address: a device pointer for CUDA memory, a host pointer
    /// otherwise.
    fn address(&self) -> u64;
    /// Length of the allocation in bytes.
    fn bytes(&self) -> usize;
}

#[repr(C)]
struct DLPackVersion {
    major: u32,
    minor: u32,
}

#[repr(C)]
struct DLManagedTensorVersioned {
    version: DLPackVersion,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensorVersioned)>,
    flags: u64,
    dl_tensor: DLTensor,
}

/// Retained by the DLPack capsule: the owning allocation and the shape and
/// stride arrays the `DLTensor` points into.
struct Exported {
    _backing: Arc<dyn Backing>,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

unsafe extern "C" fn release(managed: *mut DLManagedTensorVersioned) {
    // SAFETY: `managed` and its context were created by `export` with
    // `Box::into_raw` and DLPack calls the deleter exactly once.
    unsafe {
        drop(Box::from_raw((*managed).manager_ctx.cast::<Exported>()));
        drop(Box::from_raw(managed));
    }
}

/// Export `shape`/`strides` (in elements) starting `offset` bytes into
/// `backing` as a TVM-FFI tensor that retains `backing`.
pub(crate) fn export(
    backing: Arc<dyn Backing>,
    offset: usize,
    dtype: DType,
    shape: &[i64],
    strides: &[i64],
) -> Result<Tensor> {
    if shape.len() != strides.len() {
        return Err(Error::Invariant(
            "a view needs one stride per dimension".to_owned(),
        ));
    }
    check_bounds(backing.bytes(), offset, dtype, shape, strides)?;

    let device = backing.device().to_dlpack();
    let data = backing.address() as *mut c_void;
    let mut exported = Box::new(Exported {
        _backing: backing,
        shape: shape.to_vec(),
        strides: strides.to_vec(),
    });
    let dl_tensor = DLTensor {
        data,
        device,
        ndim: shape.len() as i32,
        dtype: dtype.to_dlpack(),
        shape: exported.shape.as_mut_ptr(),
        strides: exported.strides.as_mut_ptr(),
        byte_offset: offset as u64,
    };
    let managed = Box::into_raw(Box::new(DLManagedTensorVersioned {
        version: DLPackVersion { major: 1, minor: 0 },
        manager_ctx: Box::into_raw(exported).cast(),
        deleter: Some(release),
        flags: 0,
        dl_tensor,
    }));

    let mut handle = std::ptr::null_mut();
    // SAFETY: `managed` is a valid versioned DLPack tensor. On success TVM-FFI
    // takes ownership and calls the deleter when the tensor is released; on
    // failure ownership stays with us and we release it here.
    let status = unsafe {
        tvm_ffi::tvm_ffi_sys::TVMFFITensorFromDLPackVersioned(managed.cast(), 0, 0, &mut handle)
    };
    if status != 0 {
        // SAFETY: TVM-FFI did not take ownership of `managed`.
        unsafe { release(managed) };
        let error = tvm_ffi::Error::from_raised();
        return Err(Error::numerical("TVMFFITensorFromDLPackVersioned", &error));
    }

    // SAFETY: `handle` is an owned tensor object reference returned above.
    let any = unsafe {
        Any::from_raw_ffi_any(tvm_ffi::TVMFFIAny {
            type_index: tvm_ffi::TypeIndex::kTVMFFITensor as i32,
            small_str_len: 0,
            data_union: std::mem::transmute::<u64, tvm_ffi::tvm_ffi_sys::TVMFFIAnyDataUnion>(
                handle as u64,
            ),
        })
    };
    Tensor::try_from(any).map_err(|error| Error::numerical("tensor export", &error))
}

/// Reject views reaching outside their backing, so a bad layout fails in
/// Rust instead of corrupting memory in a kernel.
fn check_bounds(
    len: usize,
    offset: usize,
    dtype: DType,
    shape: &[i64],
    strides: &[i64],
) -> Result<()> {
    if shape.iter().any(|&extent| extent < 0) || strides.iter().any(|&stride| stride < 0) {
        return Err(Error::Invariant(
            "views require non-negative extents and strides".to_owned(),
        ));
    }
    if shape.contains(&0) {
        return if offset <= len {
            Ok(())
        } else {
            Err(Error::Invariant(
                "an empty view starts past its backing".to_owned(),
            ))
        };
    }
    let last = shape
        .iter()
        .zip(strides)
        .map(|(&extent, &stride)| (extent - 1) as u128 * stride as u128)
        .sum::<u128>();
    let end = offset as u128 + (last + 1) * dtype.bytes() as u128;
    if end > len as u128 {
        return Err(Error::Invariant(format!(
            "a view of {end} bytes exceeds its {len}-byte backing"
        )));
    }
    Ok(())
}

/// Contiguous strides, in elements, for `shape`.
pub(crate) fn contiguous_strides(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![1; shape.len()];
    for index in (0..shape.len().saturating_sub(1)).rev() {
        strides[index] = strides[index + 1] * shape[index + 1].max(1);
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_cover_the_last_element() {
        assert!(check_bounds(64, 0, DType::F32, &[4, 4], &[4, 1]).is_ok());
        assert!(check_bounds(63, 0, DType::F32, &[4, 4], &[4, 1]).is_err());
        assert!(check_bounds(64, 4, DType::F32, &[15], &[1]).is_ok());
        assert!(check_bounds(64, 8, DType::F32, &[15], &[1]).is_err());
        assert!(check_bounds(0, 0, DType::F32, &[0, 4], &[4, 1]).is_ok());
    }

    #[test]
    fn contiguous_strides_are_row_major() {
        assert_eq!(contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(contiguous_strides(&[]), Vec::<i64>::new());
    }
}
