//! The CUDA device of a GPU, and the meta device.
//!
//! A tensor's device says where its storage is. Two CUDA devices are the same device when they
//! name the same GPU. The meta device is the device of tensors that have a shape and an element
//! type and no data; it is the same device as every CUDA device, in both directions, because a
//! forward under a meta context meets loaded weights, which are on a GPU, and produces meta
//! tensors from them.
//!
//! Neither device decides what a new storage is. Storage a device is asked for comes from the
//! context installed on the thread: a block of its allocator, on the context's GPU under a
//! context that launches and meta under a meta context, whichever device value the request
//! came through.

use std::any::Any;
use std::sync::Arc;

use tensor::backend::{BackendDevice, BackendStorage};
use tensor::{CpuStorage, DType, Device, DeviceLocation, Layout, Shape};

use crate::primitives;
use crate::storage::CudaStorage;
use crate::{Error, Result, context, input};

/// The CUDA device of GPU `ordinal`.
pub fn cuda(ordinal: usize) -> Device {
    Device::Backend(Arc::new(CudaDevice::new(ordinal)))
}

/// The meta device: the device of tensors with a shape, an element type and no data.
pub fn meta() -> Device {
    Device::Backend(Arc::new(MetaDevice))
}

/// The GPU `device` names: `Some` for a CUDA device and `None` for the meta device.
///
/// # Errors
///
/// [`Error::ForeignTensor`], naming `operation`, for the CPU and for a device of another
/// backend.
pub(crate) fn gpu(device: &Device, operation: &'static str) -> Result<Option<usize>> {
    if let Device::Backend(backend) = device {
        if let Some(cuda) = backend.as_any().downcast_ref::<CudaDevice>() {
            return Ok(Some(cuda.ordinal));
        }
        if backend.as_any().is::<MetaDevice>() {
            return Ok(None);
        }
    }

    Err(Error::ForeignTensor {
        operation,
        location: device.location(),
    })
}

/// The CUDA device of one GPU.
#[derive(Debug)]
pub(crate) struct CudaDevice {
    /// The GPU's index among the host's CUDA devices.
    ordinal: usize,
}

impl CudaDevice {
    pub(crate) fn new(ordinal: usize) -> Self {
        Self { ordinal }
    }
}

/// The meta device.
#[derive(Debug)]
pub(crate) struct MetaDevice;

impl BackendDevice for CudaDevice {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn location(&self) -> DeviceLocation {
        DeviceLocation::Backend {
            kind: "cuda",
            ordinal: self.ordinal,
        }
    }

    fn same_device(&self, other: &dyn BackendDevice) -> bool {
        let other = other.as_any();

        other
            .downcast_ref::<Self>()
            .is_some_and(|other| other.ordinal == self.ordinal)
            || other.is::<MetaDevice>()
    }

    fn zeros(&self, shape: &Shape, dtype: DType) -> tensor::Result<Box<dyn BackendStorage>> {
        Ok(Box::new(zeros(shape, dtype, Some(self.ordinal))?))
    }

    unsafe fn alloc_uninit(
        &self,
        shape: &Shape,
        dtype: DType,
    ) -> tensor::Result<Box<dyn BackendStorage>> {
        let storage = CudaStorage::of_elements(shape.elem_count(), dtype, Some(self.ordinal))?;
        Ok(Box::new(storage))
    }

    fn storage_from_cpu_storage(
        &self,
        storage: &CpuStorage,
    ) -> tensor::Result<Box<dyn BackendStorage>> {
        Ok(Box::new(from_host(storage, Some(self.ordinal))?))
    }

    fn synchronize(&self) -> tensor::Result<()> {
        Ok(synchronize()?)
    }
}

impl BackendDevice for MetaDevice {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn location(&self) -> DeviceLocation {
        DeviceLocation::Backend {
            kind: "meta",
            ordinal: 0,
        }
    }

    fn same_device(&self, other: &dyn BackendDevice) -> bool {
        let other = other.as_any();

        other.is::<Self>() || other.is::<CudaDevice>()
    }

    fn zeros(&self, shape: &Shape, dtype: DType) -> tensor::Result<Box<dyn BackendStorage>> {
        Ok(Box::new(zeros(shape, dtype, None)?))
    }

    unsafe fn alloc_uninit(
        &self,
        shape: &Shape,
        dtype: DType,
    ) -> tensor::Result<Box<dyn BackendStorage>> {
        let storage = CudaStorage::of_elements(shape.elem_count(), dtype, None)?;
        Ok(Box::new(storage))
    }

    fn storage_from_cpu_storage(
        &self,
        storage: &CpuStorage,
    ) -> tensor::Result<Box<dyn BackendStorage>> {
        Ok(Box::new(from_host(storage, None)?))
    }

    fn synchronize(&self) -> tensor::Result<()> {
        Ok(synchronize()?)
    }
}

/// A storage of `shape` zero elements of `dtype` from the thread's context, asked through the
/// CUDA device of GPU `asked` or through the meta device.
fn zeros(shape: &Shape, dtype: DType, asked: Option<usize>) -> Result<CudaStorage> {
    let storage = CudaStorage::of_elements(shape.elem_count(), dtype, asked)?;
    primitives::fill(&storage, &Layout::contiguous(shape.clone()), 0)?;
    Ok(storage)
}

/// The host-to-device copy: a storage of the thread's context with the elements of `storage`.
///
/// The block comes from the context's allocator at addresses no tensor of this forward had, so
/// an input copy issued before a replay is not overwritten by the replay's earlier kernels; the
/// bytes are written into the context's
/// input buffer and copied, recorded as an input copy or only counted, as the context is
/// ([`input::copy_to_device`]).
///
/// # Errors
///
/// As [`CudaStorage::allocate`] and [`input::copy_to_device`]; [`Error::ElementType`] for a
/// 6-bit element type.
fn from_host(storage: &CpuStorage, asked: Option<usize>) -> Result<CudaStorage> {
    // The 6-bit elements have no byte layout on the device, as for every other storage.
    if matches!(storage, CpuStorage::F6E2M3(_) | CpuStorage::F6E3M2(_)) {
        return Err(Error::ElementType {
            operation: "a host-to-device copy",
            dtype: storage.dtype(),
            reason: "the element is narrower than a byte and has no packed layout here",
        });
    }

    let bytes = host_bytes(storage);
    let device = CudaStorage::allocate_unused(bytes.len() as u64, storage.dtype(), asked)?;

    // SAFETY: the address is the block just allocated for exactly these bytes. The storage that
    // owns it is returned to the caller's tensor, which the forward keeps for as long as a
    // kernel or a recorded input copy reads it.
    unsafe { input::copy_to_device(device.address(), bytes)? };
    Ok(device)
}

/// The elements of a host storage as the bytes the device holds them in.
fn host_bytes(storage: &CpuStorage) -> &[u8] {
    /// The memory of `values`. Every element type of a host storage is plain data without
    /// padding, so each of its bytes is initialized.
    fn bytes_of<T: Copy>(values: &[T]) -> &[u8] {
        // SAFETY: the slice's memory is `size_of_val(values)` initialized bytes, borrowed for
        // the slice's lifetime.
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), size_of_val(values)) }
    }

    match storage {
        CpuStorage::U8(values) => bytes_of(values),
        CpuStorage::U32(values) => bytes_of(values),
        CpuStorage::I16(values) => bytes_of(values),
        CpuStorage::I32(values) => bytes_of(values),
        CpuStorage::I64(values) => bytes_of(values),
        CpuStorage::BF16(values) => bytes_of(values),
        CpuStorage::F16(values) => bytes_of(values),
        CpuStorage::F32(values) => bytes_of(values),
        CpuStorage::F64(values) => bytes_of(values),
        CpuStorage::F8E4M3(values) => bytes_of(values),
        // The packed types are held on the host as their bytes.
        CpuStorage::F6E2M3(values)
        | CpuStorage::F6E3M2(values)
        | CpuStorage::F4(values)
        | CpuStorage::F8E8M0(values) => values,
    }
}

/// Waits for everything issued on the stream of the thread's context. A meta context issued
/// nothing.
fn synchronize() -> Result<()> {
    let context = context::current("a wait for the device")?;

    match context.stream() {
        Some(stream) => Ok(stream.synchronize()?),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use tensor::{Device, DeviceLocation};

    use super::{cuda, meta};

    #[test]
    fn two_cuda_devices_are_the_same_device_when_their_ordinals_agree() {
        assert!(cuda(1).same_device(&cuda(1)));
        assert!(!cuda(0).same_device(&cuda(1)));
    }

    #[test]
    fn the_meta_device_is_the_same_device_as_every_cuda_device_in_both_directions() {
        assert!(meta().same_device(&meta()));
        assert!(meta().same_device(&cuda(0)));
        assert!(cuda(3).same_device(&meta()));
    }

    #[test]
    fn neither_device_is_the_cpu() {
        assert!(!cuda(0).same_device(&Device::Cpu));
        assert!(!Device::Cpu.same_device(&meta()));
    }

    #[test]
    fn a_device_reports_its_kind_and_ordinal() {
        assert_eq!(
            cuda(2).location(),
            DeviceLocation::Backend {
                kind: "cuda",
                ordinal: 2
            }
        );
        assert_eq!(
            meta().location(),
            DeviceLocation::Backend {
                kind: "meta",
                ordinal: 0
            }
        );
    }
}
