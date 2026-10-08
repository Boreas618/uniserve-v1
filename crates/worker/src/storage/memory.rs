//! Rust-owned allocations and the typed views exported to Python.
//!
//! Runtime storage (KV planes, graph inputs and outputs, workspaces, staging)
//! is allocated here and handed to Python as DLPack tensors. A [`Memory`]
//! value owns one allocation; a [`StorageView`] is a typed, strided window
//! into it that retains the allocation. Dropping the last Rust and Python
//! reference releases the memory through `uniserve-cuda`'s reclaimer, which
//! never waits for device work: owners must retire all device accesses to an
//! allocation before dropping it.

use std::sync::Arc;

use uniserve_cuda::{DeviceAllocation, DeviceId, PinnedBuffer};

use crate::ffi::{Backing, DType, Device, Tensor, contiguous_strides, export};
use crate::{Error, Result};

/// One allocation owned by the worker.
pub struct Memory {
    kind: Kind,
}

enum Kind {
    Device(DeviceAllocation),
    Pinned(PinnedBuffer),
    /// Pageable host memory for CPU lanes. Boxed so its address is stable.
    Host(Box<[u8]>),
}

impl Memory {
    /// Device memory on `device`. Contents are uninitialized.
    pub fn device(device: i32, bytes: usize) -> Result<Arc<Self>> {
        let allocation = DeviceAllocation::linear(DeviceId(device), bytes)?;
        Ok(Arc::new(Self {
            kind: Kind::Device(allocation),
        }))
    }

    /// Page-locked host memory usable as a CUDA copy source or destination
    /// for `device`. Contents are uninitialized.
    pub fn pinned(device: i32, bytes: usize) -> Result<Arc<Self>> {
        let buffer = PinnedBuffer::new(DeviceId(device), bytes)?;
        Ok(Arc::new(Self {
            kind: Kind::Pinned(buffer),
        }))
    }

    /// Zeroed pageable host memory.
    pub fn host(bytes: usize) -> Arc<Self> {
        Arc::new(Self {
            kind: Kind::Host(vec![0; bytes].into_boxed_slice()),
        })
    }

    /// Adopt an existing device allocation, for example an exportable VMM
    /// allocation created by a transport.
    pub fn from_device(allocation: DeviceAllocation) -> Arc<Self> {
        Arc::new(Self {
            kind: Kind::Device(allocation),
        })
    }

    /// The allocation's host-visible bytes. `None` for device memory.
    ///
    /// # Safety
    ///
    /// No device or other host access may write the returned range while the
    /// slice is alive.
    pub unsafe fn host_bytes(&self) -> Option<&[u8]> {
        match &self.kind {
            Kind::Device(_) => None,
            // SAFETY: the buffer is owned by `self` and the caller excludes
            // concurrent writers.
            Kind::Pinned(buffer) => {
                Some(unsafe { std::slice::from_raw_parts(buffer.as_ptr(), buffer.len()) })
            }
            Kind::Host(bytes) => Some(bytes),
        }
    }
}

impl Backing for Memory {
    fn device(&self) -> Device {
        match &self.kind {
            Kind::Device(allocation) => Device::Cuda(allocation.device().0),
            Kind::Pinned(_) => Device::CudaHost,
            Kind::Host(_) => Device::Cpu,
        }
    }

    fn address(&self) -> u64 {
        match &self.kind {
            Kind::Device(allocation) => allocation.address(),
            Kind::Pinned(buffer) => buffer.address(),
            Kind::Host(bytes) => bytes.as_ptr() as u64,
        }
    }

    fn len(&self) -> usize {
        match &self.kind {
            Kind::Device(allocation) => allocation.len(),
            Kind::Pinned(buffer) => buffer.len(),
            Kind::Host(bytes) => bytes.len(),
        }
    }
}

/// A typed, strided window into an allocation.
///
/// `offset` is in bytes; `shape` and `strides` are in elements. Construction
/// checks that the window lies inside its allocation.
#[derive(Clone)]
pub struct StorageView {
    memory: Arc<dyn Backing>,
    offset: usize,
    dtype: DType,
    shape: Vec<i64>,
    strides: Vec<i64>,
}

impl StorageView {
    /// A contiguous view of `shape` starting `offset` bytes into `memory`.
    pub fn contiguous(
        memory: Arc<dyn Backing>,
        offset: usize,
        dtype: DType,
        shape: &[i64],
    ) -> Result<Self> {
        Self::strided(memory, offset, dtype, shape, &contiguous_strides(shape))
    }

    pub fn strided(
        memory: Arc<dyn Backing>,
        offset: usize,
        dtype: DType,
        shape: &[i64],
        strides: &[i64],
    ) -> Result<Self> {
        let view = Self {
            memory,
            offset,
            dtype,
            shape: shape.to_vec(),
            strides: strides.to_vec(),
        };
        view.byte_len()?;
        Ok(view)
    }

    /// A whole allocation viewed as `[len / dtype.bytes()]` elements.
    pub fn whole(memory: Arc<dyn Backing>, dtype: DType) -> Result<Self> {
        let elements = memory.len() / dtype.bytes();
        Self::contiguous(memory, 0, dtype, &[elements as i64])
    }

    pub fn device(&self) -> Device {
        self.memory.device()
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    pub fn strides(&self) -> &[i64] {
        &self.strides
    }

    /// Address of the first element.
    pub fn address(&self) -> u64 {
        self.memory.address() + self.offset as u64
    }

    pub fn is_contiguous(&self) -> bool {
        self.strides == contiguous_strides(&self.shape)
    }

    /// Bytes from the first to one past the last element.
    pub fn byte_len(&self) -> Result<usize> {
        if self.shape.contains(&0) {
            return Ok(0);
        }
        if self
            .shape
            .iter()
            .chain(&self.strides)
            .any(|&value| value < 0)
        {
            return Err(Error::Invariant(
                "views require non-negative extents and strides".to_owned(),
            ));
        }
        let last: i64 = self
            .shape
            .iter()
            .zip(&self.strides)
            .map(|(&extent, &stride)| (extent - 1) * stride)
            .sum();
        let bytes = (last as usize + 1) * self.dtype.bytes();
        if self.offset + bytes > self.memory.len() {
            return Err(Error::Invariant(format!(
                "a view of {bytes} bytes at offset {} exceeds its {}-byte allocation",
                self.offset,
                self.memory.len()
            )));
        }
        Ok(bytes)
    }

    /// The leading `count` entries along dimension 0.
    pub fn prefix(&self, count: i64) -> Result<Self> {
        let mut view = self.clone();
        let Some(extent) = view.shape.first_mut() else {
            return Err(Error::Invariant(
                "a scalar view has no leading dimension".to_owned(),
            ));
        };
        if count > *extent {
            return Err(Error::Invariant(format!(
                "prefix of {count} exceeds the leading extent {extent}"
            )));
        }
        *extent = count;
        Ok(view)
    }

    /// The same bytes reinterpreted as a contiguous `shape` of `dtype`.
    pub fn reshape(&self, dtype: DType, shape: &[i64]) -> Result<Self> {
        if !self.is_contiguous() {
            return Err(Error::Invariant(
                "only contiguous views can be reshaped".to_owned(),
            ));
        }
        let view = Self::contiguous(Arc::clone(&self.memory), self.offset, dtype, shape)?;
        if view.byte_len()? > self.byte_len()? {
            return Err(Error::Invariant(
                "a reshape cannot grow its view".to_owned(),
            ));
        }
        Ok(view)
    }

    /// Export to Python as a DLPack tensor retaining the allocation.
    pub fn to_tensor(&self) -> Result<Tensor> {
        export(
            Arc::clone(&self.memory),
            self.offset,
            self.dtype,
            &self.shape,
            &self.strides,
        )
    }
}

impl std::fmt::Debug for StorageView {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StorageView")
            .field("device", &self.device())
            .field("offset", &self.offset)
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn views_stay_inside_their_allocation() -> Result<()> {
        let memory: Arc<dyn Backing> = Memory::host(64);
        assert!(StorageView::contiguous(Arc::clone(&memory), 0, DType::F32, &[4, 4]).is_ok());
        assert!(StorageView::contiguous(Arc::clone(&memory), 4, DType::F32, &[4, 4]).is_err());

        let rows = StorageView::contiguous(Arc::clone(&memory), 0, DType::I64, &[8])?;
        assert_eq!(rows.prefix(3)?.byte_len()?, 24);
        assert!(rows.prefix(9).is_err());
        assert_eq!(rows.reshape(DType::I32, &[16])?.byte_len()?, 64);
        assert!(rows.reshape(DType::I32, &[17]).is_err());
        Ok(())
    }
}
