//! The storage of a tensor on the CUDA and meta devices, and the tensors execution and `nn` make
//! over it.
//!
//! A storage is a device address, a byte length and an element type. Who frees the bytes is one
//! of three owners:
//!
//! - **allocated**: a block of the allocator of the context the storage was made under, which
//!   returns to that allocator when the last tensor over the storage drops, on whatever thread
//!   that is ([`empty`], and every result of a tensor operation);
//! - **retained**: a weight or a published constant, which no allocator ever takes back
//!   ([`retained`]);
//! - **external**: memory execution owns and lends for the duration of a call, such as KV pages,
//!   state buffers, latents and a graph's inputs ([`external`]).
//!
//! The last two are one representation, a storage with no block to return; what tells them apart
//! is who keeps the bytes alive, which is the caller's obligation and stated by the constructor.
//!
//! A storage is a **meta storage** when it has no data: one allocated under a meta context,
//! whose block is an address nothing reads, and one made over the meta device. It reports the
//! meta device, every operation on it states the kernel it would launch, and reading it is an
//! error.

use std::sync::Arc;

use ops::platform::{CaptureStatus, Stream, copy_to_host};
use ops::{DevicePtr, MAX_RANK, Operand, Strides};
use uniserve_tensor::backend::{BackendDevice, BackendStorage};
use uniserve_tensor::{CpuStorage, DType, Device, Layout, Shape, Storage, Tensor};

use crate::allocator::Allocator;
use crate::device::{self, CudaDevice, MetaDevice};
use crate::{Error, Result, context};

/// The elements of a tensor on a GPU, or the extent of a tensor that has none.
#[derive(Debug)]
pub(crate) struct CudaStorage {
    /// The device address of the first byte. Nothing reads it for a meta storage.
    address: u64,
    /// The length in bytes.
    bytes: u64,
    dtype: DType,
    /// The GPU the bytes are on, or `None` for a meta storage, which has no data and reports the
    /// meta device.
    gpu: Option<usize>,
    /// The allocator an allocated storage returns its block to on drop, with the reset the
    /// block was allocated under. `None` for retained and external storage, which this type
    /// never frees.
    block: Option<(Arc<Allocator>, u64)>,
}

impl CudaStorage {
    /// A storage of `bytes` bytes of `dtype` from the allocator of the context installed on this
    /// thread: on the context's GPU, or meta under a meta context.
    ///
    /// `asked` is the GPU of the device the caller named, when it named a CUDA device. A meta
    /// context gives a meta storage whatever was named.
    ///
    /// # Errors
    ///
    /// [`Error::NoContext`]; [`Error::WrongDevice`] when a launching context computes on another
    /// GPU than `asked`; [`Error::OutOfMemory`].
    pub(crate) fn allocate(bytes: u64, dtype: DType, asked: Option<usize>) -> Result<Self> {
        Self::from_allocator(bytes, dtype, asked, Allocator::allocate)
    }

    /// As [`CudaStorage::allocate`], at addresses no storage of this forward had
    /// ([`Allocator::allocate_unused`]): the destination of a host-to-device copy.
    pub(crate) fn allocate_unused(bytes: u64, dtype: DType, asked: Option<usize>) -> Result<Self> {
        Self::from_allocator(bytes, dtype, asked, Allocator::allocate_unused)
    }

    /// A storage whose block `take` gives from the thread's context's allocator.
    fn from_allocator(
        bytes: u64,
        dtype: DType,
        asked: Option<usize>,
        take: fn(&Allocator, u64) -> Result<(u64, u64)>,
    ) -> Result<Self> {
        let context = context::current("an allocation")?;
        let gpu = context.ordinal();

        if let (Some(asked), Some(context)) = (asked, gpu)
            && asked != context
        {
            return Err(Error::WrongDevice { asked, context });
        }

        let allocator = Arc::clone(context.allocator());
        let (address, reset) = take(&allocator, bytes)?;

        Ok(Self {
            address,
            bytes,
            dtype,
            gpu,
            block: Some((allocator, reset)),
        })
    }

    /// A storage of `elements` elements of `dtype` from the thread's context
    /// ([`CudaStorage::allocate`]).
    pub(crate) fn of_elements(elements: usize, dtype: DType, asked: Option<usize>) -> Result<Self> {
        let bytes = byte_length(dtype, elements as u64, "an allocation")?;
        Self::allocate(bytes, dtype, asked)
    }

    /// A storage over `elements` elements of `dtype` at `address` that this type never frees:
    /// on GPU `gpu`, or a meta storage for `None`.
    fn borrowed(address: u64, elements: usize, dtype: DType, gpu: Option<usize>) -> Result<Self> {
        Ok(Self {
            address,
            bytes: byte_length(dtype, elements as u64, "a tensor over lent memory")?,
            dtype,
            gpu,
            block: None,
        })
    }

    /// The storage of a backend tensor as this backend's storage.
    ///
    /// # Errors
    ///
    /// [`Error::ForeignTensor`] for the storage of another backend.
    pub(crate) fn of(storage: &dyn BackendStorage) -> Result<&Self> {
        storage
            .as_any()
            .downcast_ref::<Self>()
            .ok_or_else(|| Error::ForeignTensor {
                operation: "a tensor operation",
                location: storage.device().location(),
            })
    }

    pub(crate) fn dtype(&self) -> DType {
        self.dtype
    }

    /// The device address of the first byte.
    pub(crate) fn address(&self) -> u64 {
        self.address
    }

    /// The device the storage reports: the CUDA device of its GPU, or the meta device.
    pub(crate) fn device(&self) -> Arc<dyn BackendDevice> {
        match self.gpu {
            Some(ordinal) => Arc::new(CudaDevice::new(ordinal)),
            None => Arc::new(MetaDevice),
        }
    }

    /// The length in bytes.
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The whole elements the storage holds.
    pub(crate) fn elements(&self) -> usize {
        let elements = match self.dtype.size_in_bytes() {
            // A packed 4-bit element: two share a byte. No other packed type is ever stored
            // (`byte_length`).
            0 => self.bytes * 2,
            size => self.bytes / size as u64,
        };

        // The byte length came from an element count that was a `usize`.
        elements as usize
    }

    /// The kernel argument of the elements `layout` views in this storage: the address of the
    /// first one, the element type, and the layout's dimensions and strides, in elements.
    ///
    /// # Errors
    ///
    /// [`Error::Rank`] for more than [`MAX_RANK`] axes; [`Error::ElementType`] for a view into a
    /// packed tensor that starts inside a byte; [`Error::MetaOperand`] for a meta storage while
    /// a context that launches is installed.
    pub(crate) fn operand(&self, layout: &Layout) -> Result<Operand> {
        let rank = layout.dims().len();
        if rank > MAX_RANK {
            return Err(Error::Rank {
                rank,
                limit: MAX_RANK,
            });
        }

        // A kernel would read an address nothing backs. Which context is installed says whether
        // one is about to: under a meta context every operand may be meta, and with no context
        // nothing launches.
        if self.gpu.is_none() && context::installed().is_some_and(|context| !context.is_meta()) {
            return Err(Error::MetaOperand);
        }

        let offset = layout.start_offset() as u64;
        let address = match self.dtype.size_in_bytes() {
            0 if offset == 0 => self.address,
            0 => {
                return Err(Error::ElementType {
                    operation: "a kernel operand",
                    dtype: self.dtype,
                    reason: "a view into a packed tensor starts inside a byte",
                });
            }
            size => self.address + offset * size as u64,
        };

        let mut dims = [0_u64; MAX_RANK];
        let mut strides = [0_u64; MAX_RANK];
        for (axis, (dim, stride)) in layout.dims().iter().zip(layout.stride()).enumerate() {
            dims[axis] = *dim as u64;
            strides[axis] = *stride as u64;
        }

        Ok(Operand::with_strides(
            DevicePtr::new(address),
            self.dtype,
            ops::Shape::new(&dims[..rank]),
            Strides::new(&strides[..rank]),
        ))
    }

    /// The device-to-host read: every element of the storage, at the same positions.
    ///
    /// The copy is issued on the stream of the thread's context after everything the stream
    /// holds, and this waits for it.
    ///
    /// # Errors
    ///
    /// [`Error::MetaRead`] for a meta storage and under a meta context; [`Error::NoContext`];
    /// [`Error::CapturedRead`] while the stream is being captured; [`Error::Platform`] and
    /// [`Error::Ops`] when the stream cannot be queried, copied on or waited for.
    pub(crate) fn read(&self) -> Result<CpuStorage> {
        if self.gpu.is_none() {
            return Err(Error::MetaRead);
        }

        let context = context::current("a device-to-host read")?;
        let Some(stream) = context.stream() else {
            return Err(Error::MetaRead);
        };
        if stream.capture_status()? != CaptureStatus::None {
            return Err(Error::CapturedRead);
        }

        // SAFETY: every element type below is plain data of which any bit pattern is a value.
        let storage = unsafe {
            match self.dtype {
                DType::U8 => CpuStorage::U8(self.read_as(stream)?),
                DType::U32 => CpuStorage::U32(self.read_as(stream)?),
                DType::I16 => CpuStorage::I16(self.read_as(stream)?),
                DType::I32 => CpuStorage::I32(self.read_as(stream)?),
                DType::I64 => CpuStorage::I64(self.read_as(stream)?),
                DType::BF16 => CpuStorage::BF16(self.read_as(stream)?),
                DType::F16 => CpuStorage::F16(self.read_as(stream)?),
                DType::F32 => CpuStorage::F32(self.read_as(stream)?),
                DType::F64 => CpuStorage::F64(self.read_as(stream)?),
                DType::F8E4M3 => CpuStorage::F8E4M3(self.read_as(stream)?),
                // The packed types are held on the host as their bytes.
                DType::F4 => CpuStorage::F4(self.read_as(stream)?),
                DType::F8E8M0 => CpuStorage::F8E8M0(self.read_as(stream)?),
                dtype => {
                    return Err(Error::ElementType {
                        operation: "a device-to-host read",
                        dtype,
                        reason: "the host storage of the element type is unknown to this backend",
                    });
                }
            }
        };
        Ok(storage)
    }

    /// The storage's bytes as a vector of `T`, copied on `stream` and waited for.
    ///
    /// # Safety
    ///
    /// Any bit pattern is a value of `T`, and the storage's address holds its bytes on the
    /// stream's device.
    unsafe fn read_as<T: Copy>(&self, stream: &Stream) -> Result<Vec<T>> {
        let count = self.bytes as usize / size_of::<T>();
        let mut values = Vec::<T>::with_capacity(count);

        // SAFETY: the vector's buffer holds `count` elements of `T`, which are the bytes copied,
        // and it is not read before the stream has finished the copy.
        unsafe {
            copy_to_host(
                values.as_mut_ptr() as u64,
                DevicePtr::new(self.address),
                (count * size_of::<T>()) as u64,
                stream,
            )?;
        }
        stream.synchronize()?;

        // SAFETY: the copy wrote every one of the `count` elements.
        unsafe { values.set_len(count) };
        Ok(values)
    }
}

impl Drop for CudaStorage {
    fn drop(&mut self) {
        // A block of an earlier reset is no longer part of its allocator's state, which the
        // allocator tells by the reset number.
        if let Some((allocator, reset)) = self.block.take() {
            allocator.free(self.address, self.bytes, reset);
        }
    }
}

/// The bytes `elements` elements of `dtype` occupy, for `operation`.
///
/// # Errors
///
/// [`Error::ElementType`] for a packed element with no byte layout; [`Error::TooLarge`].
fn byte_length(dtype: DType, elements: u64, operation: &'static str) -> Result<u64> {
    let bytes = match (dtype.size_in_bytes(), dtype) {
        // NVFP4: two elements share a byte, the last byte of an odd count half used.
        (0, DType::F4) => Some(elements.div_ceil(2)),
        (0, _) => {
            return Err(Error::ElementType {
                operation,
                dtype,
                reason: "the element is narrower than a byte and has no packed layout here",
            });
        }
        (size, _) => elements.checked_mul(size as u64),
    };

    bytes.ok_or(Error::TooLarge { elements, dtype })
}

/// A new tensor of `shape` elements of `dtype` from the context installed on this thread: a
/// block of its allocator, on the context's GPU, and a meta tensor under a meta context.
///
/// The elements are unspecified until something writes them: a fused operator allocates its
/// output with this and launches the kernel that fills it.
///
/// # Errors
///
/// [`Error::NoContext`]; [`Error::OutOfMemory`]; [`Error::ElementType`] for an element type
/// with no byte layout.
pub fn empty(shape: impl Into<Shape>, dtype: DType) -> Result<Tensor> {
    let shape = shape.into();
    let storage = CudaStorage::of_elements(shape.elem_count(), dtype, None)?;

    Ok(Tensor::from_storage(
        Storage::Backend(Box::new(storage)),
        shape,
    ))
}

/// A tensor over memory the caller owns and lends for the duration of a call: KV pages, state
/// buffers, latents, a graph's inputs. Dropping the tensor frees nothing.
///
/// `device` is the CUDA device the memory is on. Over the meta device the tensor is a meta
/// tensor that takes no block of any allocator and whose `address` is never read, which is how
/// execution states the inputs of a profile run and of a `prepare_inputs` forward.
///
/// # Errors
///
/// [`Error::ForeignTensor`] when `device` is not a device of this backend;
/// [`Error::ElementType`] for an element type with no byte layout.
///
/// # Safety
///
/// `address` is the first of `shape.elem_count()` elements of `dtype` on `device` that stay
/// valid, and that nothing else writes while a kernel launched on the tensor may read them,
/// until every tensor over the storage is dropped and the stream work issued on them, including
/// the replays of a graph captured over them, has completed.
pub unsafe fn external(
    address: u64,
    shape: impl Into<Shape>,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    borrowed(address, shape.into(), dtype, device)
}

/// A tensor over retained storage: a weight or a published constant, which stays mapped at
/// `address` for as long as the process serves and which no allocator takes back.
///
/// Over the meta device the tensor is a meta tensor: a weight of a profile run, which has a
/// shape and an element type and no bytes.
///
/// # Errors
///
/// As [`external`].
pub fn retained(
    address: u64,
    shape: impl Into<Shape>,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    borrowed(address, shape.into(), dtype, device)
}

/// The tensor of `shape` elements of `dtype` at `address` on `device`, which nothing frees.
fn borrowed(address: u64, shape: Shape, dtype: DType, device: &Device) -> Result<Tensor> {
    let gpu = device::gpu(device, "a tensor over lent memory")?;
    let storage = CudaStorage::borrowed(address, shape.elem_count(), dtype, gpu)?;

    Ok(Tensor::from_storage(
        Storage::Backend(Box::new(storage)),
        shape,
    ))
}

/// The kernel argument of `tensor`: the address of its first element, its element type, and
/// its dimensions and strides in elements, as the tensor's layout views its storage.
///
/// The operand borrows the tensor's memory: the caller keeps the tensor alive until the launch
/// that takes the operand has been issued.
///
/// # Errors
///
/// [`Error::ForeignTensor`] for a CPU tensor and for a tensor of another backend;
/// [`Error::Rank`] for more than [`MAX_RANK`] axes; [`Error::ElementType`] for a view into a
/// packed tensor that starts inside a byte; [`Error::MetaOperand`] for a meta tensor while a
/// context that launches is installed.
pub fn operand(tensor: &Tensor) -> Result<Operand> {
    let (storage, layout) = tensor.storage_and_layout();

    match &*storage {
        Storage::Backend(storage) => CudaStorage::of(storage.as_ref())?.operand(layout),
        Storage::Cpu(_) => Err(Error::ForeignTensor {
            operation: "a kernel operand",
            location: tensor.device().location(),
        }),
    }
}
