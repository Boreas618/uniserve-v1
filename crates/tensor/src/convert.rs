//! Implement conversion traits for tensors
use crate::cpu_backend::CpuStorage;
use crate::tensor::from_storage;
use crate::{DType, Device, Error, Storage, Tensor, WithDType};
use half::{bf16, f16, slice::HalfFloatSliceExt};
use std::convert::TryFrom;

impl<T: WithDType> TryFrom<&Tensor> for Vec<T> {
    type Error = Error;
    fn try_from(tensor: &Tensor) -> Result<Self, Self::Error> {
        tensor.to_vec1::<T>()
    }
}

impl<T: WithDType> TryFrom<&Tensor> for Vec<Vec<T>> {
    type Error = Error;
    fn try_from(tensor: &Tensor) -> Result<Self, Self::Error> {
        tensor.to_vec2::<T>()
    }
}

impl<T: WithDType> TryFrom<&Tensor> for Vec<Vec<Vec<T>>> {
    type Error = Error;
    fn try_from(tensor: &Tensor) -> Result<Self, Self::Error> {
        tensor.to_vec3::<T>()
    }
}

impl<T: WithDType> TryFrom<Tensor> for Vec<T> {
    type Error = Error;
    fn try_from(tensor: Tensor) -> Result<Self, Self::Error> {
        Vec::<T>::try_from(&tensor)
    }
}

impl<T: WithDType> TryFrom<Tensor> for Vec<Vec<T>> {
    type Error = Error;
    fn try_from(tensor: Tensor) -> Result<Self, Self::Error> {
        Vec::<Vec<T>>::try_from(&tensor)
    }
}

impl<T: WithDType> TryFrom<Tensor> for Vec<Vec<Vec<T>>> {
    type Error = Error;
    fn try_from(tensor: Tensor) -> Result<Self, Self::Error> {
        Vec::<Vec<Vec<T>>>::try_from(&tensor)
    }
}

impl<T: WithDType> TryFrom<&[T]> for Tensor {
    type Error = Error;
    fn try_from(v: &[T]) -> Result<Self, Self::Error> {
        Tensor::from_slice(v, v.len(), &Device::Cpu)
    }
}

impl<T: WithDType> TryFrom<Vec<T>> for Tensor {
    type Error = Error;
    fn try_from(v: Vec<T>) -> Result<Self, Self::Error> {
        let len = v.len();
        Tensor::from_vec(v, len, &Device::Cpu)
    }
}

macro_rules! from_tensor {
    ($typ:ident) => {
        impl TryFrom<&Tensor> for $typ {
            type Error = Error;

            fn try_from(tensor: &Tensor) -> Result<Self, Self::Error> {
                tensor.to_scalar::<$typ>()
            }
        }

        impl TryFrom<Tensor> for $typ {
            type Error = Error;

            fn try_from(tensor: Tensor) -> Result<Self, Self::Error> {
                $typ::try_from(&tensor)
            }
        }

        impl TryFrom<$typ> for Tensor {
            type Error = Error;

            fn try_from(v: $typ) -> Result<Self, Self::Error> {
                Tensor::new(v, &Device::Cpu)
            }
        }
    };
}

from_tensor!(f64);
from_tensor!(f32);
from_tensor!(f16);
from_tensor!(bf16);
from_tensor!(i64);
from_tensor!(i32);
from_tensor!(i16);
from_tensor!(u32);
from_tensor!(u8);

impl Tensor {
    pub fn write_bytes<W: std::io::Write>(&self, f: &mut W) -> crate::Result<()> {
        use byteorder::{LittleEndian, WriteBytesExt};

        let vs = self.flatten_all()?;
        match self.dtype() {
            DType::BF16 => {
                let vs = vs.to_vec1::<bf16>()?;
                for &v in vs.reinterpret_cast() {
                    f.write_u16::<LittleEndian>(v)?
                }
            }
            DType::F16 => {
                let vs = vs.to_vec1::<f16>()?;
                for &v in vs.reinterpret_cast() {
                    f.write_u16::<LittleEndian>(v)?
                }
            }
            DType::F32 => {
                // TODO: Avoid using a buffer when data is already on the CPU.
                for v in vs.to_vec1::<f32>()? {
                    f.write_f32::<LittleEndian>(v)?
                }
            }
            DType::F64 => {
                for v in vs.to_vec1::<f64>()? {
                    f.write_f64::<LittleEndian>(v)?
                }
            }
            DType::U32 => {
                for v in vs.to_vec1::<u32>()? {
                    f.write_u32::<LittleEndian>(v)?
                }
            }
            DType::I16 => {
                for v in vs.to_vec1::<i16>()? {
                    f.write_i16::<LittleEndian>(v)?
                }
            }
            DType::I32 => {
                for v in vs.to_vec1::<i32>()? {
                    f.write_i32::<LittleEndian>(v)?
                }
            }
            DType::I64 => {
                for v in vs.to_vec1::<i64>()? {
                    f.write_i64::<LittleEndian>(v)?
                }
            }
            DType::U8 => {
                let vs = vs.to_vec1::<u8>()?;
                f.write_all(&vs)?;
            }
            DType::F8E4M3 => {
                let vs = vs.to_vec1::<float8::F8E4M3>()?;
                for v in vs {
                    f.write_u8(v.to_bits())?
                }
            }
            DType::F4 => f.write_all(&packed_bytes(&vs)?)?,
        }
        Ok(())
    }
}

/// The packed bytes of an F4 tensor whose elements are its whole storage, as
/// [`Tensor::from_raw_buffer`] builds one. A view of part of a storage is refused: its first or
/// last byte may hold an element outside of it.
fn packed_bytes(tensor: &Tensor) -> crate::Result<Vec<u8>> {
    let host = tensor.to_device(&Device::Cpu)?;
    let (storage, layout) = host.storage_and_layout();
    let Storage::Cpu(CpuStorage::F4(bytes)) = &*storage else {
        return Err(Error::UnexpectedDType {
            msg: "the storage of an F4 tensor",
            expected: DType::F4,
            got: tensor.dtype(),
        }
        .bt());
    };

    let whole = DType::F4.storage_bytes(layout.shape().elem_count()) == Some(bytes.len());
    if layout.start_offset() != 0 || !whole {
        return Err(Error::UnsupportedDTypeForOp(DType::F4, "write_bytes of a view").bt());
    }
    Ok(bytes.clone())
}

impl Tensor {
    /// A tensor of `dtype` and `shape` on `device` over the little-endian element bytes `data`,
    /// copied. The packed bytes of an `F4` tensor are kept as they are: `data` holds
    /// [`DType::storage_bytes`] of the shape's elements.
    pub fn from_raw_buffer(
        data: &[u8],
        dtype: DType,
        shape: &[usize],
        device: &Device,
    ) -> crate::Result<Self> {
        match dtype {
            DType::U8 => convert_slice::<u8>(data, shape, device),
            DType::U32 => convert_slice::<u32>(data, shape, device),
            DType::I16 => convert_slice::<i16>(data, shape, device),
            DType::I32 => convert_slice::<i32>(data, shape, device),
            DType::I64 => convert_slice::<i64>(data, shape, device),
            DType::BF16 => convert_slice::<half::bf16>(data, shape, device),
            DType::F16 => convert_slice::<half::f16>(data, shape, device),
            DType::F32 => convert_slice::<f32>(data, shape, device),
            DType::F64 => convert_slice::<f64>(data, shape, device),
            DType::F8E4M3 => convert_slice::<float8::F8E4M3>(data, shape, device),
            DType::F4 => packed_tensor(data, shape, device),
        }
    }
}

fn convert_slice<T: WithDType>(
    data: &[u8],
    shape: &[usize],
    device: &Device,
) -> crate::Result<Tensor> {
    let size_in_bytes = size_of::<T>();
    let elem_count = data.len() / size_in_bytes;
    if (data.as_ptr() as usize).is_multiple_of(size_in_bytes) {
        // SAFETY: `data` is aligned for `T` (checked above) and holds `elem_count` whole
        // elements of it; `T` is a plain numeric type for which any bits are a value.
        let data: &[T] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const T, elem_count) };
        Tensor::from_slice(data, shape, device)
    } else {
        // XXX: We need to specify `T` here, otherwise the compiler will infer u8 because of the following cast
        // Making this vector too small to fit a full f16/f32/f64 weights, resulting in out-of-bounds access
        let mut c: Vec<T> = Vec::with_capacity(elem_count);
        // SAFETY: We just created c, so the allocated memory is necessarily
        // contiguous and non overlapping with the view's data.
        // We're downgrading the `c` pointer from T to u8, which removes alignment
        // constraints.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), c.as_mut_ptr() as *mut u8, data.len());
            c.set_len(elem_count)
        }
        Tensor::from_slice(&c, shape, device)
    }
}

/// An F4 tensor of `shape` over the packed bytes `data`, which it keeps as they are.
///
/// # Errors
///
/// [`Error::ShapeMismatch`] when `data` is not the bytes of the shape's elements.
fn packed_tensor(data: &[u8], shape: &[usize], device: &Device) -> crate::Result<Tensor> {
    let shape = crate::Shape::from(shape);
    if DType::F4.storage_bytes(shape.elem_count()) != Some(data.len()) {
        return Err(Error::ShapeMismatch {
            buffer_size: data.len(),
            shape,
        }
        .bt());
    }

    let storage = device.storage_from_cpu_storage(CpuStorage::F4(data.to_vec()))?;
    Ok(from_storage(storage, shape))
}
