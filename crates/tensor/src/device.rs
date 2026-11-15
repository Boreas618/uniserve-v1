use crate::backend::BackendDevice;
use crate::cpu_backend::CpuDevice;
use crate::{CpuStorage, DType, Result, Shape, Storage, WithDType};
use std::sync::Arc;

/// Where a device is: the host, or a backend's kind and ordinal, such as `("cuda", 0)`.
///
/// Several [`Device`] values can have the same location.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum DeviceLocation {
    Cpu,
    Backend { kind: &'static str, ordinal: usize },
}

/// Where the storage of a tensor is: the built-in CPU backend, or a device backend implemented
/// outside this crate.
#[derive(Debug, Clone)]
pub enum Device {
    Cpu,
    Backend(Arc<dyn BackendDevice>),
}

pub trait NdArray {
    fn shape(&self) -> Result<Shape>;

    fn to_cpu_storage(&self) -> CpuStorage;
}

impl<S: WithDType> NdArray for S {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage(&[*self])
    }
}

impl<S: WithDType, const N: usize> NdArray for &[S; N] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage(self.as_slice())
    }
}

impl<S: WithDType> NdArray for &[S] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage(self)
    }
}

impl<S: WithDType, const N: usize, const M: usize> NdArray for &[[S; N]; M] {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((M, N)))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage_owned(self.concat())
    }
}

impl<S: WithDType, const N1: usize, const N2: usize, const N3: usize> NdArray
    for &[[[S; N3]; N2]; N1]
{
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((N1, N2, N3)))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        let mut vec = Vec::with_capacity(N1 * N2 * N3);
        for i1 in 0..N1 {
            for i2 in 0..N2 {
                vec.extend(self[i1][i2])
            }
        }
        S::to_cpu_storage_owned(vec)
    }
}

impl<S: WithDType, const N1: usize, const N2: usize, const N3: usize, const N4: usize> NdArray
    for &[[[[S; N4]; N3]; N2]; N1]
{
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from((N1, N2, N3, N4)))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        let mut vec = Vec::with_capacity(N1 * N2 * N3 * N4);
        for i1 in 0..N1 {
            for i2 in 0..N2 {
                for i3 in 0..N3 {
                    vec.extend(self[i1][i2][i3])
                }
            }
        }
        S::to_cpu_storage_owned(vec)
    }
}

impl<S: WithDType> NdArray for Vec<S> {
    fn shape(&self) -> Result<Shape> {
        Ok(Shape::from(self.len()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        S::to_cpu_storage(self.as_slice())
    }
}

impl<S: WithDType> NdArray for Vec<&[S]> {
    fn shape(&self) -> Result<Shape> {
        if self.is_empty() {
            crate::bail!("empty array")
        }
        let n = self.len();
        let m = self[0].len();
        for v in self.iter() {
            if v.len() != m {
                crate::bail!("two elements have different len {m} {}", v.len())
            }
        }
        Ok(Shape::from((n, m)))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        let data = self.iter().copied().flatten().copied().collect::<Vec<_>>();
        S::to_cpu_storage_owned(data)
    }
}

impl<S: WithDType> NdArray for Vec<Vec<S>> {
    fn shape(&self) -> Result<Shape> {
        if self.is_empty() {
            crate::bail!("empty array")
        }
        let n = self.len();
        let m = self[0].len();
        for v in self.iter() {
            if v.len() != m {
                crate::bail!("two elements have different len {m} {}", v.len())
            }
        }
        Ok(Shape::from((n, m)))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        let len: usize = self.iter().map(|v| v.len()).sum();
        let mut dst = Vec::with_capacity(len);
        for v in self.iter() {
            dst.extend(v.iter().copied());
        }
        S::to_cpu_storage_owned(dst)
    }
}

impl<S: WithDType> NdArray for Vec<Vec<Vec<S>>> {
    fn shape(&self) -> Result<Shape> {
        if self.is_empty() {
            crate::bail!("empty array")
        }
        let shape0 = self[0].shape()?;
        let n = self.len();
        for v in self.iter() {
            let shape = v.shape()?;
            if shape != shape0 {
                crate::bail!("two elements have different shapes {shape:?} {shape0:?}")
            }
        }
        Ok(Shape::from([[n].as_slice(), shape0.dims()].concat()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        if self.is_empty() {
            return S::to_cpu_storage_owned(vec![]);
        }
        let len: usize = self
            .iter()
            .map(|v| v.iter().map(|v| v.len()).sum::<usize>())
            .sum();
        let mut dst = Vec::with_capacity(len);
        for v1 in self.iter() {
            for v2 in v1.iter() {
                dst.extend(v2.iter().copied());
            }
        }
        S::to_cpu_storage_owned(dst)
    }
}

impl<S: WithDType> NdArray for Vec<Vec<Vec<Vec<S>>>> {
    fn shape(&self) -> Result<Shape> {
        if self.is_empty() {
            crate::bail!("empty array")
        }
        let shape0 = self[0].shape()?;
        let n = self.len();
        for v in self.iter() {
            let shape = v.shape()?;
            if shape != shape0 {
                crate::bail!("two elements have different shapes {shape:?} {shape0:?}")
            }
        }
        Ok(Shape::from([[n].as_slice(), shape0.dims()].concat()))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        let len: usize = self
            .iter()
            .map(|v| {
                v.iter()
                    .map(|v| v.iter().map(|v| v.len()).sum::<usize>())
                    .sum::<usize>()
            })
            .sum();
        let mut dst = Vec::with_capacity(len);
        for v1 in self.iter() {
            for v2 in v1.iter() {
                for v3 in v2.iter() {
                    dst.extend(v3.iter().copied());
                }
            }
        }
        S::to_cpu_storage_owned(dst)
    }
}

impl Device {
    /// Whether tensors of the two devices may meet in one operation.
    ///
    /// The CPU is the same device as itself only. Two backend devices are the same device when
    /// the backend of `self` says so.
    pub fn same_device(&self, rhs: &Self) -> bool {
        match (self, rhs) {
            (Self::Cpu, Self::Cpu) => true,
            (Self::Backend(lhs), Self::Backend(rhs)) => lhs.same_device(rhs.as_ref()),
            _ => false,
        }
    }

    pub fn location(&self) -> DeviceLocation {
        match self {
            Self::Cpu => DeviceLocation::Cpu,
            Self::Backend(device) => device.location(),
        }
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu)
    }

    pub(crate) fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        match self {
            Self::Cpu => Ok(Storage::Cpu(CpuDevice.zeros(shape, dtype)?)),
            Self::Backend(device) => Ok(Storage::Backend(device.zeros(shape, dtype)?)),
        }
    }

    /// A storage of `shape.elem_count()` elements of `dtype` whose content is unspecified.
    ///
    /// # Safety
    ///
    /// The caller writes every element before it reads one.
    pub(crate) unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Storage> {
        // SAFETY: the caller writes every element before it reads one, as this function requires.
        unsafe {
            match self {
                Self::Cpu => Ok(Storage::Cpu(CpuDevice.alloc_uninit(shape, dtype)?)),
                Self::Backend(device) => Ok(Storage::Backend(device.alloc_uninit(shape, dtype)?)),
            }
        }
    }

    /// The storage of host values on this device. A backend device receives them through its
    /// host-to-device copy.
    pub(crate) fn storage_from_cpu_storage(&self, storage: CpuStorage) -> Result<Storage> {
        match self {
            Self::Cpu => Ok(Storage::Cpu(storage)),
            Self::Backend(device) => {
                Ok(Storage::Backend(device.storage_from_cpu_storage(&storage)?))
            }
        }
    }

    pub(crate) fn storage_from_slice<D: WithDType>(&self, data: &[D]) -> Result<Storage> {
        self.storage_from_cpu_storage(data.to_cpu_storage())
    }

    pub(crate) fn storage<A: NdArray>(&self, array: A) -> Result<Storage> {
        self.storage_from_cpu_storage(array.to_cpu_storage())
    }

    pub(crate) fn storage_owned<S: WithDType>(&self, data: Vec<S>) -> Result<Storage> {
        self.storage_from_cpu_storage(S::to_cpu_storage_owned(data))
    }

    /// Blocks until every operation issued on the device has completed.
    pub fn synchronize(&self) -> Result<()> {
        match self {
            Self::Cpu => Ok(()),
            Self::Backend(device) => device.synchronize(),
        }
    }
}
