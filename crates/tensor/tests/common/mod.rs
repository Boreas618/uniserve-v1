//! Support shared by the integration tests: a device backend implemented outside the crate, the
//! macro that runs one test function on every device, and sample data.
//!
//! The backend keeps its elements in host memory and computes every primitive with the crate's
//! CPU storage. It uses nothing but the crate's public interface, as a backend in another
//! repository does, so a test that passes on it has gone through `Device::Backend` and
//! `Storage::Backend` for every tensor operation it performs.

// Each test binary uses its own part of this module.
#![allow(dead_code)]

use std::any::Any;
use std::sync::Arc;
use tensor::backend::{BackendDevice, BackendStorage};
use tensor::conv::{ParamsConv1D, ParamsConv2D, ParamsConvTranspose1D, ParamsConvTranspose2D};
use tensor::cpu_backend::CpuDevice;
use tensor::op::{BinaryOp, CmpOp, ReduceOp, UnaryOp};
use tensor::scalar::Scalar;
use tensor::{CpuStorage, DType, Device, DeviceLocation, Layout, Result, Shape, Tensor, bail};

/// Runs the test function `$test`, which takes a `&Device` and returns a `Result`, once on the
/// CPU device and once on the host backend device, as the tests `$test::cpu` and
/// `$test::backend`.
macro_rules! test_device {
    ($test:ident) => {
        mod $test {
            #[test]
            fn cpu() {
                super::$test(&tensor::Device::Cpu).unwrap()
            }

            #[test]
            fn backend() {
                super::$test(&$crate::common::host_device(0)).unwrap()
            }
        }
    };
}

/// The backend device `ordinal` of kind `host`. Two of them are the same device when their
/// ordinals are equal.
pub(crate) fn host_device(ordinal: usize) -> Device {
    Device::Backend(Arc::new(HostDevice { ordinal }))
}

/// A tensor of F32 values in `[-1, 1)` that depend on `seed` only, for tests whose expected
/// values are computed from their inputs.
pub(crate) fn sample<S: Into<Shape>>(seed: u64, shape: S, device: &Device) -> Result<Tensor> {
    let shape = shape.into();

    // SplitMix64: consecutive states are well mixed whatever the seed.
    let mut state = seed;
    let values = (0..shape.elem_count())
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;

            // The top 24 bits are exact in an f32.
            (z >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        })
        .collect::<Vec<_>>();

    Tensor::from_vec(values, shape, device)
}

#[derive(Debug)]
pub(crate) struct HostDevice {
    ordinal: usize,
}

#[derive(Debug)]
pub(crate) struct HostStorage {
    storage: CpuStorage,
    device: Arc<HostDevice>,
}

impl HostStorage {
    /// The storage of a backend tensor as this backend's storage; an error when it belongs to
    /// another backend.
    pub(crate) fn from_dyn(storage: &dyn BackendStorage) -> Result<&Self> {
        match storage.as_any().downcast_ref::<Self>() {
            Some(storage) => Ok(storage),
            None => bail!("expected a host storage, got {storage:?}"),
        }
    }

    pub(crate) fn from_dyn_mut(storage: &mut dyn BackendStorage) -> Result<&mut Self> {
        match storage.as_any_mut().downcast_mut::<Self>() {
            Some(storage) => Ok(storage),
            None => bail!("expected a host storage"),
        }
    }

    pub(crate) fn cpu(&self) -> &CpuStorage {
        &self.storage
    }

    pub(crate) fn cpu_mut(&mut self) -> &mut CpuStorage {
        &mut self.storage
    }

    /// A storage on the device of `self` holding `storage`.
    pub(crate) fn on_same_device(&self, storage: CpuStorage) -> Box<dyn BackendStorage> {
        Box::new(Self {
            storage,
            device: Arc::clone(&self.device),
        })
    }
}

/// The CPU storage under an operand of this backend.
fn cpu(storage: &dyn BackendStorage) -> Result<&CpuStorage> {
    Ok(HostStorage::from_dyn(storage)?.cpu())
}

impl BackendDevice for HostDevice {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn location(&self) -> DeviceLocation {
        DeviceLocation::Backend {
            kind: "host",
            ordinal: self.ordinal,
        }
    }

    fn same_device(&self, other: &dyn BackendDevice) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| other.ordinal == self.ordinal)
    }

    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>> {
        let storage = CpuDevice.zeros(shape, dtype)?;
        Ok(self.storage(storage))
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>> {
        // SAFETY: the caller writes every element before it reads one, as this method requires.
        unsafe {
            let storage = CpuDevice.alloc_uninit(shape, dtype)?;
            Ok(self.storage(storage))
        }
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Box<dyn BackendStorage>> {
        Ok(self.storage(storage.clone()))
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

impl HostDevice {
    fn storage(&self, storage: CpuStorage) -> Box<dyn BackendStorage> {
        Box::new(HostStorage {
            storage,
            device: Arc::new(Self {
                ordinal: self.ordinal,
            }),
        })
    }
}

impl BackendStorage for HostStorage {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn dtype(&self) -> DType {
        self.storage.dtype()
    }

    fn device(&self) -> Arc<dyn BackendDevice> {
        Arc::<HostDevice>::clone(&self.device)
    }

    fn try_clone(&self, _layout: &Layout) -> Result<Box<dyn BackendStorage>> {
        Ok(self.on_same_device(self.storage.clone()))
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        Ok(self.storage.clone())
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Box<dyn BackendStorage>> {
        let storage = self.storage.affine(layout, mul, add)?;
        Ok(self.on_same_device(storage))
    }

    fn powf(&self, layout: &Layout, exponent: f64) -> Result<Box<dyn BackendStorage>> {
        let storage = self.storage.powf(layout, exponent)?;
        Ok(self.on_same_device(storage))
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Box<dyn BackendStorage>> {
        let storage = self.storage.elu(layout, alpha)?;
        Ok(self.on_same_device(storage))
    }

    fn unary(&self, op: UnaryOp, layout: &Layout) -> Result<Box<dyn BackendStorage>> {
        let storage = self.storage.unary(op, layout)?;
        Ok(self.on_same_device(storage))
    }

    fn binary(
        &self,
        op: BinaryOp,
        rhs: &dyn BackendStorage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Box<dyn BackendStorage>> {
        let storage = self.storage.binary(op, cpu(rhs)?, lhs_layout, rhs_layout)?;
        Ok(self.on_same_device(storage))
    }

    fn cmp(
        &self,
        op: CmpOp,
        rhs: &dyn BackendStorage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Box<dyn BackendStorage>> {
        let storage = self.storage.cmp(op, cpu(rhs)?, lhs_layout, rhs_layout)?;
        Ok(self.on_same_device(storage))
    }

    fn where_cond(
        &self,
        layout: &Layout,
        on_true: &dyn BackendStorage,
        true_layout: &Layout,
        on_false: &dyn BackendStorage,
        false_layout: &Layout,
    ) -> Result<Box<dyn BackendStorage>> {
        let storage = self.storage.where_cond(
            layout,
            cpu(on_true)?,
            true_layout,
            cpu(on_false)?,
            false_layout,
        )?;
        Ok(self.on_same_device(storage))
    }

    fn reduce(
        &self,
        op: ReduceOp,
        layout: &Layout,
        dims: &[usize],
    ) -> Result<Box<dyn BackendStorage>> {
        let storage = self.storage.reduce(op, layout, dims)?;
        Ok(self.on_same_device(storage))
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Box<dyn BackendStorage>> {
        let storage = self.storage.to_dtype(layout, dtype)?;
        Ok(self.on_same_device(storage))
    }

    /// The elements are a typed host vector, so the reinterpreted storage is a copy of its
    /// bytes, as the CPU backend's is.
    fn view(&self, dtype: DType) -> Result<Box<dyn BackendStorage>> {
        let storage = self.storage.view(dtype)?;
        Ok(self.on_same_device(storage))
    }

    fn matmul(
        &self,
        rhs: &dyn BackendStorage,
        bmnk: (usize, usize, usize, usize),
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Box<dyn BackendStorage>> {
        let storage = self
            .storage
            .matmul(cpu(rhs)?, bmnk, lhs_layout, rhs_layout)?;
        Ok(self.on_same_device(storage))
    }

    fn index_select(
        &self,
        ids: &dyn BackendStorage,
        layout: &Layout,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<Box<dyn BackendStorage>> {
        let storage = self
            .storage
            .index_select(cpu(ids)?, layout, ids_layout, dim)?;
        Ok(self.on_same_device(storage))
    }

    fn gather(
        &self,
        layout: &Layout,
        ids: &dyn BackendStorage,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<Box<dyn BackendStorage>> {
        let storage = self.storage.gather(layout, cpu(ids)?, ids_layout, dim)?;
        Ok(self.on_same_device(storage))
    }

    fn index_add(
        &self,
        layout: &Layout,
        ids: &dyn BackendStorage,
        ids_layout: &Layout,
        src: &dyn BackendStorage,
        src_layout: &Layout,
        dim: usize,
    ) -> Result<Box<dyn BackendStorage>> {
        let storage =
            self.storage
                .index_add(layout, cpu(ids)?, ids_layout, cpu(src)?, src_layout, dim)?;
        Ok(self.on_same_device(storage))
    }

    fn scatter_set(
        &mut self,
        layout: &Layout,
        ids: &dyn BackendStorage,
        ids_layout: &Layout,
        src: &dyn BackendStorage,
        src_layout: &Layout,
        dim: usize,
    ) -> Result<()> {
        self.storage
            .scatter_set(layout, cpu(ids)?, ids_layout, cpu(src)?, src_layout, dim)
    }

    fn scatter_add_set(
        &mut self,
        layout: &Layout,
        ids: &dyn BackendStorage,
        ids_layout: &Layout,
        src: &dyn BackendStorage,
        src_layout: &Layout,
        dim: usize,
    ) -> Result<()> {
        self.storage
            .scatter_add_set(layout, cpu(ids)?, ids_layout, cpu(src)?, src_layout, dim)
    }

    fn copy_strided_src(
        &self,
        dst: &mut dyn BackendStorage,
        dst_offset: usize,
        src_layout: &Layout,
    ) -> Result<()> {
        let dst = HostStorage::from_dyn_mut(dst)?.cpu_mut();
        self.storage.copy_strided_src(dst, dst_offset, src_layout)
    }

    fn copy2d(
        &self,
        dst: &mut dyn BackendStorage,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> Result<()> {
        let dst = HostStorage::from_dyn_mut(dst)?.cpu_mut();
        self.storage.copy2d(
            dst,
            d1,
            d2,
            src_stride1,
            dst_stride1,
            src_offset,
            dst_offset,
        )
    }

    fn const_set(&mut self, value: Scalar, layout: &Layout) -> Result<()> {
        self.storage.const_set(value, layout)
    }

    fn conv1d(
        &self,
        layout: &Layout,
        kernel: &dyn BackendStorage,
        kernel_layout: &Layout,
        params: &ParamsConv1D,
    ) -> Result<Box<dyn BackendStorage>> {
        let storage = self
            .storage
            .conv1d(layout, cpu(kernel)?, kernel_layout, params)?;
        Ok(self.on_same_device(storage))
    }

    fn conv_transpose1d(
        &self,
        layout: &Layout,
        kernel: &dyn BackendStorage,
        kernel_layout: &Layout,
        params: &ParamsConvTranspose1D,
    ) -> Result<Box<dyn BackendStorage>> {
        let storage = self
            .storage
            .conv_transpose1d(layout, cpu(kernel)?, kernel_layout, params)?;
        Ok(self.on_same_device(storage))
    }

    fn conv2d(
        &self,
        layout: &Layout,
        kernel: &dyn BackendStorage,
        kernel_layout: &Layout,
        params: &ParamsConv2D,
    ) -> Result<Box<dyn BackendStorage>> {
        let storage = self
            .storage
            .conv2d(layout, cpu(kernel)?, kernel_layout, params)?;
        Ok(self.on_same_device(storage))
    }

    fn conv_transpose2d(
        &self,
        layout: &Layout,
        kernel: &dyn BackendStorage,
        kernel_layout: &Layout,
        params: &ParamsConvTranspose2D,
    ) -> Result<Box<dyn BackendStorage>> {
        let storage = self
            .storage
            .conv_transpose2d(layout, cpu(kernel)?, kernel_layout, params)?;
        Ok(self.on_same_device(storage))
    }

    fn avg_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Box<dyn BackendStorage>> {
        let storage = self.storage.avg_pool2d(layout, kernel_size, stride)?;
        Ok(self.on_same_device(storage))
    }

    fn max_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Box<dyn BackendStorage>> {
        let storage = self.storage.max_pool2d(layout, kernel_size, stride)?;
        Ok(self.on_same_device(storage))
    }

    fn upsample_nearest1d(&self, layout: &Layout, size: usize) -> Result<Box<dyn BackendStorage>> {
        let storage = self.storage.upsample_nearest1d(layout, size)?;
        Ok(self.on_same_device(storage))
    }

    fn upsample_nearest2d(
        &self,
        layout: &Layout,
        height: usize,
        width: usize,
    ) -> Result<Box<dyn BackendStorage>> {
        let storage = self.storage.upsample_nearest2d(layout, height, width)?;
        Ok(self.on_same_device(storage))
    }

    fn upsample_bilinear2d(
        &self,
        layout: &Layout,
        height: usize,
        width: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Box<dyn BackendStorage>> {
        let storage = self.storage.upsample_bilinear2d(
            layout,
            height,
            width,
            align_corners,
            scale_h,
            scale_w,
        )?;
        Ok(self.on_same_device(storage))
    }
}

/// Asserts that `t1` and `t2` have one shape and equal elements.
pub(crate) fn assert_tensor_eq(t1: &Tensor, t2: &Tensor) -> Result<()> {
    assert_eq!(t1.shape(), t2.shape());
    // Default U8 may not be large enough to hold the sum (`t.sum_all` defaults to the dtype of `t`)
    let eq_tensor = t1.eq(t2)?.to_dtype(DType::U32)?;
    let all_equal = eq_tensor.sum_all()?;
    assert_eq!(all_equal.to_scalar::<u32>()?, eq_tensor.elem_count() as u32);
    Ok(())
}

/// The F32 scalar `t` rounded to `digits` decimal digits.
pub(crate) fn to_vec0_round(t: &Tensor, digits: i32) -> Result<f32> {
    let b = 10f32.powi(digits);
    let t = t.to_vec0::<f32>()?;
    Ok(f32::round(t * b) / b)
}

/// The F32 vector `t` with each element rounded to `digits` decimal digits.
pub(crate) fn to_vec1_round(t: &Tensor, digits: i32) -> Result<Vec<f32>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec1::<f32>()?;
    let t = t.iter().map(|t| f32::round(t * b) / b).collect();
    Ok(t)
}

/// The F32 matrix `t` with each element rounded to `digits` decimal digits.
pub(crate) fn to_vec2_round(t: &Tensor, digits: i32) -> Result<Vec<Vec<f32>>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec2::<f32>()?;
    let t = t
        .iter()
        .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
        .collect();
    Ok(t)
}

/// The F32 rank-3 tensor `t` with each element rounded to `digits` decimal digits.
pub(crate) fn to_vec3_round(t: &Tensor, digits: i32) -> Result<Vec<Vec<Vec<f32>>>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec3::<f32>()?;
    let t = t
        .iter()
        .map(|t| {
            t.iter()
                .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
                .collect()
        })
        .collect();
    Ok(t)
}
