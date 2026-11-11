//! The device backend interface.
//!
//! The CPU backend is built into this crate. Every other device is a pair of types that
//! implement [`BackendDevice`] and [`BackendStorage`] outside of it: a tensor on such a device
//! has [`Device::Backend`](crate::Device::Backend) as its device and
//! [`Storage::Backend`](crate::Storage::Backend) as its storage, and every tensor operation on
//! it is one call of a method below.
//!
//! Both traits are object safe. A method that takes another storage receives it as
//! `&dyn BackendStorage`; the tensor library has already checked with
//! [`BackendDevice::same_device`] that the operands may meet, so an implementation downcasts
//! through [`BackendStorage::as_any`] to its own storage type.
//!
//! A backend reports a failure of its own as [`Error::Backend`](crate::Error::Backend), which
//! keeps the backend's error type.
//!
//! A layout argument describes how a tensor views its storage: shape, strides and start
//! offset, all in elements. Unless stated otherwise the returned storage holds the result
//! contiguously in row-major order from offset zero. Index operands are `U8`, `U32` or `I64`.
use crate::conv::{ParamsConv1D, ParamsConv2D, ParamsConvTranspose1D, ParamsConvTranspose2D};
use crate::op::{BinaryOp, CmpOp, ReduceOp, UnaryOp};
use crate::scalar::Scalar;
use crate::{CpuStorage, DType, DeviceLocation, Layout, Result, Shape};
use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

/// A device that is not the CPU: where backend storages live and how they are created.
pub trait BackendDevice: Debug + Send + Sync + 'static {
    /// The device as [`Any`], for a backend to downcast to its own device type.
    fn as_any(&self) -> &dyn Any;

    /// The kind and ordinal of the device, such as `("cuda", 0)`.
    fn location(&self) -> DeviceLocation;

    /// Whether tensors of the two devices may meet in one operation.
    ///
    /// The tensor library asks the device of the first operand about the device of each other
    /// operand, so an implementation answers the same way in both directions.
    fn same_device(&self, other: &dyn BackendDevice) -> bool;

    /// A storage of `shape.elem_count()` elements of `dtype`, all zero.
    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>>;

    /// A storage of `shape.elem_count()` elements of `dtype` whose content is unspecified.
    ///
    /// # Safety
    ///
    /// The caller writes every element before it reads one.
    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Box<dyn BackendStorage>>;

    /// The host-to-device copy: a storage on this device with the elements of `storage`.
    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Box<dyn BackendStorage>>;

    /// Blocks until every operation issued on the device has completed.
    fn synchronize(&self) -> Result<()>;
}

/// The elements of a tensor on a [`BackendDevice`], and the primitives tensor operations are
/// built from.
pub trait BackendStorage: Debug + Send + Sync + 'static {
    /// The storage as [`Any`], for a backend to downcast to its own storage type.
    fn as_any(&self) -> &dyn Any;

    /// The mutable counterpart of [`BackendStorage::as_any`].
    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn dtype(&self) -> DType;

    fn device(&self) -> Arc<dyn BackendDevice>;

    /// A storage with its own copy of the elements. The copy keeps the element positions of
    /// `self`, so `layout` views it exactly as it views `self`.
    fn try_clone(&self, layout: &Layout) -> Result<Box<dyn BackendStorage>>;

    /// The device-to-host copy: every element of the storage, at the same positions.
    fn to_cpu_storage(&self) -> Result<CpuStorage>;

    /// `x * mul + add` for every element.
    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Box<dyn BackendStorage>>;

    /// `x.powf(exponent)` for every element.
    fn powf(&self, layout: &Layout, exponent: f64) -> Result<Box<dyn BackendStorage>>;

    /// `x` where it is positive and `alpha * (exp(x) - 1)` elsewhere.
    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Box<dyn BackendStorage>>;

    fn unary(&self, op: UnaryOp, layout: &Layout) -> Result<Box<dyn BackendStorage>>;

    /// Element-wise `op` of two operands of the same shape and dtype.
    fn binary(
        &self,
        op: BinaryOp,
        rhs: &dyn BackendStorage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Box<dyn BackendStorage>>;

    /// Element-wise comparison of two operands of the same shape and dtype. The result is `U8`,
    /// one where the comparison holds and zero elsewhere.
    fn cmp(
        &self,
        op: CmpOp,
        rhs: &dyn BackendStorage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Box<dyn BackendStorage>>;

    /// `self` is the condition, of an integer dtype: the result takes `on_true` where the
    /// condition is non-zero and `on_false` elsewhere.
    fn where_cond(
        &self,
        layout: &Layout,
        on_true: &dyn BackendStorage,
        true_layout: &Layout,
        on_false: &dyn BackendStorage,
        false_layout: &Layout,
    ) -> Result<Box<dyn BackendStorage>>;

    /// Reduces over `dims`, each of which has size one in the result. `Sum` reduces over any
    /// number of dimensions; `Min`, `Max`, `ArgMin` and `ArgMax` receive one. `ArgMin` and
    /// `ArgMax` produce `U32` indices along that dimension.
    fn reduce(
        &self,
        op: ReduceOp,
        layout: &Layout,
        dims: &[usize],
    ) -> Result<Box<dyn BackendStorage>>;

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Box<dyn BackendStorage>>;

    /// Batched matrix product. `bmnk` is `(b, m, n, k)`: `self` views `b` matrices of `m` by
    /// `k`, `rhs` views `b` matrices of `k` by `n`, and the result holds `b` matrices of `m` by
    /// `n`.
    fn matmul(
        &self,
        rhs: &dyn BackendStorage,
        bmnk: (usize, usize, usize, usize),
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Box<dyn BackendStorage>>;

    /// Selects along `dim` the slices of `self` at the one-dimensional indices `ids`.
    fn index_select(
        &self,
        ids: &dyn BackendStorage,
        layout: &Layout,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<Box<dyn BackendStorage>>;

    /// The result has the shape of `ids`; each element is the element of `self` whose index
    /// along `dim` is the value of `ids` and whose other indices are those of the result.
    fn gather(
        &self,
        layout: &Layout,
        ids: &dyn BackendStorage,
        ids_layout: &Layout,
        dim: usize,
    ) -> Result<Box<dyn BackendStorage>>;

    /// A copy of `self` where the slices of `src` along `dim` have been added at the
    /// one-dimensional indices `ids`.
    fn index_add(
        &self,
        layout: &Layout,
        ids: &dyn BackendStorage,
        ids_layout: &Layout,
        src: &dyn BackendStorage,
        src_layout: &Layout,
        dim: usize,
    ) -> Result<Box<dyn BackendStorage>>;

    /// Writes each element of `src` into `self` at the index along `dim` given by `ids`, which
    /// has the shape of `src`.
    fn scatter_set(
        &mut self,
        layout: &Layout,
        ids: &dyn BackendStorage,
        ids_layout: &Layout,
        src: &dyn BackendStorage,
        src_layout: &Layout,
        dim: usize,
    ) -> Result<()>;

    /// The accumulating counterpart of [`BackendStorage::scatter_set`].
    fn scatter_add_set(
        &mut self,
        layout: &Layout,
        ids: &dyn BackendStorage,
        ids_layout: &Layout,
        src: &dyn BackendStorage,
        src_layout: &Layout,
        dim: usize,
    ) -> Result<()>;

    /// Copies the elements `src_layout` views in `self`, in row-major order, into `dst` from
    /// element `dst_offset` on.
    fn copy_strided_src(
        &self,
        dst: &mut dyn BackendStorage,
        dst_offset: usize,
        src_layout: &Layout,
    ) -> Result<()>;

    /// Copies `d1` rows of `d2` elements. Row `i` starts at `src_offset + i * src_stride1` in
    /// `self` and at `dst_offset + i * dst_stride1` in `dst`; all values are in elements.
    #[allow(clippy::too_many_arguments)]
    fn copy2d(
        &self,
        dst: &mut dyn BackendStorage,
        d1: usize,
        d2: usize,
        src_stride1: usize,
        dst_stride1: usize,
        src_offset: usize,
        dst_offset: usize,
    ) -> Result<()>;

    /// Sets every element `layout` views to `value`.
    fn const_set(&mut self, value: Scalar, layout: &Layout) -> Result<()>;

    fn conv1d(
        &self,
        layout: &Layout,
        kernel: &dyn BackendStorage,
        kernel_layout: &Layout,
        params: &ParamsConv1D,
    ) -> Result<Box<dyn BackendStorage>>;

    fn conv_transpose1d(
        &self,
        layout: &Layout,
        kernel: &dyn BackendStorage,
        kernel_layout: &Layout,
        params: &ParamsConvTranspose1D,
    ) -> Result<Box<dyn BackendStorage>>;

    fn conv2d(
        &self,
        layout: &Layout,
        kernel: &dyn BackendStorage,
        kernel_layout: &Layout,
        params: &ParamsConv2D,
    ) -> Result<Box<dyn BackendStorage>>;

    fn conv_transpose2d(
        &self,
        layout: &Layout,
        kernel: &dyn BackendStorage,
        kernel_layout: &Layout,
        params: &ParamsConvTranspose2D,
    ) -> Result<Box<dyn BackendStorage>>;

    fn avg_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Box<dyn BackendStorage>>;

    fn max_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Box<dyn BackendStorage>>;

    fn upsample_nearest1d(&self, layout: &Layout, size: usize) -> Result<Box<dyn BackendStorage>>;

    fn upsample_nearest2d(
        &self,
        layout: &Layout,
        height: usize,
        width: usize,
    ) -> Result<Box<dyn BackendStorage>>;

    /// `scale_h` and `scale_w` are the scale factors the caller asked for, when the output size
    /// was derived from them; the source coordinates are then computed from the factors rather
    /// than from the size ratio, as PyTorch does.
    fn upsample_bilinear2d(
        &self,
        layout: &Layout,
        height: usize,
        width: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Box<dyn BackendStorage>>;
}
