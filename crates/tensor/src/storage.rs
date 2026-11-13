use crate::backend::BackendStorage;
use crate::conv::{ParamsConv1D, ParamsConv2D, ParamsConvTranspose1D, ParamsConvTranspose2D};
use crate::custom_op::{all_distinct, InplaceOpN, Src};
use crate::op::{BinaryOp, CmpOp, ReduceOp, UnaryOp};
use crate::scalar::Scalar;
use crate::{CpuStorage, DType, Device, Error, Layout, Result, Shape};
use crate::{CustomOp1, CustomOp2, CustomOp3};
use parking_lot::{RwLockReadGuard, RwLockWriteGuard};

/// The elements of a tensor: host memory of the built-in CPU backend, or the storage of a
/// device backend implemented outside this crate.
// We do not want to implement Clone on Storage as cloning may fail because of
// out of memory. Instead try_clone should be used.
#[derive(Debug)]
pub enum Storage {
    Cpu(CpuStorage),
    Backend(Box<dyn BackendStorage>),
}

pub type StorageRef<'a> = RwLockReadGuard<'a, Storage>;
pub type StorageMutRef<'a> = RwLockWriteGuard<'a, Storage>;

impl Storage {
    pub fn try_clone(&self, layout: &Layout) -> Result<Self> {
        match self {
            Self::Cpu(storage) => Ok(Self::Cpu(storage.clone())),
            Self::Backend(storage) => Ok(Self::Backend(storage.try_clone(layout)?)),
        }
    }

    pub fn device(&self) -> Device {
        match self {
            Self::Cpu(_) => Device::Cpu,
            Self::Backend(storage) => Device::Backend(storage.device()),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::Cpu(storage) => storage.dtype(),
            Self::Backend(storage) => storage.dtype(),
        }
    }

    /// The error of an operation whose operands are on devices that may not meet.
    fn device_mismatch(&self, rhs: &Self, op: &'static str) -> Error {
        Error::DeviceMismatchBinaryOp {
            lhs: self.device().location(),
            rhs: rhs.device().location(),
            op,
        }
        .bt()
    }

    pub(crate) fn same_device(&self, rhs: &Self, op: &'static str) -> Result<()> {
        if self.device().same_device(&rhs.device()) {
            Ok(())
        } else {
            Err(self.device_mismatch(rhs, op))
        }
    }

    pub(crate) fn same_dtype(&self, rhs: &Self, op: &'static str) -> Result<()> {
        let lhs = self.dtype();
        let rhs = rhs.dtype();
        if lhs != rhs {
            Err(Error::DTypeMismatchBinaryOp { lhs, rhs, op }.bt())
        } else {
            Ok(())
        }
    }

    pub(crate) fn const_set(&mut self, v: Scalar, l: &Layout) -> Result<()> {
        match self {
            Self::Cpu(storage) => storage.const_set(v, l),
            Self::Backend(storage) => storage.const_set(v, l),
        }
    }

    pub(crate) fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        match self {
            Self::Cpu(storage) => Ok(Self::Cpu(storage.affine(layout, mul, add)?)),
            Self::Backend(storage) => Ok(Self::Backend(storage.affine(layout, mul, add)?)),
        }
    }

    pub(crate) fn powf(&self, layout: &Layout, exponent: f64) -> Result<Self> {
        match self {
            Self::Cpu(storage) => Ok(Self::Cpu(storage.powf(layout, exponent)?)),
            Self::Backend(storage) => Ok(Self::Backend(storage.powf(layout, exponent)?)),
        }
    }

    pub(crate) fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        match self {
            Self::Cpu(storage) => Ok(Self::Cpu(storage.elu(layout, alpha)?)),
            Self::Backend(storage) => Ok(Self::Backend(storage.elu(layout, alpha)?)),
        }
    }

    pub(crate) fn cmp(
        &self,
        op: CmpOp,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        self.same_device(rhs, "cmp")?;
        self.same_dtype(rhs, "cmp")?;
        match (self, rhs) {
            (Self::Cpu(lhs), Self::Cpu(rhs)) => {
                Ok(Self::Cpu(lhs.cmp(op, rhs, lhs_layout, rhs_layout)?))
            }
            (Self::Backend(lhs), Self::Backend(rhs)) => Ok(Self::Backend(lhs.cmp(
                op,
                rhs.as_ref(),
                lhs_layout,
                rhs_layout,
            )?)),
            (lhs, rhs) => Err(lhs.device_mismatch(rhs, "cmp")),
        }
    }

    pub(crate) fn reduce(&self, op: ReduceOp, layout: &Layout, dims: &[usize]) -> Result<Self> {
        match self {
            Self::Cpu(storage) => Ok(Self::Cpu(storage.reduce(op, layout, dims)?)),
            Self::Backend(storage) => Ok(Self::Backend(storage.reduce(op, layout, dims)?)),
        }
    }

    pub(crate) fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        match self {
            Self::Cpu(storage) => Ok(Self::Cpu(storage.to_dtype(layout, dtype)?)),
            Self::Backend(storage) => Ok(Self::Backend(storage.to_dtype(layout, dtype)?)),
        }
    }

    /// The elements of the storage read as `dtype`, an element type of the same size in bytes.
    /// A backend storage shares its memory with the result; a CPU storage is copied.
    pub(crate) fn view(&self, dtype: DType) -> Result<Self> {
        match self {
            Self::Cpu(storage) => Ok(Self::Cpu(storage.view(dtype)?)),
            Self::Backend(storage) => Ok(Self::Backend(storage.view(dtype)?)),
        }
    }

    pub(crate) fn apply_op1(&self, l: &Layout, c: &dyn CustomOp1) -> Result<(Self, Shape)> {
        match self {
            Self::Cpu(storage) => {
                let (storage, shape) = c.cpu_fwd(storage, l)?;
                Ok((Self::Cpu(storage), shape))
            }
            Self::Backend(storage) => {
                let (storage, shape) = c.backend_fwd(storage.as_ref(), l)?;
                Ok((Self::Backend(storage), shape))
            }
        }
    }

    pub(crate) fn apply_op2(
        &self,
        l1: &Layout,
        t2: &Self,
        l2: &Layout,
        c: &dyn CustomOp2,
    ) -> Result<(Self, Shape)> {
        self.same_device(t2, c.name())?;
        match (self, t2) {
            (Self::Cpu(s1), Self::Cpu(s2)) => {
                let (s, shape) = c.cpu_fwd(s1, l1, s2, l2)?;
                Ok((Self::Cpu(s), shape))
            }
            (Self::Backend(s1), Self::Backend(s2)) => {
                let (s, shape) = c.backend_fwd(s1.as_ref(), l1, s2.as_ref(), l2)?;
                Ok((Self::Backend(s), shape))
            }
            (lhs, rhs) => Err(lhs.device_mismatch(rhs, c.name())),
        }
    }

    pub(crate) fn apply_op3(
        &self,
        l1: &Layout,
        t2: &Self,
        l2: &Layout,
        t3: &Self,
        l3: &Layout,
        c: &dyn CustomOp3,
    ) -> Result<(Self, Shape)> {
        self.same_device(t2, c.name())?;
        self.same_device(t3, c.name())?;
        match (self, t2, t3) {
            (Self::Cpu(s1), Self::Cpu(s2), Self::Cpu(s3)) => {
                let (s, shape) = c.cpu_fwd(s1, l1, s2, l2, s3, l3)?;
                Ok((Self::Cpu(s), shape))
            }
            (Self::Backend(s1), Self::Backend(s2), Self::Backend(s3)) => {
                let (s, shape) =
                    c.backend_fwd(s1.as_ref(), l1, s2.as_ref(), l2, s3.as_ref(), l3)?;
                Ok((Self::Backend(s), shape))
            }
            (lhs, rhs, _) => Err(lhs.device_mismatch(rhs, c.name())),
        }
    }

    /// Applies an custom in-place op for the `self` tensor.
    ///
    /// [`Src::Aliased`] use the same underlying storage as `self`, while [`Src::Distinct`] does not.
    /// If there are aliases present the aliased forward function on `InplaceOpN` is called. Otherwise
    /// the normal forward function can be used.
    pub(crate) fn inplace_op<const N: usize, C: InplaceOpN<N>>(
        &mut self,
        dst_l: &Layout,
        srcs: [(Src<'_, Storage>, &Layout); N],
        c: &C,
    ) -> Result<()> {
        // Aliased sources are `self`. Only distinct needs checking.
        for (s, _) in srcs.iter() {
            if let Src::Distinct(s) = s {
                self.same_device(s, c.name())?;
            }
        }

        match self {
            Self::Cpu(dst) => {
                let operands: [(Src<'_, CpuStorage>, &Layout); N] = std::array::from_fn(|i| {
                    let (s, l) = srcs[i];
                    let s = match s {
                        Src::Distinct(Self::Cpu(s)) => Src::Distinct(s),
                        Src::Aliased(rel) => Src::Aliased(rel),
                        Src::Distinct(Self::Backend(_)) => {
                            unreachable!("same_device above rejects a backend source")
                        }
                    };
                    (s, l)
                });

                match all_distinct(&operands) {
                    Some(distinct) => c.cpu_fwd(dst, dst_l, distinct),
                    None => c.cpu_fwd_aliased(dst, dst_l, operands),
                }
            }
            Self::Backend(dst) => {
                let operands: [(Src<'_, dyn BackendStorage>, &Layout); N] =
                    std::array::from_fn(|i| {
                        let (s, l) = srcs[i];
                        let s = match s {
                            Src::Distinct(Self::Backend(s)) => Src::Distinct(s.as_ref()),
                            Src::Aliased(rel) => Src::Aliased(rel),
                            Src::Distinct(Self::Cpu(_)) => {
                                unreachable!("same_device above rejects a cpu source")
                            }
                        };
                        (s, l)
                    });

                match all_distinct(&operands) {
                    Some(distinct) => c.backend_fwd(dst.as_mut(), dst_l, distinct),
                    None => c.backend_fwd_aliased(dst.as_mut(), dst_l, operands),
                }
            }
        }
    }

    pub(crate) fn unary(&self, op: UnaryOp, layout: &Layout) -> Result<Self> {
        match self {
            Self::Cpu(storage) => Ok(Self::Cpu(storage.unary(op, layout)?)),
            Self::Backend(storage) => Ok(Self::Backend(storage.unary(op, layout)?)),
        }
    }

    pub(crate) fn binary(
        &self,
        op: BinaryOp,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        self.same_device(rhs, op.name())?;
        self.same_dtype(rhs, op.name())?;
        match (self, rhs) {
            (Self::Cpu(lhs), Self::Cpu(rhs)) => {
                Ok(Self::Cpu(lhs.binary(op, rhs, lhs_layout, rhs_layout)?))
            }
            (Self::Backend(lhs), Self::Backend(rhs)) => Ok(Self::Backend(lhs.binary(
                op,
                rhs.as_ref(),
                lhs_layout,
                rhs_layout,
            )?)),
            (lhs, rhs) => Err(lhs.device_mismatch(rhs, op.name())),
        }
    }

    pub(crate) fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &ParamsConv1D,
    ) -> Result<Self> {
        self.same_device(kernel, "conv1d")?;
        self.same_dtype(kernel, "conv1d")?;
        match (self, kernel) {
            (Self::Cpu(inp), Self::Cpu(kernel)) => {
                Ok(Self::Cpu(inp.conv1d(l, kernel, kernel_l, params)?))
            }
            (Self::Backend(inp), Self::Backend(kernel)) => Ok(Self::Backend(inp.conv1d(
                l,
                kernel.as_ref(),
                kernel_l,
                params,
            )?)),
            (lhs, rhs) => Err(lhs.device_mismatch(rhs, "conv1d")),
        }
    }

    pub(crate) fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &ParamsConvTranspose1D,
    ) -> Result<Self> {
        self.same_device(kernel, "conv-transpose1d")?;
        self.same_dtype(kernel, "conv-transpose1d")?;
        match (self, kernel) {
            (Self::Cpu(inp), Self::Cpu(kernel)) => Ok(Self::Cpu(
                inp.conv_transpose1d(l, kernel, kernel_l, params)?,
            )),
            (Self::Backend(inp), Self::Backend(kernel)) => Ok(Self::Backend(
                inp.conv_transpose1d(l, kernel.as_ref(), kernel_l, params)?,
            )),
            (lhs, rhs) => Err(lhs.device_mismatch(rhs, "conv-transpose1d")),
        }
    }

    pub(crate) fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &ParamsConv2D,
    ) -> Result<Self> {
        self.same_device(kernel, "conv2d")?;
        self.same_dtype(kernel, "conv2d")?;
        match (self, kernel) {
            (Self::Cpu(inp), Self::Cpu(kernel)) => {
                Ok(Self::Cpu(inp.conv2d(l, kernel, kernel_l, params)?))
            }
            (Self::Backend(inp), Self::Backend(kernel)) => Ok(Self::Backend(inp.conv2d(
                l,
                kernel.as_ref(),
                kernel_l,
                params,
            )?)),
            (lhs, rhs) => Err(lhs.device_mismatch(rhs, "conv2d")),
        }
    }

    pub(crate) fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &ParamsConvTranspose2D,
    ) -> Result<Self> {
        self.same_device(kernel, "conv_transpose2d")?;
        self.same_dtype(kernel, "conv_transpose2d")?;
        match (self, kernel) {
            (Self::Cpu(inp), Self::Cpu(kernel)) => Ok(Self::Cpu(
                inp.conv_transpose2d(l, kernel, kernel_l, params)?,
            )),
            (Self::Backend(inp), Self::Backend(kernel)) => Ok(Self::Backend(
                inp.conv_transpose2d(l, kernel.as_ref(), kernel_l, params)?,
            )),
            (lhs, rhs) => Err(lhs.device_mismatch(rhs, "conv_transpose2d")),
        }
    }

    pub(crate) fn avg_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        match self {
            Self::Cpu(storage) => Ok(Self::Cpu(storage.avg_pool2d(
                layout,
                kernel_size,
                stride,
            )?)),
            Self::Backend(storage) => Ok(Self::Backend(storage.avg_pool2d(
                layout,
                kernel_size,
                stride,
            )?)),
        }
    }

    pub(crate) fn max_pool2d(
        &self,
        layout: &Layout,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self> {
        match self {
            Self::Cpu(storage) => Ok(Self::Cpu(storage.max_pool2d(
                layout,
                kernel_size,
                stride,
            )?)),
            Self::Backend(storage) => Ok(Self::Backend(storage.max_pool2d(
                layout,
                kernel_size,
                stride,
            )?)),
        }
    }

    pub(crate) fn upsample_nearest1d(&self, layout: &Layout, sz: usize) -> Result<Self> {
        match self {
            Self::Cpu(storage) => Ok(Self::Cpu(storage.upsample_nearest1d(layout, sz)?)),
            Self::Backend(storage) => Ok(Self::Backend(storage.upsample_nearest1d(layout, sz)?)),
        }
    }

    pub(crate) fn upsample_nearest2d(&self, layout: &Layout, h: usize, w: usize) -> Result<Self> {
        match self {
            Self::Cpu(storage) => Ok(Self::Cpu(storage.upsample_nearest2d(layout, h, w)?)),
            Self::Backend(storage) => Ok(Self::Backend(storage.upsample_nearest2d(layout, h, w)?)),
        }
    }

    pub(crate) fn upsample_bilinear2d(
        &self,
        layout: &Layout,
        h: usize,
        w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Self> {
        match self {
            Self::Cpu(storage) => Ok(Self::Cpu(storage.upsample_bilinear2d(
                layout,
                h,
                w,
                align_corners,
                scale_h,
                scale_w,
            )?)),
            Self::Backend(storage) => Ok(Self::Backend(storage.upsample_bilinear2d(
                layout,
                h,
                w,
                align_corners,
                scale_h,
                scale_w,
            )?)),
        }
    }

    pub(crate) fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        layout_t: &Layout,
        f: &Self,
        layout_f: &Layout,
    ) -> Result<Self> {
        self.same_device(t, "where")?;
        self.same_device(f, "where")?;
        t.same_dtype(f, "where")?;
        match (self, t, f) {
            (Self::Cpu(cond), Self::Cpu(t), Self::Cpu(f)) => Ok(Self::Cpu(
                cond.where_cond(layout, t, layout_t, f, layout_f)?,
            )),
            (Self::Backend(cond), Self::Backend(t), Self::Backend(f)) => Ok(Self::Backend(
                cond.where_cond(layout, t.as_ref(), layout_t, f.as_ref(), layout_f)?,
            )),
            (_, lhs, rhs) => Err(lhs.device_mismatch(rhs, "where")),
        }
    }

    pub(crate) fn gather(
        &self,
        l: &Layout,
        indexes: &Self,
        indexes_l: &Layout,
        d: usize,
    ) -> Result<Self> {
        self.same_device(indexes, "gather")?;
        match (self, indexes) {
            (Self::Cpu(s), Self::Cpu(indexes)) => {
                Ok(Self::Cpu(s.gather(l, indexes, indexes_l, d)?))
            }
            (Self::Backend(s), Self::Backend(indexes)) => Ok(Self::Backend(s.gather(
                l,
                indexes.as_ref(),
                indexes_l,
                d,
            )?)),
            (lhs, rhs) => Err(lhs.device_mismatch(rhs, "gather")),
        }
    }

    pub(crate) fn scatter_set(
        &mut self,
        l: &Layout,
        indexes: &Self,
        indexes_l: &Layout,
        source: &Self,
        source_l: &Layout,
        d: usize,
    ) -> Result<()> {
        self.same_device(indexes, "scatter-set")?;
        self.same_device(source, "scatter-set")?;
        match (self, indexes, source) {
            (Self::Cpu(s), Self::Cpu(indexes), Self::Cpu(source)) => {
                s.scatter_set(l, indexes, indexes_l, source, source_l, d)
            }
            (Self::Backend(s), Self::Backend(indexes), Self::Backend(source)) => {
                s.scatter_set(l, indexes.as_ref(), indexes_l, source.as_ref(), source_l, d)
            }
            (lhs, rhs, _) => Err(lhs.device_mismatch(rhs, "scatter-set")),
        }
    }

    pub(crate) fn scatter_add_set(
        &mut self,
        l: &Layout,
        indexes: &Self,
        indexes_l: &Layout,
        source: &Self,
        source_l: &Layout,
        d: usize,
    ) -> Result<()> {
        self.same_device(indexes, "scatter-add")?;
        self.same_device(source, "scatter-add")?;
        match (self, indexes, source) {
            (Self::Cpu(s), Self::Cpu(indexes), Self::Cpu(source)) => {
                s.scatter_add_set(l, indexes, indexes_l, source, source_l, d)
            }
            (Self::Backend(s), Self::Backend(indexes), Self::Backend(source)) => {
                s.scatter_add_set(l, indexes.as_ref(), indexes_l, source.as_ref(), source_l, d)
            }
            (lhs, rhs, _) => Err(lhs.device_mismatch(rhs, "scatter-add")),
        }
    }

    pub(crate) fn index_add(
        &self,
        l: &Layout,
        indexes: &Self,
        indexes_l: &Layout,
        source: &Self,
        source_l: &Layout,
        d: usize,
    ) -> Result<Self> {
        self.same_device(indexes, "index-add")?;
        self.same_device(source, "index-add")?;
        match (self, indexes, source) {
            (Self::Cpu(s), Self::Cpu(indexes), Self::Cpu(source)) => Ok(Self::Cpu(
                s.index_add(l, indexes, indexes_l, source, source_l, d)?,
            )),
            (Self::Backend(s), Self::Backend(indexes), Self::Backend(source)) => Ok(Self::Backend(
                s.index_add(l, indexes.as_ref(), indexes_l, source.as_ref(), source_l, d)?,
            )),
            (lhs, rhs, _) => Err(lhs.device_mismatch(rhs, "index-add")),
        }
    }

    pub(crate) fn index_select(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
        d: usize,
    ) -> Result<Self> {
        self.same_device(rhs, "index-select")?;
        match (self, rhs) {
            (Self::Cpu(lhs), Self::Cpu(rhs)) => {
                Ok(Self::Cpu(lhs.index_select(rhs, lhs_l, rhs_l, d)?))
            }
            (Self::Backend(lhs), Self::Backend(rhs)) => Ok(Self::Backend(lhs.index_select(
                rhs.as_ref(),
                lhs_l,
                rhs_l,
                d,
            )?)),
            (lhs, rhs) => Err(lhs.device_mismatch(rhs, "index-select")),
        }
    }

    pub(crate) fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        self.same_device(rhs, "matmul")?;
        self.same_dtype(rhs, "matmul")?;
        match (self, rhs) {
            (Self::Cpu(lhs), Self::Cpu(rhs)) => {
                Ok(Self::Cpu(lhs.matmul(rhs, bmnk, lhs_layout, rhs_layout)?))
            }
            (Self::Backend(lhs), Self::Backend(rhs)) => Ok(Self::Backend(lhs.matmul(
                rhs.as_ref(),
                bmnk,
                lhs_layout,
                rhs_layout,
            )?)),
            (lhs, rhs) => Err(lhs.device_mismatch(rhs, "matmul")),
        }
    }

    // self, the source can be strided whereas dst is contiguous.
    pub(crate) fn copy_strided_src(
        &self,
        dst: &mut Self,
        dst_offset: usize,
        src_l: &Layout,
    ) -> Result<()> {
        self.same_device(dst, "copy")?;
        match (self, dst) {
            (Self::Cpu(src), Self::Cpu(dst)) => src.copy_strided_src(dst, dst_offset, src_l),
            (Self::Backend(src), Self::Backend(dst)) => {
                src.copy_strided_src(dst.as_mut(), dst_offset, src_l)
            }
            (lhs, rhs) => Err(lhs.device_mismatch(rhs, "copy")),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()> {
        self.same_device(dst, "copy2d")?;
        match (self, dst) {
            (Self::Cpu(src), Self::Cpu(dst)) => src.copy2d(dst, d1, d2, src_s, dst_s, src_o, dst_o),
            (Self::Backend(src), Self::Backend(dst)) => {
                src.copy2d(dst.as_mut(), d1, d2, src_s, dst_s, src_o, dst_o)
            }
            (lhs, rhs) => Err(lhs.device_mismatch(rhs, "copy2d")),
        }
    }
}
