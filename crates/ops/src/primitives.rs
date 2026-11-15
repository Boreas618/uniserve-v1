//! The tensor library's primitives on the CUDA and meta devices.
//!
//! Every tensor operation reaches a backend storage as one method of `BackendStorage`, and each
//! method here is the launch of one generic primitive operator of `ops` with an output from the
//! allocator of the thread's context:
//!
//! | Primitive | Operator |
//! | --- | --- |
//! | `unary`, `affine`, `powf`, `elu` | `elementwise_unary` |
//! | `binary`, `cmp` | `elementwise_binary` |
//! | `where_cond` | `elementwise_where` |
//! | `to_dtype` | `cast`, and `copy_strided` between one element type and itself |
//! | `view` | none: a storage over the same bytes |
//! | `copy_strided_src`, `copy2d`, `try_clone` | `copy_strided` |
//! | `reduce` | `reduce` |
//! | `index_select`, `gather` | `index` (gather) |
//! | `index_add` | `copy_strided`, then `index` (scatter_add) |
//! | `scatter_set`, `scatter_add_set` | `index` (scatter_set, scatter_add) |
//! | `const_set`, and a device's `zeros` | `fill` |
//! | `matmul` | `gemm`, once per product (see [`CudaStorage::product`]) |
//! | `conv1d`, `conv_transpose1d` | `conv` over one spatial axis |
//! | `conv2d` | `conv` over three spatial axes, a depth of one |
//!
//! Each operator's own `Descriptor::of` states the descriptor of the call, so a call has one
//! key. An operation over no element launches nothing, because the operators refuse an operand
//! without elements, and returns its empty output.
//!
//! Nothing here asks whether a tensor is meta. A launch under a meta context checks its
//! arguments and records its key ([`launch`]), and the output's block comes from the meta
//! context's allocator, so a profile run states the keys and the peak a launching forward
//! produces.
//!
//! The generic primitives compute on `u8`, `u32`, `i16`, `i32`, `i64`, `f16`, `bf16`, `f32` and
//! `f64`; `remainder`, the `bitwise_` combinations and the shifts exist on the five integer
//! types alone. What an operator does not compute is refused by its own check.
//!
//! An FP8 tensor (`f8e4m3`) is moved and never computed with: `fill`, `copy_strided`, `index`
//! (gather and scatter_set) and `elementwise_where` carry its elements as the bytes they are,
//! so it is zeroed, set, copied, cloned, concatenated, padded, gathered and selected like any
//! other tensor, and read as `u8` and back by `view`. Arithmetic on it, a reduction, a
//! `scatter_add` and a conversion to or from another element type are refused: they are the
//! fused operators' of `nn`, which decode an FP8 operand themselves.
//!
//! An NVFP4 tensor (`f4`) packs two elements into a byte, so no layout addresses one of them
//! and every primitive refuses it. Its storage is moved as bytes: whoever owns the memory
//! states it as a `u8` tensor of the packed bytes ([`crate::retained`], [`crate::external`]),
//! which every primitive above then carries, while the `f4` tensor over the same memory is
//! the operand of the operators that decode it.
//!
//! Pooling, upsampling and `conv_transpose2d` have no operator and are refused with
//! [`Error::Unsupported`].

use std::any::Any;
use std::sync::Arc;

use crate::cast::Cast;
use crate::conv::Conv;
use crate::copy_strided::CopyStrided;
use crate::elementwise::{
    BinaryArgs, BinaryDescriptor, BinaryFunction, ElementwiseBinary, ElementwiseUnary,
    ElementwiseWhere, UnaryArgs, UnaryDescriptor, UnaryFunction, WhereArgs, WhereDescriptor,
};
use crate::fill::Fill;
use crate::gemm::{Gemm, WeightOrder};
use crate::index::Index;
use crate::reduce::Reduce;
use crate::{Operand, cast, conv, copy_strided, gemm, index, reduce};
use tensor::backend::{BackendDevice, BackendStorage};
use tensor::conv::{ParamsConv1D, ParamsConv2D, ParamsConvTranspose1D, ParamsConvTranspose2D};
use tensor::op::{BinaryOp, CmpOp, ReduceOp, UnaryOp};
use tensor::scalar::Scalar;
use tensor::{CpuStorage, DType, Layout, Shape};

use crate::storage::CudaStorage;
use crate::{Error, Result, launch};

/// Elements one thread block of a primitive's kernel computes.
///
/// A primitive's package compiles any power-of-two block and the block is part of the key, so
/// the backend states one: every primitive of every forward shares the modules of this block.
const BLOCK: u32 = 1024;

/// Whether a primitive's kernel is compiled for programmatic dependent launch. Like the block
/// it is part of the key, and the backend states one value for every primitive.
const PDL: bool = false;

/// A new contiguous storage of `shape` elements of `dtype` from the thread's context, with its
/// kernel argument.
fn output(shape: &Shape, dtype: DType) -> Result<(CudaStorage, Operand)> {
    let storage = CudaStorage::of_elements(shape.elem_count(), dtype, None)?;
    let operand = storage.operand(&Layout::contiguous(shape.clone()))?;

    Ok((storage, operand))
}

/// Launches `copy_strided` of `source` into `destination`, which have one shape and at least
/// one element.
fn copy(source: Operand, destination: Operand) -> Result<()> {
    let args = copy_strided::Args {
        source,
        destination,
    };

    launch::<CopyStrided>(&copy_strided::Descriptor::of(&args, BLOCK, PDL), args)
}

/// Stores the element whose bits are `bits` at every position `layout` views in `storage`.
pub(crate) fn fill(storage: &CudaStorage, layout: &Layout, bits: u64) -> Result<()> {
    let destination = storage.operand(layout)?;
    if destination.numel() == 0 {
        return Ok(());
    }

    let args = crate::fill::Args { destination, bits };
    launch::<Fill>(&crate::fill::Descriptor::of(&args, BLOCK, PDL), args)
}

/// The refusal of `operation`, for which this backend's kernels lack `reason`.
fn unsupported(operation: &'static str, reason: impl Into<String>) -> Error {
    Error::Unsupported {
        operation,
        reason: reason.into(),
    }
}

/// A convolution parameter as the 32-bit value a convolution descriptor carries.
fn word(value: usize, operation: &'static str) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        unsupported(
            operation,
            format!("{value} exceeds a convolution's 32-bit parameters"),
        )
    })
}

/// The element as the bits `fill` stores: the element's own encoding in the low bytes.
fn scalar_bits(value: Scalar) -> u64 {
    match value {
        Scalar::U8(value) => u64::from(value),
        Scalar::U32(value) => u64::from(value),
        // A signed integer travels as its two's complement at the element's width.
        Scalar::I16(value) => u64::from(value as u16),
        Scalar::I32(value) => u64::from(value as u32),
        Scalar::I64(value) => value as u64,
        Scalar::BF16(value) => u64::from(value.to_bits()),
        Scalar::F16(value) => u64::from(value.to_bits()),
        Scalar::F32(value) => u64::from(value.to_bits()),
        Scalar::F64(value) => value.to_bits(),
        Scalar::F8E4M3(value) => u64::from(value.to_bits()),
    }
}

/// The stride between consecutive rows of `layout` read as a matrix whose rows are its last
/// axis, when it is one: the last axis is contiguous, the axes before it step as one row axis,
/// and a row does not overlap the next.
///
/// That is the form `gemm` reads a matrix in: a row count, a row length and a row stride, which
/// its library takes as a leading dimension no shorter than a row.
fn row_stride(layout: &Layout) -> Option<usize> {
    let (dims, strides) = (layout.dims(), layout.stride());
    let (&columns, leading) = dims.split_last()?;

    if columns != 1 && strides[leading.len()] != 1 {
        return None;
    }

    // The leading axes folded from the innermost outward: `rows` of them `stride` apart so far.
    let mut rows = 1;
    let mut stride = columns;
    for (&extent, &step) in leading.iter().zip(&strides[..leading.len()]).rev() {
        if extent == 1 {
            continue;
        }
        if rows == 1 {
            stride = step;
        } else if step != rows * stride {
            return None;
        }
        rows *= extent;
    }

    (rows == 1 || stride >= columns).then_some(stride)
}

/// The matrix of `rows` rows of `columns` elements that `layout` views in `storage`, as `gemm`
/// reads one, or `None` when its rows are not evenly spaced contiguous runs ([`row_stride`]).
fn matrix(
    storage: &CudaStorage,
    layout: &Layout,
    rows: usize,
    columns: usize,
) -> Result<Option<Operand>> {
    let Some(stride) = row_stride(layout) else {
        return Ok(None);
    };

    let rows = Layout::new(
        Shape::from_dims(&[rows, columns]),
        vec![stride, 1],
        layout.start_offset(),
    );
    storage.operand(&rows).map(Some)
}

impl CudaStorage {
    /// `function` of every element `layout` views, with the function's `scalars`.
    fn map(&self, function: UnaryFunction, scalars: [f64; 2], layout: &Layout) -> Result<Self> {
        let x = self.operand(layout)?;
        let (storage, out) = output(layout.shape(), self.dtype())?;

        if out.numel() > 0 {
            let args = UnaryArgs { x, out, scalars };
            launch::<ElementwiseUnary>(&UnaryDescriptor::of(function, &args, BLOCK, PDL), args)?;
        }
        Ok(storage)
    }

    /// `function` of the elements of `self` and `rhs`, which their layouts view in one shape.
    fn zip(
        &self,
        function: BinaryFunction,
        rhs: &dyn BackendStorage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        let lhs = self.operand(lhs_layout)?;
        let rhs = Self::of(rhs)?.operand(rhs_layout)?;
        let (storage, out) = output(lhs_layout.shape(), function.output(self.dtype()))?;

        if out.numel() > 0 {
            let args = BinaryArgs { lhs, rhs, out };
            launch::<ElementwiseBinary>(&BinaryDescriptor::of(function, &args, BLOCK, PDL), args)?;
        }
        Ok(storage)
    }

    /// The elements `layout` views, contiguous in a new storage: one `copy_strided`.
    fn packed(&self, layout: &Layout) -> Result<Self> {
        let source = self.operand(layout)?;
        let (storage, destination) = output(layout.shape(), self.dtype())?;

        if destination.numel() > 0 {
            copy(source, destination)?;
        }
        Ok(storage)
    }

    /// The elements `layout` views converted to `dtype`, contiguous in a new storage.
    fn converted(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        // A conversion between one type and itself moves elements, which is `copy_strided`'s.
        if dtype == self.dtype() {
            return self.packed(layout);
        }

        let x = self.operand(layout)?;
        let (storage, out) = output(layout.shape(), dtype)?;

        if out.numel() > 0 {
            let args = cast::Args { x, out };
            launch::<Cast>(&cast::Descriptor::of(&args, BLOCK, PDL), args)?;
        }
        Ok(storage)
    }

    /// The elements `layout` views folded along `axes` by `function`.
    fn folded(&self, function: reduce::Function, layout: &Layout, axes: &[usize]) -> Result<Self> {
        let mut dims = layout.dims().to_vec();
        for &axis in axes {
            dims[axis] = 1;
        }

        let x = self.operand(layout)?;
        let (storage, out) = output(&Shape::from_dims(&dims), function.output(self.dtype()))?;

        // A sum over an axis of no element still has outputs to write, which are zero.
        if out.numel() > 0 {
            let args = reduce::Args { x, out, axes };
            launch::<Reduce>(&reduce::Descriptor::of(function, &args, BLOCK, PDL), args)?;
        }
        Ok(storage)
    }

    /// `self` read along `axis` at `indices`, whose kernel argument has the shape `dims` of the
    /// result.
    fn gathered(
        &self,
        layout: &Layout,
        indices: Operand,
        dims: &[usize],
        axis: usize,
    ) -> Result<Self> {
        let source = self.operand(layout)?;
        let (storage, destination) = output(&Shape::from_dims(dims), self.dtype())?;

        if destination.numel() > 0 {
            let args = index::Args {
                source,
                indices,
                destination,
                axis,
            };
            launch::<Index>(
                &index::Descriptor::of(index::Function::Gather, &args, BLOCK, PDL),
                args,
            )?;
        }
        Ok(storage)
    }

    /// Stores or adds `source` at the positions `indices` name along `axis` of `destination`,
    /// which is contiguous.
    fn scatter(
        function: index::Function,
        source: Operand,
        indices: Operand,
        destination: Operand,
        axis: usize,
    ) -> Result<()> {
        if source.numel() == 0 {
            return Ok(());
        }

        let args = index::Args {
            source,
            indices,
            destination,
            axis,
        };
        launch::<Index>(&index::Descriptor::of(function, &args, BLOCK, PDL), args)
    }

    /// The kernel argument of the one-dimensional indices `layout` views in `self`, repeated
    /// off `axis` to the shape `dims`: one index per position along the axis, stated as every
    /// broadcast operand is, with a zero stride on the other axes.
    fn along(&self, layout: &Layout, dims: &[usize], axis: usize) -> Result<Operand> {
        let mut strides = vec![0; dims.len()];
        strides[axis] = layout.stride().first().copied().unwrap_or(0);

        self.operand(&Layout::new(
            Shape::from_dims(dims),
            strides,
            layout.start_offset(),
        ))
    }

    /// The batched matrix product: `b` products of an `m` by `k` matrix of `self` with a `k` by
    /// `n` matrix of `rhs`.
    ///
    /// `gemm` multiplies one matrix of rows by one weight matrix. Where every batch member
    /// multiplies the same right-hand matrix (a batch of one, and a right-hand side broadcast
    /// over the batch), the batch folds into the rows and the product is one launch; the
    /// right-hand matrix is read in place when its rows or its columns are contiguous runs (a
    /// transposed weight is read in the order it is stored), and through a contiguous copy
    /// otherwise. A batch of distinct right-hand matrices is one launch per member, over
    /// contiguous operands. An operand `gemm` cannot read in place costs one `copy_strided`.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] for an element type `gemm` does not multiply.
    fn product(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> Result<Self> {
        let dtype = self.dtype();
        if !matches!(dtype, DType::BF16 | DType::F16 | DType::F32) {
            return Err(unsupported(
                "matmul",
                format!("the gemm operator multiplies bf16, f16 and f32 matrices, not {dtype:?}"),
            ));
        }
        if u32::try_from(k).is_err() {
            return Err(unsupported(
                "matmul",
                format!("a contraction over {k} elements exceeds the gemm operator's length"),
            ));
        }

        let shape = Shape::from_dims(&[b * m, n]);
        let (storage, out) = output(&shape, dtype)?;
        if out.numel() == 0 {
            return Ok(storage);
        }
        if k == 0 {
            // A sum of no product.
            fill(&storage, &Layout::contiguous(shape), 0)?;
            return Ok(storage);
        }

        // The product accumulates in FP32 and rounds once into the element type, as the CPU
        // backend's product of 16-bit floats does, which is the descriptor every product states.
        let multiply = |input: Operand, weight: Operand, out: Operand, order: WeightOrder| {
            let args = gemm::Args {
                input,
                weight,
                bias: None,
                out,
                alpha: 1.0,
            };

            launch::<Gemm>(&gemm::Descriptor::of(&args, order)?, args)
        };

        let batch = rhs_layout.dims().len() - 2;
        let shared = rhs_layout.dims()[..batch]
            .iter()
            .zip(rhs_layout.stride())
            .all(|(&extent, &stride)| extent == 1 || stride == 0);

        if shared {
            // A copy lives until the launch that reads it has been issued.
            let lhs_copy;
            let input = match matrix(self, lhs_layout, b * m, k)? {
                Some(input) => input,
                None => {
                    lhs_copy = self.packed(lhs_layout)?;
                    lhs_copy.operand(&Layout::contiguous((b * m, k)))?
                }
            };

            // The one right-hand matrix: the last two axes at the first batch member.
            let strides = &rhs_layout.stride()[batch..];
            let stored = Layout::new(
                Shape::from_dims(&[k, n]),
                strides.to_vec(),
                rhs_layout.start_offset(),
            );
            let transposed = Layout::new(
                Shape::from_dims(&[n, k]),
                vec![strides[1], strides[0]],
                rhs_layout.start_offset(),
            );

            let rhs_copy;
            let (weight, order) = if let Some(weight) = matrix(rhs, &stored, k, n)? {
                (weight, WeightOrder::KxN)
            } else if let Some(weight) = matrix(rhs, &transposed, n, k)? {
                (weight, WeightOrder::NxK)
            } else {
                rhs_copy = rhs.packed(&stored)?;
                let weight = rhs_copy.operand(&Layout::contiguous((k, n)))?;
                (weight, WeightOrder::KxN)
            };

            multiply(input, weight, out, order)?;
            return Ok(storage);
        }

        // Distinct right-hand matrices: member `i` of each contiguous operand is one product.
        let (lhs_copy, rhs_copy);
        let (lhs, lhs_start) = if lhs_layout.is_contiguous() {
            (self, lhs_layout.start_offset())
        } else {
            lhs_copy = self.packed(lhs_layout)?;
            (&lhs_copy, 0)
        };
        let (rhs, rhs_start) = if rhs_layout.is_contiguous() {
            (rhs, rhs_layout.start_offset())
        } else {
            rhs_copy = rhs.packed(rhs_layout)?;
            (&rhs_copy, 0)
        };

        for member in 0..b {
            let member_of = |operand: &Self, start: usize, rows: usize, columns: usize| {
                let layout = Layout::contiguous_with_offset(
                    (rows, columns),
                    start + member * rows * columns,
                );
                operand.operand(&layout)
            };

            multiply(
                member_of(lhs, lhs_start, m, k)?,
                member_of(rhs, rhs_start, k, n)?,
                member_of(&storage, 0, m, n)?,
                WeightOrder::KxN,
            )?;
        }
        Ok(storage)
    }

    /// The convolution of the input `layout` views with the filter `kernel_layout` views in
    /// `kernel`, launched as the `conv` operator over packed operands.
    ///
    /// `dims` are the output's dimensions, `spatial` the operator's spatial rank and `window` the
    /// convolution's stride, padding and dilation per spatial axis; a spatial rank of three
    /// states a two-dimensional convolution as the three-dimensional one over a depth of one,
    /// and every operand then gains that axis. The tensor library splits a grouped convolution
    /// into convolutions of one group, and the permission to narrow FP32 inputs to TF32 is a
    /// model's numerical declaration, which a tensor operation does not carry: the descriptor
    /// ([`conv::Descriptor::of`]) states one group and exact FP32 arithmetic.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`], naming `operation`, for an element type the convolution packages
    /// do not compute.
    fn convolved(
        &self,
        layout: &Layout,
        (kernel, kernel_layout): (&Self, &Layout),
        dims: &[usize],
        (spatial, transposed, window): (u32, bool, [[u32; 3]; 3]),
        operation: &'static str,
    ) -> Result<Self> {
        if !matches!(self.dtype(), DType::BF16 | DType::F32) {
            return Err(unsupported(
                operation,
                format!(
                    "the convolution operator computes bf16 and f32, not {:?}",
                    self.dtype()
                ),
            ));
        }

        // The operator's entry derives row-major strides for every operand.
        let (input_copy, kernel_copy);
        let (input, input_start) = if layout.is_contiguous() {
            (self, layout.start_offset())
        } else {
            input_copy = self.packed(layout)?;
            (&input_copy, 0)
        };
        let (kernel, kernel_start) = if kernel_layout.is_contiguous() {
            (kernel, kernel_layout.start_offset())
        } else {
            kernel_copy = kernel.packed(kernel_layout)?;
            (&kernel_copy, 0)
        };

        let storage = Self::of_elements(dims.iter().product(), self.dtype(), None)?;
        if dims.contains(&0) {
            return Ok(storage);
        }

        // `[n, c, h, w]` as `[n, c, 1, h, w]` for the three-dimensional operator.
        let view = |operand: &Self, dims: &[usize], start: usize| {
            let dims = match spatial {
                3 => vec![dims[0], dims[1], 1, dims[2], dims[3]],
                _ => dims.to_vec(),
            };
            operand.operand(&Layout::contiguous_with_offset(dims, start))
        };

        let args = conv::Args {
            input: view(input, layout.dims(), input_start)?,
            weight: view(kernel, kernel_layout.dims(), kernel_start)?,
            bias: None,
            out: view(&storage, dims, 0)?,
        };
        let descriptor = conv::Descriptor::of(&args, transposed, window, 1, false)?;
        launch::<Conv>(&descriptor, args)?;

        Ok(storage)
    }
}

impl BackendStorage for CudaStorage {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn dtype(&self) -> DType {
        Self::dtype(self)
    }

    fn device(&self) -> Arc<dyn BackendDevice> {
        Self::device(self)
    }

    fn try_clone(&self, _layout: &Layout) -> tensor::Result<Box<dyn BackendStorage>> {
        // The copy keeps every element at its position, so it is the whole storage that moves.
        let whole = Layout::contiguous(self.elements());
        let source = self.operand(&whole)?;

        let clone = Self::allocate(self.bytes(), self.dtype(), None)?;
        let destination = clone.operand(&whole)?;
        if destination.numel() > 0 {
            copy(source, destination)?;
        }

        Ok(Box::new(clone))
    }

    fn to_cpu_storage(&self) -> tensor::Result<CpuStorage> {
        Ok(self.read()?)
    }

    fn affine(
        &self,
        layout: &Layout,
        mul: f64,
        add: f64,
    ) -> tensor::Result<Box<dyn BackendStorage>> {
        Ok(Box::new(self.map(
            UnaryFunction::Affine,
            [mul, add],
            layout,
        )?))
    }

    fn powf(&self, layout: &Layout, exponent: f64) -> tensor::Result<Box<dyn BackendStorage>> {
        Ok(Box::new(self.map(
            UnaryFunction::Powf,
            [exponent, 0.0],
            layout,
        )?))
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> tensor::Result<Box<dyn BackendStorage>> {
        Ok(Box::new(self.map(
            UnaryFunction::Elu,
            [alpha, 0.0],
            layout,
        )?))
    }

    fn unary(&self, op: UnaryOp, layout: &Layout) -> tensor::Result<Box<dyn BackendStorage>> {
        let function = match op {
            UnaryOp::Exp => UnaryFunction::Exp,
            UnaryOp::Log => UnaryFunction::Log,
            UnaryOp::Sin => UnaryFunction::Sin,
            UnaryOp::Cos => UnaryFunction::Cos,
            UnaryOp::Abs => UnaryFunction::Abs,
            UnaryOp::Neg => UnaryFunction::Neg,
            UnaryOp::Recip => UnaryFunction::Recip,
            UnaryOp::Sqr => UnaryFunction::Sqr,
            UnaryOp::Sqrt => UnaryFunction::Sqrt,
            UnaryOp::Gelu => UnaryFunction::Gelu,
            UnaryOp::GeluErf => UnaryFunction::GeluErf,
            UnaryOp::Erf => UnaryFunction::Erf,
            UnaryOp::Relu => UnaryFunction::Relu,
            UnaryOp::Silu => UnaryFunction::Silu,
            UnaryOp::Sigmoid => UnaryFunction::Sigmoid,
            UnaryOp::Tanh => UnaryFunction::Tanh,
            UnaryOp::Floor => UnaryFunction::Floor,
            UnaryOp::Ceil => UnaryFunction::Ceil,
            UnaryOp::Round => UnaryFunction::Round,
            UnaryOp::Sign => UnaryFunction::Sign,
        };

        Ok(Box::new(self.map(function, [0.0; 2], layout)?))
    }

    fn binary(
        &self,
        op: BinaryOp,
        rhs: &dyn BackendStorage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> tensor::Result<Box<dyn BackendStorage>> {
        let function = match op {
            BinaryOp::Add => BinaryFunction::Add,
            BinaryOp::Mul => BinaryFunction::Mul,
            BinaryOp::Sub => BinaryFunction::Sub,
            BinaryOp::Div => BinaryFunction::Div,
            BinaryOp::Maximum => BinaryFunction::Maximum,
            BinaryOp::Minimum => BinaryFunction::Minimum,
            BinaryOp::Remainder => BinaryFunction::Remainder,
            BinaryOp::BitwiseAnd => BinaryFunction::BitwiseAnd,
            BinaryOp::BitwiseOr => BinaryFunction::BitwiseOr,
            BinaryOp::BitwiseXor => BinaryFunction::BitwiseXor,
            BinaryOp::BitwiseLeftShift => BinaryFunction::BitwiseLeftShift,
            BinaryOp::BitwiseRightShift => BinaryFunction::BitwiseRightShift,
        };

        Ok(Box::new(self.zip(function, rhs, lhs_layout, rhs_layout)?))
    }

    fn cmp(
        &self,
        op: CmpOp,
        rhs: &dyn BackendStorage,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> tensor::Result<Box<dyn BackendStorage>> {
        let function = match op {
            CmpOp::Eq => BinaryFunction::Eq,
            CmpOp::Ne => BinaryFunction::Ne,
            CmpOp::Le => BinaryFunction::Le,
            CmpOp::Ge => BinaryFunction::Ge,
            CmpOp::Lt => BinaryFunction::Lt,
            CmpOp::Gt => BinaryFunction::Gt,
        };

        Ok(Box::new(self.zip(function, rhs, lhs_layout, rhs_layout)?))
    }

    fn where_cond(
        &self,
        layout: &Layout,
        on_true: &dyn BackendStorage,
        true_layout: &Layout,
        on_false: &dyn BackendStorage,
        false_layout: &Layout,
    ) -> tensor::Result<Box<dyn BackendStorage>> {
        let on_true = Self::of(on_true)?;
        let on_false = Self::of(on_false)?;

        let condition = self.operand(layout)?;
        let (on_true, dtype) = (on_true.operand(true_layout)?, on_true.dtype());
        let on_false = on_false.operand(false_layout)?;
        let (storage, out) = output(layout.shape(), dtype)?;

        if out.numel() > 0 {
            let args = WhereArgs {
                condition,
                on_true,
                on_false,
                out,
            };
            launch::<ElementwiseWhere>(&WhereDescriptor::of(&args, BLOCK, PDL), args)?;
        }
        Ok(Box::new(storage))
    }

    fn reduce(
        &self,
        op: ReduceOp,
        layout: &Layout,
        dims: &[usize],
    ) -> tensor::Result<Box<dyn BackendStorage>> {
        let function = match op {
            ReduceOp::Sum => reduce::Function::Sum,
            ReduceOp::Min => reduce::Function::Min,
            ReduceOp::Max => reduce::Function::Max,
            ReduceOp::ArgMin => reduce::Function::ArgMin,
            ReduceOp::ArgMax => reduce::Function::ArgMax,
        };

        Ok(Box::new(self.folded(function, layout, dims)?))
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> tensor::Result<Box<dyn BackendStorage>> {
        Ok(Box::new(self.converted(layout, dtype)?))
    }

    fn view(&self, dtype: DType) -> tensor::Result<Box<dyn BackendStorage>> {
        Ok(Box::new(self.reinterpreted(dtype)?))
    }

    fn matmul(
        &self,
        rhs: &dyn BackendStorage,
        bmnk: (usize, usize, usize, usize),
        lhs_layout: &Layout,
        rhs_layout: &Layout,
    ) -> tensor::Result<Box<dyn BackendStorage>> {
        let rhs = Self::of(rhs)?;

        Ok(Box::new(self.product(rhs, bmnk, lhs_layout, rhs_layout)?))
    }

    fn index_select(
        &self,
        ids: &dyn BackendStorage,
        layout: &Layout,
        ids_layout: &Layout,
        dim: usize,
    ) -> tensor::Result<Box<dyn BackendStorage>> {
        let mut dims = layout.dims().to_vec();
        dims[dim] = ids_layout.shape().elem_count();

        let indices = Self::of(ids)?.along(ids_layout, &dims, dim)?;
        Ok(Box::new(self.gathered(layout, indices, &dims, dim)?))
    }

    fn gather(
        &self,
        layout: &Layout,
        ids: &dyn BackendStorage,
        ids_layout: &Layout,
        dim: usize,
    ) -> tensor::Result<Box<dyn BackendStorage>> {
        let indices = Self::of(ids)?.operand(ids_layout)?;

        Ok(Box::new(self.gathered(
            layout,
            indices,
            ids_layout.dims(),
            dim,
        )?))
    }

    fn index_add(
        &self,
        layout: &Layout,
        ids: &dyn BackendStorage,
        ids_layout: &Layout,
        src: &dyn BackendStorage,
        src_layout: &Layout,
        dim: usize,
    ) -> tensor::Result<Box<dyn BackendStorage>> {
        // The result is a copy of `self` that the slices of `src` are then added into.
        let result = self.packed(layout)?;
        let destination = result.operand(&Layout::contiguous(layout.shape().clone()))?;

        if destination.numel() > 0 {
            let source = Self::of(src)?.operand(src_layout)?;
            let indices = Self::of(ids)?.along(ids_layout, src_layout.dims(), dim)?;

            Self::scatter(
                index::Function::ScatterAdd,
                source,
                indices,
                destination,
                dim,
            )?;
        }
        Ok(Box::new(result))
    }

    fn scatter_set(
        &mut self,
        layout: &Layout,
        ids: &dyn BackendStorage,
        ids_layout: &Layout,
        src: &dyn BackendStorage,
        src_layout: &Layout,
        dim: usize,
    ) -> tensor::Result<()> {
        let source = Self::of(src)?.operand(src_layout)?;
        let indices = Self::of(ids)?.operand(ids_layout)?;
        let destination = self.operand(layout)?;

        Ok(Self::scatter(
            index::Function::ScatterSet,
            source,
            indices,
            destination,
            dim,
        )?)
    }

    fn scatter_add_set(
        &mut self,
        layout: &Layout,
        ids: &dyn BackendStorage,
        ids_layout: &Layout,
        src: &dyn BackendStorage,
        src_layout: &Layout,
        dim: usize,
    ) -> tensor::Result<()> {
        let source = Self::of(src)?.operand(src_layout)?;
        let indices = Self::of(ids)?.operand(ids_layout)?;
        let destination = self.operand(layout)?;

        Ok(Self::scatter(
            index::Function::ScatterAdd,
            source,
            indices,
            destination,
            dim,
        )?)
    }

    fn copy_strided_src(
        &self,
        dst: &mut dyn BackendStorage,
        dst_offset: usize,
        src_layout: &Layout,
    ) -> tensor::Result<()> {
        let source = self.operand(src_layout)?;
        let destination = Self::of(dst)?.operand(&Layout::contiguous_with_offset(
            src_layout.shape().clone(),
            dst_offset,
        ))?;

        if source.numel() > 0 {
            copy(source, destination)?;
        }
        Ok(())
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
    ) -> tensor::Result<()> {
        // `d1` rows of `d2` contiguous elements, the rows a stride apart on each side.
        let rows = |stride: usize, offset: usize| {
            Layout::new(Shape::from_dims(&[d1, d2]), vec![stride, 1], offset)
        };
        let source = self.operand(&rows(src_stride1, src_offset))?;
        let destination = Self::of(dst)?.operand(&rows(dst_stride1, dst_offset))?;

        if source.numel() > 0 {
            copy(source, destination)?;
        }
        Ok(())
    }

    fn const_set(&mut self, value: Scalar, layout: &Layout) -> tensor::Result<()> {
        if value.dtype() != self.dtype() {
            return Err(tensor::Error::DTypeMismatchBinaryOp {
                lhs: self.dtype(),
                rhs: value.dtype(),
                op: "const-set",
            }
            .bt());
        }

        Ok(fill(self, layout, scalar_bits(value))?)
    }

    fn conv1d(
        &self,
        layout: &Layout,
        kernel: &dyn BackendStorage,
        kernel_layout: &Layout,
        params: &ParamsConv1D,
    ) -> tensor::Result<Box<dyn BackendStorage>> {
        const NAME: &str = "conv1d";

        let window = [
            [word(params.stride, NAME)?, 1, 1],
            [word(params.padding, NAME)?, 0, 0],
            [word(params.dilation, NAME)?, 1, 1],
        ];

        Ok(Box::new(self.convolved(
            layout,
            (Self::of(kernel)?, kernel_layout),
            &params.out_dims(),
            (1, false, window),
            NAME,
        )?))
    }

    fn conv_transpose1d(
        &self,
        layout: &Layout,
        kernel: &dyn BackendStorage,
        kernel_layout: &Layout,
        params: &ParamsConvTranspose1D,
    ) -> tensor::Result<Box<dyn BackendStorage>> {
        const NAME: &str = "conv_transpose1d";

        if params.output_padding != 0 {
            return Err(unsupported(
                NAME,
                "the conv operator's transposed pass has no output padding: its output length \
                 follows from the stride, the padding and the filter alone",
            )
            .into());
        }

        let window = [
            [word(params.stride, NAME)?, 1, 1],
            [word(params.padding, NAME)?, 0, 0],
            [word(params.dilation, NAME)?, 1, 1],
        ];

        Ok(Box::new(self.convolved(
            layout,
            (Self::of(kernel)?, kernel_layout),
            &params.out_dims(),
            (1, true, window),
            NAME,
        )?))
    }

    fn conv2d(
        &self,
        layout: &Layout,
        kernel: &dyn BackendStorage,
        kernel_layout: &Layout,
        params: &ParamsConv2D,
    ) -> tensor::Result<Box<dyn BackendStorage>> {
        const NAME: &str = "conv2d";

        // The three-dimensional operator over a depth of one: a filter one deep, stepped,
        // padded and dilated along height and width alone.
        let (stride, padding, dilation) = (
            word(params.stride, NAME)?,
            word(params.padding, NAME)?,
            word(params.dilation, NAME)?,
        );
        let window = [
            [1, stride, stride],
            [0, padding, padding],
            [1, dilation, dilation],
        ];

        Ok(Box::new(self.convolved(
            layout,
            (Self::of(kernel)?, kernel_layout),
            &params.out_dims(),
            (3, false, window),
            NAME,
        )?))
    }

    fn conv_transpose2d(
        &self,
        _layout: &Layout,
        _kernel: &dyn BackendStorage,
        _kernel_layout: &Layout,
        _params: &ParamsConvTranspose2D,
    ) -> tensor::Result<Box<dyn BackendStorage>> {
        Err(unsupported(
            "conv_transpose2d",
            "the transposed convolution operator is one-dimensional",
        )
        .into())
    }

    fn avg_pool2d(
        &self,
        _layout: &Layout,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
    ) -> tensor::Result<Box<dyn BackendStorage>> {
        Err(unsupported("avg_pool2d", "no operator pools a window").into())
    }

    fn max_pool2d(
        &self,
        _layout: &Layout,
        _kernel_size: (usize, usize),
        _stride: (usize, usize),
    ) -> tensor::Result<Box<dyn BackendStorage>> {
        Err(unsupported("max_pool2d", "no operator pools a window").into())
    }

    fn upsample_nearest1d(
        &self,
        _layout: &Layout,
        _size: usize,
    ) -> tensor::Result<Box<dyn BackendStorage>> {
        Err(unsupported("upsample_nearest1d", "no operator resamples a tensor").into())
    }

    fn upsample_nearest2d(
        &self,
        _layout: &Layout,
        _height: usize,
        _width: usize,
    ) -> tensor::Result<Box<dyn BackendStorage>> {
        Err(unsupported("upsample_nearest2d", "no operator resamples a tensor").into())
    }

    fn upsample_bilinear2d(
        &self,
        _layout: &Layout,
        _height: usize,
        _width: usize,
        _align_corners: bool,
        _scale_h: Option<f64>,
        _scale_w: Option<f64>,
    ) -> tensor::Result<Box<dyn BackendStorage>> {
        Err(unsupported("upsample_bilinear2d", "no operator resamples a tensor").into())
    }
}
