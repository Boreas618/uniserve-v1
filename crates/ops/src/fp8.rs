//! The per-tensor FP8 product: a 16-bit activation encoded to E4M3 by its static scale, times an
//! E4M3 weight, accumulated in FP32 and dequantized by the two scales (ModelOpt's `FP8`).
//!
//! ```text
//! q[m, k]   = e4m3(clamp(x[m, k] · (1 / input_scale), ±448))      round to nearest even
//! out[m, n] = (Σ_k q[m, k] · w[n, k]) · alpha  (+ bias[n])        alpha = input_scale · weight_scale
//! ```
//!
//! The encoding is fused into the product's operand load, so the activation is never stored in
//! E4M3: the arithmetic is the reference's (`static_scaled_fp8_quant` followed by a scaled
//! `fp8 × fp8 → fp32` product), one launch instead of two.

use std::fmt;

use uniserve_tensor::DType;

use crate::abi;
use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{Operand, Shape, Strides};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// The largest finite E4M3 magnitude, where an encoded activation saturates.
pub const E4M3_MAX: f32 = 448.0;

/// One compilation of `fp8_gemm`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Fp8GemmDescriptor {
    /// Element type of the activation and the output.
    pub dtype: DType,
    /// Rows of the activation one program covers.
    pub block_m: u32,
    /// Output columns one program covers.
    pub block_n: u32,
    /// Contraction elements one step of a program loads.
    pub block_k: u32,
    /// Whether a bias of the output's columns is added.
    pub bias: bool,
}

impl fmt::Display for Fp8GemmDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("block_m", self.block_m)
            .field("block_n", self.block_n)
            .field("block_k", self.block_k)
            .field("bias", self.bias)
            .finish(formatter)
    }
}

/// The arguments of `fp8_gemm`.
#[derive(Clone, Copy)]
pub struct Fp8GemmArgs {
    /// `[m, k]`, unit-strided per row, rows any stride apart: the 16-bit activation.
    pub input: Operand,
    /// `[n, k]` E4M3, contiguous: the weight, one row per output column.
    pub weight: Operand,
    /// `[n]` in the output type, present exactly when the descriptor adds one.
    pub bias: Option<Operand>,
    /// `[m, n]`, contiguous: receives the product.
    pub out: Operand,
    /// The calibration's static activation scale: an activation is divided by it before encoding.
    pub input_scale: f32,
    /// `input_scale · weight_scale`: the factor the FP32 accumulator is dequantized by.
    pub alpha: f32,
}

/// The per-tensor FP8 product; see the module documentation.
pub struct Fp8Gemm;

impl Op for Fp8Gemm {
    const NAME: &'static str = proto::module::operator::FP8_GEMM;

    type Descriptor = Fp8GemmDescriptor;
    type Args<'a> = Fp8GemmArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        if descriptor.block_m == 0 || descriptor.block_n == 0 || descriptor.block_k == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a tile of no extent covers nothing".to_owned(),
            ));
        }
        args.weight.require_rank(Self::NAME, 2)?;
        let (n, k) = (args.weight.shape.dim(0), args.weight.shape.dim(1));
        args.weight.require_row(Self::NAME, DType::F8E4M3, k)?;
        if args.weight.strides != Strides::row_major(args.weight.shape) {
            return Err(Error::argument(
                Self::NAME,
                "the weight is not contiguous".to_owned(),
            ));
        }
        args.input.require_rank(Self::NAME, 2)?;
        args.input.require_row(Self::NAME, descriptor.dtype, k)?;
        let m = args.input.shape.dim(0);
        args.out.require_shape(Self::NAME, Shape::new(&[m, n]))?;
        args.out.require_row(Self::NAME, descriptor.dtype, n)?;
        if args.out.strides != Strides::row_major(args.out.shape) {
            return Err(Error::argument(
                Self::NAME,
                "the output is not contiguous".to_owned(),
            ));
        }
        match (descriptor.bias, &args.bias) {
            (true, Some(bias)) => {
                bias.require_shape(Self::NAME, Shape::new(&[n]))?;
                bias.require_row(Self::NAME, descriptor.dtype, n)?;
            }
            (false, None) => {}
            _ => {
                return Err(Error::argument(
                    Self::NAME,
                    "the bias and the descriptor disagree".to_owned(),
                ));
            }
        }
        if !(args.input_scale.is_finite() && args.input_scale > 0.0 && args.alpha.is_finite()) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "an input scale of {} and a factor of {} dequantize nothing",
                    args.input_scale, args.alpha
                ),
            ));
        }
        let widest = (m * args.input.strides.get(0)).max(m * n).max(n * k);
        if widest > i32::MAX as u64 {
            return Err(Error::argument(
                Self::NAME,
                format!("the operands span {widest} elements, past the kernel's 32-bit offsets"),
            ));
        }
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let (m, k) = (args.input.shape.dim(0), args.input.shape.dim(1));
        let n = args.weight.shape.dim(0);

        let mut arguments = Arguments::new();
        // A program compiled without a bias never reads its bias port; the output stands in.
        arguments
            .view(args.input)
            .view(args.weight)
            .view(args.bias.unwrap_or(args.out))
            .view(args.out)
            .i32(m as i32)
            .i32(n as i32)
            .i32(k as i32)
            .i32(args.input.strides.get(0) as i32)
            .f32(args.input_scale.recip())
            .f32(args.alpha);
        kernel.launch(
            LaunchGrid::new([
                m.div_ceil(u64::from(descriptor.block_m)),
                n.div_ceil(u64::from(descriptor.block_n)),
                1,
            ])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        Fp8GemmArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // Both implementations encode the same activation to the same E4M3 codes, so they differ
        // only by their FP32 association of the k products and by the one rounding into the
        // output; the reduction length is a launch value, so the bound is stated for the widest
        // contraction a model's projection has (16384).
        Tolerance::dot(16_384, DType::F32, descriptor.dtype)
    }
}
