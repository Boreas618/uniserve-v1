//! The per-tensor FP8 product: a 16-bit activation encoded to E4M3 by its scale, times an E4M3
//! weight, accumulated in FP32 and dequantized by the two scales.
//!
//! ```text
//! fp8_quantize  q[m, k]   = e4m3(clamp(x[m, k] · encode, ±448))           nearest even
//! fp8_gemm      out[m, n] = (Σ_k q[m, k] · w[n, k]) · alpha  (+ bias[n])
//!                                                     alpha = input_scale · weight_scale
//! ```
//!
//! `encode` is the reciprocal of the activation's scale. The arithmetic is the reference's
//! (`scaled_fp8_quant` followed by a scaled `fp8 × fp8 → fp32` product), in two launches: the
//! activation is encoded once into a buffer of codes, and the product loads both operands as
//! stored E4M3 tiles, which runs at twice the rate of a loop that encodes each tile it loads (a
//! 32768 × 28672 × 5376 product on a B200: 1525 TFLOPS against 671 to 721) for one more byte
//! written and read per activation element.
//!
//! Both operators load whole vectors of a row along the contraction, so a contraction and a row
//! stride are multiples of sixteen elements (one vector of E4M3, two of a 16-bit type): their
//! packages compile them as such, and a call of another is refused. A product compiled without
//! that statement ran forty times slower.
//!
//! Both factors are read from device memory. A calibrated export's static scale (ModelOpt's
//! `FP8`) is published at load as two constants; a scale measured from the activation on every
//! call is written by the launch before the encode ([`crate::dynamic_scale`]), which a host
//! scalar could not follow without a synchronization.

use std::fmt;

use platform::Stream;
use tensor::DType;

use crate::abi;
use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::operand::Operand;
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// The largest finite E4M3 magnitude, where an encoded activation saturates.
pub const E4M3_MAX: f32 = 448.0;

/// The elements of one vector a row is loaded in: a contraction and an activation's row stride
/// are multiples of it (module documentation).
pub const VECTOR: u64 = 16;

/// Refuses a contraction of `k` elements or an activation's row stride that is not whole
/// vectors.
fn require_vectors(operator: &'static str, k: u64, stride: u64) -> Result<()> {
    if !k.is_multiple_of(VECTOR) || !stride.is_multiple_of(VECTOR) {
        return Err(Error::argument(
            operator,
            format!(
                "a contraction of {k} elements with rows {stride} apart is not whole vectors of \
                 {VECTOR}"
            ),
        ));
    }
    Ok(())
}

/// One compilation of `fp8_quantize`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Fp8QuantizeDescriptor {
    /// Element type of the activation: BF16 or FP16.
    pub dtype: DType,
    /// Elements of a row one program encodes; a power of two.
    pub block: u32,
}

impl fmt::Display for Fp8QuantizeDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("block", self.block)
            .finish(formatter)
    }
}

/// The arguments of `fp8_quantize`.
#[derive(Clone)]
pub struct Fp8QuantizeArgs {
    /// `[m, k]`, unit-strided per row, rows any stride apart: the 16-bit activation.
    pub input: Operand,
    /// `[m, k]` E4M3, contiguous: receives the codes.
    pub out: Operand,
    /// `[1]` F32: the reciprocal of the activation's scale, which an element is multiplied by
    /// before it is rounded.
    pub encode: Operand,
}

/// Encoding a 16-bit activation into per-tensor E4M3; see the module documentation.
pub struct Fp8Quantize;

impl Op for Fp8Quantize {
    const NAME: &'static str = proto::module::operator::FP8_QUANTIZE;

    type Descriptor = Fp8QuantizeDescriptor;
    type Args<'a> = Fp8QuantizeArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        if !matches!(descriptor.dtype, DType::BF16 | DType::F16) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} activations; E4M3 is encoded from BF16 or FP16",
                    abi::name(descriptor.dtype)
                ),
            ));
        }
        if !descriptor.block.is_power_of_two() {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a block of {} elements is not a power of two",
                    descriptor.block
                ),
            ));
        }
        args.input.require_rank(Self::NAME, 2)?;
        let (m, k) = (args.input.dim(0), args.input.dim(1));
        args.input.require_row(Self::NAME, descriptor.dtype, k)?;
        require_vectors(Self::NAME, k, args.input.stride(0))?;
        args.out.require_shape(Self::NAME, &[m, k])?;
        args.out.require_row(Self::NAME, DType::F8E4M3, k)?;
        if !args.out.is_row_major() {
            return Err(Error::argument(
                Self::NAME,
                "the codes are not contiguous".to_owned(),
            ));
        }
        args.encode.require_shape(Self::NAME, &[1])?;
        args.encode.require_row(Self::NAME, DType::F32, 1)?;
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let (m, k) = (args.input.dim(0), args.input.dim(1));

        let mut arguments = Arguments::new();
        arguments
            .view(&args.input)
            .view(&args.out)
            .u64(k)
            .u64(args.input.stride(0))
            .view(&args.encode);

        // One row a grid row, its blocks across the grid's columns.
        kernel.launch(
            LaunchGrid::new([m, k.div_ceil(u64::from(descriptor.block)), 1])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        Fp8QuantizeArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // One product rounded to FP32 and one rounding to E4M3, both to nearest: two
        // implementations write the same codes.
        Tolerance::exact()
    }
}

/// One compilation of `fp8_gemm`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Fp8GemmDescriptor {
    /// Element type of the output and of the bias.
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
#[derive(Clone)]
pub struct Fp8GemmArgs {
    /// `[m, k]` E4M3, unit-strided per row, rows any stride apart: the encoded activation.
    pub input: Operand,
    /// `[n, k]` E4M3, contiguous: the weight, one row per output column.
    pub weight: Operand,
    /// `[n]` in the output type, present exactly when the descriptor adds one.
    pub bias: Option<Operand>,
    /// `[m, n]`, contiguous: receives the product.
    pub out: Operand,
    /// `[1]` F32: `input_scale · weight_scale`, the factor the FP32 accumulator is dequantized by.
    pub alpha: Operand,
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
        let (n, k) = (args.weight.dim(0), args.weight.dim(1));
        args.weight.require_row(Self::NAME, DType::F8E4M3, k)?;
        if !args.weight.is_row_major() {
            return Err(Error::argument(
                Self::NAME,
                "the weight is not contiguous".to_owned(),
            ));
        }
        args.input.require_rank(Self::NAME, 2)?;
        args.input.require_row(Self::NAME, DType::F8E4M3, k)?;
        require_vectors(Self::NAME, k, args.input.stride(0))?;
        let m = args.input.dim(0);
        args.out.require_shape(Self::NAME, &[m, n])?;
        args.out.require_row(Self::NAME, descriptor.dtype, n)?;
        if !args.out.is_row_major() {
            return Err(Error::argument(
                Self::NAME,
                "the output is not contiguous".to_owned(),
            ));
        }
        match (descriptor.bias, &args.bias) {
            (true, Some(bias)) => {
                bias.require_shape(Self::NAME, &[n])?;
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
        args.alpha.require_shape(Self::NAME, &[1])?;
        args.alpha.require_row(Self::NAME, DType::F32, 1)?;
        let widest = (m * args.input.stride(0)).max(m * n).max(n * k);
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
        let (m, k) = (args.input.dim(0), args.input.dim(1));
        let n = args.weight.dim(0);

        let mut arguments = Arguments::new();
        // A program compiled without a bias never reads its bias port; the output stands in.
        arguments
            .view(&args.input)
            .view(&args.weight)
            .view(args.bias.as_ref().unwrap_or(&args.out))
            .view(&args.out)
            .i32(m as i32)
            .i32(n as i32)
            .i32(k as i32)
            .i32(args.input.stride(0) as i32)
            .view(&args.alpha);
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

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        Fp8GemmArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // Both implementations multiply the same E4M3 codes, so they differ only by their FP32
        // association of the k products and by the one rounding into the output; the reduction
        // length is a launch value, so the bound is stated for the widest contraction a served
        // projection has (the MiniMax-H3 text encoder's 25600-element down projection).
        Tolerance::dot(32_768, DType::F32, descriptor.dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use platform::DevicePtr;

    fn operand(dtype: DType, dims: &[u64], strides: &[u64]) -> Operand {
        Operand::with_strides(DevicePtr::from_address(0x1000), dtype, dims, strides)
    }

    /// A product over whole vectors of a row is accepted; a contraction or an activation row
    /// stride of a partial vector is refused, since the modules load whole vectors.
    #[test]
    fn a_product_takes_whole_vectors_of_its_rows() {
        let descriptor = Fp8GemmDescriptor {
            dtype: DType::BF16,
            block_m: 128,
            block_n: 128,
            block_k: 128,
            bias: false,
        };
        let args = |k: u64, stride: u64| Fp8GemmArgs {
            input: operand(DType::F8E4M3, &[8, k], &[stride, 1]),
            weight: operand(DType::F8E4M3, &[32, k], &[k, 1]),
            bias: None,
            out: operand(DType::BF16, &[8, 32], &[32, 1]),
            alpha: operand(DType::F32, &[1], &[1]),
        };

        assert!(Fp8Gemm::check(&descriptor, &args(64, 64)).is_ok());
        assert!(Fp8Gemm::check(&descriptor, &args(64, 96)).is_ok());
        for (k, stride) in [(40, 40), (64, 72)] {
            let refusal = Fp8Gemm::check(&descriptor, &args(k, stride))
                .unwrap_err()
                .to_string();
            assert!(refusal.contains("not whole vectors of 16"), "{refusal}");
        }

        let quantize = Fp8QuantizeDescriptor {
            dtype: DType::BF16,
            block: 1024,
        };
        let encode = |k: u64| Fp8QuantizeArgs {
            input: operand(DType::BF16, &[8, k], &[k, 1]),
            out: operand(DType::F8E4M3, &[8, k], &[k, 1]),
            encode: operand(DType::F32, &[1], &[1]),
        };
        assert!(Fp8Quantize::check(&quantize, &encode(48)).is_ok());
        assert!(Fp8Quantize::check(&quantize, &encode(40)).is_err());
    }
}
