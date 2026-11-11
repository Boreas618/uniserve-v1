//! `channel_affine`: one scale and one shift per channel.
//!
//! ```text
//! out[b, c, p] = x[b, c, p] · scale[c] + shift[c]
//! ```
//!
//! over a contiguous `[batch, channels, positions]` view, which is any contiguous tensor read with
//! its channel axis in the middle: the axes before the channel fold into `batch` and the ones after
//! it into `positions`. A latent restore (`latents * std + mean`) is the map.
//!
//! The product and the sum are two FP32 roundings, as two tensor operations compute them, and never
//! a fused multiply-add: the kernel issues each as one round-to-nearest instruction the compiler
//! cannot contract, and the store rounds into the element type. That is why the tolerance is exact.
//!
//! The geometry is launch words; the key names the element type, the block and the PDL option.

use std::fmt;

use uniserve_tensor::DType;

use crate::abi;
use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{Operand, Shape, Strides};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// One compilation of `channel_affine`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of every operand.
    pub dtype: DType,
    /// Elements one program maps.
    pub block: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("block", self.block)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one `channel_affine`.
#[derive(Clone, Copy)]
pub struct Args {
    /// `[batch, channels, positions]`, contiguous.
    pub x: Operand,
    /// `[channels]`: each channel's factor.
    pub scale: Operand,
    /// `[channels]`: each channel's offset, added after the product.
    pub shift: Operand,
    /// `[batch, channels, positions]`, contiguous: the result; it may be `x` itself.
    pub out: Operand,
}

/// The per-channel affine map.
pub struct ChannelAffine;

impl Op for ChannelAffine {
    const NAME: &'static str = proto::module::operator::CHANNEL_AFFINE;

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        if descriptor.block == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a block of zero elements maps nothing".to_owned(),
            ));
        }
        args.x.require_rank(Self::NAME, 3)?;
        let shape = args.x.shape;
        for (name, view) in [("x", &args.x), ("out", &args.out)] {
            view.require_shape(Self::NAME, shape)?;
            view.require_row(Self::NAME, descriptor.dtype, shape.dim(2))?;
            if view.strides != Strides::row_major(shape) {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "{name} is read as contiguous; found strides {}",
                        view.strides
                    ),
                ));
            }
        }
        let channels = Shape::new(&[shape.dim(1)]);
        for view in [&args.scale, &args.shift] {
            view.require_shape(Self::NAME, channels)?;
            view.require_row(Self::NAME, descriptor.dtype, shape.dim(1))?;
        }
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let shape = args.x.shape;
        let elements = shape.numel();
        let mut arguments = Arguments::new();
        arguments
            .view(args.x)
            .view(args.scale)
            .view(args.shift)
            .view(args.out)
            .u64(shape.dim(1))
            .u64(shape.dim(2))
            .u64(elements)
            .flag(descriptor.pdl);
        kernel.launch(
            LaunchGrid::covering(elements, u64::from(descriptor.block))?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        Args {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // A product and a sum, each rounded once in FP32 as the reference's two tensor operations
        // round them: a fused multiply-add is another computation, not a rounding of this one.
        let _ = descriptor;
        Tolerance::exact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The key names no geometry: one module maps every latent shape of an element type.
    #[test]
    fn the_key_holds_no_geometry() {
        let descriptor = Descriptor {
            dtype: DType::F32,
            block: 1024,
            pdl: false,
        };
        assert_eq!(
            crate::descriptor::key(&descriptor),
            "dtype_f32_block_1024_pdl_false"
        );
    }
}
