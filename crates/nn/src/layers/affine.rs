//! The per-channel affine map `out = x · scale[c] + shift[c]`, with the channel on any axis.
//!
//! A latent restore (`latents * std + mean`) is the established use: a denoiser publishes
//! unit-variance latents and a decoder restores each channel's statistics before its first layer.
//! The product and the sum are two FP32 roundings, as the reference's two tensor operations round
//! them, and never a fused multiply-add, which rounds once and differs in the last bit
//! (`ops::channel_affine`).

use ops::channel_affine::{self, ChannelAffine as ChannelAffineOp};
use ops::{Shape, Strides, Operand};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::launch::PDL;
use crate::weights::Weight;

/// Elements one program maps.
const BLOCK: u32 = 1024;

/// One scale and one shift per channel of the `axis` axis.
#[derive(Clone, Copy, Debug)]
pub struct ChannelAffine {
    scale: Weight,
    shift: Weight,
    axis: usize,
}

impl ChannelAffine {
    /// The map scaling each channel of axis `axis` by `scale` and then shifting it by `shift`, both
    /// `[channels]`.
    ///
    /// # Errors
    ///
    /// A scale or shift that is not one vector, or the two of different lengths.
    pub fn new(scale: Weight, shift: Weight, axis: usize) -> Result<Self> {
        if scale.shape().rank() != 1 || scale.shape() != shift.shape() {
            return Err(Error::shape(
                "ChannelAffine",
                format!(
                    "a scale of {:?} and a shift of {:?} are not one [channels] vector each",
                    scale.shape().dims(),
                    shift.shape().dims()
                ),
            ));
        }
        Ok(Self { scale, shift, axis })
    }

    /// `x` mapped channel by channel into a fresh tensor of its shape and element type; `x` is a
    /// contiguous view whose `axis` axis holds the channels.
    ///
    /// # Errors
    ///
    /// A non-contiguous view, a channel axis past the rank or of another extent, an arena that
    /// cannot hold the output, or the operator's refusal.
    pub fn forward(&self, f: &Forward, x: Operand) -> Result<Operand> {
        const OWNER: &str = "ChannelAffine";
        let shape = x.shape;
        let channels = self.scale.shape().numel();
        if self.axis >= shape.rank() || shape.dim(self.axis) != channels {
            return Err(Error::shape(
                OWNER,
                format!(
                    "{:?} has no axis {} of {channels} channels",
                    shape.dims(),
                    self.axis
                ),
            ));
        }
        if x.strides != Strides::row_major(shape) {
            return Err(Error::shape(
                OWNER,
                format!("x is not a contiguous {:?}", shape.dims()),
            ));
        }
        let out = f.empty(x.dtype, shape)?;
        // The axes before the channel fold into one batch axis and the ones after it into one
        // position axis: a contiguous tensor read as `[batch, channels, positions]`.
        let dims = shape.dims();
        let batch = dims[..self.axis].iter().product::<u64>();
        let positions = dims[self.axis + 1..].iter().product::<u64>();
        let folded = Shape::new(&[batch, channels, positions]);
        let descriptor = channel_affine::Descriptor {
            dtype: x.dtype,
            block: BLOCK,
            pdl: PDL,
        };
        let args = channel_affine::Args {
            x: x.reshape(folded),
            scale: self.scale.view(),
            shift: self.shift.view(),
            out: out.reshape(folded),
        };
        f.operate::<ChannelAffineOp>(&descriptor, args)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Arena;
    use uniserve_tensor::DType;

    /// The channel axis may be any axis of a contiguous view; the map refuses one whose extent is
    /// not the parameters' length and a strided view it cannot fold.
    #[test]
    fn the_channel_axis_is_any_axis_of_the_parameters_length() {
        let arena = Arena::counting();
        let f = Forward::recording(&arena);
        let view = |dims: &[u64]| arena.tensor(DType::F32, Shape::new(dims)).unwrap();
        let held = |view: Operand| Weight {
            backing: crate::Backing::new(view.ptr, view.numel() * 4),
            view,
        };
        let (scale, shift) = (held(view(&[24])), held(view(&[24])));
        let last = ChannelAffine::new(scale, shift, 1).unwrap();
        let rows = view(&[100, 24]);
        assert!(last.forward(&f, rows).is_ok());
        let middle = view(&[2, 24, 50]);
        assert!(last.forward(&f, middle).is_ok());
        assert!(last.forward(&f, view(&[24, 100])).is_err());
        assert!(
            ChannelAffine::new(scale, shift, 2)
                .unwrap()
                .forward(&f, rows)
                .is_err()
        );

        let transposed = Operand::with_strides(
            rows.ptr,
            DType::F32,
            Shape::new(&[100, 24]),
            Strides::new(&[1, 100]),
        );
        assert!(last.forward(&f, transposed).is_err());
        assert!(ChannelAffine::new(scale, held(view(&[12])), 1).is_err());
    }
}
