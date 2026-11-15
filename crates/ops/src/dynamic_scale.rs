//! `dynamic_scale`: the two factors a product over an encoded activation reads, measured from the
//! activation itself.
//!
//! ```text
//! amax   = max(max_i partials[i], floor)                 the activation's largest magnitude
//! encode = range / amax                                  the encoder's multiplier
//! alpha  = (amax / range) · (weight_amax / range)        the product's dequantization factor
//! ```
//!
//! `partials` are the largest magnitudes of parts of the activation (each row's, from `reduce`'s
//! `absmax`), `weight_amax` the largest magnitude of the weight the activation is multiplied by,
//! and `range` the format's largest code times its largest scale: `448` for per-tensor E4M3, and
//! `6 · 448` for NVFP4, whose E2M1 codes reach 6 under an E4M3 block scale of at most 448. An
//! activation divided by `amax / range` spans the format exactly, and the product of two operands
//! encoded that way is dequantized by the two scales' product.
//!
//! The divisions round to nearest (IEEE), as does the one product, so every implementation writes
//! the same bits: the weight's scale `weight_amax / range` is the number the loader encoded the
//! weight with. `floor` keeps `encode` finite for an activation of zeros, whose codes are zeros at
//! any multiplier.
//!
//! One program reads every partial, `tile` at a time: the count is a row count, a few tens of
//! thousands at most, and the result is two words that the encode and the product wait for.

use std::fmt;

use platform::Stream;
use tensor::DType;

use crate::abi;
use crate::descriptor::{self, Key};
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::launch::PDL;
use crate::operand::Operand;
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// The range of per-tensor E4M3: its largest finite magnitude.
pub const FP8_RANGE: f32 = crate::fp8::E4M3_MAX;

/// The range of NVFP4: the largest E2M1 code under the largest E4M3 block scale, `6 · 448`.
pub const NVFP4_RANGE: f32 = crate::fp4::E2M1_MAX * crate::fp8::E4M3_MAX;

/// The least largest magnitude the factors are computed from: below every magnitude a trained
/// network's 16-bit activation carries that is not zero, and small enough that both ranges'
/// multipliers stay finite.
pub const FLOOR: f32 = 1e-12;

/// One compilation of `dynamic_scale`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of the partial magnitudes: an activation's own (BF16, FP16) or FP32.
    pub dtype: DType,
    /// Partials the one program reads at a time; a power of two.
    pub tile: u32,
    /// Whether the module was compiled for programmatic dependent launch.
    pub pdl: bool,
}

impl Descriptor {
    /// The compilation that folds partials of `dtype` magnitudes into the two factors.
    pub fn new(dtype: DType) -> Self {
        Self {
            dtype,
            tile: descriptor::BLOCK,
            pdl: PDL,
        }
    }
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("tile", self.tile)
            .field("pdl", self.pdl)
            .finish(formatter)
    }
}

/// The arguments of one `dynamic_scale`.
#[derive(Clone)]
pub struct Args {
    /// `[count]`, contiguous, at least one: the largest magnitudes of the activation's parts.
    pub partials: Operand,
    /// `[1]` F32: the largest magnitude of the weight.
    pub weight_amax: Operand,
    /// `[1]` F32: receives the encoder's multiplier `range / amax`.
    pub encode: Operand,
    /// `[1]` F32: receives the product's factor `(amax / range) · (weight_amax / range)`.
    pub alpha: Operand,
    /// The format's largest code times its largest scale.
    pub range: f32,
    /// The least `amax` the factors are computed from.
    pub floor: f32,
}

/// The factors of a product over an activation encoded at its own measured scale.
pub struct DynamicScale;

impl Op for DynamicScale {
    const NAME: &'static str = proto::module::operator::DYNAMIC_SCALE;

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        if !matches!(descriptor.dtype, DType::BF16 | DType::F16 | DType::F32) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} partials; the magnitudes are BF16, FP16 or FP32",
                    abi::name(descriptor.dtype)
                ),
            ));
        }
        if !descriptor.tile.is_power_of_two() {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a tile of {} partials is not a power of two",
                    descriptor.tile
                ),
            ));
        }

        let count = args.partials.numel();
        if args.partials.rank() != 1 || count == 0 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the partials are one axis of at least one magnitude, not {:?}",
                    args.partials.dims()
                ),
            ));
        }
        args.partials
            .require_row(Self::NAME, descriptor.dtype, count)?;

        for scalar in [&args.weight_amax, &args.encode, &args.alpha] {
            scalar.require_shape(Self::NAME, &[1])?;
            scalar.require_row(Self::NAME, DType::F32, 1)?;
        }

        for (what, value) in [("range", args.range), ("floor", args.floor)] {
            if !(value.is_finite() && value > 0.0) {
                return Err(Error::argument(
                    Self::NAME,
                    format!("a {what} of {value} is not a positive number"),
                ));
            }
        }
        if !(args.range / args.floor).is_finite() {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a range of {} over a floor of {} has no finite multiplier",
                    args.range, args.floor
                ),
            ));
        }
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        _descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("partials", &args.partials)
            .view("weight_amax", &args.weight_amax)
            .view("encode", &args.encode)
            .view("alpha", &args.alpha)
            .u64("count", args.partials.numel())
            .f32("range_", args.range)
            .f32("floor", args.floor);

        // One program: the result is one pair of words.
        kernel.launch(LaunchGrid::new([1, 1, 1])?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        Args {
            encode: twins.shadow_of(args.encode),
            alpha: twins.shadow_of(args.alpha),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // A maximum is one of its operands, and the two divisions and the product each round to
        // nearest: two implementations write the same bits.
        Tolerance::exact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use platform::DevicePtr;

    fn operand(address: u64, dtype: DType, shape: &[u64]) -> Operand {
        Operand::new(DevicePtr::from_address(address), dtype, shape)
    }

    fn args(partials: Operand) -> Args {
        Args {
            partials,
            weight_amax: operand(0x2000, DType::F32, &[1]),
            encode: operand(0x3000, DType::F32, &[1]),
            alpha: operand(0x4000, DType::F32, &[1]),
            range: 448.0,
            floor: 1e-12,
        }
    }

    #[test]
    fn the_key_names_the_partials_type_and_the_tile() {
        let descriptor = Descriptor {
            dtype: DType::BF16,
            tile: 1024,
            pdl: false,
        };
        assert_eq!(descriptor.to_string(), "dtype_bf16_tile_1024_pdl_false");
        DynamicScale::check(&descriptor, &args(operand(0x1000, DType::BF16, &[37]))).unwrap();
    }

    #[test]
    fn partials_of_no_element_or_another_type_and_scales_of_another_shape_are_refused() {
        let descriptor = Descriptor {
            dtype: DType::F16,
            tile: 256,
            pdl: false,
        };
        for refused in [
            args(operand(0x1000, DType::F16, &[0])),
            args(operand(0x1000, DType::BF16, &[8])),
            args(operand(0x1000, DType::F16, &[2, 4])),
            Args {
                alpha: operand(0x4000, DType::F32, &[2]),
                ..args(operand(0x1000, DType::F16, &[8]))
            },
            Args {
                range: 0.0,
                ..args(operand(0x1000, DType::F16, &[8]))
            },
            Args {
                floor: f32::MIN_POSITIVE,
                range: 2688.0,
                ..args(operand(0x1000, DType::F16, &[8]))
            },
        ] {
            assert!(DynamicScale::check(&descriptor, &refused).is_err());
        }
    }
}
