//! Initial noise, drawn on the device from a request's seed.
//!
//! A diffusion request starts from standard normal noise. Drawing it on the device means every
//! element has an address, so a restored request redraws exactly the same values from its seed, and
//! two requests in one batch draw independent streams (specs/worker.md 9, the initial-noise row).
//! The address is the one `crates/ipc/src/philox.rs` gives a random draw:
//!
//! ```text
//! counter = [pair_low, pair_high, stage, draw]      pair = element_index / 2
//! key     = [key_low, key_high]                     the row's key, an i64 of the key's bits
//! words   = philox4x32_10(counter, key)
//! u1      = 1 − (words[0] >> 8) / 2^24              in (0, 1]
//! u2      = (words[1] >> 8) / 2^24                  in [0, 1)
//! r       = sqrt(−2 · ln u1)
//! out[i]  = r · cos(2π · u2)                        for even i
//! out[i]  = r · sin(2π · u2)                        for odd i
//! ```
//!
//! Both elements of a pair share one counter, so the draw consumes the stream in element order and
//! never in batch order. The uniform is the exactly representable 24-bit value Philox's high word
//! yields; the normal's arithmetic is single precision, and the result rounds once into the output's
//! type.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// The elements one program draws.
pub const NOISE_BLOCK: u32 = 1024;

/// One compilation of `philox_noise`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NoiseDescriptor {
    /// Element type of the drawn noise: FP16, BF16 or FP32.
    pub dtype: DType,
    /// The elements one program draws.
    pub block: u32,
}

impl fmt::Display for NoiseDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", self.dtype)
            .field("block", self.block)
            .finish(formatter)
    }
}

/// The arguments of `philox_noise`.
#[derive(Clone, Copy)]
pub struct NoiseArgs {
    /// `[rows, elements]`, contiguous: receives one standard normal sample per row.
    pub out: TensorView,
    /// `[rows]` i64: each row's sampling key as its bit pattern, from `ipc`'s
    /// `request_sampling_key`.
    pub keys: TensorView,
    /// The processor stage the draw belongs to.
    pub stage: u32,
    /// The draw index within the token and stage.
    pub draw: u32,
}

/// Standard normal noise addressed by (key, element).
pub struct PhiloxNoise;

impl Op for PhiloxNoise {
    const NAME: &'static str = "philox_noise";

    type Descriptor = NoiseDescriptor;
    type Args<'a> = NoiseArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        if !matches!(descriptor.dtype, DType::F16 | DType::BF16 | DType::F32) {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} is not a floating noise type: the draw is computed in FP32",
                    descriptor.dtype
                ),
            ));
        }
        if descriptor.block == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a block of zero elements draws nothing".to_owned(),
            ));
        }
        args.out.require_rank(Self::NAME, 2)?;
        if args.out.dtype != descriptor.dtype {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} noise, found {}",
                    descriptor.dtype, args.out.dtype
                ),
            ));
        }
        if args.out.strides != crate::tensor::Strides::row_major(args.out.shape) {
            return Err(Error::argument(
                Self::NAME,
                "the noise buffer is not contiguous".to_owned(),
            ));
        }
        let rows = args.out.shape.dim(0);
        let elements = args.out.shape.dim(1);
        args.keys.require_shape(Self::NAME, Shape::new(&[rows]))?;
        args.keys.require_row(Self::NAME, DType::I64, rows)?;
        if args.keys.dtype != DType::I64 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the sampling keys are {}, not i64 holding the key's bits",
                    args.keys.dtype
                ),
            ));
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.out)
            .view(args.keys)
            .u64(elements)
            .u32(args.stage)
            .u32(args.draw);
        kernel.launch(
            LaunchGrid::new([rows, elements.div_ceil(u64::from(descriptor.block)), 1])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        NoiseArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // Philox is integer arithmetic, so the uniforms agree bit for bit; the normal then costs the
        // logarithm, the square root, the angle and its cosine or sine before the store rounds into
        // the output type.
        Tolerance::new(0.0, 4.0 * descriptor.dtype.unit_roundoff() as f32)
    }
}
