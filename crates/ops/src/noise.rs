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
//!
//! A draw may also start at an **element offset** into that stream: the reference draws every
//! condition, the video latent and the audio latent through one Philox stream, so a shard that owns
//! part of a request's elements addresses the stream at the global flat index it starts at. An
//! element index is `offset + i`, which means an odd offset is legal: it draws the high half of a
//! word first, and there is no alignment rule to violate. Two draws at adjacent offsets reproduce
//! one longer draw, which is what lets a rank draw its own slice and the batch still be the same
//! stream.
//!
//! The key is the request's, and it arrives in one of two forms. **Rows** gives one i64 key per row,
//! so a batch draws one independent stream per request. **Scalar** gives one `u64` key for the whole
//! draw and addresses the buffer flattened: `pair = flat element index / 2`, so a value is a pure
//! function of `(key, element)` with no dependence on the program's id or its position in the grid.
//! The second form is what a diffusion denoiser needs, because its seed is one host value and every
//! element of the latent — of any shape the request happens to have — must redraw from it.
//!
//! A request may also need a **uniform** draw instead of a normal one — the diffusion schedule's
//! noise keeps the same stream so a redraw is reproducible — and that is the same addressing with
//! one word: `out[i] = (words[0] >> 8) / 2^24` in `[0, 1)` for every element, one sample per element
//! rather than one pair. The kind is part of the compilation, because the two draw different values
//! from the same counter and their bounds differ (a normal has no absolute envelope).

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::{self, Stream};
use crate::tensor::{DType, Shape, TensorView};
use crate::tolerance::Tolerance;
use crate::{DevicePtr, Error, Op, Result, Twins};

/// The elements one program draws.
pub const NOISE_BLOCK: u32 = 1024;

/// What a `philox_noise` draw yields.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum NoiseKind {
    /// One standard normal sample per element, `r·cos` for an even and `r·sin` for an odd index.
    Normal,
    /// One sample per element in `[0, 1)`: the 24 bits Philox's first word yields.
    Uniform,
}

impl fmt::Display for NoiseKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Normal => "normal",
            Self::Uniform => "uniform",
        })
    }
}

/// Where a draw's Philox key comes from.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum KeySource {
    /// One i64 key per row of `out`, read from the `keys` view: one stream per row.
    #[default]
    Rows,
    /// One `u64` key for the whole draw, a launch scalar: one stream over the flattened buffer.
    Scalar,
}

impl fmt::Display for KeySource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Rows => "rows",
            Self::Scalar => "scalar",
        })
    }
}

/// One compilation of `philox_noise`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NoiseDescriptor {
    /// Element type of the drawn noise: FP16, BF16 or FP32.
    pub dtype: DType,
    /// The elements one program draws.
    pub block: u32,
    /// Whether the draw is normal or uniform.
    pub kind: NoiseKind,
    /// Where the draw's key comes from: a view of per-row keys, or one launch scalar.
    pub source: KeySource,
}

impl fmt::Display for NoiseDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let key = Key::new()
            .field("dtype", self.dtype)
            .field("block", self.block)
            .field("kind", self.kind);
        // The per-row form is the default, so its key is the one it always had: `source` appears
        // only for the scalar form, which addresses the buffer differently and is another program.
        match self.source {
            KeySource::Rows => key.finish(formatter),
            KeySource::Scalar => key.field("source", self.source).finish(formatter),
        }
    }
}

/// The arguments of `philox_noise`.
#[derive(Clone, Copy)]
pub struct NoiseArgs {
    /// `[rows, elements]`, contiguous: receives one standard normal sample per row.
    pub out: TensorView,
    /// `[rows]` i64: each row's sampling key as its bit pattern, from `ipc`'s
    /// `request_sampling_key`. Present under `KeySource::Rows` and `None` under
    /// `KeySource::Scalar`, where `key` replaces it; a mismatch of the two is refused.
    pub keys: Option<TensorView>,
    /// The whole draw's key as its bit pattern, read under `KeySource::Scalar` only.
    pub key: u64,
    /// The processor stage the draw belongs to.
    pub stage: u32,
    /// The draw index within the token and stage.
    pub draw: u32,
    /// The element of the stream the draw starts at: element `i` of this call is element
    /// `offset + i` of the request's stream.
    ///
    /// Under `KeySource::Scalar` it is the flat index into the whole request's draw; under
    /// `KeySource::Rows` it is the index within each row, so a row sharded across ranks starts each
    /// piece at its own offset. An odd offset is legal: the first element then takes the high half
    /// of a Philox word.
    pub offset: u64,
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
        let plan = plan(descriptor, args)?;

        let mut arguments = Arguments::new();
        arguments
            .view(args.out)
            .pointer(DevicePtr::new(plan.keys))
            .u64(plan.elements)
            .u32(args.stage)
            .u32(args.draw)
            // Last, so the per-row form's leading arguments are the ones it always had.
            .u64(args.key)
            // And the offset after it: an existing package that reads neither is correct at 0.
            .u64(args.offset);
        kernel.launch(plan.grid, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        NoiseArgs {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        match descriptor.kind {
            // Philox is integer arithmetic, so the uniforms agree bit for bit; the normal then costs
            // the logarithm, the square root, the angle and its cosine or sine before the store
            // rounds into the output type.
            NoiseKind::Normal => Tolerance::new(0.0, 4.0 * descriptor.dtype.unit_roundoff() as f32),
            // A 24-bit value scaled by 2^-24 is exact in FP32 and rounds once into a narrower output.
            NoiseKind::Uniform => Tolerance::exact(),
        }
    }
}

/// The Philox word an element of a draw falls in, and which half of it: element `index` of a draw
/// that starts at `offset` is element `offset + index` of the request's stream.
///
/// One program indexes one element by this, so an odd offset draws the high half of a word first and
/// two draws at adjacent offsets reproduce one longer draw. It is the host statement of the rule the
/// kernel program applies, and the tests below pin it to the chained-stream property.
pub const fn word_and_half(offset: u64, index: u64) -> (u64, u32) {
    let element = offset + index;
    (element / 2, (element % 2) as u32)
}

/// How one launch addresses the draw: where the keys are, how many elements it covers and the grid.
struct Plan {
    /// The `keys` view's address under [`KeySource::Rows`], zero under [`KeySource::Scalar`].
    keys: u64,
    /// The row length under [`KeySource::Rows`], the flattened element count under
    /// [`KeySource::Scalar`]: the number of consecutive elements a program indexes.
    elements: u64,
    /// `[rows, ceil(elements / block), 1]` under [`KeySource::Rows`],
    /// `[ceil(elements / block), 1, 1]` under [`KeySource::Scalar`].
    grid: LaunchGrid,
}

/// The addressing of one draw, checked against the form its descriptor names.
fn plan(descriptor: &NoiseDescriptor, args: NoiseArgs) -> Result<Plan> {
    let rows = args.out.shape.dim(0);
    let row_elements = args.out.shape.dim(1);

    // The two key forms address the draw differently, so the view and the grid follow the
    // descriptor and a call that hands the wrong one is refused rather than read.
    match (descriptor.source, args.keys) {
        (KeySource::Rows, Some(keys)) => {
            keys.require_shape(PhiloxNoise::NAME, Shape::new(&[rows]))?;
            keys.require_row(PhiloxNoise::NAME, DType::I64, rows)?;
            if args.key != 0 {
                return Err(Error::argument(
                    PhiloxNoise::NAME,
                    format!(
                        "the per-row form draws from the keys view; key {} is read in the scalar \
                         form only",
                        args.key
                    ),
                ));
            }
            Ok(Plan {
                keys: platform::address(keys.ptr),
                elements: row_elements,
                // One row per program column, the whole row in the other.
                grid: LaunchGrid::new([
                    rows,
                    row_elements.div_ceil(u64::from(descriptor.block)),
                    1,
                ])?,
            })
        }
        (KeySource::Scalar, None) => {
            // One grid over the flattened buffer: the element index is the flat index, so a value
            // depends on the key and its own position and on nothing about the launch.
            let elements = rows * row_elements;
            Ok(Plan {
                keys: 0,
                elements,
                grid: LaunchGrid::new([elements.div_ceil(u64::from(descriptor.block)), 1, 1])?,
            })
        }
        (KeySource::Rows, None) => Err(Error::argument(
            PhiloxNoise::NAME,
            "the per-row form needs its keys view".to_owned(),
        )),
        (KeySource::Scalar, Some(_)) => Err(Error::argument(
            PhiloxNoise::NAME,
            "the scalar form draws from one key and takes no keys view".to_owned(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn out(rows: u64, elements: u64) -> TensorView {
        TensorView::new(
            DevicePtr::new(0x1000),
            DType::BF16,
            Shape::new(&[rows, elements]),
        )
    }

    fn keys(rows: u64) -> TensorView {
        TensorView::new(DevicePtr::new(0x2000), DType::I64, Shape::new(&[rows]))
    }

    fn descriptor(source: KeySource) -> NoiseDescriptor {
        NoiseDescriptor {
            dtype: DType::BF16,
            block: NOISE_BLOCK,
            kind: NoiseKind::Normal,
            source,
        }
    }

    fn args(out: TensorView, keys: Option<TensorView>, key: u64) -> NoiseArgs {
        NoiseArgs {
            out,
            keys,
            key,
            stage: 0,
            draw: 0,
            offset: 0,
        }
    }

    #[test]
    fn an_element_offset_zeroes_the_addressing_it_already_had() {
        // A draw with no offset addresses the stream exactly as before: element i is the low half of
        // word i/2 for even i and the high half for odd i.
        for index in 0..8_u64 {
            assert_eq!(word_and_half(0, index), (index / 2, (index % 2) as u32));
        }
    }

    #[test]
    fn a_draw_at_an_offset_is_the_slice_of_one_longer_draw() {
        // The chained-stream property, for an even and an odd start: element i of a draw at offset k
        // addresses the same word and half as element k + i of a draw at zero.
        for offset in [6_u64, 7] {
            for index in 0..5_u64 {
                assert_eq!(
                    word_and_half(offset, index),
                    word_and_half(0, offset + index),
                    "offset {offset}, element {index}"
                );
            }
        }
        // And an odd offset really does start mid-word rather than rounding down.
        assert_eq!(word_and_half(7, 0), (3, 1));
        assert_eq!(word_and_half(6, 0), (3, 0));
    }

    #[test]
    fn two_draws_at_adjacent_offsets_are_one_draw() {
        // A rank draws its own slice of one stream: video at 0..n, then audio at n.., equals the
        // concatenated draw element for element.
        let video = 5_u64;
        let drawn: Vec<(u64, u32)> = (0..video)
            .map(|index| word_and_half(0, index))
            .chain((0..4).map(|index| word_and_half(video, index)))
            .collect();
        let one: Vec<(u64, u32)> = (0..video + 4)
            .map(|index| word_and_half(0, index))
            .collect();
        assert_eq!(drawn, one);
    }

    #[test]
    fn the_scalar_form_is_the_only_one_that_renames_the_key() {
        // The per-row form's key is the one every published fixture carries; the scalar form
        // addresses the buffer differently and must not share a module with it.
        assert_eq!(
            descriptor(KeySource::Rows).to_string(),
            "dtype_bf16_block_1024_kind_normal"
        );
        assert_eq!(
            descriptor(KeySource::Scalar).to_string(),
            "dtype_bf16_block_1024_kind_normal_source_scalar"
        );
    }

    #[test]
    fn the_scalar_form_covers_the_buffer_flattened_in_one_dimension() {
        // 4 rows of 4096 elements at a block of 1024: the per-row form is a 4 x 4 grid (one column
        // per row) and the scalar form is one row of 16 blocks, so an element's program does not
        // depend on which row it is in.
        let rows_form = plan(
            &descriptor(KeySource::Rows),
            args(out(4, 4096), Some(keys(4)), 0),
        )
        .expect("the per-row form with its keys view");
        assert_eq!(rows_form.grid.blocks, [4, 4, 1]);
        assert_eq!(rows_form.elements, 4096);

        let scalar_form = plan(&descriptor(KeySource::Scalar), args(out(4, 4096), None, 7))
            .expect("the scalar form with its key");
        assert_eq!(scalar_form.grid.blocks, [16, 1, 1]);
        assert_eq!(scalar_form.elements, 4 * 4096);
        assert_eq!(scalar_form.keys, 0);
    }

    #[test]
    fn a_call_that_hands_the_wrong_key_form_is_refused_both_ways() {
        let view = keys(4);
        assert!(
            plan(&descriptor(KeySource::Rows), args(out(4, 16), None, 0)).is_err(),
            "the per-row form read a draw with no keys"
        );
        assert!(
            plan(
                &descriptor(KeySource::Scalar),
                args(out(4, 16), Some(view), 7)
            )
            .is_err(),
            "the scalar form read a keys view it does not address"
        );
        assert!(
            plan(
                &descriptor(KeySource::Rows),
                args(out(4, 16), Some(view), 7)
            )
            .is_err(),
            "the per-row form read a key it does not use"
        );
    }
}
