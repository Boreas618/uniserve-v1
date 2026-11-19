//! The records of a paged cache field: how one entry's values are encoded into bytes.
//!
//! A cache field stores one record per entry: a token's window latent, a group's compressed
//! latent, a group's index key. A record is `width` values in one of the [`Format`]s below, and
//! a plane of the field is `[pages, entries, record]`: `entries` records per page, each
//! `record` bytes (or `width` BF16 elements). Entry `e` of a request lies on page
//! `table[row, e / entries]`, at slot `e % entries` of it.
//!
//! The block-scaled formats keep one scale per block of consecutive values:
//!
//! | Format | Values | Scale per block | Record of 512 | Record of 128 |
//! | --- | --- | --- | --- | --- |
//! | `bf16` | BF16 | — | 1024 B | 256 B |
//! | `mxfp8` | E4M3 | UE8M0 per 32 | 512 + 16 = 528 B | 128 + 4 = 132 B |
//! | `nvfp4` | E2M1, two per byte | E4M3 per 16 | 256 + 32 = 288 B | 64 + 8 = 72 B |
//! | `mxfp4` | E2M1, two per byte | UE8M0 per 32 | 256 + 16 = 272 B | 64 + 4 = 68 B |
//!
//! An E2M1 byte holds the even element of a pair in its low nibble and the odd one in its high
//! nibble. A UE8M0 scale byte `b` is `2^(b − 127)`; an E4M3 scale byte is the E4M3 value.
//!
//! **Where the scales lie** is the [`Layout`]: after the values of their own record
//! ([`Layout::Entry`], the latent records FlashMLA reads), or after the values of every record of
//! the page ([`Layout::Page`], the index-key pages DeepGEMM's paged logits read). Either way a
//! page is `entries × record` bytes.
//!
//! **The encoding** is DeepSeek's reference fake quantization (`inference/kernel.py`,
//! `act_quant` and `fp4_act_quant`), so that a decoded record is exactly the value the reference
//! attends:
//!
//! ```text
//! mxfp8   amax = max(max|x|, 1e-4)        s = 2^ceil(log2(amax · f32(1/448)))   q = e4m3(clamp(x / s, ±448))
//! nvfp4   amax = max(max|x|, 6 · 2^-9)    s = e4m3(amax / 6)                     q = e2m1(clamp(x / s, ±6))
//! mxfp4   amax = max(max|x|, 6 · 2^-126)  s = 2^ceil(log2(amax · f32(1/6)))     q = e2m1(clamp(x / s, ±6))
//! ```
//!
//! with `ceil(log2(·))` read off the FP32 bits (the exponent, plus one for a non-zero mantissa)
//! rather than evaluated, every conversion rounding to nearest even and saturating, and the
//! decoded value `q · s`. Each value's code is a function of its block alone, so two
//! implementations of an encoding agree bit for bit; the scalar rules here are the definition
//! the CPU formulas use and the kernels restate.

use std::fmt;

use platform::Stream;
use tensor::DType;

use crate::abi;
use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::operand::Operand;
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// How a record encodes its values; see the module table.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Format {
    /// BF16 values, no scale.
    Bf16,
    /// E4M3 values with a UE8M0 scale per 32.
    Mxfp8,
    /// E2M1 values with an E4M3 scale per 16.
    Nvfp4,
    /// E2M1 values with a UE8M0 scale per 32.
    Mxfp4,
}

impl Format {
    /// The values one scale covers; `None` for an unscaled format.
    pub const fn block(self) -> Option<u64> {
        match self {
            Self::Bf16 => None,
            Self::Mxfp8 | Self::Mxfp4 => Some(32),
            Self::Nvfp4 => Some(16),
        }
    }

    /// Whether the format packs bytes rather than BF16 elements, so a plane of it is U8.
    pub const fn packed(self) -> bool {
        !matches!(self, Self::Bf16)
    }

    /// The bytes of `width` encoded values, without their scales.
    pub const fn value_bytes(self, width: u64) -> u64 {
        match self {
            Self::Bf16 => 2 * width,
            Self::Mxfp8 => width,
            Self::Nvfp4 | Self::Mxfp4 => width / 2,
        }
    }

    /// The bytes of the scales of `width` values.
    pub const fn scale_bytes(self, width: u64) -> u64 {
        match self.block() {
            Some(block) => width / block,
            None => 0,
        }
    }

    /// The bytes of one record of `width` values: its values and its scales.
    pub const fn record_bytes(self, width: u64) -> u64 {
        self.value_bytes(width) + self.scale_bytes(width)
    }

    /// The element type a plane of this format is viewed in, and the extent of its last axis
    /// for records of `width` values: BF16 elements for [`Format::Bf16`], bytes otherwise.
    pub const fn plane(self, width: u64) -> (DType, u64) {
        if self.packed() {
            (DType::U8, self.record_bytes(width))
        } else {
            (DType::BF16, width)
        }
    }

    /// Refuses a record width the format cannot encode: a positive whole number of scale blocks.
    ///
    /// # Errors
    ///
    /// [`Error::Argument`] naming `operator`.
    pub fn check_width(self, operator: &'static str, width: u64) -> Result<()> {
        let block = self.block().unwrap_or(2);
        if width == 0 || !width.is_multiple_of(block) {
            return Err(Error::argument(
                operator,
                format!("{self} records of {width} values: a positive multiple of {block}"),
            ));
        }
        Ok(())
    }

    /// Checks `plane`, `[pages, entries, …]` records of `width` values in this format, and
    /// answers its page count.
    ///
    /// # Errors
    ///
    /// [`Error::Argument`] naming `operator` and the plane `name`.
    pub fn check_plane(
        self,
        operator: &'static str,
        name: &str,
        plane: &Operand,
        entries: u64,
        width: u64,
    ) -> Result<u64> {
        let (dtype, last) = self.plane(width);
        plane
            .require_rank(operator, 3)
            .and_then(|()| plane.require_shape(operator, &[plane.dim(0), entries, last]))
            .and_then(|()| plane.require_row(operator, dtype, last))
            .map_err(|error| Error::argument(operator, format!("the {name} plane: {error}")))?;
        if !plane.is_row_major() || plane.dim(0) == 0 {
            return Err(Error::argument(
                operator,
                format!("the {name} plane is not contiguous pages of records"),
            ));
        }
        Ok(plane.dim(0))
    }
}

impl fmt::Display for Format {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Bf16 => "bf16",
            Self::Mxfp8 => "mxfp8",
            Self::Nvfp4 => "nvfp4",
            Self::Mxfp4 => "mxfp4",
        })
    }
}

/// Where a page keeps its records' scales; see the module documentation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Layout {
    /// Each record's scales follow its own values.
    Entry,
    /// The page holds every record's values, then every record's scales.
    Page,
}

impl Layout {
    /// The byte offsets, within its page, of entry `slot`'s values and of its scales, for a page
    /// of `entries` records of `width` values in `format`.
    pub const fn offsets(self, format: Format, entries: u64, width: u64, slot: u64) -> (u64, u64) {
        let (values, scales) = (format.value_bytes(width), format.scale_bytes(width));
        match self {
            Self::Entry => {
                let start = slot * (values + scales);
                (start, start + values)
            }
            Self::Page => (slot * values, entries * values + slot * scales),
        }
    }
}

impl fmt::Display for Layout {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Entry => "entry",
            Self::Page => "page",
        })
    }
}

/// The largest E4M3 magnitude.
pub const E4M3_MAX: f32 = 448.0;

/// The largest E2M1 magnitude.
pub const E2M1_MAX: f32 = 6.0;

/// The E2M1 magnitudes, by the code's low three bits.
pub const E2M1_VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// `ceil(log2(x))` of a positive FP32 value, read off its bits: the unbiased exponent, plus one
/// when the mantissa is not zero. The reference's `fast_log2_ceil`; a subnormal reads as `−126`.
pub const fn log2_ceil(x: f32) -> i32 {
    let bits = x.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127;
    let fraction = (bits & 0x7f_ffff != 0) as i32;
    exponent + fraction
}

/// `2^exponent` for an exponent of a normal FP32 value.
pub fn pow2(exponent: i32) -> f32 {
    f32::from_bits(((exponent + 127) as u32) << 23)
}

/// The E4M3 code of `x` (finite), rounded to nearest even and saturated at ±448.
pub fn e4m3_code(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 0x80 } else { 0 };
    let magnitude = f64::from(x.abs()).min(f64::from(E4M3_MAX));

    // The subnormal step is 2^-9 below the smallest normal 2^-6; a normal value of exponent `e`
    // keeps three fraction bits, a step of 2^(e - 3). A rounding up to the next power of two
    // is the next binade's first code, which the arithmetic below reaches by itself.
    let code = if magnitude < 2f64.powi(-6) {
        (magnitude / 2f64.powi(-9)).round_ties_even() as u32
    } else {
        let exponent = magnitude.log2().floor() as i32;
        let steps = (magnitude / 2f64.powi(exponent - 3)).round_ties_even() as u32;
        // `steps` is in 8..=16; 16 is the next binade's first value.
        let (exponent, steps) = if steps == 16 {
            (exponent + 1, 8)
        } else {
            (exponent, steps)
        };
        (((exponent + 7) as u32) << 3) | (steps - 8)
    };
    sign | code as u8
}

/// The value of the E4M3 code `code`.
pub fn e4m3_value(code: u8) -> f32 {
    let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((code >> 3) & 0x0f);
    let fraction = f32::from(code & 0x07);
    let magnitude = if exponent == 0 {
        fraction * 2f32.powi(-9)
    } else {
        (1.0 + fraction / 8.0) * 2f32.powi(exponent - 7)
    };
    sign * magnitude
}

/// The E2M1 code of `x` (finite), rounded to nearest even and saturated at ±6: the four bits
/// `sign · 8 + code`, the sign set for a negative `x`.
pub fn e2m1_code(x: f32) -> u8 {
    let sign = if x < 0.0 { 0x8 } else { 0 };
    let magnitude = x.abs();

    // The boundaries between neighbouring codes are their midpoints; a midpoint goes to the
    // even code (`round_ties_even` over the E2M1 grid).
    let code = match magnitude {
        m if m <= 0.25 => 0,
        m if m < 0.75 => 1,
        m if m <= 1.25 => 2,
        m if m < 1.75 => 3,
        m if m <= 2.5 => 4,
        m if m < 3.5 => 5,
        m if m <= 5.0 => 6,
        _ => 7,
    };
    sign | code
}

/// The value of the E2M1 code `code` (its low four bits).
pub fn e2m1_value(code: u8) -> f32 {
    let magnitude = E2M1_VALUES[usize::from(code & 0x7)];
    if code & 0x8 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

/// Encodes `values`, one record of `format`, into its value bytes and its scale bytes.
///
/// The values are FP32; a BF16 record rounds each to nearest even.
pub fn encode(format: Format, values: &[f32]) -> (Vec<u8>, Vec<u8>) {
    match format {
        Format::Bf16 => (
            values
                .iter()
                .flat_map(|value| half::bf16::from_f32(*value).to_le_bytes())
                .collect(),
            Vec::new(),
        ),
        Format::Mxfp8 => {
            let mut codes = Vec::with_capacity(values.len());
            let mut scales = Vec::with_capacity(values.len() / 32);
            for block in values.chunks(32) {
                let exponent = power_scale(block, 1e-4, E4M3_MAX);
                let scale = pow2(exponent);
                codes.extend(
                    block
                        .iter()
                        .map(|value| e4m3_code((value / scale).clamp(-E4M3_MAX, E4M3_MAX))),
                );
                scales.push((exponent + 127) as u8);
            }
            (codes, scales)
        }
        Format::Nvfp4 => {
            let mut codes = Vec::with_capacity(values.len());
            let mut scales = Vec::with_capacity(values.len() / 16);
            for block in values.chunks(16) {
                let amax = block
                    .iter()
                    .fold(6.0 * 2f32.powi(-9), |peak, value| peak.max(value.abs()));
                let scale_code = e4m3_code(amax / E2M1_MAX);
                let scale = e4m3_value(scale_code);
                codes.extend(
                    block
                        .iter()
                        .map(|value| e2m1_code((value / scale).clamp(-E2M1_MAX, E2M1_MAX))),
                );
                scales.push(scale_code);
            }
            (pack_nibbles(&codes), scales)
        }
        Format::Mxfp4 => {
            let mut codes = Vec::with_capacity(values.len());
            let mut scales = Vec::with_capacity(values.len() / 32);
            for block in values.chunks(32) {
                let exponent = power_scale(block, 6.0 * 2f32.powi(-126), E2M1_MAX);
                let scale = pow2(exponent);
                codes.extend(
                    block
                        .iter()
                        .map(|value| e2m1_code((value / scale).clamp(-E2M1_MAX, E2M1_MAX))),
                );
                scales.push((exponent + 127) as u8);
            }
            (pack_nibbles(&codes), scales)
        }
    }
}

/// Decodes one record of `width` values from its value bytes and scale bytes into FP32.
pub fn decode(format: Format, width: usize, values: &[u8], scales: &[u8]) -> Vec<f32> {
    match format {
        Format::Bf16 => values
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| half::bf16::from_le_bytes(*pair).to_f32())
            .collect(),
        Format::Mxfp8 => (0..width)
            .map(|index| e4m3_value(values[index]) * pow2(i32::from(scales[index / 32]) - 127))
            .collect(),
        Format::Nvfp4 => (0..width)
            .map(|index| e2m1_value(nibble(values, index)) * e4m3_value(scales[index / 16]))
            .collect(),
        Format::Mxfp4 => (0..width)
            .map(|index| {
                e2m1_value(nibble(values, index)) * pow2(i32::from(scales[index / 32]) - 127)
            })
            .collect(),
    }
}

/// The exponent of a power-of-two scale of `block`: `log2_ceil(max(max|x|, floor) · f32(1/max))`.
fn power_scale(block: &[f32], floor: f32, max: f32) -> i32 {
    let amax = block
        .iter()
        .fold(floor, |peak, value| peak.max(value.abs()));
    log2_ceil(amax * (1.0 / max))
}

/// E2M1 codes two per byte, the even element in the low nibble.
fn pack_nibbles(codes: &[u8]) -> Vec<u8> {
    codes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|[even, odd]| (even & 0x0f) | (odd << 4))
        .collect()
}

/// The E2M1 code of element `index` of packed `values`.
fn nibble(values: &[u8], index: usize) -> u8 {
    let byte = values[index / 2];
    if index.is_multiple_of(2) {
        byte & 0x0f
    } else {
        byte >> 4
    }
}

/// One compilation of `record_write`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct RecordWriteDescriptor {
    /// Element type of the rows written.
    pub dtype: DType,
    /// Values of a row and of a record.
    pub width: u32,
    /// Whether the row is RMS-normalized by a learned gain before it is rotated.
    pub norm: bool,
    /// The trailing channels of the row that rotate, in adjacent pairs; zero rotates none.
    pub rotated: u32,
    /// Tokens one entry stands for: the rotation turns entry `e` by the angles of position
    /// `e · ratio`, the first token of its group.
    pub ratio: u32,
    /// The records' format.
    pub format: Format,
    /// Where a page keeps its records' scales.
    pub layout: Layout,
    /// Records per page.
    pub entries: u32,
}

impl fmt::Display for RecordWriteDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("dtype", abi::name(self.dtype))
            .field("width", self.width)
            .field("norm", self.norm)
            .field("rotated", self.rotated)
            .field("ratio", self.ratio)
            .field("format", self.format)
            .field("layout", self.layout)
            .field("entries", self.entries)
            .finish(formatter)
    }
}

/// The arguments of `record_write`.
#[derive(Clone)]
pub struct RecordWriteArgs {
    /// `[tokens, width]`, unit stride along a row: the rows to write.
    pub x: Operand,
    /// `[width]` in the rows' type: the normalization's gain, exactly when the descriptor normalizes.
    pub weight: Option<Operand>,
    /// `[rotated / 2]` F32: the inverse frequency of each rotated pair, exactly when the
    /// descriptor rotates.
    pub inv_freq: Option<Operand>,
    /// `[tokens]` I32: the entry each token's row is written to, in its request's order; a
    /// negative entry writes nothing.
    pub entries: Operand,
    /// `[tokens]` U32: each token's row of the page table.
    pub rows: Operand,
    /// `[rows, max_pages]` U32: each row's pages.
    pub table: Operand,
    /// `[pages, entries, record]`: the plane written, in place.
    pub plane: Operand,
    /// The normalization's epsilon.
    pub eps: f32,
}

/// The write of one record per token into a paged cache plane.
///
/// ```text
/// y      = norm ? x · rsqrt(mean(x²) + eps) · gain : x          FP32, then rounded to dtype
/// y'     = y with its trailing `rotated` channels turned by position e · ratio, adjacent pairs:
///          (y[2j], y[2j+1]) ← (y[2j]·cos − y[2j+1]·sin, y[2j+1]·cos + y[2j]·sin),
///          angle = f32(e · ratio) · inv_freq[j], each value then rounded to dtype
/// record = encode(format, y')                                      (module documentation)
/// ```
///
/// for each token whose entry `e` is not negative, into slot `e mod entries` of page
/// `table[row, (e / entries) mod max_pages]`. The two roundings into the rows' type are the
/// reference's: its normalization returns the row in its input type and its rotation writes
/// back into it. One program writes one record, so two tokens of one launch name distinct
/// entries.
pub struct RecordWrite;

impl Op for RecordWrite {
    const NAME: &'static str = proto::module::operator::RECORD_WRITE;

    type Descriptor = RecordWriteDescriptor;
    type Args<'a> = RecordWriteArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let d = descriptor;
        let width = u64::from(d.width);
        d.format.check_width(Self::NAME, width)?;
        if d.rotated > d.width
            || !d.rotated.is_multiple_of(2)
            || d.ratio == 0
            || d.entries == 0
            || !matches!(d.dtype, DType::BF16 | DType::F32)
        {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "{} rows of {}, {} of them rotated, entries of {} tokens in pages of {}: BF16 \
                     or F32 rows, an even rotation within the row",
                    abi::name(d.dtype),
                    d.width,
                    d.rotated,
                    d.ratio,
                    d.entries
                ),
            ));
        }

        args.x.require_rank(Self::NAME, 2)?;
        let tokens = args.x.dim(0);
        args.x
            .require_shape(Self::NAME, &[tokens, width])
            .and_then(|()| args.x.require_row(Self::NAME, d.dtype, width))
            .map_err(|error| Error::argument(Self::NAME, format!("the rows: {error}")))?;
        if args.x.last_stride() != 1 {
            return Err(Error::argument(
                Self::NAME,
                "the rows are not contiguous along a row".to_owned(),
            ));
        }

        match (&args.weight, d.norm) {
            (Some(weight), true) => {
                weight
                    .require_shape(Self::NAME, &[width])
                    .map_err(|error| Error::argument(Self::NAME, format!("the gain: {error}")))?;
                // The gain is read in the rows' type: the program has one pointer type for it.
                if weight.dtype != d.dtype {
                    return Err(Error::argument(
                        Self::NAME,
                        format!(
                            "a gain of {} elements for {} rows",
                            abi::name(weight.dtype),
                            abi::name(d.dtype)
                        ),
                    ));
                }
            }
            (None, false) => {}
            _ => {
                return Err(Error::argument(
                    Self::NAME,
                    "a gain is passed exactly when the compilation normalizes".to_owned(),
                ));
            }
        }
        match (&args.inv_freq, d.rotated) {
            (Some(frequencies), rotated) if rotated > 0 => {
                let pairs = u64::from(rotated / 2);
                frequencies
                    .require_shape(Self::NAME, &[pairs])
                    .and_then(|()| frequencies.require_row(Self::NAME, DType::F32, pairs))
                    .map_err(|error| {
                        Error::argument(Self::NAME, format!("the inverse frequencies: {error}"))
                    })?;
            }
            (None, 0) => {}
            _ => {
                return Err(Error::argument(
                    Self::NAME,
                    "inverse frequencies are passed exactly when the compilation rotates"
                        .to_owned(),
                ));
            }
        }

        for (name, column, dtype) in [
            ("entries", &args.entries, DType::I32),
            ("rows", &args.rows, DType::U32),
        ] {
            column
                .require_shape(Self::NAME, &[tokens])
                .and_then(|()| column.require_row(Self::NAME, dtype, tokens))
                .map_err(|error| Error::argument(Self::NAME, format!("the {name}: {error}")))?;
        }
        args.table
            .require_rank(Self::NAME, 2)
            .and_then(|()| {
                args.table
                    .require_row(Self::NAME, DType::U32, args.table.dim(1))
            })
            .map_err(|error| Error::argument(Self::NAME, format!("the page table: {error}")))?;
        if !args.table.is_row_major() || args.table.dim(1) == 0 {
            return Err(Error::argument(
                Self::NAME,
                "the page table is not contiguous rows of pages".to_owned(),
            ));
        }
        d.format.check_plane(
            Self::NAME,
            "cache",
            &args.plane,
            u64::from(d.entries),
            width,
        )?;
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        _descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let tokens = args.x.dim(0);

        // A call that neither normalizes nor rotates passes the unread operand for each, which
        // its program, compiled without them, never dereferences.
        let unread = Operand::unread();
        let mut arguments = Arguments::new(Self::NAME);
        arguments
            .view("x", &args.x)
            .view("weight", args.weight.as_ref().unwrap_or(&unread))
            .view("inv_freq", args.inv_freq.as_ref().unwrap_or(&unread))
            .view("entries", &args.entries)
            .view("rows", &args.rows)
            .view("table", &args.table)
            .view("plane", &args.plane)
            .u64("tokens", tokens)
            .u64("x_stride", args.x.stride(0))
            .u64("max_pages", args.table.dim(1))
            .f32("eps", args.eps);
        kernel.launch(LaunchGrid::linear(tokens)?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        // The plane is read-modify-write: the comparison sees the untouched records as well as
        // the written ones.
        RecordWriteArgs {
            plane: twins.in_place(args.plane),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // The bytes of a record are a function of its rotated row alone (module documentation),
        // so two implementations that evaluate the rotation's cosines and sines to the same FP32
        // values agree bit for bit; the plane is bytes, which no elementwise bound describes.
        Tolerance::exact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_write_key_states_the_record() {
        let write = RecordWriteDescriptor {
            dtype: DType::BF16,
            width: 128,
            norm: true,
            rotated: 64,
            ratio: 2,
            format: Format::Mxfp4,
            layout: Layout::Page,
            entries: 32,
        };
        assert_eq!(
            write.to_string(),
            "dtype_bf16_width_128_norm_true_rotated_64_ratio_2_format_mxfp4_layout_page_entries_32"
        );
    }

    #[test]
    fn the_record_sizes_are_the_cache_formats() {
        assert_eq!(Format::Mxfp8.record_bytes(512), 528);
        assert_eq!(Format::Nvfp4.record_bytes(512), 288);
        assert_eq!(Format::Mxfp4.record_bytes(128), 68);
        assert_eq!(Format::Bf16.plane(512), (DType::BF16, 512));
        assert_eq!(
            Layout::Page.offsets(Format::Mxfp4, 64, 128, 3),
            (3 * 64, 64 * 64 + 3 * 4)
        );
        assert_eq!(
            Layout::Entry.offsets(Format::Mxfp8, 64, 512, 2),
            (2 * 528, 2 * 528 + 512)
        );
    }

    #[test]
    fn every_e4m3_and_e2m1_code_encodes_its_own_value() {
        for code in 0..=255_u8 {
            // 0x7f and 0xff are E4M3's NaN; -0 encodes as +0's sign-carrying twin.
            if code & 0x7f == 0x7f {
                continue;
            }
            assert_eq!(e4m3_code(e4m3_value(code)), code, "E4M3 {code:#x}");
        }
        for code in 0..16_u8 {
            if code == 0x8 {
                continue;
            }
            assert_eq!(e2m1_code(e2m1_value(code)), code, "E2M1 {code:#x}");
        }
    }

    #[test]
    fn a_midpoint_rounds_to_the_even_code_and_beyond_the_range_saturates() {
        assert_eq!(e2m1_value(e2m1_code(0.75)), 1.0);
        assert_eq!(e2m1_value(e2m1_code(1.25)), 1.0);
        assert_eq!(e2m1_value(e2m1_code(2.5)), 2.0);
        assert_eq!(e2m1_value(e2m1_code(5.0)), 4.0);
        assert_eq!(e2m1_value(e2m1_code(7.0)), 6.0);
        // 1.0625 lies midway between E4M3's 1.0 and 1.125.
        assert_eq!(e4m3_value(e4m3_code(1.0625)), 1.0);
        assert_eq!(e4m3_value(e4m3_code(1.1875)), 1.25);
        assert_eq!(e4m3_value(e4m3_code(500.0)), 448.0);
        assert_eq!(log2_ceil(1.0), 0);
        assert_eq!(log2_ceil(1.5), 1);
        assert_eq!(log2_ceil(0.25), -2);
    }

    #[test]
    fn a_block_decodes_within_its_scale_and_keeps_its_peak() {
        let values: Vec<f32> = (0..64).map(|index| (index as f32 - 31.5) * 0.37).collect();
        for format in [Format::Mxfp8, Format::Nvfp4, Format::Mxfp4, Format::Bf16] {
            let (codes, scales) = encode(format, &values);
            assert_eq!(
                (codes.len() + scales.len()) as u64,
                format.record_bytes(64),
                "{format}"
            );
            let decoded = decode(format, 64, &codes, &scales);
            let block = format.block().unwrap_or(64) as usize;
            for (start, chunk) in values.chunks(block).enumerate() {
                let peak = chunk
                    .iter()
                    .fold(0.0_f32, |peak, value| peak.max(value.abs()));
                // A relative step of the format's mantissa at the block's peak.
                let step = match format {
                    Format::Mxfp8 => peak / 8.0,
                    Format::Nvfp4 | Format::Mxfp4 => peak / 2.0,
                    Format::Bf16 => peak / 128.0,
                };
                for (index, value) in chunk.iter().enumerate() {
                    let got = decoded[start * block + index];
                    assert!(
                        (got - value).abs() <= step,
                        "{format}: {value} decoded as {got}"
                    );
                }
            }
        }
    }
}
