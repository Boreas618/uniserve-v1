//! The container the block-by-block diagnosis writes: `.npy`, as `numpy.save` writes it.
//!
//! `notes/h3/diagnosis.md` states the protocol: both implementations write the tensors they computed
//! into their own tree, the reference's through `tools/dump_*.py` and this crate's through a
//! harness's `--diag <dir>`, and `tools/diagnosis/compare.py` pairs the two trees by relative path.
//! The two `.h3t` containers beside these files are this crate's own and are read by
//! `examples/compare_blocks.rs`; a `.npy` is what a comparison with the reference can use, so a
//! tensor that is meant to be compared is written in this one.
//!
//! **Nothing here is a diagnostic hook.** No model launch reaches this module: it encodes bytes a
//! caller already has on the host, and a run that names no `--diag` never calls it. A hook in the
//! model would put a file write in the control path of every request, which is why the protocol puts
//! the runner's dumps in the executor and leaves the model pure arithmetic.

use ops::DType;
use std::path::Path;

/// Writes one `.npy` tensor: version 1.0, little-endian floats, C order.
///
/// `payload` is the tensor's bytes exactly as the device produced them, so a `.npy` written here and
/// the `.h3t` written beside it are the same numbers in two containers rather than two computations.
/// The shape is in elements and the payload must hold `4 * product(shape)` bytes, which is why this
/// function takes `f32` data only: the reference's `numpy.save` output reads with it, and a type the
/// reader must widen is a type the comparator widens on its own side.
///
/// The header arithmetic is `numpy`'s own: a preamble of magic (6 bytes), version (2), a 2-byte
/// little-endian header length, then a dictionary padded with spaces and closed by a newline so that
/// the preamble is a multiple of 64 bytes. A reader rejects a file whose padding is short, so this is
/// not a cosmetic detail.
pub fn write_npy(path: &Path, shape: &[u64], payload: &[u8]) -> Result<(), String> {
    let elements: u64 = shape.iter().product();
    if payload.len() as u64 != elements * 4 {
        return Err(format!(
            "{}: {} bytes for {} elements of f32",
            path.display(),
            payload.len(),
            elements
        ));
    }
    let extents = shape
        .iter()
        .map(|extent| extent.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({extents},), }}");
    let preamble = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - preamble % 64) % 64));
    header.push('\n');

    let mut bytes = Vec::with_capacity(10 + header.len() + payload.len());
    bytes.extend_from_slice(b"\x93NUMPY");
    bytes.push(1);
    bytes.push(0);
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(payload);
    std::fs::write(path, bytes).map_err(|error| format!("{}: {error}", path.display()))
}

/// The `f32` bytes of a slice, which is the payload [`write_npy`] takes.
pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// The same tensor in FP32, for a `.npy` written from a tensor of another type.
///
/// BF16 widens by shifting its sixteen bits into the high half of a float and F16 by the standard
/// exponent/mantissa re-bias; both are exact, so a row compared against the reference differs only
/// where the two computations differ. A type this does not widen is refused by name rather than
/// written as if it had been converted.
pub fn f32_bytes_of(dtype: DType, payload: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(payload.len() * 2);
    match dtype {
        DType::BF16 => {
            for pair in payload.as_chunks::<2>().0 {
                let bits = u32::from(u16::from_le_bytes(*pair)) << 16;
                out.extend_from_slice(&f32::from_bits(bits).to_le_bytes());
            }
        }
        DType::F16 => {
            for pair in payload.as_chunks::<2>().0 {
                out.extend_from_slice(&half_to_f32(u16::from_le_bytes(*pair)).to_le_bytes());
            }
        }
        DType::F32 => out.extend_from_slice(payload),
        other => {
            return Err(format!(
                "a `.npy` twin is written from BF16, F16 or F32; this tensor is {other}"
            ));
        }
    }
    Ok(out)
}

/// One IEEE binary16 value as the float it denotes: exact for every finite half, including the
/// subnormals, which is why this is written out rather than approximated by a scale.
fn half_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = u32::from(bits & 0x03ff);
    let value = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            // A subnormal half: shift the mantissa up until its leading bit is the implicit one.
            let mut shift = 0;
            let mut mantissa = mantissa;
            while mantissa & 0x0400 == 0 {
                mantissa <<= 1;
                shift += 1;
            }
            sign | ((127 - 15 - shift + 1) << 23) | ((mantissa & 0x03ff) << 13)
        }
    } else if exponent == 0x1f {
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | (u32::from(exponent) + (127 - 15)) << 23 | (mantissa << 13)
    };
    f32::from_bits(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact values a half denotes, including the three cases a shift-based conversion gets
    /// wrong: a subnormal, the largest finite half, and infinity.
    #[test]
    fn a_half_widens_to_the_float_it_denotes() {
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0x3800), 0.5);
        assert_eq!(half_to_f32(0xc000), -2.0);
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x7bff), 65504.0);
        assert_eq!(half_to_f32(0x0001), 5.960_464_5e-8);
        assert!(half_to_f32(0x7c00).is_infinite());
        assert!(half_to_f32(0xfc00).is_infinite() && half_to_f32(0xfc00) < 0.0);
    }

    /// BF16 is the high half of a float, so the widening is a shift and every value round-trips.
    #[test]
    fn a_bf16_widens_by_shifting_its_bits() {
        let one: u16 = 0x3f80;
        let bytes = one.to_le_bytes();
        assert_eq!(
            f32_bytes_of(DType::BF16, &bytes).unwrap(),
            1.0_f32.to_le_bytes()
        );
        let negative: u16 = 0xbf80;
        assert_eq!(
            f32_bytes_of(DType::BF16, &negative.to_le_bytes()).unwrap(),
            (-1.0_f32).to_le_bytes()
        );
    }

    /// A type this cannot widen exactly is refused by name rather than written as if converted.
    #[test]
    fn a_type_with_no_exact_widening_is_refused() {
        let error = f32_bytes_of(DType::F8E4M3, &[0, 0]).unwrap_err();
        assert!(error.contains("F8E4M3"), "{error}");
    }

    /// An FP32 tensor passes through: the twin of an FP32 run is its own bytes.
    #[test]
    fn f32_passes_through() {
        let values = [0.0_f32, 1.5, -2.25];
        let payload: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        assert_eq!(f32_bytes_of(DType::F32, &payload).unwrap(), payload);
    }
}
