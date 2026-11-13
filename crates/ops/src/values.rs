//! Host values in a tensor's element type: the one place a number becomes the bytes of a `DType`
//! and back.
//!
//! Every host-side producer of device bytes — a constant table, a loaded weight's conversion, a
//! media sample — encodes through [`encode`] and reads through [`decode`]. The scalar conversions
//! are the `half` crate's, which the tensor library's own element types are, and follow the
//! device's own conversion so a value narrowed on the host and one narrowed on the card agree bit
//! for bit:
//!
//! - **BF16** keeps an `f32`'s top sixteen bits, rounded to nearest with ties to even. A NaN stays a
//!   quiet NaN, and an infinity stays one.
//! - **F16** rounds to nearest with ties to even, through the subnormal range (a whole number of
//!   `2^-24` units), and overflows to a signed infinity past the largest finite half.
//!
//! The bytes are little-endian, the order of every device this tree serves and of a safetensors
//! payload.

use half::{bf16, f16};
use tensor::DType;

use crate::abi;
use crate::error::{Error, Result};

/// `values` as the little-endian bytes of the floating-point type `dtype`, each rounded to nearest
/// with ties to even.
///
/// # Errors
///
/// For a type other than F32, BF16 or F16: a packed or integer type has no rounding from a real
/// number that a host producer may choose for it.
pub fn encode(dtype: DType, values: &[f32]) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    match dtype {
        DType::F32 => values
            .iter()
            .for_each(|value| bytes.extend_from_slice(&value.to_le_bytes())),
        DType::BF16 => values
            .iter()
            .for_each(|value| bytes.extend_from_slice(&bf16::from_f32(*value).to_le_bytes())),
        DType::F16 => values
            .iter()
            .for_each(|value| bytes.extend_from_slice(&f16::from_f32(*value).to_le_bytes())),
        other => {
            return Err(Error::Shape(format!(
                "real numbers are encoded as f32, bf16 or f16, not {}",
                abi::name(other)
            )));
        }
    }
    Ok(bytes)
}

/// The real values the little-endian bytes of the floating-point type `dtype` hold, exactly.
///
/// # Errors
///
/// For a type other than F32, BF16 or F16, and for a byte count that is not whole elements.
pub fn decode(dtype: DType, bytes: &[u8]) -> Result<Vec<f32>> {
    let width = match dtype {
        DType::F32 => 4,
        DType::BF16 | DType::F16 => 2,
        other => {
            return Err(Error::Shape(format!(
                "real numbers are decoded from f32, bf16 or f16, not {}",
                abi::name(other)
            )));
        }
    };
    if !bytes.len().is_multiple_of(width) {
        return Err(Error::Shape(format!(
            "{} bytes are not whole {} elements",
            bytes.len(),
            abi::name(dtype)
        )));
    }
    Ok(bytes
        .chunks_exact(width)
        .map(|chunk| match dtype {
            DType::F32 => f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
            DType::BF16 => bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32(),
            _ => f16::from_le_bytes([chunk[0], chunk[1]]).to_f32(),
        })
        .collect())
}

/// The bytes of one element of value one: the coefficient a fused operator reads where a
/// scale of one is meant.
///
/// # Errors
///
/// As [`encode`].
pub fn one(dtype: DType) -> Result<Vec<u8>> {
    encode(dtype, &[1.0])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bit pattern `encode` gives `value` in the 16-bit type `dtype`.
    fn bits(dtype: DType, value: f32) -> u16 {
        let bytes = encode(dtype, &[value]).unwrap();
        u16::from_le_bytes([bytes[0], bytes[1]])
    }

    /// The value `decode` reads from the 16-bit pattern `bits` of `dtype`.
    fn value(dtype: DType, bits: u16) -> f32 {
        decode(dtype, &bits.to_le_bytes()).unwrap()[0]
    }

    #[test]
    fn a_half_round_trips_the_values_a_checkpoint_holds() {
        // A subnormal half has an absolute precision of 2^-24, not a relative one, so the bound is
        // half a unit there and relative only from the smallest normal upward.
        const SMALLEST_NORMAL: f32 = 6.103_515_6e-5;
        const HALF_UNIT: f32 = 2.980_232_2e-8;
        for original in [0.0_f32, 1.0, -1.0, 0.5, 0.1, 65_504.0, 1e-5, 1.192_092_9e-7] {
            let back = value(DType::F16, bits(DType::F16, original));
            let bound = if original.abs() < SMALLEST_NORMAL {
                HALF_UNIT
            } else {
                original.abs() * 1e-3
            };
            assert!(
                (back - original).abs() <= bound,
                "{original:e} became {back:e}"
            );
        }
    }

    #[test]
    fn a_half_holds_the_subnormal_range_and_the_specials() {
        // A quarter of the half's magnitude range is subnormal, and a checkpoint holds RMS weights
        // and epsilon-like constants there; the encoding is a whole number of 2^-24 units.
        for (original, units) in [
            (1e-5_f32, 168_u16),
            (5.960_464_5e-8, 1),
            (1.192_092_9e-7, 2),
            (6.097_555_6e-5, 1023),
        ] {
            assert_eq!(bits(DType::F16, original), units, "{original:e}");
            assert!(
                (value(DType::F16, units) - original).abs() <= 5.960_464_5e-8,
                "{original:e}"
            );
        }
        // Below half a unit of the smallest subnormal the value is zero; exactly half a unit ties
        // between 0 and 1, where even wins.
        assert_eq!(bits(DType::F16, 2.0e-8), 0);
        assert_eq!(bits(DType::F16, -2.0e-8), 0x8000);
        assert_eq!(bits(DType::F16, 2.980_232_2e-8), 0);
        assert_eq!(bits(DType::F16, 4.5e-8), 1);
        // A value that rounds up out of the subnormals becomes the smallest normal.
        assert_eq!(bits(DType::F16, 6.101e-5), 0x0400);
        // The largest finite magnitude survives; past it a value overflows to infinity.
        assert_eq!(value(DType::F16, bits(DType::F16, 65504.0)), 65504.0);
        assert_eq!(bits(DType::F16, 65_520.0), 0x7c00);
        assert_eq!(bits(DType::F16, 70_000.0), 0x7c00);
        assert_eq!(bits(DType::F16, -70_000.0), 0xfc00);
        assert!(bits(DType::F16, f32::NAN) & 0x7c00 == 0x7c00);
        assert!(bits(DType::F16, f32::NAN) & 0x03ff != 0);
        assert_eq!(bits(DType::F16, f32::INFINITY), 0x7c00);
        assert_eq!(value(DType::F16, 0x8000), -0.0);
    }

    #[test]
    fn a_bf16_rounds_like_the_hardware() {
        // A value whose low half is zero keeps exactly its top sixteen bits.
        let tiny = f32::from_bits(0x0c00_0000);
        let large = f32::from_bits(0x7f7f_0000);
        for original in [0.0_f32, 1.0, -1.0, 2.0, 0.5, tiny, large] {
            let rounded = bits(DType::BF16, original);
            assert_eq!(rounded, (original.to_bits() >> 16) as u16, "{original:e}");
            assert_eq!(value(DType::BF16, rounded), original);
        }
        // Half a unit in the last place ties between two neighbours and the even one wins; a value
        // past the tie rounds up.
        assert_eq!(bits(DType::BF16, 1.0 + 2.0_f32.powi(-8)), 0x3f80);
        assert_eq!(bits(DType::BF16, 1.0 + 3.0 * 2.0_f32.powi(-9)), 0x3f81);
        // The largest finite value's neighbourhood rounds to infinity, and the specials keep their
        // meaning: a NaN stays a NaN instead of carrying into that infinity.
        assert_eq!(bits(DType::BF16, f32::from_bits(0x7f7f_ffff)), 0x7f80);
        assert_eq!(bits(DType::BF16, f32::INFINITY), 0x7f80);
        assert_eq!(bits(DType::BF16, f32::NEG_INFINITY), 0xff80);
        assert!(value(DType::BF16, bits(DType::BF16, f32::NAN)).is_nan());
        assert!(value(DType::BF16, bits(DType::BF16, f32::from_bits(0x7fff_ffff))).is_nan());
    }

    #[test]
    fn a_type_encodes_and_decodes_its_own_bytes() {
        let values = [1.0_f32, -0.5, 3.0];
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            let bytes = encode(dtype, &values).unwrap();
            assert_eq!(bytes.len() as u64, 3 * abi::bytes(dtype).unwrap());
            assert_eq!(decode(dtype, &bytes).unwrap(), values, "{dtype:?}");
            assert_eq!(one(dtype).unwrap(), encode(dtype, &[1.0]).unwrap());
        }
        assert_eq!(one(DType::BF16).unwrap(), [0x80, 0x3f]);
        assert_eq!(one(DType::F16).unwrap(), [0x00, 0x3c]);
        assert!(encode(DType::I32, &values).is_err());
        assert!(decode(DType::U8, &[0]).is_err());
        assert!(decode(DType::BF16, &[0; 3]).is_err());
    }
}
