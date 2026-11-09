//! Host values in a tensor's element type: the one place a number becomes the bytes of a `DType`
//! and back.
//!
//! Every host-side producer of device bytes — a constant table, a loaded weight's conversion, a
//! media sample — encodes through [`DType::encode`] and reads through [`DType::decode`], and the
//! scalar converters below are what both use. They follow the device's own conversion so a value
//! narrowed on the host and one narrowed on the card agree bit for bit:
//!
//! - **BF16** keeps an `f32`'s top sixteen bits, rounded to nearest with ties to even. A NaN stays a
//!   quiet NaN with its sign and the payload's top bits (the rounding add would otherwise carry a
//!   large payload into an infinity), and an infinity stays one.
//! - **F16** rounds to nearest with ties to even, through the subnormal range (a whole number of
//!   `2^-24` units), and overflows to a signed infinity past the largest finite half.
//!
//! The bytes are little-endian, the order of every device this tree serves and of a safetensors
//! payload.

use crate::error::{Error, Result};
use crate::tensor::DType;

/// The bfloat16 bit pattern nearest to `value`, ties to even.
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        // Quiet, with the payload's top six bits: an `f32` NaN is any exponent of all ones with a
        // non-zero mantissa, and the truncation keeps both.
        return ((bits >> 16) as u16) | 0x0040;
    }
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

/// The `f32` value of a bfloat16 bit pattern: exact, since every bfloat16 is an `f32`.
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// `value` divided by `2^shift`, rounded to nearest with ties to even.
fn round_even(value: u32, shift: u32) -> u32 {
    if shift == 0 {
        return value;
    }
    let quotient = value >> shift;
    let remainder = value & ((1 << shift) - 1);
    let half = 1 << (shift - 1);
    if remainder > half || (remainder == half && quotient & 1 == 1) {
        quotient + 1
    } else {
        quotient
    }
}

/// The IEEE half-precision bit pattern nearest to `value`, ties to even.
pub fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = (bits >> 23) & 0xff;
    let mantissa = bits & 0x007f_ffff;
    if exponent == 0 {
        // A zero or subnormal `f32` underflows to a signed zero in the half's exponent range; the
        // largest half subnormal is still an order of magnitude above the largest `f32` subnormal.
        return sign;
    }
    if exponent == 0xff {
        return sign | 0x7c00 | if mantissa == 0 { 0 } else { 0x0200 };
    }
    let unbiased = exponent as i32 - 127;
    if unbiased > 15 {
        return sign | 0x7c00;
    }
    if unbiased >= -14 {
        // A normal half: drop thirteen mantissa bits, round to nearest with ties to even, and let
        // the rounding carry into the exponent — a carry out of the largest finite half is the
        // infinity that nearest-rounding gives.
        let half_exponent = ((unbiased + 15) as u32) << 10;
        let total = half_exponent + round_even(mantissa, 13);
        return if total >= 0x7c00 {
            sign | 0x7c00
        } else {
            sign | total as u16
        };
    }
    // A subnormal half is a whole number of `2^-24` units: the significand at the float's own scale
    // (`1.m · 2^23`) divided by `2^(-unbiased - 1)`, rounded to nearest with ties to even.
    let significand = mantissa | 0x0080_0000;
    let shift = (-unbiased - 1) as u32;
    if shift > 24 {
        return sign;
    }
    let rounded = round_even(significand, shift);
    if rounded >= 0x0400 {
        // Rounded up out of the subnormals into the smallest normal.
        return sign | 0x0400;
    }
    sign | rounded as u16
}

/// The `f32` value of an IEEE half-precision bit pattern: exact, subnormals included.
pub fn f16_to_f32(half: u16) -> f32 {
    let sign = u32::from(half & 0x8000) << 16;
    let exponent = (half >> 10) & 0x1f;
    let mantissa = u32::from(half & 0x03ff);
    let bits = match exponent {
        0 => {
            if mantissa == 0 {
                sign
            } else {
                // A subnormal half is `mantissa · 2^-24`. Normalizing it left until the leading bit
                // reaches the half's implicit position gives the float's exponent and fraction.
                let leading = 31 - mantissa.leading_zeros();
                let shift = 10 - leading;
                let fraction = (mantissa << shift) & 0x03ff;
                let exponent = 127 - 15 + 1 - shift as i32;
                sign | ((exponent as u32) << 23) | (fraction << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((u32::from(exponent) + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(bits)
}

impl DType {
    /// `values` as the little-endian bytes of this floating-point type, each rounded to nearest
    /// with ties to even.
    ///
    /// # Errors
    ///
    /// For a type other than F32, BF16 or F16: a packed or integer type has no rounding from a real
    /// number that a host producer may choose for it.
    pub fn encode(self, values: &[f32]) -> Result<Vec<u8>> {
        let mut bytes = Vec::with_capacity(values.len() * 4);
        match self {
            Self::F32 => values
                .iter()
                .for_each(|value| bytes.extend_from_slice(&value.to_le_bytes())),
            Self::BF16 => values
                .iter()
                .for_each(|value| bytes.extend_from_slice(&f32_to_bf16(*value).to_le_bytes())),
            Self::F16 => values
                .iter()
                .for_each(|value| bytes.extend_from_slice(&f32_to_f16(*value).to_le_bytes())),
            other => {
                return Err(Error::Shape(format!(
                    "real numbers are encoded as f32, bf16 or f16, not {other}"
                )));
            }
        }
        Ok(bytes)
    }

    /// The real values the little-endian bytes of this floating-point type hold, exactly.
    ///
    /// # Errors
    ///
    /// For a type other than F32, BF16 or F16, and for a byte count that is not whole elements.
    pub fn decode(self, bytes: &[u8]) -> Result<Vec<f32>> {
        let width = match self {
            Self::F32 => 4,
            Self::BF16 | Self::F16 => 2,
            other => {
                return Err(Error::Shape(format!(
                    "real numbers are decoded from f32, bf16 or f16, not {other}"
                )));
            }
        };
        if !bytes.len().is_multiple_of(width) {
            return Err(Error::Shape(format!(
                "{} bytes are not whole {self} elements",
                bytes.len()
            )));
        }
        Ok(bytes
            .chunks_exact(width)
            .map(|chunk| match self {
                Self::F32 => f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
                Self::BF16 => bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])),
                _ => f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])),
            })
            .collect())
    }

    /// The bytes of one element of value one: the coefficient a fused operator reads where a
    /// scale of one is meant.
    ///
    /// # Errors
    ///
    /// As [`Self::encode`].
    pub fn one(self) -> Result<Vec<u8>> {
        self.encode(&[1.0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_half_round_trips_the_values_a_checkpoint_holds() {
        // A subnormal half has an absolute precision of 2^-24, not a relative one, so the bound is
        // half a unit there and relative only from the smallest normal upward.
        const SMALLEST_NORMAL: f32 = 6.103_515_6e-5;
        const HALF_UNIT: f32 = 2.980_232_2e-8;
        for value in [0.0_f32, 1.0, -1.0, 0.5, 0.1, 65_504.0, 1e-5, 1.192_092_9e-7] {
            let back = f16_to_f32(f32_to_f16(value));
            let bound = if value.abs() < SMALLEST_NORMAL {
                HALF_UNIT
            } else {
                value.abs() * 1e-3
            };
            assert!((back - value).abs() <= bound, "{value:e} became {back:e}");
        }
    }

    #[test]
    fn a_half_holds_the_subnormal_range_and_the_specials() {
        // A quarter of the half's magnitude range is subnormal, and a checkpoint holds RMS weights
        // and epsilon-like constants there; the encoding is a whole number of 2^-24 units.
        for (value, units) in [
            (1e-5_f32, 168_u16),
            (5.960_464_5e-8, 1),
            (1.192_092_9e-7, 2),
            (6.097_555_6e-5, 1023),
        ] {
            assert_eq!(f32_to_f16(value), units, "{value:e}");
            assert!(
                (f16_to_f32(units) - value).abs() <= 5.960_464_5e-8,
                "{value:e}"
            );
        }
        // Below half a unit of the smallest subnormal the value is zero; exactly half a unit ties
        // between 0 and 1, where even wins.
        assert_eq!(f32_to_f16(2.0e-8), 0);
        assert_eq!(f32_to_f16(-2.0e-8), 0x8000);
        assert_eq!(f32_to_f16(2.980_232_2e-8), 0);
        assert_eq!(f32_to_f16(4.5e-8), 1);
        // A value that rounds up out of the subnormals becomes the smallest normal.
        assert_eq!(f32_to_f16(6.101e-5), 0x0400);
        // The largest finite magnitude survives; past it a value overflows to infinity.
        assert_eq!(f16_to_f32(f32_to_f16(65504.0)), 65504.0);
        assert_eq!(f32_to_f16(65_520.0), 0x7c00);
        assert_eq!(f32_to_f16(70_000.0), 0x7c00);
        assert_eq!(f32_to_f16(-70_000.0), 0xfc00);
        assert!(f32_to_f16(f32::NAN) & 0x7c00 == 0x7c00);
        assert!(f32_to_f16(f32::NAN) & 0x03ff != 0);
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
        assert_eq!(f16_to_f32(0x8000), -0.0);
    }

    #[test]
    fn a_bf16_rounds_like_the_hardware() {
        // A value whose low half is zero keeps exactly its top sixteen bits.
        let tiny = f32::from_bits(0x0c00_0000);
        let large = f32::from_bits(0x7f7f_0000);
        for value in [0.0_f32, 1.0, -1.0, 2.0, 0.5, tiny, large] {
            assert_eq!(
                f32_to_bf16(value),
                (value.to_bits() >> 16) as u16,
                "{value:e}"
            );
            assert_eq!(bf16_to_f32(f32_to_bf16(value)), value);
        }
        // Half a unit in the last place ties between two neighbours and the even one wins; a value
        // past the tie rounds up.
        assert_eq!(f32_to_bf16(1.0 + 2.0_f32.powi(-8)), 0x3f80);
        assert_eq!(f32_to_bf16(1.0 + 3.0 * 2.0_f32.powi(-9)), 0x3f81);
        // The largest finite value's neighbourhood rounds to infinity, and the specials keep their
        // meaning: a NaN stays a NaN instead of carrying into that infinity.
        assert_eq!(f32_to_bf16(f32::from_bits(0x7f7f_ffff)), 0x7f80);
        assert_eq!(f32_to_bf16(f32::INFINITY), 0x7f80);
        assert_eq!(f32_to_bf16(f32::NEG_INFINITY), 0xff80);
        assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
        assert!(bf16_to_f32(f32_to_bf16(f32::from_bits(0x7fff_ffff))).is_nan());
    }

    #[test]
    fn a_type_encodes_and_decodes_its_own_bytes() {
        let values = [1.0_f32, -0.5, 3.0];
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            let bytes = dtype.encode(&values).unwrap();
            assert_eq!(bytes.len() as u64, 3 * dtype.bytes().unwrap());
            assert_eq!(dtype.decode(&bytes).unwrap(), values, "{dtype}");
            assert_eq!(dtype.one().unwrap(), dtype.encode(&[1.0]).unwrap());
        }
        assert_eq!(DType::BF16.one().unwrap(), [0x80, 0x3f]);
        assert_eq!(DType::F16.one().unwrap(), [0x00, 0x3c]);
        assert!(DType::I32.encode(&values).is_err());
        assert!(DType::U8.decode(&[0]).is_err());
        assert!(DType::BF16.decode(&[0; 3]).is_err());
    }
}
