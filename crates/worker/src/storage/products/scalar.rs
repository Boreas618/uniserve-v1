//! Scalar products and their writes.

use super::*;
use crate::ffi::{DType, Device};
use crate::storage::memory::StorageView;
use crate::stream::{Fill, Issue};
use crate::{Error, Result};
use std::collections::HashSet;
use std::sync::Arc;
use uniserve_cuda::{CopyBatch, SourceOrder};

/// A scalar written into a one-element product.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scalar {
    Int(i64),
    Float(f64),
}

/// A one-element value's bits in `dtype` and their width in bytes.
pub(super) fn scalar_bits(value: Scalar, dtype: DType) -> Result<(u64, usize)> {
    let float = match value {
        Scalar::Int(value) => value as f64,
        Scalar::Float(value) => value,
    };
    let integer = match value {
        Scalar::Int(value) => value,
        // Float fills of integer storage truncate toward zero.
        Scalar::Float(value) => value as i64,
    };
    Ok(match dtype {
        DType::Bool => (u64::from(float != 0.0), 1),
        DType::U8 => (u64::from(integer as u8), 1),
        DType::I8 => (u64::from(integer as i8 as u8), 1),
        DType::I16 => (u64::from(integer as i16 as u16), 2),
        DType::I32 => (u64::from(integer as i32 as u32), 4),
        DType::U32 => (u64::from(integer as u32), 4),
        DType::I64 => (integer as u64, 8),
        DType::F16 => (u64::from(f16_bits(float as f32)), 2),
        DType::BF16 => (u64::from(bf16_bits(float as f32)), 2),
        DType::F32 => (u64::from((float as f32).to_bits()), 4),
        DType::F64 => (float.to_bits(), 8),
        DType::F8E4M3 | DType::F8E5M2 => {
            return Err(Error::Unsupported(
                "scalar products are not stored as FP8".into(),
            ));
        }
    })
}

/// IEEE binary16 bits of `value`, rounded to nearest even.
pub(super) fn f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00 | if mantissa != 0 { 0x200 } else { 0 };
    }
    let biased = exponent - 127 + 15;
    if biased >= 0x1f {
        return sign | 0x7c00;
    }
    if biased <= 0 {
        if biased < -10 {
            return sign;
        }
        // Subnormal: the value in units of 2^-24, rounded to nearest even.
        let significand = mantissa | 0x80_0000;
        let shift = (14 - biased) as u32;
        let truncated = significand >> shift;
        let remainder = significand & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        let rounded = truncated
            + u32::from(remainder > halfway || (remainder == halfway && truncated & 1 == 1));
        return sign | rounded as u16;
    }
    let truncated = ((biased as u32) << 10) | (mantissa >> 13);
    let remainder = mantissa & 0x1fff;
    // A carry out of the mantissa correctly increments the exponent, up to
    // infinity.
    let rounded =
        truncated + u32::from(remainder > 0x1000 || (remainder == 0x1000 && truncated & 1 == 1));
    sign | rounded as u16
}

/// bfloat16 bits of `value`, rounded to nearest even.
pub(super) fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return ((bits >> 16) | 0x40) as u16;
    }
    let rounding = 0x7fff + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding) >> 16) as u16
}

impl ProductStore {
    /// Copy one scalar per product from a contiguous source column of the
    /// products' dtype, in one batched copy on the producer `stream`.
    pub fn write_scalars(
        &self,
        writes: &[ProductBuffer],
        values: &StorageView,
        stream: impl Issue,
    ) -> Result<Vec<StorageView>> {
        let state = lock(&self.state);
        let mut targets = Vec::with_capacity(writes.len());
        let mut seen = HashSet::new();
        for write in writes {
            state.require_buffer(write)?;
            if lock(write).state.produced() || !seen.insert(Arc::as_ptr(&write.0) as usize) {
                return Err(Error::Invariant(
                    "device product was published more than once".into(),
                ));
            }
            targets.push(write.view());
        }
        let Some(first) = targets.first() else {
            return Ok(targets);
        };
        if values.numel() != targets.len() as u64 || !values.is_contiguous() {
            return Err(Error::Invalid(
                "batched device-product export requires one contiguous scalar per output".into(),
            ));
        }
        if targets.iter().any(|target| {
            target.numel() != 1
                || target.device() != first.device()
                || target.dtype() != first.dtype()
        }) || values.dtype() != first.dtype()
        {
            return Err(Error::Invalid(
                "batched device-product export spans incompatible storage".into(),
            ));
        }

        let width = first.dtype().bytes();
        let mut batch = CopyBatch::new(SourceOrder::Stream);
        for (index, target) in targets.iter().enumerate() {
            batch.push(
                target.address(),
                values.address() + (index * width) as u64,
                width,
            );
        }
        match first.device() {
            Device::Cuda(_) => {
                if stream.stream().is_none() {
                    return Err(Error::Invalid(
                        "a device product write requires its producer stream".into(),
                    ));
                }
                // SAFETY: each destination is one reserved element; the
                // producer keeps its column until its stream passes the copy.
                unsafe { stream.copy(&batch) }?;
            }
            // SAFETY: host elements of reserved products, copied in the
            // producer's order.
            _ => unsafe { stream.host().copy(&batch) }?,
        }

        let fence = self.producer_fence(first.device(), stream, writes.len())?;
        for write in writes {
            lock(write).produced(vec![1], 1, fence.clone());
        }
        Ok(targets)
    }

    /// Fill a one-element product with `value` (its bit pattern in the
    /// product's dtype) on the producer `stream`.
    pub fn write_scalar(
        &self,
        buffer: &ProductBuffer,
        value: Scalar,
        stream: impl Issue,
    ) -> Result<StorageView> {
        let state = lock(&self.state);
        state.require_buffer(buffer)?;
        if lock(buffer).state.produced() {
            return Err(Error::Invariant(
                "device product was published more than once".into(),
            ));
        }
        let view = value_view(&buffer.view(), &[1], 1)?;
        let (bits, width) = scalar_bits(value, view.dtype())?;
        let address = view.address();
        // An eight-byte element is two words, low word first.
        let words = [(address, bits as u32), (address + 4, (bits >> 32) as u32)];
        let fills = words[..width.div_ceil(4)]
            .iter()
            .map(|&(address, value)| Fill {
                address,
                value,
                width: width.min(4),
                count: 1,
            });
        match view.device() {
            Device::Cuda(_) => {
                if stream.stream().is_none() {
                    return Err(Error::Invalid(
                        "a device product write requires its producer stream".into(),
                    ));
                }
                for fill in fills {
                    // SAFETY: one reserved element of `width` bytes on the
                    // stream's device.
                    unsafe { stream.fill(fill) }?;
                }
            }
            _ => {
                for fill in fills {
                    // SAFETY: one reserved host element of `width` bytes.
                    unsafe { stream.host().fill(fill) }?;
                }
            }
        }
        let fence = self.producer_fence(view.device(), stream, 1)?;
        lock(buffer).produced(vec![1], 1, fence);
        Ok(view)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_precision_bits_round_to_nearest_even() {
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(-2.5), 0xc100);
        assert_eq!(f16_bits(65504.0), 0x7bff);
        assert_eq!(f16_bits(1.0e6), 0x7c00);
        assert_eq!(f16_bits(5.960_464_5e-8), 0x0001);
        assert_eq!(f16_bits(1.0 + 1.0 / 2048.0), 0x3c00);
        assert_eq!(bf16_bits(1.0), 0x3f80);
        assert_eq!(bf16_bits(-3.0), 0xc040);
        assert_eq!(bf16_bits(1.0 + 1.0 / 256.0), 0x3f80);
    }

    #[test]
    fn scalar_bits_follow_the_storage_dtype() -> Result<()> {
        assert_eq!(scalar_bits(Scalar::Int(-1), DType::I32)?, (0xffff_ffff, 4));
        assert_eq!(scalar_bits(Scalar::Float(2.9), DType::I64)?, (2, 8));
        assert_eq!(scalar_bits(Scalar::Int(1), DType::F32)?, (0x3f80_0000, 4));
        assert!(scalar_bits(Scalar::Int(1), DType::F8E4M3).is_err());
        Ok(())
    }
}
