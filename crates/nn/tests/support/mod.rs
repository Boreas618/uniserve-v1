//! What the device cases of nn's integration tests share: fixed-seed inputs, the element codecs
//! of the float types, and a bench that runs a layer's forward on a device with kernels a Build
//! Server supplies.
//!
//! Each test file that declares `mod support;` compiles its own copy and uses part of it.
#![allow(dead_code, reason = "each test crate uses part of the shared support")]

use nn::DType;
use ops::values::{f16_to_f32, f32_to_bf16, f32_to_f16};

pub(crate) mod device;

/// Every harness failure: a missing asset, a refused launch, a device or Build Server error.
pub(crate) type Outcome<T = ()> = Result<T, Box<dyn std::error::Error>>;

// ---- Fixed-seed inputs ----------------------------------------------------------------------

/// A SplitMix64 stream: the same seed draws the same inputs on every host.
pub(crate) struct Draw(u64);

impl Draw {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut mixed = self.0;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        mixed ^ (mixed >> 31)
    }

    /// A uniform value in `(0, 1]`, from the top 53 bits.
    pub(crate) fn unit(&mut self) -> f64 {
        ((self.next() >> 11) + 1) as f64 / (1_u64 << 53) as f64
    }

    /// `count` values uniform in `[low, high)`.
    pub(crate) fn uniform(&mut self, count: usize, low: f32, high: f32) -> Vec<f32> {
        (0..count)
            .map(|_| low + (high - low) * (1.0 - self.unit()) as f32)
            .collect()
    }

    /// One standard normal value (Box-Muller, one of each pair).
    pub(crate) fn normal(&mut self) -> f64 {
        let radius = (-2.0 * self.unit().ln()).sqrt();
        radius * (std::f64::consts::TAU * self.unit()).cos()
    }

    /// `count` values of `N(0, std²)`.
    pub(crate) fn normals(&mut self, count: usize, std: f32) -> Vec<f32> {
        (0..count).map(|_| (self.normal() as f32) * std).collect()
    }
}

// ---- Element bytes ----------------------------------------------------------------------------

/// `values` rounded to `dtype` (round to nearest even), as little-endian bytes.
pub(crate) fn encode(dtype: DType, values: &[f32]) -> Outcome<Vec<u8>> {
    Ok(match dtype {
        DType::F32 => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        DType::F16 => values
            .iter()
            .flat_map(|value| f32_to_f16(*value).to_le_bytes())
            .collect(),
        DType::BF16 => values
            .iter()
            .flat_map(|value| f32_to_bf16(*value).to_le_bytes())
            .collect(),
        other => return Err(format!("no float encoding for {other}").into()),
    })
}

/// The values of little-endian `bytes` of `dtype`, widened to `f32` exactly.
pub(crate) fn decode(dtype: DType, bytes: &[u8]) -> Outcome<Vec<f32>> {
    Ok(match dtype {
        DType::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect(),
        DType::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|half| f16_to_f32(u16::from_le_bytes(*half)))
            .collect(),
        DType::BF16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|half| f32::from_bits(u32::from(u16::from_le_bytes(*half)) << 16))
            .collect(),
        other => return Err(format!("no float decoding for {other}").into()),
    })
}
