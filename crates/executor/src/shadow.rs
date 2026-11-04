//! Shadow execution: run a changed operator's new kernel beside its old one and compare.
//!
//! Precopy checks a program-layer update on real work before switchover (specs/worker.md 7.3): for
//! every operator whose kernel program changed, one paired forward runs on the batch's stream
//! without a graph, where the old kernel writes the real outputs and the new one writes twins
//! (`Op::shadow`), and the two results are compared element by element within `Op::tolerance`.
//!
//! Two properties make the comparison meaningful, and this module is where they are kept:
//!
//! - **Errors do not accumulate.** Each paired forward writes the new kernel's result into the
//!   twin region only; the forward continues with the old kernel's real output, so every operator
//!   is compared on the same inputs its old kernel saw.
//! - **Resident memory is never written by the new kernel.** The twins come from the scratch pool
//!   (`ops::Twins`), the activations from the scratch arena, and the KV the paired forward writes
//!   goes to scratch pages. Nothing a serving request reads is touched.
//!
//! The comparison copies each pair to host page-locked staging and compares there; it runs in
//! precopy only, never on a serving path, so the copy cost is paid where it cannot delay a batch.

use std::sync::Arc;

use ops::{DType, TensorView, Tolerance, Twin, Twins};
use platform::{DeviceId, Issue as _, PinnedBuffer, Stream};

use crate::{Error, Result};

/// What one paired forward compared.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    /// How many twin buffers were compared.
    pub pairs: usize,
    /// How many elements were compared, over all pairs.
    pub elements: u64,
}

/// The paired-execution harness of one runner.
///
/// The region is allocated once per runner, sized for the largest shadow set of the buckets the
/// runner shadows, and reset for each paired forward; see `ops::Twins`.
pub struct Shadow {
    twins: Twins,
    staging: Staging,
}

impl Shadow {
    /// A harness over `twins`, who	staging lives on `device`.
    pub fn new(device: DeviceId, twins: Twins) -> Result<Self> {
        Ok(Self {
            twins,
            staging: Staging::new(device, 1 << 20)?,
        })
    }

    /// The shadow region, for the model to write twins into during the paired forward.
    pub fn twins(&mut self) -> &mut Twins {
        &mut self.twins
    }

    /// Prepares the region for one paired forward: no pairs from the previous one survive.
    ///
    /// A region that could not fit the previous paired forward keeps its failure; the caller reads
    /// [`Shadow::failure`] after the forward and fails the precopy there.
    pub fn reset(&mut self) {
        self.twins.reset();
    }

    /// The failure a carving recorded inside the paired forward, if one did.
    pub fn failure(&self) -> Option<&ops::Error> {
        self.twins.failure()
    }

    /// Compares every pair of the paired forward on `stream` within `tolerance`.
    ///
    /// The stream is the one the paired forward ran on; the comparison waits for it, which is
    /// correct in precopy and useless on a serving path.
    pub fn compare(&mut self, stream: &Stream, tolerance: Tolerance) -> Result<Summary> {
        if let Some(error) = self.twins.failure() {
            return Err(Error::Invariant(format!(
                "the shadow region could not hold the paired forward: {error}"
            )));
        }
        let pairs = self.twins.pairs();
        let mut summary = Summary {
            pairs: pairs.len(),
            elements: 0,
        };
        for pair in pairs {
            let bytes = pair.real.byte_len()?;
            if bytes != pair.shadow.byte_len()? {
                return Err(Error::Invariant(
                    "the twin and the real buffer have different sizes".into(),
                ));
            }
            let bytes = usize::try_from(bytes)
                .map_err(|_| Error::Invariant("a twin buffer is larger than memory".into()))?;
            self.staging.reserve(bytes)?;
            self.staging.read(stream, &pair.real, &pair.shadow)?;
            summary.elements += compare_pair(pair, self.staging.real(), self.staging.shadow(), tolerance)?;
        }
        self.twins.reset();
        Ok(summary)
    }
}

/// Compares one pair's host copies within `tolerance` and answers how many elements it compared.
fn compare_pair(pair: &Twin, real: &[u8], shadow: &[u8], tolerance: Tolerance) -> Result<u64> {
    let dtype = pair.real.dtype;
    let numel = pair.real.numel();
    let bytes_of = |dtype: DType| -> usize {
        dtype
            .bytes()
            .map(|bytes| bytes as usize)
            .unwrap_or_default()
    };
    let width = bytes_of(dtype);
    if width == 0 {
        return Err(Error::Invariant(format!(
            "shadow comparison of a sub-byte element type ({dtype:?}) is not supported"
        )));
    }
    let expected = usize::try_from(numel)
        .ok()
        .and_then(|count| count.checked_mul(width))
        .ok_or_else(|| Error::Invariant("a twin buffer's extent overflows".into()))?;
    if real.len() != expected || shadow.len() != expected {
        return Err(Error::Invariant(
            "a twin buffer's copy has an unexpected size".into(),
        ));
    }
    for index in 0..numel as usize {
        let offset = index * width;
        let (reference, candidate) = match dtype {
            DType::F32 => (
                f32::from_ne_bytes(real[offset..offset + 4].try_into().unwrap()) as f64,
                f32::from_ne_bytes(shadow[offset..offset + 4].try_into().unwrap()) as f64,
            ),
            DType::BF16 => (
                bf16_to_f32(&real[offset..offset + 2]) as f64,
                bf16_to_f32(&shadow[offset..offset + 2]) as f64,
            ),
            DType::F16 => (
                f16_to_f32(u16::from_ne_bytes(real[offset..offset + 2].try_into().unwrap())) as f64,
                f16_to_f32(u16::from_ne_bytes(
                    shadow[offset..offset + 2].try_into().unwrap(),
                )) as f64,
            ),
            DType::I32 => (
                i32::from_ne_bytes(real[offset..offset + 4].try_into().unwrap()) as f64,
                i32::from_ne_bytes(shadow[offset..offset + 4].try_into().unwrap()) as f64,
            ),
            DType::I64 => (
                i64::from_ne_bytes(real[offset..offset + 8].try_into().unwrap()) as f64,
                i64::from_ne_bytes(shadow[offset..offset + 8].try_into().unwrap()) as f64,
            ),
            DType::U8 => (real[offset] as f64, shadow[offset] as f64),
            other => {
                return Err(Error::Invariant(format!(
                    "shadow comparison of {other:?} is not supported"
                )));
            }
        };
        let (reference, candidate) = (reference as f32, candidate as f32);
        // A NaN in one buffer and a number in the other is a difference; two NaNs agree.
        let agree = if reference.is_nan() && candidate.is_nan() {
            true
        } else {
            tolerance.allows(reference, candidate)
        };
        if !agree {
            return Err(Error::Invariant(format!(
                "element {index} differs beyond tolerance: reference {reference}, candidate \
                 {candidate}, tolerance atol {} rtol {}",
                tolerance.atol, tolerance.rtol
            )));
        }
    }
    Ok(numel)
}

fn bf16_to_f32(bytes: &[u8]) -> f32 {
    f32::from_bits((u16::from_ne_bytes([bytes[0], bytes[1]]) as u32) << 16)
}

fn f16_to_f32(bits: u16) -> f32 {
    f32::from_bits(halves_to_f32_bits(bits))
}

/// IEEE 754 half to single, including subnormals, infinities and NaN.
fn halves_to_f32_bits(bits: u16) -> u32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = u32::from(bits & 0x03ff);
    match exponent {
        0 => {
            if mantissa == 0 {
                return sign;
            }
            // Subnormal: normalize into a single-precision value.
            let mut exponent = 127 - 15 + 1;
            let mut mantissa = mantissa;
            while mantissa & 0x0400 == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            let mantissa = (mantissa & 0x03ff) << 13;
            sign | ((exponent as u32) << 23) | mantissa
        }
        0x1f => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((u32::from(exponent) + 127 - 15) << 23) | (mantissa << 13),
    }
}

/// Host staging for the comparison: two page-locked buffers, grown as pairs need.
struct Staging {
    device: DeviceId,
    real: PinnedBuffer,
    shadow: PinnedBuffer,
}

impl Staging {
    fn new(device: DeviceId, bytes: usize) -> Result<Self> {
        Ok(Self {
            device,
            real: PinnedBuffer::new(device, bytes)?,
            shadow: PinnedBuffer::new(device, bytes)?,
        })
    }

    fn reserve(&mut self, bytes: usize) -> Result<()> {
        if self.real.len() >= bytes {
            return Ok(());
        }
        self.real = PinnedBuffer::new(self.device, bytes)?;
        self.shadow = PinnedBuffer::new(self.device, bytes)?;
        Ok(())
    }

    /// Copies `real` and `shadow` to the host and waits for both copies.
    fn read(&mut self, stream: &Stream, real: &TensorView, shadow: &TensorView) -> Result<()> {
        let bytes = real.byte_len()? as usize;
        // SAFETY: both host destinations are page-locked storage of `bytes`, and each device
        // source view owns `bytes` of the paired forward's buffers, which the stream's work the
        // caller waited for no longer touches.
        unsafe {
            platform::memcpy_async(stream, self.real.address(), real.ptr.address(), bytes)?;
            platform::memcpy_async(stream, self.shadow.address(), shadow.ptr.address(), bytes)?;
        }
        stream.synchronize()?;
        Ok(())
    }

    fn real(&self) -> &[u8] {
        // SAFETY: the buffer is page-locked storage of `len` bytes written by the copy above.
        unsafe { std::slice::from_raw_parts(self.real.as_ptr(), self.real.len()) }
    }

    fn shadow(&self) -> &[u8] {
        // SAFETY: as for `real`.
        unsafe { std::slice::from_raw_parts(self.shadow.as_ptr(), self.shadow.len()) }
    }
}

/// The tolerance of the operator being shadowed, from its descriptor.
pub fn tolerance_of<O: ops::Op>(descriptor: &O::Descriptor) -> Tolerance {
    O::tolerance(descriptor)
}

/// The harness's twin region, for a model that writes through it.
pub type Region = Arc<Twins>;
