//! Shadow: a changed operator's new kernel beside the one it replaces, on one paired forward.
//!
//! Before a generation is switched to, each operator whose kernel changed runs one paired
//! forward: an eager forward of a bucket under the generation being prepared whose route
//! ([`Pairing`]) launches every call of that operator twice over the same inputs. The new kernel
//! writes shadow outputs (`ops::Op::shadow`) and the serving kernel then writes the real
//! outputs, which the forward continues on. Two properties make the comparison meaningful:
//!
//! - the difference of one operator does not accumulate through the next, because nothing reads
//!   a shadow output but the comparison;
//! - each real output is copied beside its shadow output the moment it is written
//!   (`ops::Twins::freeze`), because a later operator of the forward writes over the block it
//!   was allocated in.
//!
//! After the forward the pairs are copied to page-locked host memory behind the forward on its
//! stream, and compared element by element in the element type's own values, each pair within
//! the tolerance of the call that produced it (`ops::Op::tolerance`).

use std::sync::{Arc, Mutex};

use half::{bf16, f16};
use ops::{Generation, Tolerance, Twins};
use platform::{DeviceId, Event, EventFlags, PinnedBuffer, Stream};
use tensor::DType;

use crate::worker::{Error, Result};

use service::lock;

/// The route of one paired forward: the calls of one operator, launched with the kernels of two
/// generations.
pub(crate) struct Pairing {
    /// The operator whose kernels are compared, by `ops::Op::NAME`.
    operator: &'static str,
    /// The generation that serves: its kernel writes the real outputs.
    serving: Arc<Generation>,
    /// The region the new kernel's shadow outputs are written in.
    outputs: Arc<Mutex<Twins>>,
    /// Per paired call, the number of pairs the region held after it and the bound its outputs
    /// are compared within. A call's descriptor decides its tolerance, and one operator is
    /// launched under several.
    calls: Mutex<Vec<(usize, Tolerance)>>,
}

impl ops::Route for Pairing {
    fn launch(
        &self,
        call: &dyn ops::Call,
        current: &Generation,
        stream: &Stream,
    ) -> ops::Result<bool> {
        if call.operator() != self.operator {
            return Ok(false);
        }

        // A call is paired when both generations hold a kernel for its key and the two differ:
        // a kernel is compared with the one it replaces, never with another compilation.
        let key = call.key();
        let (Ok(serving), Ok(new)) = (
            self.serving.kernel(self.operator, &key),
            current.kernel(self.operator, &key),
        ) else {
            return Ok(false);
        };
        if Arc::ptr_eq(serving, new) {
            return Ok(false);
        }

        // The new kernel writes the shadow outputs; the serving kernel then writes the real
        // ones, which are copied beside them before anything writes over their block.
        let mut outputs = lock(&self.outputs);
        call.launch_redirected(new, &mut outputs, stream)?;
        call.launch(serving, stream)?;
        outputs.freeze(stream);

        lock(&self.calls).push((outputs.pairs().len(), call.tolerance()));
        Ok(true)
    }
}

/// What a runner's graphs compare paired forwards with: the region shadow outputs are written
/// in, and the page-locked memory the pairs are copied to.
pub(crate) struct Shadow {
    outputs: Arc<Mutex<Twins>>,
    device: DeviceId,
    /// The host copies of the real and of the shadow outputs, back to back per pair; grown to
    /// the largest paired forward compared so far.
    host: Option<(PinnedBuffer, PinnedBuffer)>,
}

impl Shadow {
    /// A shadow over `outputs`, device memory of the scratch pool of `device`.
    pub(crate) fn new(device: DeviceId, outputs: Twins) -> Self {
        Self {
            outputs: Arc::new(Mutex::new(outputs)),
            device,
            host: None,
        }
    }

    /// The route of one paired forward of `operator` against the kernels `serving` holds, with
    /// the region emptied for it.
    pub(crate) fn pairing(&self, operator: &'static str, serving: Arc<Generation>) -> Arc<Pairing> {
        lock(&self.outputs).reset();

        Arc::new(Pairing {
            operator,
            serving,
            outputs: Arc::clone(&self.outputs),
            calls: Mutex::new(Vec::new()),
        })
    }

    /// Compares the pairs of the paired forward `pairing` routed on `stream`, and returns the
    /// elements compared. A pair whose two outputs have other sizes cannot be compared: it is
    /// named in `uncompared` and skipped. A forward whose operator's outputs exceed the region
    /// compares nothing: the operator is named in `uncompared`, and the batch has no evidence
    /// for it. Its calls whose shadow outputs had no room launched the new kernel over the
    /// forward's own outputs, which the serving kernel then wrote again (`ops::Twins`), so the
    /// forward still wrote nothing a serving request reads.
    ///
    /// The copies to the host are issued behind the forward and waited for on their own event.
    ///
    /// # Errors
    ///
    /// When a call's outputs could not be shadowed for another reason than the region's size,
    /// when a pair differs beyond its call's tolerance (the error names how many elements, the
    /// widest and the bound), or when the driver refuses a copy or the wait.
    pub(crate) fn compare(
        &mut self,
        pairing: &Pairing,
        stream: &Stream,
        uncompared: &mut Vec<String>,
    ) -> Result<u64> {
        let calls = std::mem::take(&mut *lock(&pairing.calls));
        let outputs = lock(&self.outputs);
        match outputs.failure() {
            Some(failure @ ops::Error::Twins { .. }) => {
                uncompared.push(format!(
                    "`{}`: its outputs in this batch exceed the shadow region of {} bytes: \
                     {failure}",
                    pairing.operator,
                    outputs.capacity()
                ));
                return Ok(0);
            }
            Some(failure) => {
                return Err(Error::Invariant(format!(
                    "the paired forward of `{}` could not shadow a call's outputs: {failure}",
                    pairing.operator
                )));
            }
            None => {}
        }

        // Each comparable pair: its element type, its element count, its offset in the host
        // copies, its bytes and the bound of its call.
        let mut pairs = Vec::new();
        let mut total = 0_usize;
        for (index, pair) in outputs.pairs().iter().enumerate() {
            let bytes = pair.real.byte_len()?;
            if bytes != pair.shadow.byte_len()? {
                uncompared.push(format!(
                    "`{}`: the real output holds {bytes} bytes of {} and the shadow output {} \
                     of {}",
                    pairing.operator,
                    ops::abi::name(pair.real.dtype),
                    pair.shadow.byte_len()?,
                    ops::abi::name(pair.shadow.dtype),
                ));
                continue;
            }

            let tolerance = calls
                .iter()
                .find(|(end, _)| index < *end)
                .map(|(_, tolerance)| *tolerance)
                .ok_or_else(|| {
                    Error::Invariant("a shadow output belongs to no paired call".into())
                })?;
            pairs.push((pair, total, bytes as usize, tolerance));
            total += bytes as usize;
        }

        let held = self.host.as_ref().map_or(0, |(real, _)| real.len());
        if held < total {
            self.host = Some((
                PinnedBuffer::new(self.device, total)?,
                PinnedBuffer::new(self.device, total)?,
            ));
        }
        let Some((real, shadow)) = &self.host else {
            return Ok(0);
        };

        for (pair, offset, bytes, _) in &pairs {
            // SAFETY: both host buffers hold `total` page-locked bytes, of which
            // `offset..offset + bytes` is this pair's alone; each source is `bytes` bytes of
            // the shadow region, which nothing writes before the next paired forward, and the
            // copies are ordered behind the forward on its stream.
            unsafe {
                platform::memcpy_async(
                    real.address() + *offset as u64,
                    pair.real.ptr.address(),
                    *bytes,
                    stream,
                )?;
                platform::memcpy_async(
                    shadow.address() + *offset as u64,
                    pair.shadow.ptr.address(),
                    *bytes,
                    stream,
                )?;
            }
        }

        let copied = Event::new(EventFlags::default());
        copied.record(stream)?;
        copied.synchronize()?;

        // SAFETY: the buffers are this shadow's own page-locked memory, read once the copies
        // into them have completed.
        let (real, shadow) = unsafe { (real.as_slice(), shadow.as_slice()) };
        let mut elements = 0;
        for (pair, offset, bytes, tolerance) in pairs {
            elements += compare_elements(
                pair.real.dtype,
                pair.real.numel(),
                &real[offset..offset + bytes],
                &shadow[offset..offset + bytes],
                tolerance,
            )
            .map_err(|error| {
                Error::Invariant(format!("shadow of `{}`: {error}", pairing.operator))
            })?;
        }
        Ok(elements)
    }
}

/// Compares the host copies of one pair within `tolerance` and returns the elements compared.
///
/// `real` and `shadow` are the bytes of the serving kernel's output and of the new kernel's
/// shadow output, `elements` elements of `dtype` each. Elements are compared as the numbers
/// they denote; two NaNs agree and a NaN against a number does not.
///
/// # Errors
///
/// When the copies do not hold `elements` elements, the element type has no comparison, or
/// elements differ beyond the bound: the error states how many, the widest one's reference,
/// candidate and difference, and the bound.
pub(crate) fn compare_elements(
    dtype: DType,
    elements: u64,
    real: &[u8],
    shadow: &[u8],
    tolerance: Tolerance,
) -> Result<u64> {
    let width = ops::abi::bytes(dtype).unwrap_or(0) as usize;
    if width == 0 {
        return Err(Error::Invariant(format!(
            "{dtype:?} elements are narrower than a byte and are not compared"
        )));
    }

    let expected = usize::try_from(elements)
        .ok()
        .and_then(|count| count.checked_mul(width));
    if expected != Some(real.len()) || expected != Some(shadow.len()) {
        return Err(Error::Invariant(format!(
            "the copies of a pair of {elements} {dtype:?} elements hold {} and {} bytes",
            real.len(),
            shadow.len()
        )));
    }

    let read = |bytes: &[u8], index: usize| -> Result<f32> {
        let at = index * width;
        let value = match dtype {
            DType::F32 => f32::from_ne_bytes(element(bytes, at)?),
            DType::BF16 => bf16::from_ne_bytes(element(bytes, at)?).to_f32(),
            DType::F16 => f16::from_ne_bytes(element(bytes, at)?).to_f32(),
            DType::I32 => i32::from_ne_bytes(element(bytes, at)?) as f32,
            DType::U32 => u32::from_ne_bytes(element(bytes, at)?) as f32,
            DType::I64 => i64::from_ne_bytes(element(bytes, at)?) as f32,
            DType::U8 => f32::from(u8::from_ne_bytes(element(bytes, at)?)),
            other => {
                return Err(Error::Invariant(format!(
                    "{other:?} elements have no shadow comparison"
                )));
            }
        };
        Ok(value)
    };

    // A bound that cancels is relative to the output's largest magnitude, so that comes first.
    let mut peak = 0.0_f32;
    for index in 0..elements as usize {
        let reference = read(real, index)?;
        if reference.is_finite() {
            peak = peak.max(reference.abs());
        }
    }

    // Every element is compared, not only up to the first difference: how many differ and by
    // how much is what tells a wrong kernel from a rounding.
    let mut differing = 0_u64;
    let mut widest: Option<(usize, f32, f32, f32)> = None;
    for index in 0..elements as usize {
        let (reference, candidate) = (read(real, index)?, read(shadow, index)?);

        // Two NaNs agree, and so do two infinities of one sign, whose difference no bound holds.
        let agree = (reference.is_nan() && candidate.is_nan())
            || reference == candidate
            || tolerance.allows_within(peak, reference, candidate);
        if agree {
            continue;
        }

        differing += 1;
        let difference = (reference - candidate).abs();
        if widest.is_none_or(|(_, _, _, held)| difference.total_cmp(&held).is_gt()) {
            widest = Some((index, reference, candidate, difference));
        }
    }

    match widest {
        Some((index, reference, candidate, difference)) => Err(Error::Invariant(format!(
            "{differing} of {elements} elements differ beyond tolerance: the widest is element \
             {index}: reference {reference}, candidate {candidate}, |difference| {difference}, \
             tolerance atol {} rtol {}",
            tolerance.atol, tolerance.rtol
        ))),
        None => Ok(elements),
    }
}

/// The `N` bytes of one element at byte `at` of a pair's copy.
fn element<const N: usize>(bytes: &[u8], at: usize) -> Result<[u8; N]> {
    bytes
        .get(at..at + N)
        .and_then(|slice| slice.try_into().ok())
        .ok_or_else(|| {
            Error::Invariant(format!(
                "a pair's copy holds no element of {N} bytes at byte {at}"
            ))
        })
}

#[cfg(test)]
mod tests {
    use ops::Tolerance;
    use tensor::DType;

    use super::compare_elements;

    fn f32s(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect()
    }

    #[test]
    fn identical_outputs_agree_and_the_count_is_elements() {
        let real = f32s(&[1.0, -2.5, 0.0, 3.0]);

        let compared =
            compare_elements(DType::F32, 4, &real, &real, Tolerance::new(0.0, 0.0)).unwrap();
        assert_eq!(compared, 4);
    }

    #[test]
    fn the_tolerance_decides_whether_a_difference_agrees() {
        let real = f32s(&[1.0, 2.0]);
        let close = f32s(&[1.0 + 1e-4, 2.0 - 1e-4]);

        compare_elements(DType::F32, 2, &real, &close, Tolerance::new(1e-3, 1e-3)).unwrap();

        // Beyond the bound, the failure names how many elements differ and the widest.
        let refused = compare_elements(DType::F32, 2, &real, &close, Tolerance::new(0.0, 0.0))
            .unwrap_err()
            .to_string();
        assert!(refused.contains("2 of 2 elements"), "{refused}");
        assert!(refused.contains("element 0"), "{refused}");
    }

    #[test]
    fn a_nan_agrees_only_with_a_nan() {
        let nans = f32s(&[f32::NAN, f32::NAN]);
        compare_elements(DType::F32, 2, &nans, &nans, Tolerance::new(0.0, 0.0)).unwrap();

        // A kernel that produced NaN everywhere must not pass against numbers, whatever the
        // bound.
        let numbers = f32s(&[1.0, 1.0]);
        let bound = Tolerance::new(f32::INFINITY, 1.0);
        assert!(compare_elements(DType::F32, 2, &nans, &numbers, bound).is_err());
    }

    #[test]
    fn elements_are_compared_as_the_numbers_of_their_element_type() {
        // 0x3f80 is 1.0 in BF16 and 0x3f81 the next value above it, 2^-7 further.
        let one = 0x3f80_u16.to_ne_bytes().to_vec();
        let next = 0x3f81_u16.to_ne_bytes().to_vec();

        compare_elements(DType::BF16, 1, &one, &one, Tolerance::new(0.0, 0.0)).unwrap();
        assert!(compare_elements(DType::BF16, 1, &one, &next, Tolerance::new(1e-3, 1e-3)).is_err());
        compare_elements(DType::BF16, 1, &one, &next, Tolerance::new(0.01, 0.01)).unwrap();
    }

    #[test]
    fn copies_of_another_size_than_the_pair_are_refused() {
        let real = f32s(&[1.0, 2.0]);
        let short = f32s(&[1.0]);

        assert!(compare_elements(DType::F32, 2, &real, &short, Tolerance::new(0.0, 0.0)).is_err());
    }
}
