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
//!
//! ## Why a wrong kernel never passed probation
//!
//! Until the scratch page set was published (`PrefixCache::scratch`), every paired forward refused
//! at the cache provider: there was no page set it was allowed to write, and writing a resident one
//! would have overwritten the KV of the very batch it was walking. A stated refusal is why the
//! delivery test could not be fooled by a deliberately wrong kernel in the meantime - there was no
//! shadow pass to fool. Now the set exists, the walk stages its own slot column into it, and the
//! refusal is gone; what remains is the comparison itself.

use std::sync::Arc;

use ops::{DType, TensorView, Tolerance, Twins};
use platform::{DeviceId, PinnedBuffer, Stream};

use crate::{Error, Result};

/// What one paired forward compared.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    /// How many twin buffers were compared.
    pub pairs: usize,
    /// How many elements were compared, over all pairs.
    pub elements: u64,
    /// The pairs that could not be compared because the two programs wrote outputs of different
    /// extents, each with the sentence that states it (7.3 step 2).
    ///
    /// An operator whose programs are not comparable is skipped, not failed: the operating system
    /// has nothing to say about which of the two is right, so the update proceeds and states what
    /// it has no evidence for rather than claiming a comparison it could not make.
    pub incomparable: Vec<String>,
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
    /// A harness over `twins`, whose staging lives on `device`.
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

    /// The tolerance the paired forward's operator recorded when it was shadowed, if it was.
    ///
    /// `Op::shadow` is where the operator's type is known, so `Forward::operate` records
    /// `O::tolerance(descriptor)` in the region there; the harness compares with it.
    pub fn tolerance(&self) -> Option<Tolerance> {
        self.twins.tolerance()
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
            incomparable: Vec::new(),
        };
        for pair in pairs {
            let bytes = pair.real.byte_len()?;
            if bytes != pair.shadow.byte_len()? {
                // The two programs wrote outputs of different extents: they are not comparable, and
                // an update is not refused for it - the pair is skipped and named, so the update's
                // evidence states what it could not measure (7.3 step 2).
                summary.incomparable.push(format!(
                    "the twin and the real buffer have different sizes: the real output holds \
                     {bytes} bytes of {} and the twin {} of {}, so the two programs are not \
                     comparable",
                    pair.real.dtype,
                    pair.shadow.byte_len()?,
                    pair.shadow.dtype,
                ));
                summary.pairs -= 1;
                continue;
            }
            let bytes = usize::try_from(bytes)
                .map_err(|_| Error::Invariant("a twin buffer is larger than memory".into()))?;
            self.staging.reserve(bytes)?;
            self.staging.read(stream, &pair.real, &pair.shadow)?;
            summary.elements += compare_buffers(
                pair.real.dtype,
                pair.real.numel(),
                &self.staging.real()[..bytes],
                &self.staging.shadow()[..bytes],
                tolerance,
            )?;
        }
        self.twins.reset();
        Ok(summary)
    }
}

/// Compares one pair's host copies within `tolerance` and answers how many elements it compared.
///
/// `real` and `shadow` are the element bytes of the old kernel's output and of the new kernel's
/// twin, `numel` elements of `dtype` each. The comparison is what 7.3 requires of a shadow run:
/// element by element, in the dtype's own value domain, within the operator's [`Tolerance`], with
/// two NaNs agreeing and a NaN against a number not.
///
/// A difference names the operator's bound as well as the measurement - how many elements
/// differ, the widest one's reference and candidate, its absolute difference, and the `atol` and
/// `rtol` it left - because "the kernels differ" does not say whether a program is wrong or was
/// compiled twice.
pub fn compare_buffers(
    dtype: DType,
    numel: u64,
    real: &[u8],
    shadow: &[u8],
    tolerance: Tolerance,
) -> Result<u64> {
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
    // Every element is compared, not only up to the first difference: the reason an update is
    // refused states how far apart the two programs are and how many elements that is, which is
    // what tells a wrong kernel from a rounding, and the scan is host memory over a bucket.
    let read = |offset: usize| -> Result<(f32, f32)> {
        // Each arm reads the element through `element`, which names the offset a buffer too short
        // for the dtype would have failed at: the check above makes that unreachable, and a panic
        // would hide the cause if it ever stopped being true.
        let values = match dtype {
            DType::F32 => (
                f32::from_ne_bytes(element::<4>(real, offset)?) as f64,
                f32::from_ne_bytes(element::<4>(shadow, offset)?) as f64,
            ),
            DType::BF16 => (
                bf16_to_f32(&element::<2>(real, offset)?) as f64,
                bf16_to_f32(&element::<2>(shadow, offset)?) as f64,
            ),
            DType::F16 => (
                f16_to_f32(u16::from_ne_bytes(element::<2>(real, offset)?)) as f64,
                f16_to_f32(u16::from_ne_bytes(element::<2>(shadow, offset)?)) as f64,
            ),
            DType::I32 => (
                i32::from_ne_bytes(element::<4>(real, offset)?) as f64,
                i32::from_ne_bytes(element::<4>(shadow, offset)?) as f64,
            ),
            DType::I64 => (
                i64::from_ne_bytes(element::<8>(real, offset)?) as f64,
                i64::from_ne_bytes(element::<8>(shadow, offset)?) as f64,
            ),
            DType::U8 => (
                *real.get(offset).ok_or_else(|| missing(offset, 1))? as f64,
                *shadow.get(offset).ok_or_else(|| missing(offset, 1))? as f64,
            ),
            other => {
                return Err(Error::Invariant(format!(
                    "shadow comparison of {other:?} is not supported"
                )));
            }
        };
        Ok((values.0 as f32, values.1 as f32))
    };
    // A bound that cancels is read against the output's largest magnitude, so that comes first.
    let mut peak = 0.0_f32;
    for index in 0..numel as usize {
        let (reference, _) = read(index * width)?;
        if reference.is_finite() {
            peak = peak.max(reference.abs());
        }
    }
    let mut differing: u64 = 0;
    let mut widest: Option<(usize, f32, f32, f32)> = None;
    for index in 0..numel as usize {
        let (reference, candidate) = read(index * width)?;
        // A NaN in one buffer and a number in the other is a difference; two NaNs agree, and so do
        // two infinities of one sign, whose difference is a NaN no bound can hold.
        let agree = if (reference.is_nan() && candidate.is_nan()) || reference == candidate {
            true
        } else {
            tolerance.allows_within(peak, reference, candidate)
        };
        if !agree {
            differing += 1;
            // `total_cmp` orders the non-finite differences too, so a NaN or an infinity is
            // selected as the widest and named rather than dropped by a comparison that would
            // answer false for it.
            let difference = (reference - candidate).abs();
            let wider =
                widest.is_none_or(|(_, _, _, widest)| difference.total_cmp(&widest).is_gt());
            if wider {
                widest = Some((index, reference, candidate, difference));
            }
        }
    }
    if let Some((index, reference, candidate, difference)) = widest {
        return Err(Error::Invariant(format!(
            "{differing} of {numel} elements differ beyond tolerance: the widest is element \
             {index}: reference {reference}, candidate {candidate}, |difference| {difference}, \
             tolerance atol {} rtol {}",
            tolerance.atol, tolerance.rtol
        )));
    }
    Ok(numel)
}

/// One element of `N` bytes at `offset`, or the error naming the offset.
fn element<const N: usize>(bytes: &[u8], offset: usize) -> Result<[u8; N]> {
    bytes
        .get(offset..offset + N)
        .and_then(|slice| slice.try_into().ok())
        .ok_or_else(|| missing(offset, N))
}

/// A buffer that holds no element of `width` bytes at `offset`.
fn missing(offset: usize, width: usize) -> Error {
    Error::Invariant(format!(
        "a twin buffer has no element of {width} bytes at offset {offset}"
    ))
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
            platform::memcpy_async(self.real.address(), real.ptr.address(), bytes, stream)?;
            platform::memcpy_async(self.shadow.address(), shadow.ptr.address(), bytes, stream)?;
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

/// The cache slots a paired forward writes: the same tokens as the served batch, in the scratch
/// set's own numbering.
///
/// A slot is flat over one plane's pages (`ops::kv`'s `Args::slots`: `page · page_size + offset`),
/// and the scratch set gives row `r` the pages `r · pages .. (r + 1) · pages` - the ids its block
/// table states (`ScratchCache`). So token `t` of row `r` at position `p` writes
/// `(r · pages + p / page_size) · page_size + p % page_size`, which is inside the row's own pages
/// and never a resident slot. A token the served batch does not write (`slots[t] < 0`, a padded
/// row) writes nothing here either.
///
/// `positions` is `[tokens]`, `query_offsets` is `[rows + 1]` packed, and `slots` is the served
/// batch's `[tokens]` slot column. The result is `[tokens]`, the column the paired forward stages
/// in place of the served batch's.
pub fn scratch_slot_mapping(
    positions: &[i32],
    query_offsets: &[i32],
    slots: &[i32],
    pages: u32,
    page_size: u32,
) -> Result<Vec<i32>> {
    if page_size == 0 || pages == 0 {
        return Err(Error::Invariant(
            "a scratch page set needs a non-zero page and page size".into(),
        ));
    }
    if query_offsets.len() < 2 {
        return Err(Error::Invariant(
            "a batch's packed query offsets name no row".into(),
        ));
    }
    match query_offsets.last().copied() {
        Some(tokens) if tokens as usize == positions.len() && positions.len() == slots.len() => {}
        _ => {
            return Err(Error::Invariant(
                "the batch's offsets, positions and slots describe different token counts".into(),
            ));
        }
    }
    let rows = query_offsets.len() - 1;
    let mut mapping = vec![-1_i32; positions.len()];
    for row in 0..rows {
        let (start, end) = (query_offsets[row], query_offsets[row + 1]);
        if start < 0 || end < start {
            return Err(Error::Invariant(format!(
                "row {row}'s packed query offsets are {start}..{end}"
            )));
        }
        for token in start as usize..end as usize {
            if slots[token] < 0 {
                continue;
            }
            let position = positions[token];
            if position < 0 {
                return Err(Error::Invariant(format!(
                    "token {token} writes cache page {position} of the batch, which is no position"
                )));
            }
            let (page, offset) = (position as u32 / page_size, position as u32 % page_size);
            if page >= pages {
                return Err(Error::Invalid(format!(
                    "row {row} writes position {position}, which needs page {page} of a scratch \
                     set of {pages} pages; the paired forward's page set is too small"
                )));
            }
            let flat = (row as u32 * pages + page) * page_size + offset;
            mapping[token] = i32::try_from(flat).map_err(|_| {
                Error::Invalid(format!(
                    "a scratch slot of {flat} is outside the slot range a cache address holds"
                ))
            })?;
        }
    }
    Ok(mapping)
}

/// The scratch slot column a paired forward is given, over the **bucket's** token extent.
///
/// A captured graph addresses the whole column, so the mapping must state a value for every token
/// of the bucket and not only for the batch's own: a token the batch does not hold names no page
/// (`-1`), because a value left over from an earlier batch would send the paired forward's writes
/// into a page a served request reads — the one thing the scratch numbering exists to prevent
/// (7.3). `query_offsets` is the staged column, whose last entry is the batch's live count; the
/// positions and slots are sliced to that count so the three describe one batch.
pub fn scratch_column(
    positions: &[i32],
    query_offsets: &[i32],
    slots: &[i32],
    live: usize,
    tokens: usize,
    pages: u32,
    page_size: u32,
) -> Result<Vec<i32>> {
    if live > positions.len() || live > slots.len() || tokens < live {
        return Err(Error::Invariant(format!(
            "a batch of {live} live token(s) in a bucket of {tokens} names {} position(s) and {} \
             slot(s)",
            positions.len(),
            slots.len()
        )));
    }
    let mut mapping = if live == 0 {
        vec![-1_i32; tokens]
    } else {
        scratch_slot_mapping(
            &positions[..live],
            query_offsets,
            &slots[..live],
            pages,
            page_size,
        )?
    };
    mapping.resize(tokens, -1);
    Ok(mapping)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tolerance(atol: f32, rtol: f32) -> Tolerance {
        Tolerance::new(atol, rtol)
    }

    fn f32s(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect()
    }

    /// The old and the new kernel agreeing exactly is a passing comparison, and the count is the
    /// elements compared, not the bytes.
    #[test]
    fn identical_buffers_agree() {
        let real = f32s(&[1.0, -2.5, 0.0, 3.0]);
        let compared = compare_buffers(DType::F32, 4, &real, &real, tolerance(0.0, 0.0))
            .expect("identical buffers agree");
        assert_eq!(compared, 4);
    }

    /// A new kernel that differs within the operator's tolerance passes; one that differs beyond it
    /// fails, and the failure names the element.
    #[test]
    fn tolerance_decides_whether_a_difference_agrees() {
        let real = f32s(&[1.0, 2.0]);
        let close = f32s(&[1.0 + 1e-4, 2.0 - 1e-4]);
        compare_buffers(DType::F32, 2, &real, &close, tolerance(1e-3, 1e-3))
            .expect("a difference within tolerance agrees");

        let error = compare_buffers(DType::F32, 2, &real, &close, tolerance(0.0, 0.0))
            .expect_err("a difference beyond tolerance fails");
        let message = error.to_string();
        assert!(message.contains("element 0"), "{message}");
    }

    /// Each row of a paired forward writes its own scratch pages: row 0's three tokens and row
    /// 1's two are addressed as `(row * pages + page) * page_size + offset`, so no two rows share
    /// a page and none of them names a resident slot.
    #[test]
    fn a_paired_forward_writes_its_own_scratch_pages() {
        let mapping =
            scratch_slot_mapping(&[5, 6, 7, 0, 1], &[0, 3, 5], &[10, 11, 12, 20, 21], 2, 4)
                .expect("every token fits the set");
        // Row 0: positions 5..8 are page 1, offsets 1..3. Row 1: positions 0..2, page 0.
        assert_eq!(mapping, vec![5, 6, 7, 8, 9]);
    }

    /// A token the served batch does not write (a padded row, `slot < 0`) writes no scratch page
    /// either, and a position past the set's pages is refused by name rather than wrapped.
    #[test]
    fn a_padded_token_writes_nothing_and_a_full_set_is_refused() {
        let mapping = scratch_slot_mapping(&[0, 0], &[0, 2], &[-1, 3], 1, 4)
            .expect("the padded token and the written one are both in range");
        assert_eq!(mapping, vec![-1, 0]);

        let error = scratch_slot_mapping(&[9], &[0, 1], &[7], 1, 4)
            .expect_err("position 9 needs page 2 of a one-page set");
        assert!(error.to_string().contains("page 2"), "{error}");
    }

    /// A bucket wider than the batch names no page for the tokens the batch does not hold: the
    /// column a captured graph addresses is the bucket's, and a leftover slot there would write
    /// into a page a served request reads (7.3).
    #[test]
    fn a_partial_buckets_scratch_tail_names_no_page() {
        let mapping = scratch_column(&[0, 1, 2], &[0, 3], &[10, 11, 12], 3, 8, 2, 4)
            .expect("three tokens of an eight-token bucket");
        assert_eq!(mapping.len(), 8);
        assert_eq!(
            &mapping[..3],
            &[0, 1, 2],
            "the batch's own tokens keep their pages"
        );
        assert!(
            mapping[3..].iter().all(|slot| *slot < 0),
            "the bucket's tail names no page: {mapping:?}"
        );

        // A batch that holds nothing writes nothing, whatever the bucket's width.
        let empty = scratch_column(&[], &[0], &[], 0, 4, 2, 4).expect("an empty batch");
        assert_eq!(empty, vec![-1; 4]);

        // A batch wider than the bucket it is served from is the case the guard exists for: the
        // mapping is sized by the bucket's token extent, so a live count past it would write past
        // the column. The case below states it with numbers that really violate it - two live
        // tokens in a bucket of one - and then with a bucket of none.
        let error = scratch_column(&[0, 1], &[0, 2], &[7, 8], 2, 1, 1, 4)
            .expect_err("two live tokens cannot name a bucket of one");
        assert!(error.to_string().contains("bucket of"), "{error}");

        let error = scratch_column(&[0], &[0, 1], &[7], 1, 0, 1, 4)
            .expect_err("one live token cannot name a bucket of none");
        assert!(error.to_string().contains("bucket of"), "{error}");

        // The three columns describe one batch: a slot or a position list shorter than the live
        // count is a batch whose own columns disagree, and the refusal names the lengths.
        let error = scratch_column(&[0, 1], &[0, 2], &[7], 2, 2, 1, 4)
            .expect_err("two positions and one slot are not one batch");
        assert!(error.to_string().contains("slot(s)"), "{error}");
    }

    /// A NaN has no distance to anything, so the comparison treats two NaNs as agreeing and a NaN
    /// against a number as a difference; otherwise a kernel that produced NaN everywhere would pass
    /// against a NaN reference.
    #[test]
    fn nans_agree_only_with_nans() {
        let nans = f32s(&[f32::NAN, f32::NAN]);
        compare_buffers(DType::F32, 2, &nans, &nans, tolerance(0.0, 0.0)).expect("two NaNs agree");

        let numbers = f32s(&[1.0, 1.0]);
        compare_buffers(
            DType::F32,
            2,
            &nans,
            &numbers,
            tolerance(f32::INFINITY, 1.0),
        )
        .expect_err("a NaN against a number differs");
    }

    /// Elements are compared in their own dtype's value domain: the same numerical value written as
    /// two element types agrees, and a bf16 value is read as the number it denotes rather than as
    /// its bit pattern.
    #[test]
    fn elements_are_compared_as_numbers_of_their_dtype() {
        let ones = f32s(&[1.0]);
        let bf16_one = 0x3f80u16.to_ne_bytes().to_vec();
        compare_buffers(DType::F32, 1, &ones, &ones, tolerance(0.0, 0.0)).expect("f32 agrees");
        compare_buffers(DType::BF16, 1, &bf16_one, &bf16_one, tolerance(0.0, 0.0))
            .expect("bf16 agrees");

        // The bf16 value next to 1.0 (one unit in the last place) differs from 1.0 by 2^-7.
        let next = 0x3f81u16.to_ne_bytes().to_vec();
        compare_buffers(DType::BF16, 1, &bf16_one, &next, tolerance(1e-3, 1e-3))
            .expect_err("one bf16 step is beyond a 1e-3 tolerance");
        compare_buffers(DType::BF16, 1, &bf16_one, &next, tolerance(0.01, 0.01))
            .expect("one bf16 step is within a 1e-2 tolerance");
    }

    /// A copy that does not hold the pair's elements is refused rather than compared short.
    #[test]
    fn a_copy_of_the_wrong_size_is_refused() {
        let real = f32s(&[1.0, 2.0]);
        let short = f32s(&[1.0]);
        compare_buffers(DType::F32, 2, &real, &short, tolerance(0.0, 0.0))
            .expect_err("a short copy is refused");
    }
}
