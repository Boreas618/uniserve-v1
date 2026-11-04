//! The batch column values the mechanism computes (design §9: the mechanism
//! writes the batch in the layout the kernels read): the KV slot each query
//! token writes, and the forward row whose hidden state feeds `logits()`.
//!
//! Everything else the kernels read is a fact of the calls the mechanism
//! builds, which it can state as it builds them; these two are computed from a
//! request's block table and from the batch's rows, so they live here as pure
//! functions the assembly calls once per call.
//!
//! A request's KV lives in physical cache units. One logical page of a cache
//! group holds `block_size` tokens and occupies `units_per_page` units
//! (`ipc::CacheGroup`, reported by the worker at handshake and kept in
//! `WorkerInfo`). The KV-write kernels address one row of the cache planes per
//! token and store it themselves as `unit * block_size + position % block_size`
//! (`ops/kernels/gather_decode_inputs`, `gather_request_rows`), so the mechanism
//! states the same number in the batch's slot column. Nothing else derives it:
//! the executor submits the column, it does not build it.
//!
//! The mapping is a pure function of a table, the group's page geometry and the
//! positions of the tokens a call evaluates, which is why it lives here rather
//! than in the call-building code.

use std::ops::Range;

/// A cache group's page shape, as the worker reported it (`ipc::CacheGroup`).
///
/// Only `block_size` is needed to turn a position into a slot; `units_per_page`
/// is carried so the mapping can refuse a page shape it cannot address (see
/// [`slot_mapping`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PageGeometry {
    /// Tokens one logical page holds.
    pub block_size: u32,
    /// Physical units one logical page occupies.
    pub units_per_page: u32,
}

/// The slot each position of `positions` writes through `table`.
///
/// A position is absolute in its request's sequence. The page holding it is
/// `position / block_size`; the table's `start_page` is the first page its
/// `unit_ids` cover, so the unit is `unit_ids[page - start_page]` and the slot
/// is that unit's first token slot plus the position's offset in its page.
///
/// A token whose KV is not written states `-1` (see [`slot_mapping_where`]).
///
/// `None` when:
/// - a position lies before `start_page` or past the table's last page, which is
///   a planning fault rather than something to clamp: a batch must not write a
///   slot it does not hold;
/// - the page geometry is unusable — `block_size == 0`, or `units_per_page != 1`.
///   A page of several units holds different layers' columns in each unit, so
///   one slot number per token addresses one of them; the kernels index one unit
///   per page today, and a multi-unit page needs a column per unit, which is a
///   decision for the executor rather than a guess here.
pub(crate) fn slot_mapping(
    table: &ipc::BlockTable,
    page: PageGeometry,
    positions: Range<u32>,
) -> Option<Vec<i32>> {
    slot_mapping_where(table, page, positions, |_| true)
}

/// The slot each position of `positions` writes through `table`, with `-1` for a
/// position the mechanism decides writes nothing.
///
/// The write decision is the mechanism's, per token: a token's KV is written
/// unless its row's write flag says otherwise, so a caller expands its rows'
/// flags to tokens and `writes` answers for each position. `-1` writes nothing
/// and initializes nothing, which is what the kernels do with it; a position
/// outside the table is still a fault rather than a `-1`.
pub(crate) fn slot_mapping_where(
    table: &ipc::BlockTable,
    page: PageGeometry,
    positions: Range<u32>,
    writes: impl Fn(u32) -> bool,
) -> Option<Vec<i32>> {
    if page.block_size == 0 || page.units_per_page != 1 {
        return None;
    }
    let block_size = u64::from(page.block_size);
    let start_page = u64::from(table.start_page);
    // The table's end page is the first page it does not cover: its absolute
    // token extent over the page size.
    let end_page = u64::from(table.allocated_tokens) / block_size;
    let mut slots = Vec::with_capacity(positions.len());
    for position in positions {
        let position = u64::from(position);
        let page_index = position / block_size;
        if page_index < start_page || page_index >= end_page {
            return None;
        }
        let index = usize::try_from(page_index - start_page).ok()?;
        let unit = u64::from(table.unit_ids.get(index)?.0);
        if !writes(u32::try_from(position).ok()?) {
            slots.push(-1);
            continue;
        }
        let slot = unit
            .checked_mul(block_size)?
            .checked_add(position % block_size)?;
        slots.push(i32::try_from(slot).ok()?);
    }
    Some(slots)
}

/// The batch's `[rows + 1]` packed query offsets: row `r` holds query tokens
/// `offsets[r]..offsets[r + 1]` of the packed token columns.
///
/// `query_lens` is one row's query length per row, in row order. `None` when a
/// length or the total does not fit the column's `i32`, which a batch of this
/// size cannot address.
pub(crate) fn query_offsets(query_lens: &[u32]) -> Option<Vec<i32>> {
    let mut offsets = Vec::with_capacity(query_lens.len() + 1);
    let mut total = 0u32;
    offsets.push(0);
    for len in query_lens {
        total = total.checked_add(*len)?;
        offsets.push(i32::try_from(total).ok()?);
    }
    Some(offsets)
}

/// The absolute positions of the query tokens a call evaluates: its first
/// token's logical position, then one per token of the call.
///
/// A call's query interval is contiguous in its request's sequence, which is
/// what `Call.coordinates.logical_position` and its input tokens state.
pub(crate) fn positions(call: &ipc::Call) -> Option<Vec<i32>> {
    let first = u32::try_from(call.coordinates.logical_position).ok()?;
    let tokens = u32::try_from(call.input_token_ids.len()).ok()?;
    let mut positions = Vec::with_capacity(call.input_token_ids.len());
    for offset in 0..tokens {
        positions.push(i32::try_from(first.checked_add(offset)?).ok()?);
    }
    Some(positions)
}

/// One entry per forward row: the row whose hidden state the row's logits come
/// from, or -1 when the row produces no logits.
///
/// `rows[i]` is the half-open span of forward rows call `i` occupies, in call
/// order and contiguous, which is how the assembly builds them. What samples
/// depends on the kind (`CausalLm`'s reading, model's answer):
///
/// - a **prefill** is sampled once, at its last query row, so only that row's
///   entry names a row;
/// - a **decode** evaluates one row per request and every row samples, so the
///   mapping is the identity;
/// - a **verify** carries one row per candidate position of its request, in
///   candidate order, so its rows are the positions that sample and the mapping
///   is the identity there too. That rests on model's confirmation that every
///   row of a verify call is a candidate position; a call that also carries a
///   row producing no logits would need that row marked -1 instead;
/// - every other kind produces latents rather than logits, and a call with no
///   consumer slots is sampled by nobody: both leave -1 on every row.
///
/// `None` when the calls and the spans disagree, when the spans are not the
/// contiguous row order of one batch, or when a sampling prefill has no row —
/// a batch must not ask a kernel to sample a row it does not have.
pub(crate) fn logits_rows(calls: &[ipc::Call], rows: &[Range<u32>]) -> Option<Vec<i32>> {
    if calls.len() != rows.len() {
        return None;
    }
    let mut total = 0usize;
    for span in rows {
        if usize::try_from(span.start).ok()? != total || span.end < span.start {
            return None;
        }
        total = usize::try_from(span.end).ok()?;
    }
    let mut logits = vec![-1i32; total];
    for (call, span) in calls.iter().zip(rows) {
        if call.consumer_slots.is_empty() {
            continue;
        }
        let start = usize::try_from(span.start).ok()?;
        let end = usize::try_from(span.end).ok()?;
        match call.kind {
            ipc::CallKind::Forward(ipc::ForwardMode::Prefill) => {
                let last = end.checked_sub(1).filter(|last| *last >= start)?;
                logits[last] = i32::try_from(last).ok()?;
            }
            ipc::CallKind::Forward(ipc::ForwardMode::Decode | ipc::ForwardMode::Verify) => {
                for row in start..end {
                    logits[row] = i32::try_from(row).ok()?;
                }
            }
            _ => {}
        }
    }
    Some(logits)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Four tokens per page, one unit per page: the geometry every token group
    /// reports today.
    fn page() -> PageGeometry {
        PageGeometry {
            block_size: 4,
            units_per_page: 1,
        }
    }

    /// A table covering pages 2..5 (positions 8..20) whose units are 7, 9 and
    /// 10 — not contiguous, because the pool assigns what it has.
    fn table() -> ipc::BlockTable {
        ipc::BlockTable {
            request_pool_idx: 1,
            group_id: 0,
            start_page: 2,
            unit_ids: vec![ipc::UnitId(7), ipc::UnitId(9), ipc::UnitId(10)],
            allocated_tokens: 20,
        }
    }

    #[test]
    fn a_token_writes_the_row_of_its_page_and_its_offset() {
        // Position 8 is page 2's first token, so unit 7's first slot; position
        // 11 is its last; position 12 opens unit 9.
        assert_eq!(
            slot_mapping(&table(), page(), 8..14).unwrap(),
            vec![7 * 4, 7 * 4 + 1, 7 * 4 + 2, 7 * 4 + 3, 9 * 4, 9 * 4 + 1]
        );
        assert_eq!(
            slot_mapping(&table(), page(), 19..20).unwrap(),
            vec![10 * 4 + 3],
            "the last token the table covers"
        );
        assert!(slot_mapping(&table(), page(), 8..8).unwrap().is_empty());
    }

    #[test]
    fn a_position_outside_the_table_has_no_slot() {
        // Before the table's first page, and past its last.
        assert_eq!(slot_mapping(&table(), page(), 7..9), None);
        assert_eq!(slot_mapping(&table(), page(), 19..21), None);
    }

    #[test]
    fn a_page_shape_the_kernels_cannot_address_has_no_slot() {
        let multi_unit = PageGeometry {
            block_size: 4,
            units_per_page: 2,
        };
        assert_eq!(slot_mapping(&table(), multi_unit, 8..9), None);
        let no_tokens = PageGeometry {
            block_size: 0,
            units_per_page: 1,
        };
        assert_eq!(slot_mapping(&table(), no_tokens, 8..9), None);
        let empty = ipc::BlockTable {
            unit_ids: Vec::new(),
            ..table()
        };
        assert_eq!(slot_mapping(&empty, page(), 8..9), None);
    }

    /// A call of `kind` with the consumer slots a sampling call has.
    fn call(kind: ipc::CallKind, samples: bool) -> ipc::Call {
        let mut call = ipc::Call::new(
            ipc::RequestKey::new(1, ipc::RequestId(1), 1),
            ipc::CallId::new(1, 0),
            kind,
        );
        call.consumer_slots = if samples { vec![0] } else { Vec::new() };
        call
    }

    fn prefill() -> ipc::CallKind {
        ipc::CallKind::Forward(ipc::ForwardMode::Prefill)
    }

    fn decode() -> ipc::CallKind {
        ipc::CallKind::Forward(ipc::ForwardMode::Decode)
    }

    #[test]
    fn a_prefill_is_sampled_at_its_last_row_only() {
        // Two calls, three rows: a prefill of two query rows, then a decode.
        let calls = [call(prefill(), true), call(decode(), true)];
        assert_eq!(logits_rows(&calls, &[0..2, 2..3]), Some(vec![-1, 1, 2]));
    }

    #[test]
    fn a_decode_and_a_verify_sample_every_row() {
        let verify = ipc::CallKind::Forward(ipc::ForwardMode::Verify);
        let calls = [call(decode(), true), call(verify, true)];
        assert_eq!(
            logits_rows(&calls, &[0..2, 2..5]),
            Some(vec![0, 1, 2, 3, 4])
        );
    }

    #[test]
    fn a_call_nobody_samples_and_a_media_call_produce_no_logits() {
        let calls = [
            call(prefill(), false),
            call(decode(), false),
            call(ipc::CallKind::Media(ipc::MediaCall::Denoising), true),
        ];
        assert_eq!(
            logits_rows(&calls, &[0..1, 1..3, 3..4]),
            Some(vec![-1, -1, -1, -1])
        );
    }

    #[test]
    fn mismatched_rows_are_a_fault() {
        let calls = [call(prefill(), true), call(decode(), true)];
        // One span for two calls.
        assert_eq!(logits_rows(&calls, &[0..1]), None);
        // Spans that are not one contiguous row order.
        assert_eq!(logits_rows(&calls, &[0..2, 3..4]), None);
        assert_eq!(logits_rows(&calls, &[1..2, 2..3]), None);
        // A sampling prefill with no row.
        assert_eq!(logits_rows(&calls, &[0..0, 0..1]), None);
    }

    fn token_call(positions: &[u32], logical_position: u32) -> ipc::Call {
        let mut call = call(prefill(), true);
        call.coordinates.logical_position = logical_position;
        call.input_token_ids = positions.to_vec();
        call
    }

    #[test]
    fn query_offsets_pack_the_rows_of_the_batch() {
        assert_eq!(query_offsets(&[2, 1, 3]), Some(vec![0, 2, 3, 6]));
        assert_eq!(query_offsets(&[]), Some(vec![0]));
        // A total the column cannot address.
        assert_eq!(query_offsets(&[u32::MAX, 1]), None);
    }

    #[test]
    fn positions_are_the_call_s_query_interval() {
        assert_eq!(positions(&token_call(&[0; 3], 12)), Some(vec![12, 13, 14]));
        assert_eq!(positions(&token_call(&[], 12)), Some(Vec::new()));
        // A position past the column's range.
        assert_eq!(positions(&token_call(&[0; 2], u32::MAX)), None);
    }

    #[test]
    fn a_token_the_mechanism_does_not_write_states_minus_one() {
        // Page 2's four tokens (8..12, unit 7), writing only the even positions.
        let slots =
            slot_mapping_where(&table(), page(), 8..12, |position| position % 2 == 0).unwrap();
        assert_eq!(slots, vec![7 * 4, -1, 7 * 4 + 2, -1]);
        // Writing nothing is still a slot decision, not a fault.
        assert_eq!(
            slot_mapping_where(&table(), page(), 8..10, |_| false).unwrap(),
            vec![-1, -1]
        );
        // A position outside the table stays a fault whatever the write flag.
        assert_eq!(
            slot_mapping_where(&table(), page(), 19..21, |_| false),
            None
        );
    }
}
