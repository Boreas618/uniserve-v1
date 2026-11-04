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
) -> Option<Vec<i64>> {
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
        let slot = unit
            .checked_mul(block_size)?
            .checked_add(position % block_size)?;
        slots.push(i64::try_from(slot).ok()?);
    }
    Some(slots)
}

/// For each call of `calls`, in order, the forward row whose hidden state feeds
/// the call's `logits()`: the last row the call occupies, when the call has
/// consumer slots that draw from it.
///
/// A row is one query interval of `forward`, grouped by call in scheduling
/// order (`ForwardBatch::call_indices`). Sampling reads the interval's last
/// token, so the row to read is the last row carrying the call's index; a call
/// with no consumer slots samples nothing and states -1, as a row that no call
/// reads does.
///
/// `None` when a call that has consumer slots occupies no row: the batch would
/// ask a kernel to sample a row it does not have.
pub(crate) fn logits_rows(calls: &[ipc::Call], forward: &ipc::ForwardBatch) -> Option<Vec<i64>> {
    let mut last_row: Vec<Option<u32>> = vec![None; calls.len()];
    for (row, call_index) in forward.call_indices.iter().enumerate() {
        let call_index = usize::try_from(*call_index).ok()?;
        if call_index >= calls.len() {
            return None;
        }
        last_row[call_index] = Some(u32::try_from(row).ok()?);
    }
    calls
        .iter()
        .zip(last_row)
        .map(|(call, row)| {
            if call.consumer_slots.is_empty() {
                return Some(-1);
            }
            // A sampling call must occupy at least one row.
            Some(i64::from(row?))
        })
        .collect()
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

    #[test]
    fn a_sampling_call_reads_its_last_row() {
        let calls = [
            call(ipc::CallKind::Forward(ipc::ForwardMode::Prefill), true),
            call(ipc::CallKind::Forward(ipc::ForwardMode::Decode), true),
            call(
                ipc::CallKind::Transfer(ipc::TransferMode::CacheExport),
                false,
            ),
        ];
        // Call 0 has rows 0 and 1, call 1 has row 2, the transfer has none.
        let mut forward = ipc::ForwardBatch::default();
        forward.push(0, 4, 12, 8, true);
        forward.push(0, 4, 20, 8, true);
        forward.push(1, 5, 30, 1, true);

        assert_eq!(logits_rows(&calls, &forward), Some(vec![1, 2, -1]));
    }

    #[test]
    fn a_sampling_call_without_a_row_is_a_fault() {
        // Call 1 samples but no row carries it.
        let calls = [
            call(ipc::CallKind::Forward(ipc::ForwardMode::Prefill), true),
            call(ipc::CallKind::Forward(ipc::ForwardMode::Decode), true),
        ];
        let mut forward = ipc::ForwardBatch::default();
        forward.push(0, 4, 12, 8, true);

        assert_eq!(logits_rows(&calls, &forward), None);

        // A row that names no call in the batch is a fault too.
        let mut stray = ipc::ForwardBatch::default();
        stray.push(7, 4, 12, 8, true);
        assert_eq!(logits_rows(&calls, &stray), None);
    }
}
