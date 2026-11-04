//! One batch's executor columns, filled from the calls the mechanism planned.
//!
//! A batch is one numerical call of one phase on one component (§4.4, `M499`),
//! and the executor reads it in the layout the kernels address: `nn::BatchRows`
//! — one [`nn::BatchRow`] per call with the half-open row range it owns, then
//! the token and row columns — beside the sampler's own columns. Nothing is
//! converted between the mechanism and the model: the mechanism writes exactly
//! what the kernels read, and the executor stages it into the pinned arena.
//!
//! The columns this module fills are the ones the calls themselves state:
//!
//! - **rows**: one `BatchRow` per call, contiguous in plan order, with the
//!   request slot the call's rows belong to, its [`nn::RowKind`] and its rows;
//! - **positions**: each call's query tokens start at its own logical position;
//! - **slot_mapping**: one absolute slot per token through the call's block
//!   table, `-1` where the call's row writes no cache page — the mechanism's
//!   decision, taken from `ipc::ForwardBatch::write_cache`, never a clamp;
//! - **seq_lens**: each row's attention length;
//! - **query_offsets**: the packed token range of every row;
//! - **logits_rows**: the rows whose logits are wanted (a prefill's last row, a
//!   decode's or a verify's every row), `-1` elsewhere;
//! - **slots**: the sampler's per-slot state row;
//! - **page_size** and **max_kv_len**: the cache's geometry.
//!
//! Two families are not filled here, because the executor's type states them but
//! nothing in the tree yet states their contents: the sampler's parameter
//! columns (`BatchColumns::parameters`, `ops::sampler::PARAMETER_COLUMNS` of
//! them per row) and the commit table (`BatchColumns::commit`). Both are asked
//! of exec (`M708`); a batch is submitted only when they are filled, so this
//! module answers the columns and the caller completes them in one place.
//!
//! A batch whose calls disagree with what the layout promises — a call whose
//! rows belong to several requests, a token outside its table, a call a token
//! batch cannot carry — answers `None`: a planning fault, never a batch that
//! would run on plausible values. The executor's own `BatchColumns::check` is
//! the second line of that defence.

use std::ops::Range;

use executor::batch::BatchColumns;
use ipc::{Call, CallKind, ForwardMode};
use nn::{BatchRow, BatchRows, RowKind};

use super::slots::{self, PageGeometry};
use crate::executor::RequestPlacement;

/// The kind of row a call occupies in a token batch, or `None` for a call a
/// token batch does not carry.
///
/// A media call's numerical extent is its own (`executor`'s media runners), and
/// a transfer moves bytes: neither is a token row. `ForwardMode::TokenDenoising`
/// likewise runs on the diffusion LM's canvas rather than on token rows, so it
/// is not a token batch of this shape either — the media fold's batch is the
/// one that carries it (`M521`, `M708` question 4).
const fn row_kind(kind: CallKind) -> Option<RowKind> {
    match kind {
        CallKind::Forward(ForwardMode::Prefill) => Some(RowKind::Prefill),
        CallKind::Forward(ForwardMode::Decode) => Some(RowKind::Decode),
        CallKind::Forward(ForwardMode::Verify) => Some(RowKind::Verify),
        _ => None,
    }
}

/// The columns of the batch formed by `requests`, whose calls' rows embed the
/// tokens `inputs` states for them, in call order.
///
/// `inputs[i]` holds the tokens of call `i`'s rows, back to back: a prefill's or
/// a verify's host tokens (`ipc::Call::input_token_ids`) or a decode's one token
/// per row. A call that states fewer tokens than its rows evaluate is a fault:
/// the mechanism never invents a token a kernel would then embed.
///
/// `page` is the cache's page geometry and `max_kv_len` the attention
/// descriptor's static bound on a sequence, both of which the mechanism states
/// to the executor rather than letting each side derive them.
pub(crate) fn columns(
    requests: &[(Call, RequestPlacement)],
    inputs: &[Vec<i32>],
    page: PageGeometry,
    max_kv_len: u32,
) -> Option<BatchColumns> {
    if requests.len() != inputs.len() {
        return None;
    }
    let mut rows = BatchRows::default();
    let mut slots = Vec::with_capacity(requests.len());
    let mut queries = Vec::new();
    let mut spans: Vec<Range<u32>> = Vec::with_capacity(requests.len());
    let mut input_ids: Vec<i32> = Vec::new();
    for ((call, placement), inputs) in requests.iter().zip(inputs) {
        let forward = &placement.forward;
        let kind = row_kind(call.kind)?;
        // Every row of one call belongs to the call's request: the placement
        // group a call by request, so rows of two requests in one call is a
        // fault of the assembly, not something to split here.
        let request = *forward.request_pool_indices.first()?;
        if forward.request_pool_indices.len() != forward.query_lens.len()
            || forward.seq_lens.len() != forward.query_lens.len()
            || forward.write_cache.len() != forward.query_lens.len()
            || forward
                .request_pool_indices
                .iter()
                .any(|index| *index != request)
        {
            return None;
        }
        let rows_of_call = u32::try_from(forward.query_lens.len()).ok()?;
        let tokens_of_call = forward
            .query_lens
            .iter()
            .try_fold(0u32, |total, length| total.checked_add(*length))?;
        if usize::try_from(tokens_of_call).ok()? != inputs.len() {
            return None;
        }

        let first = u32::try_from(call.coordinates.logical_position).ok()?;
        rows.positions.extend(slots::positions_of(first, tokens_of_call)?);

        // One `-1` per token whose row writes no cache page: the write decision
        // is per row, so it expands over the row's tokens here and nowhere else.
        let writes: Vec<bool> = forward
            .write_cache
            .iter()
            .zip(&forward.query_lens)
            .flat_map(|(writes, length)| std::iter::repeat_n(*writes, *length as usize))
            .collect();
        let table = placement
            .block_tables
            .iter()
            .find(|table| table.group_id == 0)?;
        rows.slot_mapping.extend(slots::slot_mapping_where(
            table,
            page,
            first..first.checked_add(tokens_of_call)?,
            |position| {
                let offset = usize::try_from(position - first).unwrap_or(usize::MAX);
                writes.get(offset).copied().unwrap_or(false)
            },
        )?);

        for seq_len in &forward.seq_lens {
            rows.seq_lens.push(i32::try_from(*seq_len).ok()?);
        }
        queries.extend_from_slice(&forward.query_lens);
        slots.extend(
            forward
                .request_pool_indices
                .iter()
                .map(|slot| i32::try_from(*slot))
                .collect::<Result<Vec<_>, _>>()
                .ok()?,
        );
        input_ids.extend_from_slice(inputs);

        let start = spans.last().map_or(0, |span| span.end);
        let span = start..start.checked_add(rows_of_call)?;
        spans.push(span.clone());
        rows.calls.push(BatchRow {
            request,
            kind,
            // One attention group today: a model with several states each
            // call's cache group in its own call, and this reads it.
            cache_group: 0,
            rows: span,
        });
    }

    rows.query_offsets = slots::query_offsets(&queries)?;
    rows.logits_rows = slots::logits_rows_of(
        requests
            .iter()
            .map(|(call, _)| (call.kind, !call.consumer_slots.is_empty())),
        &spans,
    )?;

    Some(BatchColumns {
        rows,
        input_ids,
        slots,
        // The sampler's parameters and the commit table are exec's to state
        // (`M708`); filling them from a guess would change sampling silently
        // rather than fail, so the batch is not submittable until they are set.
        parameters: Vec::new(),
        commit: Vec::new(),
        page_size: page.block_size,
        max_kv_len,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::WorkerGroupId;

    /// Four tokens per page, one unit per page: the geometry every token group
    /// reports today.
    fn page() -> PageGeometry {
        PageGeometry {
            block_size: 4,
            units_per_page: 1,
        }
    }

    /// A table covering pages 2..5 (positions 8..20) whose units are 7, 9 and
    /// 10.
    fn table() -> ipc::BlockTable {
        ipc::BlockTable {
            request_pool_idx: 1,
            group_id: 0,
            start_page: 2,
            unit_ids: vec![ipc::UnitId(7), ipc::UnitId(9), ipc::UnitId(10)],
            allocated_tokens: 20,
        }
    }

    /// A call of `kind` whose rows are `query_lens`, the first starting at
    /// `position`, with `write_cache` per row.
    fn call(kind: CallKind, position: u32, query_lens: &[u32], write_cache: bool) -> Call {
        let mut call = Call::new(
            ipc::RequestKey::new(1, ipc::RequestId(1), 1),
            ipc::CallId::new(1, 0),
            kind,
        );
        call.consumer_slots = vec![0];
        call.coordinates.logical_position = position;
        call.input_token_ids = (0..query_lens.iter().sum()).map(|_| 1u32).collect();
        call
    }

    /// The placement of a call whose rows are `query_lens`, all in request slot
    /// `slot`, writing the cache where `writes` says.
    fn placement(slot: u32, query_lens: &[u32], writes: &[bool]) -> RequestPlacement {
        let mut forward = ipc::ForwardBatch::default();
        let mut seq_len = 8;
        for (length, write) in query_lens.iter().zip(writes) {
            forward.push(0, slot, seq_len, *length, *write);
            seq_len += length;
        }
        RequestPlacement {
            worker: WorkerGroupId("w".to_owned()),
            request_pool_idx: Some(slot),
            block_tables: vec![table()],
            new_cache_units: Vec::new(),
            forward,
            latent: None,
            decode: None,
            buffers: Vec::new(),
            page_loads: Vec::new(),
            readers: None,
        }
    }

    fn prefill() -> CallKind {
        CallKind::Forward(ForwardMode::Prefill)
    }

    fn verify() -> CallKind {
        CallKind::Forward(ForwardMode::Verify)
    }

    #[test]
    fn an_empty_batch_states_its_geometry_and_nothing_else() {
        let columns = columns(&[], &[], page(), 2048).expect("an empty batch is a batch");
        assert_eq!(columns.rows(), 0);
        assert_eq!(columns.tokens(), 0);
        assert!(columns.rows.query_offsets.is_empty() || columns.rows.query_offsets == vec![0]);
        assert_eq!(columns.page_size, 4);
        assert_eq!(columns.max_kv_len, 2048);
    }

    #[test]
    fn two_prefills_are_two_row_spans_with_their_own_positions_and_slots() {
        // Call A: 4 tokens at position 8 (page 2, unit 7); call B: 2 tokens at
        // position 12 (page 3, unit 9).
        let requests = vec![
            (
                call(prefill(), 8, &[4], true),
                placement(1, &[4], &[true]),
            ),
            (
                call(prefill(), 12, &[2], true),
                placement(2, &[2], &[true]),
            ),
        ];
        let inputs = vec![vec![11, 12, 13, 14], vec![21, 22]];
        let columns = columns(&requests, &inputs, page(), 2048).expect("a prefill batch");

        assert_eq!(columns.rows(), 2, "one row per call");
        assert_eq!(columns.tokens(), 6);
        assert_eq!(
            columns.rows.calls,
            vec![
                BatchRow {
                    request: 1,
                    kind: RowKind::Prefill,
                    cache_group: 0,
                    rows: 0..1,
                },
                BatchRow {
                    request: 2,
                    kind: RowKind::Prefill,
                    cache_group: 0,
                    rows: 1..2,
                },
            ]
        );
        assert_eq!(columns.rows.positions, vec![8, 9, 10, 11, 12, 13]);
        assert_eq!(
            columns.rows.slot_mapping,
            vec![7 * 4, 7 * 4 + 1, 7 * 4 + 2, 7 * 4 + 3, 9 * 4, 9 * 4 + 1]
        );
        assert_eq!(columns.rows.seq_lens, vec![12, 10]);
        assert_eq!(columns.rows.query_offsets, vec![0, 4, 6]);
        assert_eq!(
            columns.rows.logits_rows,
            vec![-1, 5],
            "each prefill samples at its last query row"
        );
        assert_eq!(columns.input_ids, vec![11, 12, 13, 14, 21, 22]);
        assert_eq!(columns.slots, vec![1, 2]);
    }

    #[test]
    fn a_row_that_writes_nothing_states_minus_one_for_its_tokens() {
        let requests = vec![(
            call(prefill(), 8, &[2], true),
            placement(1, &[2], &[false]),
        )];
        let columns = columns(&requests, &[vec![1, 2]], page(), 64).expect("a prefill batch");
        assert_eq!(columns.rows.slot_mapping, vec![-1, -1]);
    }

    #[test]
    fn a_verify_batch_samples_every_candidate_row() {
        // Three candidate positions, one row each, all sampling.
        let requests = vec![(
            call(verify(), 9, &[1, 1, 1], true),
            placement(1, &[1, 1, 1], &[true, true, true]),
        )];
        let columns = columns(&requests, &[vec![7, 8, 9]], page(), 64).expect("a verify batch");
        assert_eq!(columns.rows(), 3);
        assert_eq!(columns.tokens(), 3);
        assert_eq!(columns.rows.positions, vec![9, 10, 11]);
        assert_eq!(
            columns.rows.logits_rows,
            vec![0, 1, 2],
            "every verify row is a candidate"
        );
        assert_eq!(columns.rows.query_offsets, vec![0, 1, 2, 3]);
        assert_eq!(
            columns.rows.calls[0].rows,
            0..3,
            "one call owns all three rows"
        );
    }

    #[test]
    fn a_media_call_is_not_a_token_batch() {
        let media = call(ipc::CallKind::Media(ipc::MediaCall::Denoising), 0, &[1], true);
        assert!(
            columns(&[(media, placement(1, &[1], &[true]))], &[vec![0]], page(), 64).is_none(),
            "a media call has no token row"
        );
    }

    #[test]
    fn a_call_that_states_fewer_tokens_than_its_rows_embed_is_refused() {
        // The decode case: its rows evaluate one token each while the call
        // states no host token (`M708` question 1). The fill never invents one.
        let decode = call(CallKind::Forward(ForwardMode::Decode), 12, &[1], true);
        let mut decode = decode;
        decode.input_token_ids.clear();
        assert!(
            columns(&[(decode, placement(1, &[1], &[true]))], &[Vec::new()], page(), 64).is_none()
        );
    }

    #[test]
    fn a_token_outside_its_table_is_a_fault_not_a_clamp() {
        // A call at position 20 lies past the table's last covered page.
        let requests = vec![(
            call(prefill(), 20, &[1], true),
            placement(1, &[1], &[true]),
        )];
        assert!(columns(&requests, &[vec![1]], page(), 64).is_none());
    }

    #[test]
    fn rows_of_two_requests_in_one_call_are_a_fault() {
        let mut placement = placement(1, &[2], &[true]);
        placement.forward.push(0, 2, 4, 1, true);
        let requests = vec![(call(prefill(), 8, &[2], true), placement)];
        assert!(columns(&requests, &[vec![1, 2]], page(), 64).is_none());
    }

    #[test]
    fn the_inputs_must_name_one_token_per_token_the_rows_embed() {
        let requests = vec![(
            call(prefill(), 8, &[2], true),
            placement(1, &[2], &[true]),
        )];
        assert!(columns(&requests, &[vec![1]], page(), 64).is_none());
        assert!(columns(&requests, &[vec![1, 2, 3]], page(), 64).is_none());
        assert!(
            columns(&requests, &[vec![1, 2], vec![3]], page(), 64).is_none(),
            "an input list per call, not one more than there are calls"
        );
    }
}
