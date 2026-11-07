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
//! - **parameters**: the sampler's columns of every row, in the order
//!   `ops::sampler` names ([`sampler_parameters`]);
//! - **input_ids**: the tokens each call's rows embed, which the caller states
//!   because a decode's token comes from the mechanism's request state and not
//!   from the call;
//! - **page_size** and **max_kv_len**: the cache's geometry.
//!
//! Three counts, one per column family, and every column follows exactly one
//! of them: a token column is as long as the batch's tokens, a row column as
//! long as its rows — the sequences, so a verification of three candidates is
//! one call and three rows — and the commit table as long as its calls. Their
//! single statement is [`executor::batch::BatchColumns`]' own documentation,
//! which the executor's `check` and its staging both read: compare a column
//! with the wrong one of the three and a legitimate batch is refused.
//!
//! - **commit**: the eight columns `ops::commit::column` names, one row per
//!   call: how the step advances the slot's device state. A decode advances by
//!   its sampled token; a prefill that samples installs its token at the host
//!   coordinates; a prefill that samples nothing only states its cache extent;
//!   a verification installs the base its accepted span advances from. The
//!   rules are the reference's (`uniserve/executor/text/commit.py`), which is
//!   where every cell of the table is decided, and nothing here is a default:
//!   a call the table cannot state — a decode or a verification that samples
//!   nothing, a row writing past its block table — answers `None`.
//!
//! A batch whose calls disagree with what the layout promises — a call whose
//! rows belong to several requests, a token outside its table, a call a token
//! batch cannot carry — answers `None`: a planning fault, never a batch that
//! would run on plausible values. The executor's own `BatchColumns::check` is
//! the second line of that defence.

use std::ops::Range;

use executor::batch::{BatchColumns, commit};
use ipc::{Call, CallKind, ForwardMode};
use nn::{BatchRow, BatchRows, RowKind};
use proto::request::SamplingParams;

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

/// The sampler's parameter columns of one row.
///
/// The order is `ops::sampler`'s, which fixes it for the kernels that read the
/// table (`PARAMETER_COLUMNS` columns per row): temperature, the top-k width,
/// the nucleus threshold, the min-p threshold, then the repetition, frequency
/// and presence penalties. The eighth column is named by no op — `ops::sampler`
/// reads only the parameters it names — so it is zero.
pub(crate) fn sampler_parameters(sampling: &SamplingParams) -> [f32; SAMPLER_PARAMETERS] {
    let mut columns = [0.0f32; SAMPLER_PARAMETERS];
    columns[0] = sampling.temperature;
    columns[1] = sampling.top_k as f32;
    columns[2] = sampling.top_p;
    columns[3] = sampling.min_p;
    columns[4] = sampling.repetition_penalty;
    columns[5] = sampling.frequency_penalty;
    columns[6] = sampling.presence_penalty;
    columns
}

/// Whether the request's token-count penalties shape its selection: the
/// reference's `SamplingParams::uses_penalties`, which the commit table states
/// per sampling row.
fn uses_penalties(sampling: &SamplingParams) -> bool {
    sampling.repetition_penalty != 1.0
        || sampling.frequency_penalty != 0.0
        || sampling.presence_penalty != 0.0
}

/// The sampler's columns per row, `ops::sampler::PARAMETER_COLUMNS`.
pub(crate) const SAMPLER_PARAMETERS: usize = 8;

/// The column of a row's parameters that states its temperature, in
/// `ops::sampler`'s order (`ops::sampler::TEMPERATURE`, which the engine does
/// not depend on directly).
const TEMPERATURE_COLUMN: usize = 0;

/// The columns of the batch formed by `requests`, whose calls' rows embed the
/// tokens `inputs` states for them and sample with the parameters `sampling`
/// states for them, all in call order.
///
/// `inputs[i]` holds the tokens of call `i`'s rows, back to back: a prefill's or
/// a verify's host tokens (`ipc::Call::input_token_ids`) or a decode's one token
/// per row. A call that states fewer tokens than its rows evaluate is a fault:
/// the mechanism never invents a token a kernel would then embed.
///
/// `draws[i]` is the request's sampling position before call `i`: the number of
/// draws it has made, which an explicit or a verification row installs as the
/// device state's sampling position.
///
/// `page` is the cache's page geometry and `max_kv_len` the attention
/// descriptor's static bound on a sequence, both of which the mechanism states
/// to the executor rather than letting each side derive them.
///
/// The slot mapping is written through the deployment's cache group: the batch
/// types one KV kind, and a deployment with more than one cache group (a hybrid
/// model's several KV kinds) would need the batch to name its group before this
/// could address the others. Until then the first group is the only one, and
/// `None` says so rather than mapping through a group the batch does not own.
/// The same holds of the page geometry the caller passes.
pub(crate) fn columns(
    requests: &[(Call, RequestPlacement)],
    tables: &[Vec<crate::scheduler::IpcBlockTable>],
    inputs: &[Vec<i32>],
    sampling: &[&SamplingParams],
    draws: &[u32],
    page: PageGeometry,
    max_kv_len: u32,
) -> Option<BatchColumns> {
    if requests.len() != inputs.len()
        || requests.len() != sampling.len()
        || requests.len() != draws.len()
        || requests.len() != tables.len()
    {
        return None;
    }
    let mut rows = BatchRows::default();
    let mut commit_rows: Vec<[i64; commit::COMMIT_FIELDS]> = Vec::with_capacity(requests.len());
    let mut tasks = 0usize;
    let mut slots = Vec::with_capacity(requests.len());
    let mut queries = Vec::new();
    let mut spans: Vec<Range<u32>> = Vec::with_capacity(requests.len());
    let mut input_ids: Vec<i32> = Vec::new();
    let mut parameters: Vec<f32> = Vec::new();
    for (index, (call, placement)) in requests.iter().enumerate() {
        let inputs = inputs.get(index)?;
        let sampling = *sampling.get(index)?;
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
        rows.positions
            .extend(slots::positions_of(first, tokens_of_call)?);

        // One `-1` per token whose row writes no cache page: the write decision
        // is per row, so it expands over the row's tokens here and nowhere else.
        let writes: Vec<bool> = forward
            .write_cache
            .iter()
            .zip(&forward.query_lens)
            .flat_map(|(writes, length)| std::iter::repeat_n(*writes, *length as usize))
            .collect();
        // The tables this call addresses, which the caller states for every
        // call: a call declares a table on the wire only when it changed
        // (`batching::call`), so reading the declaration here left every later
        // call of the same request without a layout and the executor without a
        // batch it could serve.
        let table = tables.get(index)?.iter().find(|table| table.group_id == 0)?;
        let mapped = slots::slot_mapping_where(
            table,
            page,
            first..first.checked_add(tokens_of_call)?,
            |position| {
                let offset = usize::try_from(position - first).unwrap_or(usize::MAX);
                writes.get(offset).copied().unwrap_or(false)
            },
        );
        rows.slot_mapping.extend(mapped?);

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
        // One parameter row per row of the call, the same values: the
        // request's sampling parameters apply to every row it samples.
        let row = sampler_parameters(sampling);
        for _ in 0..rows_of_call {
            parameters.extend_from_slice(&row);
        }

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

        // The commit row this call installs: how the sampling step advances the
        // slot's device state. A row names its slot, its mode, the task whose
        // sampled token it installs, the host length and positions an explicit
        // or a verification row writes, and whether its token counts penalties.
        let query = *forward.query_lens.last()?;
        // A call samples when it declares a sampled-token output: a decode and
        // a verification always do, and a prompt chunk does only when it is the
        // chunk that ends the prompt. `consumer_slots` is the executor's, filled
        // after this runs, so the mechanism's own answer is `token_output`.
        let sampled = call.token_output.is_some();
        if matches!(kind, RowKind::Decode | RowKind::Verify) && !sampled {
            return None;
        }
        let mode = match kind {
            RowKind::Decode => commit::mode::DECODE,
            RowKind::Verify => commit::mode::VERIFY,
            RowKind::Prefill if sampled => commit::mode::EXPLICIT,
            RowKind::Prefill => commit::mode::CACHE,
        };
        let explicit = mode == commit::mode::EXPLICIT;
        let verify = mode == commit::mode::VERIFY;
        // The tokens this call commits to the cache: a decode its one token,
        // a prefill the query of every row that writes, and a verification
        // none, its accepted span being the device's to install. A call that
        // would write past its block table is a planning fault, not a clamp.
        let committed = match kind {
            RowKind::Decode => 1,
            RowKind::Verify => 0,
            RowKind::Prefill => forward
                .query_lens
                .iter()
                .zip(&forward.write_cache)
                .filter(|(_, writes)| **writes)
                .try_fold(0u32, |total, (length, _)| total.checked_add(*length))?,
        };
        let visible = call.coordinates.cache_visible_len;
        let length = visible.checked_add(committed)?;
        if committed > query || (!verify && length > table.allocated_tokens) {
            return None;
        }
        let draw = *draws.get(index)?;
        let mut cells = [0i64; commit::COMMIT_FIELDS];
        cells[commit::column::SLOT] = i64::from(request);
        cells[commit::column::MODE] = mode;
        cells[commit::column::TASK] = if sampled {
            let task = i64::try_from(tasks).ok()?;
            tasks = tasks.checked_add(1)?;
            task
        } else {
            -1
        };
        // Every prefill states its cache extent, sampled or not; a decode and a
        // verification advance it on the device.
        cells[commit::column::CACHE_LENGTH] = if matches!(kind, RowKind::Prefill) {
            i64::from(length)
        } else {
            0
        };
        cells[commit::column::LOGICAL] = if explicit {
            i64::from(first).saturating_add(i64::from(query))
        } else if verify {
            i64::from(first)
        } else {
            0
        };
        cells[commit::column::SAMPLING] = if explicit {
            i64::from(draw).saturating_add(1)
        } else if verify {
            i64::from(draw)
        } else {
            0
        };
        cells[commit::column::VISIBLE] = if verify { i64::from(visible) } else { 0 };
        cells[commit::column::PENALTIES] = if sampled && uses_penalties(sampling) {
            1
        } else {
            0
        };
        commit_rows.push(cells);
    }

    // The table is column-major: one run of every column over the batch's rows.
    // Column-major over the batch's calls: `ops::commit` reads `[column][row]`,
    // and `commit::COMMIT_FIELDS` columns of one row each per call of the batch.
    let row_count = commit_rows.len();
    let mut commit = vec![0i64; commit::COMMIT_FIELDS * row_count];
    for (row, cells) in commit_rows.iter().enumerate() {
        for (column, value) in cells.iter().enumerate() {
            commit[column * row_count + row] = *value;
        }
    }

    rows.query_offsets = slots::query_offsets(&queries)?;
    rows.logits_rows = slots::logits_rows_of(
        requests
            .iter()
            .map(|(call, _)| (call.kind, call.token_output.is_some())),
        &spans,
    )?;

    Some(BatchColumns {
        rows,
        input_ids,
        slots,
        parameters,
        commit,
        page_size: page.block_size,
        max_kv_len,
    })
}

/// The captured shape a batch of these columns is served from.
///
/// `rows` and `tokens` are the columns' own counts. The variant follows the two
/// decisions the columns already state:
///
/// - a prefill projects last hidden states exactly when a row's logits are
///   wanted, which `logits_rows` names with a row index or `-1`;
/// - a decode is served greedily only when no row draws. `ops::sampler` divides
///   a temperature-zero row by one and leaves its order, and **no draw path
///   produces a greedy token** — `argmax` is a separate step of the sampler — so
///   a greedy row served from the sampler graph would sample where its request
///   asked for the largest candidate. One batch serves one variant, so a burst
///   that mixes greedy and sampling rows is sampled.
///
/// A prefill's prompt endings choose the same way: a batch whose endings are all greedy takes no
/// draw, because the draw is the expensive tail of the graph.
///
/// A verification's candidates are decode-class rows: they draw, so they are
/// sampled too.
pub(crate) fn batch_shape(columns: &BatchColumns) -> executor::plan::Shape {
    use executor::plan::Variant;

    let rows = u32::try_from(columns.rows()).unwrap_or(u32::MAX);
    let tokens = u32::try_from(columns.tokens()).unwrap_or(u32::MAX);
    let prefill = columns
        .rows
        .calls
        .iter()
        .any(|call| matches!(call.kind, RowKind::Prefill));
    let variant = if prefill {
        let mut endings = columns
            .parameters
            .chunks_exact(SAMPLER_PARAMETERS)
            .zip(&columns.rows.logits_rows)
            .filter(|(_, row)| **row >= 0)
            .peekable();
        if endings.peek().is_none() {
            Variant::PrefillWrite
        } else if endings.all(|(row, _)| row[TEMPERATURE_COLUMN] == 0.0) {
            Variant::PrefillGreedy
        } else {
            Variant::PrefillOutput
        }
    } else if columns
        .parameters
        .chunks_exact(SAMPLER_PARAMETERS)
        .all(|row| row[TEMPERATURE_COLUMN] == 0.0)
    {
        Variant::DecodeGreedy
    } else {
        Variant::DecodeSampled
    };
    executor::plan::Shape {
        rows,
        tokens,
        variant,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::ExecutorGroupId;

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
    /// The tables each call addresses, as the mechanism states them for the
    /// layout: the placement's own declaration, which is the whole table in a
    /// fixture.
    fn tables_of(requests: &[(Call, crate::executor::RequestPlacement)]) -> Vec<Vec<crate::scheduler::IpcBlockTable>> {
        requests
            .iter()
            .map(|(_, placement)| placement.block_tables.clone())
            .collect()
    }

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
    fn call(kind: CallKind, position: u32, query_lens: &[u32]) -> Call {
        let mut call = Call::new(
            ipc::RequestKey::new(1, ipc::RequestId(1), 1),
            ipc::CallId::new(1, 0),
            kind,
        );
        // A call that samples declares the output it samples into, which is
        // what the planner sets (`plan_decode`, `plan_verify`, a sampling
        // `plan_prefill`); `consumer_slots` is the executor's, filled later.
        call.token_output = Some(ipc::TensorRef {
            request_key: ipc::RequestKey::new(1, ipc::RequestId(1), 1),
            producer_call_id: ipc::CallId::new(1, 0),
            output_index: 0,
            generation: 1,
            dtype: ipc::DType::I64,
            shape_bound: Default::default(),
        });
        call.coordinates.logical_position = position;
        call.input_token_ids = (0..query_lens.iter().sum()).map(|_| 1u32).collect();
        call
    }

    /// The placement of a call whose rows are `query_lens`, all in request slot
    /// `slot`, writing the cache where `writes` says.
    fn placement(slot: u32, query_lens: &[u32], writes: &[bool]) -> RequestPlacement {
        let mut forward = ipc::ForwardBatch::default();
        // `ForwardBatch::seq_lens` is a row's total attention length, cached
        // prefix plus its own query, which is what the assembly states.
        let mut prefix = 8;
        for (length, write) in query_lens.iter().zip(writes) {
            forward.push(0, slot, prefix + length, *length, *write);
            prefix += length;
        }
        RequestPlacement {
            worker: ExecutorGroupId("w".to_owned()),
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

    /// A request's sampling parameters, distinguishable per field so a column
    /// order swapped by mistake fails a test.
    fn sampling() -> SamplingParams {
        SamplingParams {
            temperature: 0.5,
            top_k: 40,
            top_p: 0.9,
            min_p: 0.05,
            repetition_penalty: 1.1,
            frequency_penalty: 0.2,
            presence_penalty: 0.3,
            ..SamplingParams::default()
        }
    }

    fn prefill() -> CallKind {
        CallKind::Forward(ForwardMode::Prefill)
    }

    fn verify() -> CallKind {
        CallKind::Forward(ForwardMode::Verify)
    }

    /// One cell of the commit table, which is column-major over the batch's
    /// calls: a call states once how the step advances its slot, however many
    /// rows it evaluated.
    fn commit_cell(columns: &BatchColumns, column: usize, row: usize) -> i64 {
        columns.commit[column * columns.calls() + row]
    }

    fn decode() -> CallKind {
        CallKind::Forward(ForwardMode::Decode)
    }

    #[test]
    fn a_decode_advances_its_own_slot_and_states_only_its_task() {
        let mut decode_call = call(decode(), 12, &[1]);
        decode_call.coordinates.cache_visible_len = 12;
        let requests = vec![(decode_call, placement(3, &[1], &[true]))];
        let columns = columns(&requests, &tables_of(&requests), &[vec![42]], &[&sampling()], &[5], page(), 2048)
            .expect("a decode batch");

        assert_eq!(columns.commit.len(), commit::COMMIT_FIELDS * columns.rows());
        // The row's rotary position is the call's logical position, not the batch row it lands in:
        // a decode of the token at position 12 states 12, and one row of a bucket is row 0. The
        // model reads this column verbatim into its factor tables, so a row index here would put
        // every decode at RoPE angle zero and produce plausible, wrong logits.
        assert_eq!(columns.rows.positions, vec![12]);
        // The row's attention length is the placement's `ForwardBatch::seq_lens` - cached prefix plus
        // this row's queries - and not the call's position: this fixture states a prefix of 8 for a
        // call at position 12, so the column carries 9. The two are separate inputs on purpose (a
        // retained cache can be shorter than the sequence), and a live decode states them equal.
        assert_eq!(columns.rows.seq_lens, vec![9]);
        assert_eq!(columns.rows.query_offsets, vec![0, 1]);
        assert_eq!(commit_cell(&columns, commit::column::SLOT, 0), 3);
        assert_eq!(
            commit_cell(&columns, commit::column::MODE, 0),
            commit::mode::DECODE
        );
        assert_eq!(
            commit_cell(&columns, commit::column::TASK, 0),
            0,
            "a decode installs its one sampled token"
        );
        for column in [
            commit::column::CACHE_LENGTH,
            commit::column::LOGICAL,
            commit::column::SAMPLING,
            commit::column::VISIBLE,
        ] {
            assert_eq!(
                commit_cell(&columns, column, 0),
                0,
                "the device advances a decode's lengths and positions from its own state"
            );
        }
        assert_eq!(
            commit_cell(&columns, commit::column::PENALTIES, 0),
            1,
            "every penalty of this request is set"
        );
    }

    #[test]
    fn a_prefill_that_samples_installs_its_token_at_the_host_coordinates() {
        let mut prefill_call = call(prefill(), 8, &[4]);
        prefill_call.coordinates.cache_visible_len = 8;
        let requests = vec![(prefill_call, placement(1, &[4], &[true]))];
        let columns = columns(
            &requests,
            &tables_of(&requests),
            &[vec![1, 2, 3, 4]],
            &[&sampling()],
            &[2],
            page(),
            2048,
        )
        .expect("a prefill batch");

        assert_eq!(
            commit_cell(&columns, commit::column::MODE, 0),
            commit::mode::EXPLICIT
        );
        assert_eq!(
            commit_cell(&columns, commit::column::CACHE_LENGTH, 0),
            12,
            "the prefix plus the query this prefill commits"
        );
        assert_eq!(
            commit_cell(&columns, commit::column::LOGICAL, 0),
            12,
            "a sampling prefill advances its logical position by its query"
        );
        assert_eq!(
            commit_cell(&columns, commit::column::SAMPLING, 0),
            3,
            "the request's draw counter after the one draw this prefill makes"
        );
        assert_eq!(commit_cell(&columns, commit::column::VISIBLE, 0), 0);
        assert_eq!(commit_cell(&columns, commit::column::TASK, 0), 0);
        assert_eq!(commit_cell(&columns, commit::column::PENALTIES, 0), 1);
    }

    #[test]
    fn a_prefill_that_samples_nothing_still_states_its_cache_extent() {
        let mut chunk = call(prefill(), 8, &[2]);
        chunk.token_output = None;
        chunk.coordinates.cache_visible_len = 8;
        let mut neutral = sampling();
        neutral.repetition_penalty = 1.0;
        neutral.frequency_penalty = 0.0;
        neutral.presence_penalty = 0.0;
        let requests = vec![(chunk, placement(1, &[2], &[true]))];
        let columns = columns(&requests, &tables_of(&requests), &[vec![5, 6]], &[&neutral], &[7], page(), 2048)
            .expect("a chunked prefill batch");

        assert_eq!(
            commit_cell(&columns, commit::column::MODE, 0),
            commit::mode::CACHE
        );
        assert_eq!(
            commit_cell(&columns, commit::column::CACHE_LENGTH, 0),
            10,
            "a prefill writes its extent whether or not it samples"
        );
        assert_eq!(
            commit_cell(&columns, commit::column::TASK, 0),
            -1,
            "a prompt chunk samples nothing"
        );
        for column in [
            commit::column::LOGICAL,
            commit::column::SAMPLING,
            commit::column::VISIBLE,
            commit::column::PENALTIES,
        ] {
            assert_eq!(commit_cell(&columns, column, 0), 0);
        }
    }

    #[test]
    fn a_verification_states_the_base_its_accepted_span_advances_from() {
        let mut verify_call = call(verify(), 9, &[1, 1, 1]);
        verify_call.coordinates.cache_visible_len = 9;
        let requests = vec![(verify_call, placement(1, &[1, 1, 1], &[true, true, true]))];
        let columns = columns(
            &requests,
            &tables_of(&requests),
            &[vec![7, 8, 9]],
            &[&sampling()],
            &[4],
            page(),
            2048,
        )
        .expect("a verify batch");

        assert_eq!(
            commit_cell(&columns, commit::column::MODE, 0),
            commit::mode::VERIFY
        );
        assert_eq!(
            commit_cell(&columns, commit::column::VISIBLE, 0),
            9,
            "the visible base the accepted count is added to"
        );
        assert_eq!(
            commit_cell(&columns, commit::column::LOGICAL, 0),
            9,
            "a verification keeps its logical position and advances by the accepted count"
        );
        assert_eq!(
            commit_cell(&columns, commit::column::SAMPLING, 0),
            4,
            "a verification installs its base draw and draws nothing itself"
        );
        assert_eq!(
            commit_cell(&columns, commit::column::CACHE_LENGTH, 0),
            0,
            "the device states a verification's cache length from the accepted count"
        );
        assert_eq!(commit_cell(&columns, commit::column::TASK, 0), 0);
    }

    #[test]
    fn tasks_are_numbered_over_the_sampling_calls_only() {
        // A prompt chunk, then a decode, then a sampling prefill: the chunk
        // installs no token and so consumes no task.
        let mut chunk = call(prefill(), 8, &[2]);
        chunk.token_output = None;
        chunk.coordinates.cache_visible_len = 8;
        let mut decode_call = call(decode(), 10, &[1]);
        decode_call.coordinates.cache_visible_len = 10;
        let mut prefill_call = call(prefill(), 11, &[2]);
        prefill_call.coordinates.cache_visible_len = 11;
        let requests = vec![
            (chunk, placement(1, &[2], &[true])),
            (decode_call, placement(2, &[1], &[true])),
            (prefill_call, placement(3, &[2], &[true])),
        ];
        let columns = columns(
            &requests,
            &tables_of(&requests),
            &[vec![1, 2], vec![3], vec![4, 5]],
            &[&sampling(), &sampling(), &sampling()],
            &[0, 0, 0],
            page(),
            2048,
        )
        .expect("a mixed batch");

        assert_eq!(
            (0..3)
                .map(|row| commit_cell(&columns, commit::column::TASK, row))
                .collect::<Vec<_>>(),
            vec![-1, 0, 1]
        );
        assert_eq!(
            (0..3)
                .map(|row| commit_cell(&columns, commit::column::SLOT, row))
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(columns.commit.len(), commit::COMMIT_FIELDS * 3);
    }

    #[test]
    fn a_call_the_commit_cannot_state_is_no_batch_rather_than_a_guess() {
        // A decode always samples: one with no consumer has no task to name.
        let mut unsampled_decode = call(decode(), 12, &[1]);
        unsampled_decode.token_output = None;
        unsampled_decode.coordinates.cache_visible_len = 12;
        assert!(
            columns(
                &[(unsampled_decode.clone(), placement(1, &[1], &[true]))],
                &tables_of(&[(unsampled_decode.clone(), placement(1, &[1], &[true]))]),
                &[vec![1]],
                &[&sampling()],
                &[0],
                page(),
                2048,
            )
            .is_none()
        );

        // A prefill writing past the extent its block table covers is a
        // planning fault, not a clamped extent.
        let mut overlong = call(prefill(), 18, &[4]);
        overlong.coordinates.cache_visible_len = 18;
        assert!(
            columns(
                &[(overlong.clone(), placement(1, &[4], &[true]))],
                &tables_of(&[(overlong.clone(), placement(1, &[4], &[true]))]),
                &[vec![1, 2, 3, 4]],
                &[&sampling()],
                &[0],
                page(),
                2048,
            )
            .is_none()
        );
    }

    #[test]
    fn an_empty_batch_states_its_geometry_and_nothing_else() {
        let columns =
            columns(&[], &[], &[], &[], &[], page(), 2048).expect("an empty batch is a batch");
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
            (call(prefill(), 8, &[4]), placement(1, &[4], &[true])),
            (call(prefill(), 12, &[2]), placement(2, &[2], &[true])),
        ];
        let inputs = vec![vec![11, 12, 13, 14], vec![21, 22]];
        let columns = columns(
            &requests,
            &tables_of(&requests),
            &inputs,
            &[&sampling(), &sampling()],
            &[0, 0],
            page(),
            2048,
        )
        .expect("a prefill batch");

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
            vec![0, 1],
            "one entry per row, each naming the row whose logits are read: both calls end \
             their prompt, so each samples at its only row"
        );
        assert_eq!(columns.input_ids, vec![11, 12, 13, 14, 21, 22]);
        assert_eq!(columns.slots, vec![1, 2]);
    }

    #[test]
    fn a_chunk_states_no_logits_row_and_the_chunk_that_ends_the_prompt_its_own() {
        // `logits_rows` has one entry per row and names, in each, the row whose
        // hidden state is projected: `-1` for a row that projects nothing, its
        // own index for a row that samples. A prefill call is one row per
        // request, so a chunk of a prompt that does not end samples nothing and
        // the chunk that ends it samples at its own row.
        let mut chunk = call(prefill(), 8, &[3]);
        chunk.token_output = None;
        let requests = vec![
            (chunk, placement(1, &[3], &[true])),
            (call(prefill(), 11, &[2]), placement(2, &[2], &[true])),
        ];
        let columns = columns(
            &requests,
            &tables_of(&requests),
            &[vec![1, 2, 3], vec![4, 5]],
            &[&sampling(), &sampling()],
            &[0, 0],
            page(),
            2048,
        )
        .expect("a chunk and the chunk that ends a prompt");

        assert_eq!(columns.rows(), 2, "one row per call");
        assert_eq!(columns.tokens(), 5);
        assert_eq!(
            columns.rows.logits_rows,
            vec![-1, 1],
            "the chunk projects nothing, and the prompt's last chunk projects its own row"
        );
        assert_eq!(
            batch_shape(&columns).variant,
            executor::plan::Variant::PrefillOutput,
            "the batch ends a prompt, so it projects logits"
        );
    }

    #[test]
    fn a_filled_batch_states_one_sampler_row_per_row_and_passes_the_executors_check() {
        let requests = vec![(call(prefill(), 8, &[4]), placement(1, &[4], &[true]))];
        let columns = columns(
            &requests,
            &tables_of(&requests),
            &[vec![1, 2, 3, 4]],
            &[&sampling()],
            &[0],
            page(),
            2048,
        )
        .expect("a prefill batch");
        assert_eq!(columns.parameters.len(), SAMPLER_PARAMETERS);
        assert_eq!(
            &columns.parameters[..7],
            &[0.5, 40.0, 0.9, 0.05, 1.1, 0.2, 0.3],
            "ops::sampler's order: temperature, top-k, top-p, min-p, repetition, frequency, presence"
        );
        let shape = executor::plan::Shape::prefill(
            columns.rows() as u32,
            columns.tokens() as u32,
            executor::plan::Variant::PrefillWrite,
        );
        columns
            .check(&shape)
            .expect("the fill's columns are a well-formed batch of its shape");
    }

    #[test]
    fn a_prefills_shape_follows_whether_a_row_projects_logits() {
        // A prefill that samples projects its last row's hidden state; one that
        // writes only the cache leaves every logits row unwritten.
        let requests = vec![(call(prefill(), 8, &[4]), placement(1, &[4], &[true]))];
        let sampling_batch = columns(
            &requests,
            &tables_of(&requests),
            &[vec![1, 2, 3, 4]],
            &[&sampling()],
            &[0],
            page(),
            2048,
        )
        .expect("a sampling prefill");
        let shape = batch_shape(&sampling_batch);
        assert_eq!(shape.variant, executor::plan::Variant::PrefillOutput);
        assert_eq!(
            (shape.rows, shape.tokens),
            (sampling_batch.rows() as u32, sampling_batch.tokens() as u32),
            "the shape's counts are the columns' own"
        );
        sampling_batch
            .check(&shape)
            .expect("a sampling prefill is a well-formed PrefillOutput batch");

        let mut chunk = call(prefill(), 8, &[4]);
        chunk.token_output = None;
        let requests = vec![(chunk, placement(1, &[4], &[true]))];
        let writing_batch = columns(
            &requests,
            &tables_of(&requests),
            &[vec![1, 2, 3, 4]],
            &[&sampling()],
            &[0],
            page(),
            2048,
        )
        .expect("a chunked prefill");
        let shape = batch_shape(&writing_batch);
        assert_eq!(shape.variant, executor::plan::Variant::PrefillWrite);
        writing_batch
            .check(&shape)
            .expect("a chunked prefill is a well-formed PrefillWrite batch");
    }

    #[test]
    fn a_batch_is_served_greedily_only_when_no_row_draws() {
        let greedy = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        let decode = CallKind::Forward(ForwardMode::Decode);
        let requests = vec![(call(decode, 12, &[1]), placement(1, &[1], &[true]))];
        let one_greedy = columns(&requests, &tables_of(&requests), &[vec![9]], &[&greedy], &[0], page(), 2048)
            .expect("a greedy decode");
        assert_eq!(
            batch_shape(&one_greedy).variant,
            executor::plan::Variant::DecodeGreedy,
            "no draw path produces a greedy token, so a greedy row is served from the greedy bucket"
        );

        let one_sampling = columns(&requests, &tables_of(&requests), &[vec![9]], &[&sampling()], &[0], page(), 2048)
            .expect("a sampling decode");
        assert_eq!(
            batch_shape(&one_sampling).variant,
            executor::plan::Variant::DecodeSampled
        );

        // One batch serves one variant, so a burst that mixes the two is
        // sampled even though one of its rows is greedy.
        let mut second = call(decode, 12, &[1]);
        second.request_key = ipc::RequestKey::new(2, ipc::RequestId(2), 1);
        second.call_id = ipc::CallId::new(2, 1);
        let requests = vec![
            (call(decode, 12, &[1]), placement(1, &[1], &[true])),
            (second, placement(2, &[1], &[true])),
        ];
        let mixed = columns(
            &requests,
            &tables_of(&requests),
            &[vec![9], vec![9]],
            &[&greedy, &sampling()],
            &[0, 0],
            page(),
            2048,
        )
        .expect("a mixed decode");
        assert_eq!(
            batch_shape(&mixed).variant,
            executor::plan::Variant::DecodeSampled,
            "the sampler graph serves both rows, and a greedy row's token is its argmax"
        );

        // A verification is served from a decode bucket, and which one follows
        // the same rule: a greedy verification argmaxes each candidate row, and
        // one whose rows draw takes the sampler graph.
        let requests = vec![(
            call(verify(), 12, &[1, 1]),
            placement(1, &[1, 1], &[true, true]),
        )];
        let greedy_verify = columns(&requests, &tables_of(&requests), &[vec![7, 8]], &[&greedy], &[0], page(), 2048)
            .expect("a greedy verification");
        assert_eq!(
            batch_shape(&greedy_verify).variant,
            executor::plan::Variant::DecodeGreedy
        );
        let sampling_verify = columns(&requests, &tables_of(&requests), &[vec![7, 8]], &[&sampling()], &[0], page(), 2048)
            .expect("a sampling verification");
        assert_eq!(
            batch_shape(&sampling_verify).variant,
            executor::plan::Variant::DecodeSampled
        );
    }

    #[test]
    fn a_row_that_writes_nothing_states_minus_one_for_its_tokens() {
        let requests = vec![(call(prefill(), 8, &[2]), placement(1, &[2], &[false]))];
        let columns = columns(&requests, &tables_of(&requests), &[vec![1, 2]], &[&sampling()], &[0], page(), 64)
            .expect("a prefill batch");
        assert_eq!(columns.rows.slot_mapping, vec![-1, -1]);
    }

    #[test]
    fn a_verify_batch_samples_every_candidate_row() {
        // Three candidate positions, one row each, all sampling.
        let requests = vec![(
            call(verify(), 9, &[1, 1, 1]),
            placement(1, &[1, 1, 1], &[true, true, true]),
        )];
        let columns = columns(
            &requests,
            &tables_of(&requests),
            &[vec![7, 8, 9]],
            &[&sampling()],
            &[0],
            page(),
            64,
        )
        .expect("a verify batch");
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
        // The batch's two row counts stay distinct: the sampler's slots and
        // parameters are per row, the commit table's are per call.
        assert_eq!(columns.calls(), 1, "one call evaluates three rows");
        assert_eq!(columns.slots.len(), 3);
        assert_eq!(columns.commit.len(), commit::COMMIT_FIELDS);
        // A verification is served from a decode bucket: one token per row,
        // three candidate rows, and the sampler graph because they draw.
        columns
            .check(&executor::plan::Shape::decode(
                3,
                executor::plan::Variant::DecodeSampled,
            ))
            .expect("the fill satisfies the executor's check for a multi-row call");
    }

    #[test]
    fn a_media_call_is_not_a_token_batch() {
        let media = call(ipc::CallKind::Media(ipc::MediaCall::Denoising), 0, &[1]);
        assert!(
            columns(
                &[(media.clone(), placement(1, &[1], &[true]))],
                &tables_of(&[(media.clone(), placement(1, &[1], &[true]))]),
                &[vec![0]],
                &[&sampling()],
                &[0],
                page(),
                64
            )
            .is_none(),
            "a media call has no token row"
        );
    }

    #[test]
    fn a_call_that_states_fewer_tokens_than_its_rows_embed_is_refused() {
        // The decode case: its rows evaluate one token each while the call
        // states no host token (`M708` question 1). The fill never invents one.
        let decode = call(CallKind::Forward(ForwardMode::Decode), 12, &[1]);
        let mut decode = decode;
        decode.input_token_ids.clear();
        assert!(
            columns(
                &[(decode.clone(), placement(1, &[1], &[true]))],
                &tables_of(&[(decode.clone(), placement(1, &[1], &[true]))]),
                &[Vec::new()],
                &[&sampling()],
                &[0],
                page(),
                64
            )
            .is_none()
        );
    }

    #[test]
    fn a_token_outside_its_table_is_a_fault_not_a_clamp() {
        // A call at position 20 lies past the table's last covered page.
        let requests = vec![(call(prefill(), 20, &[1]), placement(1, &[1], &[true]))];
        assert!(columns(&requests, &tables_of(&requests), &[vec![1]], &[&sampling()], &[0], page(), 64).is_none());
    }

    #[test]
    fn rows_of_two_requests_in_one_call_are_a_fault() {
        let mut placement = placement(1, &[2], &[true]);
        placement.forward.push(0, 2, 4, 1, true);
        let requests = vec![(call(prefill(), 8, &[2]), placement)];
        assert!(columns(&requests, &tables_of(&requests), &[vec![1, 2]], &[&sampling()], &[0], page(), 64).is_none());
    }

    #[test]
    fn the_inputs_must_name_one_token_per_token_the_rows_embed() {
        let requests = vec![(call(prefill(), 8, &[2]), placement(1, &[2], &[true]))];
        assert!(columns(&requests, &tables_of(&requests), &[vec![1]], &[&sampling()], &[0], page(), 64).is_none());
        assert!(
            columns(
                &requests,
                &tables_of(&requests),
                &[vec![1, 2, 3]],
                &[&sampling()],
                &[0],
                page(),
                64
            )
            .is_none()
        );
        assert!(
            columns(
                &requests,
                &tables_of(&requests),
                &[vec![1, 2], vec![3]],
                &[&sampling(), &sampling()],
                &[0, 0],
                page(),
                64
            )
            .is_none(),
            "an input list per call, not one more than there are calls"
        );
    }
}
