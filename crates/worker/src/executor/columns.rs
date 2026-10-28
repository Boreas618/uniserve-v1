//! The step a batch of a computation that does not prepare hands the executor
//! ([`ColumnStep`]): host columns, a small JSON and the batch's output words,
//! into which the executor copies its device values, with a host tensor it
//! writes its completion records into (`crate::executor::records`).
//!
//! The columns are one column of [`CALL_FIELDS`] integers per call (`int64
//! [CALL_FIELDS, C]`), the scheduler's forward rows grouped by call with
//! their attention pages (`int64 [FORWARD_FIELDS + 2G, R]`), each laid out
//! field-major, so a field is one contiguous row the executor indexes by
//! call or by row, and the calls' input tokens back to back (`int64 [N]`).
//! The executor recomputes nothing these state: a call's slot, kind,
//! coordinates and accepted progress, its rows and tokens, which of its
//! outputs exist; a row's slot, lengths and, per cache group `g`, the first
//! page it reads and how many (`page_start_g`, `page_count_g`, from
//! `PrefixCache::prepare_attention`). The JSON holds only what a call
//! carries beside them: its sampling state, the features it injects
//! (leased like every read of the batch) and where a readout's slots and
//! candidates lie among the tokens.
//!
//! `uniserve.executor.text.protocol` names the fields.

use std::ops::Range;
use std::sync::Arc;

use ipc::{Batch as BatchPlan, Call, CallKind, DType, ForwardBatch, ForwardMode, TensorRef};

use crate::executor::batch::Batch;
use crate::executor::computations::CallResources;
use crate::executor::pages::PageReservation;
use crate::executor::records::{self, CallLimits};
use crate::ffi::{Backing, DType as Element};
use crate::job::Recorder;
use crate::memory::Metadata;
use crate::memory::allocation::{Storage, StorageView};
use crate::request::RequestProgress;
use crate::{Error, ProductStore, Result, lock};

/// Fields of one call's column.
pub(crate) const CALL_FIELDS: usize = 22;

/// The call fields' names in row order, which the executor's manifest
/// echoes (`uniserve.executor.text.protocol.CALL_FIELD_NAMES`).
pub(crate) const CALL_FIELD_NAMES: [&str; CALL_FIELDS] = [
    "slot",
    "kind",
    "active",
    "logical",
    "visible",
    "computed",
    "rng",
    "prompt_ready",
    "allocated",
    "predicate",
    "row_start",
    "row_count",
    "token_start",
    "token_count",
    "max_tokens",
    "flags",
    "canvas_block",
    "canvas_step",
    "request_id",
    "epoch",
    "proposals",
    "lookahead",
];

/// Fields of one forward row before its page columns.
pub(crate) const FORWARD_FIELDS: usize = 5;

/// Field indices of the call columns.
pub(crate) mod field {
    /// The call's request slot.
    pub(crate) const SLOT: usize = 0;
    /// 0 prefill, 1 decode, 2 verify, 3 token denoising.
    pub(crate) const KIND: usize = 1;
    /// Whether the call executes: its request was not refused. A false
    /// predicate gates the call on the device instead (`PREDICATE`).
    pub(crate) const ACTIVE: usize = 2;
    /// The call's coordinates: its first token's logical position, the
    /// visible and computed prefix cache lengths.
    pub(crate) const LOGICAL: usize = 3;
    pub(crate) const VISIBLE: usize = 4;
    pub(crate) const COMPUTED: usize = 5;
    /// Draws the request's sampling stream made, as accepted.
    pub(crate) const RNG: usize = 6;
    /// Whether the request's retained prompt logits are ready, as accepted.
    pub(crate) const PROMPT_READY: usize = 7;
    /// The tokens the slot's block tables hold once the batch installed
    /// them; zero for a call that does not execute.
    pub(crate) const ALLOCATED: usize = 8;
    /// 0 none, 1 a `uint8` predicate the executor gates the call on, on the
    /// device: the feature the call's JSON names (`predicate.tensor`), or a
    /// token-denoising step's, its slot state; 2 an `int64` relayed
    /// continuation, which the executor gates on its slot state.
    pub(crate) const PREDICATE: usize = 9;
    /// The call's rows among the forward columns, which hold each call's
    /// rows together in row order.
    pub(crate) const ROW_START: usize = 10;
    pub(crate) const ROW_COUNT: usize = 11;
    /// The call's input tokens in the flat token column.
    pub(crate) const TOKEN_START: usize = 12;
    pub(crate) const TOKEN_COUNT: usize = 13;
    pub(crate) const MAX_TOKENS: usize = 14;
    /// [`super::flag`] bits.
    pub(crate) const FLAGS: usize = 15;
    /// The canvas step a token-denoising step runs, -1 for other calls.
    pub(crate) const CANVAS_BLOCK: usize = 16;
    pub(crate) const CANVAS_STEP: usize = 17;
    /// The call's request, which the executor matches with the slot's
    /// started request.
    pub(crate) const REQUEST_ID: usize = 18;
    pub(crate) const EPOCH: usize = 19;
    /// Draft tokens the call proposes for its request's next verification
    /// (`ipc::DraftWork::proposals`).
    pub(crate) const PROPOSALS: usize = 20;
    /// The token after the call's last input token, which its last draft
    /// cache entry reads; -1 for none (`ipc::DraftWork::lookahead`).
    pub(crate) const LOOKAHEAD: usize = 21;
}

/// Bits of a call's FLAGS field: which outputs the call declares, which
/// state it writes, and which entries its JSON holds.
pub(crate) mod flag {
    pub(crate) const TOKEN_OUTPUT: i64 = 1;
    pub(crate) const TRANSITION_OUTPUT: i64 = 1 << 1;
    pub(crate) const COMPLETION_OUTPUT: i64 = 1 << 2;
    /// The call is a context prefill placing vision blocks.
    pub(crate) const WRITES_CONTEXT: i64 = 1 << 3;
    /// The call writes one visual-state row (image feedback or a latent).
    pub(crate) const WRITES_VISUAL_STATE: i64 = 1 << 4;
    /// The JSON holds the call's `sampling_state`.
    pub(crate) const SAMPLING_STATE: i64 = 1 << 5;
    /// The JSON holds the call's `vision` or `latent` features.
    pub(crate) const FEATURES: i64 = 1 << 6;
    /// The JSON holds the call's `readout`.
    pub(crate) const READOUT: i64 = 1 << 7;
    /// The call refills its request's draft cache
    /// (`ipc::DraftWork::refill`): it evaluates tokens the cache holds,
    /// writes no target K/V and keeps no per-request state.
    pub(crate) const DRAFT_REFILL: i64 = 1 << 8;
}

/// The value of a call's KIND field.
pub(crate) fn kind_code(kind: CallKind) -> Result<i64> {
    match kind {
        CallKind::Forward(ForwardMode::Prefill) => Ok(0),
        CallKind::Forward(ForwardMode::Decode) => Ok(1),
        CallKind::Forward(ForwardMode::Verify) => Ok(2),
        CallKind::Forward(ForwardMode::TokenDenoising) => Ok(3),
        other => Err(Error::Invalid(format!(
            "{} is no token call",
            other.as_str()
        ))),
    }
}

/// What one call states in its column beside the call itself.
pub(crate) struct CallFacts<'a> {
    pub(crate) call: &'a Call,
    pub(crate) slot: usize,
    pub(crate) active: bool,
    /// The call's progress before it runs: its coordinates, with the
    /// request's accepted draws and prompt logits.
    pub(crate) progress: RequestProgress,
    pub(crate) allocated: u32,
    /// The call's rows among the grouped forward rows.
    pub(crate) rows: Range<usize>,
    /// The call's input tokens in the flat token column.
    pub(crate) tokens: Range<usize>,
    /// The [`flag`] bits of the JSON entries the batch states for it.
    pub(crate) entries: i64,
}

/// One call's column.
pub(crate) fn call_column(facts: &CallFacts<'_>) -> Result<[i64; CALL_FIELDS]> {
    let call = facts.call;
    let wide = |value: u64, name: &str| {
        i64::try_from(value).map_err(|_| Error::Invalid(format!("a call's {name} exceeds int64")))
    };
    let mut flags = facts.entries;
    for (present, bit) in [
        (call.token_output.is_some(), flag::TOKEN_OUTPUT),
        (call.transition_output.is_some(), flag::TRANSITION_OUTPUT),
        (call.completion_output.is_some(), flag::COMPLETION_OUTPUT),
        (call.writes_context(), flag::WRITES_CONTEXT),
        (call.writes_visual_state(), flag::WRITES_VISUAL_STATE),
        (call.draft.refill.is_some(), flag::DRAFT_REFILL),
    ] {
        if present {
            flags |= bit;
        }
    }
    let predicate = match call.predicate.as_ref().map(|predicate| predicate.dtype) {
        None => 0,
        Some(DType::U8) => 1,
        Some(DType::I64) => 2,
        Some(other) => {
            return Err(Error::Invalid(format!(
                "a call predicate is uint8 or int64, not {other:?}"
            )));
        }
    };
    let progress = facts.progress;
    let mut column = [0; CALL_FIELDS];
    column[field::SLOT] = facts.slot as i64;
    column[field::KIND] = kind_code(call.kind)?;
    column[field::ACTIVE] = i64::from(facts.active);
    column[field::LOGICAL] = wide(progress.logical_position, "logical position")?;
    column[field::VISIBLE] = wide(progress.cache_visible_len, "visible length")?;
    column[field::COMPUTED] = wide(progress.cache_computed_len, "computed length")?;
    column[field::RNG] = wide(progress.rng_counter, "draw count")?;
    column[field::PROMPT_READY] = i64::from(progress.prompt_logits_ready);
    column[field::ALLOCATED] = i64::from(facts.allocated);
    column[field::PREDICATE] = predicate;
    column[field::ROW_START] = facts.rows.start as i64;
    column[field::ROW_COUNT] = facts.rows.len() as i64;
    column[field::TOKEN_START] = facts.tokens.start as i64;
    column[field::TOKEN_COUNT] = facts.tokens.len() as i64;
    column[field::MAX_TOKENS] = i64::from(call.bounds.max_tokens);
    column[field::FLAGS] = flags;
    column[field::CANVAS_BLOCK] = call.canvas.map_or(-1, |step| i64::from(step.block));
    column[field::CANVAS_STEP] = call.canvas.map_or(-1, |step| i64::from(step.step));
    column[field::REQUEST_ID] = wide(call.request_key.request_id.0, "request id")?;
    column[field::EPOCH] = wide(call.request_key.request_epoch, "request epoch")?;
    column[field::PROPOSALS] = i64::from(call.draft.proposals);
    column[field::LOOKAHEAD] = call.draft.lookahead.map_or(-1, i64::from);
    Ok(column)
}

/// Columns laid out field-major: field `f` of column `c` at `f * C + c`.
pub(crate) fn field_major<const F: usize>(columns: &[[i64; F]]) -> Vec<i64> {
    let count = columns.len();
    let mut laid = vec![0; F * count];
    for (index, column) in columns.iter().enumerate() {
        for (field, &value) in column.iter().enumerate() {
            laid[field * count + index] = value;
        }
    }
    laid
}

/// A host `int64` tensor of `shape` holding `values`, for the executor.
pub(crate) fn host_tensor(values: &[i64], shape: &[i64]) -> Result<StorageView> {
    let memory = Storage::host(size_of_val(values).max(size_of::<i64>()));
    // SAFETY: fresh host memory of at least the values' bytes, which
    // nothing else references yet.
    unsafe {
        std::ptr::copy_nonoverlapping(values.as_ptr(), memory.address() as *mut i64, values.len());
    }
    StorageView::contiguous(memory as Arc<dyn Backing>, 0, Element::I64, shape)
}

/// What a batch hands the executor, and what checks the records it writes.
pub(crate) struct ColumnStep {
    /// The step's JSON.
    pub(crate) step: serde_json::Value,
    /// In protocol order: the call columns, the forward columns, the
    /// tokens, the records, the lent output words, then each feature.
    pub(crate) tensors: Vec<StorageView>,
    /// Record words the records tensor holds.
    pub(crate) capacity: usize,
    /// What each call's record is checked against.
    pub(crate) limits: Vec<CallLimits>,
    /// The lease's first word lent to the executor, and how many it lent.
    pub(crate) lent: usize,
    pub(crate) lent_words: usize,
}

impl ColumnStep {
    /// The records' position among [`Self::tensors`].
    const RECORDS: usize = 3;

    /// The tensor the executor writes its records into.
    pub(crate) fn records(&self) -> &StorageView {
        &self.tensors[Self::RECORDS]
    }
}

/// Build computation `computation`'s step of `batch`, whose pages `reservation`
/// holds: lease the active calls' features on `issue`'s stream behind their
/// producers (they end with the batch's other reads), lay out the columns and
/// tokens, allocate the records and lend the executor every output word the
/// lease has left.
pub(crate) fn column_step(
    tensors: &ProductStore,
    plan: &BatchPlan,
    computation: usize,
    batch: &mut Batch<CallResources>,
    reservation: &PageReservation,
    issue: Recorder<'_>,
) -> Result<ColumnStep> {
    let forward = &plan.forward;
    let memory = batch.target();

    // The features of vision blocks and latent conditioning.
    let mut features: Vec<StorageView> = Vec::new();
    let mut entries: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    let mut lease = |reference: &TensorRef, call: &Call| -> Result<(usize, u32, u32)> {
        let read = tensors.consume(reference, call.call_id, Some(memory), issue)?;
        batch.reads.push(Arc::clone(&read));
        let Some(Metadata::Feature { height, width }) = read.metadata() else {
            return Err(Error::Invalid(
                "visual input requires encoder feature metadata".into(),
            ));
        };
        features.push(read.value().clone());
        Ok((features.len() - 1, height, width))
    };
    let mut bits = vec![0_i64; plan.calls.len()];
    for (index, call) in plan.calls.iter().enumerate() {
        if !batch.calls[index].active {
            continue;
        }
        let mut entry = serde_json::Map::new();
        if let Some(state) = &call.sampling_state {
            entry.insert(
                "sampling_state".into(),
                serde_json::to_value(state).map_err(|error| {
                    Error::Invalid(format!("encoding a sampling state: {error}"))
                })?,
            );
            bits[index] |= flag::SAMPLING_STATE;
        }
        if !call.vision_inputs.is_empty() {
            let mut vision = Vec::with_capacity(call.vision_inputs.len());
            for input in &call.vision_inputs {
                let (tensor, height, width) = lease(&input.feature, call)?;
                vision.push(serde_json::json!({
                    "offset": input.offset,
                    "tensor": tensor,
                    "height": height,
                    "width": width,
                }));
            }
            entry.insert("vision".into(), serde_json::Value::Array(vision));
            bits[index] |= flag::FEATURES;
        }
        if let Some(latent) = &call.latent_feature_input {
            let (tensor, height, width) = lease(latent, call)?;
            entry.insert(
                "latent".into(),
                serde_json::json!({"tensor": tensor, "height": height, "width": width}),
            );
            bits[index] |= flag::FEATURES;
        }
        if !entry.is_empty() {
            entries.insert(index.to_string(), serde_json::Value::Object(entry));
        }
    }

    // The leased `uint8` predicates (`Batch::lease_predicates`), which gate
    // their calls on the device.
    for (index, state) in batch.calls.iter().enumerate() {
        let Some(predicate) = state.predicate.clone() else {
            continue;
        };
        features.push(predicate);
        let entry = entries
            .entry(index.to_string())
            .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
        entry["predicate"] = serde_json::json!({"tensor": features.len() - 1});
    }

    // The flat tokens: each active call's input tokens, then each readout's
    // slot tokens, candidate offsets and candidate ids, `[start, count]` of
    // each in its JSON.
    let mut tokens: Vec<i64> = Vec::new();
    let mut spans = vec![0..0; plan.calls.len()];
    for (index, call) in plan.calls.iter().enumerate() {
        let start = tokens.len();
        if batch.calls[index].active {
            tokens.extend(call.input_token_ids.iter().map(|&token| i64::from(token)));
        }
        spans[index] = start..tokens.len();
    }
    for (index, call) in plan.calls.iter().enumerate() {
        let Some(readout) = call.readout.as_ref().filter(|_| batch.calls[index].active) else {
            continue;
        };
        let mut place = |values: &[u32]| {
            let start = tokens.len();
            tokens.extend(values.iter().map(|&value| i64::from(value)));
            serde_json::json!([start, values.len()])
        };
        let located = serde_json::json!({
            "slot_tokens": place(&readout.slot_tokens),
            "candidate_offsets": place(&readout.candidate_offsets),
            "candidate_ids": place(&readout.candidate_ids),
        });
        let entry = entries
            .entry(index.to_string())
            .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
        entry["readout"] = located;
        bits[index] |= flag::READOUT;
    }

    // The call and forward columns.
    let rows = &reservation.rows;
    let mut columns = Vec::with_capacity(plan.calls.len());
    let mut limits = Vec::with_capacity(plan.calls.len());
    for (index, (call, state)) in plan.calls.iter().zip(&batch.calls).enumerate() {
        columns.push(call_column(&CallFacts {
            call,
            slot: state.slot,
            active: state.active,
            progress: state.output.progress,
            allocated: state.forward.pages.allocated,
            rows: rows.span(index),
            tokens: spans[index].clone(),
            entries: bits[index],
        })?);
        limits.push(CallLimits {
            active: state.active,
            logical: state.output.progress.logical_position,
            rng: state.output.progress.rng_counter,
            allocated: u64::from(state.forward.pages.allocated),
            max_tokens: u64::from(call.bounds.max_tokens),
            proposals: u64::from(call.draft.proposals),
        });
    }
    let calls = plan.calls.len();
    let row_count = forward.call_indices.len();
    let capacity = records::capacity(calls, tokens.len());
    let (lent, words) = lock(&batch.buffer).lend_words()?;
    let lent_words = words.shape()[0] as usize;
    let mut tensors = vec![
        host_tensor(&field_major(&columns), &[CALL_FIELDS as i64, calls as i64])?,
        host_tensor(
            &forward_columns(forward, rows, &reservation.pages),
            &[
                (FORWARD_FIELDS + 2 * reservation.pages.len()) as i64,
                row_count as i64,
            ],
        )?,
        host_tensor(&tokens, &[tokens.len() as i64])?,
        host_tensor(&vec![0; capacity], &[capacity as i64])?,
        words,
    ];
    tensors.extend(features);
    Ok(ColumnStep {
        step: serde_json::json!({
            "batch": plan.batch_id,
            "computation": computation,
            "collective_seq": plan.collective_seq,
            "calls": entries,
        }),
        tensors,
        capacity,
        limits,
        lent,
        lent_words,
    })
}

/// A batch plan's forward rows grouped by the call they belong to, each
/// call's rows in row order: a counting sort of the rows' call indices.
pub(crate) struct ForwardRows {
    /// Row indices, call by call.
    rows: Vec<usize>,
    /// Where each call's rows start in `rows`, then where the last call's
    /// end.
    starts: Vec<usize>,
}

impl ForwardRows {
    /// Group the rows of `call_indices` among `calls` calls. A row naming
    /// no call is invalid.
    pub(crate) fn new(call_indices: &[u32], calls: usize) -> Result<Self> {
        // Each call's row count at `starts[call + 1]`, summed into where
        // each call's rows start.
        let mut starts = vec![0; calls + 1];
        for &call in call_indices {
            let call = call as usize;
            if call >= calls {
                return Err(Error::Invalid("forward row names no call".into()));
            }
            starts[call + 1] += 1;
        }
        for call in 0..calls {
            starts[call + 1] += starts[call];
        }
        // Placing a call's rows advances its start to the next call's;
        // shifting the starts up by one restores them.
        let mut rows = vec![0; call_indices.len()];
        for (row, &call) in call_indices.iter().enumerate() {
            let next = &mut starts[call as usize];
            rows[*next] = row;
            *next += 1;
        }
        starts.copy_within(0..calls, 1);
        starts[0] = 0;
        Ok(Self { rows, starts })
    }

    /// The forward rows of call `call`, in row order.
    pub(crate) fn of(&self, call: usize) -> &[usize] {
        &self.rows[self.starts[call]..self.starts[call + 1]]
    }

    /// Where call `call`'s rows lie among the grouped rows.
    pub(crate) fn span(&self, call: usize) -> Range<usize> {
        self.starts[call]..self.starts[call + 1]
    }

    /// Every row, call by call.
    pub(crate) fn order(&self) -> &[usize] {
        &self.rows
    }
}

/// The pages one cache group's tables select for each forward row, by the
/// row's index in the plan: zero pages for a row nothing prepared (a
/// skipped call's).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct GroupPages {
    pub(crate) starts: Vec<i64>,
    pub(crate) counts: Vec<i64>,
}

impl GroupPages {
    /// No pages for any of `rows` rows.
    pub(crate) fn empty(rows: usize) -> Self {
        Self {
            starts: vec![0; rows],
            counts: vec![0; rows],
        }
    }
}

/// The forward columns `[FORWARD_FIELDS + 2G, R]`: each row's call index,
/// request slot, sequence and query lengths and cache write, then per cache
/// group its first page and page count, with the rows in `rows`' grouped
/// order.
pub(crate) fn forward_columns(
    forward: &ForwardBatch,
    rows: &ForwardRows,
    pages: &[GroupPages],
) -> Vec<i64> {
    let order = rows.order();
    let fields: Vec<[i64; FORWARD_FIELDS]> = order
        .iter()
        .map(|&row| {
            [
                i64::from(forward.call_indices[row]),
                i64::from(forward.request_pool_indices[row]),
                i64::from(forward.seq_lens[row]),
                i64::from(forward.query_lens[row]),
                i64::from(forward.write_cache[row]),
            ]
        })
        .collect();
    let mut laid = field_major(&fields);
    // Field-major, each group's page field is one more row of the columns.
    for group in pages {
        laid.extend(order.iter().map(|&row| group.starts[row]));
        laid.extend(order.iter().map(|&row| group.counts[row]));
    }
    laid
}

#[cfg(test)]
mod tests {
    use ipc::{CallId, CanvasStep, RequestId, RequestKey, ShapeBound, TensorRef};

    use super::*;

    #[test]
    fn forward_rows_group_each_calls_rows_in_row_order() -> Result<()> {
        // Rows of calls 2, 0, 2, 3, 0 among four calls; call 1 has none.
        let rows = ForwardRows::new(&[2, 0, 2, 3, 0], 4)?;
        assert_eq!(rows.of(0), &[1, 4]);
        assert!(rows.of(1).is_empty());
        assert_eq!(rows.of(2), &[0, 2]);
        assert_eq!(rows.of(3), &[3]);
        assert_eq!(rows.span(2), 2..4);
        assert_eq!(rows.order(), &[1, 4, 0, 2, 3]);

        assert!(ForwardRows::new(&[0, 4], 4).is_err());
        Ok(())
    }

    fn reference(dtype: DType) -> TensorRef {
        TensorRef {
            request_key: RequestKey::new(1, RequestId(9), 4),
            producer_call_id: CallId::new(1, 0),
            output_index: 0,
            generation: 1,
            dtype,
            shape_bound: ShapeBound::default(),
        }
    }

    /// A call's column states its slot, kind, coordinates, accepted
    /// progress, rows, tokens and outputs; the columns of a batch lie
    /// field by field.
    #[test]
    fn call_columns_lie_field_by_field() -> Result<()> {
        let mut decode = Call::new(
            RequestKey::new(1, RequestId(9), 4),
            CallId::new(2, 0),
            CallKind::Forward(ForwardMode::Decode),
        );
        decode.bounds.max_tokens = 1;
        decode.token_output = Some(reference(DType::I64));
        decode.predicate = Some(reference(DType::I64));
        let mut step = Call::new(
            RequestKey::new(1, RequestId(11), 2),
            CallId::new(2, 1),
            CallKind::Forward(ForwardMode::TokenDenoising),
        );
        step.canvas = Some(CanvasStep { block: 3, step: 1 });
        step.completion_output = Some(reference(DType::U8));
        step.predicate = Some(reference(DType::U8));
        let progress = RequestProgress {
            logical_position: 40,
            rng_counter: 7,
            cache_visible_len: 39,
            cache_computed_len: 39,
            prompt_logits_ready: true,
        };

        let first = call_column(&CallFacts {
            call: &decode,
            slot: 5,
            active: true,
            progress,
            allocated: 48,
            rows: 0..1,
            tokens: 0..0,
            entries: flag::SAMPLING_STATE,
        })?;
        assert_eq!(
            first,
            [
                5,
                1,
                1,
                40,
                39,
                39,
                7,
                1,
                48,
                2,
                0,
                1,
                0,
                0,
                1,
                1 | (1 << 5),
                -1,
                -1,
                9,
                4,
                0,
                -1
            ]
        );
        let second = call_column(&CallFacts {
            call: &step,
            slot: 2,
            active: false,
            progress: RequestProgress::default(),
            allocated: 0,
            rows: 1..2,
            tokens: 0..4,
            entries: 0,
        })?;
        assert_eq!(
            second,
            [
                2,
                3,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                0,
                4,
                0,
                1 << 2,
                3,
                1,
                11,
                2,
                0,
                -1
            ]
        );

        let laid = field_major(&[first, second]);
        assert_eq!(laid.len(), 2 * CALL_FIELDS);
        // Field by field: both slots, then both kinds, and so on.
        assert_eq!(&laid[..4], &[5, 2, 1, 3]);
        assert_eq!(
            &laid[2 * field::REQUEST_ID..2 * field::REQUEST_ID + 2],
            &[9, 11]
        );

        // A media call is no token call.
        let image = Call::new(
            RequestKey::new(1, RequestId(1), 1),
            CallId::new(2, 2),
            CallKind::Media(ipc::MediaCall::Denoising),
        );
        assert!(kind_code(image.kind).is_err());
        Ok(())
    }

    /// The forward columns hold each call's rows together in row order,
    /// with each cache group's pages of the row.
    #[test]
    fn forward_columns_group_rows_by_call_with_their_pages() -> Result<()> {
        let mut forward = ForwardBatch::default();
        // Rows of calls 1, 0, 1: a context prefill's two rows and a decode.
        forward.push(1, 4, 12, 8, true);
        forward.push(0, 3, 30, 1, true);
        forward.push(1, 4, 20, 8, true);
        let rows = ForwardRows::new(&forward.call_indices, 2)?;
        let pages = [
            GroupPages {
                starts: vec![0, 1, 0],
                counts: vec![1, 2, 2],
            },
            GroupPages {
                starts: vec![2, 3, 4],
                counts: vec![5, 6, 7],
            },
        ];
        let columns = forward_columns(&forward, &rows, &pages);
        // Rows in grouped order 1, 0, 2.
        assert_eq!(
            columns,
            [
                0, 1, 1, // call index
                3, 4, 4, // slot
                30, 12, 20, // sequence length
                1, 8, 8, // query length
                1, 1, 1, // cache write
                1, 0, 0, // group 0 first page
                2, 1, 2, // group 0 pages
                3, 2, 4, // group 1 first page
                6, 5, 7, // group 1 pages
            ]
        );
        Ok(())
    }
}
