//! A causal forward's outputs, and the completion of a step assembled from them.
//!
//! ## The outputs of a forward
//!
//! `CausalLmRunner::forward` returns its outputs in one order, of which a bucket's variant and
//! the runner's config select the entries present. `rows` and `tokens` are the bucket's.
//!
//! | Entry | Type and extent | Present |
//! | --- | --- | --- |
//! | the chosen tokens | U32 `[rows]` | always |
//! | their log-probabilities | F32 `[rows]` | always |
//! | their ranks | I32 `[rows]` | always |
//! | each call's logical position after the commit | I32 `[rows]` | always |
//! | each call's sampling position after the commit | I32 `[rows]` | always |
//! | each row's most likely tokens' log-probabilities, then their ids | F32 and U32 `[rows, candidates]` | when the runner reports candidates |
//! | whether each row's candidates formed a distribution | I32 `[rows]` | in a bucket that draws |
//! | each token's target's log-probability, then its rank | F32 and I32 `[tokens]` | in a bucket that scores its prompt |
//! | each token's most likely tokens' log-probabilities, then their ids | F32 and U32 `[tokens, candidates]` | in a bucket that scores its prompt, when the runner reports candidates |
//!
//! The first three and the candidates are indexed by row, the positions by call (a call's
//! entry is its slot's), and the prompt scores by token. The join bucket of an expert union
//! returns no output.
//!
//! A verification bucket returns the entries above up to the validity flags, the rows being
//! its tokens and the calls its verifications, then each verification's agreed length, U32
//! `[verifications]`. The drafter's bucket returns one entry per chained step: the draft each
//! row proposes at that step, U32 `[rows]`. A runner that speculates returns its sampling
//! outputs from the memory it keeps them in, in the same order.
//!
//! ## The completion
//!
//! [`calls_output`] names one [`CallOutput`] per call of the batch, in the batch's order:
//!
//! - a call's tokens are those of the rows it owns whose logits the batch asked for, in row
//!   order, each with its scores; a row the batch asked nothing of contributes nothing, though
//!   the forward chose a token for it;
//! - a call's state is its slot's positions after the commit, and the default for a call that
//!   runs on no slot;
//! - a call that scores its prompt reports, for each of its tokens that names a target, the
//!   target's scores, in token order;
//! - a call one of whose rows could not be sampled (no candidate formed a distribution, or the
//!   greedy choice is no token of the vocabulary) fails alone, as a numerical fault: the other
//!   calls of the batch complete;
//! - a call whose tokens name a cache entry the batch may not write (`columns::slot_mapping`)
//!   wrote nothing and fails alone, as an invariant violation ([`refuse`]): the control plane
//!   scheduled a row outside the entries its owner may write.
//!
//! [`verification_output`] names the same for a verification batch: a call's tokens are its
//! accepted prefix, the drafts its own samples agreed with and the sample after them, each
//! with its scores. The drafts a call proposes are added by `speculation::propose`.

use crate::batch::{CallOutput, CallStatus, Columns, ErrorCode};
use tensor::{Tensor, WithDType};

use super::columns::Refusal;
use super::speculation::token_rows;
use crate::runner::buckets::Bucket;
use crate::runner::performed;
use crate::runner::sampler::scored;
use crate::worker::{Error, Result};

/// The elements of `tensor` in row-major order, read under the installed context.
fn values<T: WithDType>(tensor: &Tensor) -> Result<Vec<T>> {
    Ok(tensor.flatten_all()?.to_vec1()?)
}

/// A position the device holds as I32, which is never negative.
const fn length(value: i32) -> u32 {
    if value < 0 { 0 } else { value as u32 }
}

/// Entry `index` of a column of positions; zero for an entry the column does not hold, so a
/// call past the column reads no other call's position.
fn position(column: &[i32], index: usize) -> u32 {
    length(column.get(index).copied().unwrap_or(0))
}

/// The completion of the step that served the batch of `columns` from `bucket`, from the
/// forward's `outputs` in their order (module documentation): one output per call of
/// `columns`, in their order, which is the order of the batch's requests.
///
/// `vocabulary` is the width of a logit row and `candidates` the most likely tokens a scored
/// row reports, as the runner's sampler is configured.
///
/// # Errors
///
/// [`Error::Invariant`] when the outputs are not the ones the bucket's forward returns, or when
/// a row whose logits the batch asked for belongs to no call.
pub(crate) fn calls_output(
    columns: &Columns,
    bucket: &Bucket,
    vocabulary: u32,
    candidates: u32,
    outputs: &[Tensor],
) -> Result<Vec<CallOutput>> {
    let rows = &columns.rows;
    let width = candidates as usize;

    // ---- The outputs, in their order ----

    let mut entries = outputs.iter();
    let mut next = || {
        entries.next().ok_or_else(|| {
            Error::Invariant(format!(
                "the forward of bucket {bucket} returned {} outputs, fewer than its variant \
                 states",
                outputs.len()
            ))
        })
    };

    let tokens: Vec<u32> = values(next()?)?;
    let logprobs: Vec<f32> = values(next()?)?;
    let ranks: Vec<i32> = values(next()?)?;
    let logical: Vec<i32> = values(next()?)?;
    let sampling: Vec<i32> = values(next()?)?;

    let (top_values, top_ids): (Vec<f32>, Vec<u32>) = if width > 0 {
        (values(next()?)?, values(next()?)?)
    } else {
        (Vec::new(), Vec::new())
    };
    let valid: Option<Vec<i32>> = if bucket.variant.greedy() {
        None
    } else {
        Some(values(next()?)?)
    };

    let prompt: Option<(Vec<f32>, Vec<i32>)> = if bucket.variant.scores() {
        Some((values(next()?)?, values(next()?)?))
    } else {
        None
    };
    let (prompt_values, prompt_ids): (Vec<f32>, Vec<u32>) = if prompt.is_some() && width > 0 {
        (values(next()?)?, values(next()?)?)
    } else {
        (Vec::new(), Vec::new())
    };

    // ---- One completion per call, with its slot's state ----

    if rows.logits_rows.len() > tokens.len() || rows.calls.len() > logical.len() {
        return Err(Error::Invariant(format!(
            "the bucket chose {} tokens and the batch holds {} rows in {} calls",
            tokens.len(),
            rows.logits_rows.len(),
            rows.calls.len()
        )));
    }

    let mut output: Vec<CallOutput> = Vec::with_capacity(rows.calls.len());
    for (index, call) in rows.calls.iter().enumerate() {
        // The commit table is one row per call, and the sampler's slots one per row: a call
        // reads the slot its rows share through the first row it owns. A call on no slot
        // reports the default state.
        let mut entry = performed();
        let slot = columns.slots.get(call.rows.start as usize);
        if slot.is_some_and(|slot| *slot > 0) {
            entry.step_index = position(&sampling, index);
            entry.cache_computed_len = position(&logical, index);
            entry.cache_visible_len = position(&sampling, index);
        }

        output.push(entry);
    }

    // ---- Each asked row's token, to the call that owns the row ----

    let candidates_of = |values: &[f32], ids: &[u32], index: usize| {
        let span = index * width..(index + 1) * width;
        (
            values.get(span.clone()).unwrap_or_default().to_vec(),
            ids.get(span).unwrap_or_default().to_vec(),
        )
    };

    for (row, logits_row) in rows.logits_rows.iter().copied().enumerate() {
        // A negative entry is a row that samples nothing: a chunk of a prefill that ends no
        // prompt. The forward chose a token for it all the same, which nothing reads.
        let Ok(logits_row) = u32::try_from(logits_row) else {
            continue;
        };
        if !rows
            .calls
            .iter()
            .any(|call| call.rows.contains(&logits_row))
        {
            return Err(Error::Invariant(format!(
                "sampling row {logits_row} names a row the batch does not hold"
            )));
        }

        // The sample belongs to the batch row, whatever row its logits were read from.
        let owner = rows
            .calls
            .iter()
            .position(|call| call.rows.contains(&(row as u32)))
            .ok_or_else(|| {
                Error::Invariant(format!("no call of the batch owns sampling row {row}"))
            })?;
        let entry = &mut output[owner];

        // A row with no candidate holds no token: a drawn one states it by its flag, and a
        // greedy one by a choice past the vocabulary.
        let token = tokens[row];
        let drawable = valid
            .as_ref()
            .is_none_or(|valid| valid.get(row).is_some_and(|flag| *flag != 0));
        let sampled = token < vocabulary && drawable;
        if !sampled {
            entry.status = CallStatus::Error;
            entry.error_code = Some(ErrorCode::NumericalFault);
            continue;
        }

        let (values, ids) = candidates_of(&top_values, &top_ids, row);
        entry.tokens.push(token);
        entry.token_logprobs.push(scored(
            token,
            logprobs.get(row).copied().unwrap_or(0.0),
            position(&ranks, row),
            &values,
            &ids,
            vocabulary,
        ));
    }

    // A call that failed reports nothing it sampled.
    for entry in &mut output {
        if entry.status == CallStatus::Error {
            entry.tokens.clear();
            entry.token_logprobs.clear();
        }
    }

    // ---- A scoring bucket's prompt scores, to the call whose rows hold the tokens ----

    if let Some((prompt_logprobs, prompt_ranks)) = &prompt {
        let offset = |row: u32| {
            rows.query_offsets
                .get(row as usize)
                .map_or(0, |offset| (*offset).max(0) as usize)
        };

        for (call, entry) in rows.calls.iter().zip(&mut output) {
            for token in offset(call.rows.start)..offset(call.rows.end) {
                // A token whose target is negative scored nothing.
                let Some(target) = columns
                    .targets
                    .get(token)
                    .and_then(|target| u32::try_from(*target).ok())
                else {
                    continue;
                };
                let Some(logprob) = prompt_logprobs.get(token) else {
                    continue;
                };

                let (values, ids) = candidates_of(&prompt_values, &prompt_ids, token);
                entry.prompt_logprobs.push(scored(
                    target,
                    *logprob,
                    position(prompt_ranks, token),
                    &values,
                    &ids,
                    vocabulary,
                ));
            }
        }
    }

    Ok(output)
}

/// The completion of the step that served the verification batch of `columns` from `bucket`,
/// from the forward's `outputs` in their order (module documentation).
///
/// Each token of a verification sampled the position after it, so the batch's completion is
/// that of its rows of one token ([`token_rows`]), of which a call keeps its accepted prefix:
/// one more token than the length the device agreed for it. A call one of whose samples is no
/// token fails alone, as in [`calls_output`].
///
/// # Errors
///
/// As [`calls_output`]; [`Error::Invariant`] for outputs that state no agreed length for a
/// call.
pub(crate) fn verification_output(
    columns: &Columns,
    bucket: &Bucket,
    vocabulary: u32,
    candidates: u32,
    outputs: &[Tensor],
) -> Result<Vec<CallOutput>> {
    let rows = token_rows(columns);
    let mut output = calls_output(&rows, bucket, vocabulary, candidates, outputs)?;

    // The agreed lengths are the forward's last output, one per verification.
    let agreed: Vec<u32> = match outputs.last() {
        Some(agreed) => values(agreed)?,
        None => Vec::new(),
    };
    if agreed.len() < output.len() {
        return Err(Error::Invariant(format!(
            "the forward of bucket {bucket} agreed lengths for {} verifications of {}",
            agreed.len(),
            output.len()
        )));
    }

    for (entry, agreed) in output.iter_mut().zip(agreed) {
        let accepted = agreed as usize + 1;

        entry.tokens.truncate(accepted);
        entry.token_logprobs.truncate(accepted);
    }
    Ok(output)
}

/// Fails each call of `refused`, by its index among `calls` with the reason it was refused
/// (`columns::slot_mapping`): its tokens wrote no cache entry, so it reports no token and no
/// state, as an invariant violation of the control plane, which scheduled a row outside the
/// entries its owner may write. The reason is logged: a failed call states a message only for a
/// refused input.
pub(crate) fn refuse(calls: &mut [CallOutput], refused: &[Refusal]) {
    for (index, reason) in refused {
        tracing::error!(call = index, %reason, "a call writes a cache entry it may not");

        if let Some(entry) = calls.get_mut(*index) {
            *entry = CallOutput {
                status: CallStatus::Error,
                error_code: Some(ErrorCode::InvariantViolation),
                ..performed()
            };
        }
    }
}
