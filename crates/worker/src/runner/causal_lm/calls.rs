//! One token call's input row, sampling task and committed coordinates.
//!
//! The rules here decide, from call metadata and accepted request progress
//! alone (never from device values), how a prefill, decode or verification
//! call reads its tokens, which logits it selects, where it samples, and how
//! its request state advances when the batch commits.

use uniserve_ipc::SamplingParams;
use uniserve_ipc::{Call, CallKind, DType, ForwardMode};

use super::context::Media;
use crate::executor::PendingOutput;
use crate::runner::TokenSelection;
use crate::runner::causal_lm::{COMMIT_CACHE as CACHE, FINISH_IDS, RowSource, TokenRow};
use crate::sampling::{SamplingControls, SamplingPath};
use crate::storage::BlockTables;
use crate::{Error, Result};

/// Commit modes of the slot-state commit (`uniserve.sampling.state`).
pub(super) const DECODE: i64 = 0;
pub(super) const EXPLICIT: i64 = 1;
pub(super) const VERIFY: i64 = 2;

/// A call's input row and how it uses the row's logits.
pub(super) struct Prepared {
    /// The call's last row: the one it samples from.
    pub(super) row: TokenRow,
    /// The rows and positions of a call placing image features.
    pub(super) media: Option<Media>,
    /// The call scores its prompt chunk.
    pub(super) scores: bool,
    /// Draft tokens of a verification.
    pub(super) drafts: Vec<u32>,
    /// The call's predicate relays a token (an I64 continuation).
    pub(super) tagged: bool,
}

pub(super) fn kind(call: &Call) -> Result<ForwardMode> {
    match call.kind {
        CallKind::Forward(
            mode @ (ForwardMode::Prefill | ForwardMode::Decode | ForwardMode::Verify),
        ) => Ok(mode),
        kind => Err(Error::Invalid(format!(
            "{} is not a text call",
            kind.as_str()
        ))),
    }
}

/// The admitted sampling parameters of a call's request.
pub(super) fn parameters(admission: &uniserve_ipc::NewRequest) -> Result<&SamplingParams> {
    admission
        .generation
        .as_ref()
        .map(|generation| &generation.sampling)
        .ok_or_else(|| {
            Error::Invalid("sequence execution requires admitted sampling parameters".into())
        })
}

/// Build a call's input row.
///
/// A decode gated by a relayed token reads its token, position and cache
/// length from the slot state; a verification gated by one reads that token
/// from the slot state instead of a host token.
pub(super) fn prepare(
    call: &Call,
    output: &PendingOutput,
    params: &SamplingParams,
    slot: u32,
    tables: &BlockTables,
) -> Result<Prepared> {
    let mode = kind(call)?;
    let decode = mode == ForwardMode::Decode;
    let verify = mode == ForwardMode::Verify;
    let scores = params.prompt_logprobs_requested();
    let (_, visible, _) = tables.coordinates(slot, output.progress.cache_visible_len)?;
    let tagged = call
        .predicate
        .as_ref()
        .is_some_and(|predicate| predicate.dtype == DType::I64);
    let indexed = decode && tagged;
    let relay = call.predicate.is_some()
        && !indexed
        && (decode || verify || call.input_token_ids.is_empty());
    let tokens: &[u32] = if indexed || (decode && relay) {
        &[]
    } else if decode {
        call.input_token_ids.get(..1).ok_or_else(|| {
            Error::Invalid("last-sampled token source has no committed token".into())
        })?
    } else {
        &call.input_token_ids
    };
    if verify && !relay && tokens.len() < 2 {
        return Err(Error::Invalid(
            "fixed-parent verification requires current and draft tokens".into(),
        ));
    }
    if tokens.is_empty() && !indexed && !relay {
        return Err(Error::Invalid(
            "text extension requires input tokens".into(),
        ));
    }

    let selection = if verify || (!decode && scores) {
        TokenSelection::AllLogits
    } else if decode || call.token_output.is_some() {
        TokenSelection::LastLogits
    } else {
        TokenSelection::Cache
    };
    let position = output.progress.logical_position;
    let source = if indexed {
        RowSource::Indexed
    } else if relay {
        RowSource::Relay {
            drafts: tokens.to_vec(),
            position,
        }
    } else {
        RowSource::Host {
            tokens: tokens.to_vec(),
            position,
        }
    };
    let query = match &source {
        RowSource::Host { tokens, .. } | RowSource::Placed { tokens, .. } => tokens.len(),
        RowSource::Relay { drafts, .. } => drafts.len() + 1,
        RowSource::Indexed => 1,
    };
    let first = usize::from(call.predicate.is_none());
    let drafts = if verify {
        call.input_token_ids
            .get(first..)
            .unwrap_or_default()
            .to_vec()
    } else {
        Vec::new()
    };
    Ok(Prepared {
        media: None,
        row: TokenRow {
            slot,
            source,
            prefix: u32::try_from(visible)
                .map_err(|_| Error::Invalid("visible cache exceeds the token range".into()))?,
            query: u32::try_from(query)
                .map_err(|_| Error::Invalid("query exceeds the token range".into()))?,
            selection,
            gated: tagged,
            force_finish: call
                .sampling_state
                .as_ref()
                .is_some_and(|state| state.force_finish),
            causal: true,
        },
        scores,
        drafts,
        tagged,
    })
}

/// Whether a call samples, and its semantic sampling positions.
///
/// A prefill without a token output only extends the prefix cache. Prefill and decode
/// sample after their last token; a verification samples one position per
/// draft plus its bonus position.
pub(super) fn sample_positions(
    call: &Call,
    prepared: &Prepared,
    start: u64,
) -> Result<Option<Vec<u64>>> {
    let mode = kind(call)?;
    if let Some(media) = &prepared.media {
        return Ok(call.token_output.is_some().then(|| vec![media.sample]));
    }
    Ok(match mode {
        ForwardMode::Prefill if call.token_output.is_none() => None,
        ForwardMode::Prefill | ForwardMode::Decode => {
            Some(vec![start + u64::from(prepared.row.query)])
        }
        _ => Some((start + 1..start + prepared.drafts.len() as u64 + 2).collect()),
    })
}

/// Sampling controls of a sampled call.
pub(super) fn controls(
    call: &Call,
    params: &SamplingParams,
    finish_tokens: &[u32],
    positions: &[u64],
    drafts: Vec<u32>,
    return_transition: bool,
) -> Result<(SamplingControls, Option<Vec<f32>>)> {
    let (mut controls, draws) =
        SamplingControls::for_call(call, params, finish_tokens, positions, drafts)?;
    controls.return_transition = return_transition;
    Ok((controls, draws))
}

/// Check the cache extent a call writes and return it (prefix plus written
/// tokens).
pub(super) fn written_length(tables: &BlockTables, row: &TokenRow, count: u64) -> Result<u64> {
    if count > u64::from(row.query) {
        return Err(Error::Invariant(
            "cache commit count is outside the task query span".into(),
        ));
    }
    let length = u64::from(row.prefix) + count;
    if length > u64::from(tables.allocated_length(row.slot)) {
        return Err(Error::Invariant(
            "cache task exceeds its scheduler block table".into(),
        ));
    }
    Ok(length)
}

/// Advance a call's progress after its step and return its commit row
/// `[slot, mode, task, cache, logical, sampling, visible, penalties]`.
///
/// `task` is the call's output position among the invocation's sampled
/// tasks; `None` for a prefill that only extends the prefix cache.
pub(super) fn finish(
    call: &Call,
    output: &mut PendingOutput,
    prepared: &Prepared,
    tables: &BlockTables,
    task: Option<usize>,
    controls: Option<&SamplingControls>,
    penalties: bool,
) -> Result<[i64; 8]> {
    let mode = kind(call)?;
    let row = &prepared.row;
    let slot = i64::from(row.slot);
    let start = output.progress.logical_position;
    let rng = output.progress.rng_counter;
    if let Some(media) = &prepared.media {
        // A call placing image features advances to explicit positions.
        let length = written_length(tables, row, u64::from(row.query))?;
        output.advance_tokens(0, Some(length), Some(media.position), task.is_some());
        return Ok(match task {
            None => [slot, CACHE, -1, length as i64, 0, 0, 0, 0],
            Some(task) => [
                slot,
                EXPLICIT,
                task as i64,
                length as i64,
                media.continued as i64,
                output.progress.rng_counter as i64,
                0,
                i64::from(penalties),
            ],
        });
    }
    match (mode, task) {
        (ForwardMode::Prefill, None) => {
            let length = written_length(tables, row, u64::from(row.query))?;
            output.advance_tokens(0, Some(length), None, false);
            Ok([slot, CACHE, -1, length as i64, 0, 0, 0, 0])
        }
        (ForwardMode::Prefill, Some(task)) => {
            let count = u64::from(row.query);
            let length = written_length(tables, row, count)?;
            output.advance_tokens(count, Some(length), None, true);
            Ok([
                slot,
                EXPLICIT,
                task as i64,
                length as i64,
                output.progress.logical_position as i64,
                output.progress.rng_counter as i64,
                0,
                i64::from(penalties),
            ])
        }
        (ForwardMode::Decode, Some(task)) => {
            let length = written_length(tables, row, 1)?;
            output.advance_tokens(1, Some(length), None, true);
            Ok([slot, DECODE, task as i64, 0, 0, 0, 0, i64::from(penalties)])
        }
        (ForwardMode::Verify, Some(task)) => {
            let controls = controls
                .ok_or_else(|| Error::Invariant("verification requires sampling inputs".into()))?;
            let visible = u64::from(row.prefix);
            let initialized = visible + u64::from(row.query);
            output.set_speculation(
                controls.draft_token_ids.clone(),
                controls.terminal_draft_prefix,
                visible,
                initialized,
            );
            Ok([
                slot,
                VERIFY,
                task as i64,
                0,
                start as i64,
                rng as i64,
                visible as i64,
                i64::from(penalties),
            ])
        }
        _ => Err(Error::Invariant(
            "decode and verification calls always sample".into(),
        )),
    }
}

/// How a decode forward selects its rows' tokens itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum OwnSelection {
    /// The argmax of each row's logits.
    Greedy,
    /// Each row sampled from its top-k candidates under its own parameters
    /// and draw; greedy rows are its zero-temperature rows.
    TopK,
}

/// The selection a decode forward can make for a call itself, or `None`
/// when the call needs a sampling method after the forward. The forward
/// selects for a decode relayed by its slot's token, with a token relay
/// output, no transition, finish tokens the forward reads (at most
/// [`FINISH_IDS`]) and the unshaped greedy or the top-k sampler as `path`.
pub(super) fn own_selection(
    call: &Call,
    prepared: &Prepared,
    controls: &SamplingControls,
    path: SamplingPath,
    token_output: bool,
    transition_output: bool,
) -> Option<OwnSelection> {
    let transitions = call
        .sampling_state
        .as_ref()
        .is_some_and(|state| !state.transition_token_ids.is_empty());
    if call.kind != CallKind::Forward(ForwardMode::Decode)
        || controls.finish_token_ids.len() > FINISH_IDS
        || transitions
        || transition_output
        || !prepared.tagged
        || !token_output
    {
        return None;
    }
    match path {
        SamplingPath::Greedy { suppressed: false } => Some(OwnSelection::Greedy),
        SamplingPath::TopK => Some(OwnSelection::TopK),
        SamplingPath::Greedy { suppressed: true } | SamplingPath::Categorical => None,
    }
}
