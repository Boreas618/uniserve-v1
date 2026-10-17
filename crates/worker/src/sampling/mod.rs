//! Token sampling: a sampled call's controls ([`SamplingControls`]) and a
//! forward's sampling metadata ([`SamplingMetadata`], or [`RowSampling`]
//! for a decode forward that samples its rows itself), which fill the
//! sampler's parameter tensors.

use std::collections::BTreeSet;

use uniserve_core::SamplingParams;
use uniserve_core::philox::{DRAW_LAYOUT_TARGET, request_sampling_key, sampling_uniform};
use uniserve_ipc::{Call, DrawLayout};

use crate::{Error, Result};

mod metadata;

pub(crate) use metadata::{
    RowSampling, SampleTask, SamplingGroup, SamplingMetadata, logprob_request, logprob_words,
    padded,
};

/// The most candidates the top-k sampler keeps per row: a row samples from
/// its top-k logits when its top-k width is at most this. The captured
/// decode selection keeps this many for every row, so its shapes do not
/// depend on the rows' widths.
pub(crate) const TOP_K_CANDIDATES: usize = 128;

/// Candidates the top-k sampler keeps per row of a `vocab`-token vocabulary.
pub(crate) fn candidates(vocab: usize) -> usize {
    TOP_K_CANDIDATES.min(vocab)
}

/// Token relays carry continuation in bit 31 and the token in the low 31 bits.
pub(crate) const TOKEN_CONTINUATION_BIT: u32 = 1 << 31;
pub(crate) const TOKEN_VALUE_MASK: u32 = TOKEN_CONTINUATION_BIT - 1;

/// Numerical selectors supported by the worker's batched sampler.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum SamplingPath {
    /// Argmax of the unshaped logits, optionally after masking suppressed
    /// tokens.
    Greedy { suppressed: bool },
    /// Each row samples from its top-k candidates under its own parameters
    /// (temperature, top-k, top-p, min-p, penalties) and draw. Every value
    /// is a tensor column, so a decode forward captures this sampler.
    TopK,
    /// The general path, for what the caller knows on the host: token
    /// sets, logit biases, typical-p, draft tokens, reported logprobs, or a
    /// sampled top-k width beyond the top-k sampler's candidates.
    Categorical,
}

/// Per-call constraints aligned with candidate logit rows. Tensor values and
/// penalties stay with the numerical backend; no device value is read here.
#[derive(Clone, Debug, Default)]
pub(crate) struct SamplingControls {
    pub(crate) allowed: Vec<Option<Vec<u32>>>,
    pub(crate) suppress: Vec<u32>,
    pub(crate) finish_token_ids: Vec<u32>,
    pub(crate) transition_token_ids: Vec<u32>,
    pub(crate) force_finish: bool,
    pub(crate) draft_token_ids: Vec<u32>,
    /// One-based prefix length ending at the first terminal draft token.
    pub(crate) terminal_draft_prefix: Option<usize>,
    pub(crate) return_transition: bool,
}

impl SamplingControls {
    /// Resolve admitted and branch-local controls. Draw coordinates describe
    /// the candidate rows, independent of scheduling order or draft acceptance.
    /// A seeded request's draws depend on its seed and positions alone, so
    /// the request repeats when sent again ([`request_sampling_key`]).
    /// The returned draws are absent only for the unshaped greedy path.
    pub(crate) fn for_call(
        call: &Call,
        parameters: &SamplingParams,
        finish_tokens: &[u32],
        positions: &[u64],
        drafts: Vec<u32>,
    ) -> Result<(Self, Option<Vec<f32>>)> {
        let state = call.sampling_state.as_ref();
        let allowed = state
            .and_then(|state| state.allowed_token_ids.as_ref())
            .or(parameters.allowed_token_ids.as_ref());
        let mut result = Self {
            allowed: positions
                .iter()
                .enumerate()
                .map(|(index, _)| {
                    parameters
                        .forced_token_ids
                        .get(index)
                        .map_or_else(|| allowed.cloned(), |token| Some(vec![*token]))
                })
                .collect(),
            suppress: state.map_or_else(Vec::new, |state| state.suppressed_token_ids.clone()),
            finish_token_ids: finish_token_ids(call, finish_tokens),
            transition_token_ids: state
                .map_or_else(Vec::new, |state| state.transition_token_ids.clone()),
            force_finish: state.is_some_and(|state| state.force_finish),
            draft_token_ids: drafts,
            ..Self::default()
        };
        result.terminal_draft_prefix = result
            .draft_token_ids
            .iter()
            .position(|token| result.finish_token_ids.contains(token))
            .map(|index| index + 1);

        let draws = if parameters.temperature > 0.0 {
            let rng = call
                .rng
                .as_ref()
                .filter(|rng| rng.draw_layout == DrawLayout::TargetSampling)
                .ok_or_else(|| {
                    Error::Invalid(
                        "stochastic sampling requires target-sampling RNG coordinates".into(),
                    )
                })?;
            if rng.seed != parameters.seed.unwrap_or(0) {
                return Err(Error::Invalid(
                    "call RNG seed disagrees with admitted sampling".into(),
                ));
            }
            if positions.iter().enumerate().any(|(index, position)| {
                rng.semantic_index_base.checked_add(index as u64) != Some(*position)
            }) {
                return Err(Error::Invalid(
                    "sampling positions disagree with registered semantic RNG coordinates".into(),
                ));
            }

            let key = request_sampling_key(
                parameters.seed,
                call.request_key.engine_id,
                call.request_key.request_id.0,
                call.request_key.request_epoch,
                DRAW_LAYOUT_TARGET,
            );
            Some(
                positions
                    .iter()
                    .map(|position| sampling_uniform(key, *position, 0, 0))
                    .collect(),
            )
        } else if result.device_greedy(parameters) {
            None
        } else {
            Some(vec![0.0; positions.len()])
        };
        Ok((result, draws))
    }

    pub(crate) fn device_greedy(&self, parameters: &SamplingParams) -> bool {
        self.draft_token_ids.is_empty()
            && parameters.device_greedy()
            && self.allowed.iter().all(Option::is_none)
    }

    /// Select a numerical implementation without inspecting logits on device.
    pub(crate) fn path(&self, parameters: &SamplingParams, vocab: usize) -> SamplingPath {
        if self.device_greedy(parameters) {
            return SamplingPath::Greedy {
                suppressed: !self.suppress.is_empty(),
            };
        }

        // The top-k sampler reads only per-row parameters and draws: no
        // token set, bias or draft, no reported logprob (whose packed layout
        // the host plans), and at most its candidates when sampling. A
        // zero-temperature row selects its largest shaped logit, which the
        // candidates always hold.
        if self.draft_token_ids.is_empty()
            && self.allowed.iter().all(Option::is_none)
            && self.suppress.is_empty()
            && parameters.logit_bias.is_empty()
            && parameters.typical_p == 1.0
            && !parameters.generated_logprobs_requested()
            && (parameters.temperature <= 0.0
                || top_k_width(parameters, vocab) <= candidates(vocab))
        {
            SamplingPath::TopK
        } else {
            SamplingPath::Categorical
        }
    }
}

/// The tokens a row's top-k cutoff keeps: its top-k width inside the
/// vocabulary, or the whole vocabulary without one.
pub(crate) fn top_k_width(parameters: &SamplingParams, vocab: usize) -> usize {
    match parameters.top_k as usize {
        0 => vocab,
        top_k => top_k.min(vocab),
    }
}

/// Merge admitted stop tokens with the current branch's terminal tokens.
pub(crate) fn finish_token_ids(call: &Call, admitted: &[u32]) -> Vec<u32> {
    let local = call
        .sampling_state
        .as_ref()
        .map_or(&[][..], |state| state.finish_token_ids.as_slice());
    if local.is_empty() {
        admitted.to_vec()
    } else if admitted.is_empty() {
        local.to_vec()
    } else {
        admitted
            .iter()
            .chain(local)
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }
}
