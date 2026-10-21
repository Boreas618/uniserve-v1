//! Text sampling parameters as a request's admission carries them.

use serde::{Deserialize, Serialize};

/// Text sampling parameters.
///
/// Worker-side math (temperature, top_k, top_p, min_p, typical_p, penalties,
/// logit_bias, allowed and forced tokens) operates on the logits tensor inside
/// the worker. The engine scheduler enforces the
/// control-flow fields on the host: `min_tokens` and `bad_words_ids` become
/// per-step suppression masks, and `ignore_eos` decides whether a sampled EOS
/// finishes the request. Fields are scalars or small id/weight lists — never
/// tensors — so they can cross the worker boundary without carrying device
/// state.
///
/// The worker receives these fields in a request's admission message and
/// stages them for the Python samplers in `uniserve.sampling`. All consumers share this FP32 representation
/// and its numerical validation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Softmax temperature; zero selects greedy decoding.
    pub temperature: f32,
    /// Maximum number of highest-logit candidates; zero disables the cutoff.
    pub top_k: u32,
    /// Cumulative probability mass retained by nucleus sampling.
    pub top_p: f32,
    /// Whether a sampled EOS token leaves the request running instead of
    /// finishing it. EOS stays eligible for sampling either way once the
    /// minimum-token floor is met.
    pub ignore_eos: bool,
    /// Optional deterministic random seed.
    pub seed: Option<u64>,
    /// Minimum-probability cutoff relative to the top token.
    pub min_p: f32,
    /// Multiplicative repetition penalty (1.0 == no-op).
    pub repetition_penalty: f32,
    /// Additive frequency penalty over the running output (0.0 == no-op).
    pub frequency_penalty: f32,
    /// Additive presence penalty over the running output (0.0 == no-op).
    pub presence_penalty: f32,
    /// Per-token logit bias, `(token_id, bias)` pairs (empty == no-op).
    pub logit_bias: Vec<(u32, f32)>,
    /// Floor on generated tokens before EOS/stop may fire.
    pub min_tokens: usize,
    /// Whether to return the sampled token's log probability.
    pub return_logprobs: bool,
    /// How many highest-probability alternatives to return per generated step.
    pub n_logprobs: u32,
    /// Whether to score prompt positions during prefill.
    pub return_prompt_logprobs: bool,
    /// How many highest-probability alternatives to return per scored prompt position.
    pub n_prompt_logprobs: u32,
    /// Bad-word token sequences: generation may not complete any of these.
    /// Enforced host-side by masking the completing token.
    pub bad_words_ids: Vec<Vec<u32>>,
    /// If set, only these token ids may be sampled (whitelist mask).
    pub allowed_token_ids: Option<Vec<u32>>,
    /// Typical-sampling mass cutoff over the locally-typical set (1.0 == no-op).
    pub typical_p: f32,
    /// Forced-decoding schedule: point `i` of a call's span is forced to
    /// `forced_token_ids[i]` when present, overriding stochastic selection.
    pub forced_token_ids: Vec<u32>,
}

/// Validation failure returned by [`SamplingParams::validate`].
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SamplingParamsError {
    /// A floating-point parameter is NaN or infinite.
    #[error("{field} must be finite, got {got}")]
    NonFinite {
        /// Name of the invalid parameter.
        field: &'static str,
        /// Supplied non-finite value.
        got: f32,
    },
    /// Temperature is below zero.
    #[error("temperature must be non-negative, got {got}")]
    NegativeTemperature {
        /// Supplied temperature.
        got: f32,
    },
    /// Nucleus probability lies outside `(0, 1]`.
    #[error("top_p must be in (0, 1], got {got}")]
    TopP {
        /// Supplied probability.
        got: f32,
    },
    /// Typical-sampling probability lies outside `(0, 1]`.
    #[error("typical_p must be in (0, 1], got {got}")]
    TypicalP {
        /// Supplied probability.
        got: f32,
    },
    /// Minimum relative probability lies outside `[0, 1]`.
    #[error("min_p must be in [0, 1], got {got}")]
    MinP {
        /// Supplied probability.
        got: f32,
    },
    /// Repetition penalty is not positive.
    #[error("repetition_penalty must be positive, got {got}")]
    RepetitionPenalty {
        /// Supplied penalty.
        got: f32,
    },
    /// An explicit allowed-token set is empty.
    #[error("allowed_token_ids must contain at least one token when present")]
    EmptyAllowedTokenIds,
    /// A bad-word sequence is empty.
    #[error("bad_words_ids[{index}] must contain at least one token")]
    EmptyBadWord {
        /// Index of the empty sequence.
        index: usize,
    },
}

impl SamplingParams {
    /// Whether token-count penalties affect selection.
    pub fn uses_penalties(&self) -> bool {
        self.repetition_penalty != 1.0
            || self.frequency_penalty != 0.0
            || self.presence_penalty != 0.0
    }

    /// Parameters that permit unpenalized device argmax without score readout.
    /// Per-call token masks and draft chains are checked by the sampler.
    pub fn device_greedy(&self) -> bool {
        self.temperature <= 0.0
            && self.top_k == 0
            && self.top_p == 1.0
            && self.typical_p == 1.0
            && !self.generated_logprobs_requested()
            && self.logit_bias.is_empty()
            && !self.uses_penalties()
    }

    /// Returns whether generated-token logprobs are enabled.
    pub fn generated_logprobs_requested(&self) -> bool {
        self.return_logprobs || self.n_logprobs > 0
    }

    /// Returns whether prompt-token logprobs are enabled.
    pub fn prompt_logprobs_requested(&self) -> bool {
        self.return_prompt_logprobs || self.n_prompt_logprobs > 0
    }

    /// Validates sampling math inputs before they reach a worker or simulator.
    ///
    /// Callers include the server's sampling lowering,
    /// `GenerationRequest::validate`, and the worker IPC
    /// `NewRequest::validate`. It checks finiteness, the numeric domain of
    /// each transform, and that token constraints are non-empty. It does not
    /// check token ids against a vocabulary or bound `top_k`, `min_tokens`, or
    /// the logprob counts.
    pub fn validate(&self) -> Result<(), SamplingParamsError> {
        // Reject non-finite scalar math inputs before applying their individual
        // range constraints.
        for (field, value) in [
            ("temperature", self.temperature),
            ("top_p", self.top_p),
            ("typical_p", self.typical_p),
            ("min_p", self.min_p),
            ("repetition_penalty", self.repetition_penalty),
            ("frequency_penalty", self.frequency_penalty),
            ("presence_penalty", self.presence_penalty),
        ] {
            if !value.is_finite() {
                return Err(SamplingParamsError::NonFinite { field, got: value });
            }
        }

        // Bias values participate directly in logit arithmetic and follow the
        // same finite-value contract as scalar processors.
        for &(_, bias) in &self.logit_bias {
            if !bias.is_finite() {
                return Err(SamplingParamsError::NonFinite {
                    field: "logit_bias",
                    got: bias,
                });
            }
        }

        // Enforce the mathematical domains of each sampling transform.
        if self.temperature < 0.0 {
            return Err(SamplingParamsError::NegativeTemperature {
                got: self.temperature,
            });
        }
        if !(0.0 < self.top_p && self.top_p <= 1.0) {
            return Err(SamplingParamsError::TopP { got: self.top_p });
        }
        if !(0.0 < self.typical_p && self.typical_p <= 1.0) {
            return Err(SamplingParamsError::TypicalP {
                got: self.typical_p,
            });
        }
        if !(0.0..=1.0).contains(&self.min_p) {
            return Err(SamplingParamsError::MinP { got: self.min_p });
        }
        if self.repetition_penalty <= 0.0 {
            return Err(SamplingParamsError::RepetitionPenalty {
                got: self.repetition_penalty,
            });
        }

        // Empty vocabulary constraints cannot express a valid sampling set.
        if self.allowed_token_ids.as_ref().is_some_and(Vec::is_empty) {
            return Err(SamplingParamsError::EmptyAllowedTokenIds);
        }
        if let Some(index) = self.bad_words_ids.iter().position(Vec::is_empty) {
            return Err(SamplingParamsError::EmptyBadWord { index });
        }
        Ok(())
    }
}

impl Default for SamplingParams {
    /// Defaults direct construction to greedy sampling.
    ///
    /// Server request lowering does not inherit this temperature: text
    /// requests start from the checkpoint's sampling defaults (temperature
    /// `1.0` when it declares none), omni requests start from fixed profile
    /// defaults, and both then apply the caller's explicit values.
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            ignore_eos: false,
            seed: None,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            logit_bias: Vec::new(),
            min_tokens: 0,
            return_logprobs: false,
            n_logprobs: 0,
            return_prompt_logprobs: false,
            n_prompt_logprobs: 0,
            bad_words_ids: Vec::new(),
            allowed_token_ids: None,
            typical_p: 1.0,
            forced_token_ids: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Direct construction stays greedy; request lowering, not this default,
    /// supplies the API temperature.
    #[test]
    fn sampling_default_is_greedy() {
        let sp = SamplingParams::default();
        assert_eq!(
            sp.temperature, 0.0,
            "core default must stay greedy (0.0); unset->1.0 is resolved in lowering, not here"
        );
        assert_eq!(sp.top_p, 1.0);
        assert_eq!(sp.repetition_penalty, 1.0);
        assert_eq!(sp.top_k, 0);
    }
}
