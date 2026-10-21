//! Token sampling resolution: the model's sampling defaults under a request's
//! controls ([`resolve_sampling`]), and its generation length
//! ([`resolve_max_tokens`]).

use uniserve_ipc::SamplingParams as EngineSamplingParams;

use super::{SamplingParams, TokenizeError};
use crate::profile::SamplingDefaults;
use crate::profile::tokenizer::HuggingFaceTokenizer;

/// Resolves a request's engine sampling parameters: the model's `defaults`
/// under the request's `controls`, with the request's resolved `seed`.
///
/// # Errors
///
/// As [`apply_sampling`].
pub(super) fn resolve_sampling(
    tokenizer: &HuggingFaceTokenizer,
    defaults: &SamplingDefaults,
    controls: &SamplingParams,
    seed: Option<u64>,
) -> Result<EngineSamplingParams, TokenizeError> {
    let mut sampling = EngineSamplingParams {
        temperature: defaults.temperature.unwrap_or(1.0),
        top_k: defaults.top_k.unwrap_or(0),
        top_p: defaults.top_p.unwrap_or(1.0),
        min_p: defaults.min_p.unwrap_or(0.0),
        repetition_penalty: defaults.repetition_penalty.unwrap_or(1.0),
        seed,
        ..EngineSamplingParams::default()
    };
    apply_sampling(tokenizer, controls, &mut sampling)?;
    Ok(sampling)
}

/// Applies public controls once, preserving model biases.
///
/// Temperature, top-k, top-p, min-p, and repetition penalty override the
/// caller's defaults only when the request sets them. Frequency and presence
/// penalties, `min_tokens`, `ignore_eos`, and every logprob, allowed-token, and
/// bad-word field are replaced from the request. Biases already in
/// `logit_bias` are summed per token with the request's biases.
///
/// Must be called once per request: a second call would add the request
/// biases again.
///
/// # Errors
///
/// Fails when a bad word cannot be tokenized or the result fails
/// `SamplingParams::validate`. `sampling` may be partially updated on error.
fn apply_sampling(
    tokenizer: &HuggingFaceTokenizer,
    controls: &SamplingParams,
    sampling: &mut EngineSamplingParams,
) -> Result<(), TokenizeError> {
    sampling.temperature = controls.temperature.unwrap_or(sampling.temperature);
    sampling.top_k = controls.top_k.unwrap_or(sampling.top_k);
    sampling.top_p = controls.top_p.unwrap_or(sampling.top_p);
    sampling.min_p = controls.min_p.unwrap_or(sampling.min_p);
    sampling.repetition_penalty = controls
        .repetition_penalty
        .unwrap_or(sampling.repetition_penalty);
    sampling.ignore_eos = controls.ignore_eos;
    sampling.min_tokens = controls.min_tokens.unwrap_or(0) as usize;
    sampling.frequency_penalty = controls.frequency_penalty.unwrap_or(0.0);
    sampling.presence_penalty = controls.presence_penalty.unwrap_or(0.0);
    // The ordered map yields `logit_bias` sorted by token ID.
    let mut biases = sampling
        .logit_bias
        .drain(..)
        .collect::<std::collections::BTreeMap<_, _>>();
    if let Some(request) = &controls.logit_bias {
        for (&token, &bias) in request {
            *biases.entry(token).or_insert(0.0) += bias;
        }
    }
    sampling.logit_bias = biases.into_iter().collect();
    sampling.return_logprobs = controls.logprobs.is_some();
    sampling.n_logprobs = logprob_count(controls.logprobs);
    sampling.return_prompt_logprobs = controls.prompt_logprobs.is_some();
    sampling.n_prompt_logprobs = logprob_count(controls.prompt_logprobs);
    sampling.allowed_token_ids = controls.allowed_token_ids.clone();
    sampling.bad_words_ids =
        tokenize_bad_words(&controls.bad_words, tokenizer)?.unwrap_or_default();
    sampling.validate()?;
    Ok(())
}

/// Converts a wire logprob count into the engine count.
///
/// The chat request validation admits a non-negative count or `-1`, which
/// requests every candidate and maps to `u32::MAX`; an absent count is zero.
fn logprob_count(value: Option<i32>) -> u32 {
    value.map_or(0, |value| u32::try_from(value).unwrap_or(u32::MAX))
}

/// Converts bad-word strings into sorted, deduplicated token-ID sequences.
///
/// Each word is encoded as written. It is also encoded with a single leading
/// space (after trimming existing leading whitespace), and that non-empty
/// variant is kept only when the as-written encoding is empty, or when both
/// encodings have the same length and different first tokens. Empty
/// encodings are dropped. Returns `None` when no sequence remains.
fn tokenize_bad_words(
    bad_words: &[String],
    tokenizer: &crate::profile::tokenizer::HuggingFaceTokenizer,
) -> std::result::Result<Option<Vec<Vec<u32>>>, crate::profile::tokenizer::TokenizerError> {
    if bad_words.is_empty() {
        return Ok(None);
    }
    let mut all_token_ids = Vec::new();
    for bad_word in bad_words {
        let without_space = tokenizer.encode(bad_word, false)?;
        let with_space = tokenizer.encode(&format!(" {}", bad_word.trim_start()), false)?;
        let keep_with_space = !with_space.is_empty()
            && (without_space.is_empty()
                || (with_space[0] != without_space[0] && with_space.len() == without_space.len()));
        if !without_space.is_empty() {
            all_token_ids.push(without_space);
        }
        if keep_with_space {
            all_token_ids.push(with_space);
        }
    }
    all_token_ids.sort();
    all_token_ids.dedup();
    Ok((!all_token_ids.is_empty()).then_some(all_token_ids))
}

/// Resolves the effective `max_tokens` for generation.
///
/// The request's `user_max_tokens` replaces `default_max_tokens` when set, and
/// when `max_model_len` is known the result is capped by the remaining
/// context, `max_model_len - prompt_len`. Returns `u32::MAX` when no limit is
/// known.
///
/// # Errors
///
/// Returns `text::Error::PromptTooLong` when `max_model_len` is set and
/// `prompt_len >= max_model_len`.
pub(super) fn resolve_max_tokens(
    user_max_tokens: Option<u32>,
    default_max_tokens: Option<u32>,
    max_model_len: Option<u32>,
    prompt_len: u32,
) -> crate::serving::text::Result<u32> {
    let model_max_tokens = match max_model_len {
        Some(max_model_len) if prompt_len >= max_model_len => {
            return Err(crate::serving::text::Error::PromptTooLong {
                max_model_len,
                prompt_len,
            });
        }
        Some(max_model_len) => Some(max_model_len - prompt_len),
        None => None,
    };

    let fallback_max_tokens = user_max_tokens.or(default_max_tokens);
    Ok([fallback_max_tokens, model_max_tokens]
        .into_iter()
        .flatten()
        .min()
        .unwrap_or(u32::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serving::text::Error;

    #[test]
    fn resolve_max_tokens_caps_by_model_len() {
        assert_eq!(
            resolve_max_tokens(Some(150), None, Some(200), 100).unwrap(),
            100
        );
    }

    #[test]
    fn resolve_max_tokens_uses_default_when_user_omits() {
        assert_eq!(
            resolve_max_tokens(None, Some(64), Some(200), 100).unwrap(),
            64
        );
    }

    #[test]
    fn resolve_max_tokens_no_limits_known_falls_back_to_u32_max() {
        assert_eq!(resolve_max_tokens(None, None, None, 100).unwrap(), u32::MAX);
    }

    #[test]
    fn resolve_max_tokens_prompt_too_long() {
        assert!(matches!(
            resolve_max_tokens(Some(10), None, Some(100), 100),
            Err(Error::PromptTooLong {
                max_model_len: 100,
                prompt_len: 100,
            })
        ));
    }
}
