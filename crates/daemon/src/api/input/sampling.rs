//! Token sampling resolution: a chat request's sampling controls over the
//! model's defaults ([`lower`]), lowered once into the engine's
//! `SamplingParams`, and its generation length ([`resolve_max_tokens`]).

use proto::request::SamplingParams;

use super::TokenizeError;
use crate::api::model::SamplingDefaults;
use crate::api::protocol::ChatCompletionRequest;
use crate::api::tokenizer::HuggingFaceTokenizer;

/// The engine's sampling parameters of `request`: each control it sets over
/// the model's `defaults`, its logit biases keyed by token ID, and its bad
/// words tokenized with `tokenizer`.
///
/// Temperature, top-k, top-p, min-p and repetition penalty fall back to the
/// model's defaults, then to the neutral value; every other control is the
/// request's own. A streamed request's prompt logprobs are not computed: a
/// stream chunk has no field for them.
///
/// # Errors
///
/// Fails when a `logit_bias` key is not a token ID, a bad word cannot be
/// tokenized or the parameters fail `SamplingParams::validate`.
pub(super) fn lower(
    request: &ChatCompletionRequest,
    defaults: &SamplingDefaults,
    tokenizer: &HuggingFaceTokenizer,
) -> Result<SamplingParams, TokenizeError> {
    let logit_bias = logit_bias(request.logit_bias.as_ref())?;
    let logprobs = request
        .logprobs
        .then_some(request.top_logprobs.unwrap_or(0));
    let prompt_logprobs = request.prompt_logprobs.filter(|_| !request.stream);
    let bad_words = request.bad_words.as_deref().unwrap_or_default();
    let sampling = SamplingParams {
        temperature: request.temperature.or(defaults.temperature).unwrap_or(1.0),
        top_k: request.top_k.or(defaults.top_k).unwrap_or(0),
        top_p: request.top_p.or(defaults.top_p).unwrap_or(1.0),
        min_p: request.min_p.or(defaults.min_p).unwrap_or(0.0),
        repetition_penalty: request
            .repetition_penalty
            .or(defaults.repetition_penalty)
            .unwrap_or(1.0),
        // The seed's signed bit pattern seeds the engine.
        seed: request.seed.map(|seed| seed as u64),
        ignore_eos: request.ignore_eos,
        min_tokens: request.min_tokens.unwrap_or(0) as usize,
        frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
        presence_penalty: request.presence_penalty.unwrap_or(0.0),
        logit_bias,
        return_logprobs: logprobs.is_some(),
        n_logprobs: logprob_count(logprobs),
        return_prompt_logprobs: prompt_logprobs.is_some(),
        n_prompt_logprobs: logprob_count(prompt_logprobs),
        allowed_token_ids: request.allowed_token_ids.clone(),
        bad_words_ids: tokenize_bad_words(bad_words, tokenizer)?.unwrap_or_default(),
        ..SamplingParams::default()
    };
    sampling.validate()?;
    Ok(sampling)
}

/// Converts OpenAI-style `logit_bias`, whose JSON object keys are decimal
/// token IDs, into `(token_id, bias)` pairs sorted by token ID.
///
/// Token IDs are not checked against the vocabulary here.
fn logit_bias(
    logit_bias: Option<&std::collections::HashMap<String, f32>>,
) -> Result<Vec<(u32, f32)>, TokenizeError> {
    let mut pairs = logit_bias
        .into_iter()
        .flatten()
        .map(|(key, value)| {
            key.parse()
                .map(|token| (token, *value))
                .map_err(|_| TokenizeError::InvalidParameter {
                    param: Some("logit_bias"),
                    message: format!(
                        "Invalid key in 'logit_bias': '{key}' is not a valid token ID. \
                         Token IDs must be non-negative integers."
                    ),
                })
        })
        .collect::<Result<Vec<(u32, f32)>, _>>()?;
    pairs.sort_by_key(|(token, _)| *token);
    Ok(pairs)
}

/// Converts a wire logprob count into the engine count.
///
/// The chat request validation admits a non-negative count or `-1`, which
/// asks for every candidate of the vocabulary and maps to `u32::MAX`; an
/// absent count is zero. The deployment's `max_logprobs` then bounds the
/// count (`GenerationLimits::covers_logprobs`), as vLLM's does: a deployment
/// computes that many candidates a position, so a larger count, `-1`
/// included, is refused.
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
    tokenizer: &crate::api::tokenizer::HuggingFaceTokenizer,
) -> std::result::Result<Option<Vec<Vec<u32>>>, crate::api::tokenizer::TokenizerError> {
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
/// Returns `TokenizeError::PromptTooLong` when `max_model_len` is set and
/// `prompt_len >= max_model_len`.
pub(super) fn resolve_max_tokens(
    user_max_tokens: Option<u32>,
    default_max_tokens: Option<u32>,
    max_model_len: Option<u32>,
    prompt_len: u32,
) -> Result<u32, TokenizeError> {
    let model_max_tokens = match max_model_len {
        Some(max_model_len) if prompt_len >= max_model_len => {
            return Err(TokenizeError::PromptTooLong {
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
            Err(TokenizeError::PromptTooLong {
                max_model_len: 100,
                prompt_len: 100,
            })
        ));
    }
}
