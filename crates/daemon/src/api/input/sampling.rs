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
/// stream chunk has no field for them. `max_logprobs` is how many candidates of
/// a position the workers compute, which a logprob count of the request may not
/// exceed ([`logprob_count`]).
///
/// # Errors
///
/// Fails when a `logit_bias` key is not a token ID, a logprob count asks for
/// every candidate of the vocabulary, a bad word cannot be tokenized or the
/// parameters fail `SamplingParams::validate`.
pub(super) fn lower(
    request: &ChatCompletionRequest,
    defaults: &SamplingDefaults,
    tokenizer: &HuggingFaceTokenizer,
    max_logprobs: u32,
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
        seed: Some(super::request_seed(request.seed.map(|seed| seed as u64))),
        ignore_eos: request.ignore_eos,
        min_tokens: request.min_tokens.unwrap_or(0) as usize,
        frequency_penalty: request.frequency_penalty.unwrap_or(0.0),
        presence_penalty: request.presence_penalty.unwrap_or(0.0),
        logit_bias,
        return_logprobs: logprobs.is_some(),
        n_logprobs: logprob_count(logprobs, "top_logprobs", max_logprobs)?,
        return_prompt_logprobs: prompt_logprobs.is_some(),
        n_prompt_logprobs: logprob_count(prompt_logprobs, "prompt_logprobs", max_logprobs)?,
        allowed_token_ids: request.allowed_token_ids.clone(),
        bad_words_ids: tokenize_bad_words(bad_words, tokenizer)?,
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

/// The engine's candidate count of the request field `parameter`, which states `value`.
///
/// An absent count is zero, and a non-negative count is itself. `-1` asks for every candidate
/// of the vocabulary, as OpenAI's and vLLM's APIs read it: the workers compute `max_logprobs`
/// candidates of each position, a count fixed by their captured graphs, so no deployment
/// computes the vocabulary and the request is refused naming the count it may ask for. A count
/// above `max_logprobs` is refused by the workers' limits (`GenerationLimits::covers_logprobs`).
///
/// # Errors
///
/// `TokenizeError::InvalidParameter`, naming `parameter`, for `-1` or a count below it.
fn logprob_count(
    value: Option<i32>,
    parameter: &'static str,
    max_logprobs: u32,
) -> Result<u32, TokenizeError> {
    let Some(value) = value else {
        return Ok(0);
    };
    u32::try_from(value).map_err(|_| TokenizeError::InvalidParameter {
        param: Some(parameter),
        message: if value == -1 {
            format!(
                "{parameter} of -1 asks for every candidate of the vocabulary; this deployment \
                 computes at most {max_logprobs} candidates of a position"
            )
        } else {
            format!("{parameter} must be a non-negative value or -1")
        },
    })
}

/// The token-ID sequences a request's bad words ban, sorted and deduplicated.
///
/// A word is banned wherever it occurs as text: written as the request states it, and after a
/// word boundary, where a tokenizer whose pre-tokenizer marks word starts (byte-level or
/// metaspace) encodes it with its leading space. Each word therefore bans its encoding as
/// written and its encoding with one leading space (its own leading whitespace replaced), the
/// non-empty ones; the two are one sequence for a tokenizer that marks no word start.
fn tokenize_bad_words(
    bad_words: &[String],
    tokenizer: &crate::api::tokenizer::HuggingFaceTokenizer,
) -> std::result::Result<Vec<Vec<u32>>, crate::api::tokenizer::TokenizerError> {
    let mut sequences = Vec::with_capacity(bad_words.len() * 2);
    for bad_word in bad_words {
        sequences.push(tokenizer.encode(bad_word, false)?);
        sequences.push(tokenizer.encode(&format!(" {}", bad_word.trim_start()), false)?);
    }
    sequences.retain(|sequence| !sequence.is_empty());
    sequences.sort();
    sequences.dedup();
    Ok(sequences)
}

/// Resolves the effective `max_tokens` for generation.
///
/// The request's `user_max_tokens` replaces `default_max_tokens` when set, and the result is
/// capped by the remaining context, `max_model_len - prompt_len`.
///
/// # Errors
///
/// Returns `TokenizeError::PromptTooLong` when `prompt_len >= max_model_len`: the prompt leaves
/// no room for a generated token.
pub(super) fn resolve_max_tokens(
    user_max_tokens: Option<u32>,
    default_max_tokens: Option<u32>,
    max_model_len: u32,
    prompt_len: u32,
) -> Result<u32, TokenizeError> {
    let room = max_model_len
        .checked_sub(prompt_len)
        .filter(|room| *room > 0)
        .ok_or(TokenizeError::PromptTooLong {
            max_model_len,
            prompt_len,
        })?;
    Ok(user_max_tokens
        .or(default_max_tokens)
        .map_or(room, |requested| requested.min(room)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_max_tokens_caps_by_model_len() {
        assert_eq!(resolve_max_tokens(Some(150), None, 200, 100).unwrap(), 100);
    }

    #[test]
    fn resolve_max_tokens_uses_default_when_user_omits() {
        assert_eq!(resolve_max_tokens(None, Some(64), 200, 100).unwrap(), 64);
    }

    /// A request that states no length, of a model that states no default, generates to the end
    /// of the context.
    #[test]
    fn resolve_max_tokens_fills_the_context_without_a_stated_length() {
        assert_eq!(resolve_max_tokens(None, None, 200, 100).unwrap(), 100);
    }

    #[test]
    fn resolve_max_tokens_prompt_too_long() {
        assert!(matches!(
            resolve_max_tokens(Some(10), None, 100, 100),
            Err(TokenizeError::PromptTooLong {
                max_model_len: 100,
                prompt_len: 100,
            })
        ));
    }

    /// A bad word bans its encoding as written and at a word boundary, once each.
    #[test]
    fn a_bad_word_bans_its_encodings_as_written_and_after_a_space() {
        let tokenizer = crate::api::test_support::configured_tokenizer();
        let mut expected = ["ab", " ab", "  ab"]
            .map(|text| tokenizer.encode(text, false).unwrap())
            .to_vec();
        expected.retain(|sequence| !sequence.is_empty());
        expected.sort();
        expected.dedup();

        let banned = tokenize_bad_words(&["ab".to_owned(), "  ab".to_owned()], &tokenizer).unwrap();
        assert_eq!(banned, expected);
        assert!(tokenize_bad_words(&[], &tokenizer).unwrap().is_empty());
    }
}
