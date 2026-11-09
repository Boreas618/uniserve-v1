//! A text checkpoint's control tokens, and what one request may be given to generate.

use std::path::Path;

use anyhow::{Context as _, Result};
use proto::request::{GenerationFeatures, GenerationLimits};

/// The control tokens a checkpoint states of itself: what begins its sequences and what ends them.
#[derive(Debug)]
pub(super) struct ControlTokens {
    /// The token that begins a sequence.
    pub(super) bos: u32,
    /// The tokens that terminate generation, the primary one first.
    pub(super) eos: Vec<u32>,
}

/// Reads the control tokens from the checkpoint the launch file names.
///
/// A launch file states where the checkpoint is, not what its vocabulary means, so the ids come
/// from the checkpoint's own files, in the order the engine reads them:
///
/// - `generation_config.json` states the generation defaults, and its `eos_token_id` is the primary
///   end of generation;
/// - `config.json` states the architecture's `bos_token_id` and `eos_token_id`, which may be one id
///   or a list;
/// - `tokenizer_config.json` names its special tokens and maps each to an id in
///   `added_tokens_decoder`.
///
/// A family may state no beginning-of-sequence token (Qwen3's `"bos_token_id": null`), and then the
/// beginning a text model has is its vocabulary's end of text, `end_of_text` as the family names it
/// ([`nn::family::TextCheckpoint::end_of_text`]), whose id is read from the tokenizer that names
/// it; it joins the eos list for the same reason: a completion that emits the vocabulary's end of
/// text must stop, and the engine acts only on the ids it is given.
///
/// The primary end of generation is `eos[0]`, because that is the token the engine gives a
/// completion that commits none (`crates/daemon/src/leader.rs::special_token_ids`, whose rule this
/// follows); the rest follow ascending.
pub(super) fn checkpoint_tokens(dir: &Path, end_of_text: Option<&str>) -> Result<ControlTokens> {
    let config = read_json(&dir.join("config.json"))?;
    // Both files are optional: a checkpoint may state its generation defaults in the architecture's
    // configuration alone, and an architecture that states a beginning states it there.
    let generation = read_json(&dir.join("generation_config.json")).ok();
    let tokenizer = read_json(&dir.join("tokenizer_config.json")).ok();
    let named = end_of_text;
    let end_of_text = tokenizer
        .as_ref()
        .zip(named)
        .and_then(|(value, content)| added_token(value, content));

    let bos = token_id(config.get("bos_token_id"))
        .or_else(|| {
            generation
                .as_ref()
                .and_then(|value| token_id(value.get("bos_token_id")))
        })
        .or(end_of_text)
        .with_context(|| {
            format!(
                "{} states no beginning-of-sequence token: `config.json` and \
                 `generation_config.json` state no `bos_token_id`, and `tokenizer_config.json` \
                 names no {}",
                dir.display(),
                named.unwrap_or("end-of-text token of the family")
            )
        })?;

    let mut eos = Vec::new();
    for value in [
        generation
            .as_ref()
            .and_then(|value| value.get("eos_token_id")),
        config.get("eos_token_id"),
    ]
    .into_iter()
    .flatten()
    {
        for id in token_ids(value) {
            if !eos.contains(&id) {
                eos.push(id);
            }
        }
    }
    if let Some(end_of_text) = end_of_text {
        if !eos.contains(&end_of_text) {
            eos.push(end_of_text);
        }
    }
    anyhow::ensure!(
        !eos.is_empty(),
        "{} states no end-of-generation token: neither `config.json` nor \
         `generation_config.json` states `eos_token_id`, and `tokenizer_config.json` names no \
         {}",
        dir.display(),
        named.unwrap_or("end-of-text token of the family")
    );
    Ok(ControlTokens { bos, eos })
}

/// Reads one JSON file of a checkpoint.
pub(super) fn read_json(path: &Path) -> Result<serde_json::Value> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

/// One token id, whether the file writes it as a number or as a list of numbers.
pub(super) fn token_id(value: Option<&serde_json::Value>) -> Option<u32> {
    let value = value?;
    match value {
        serde_json::Value::Number(number) => number.as_u64().and_then(|id| u32::try_from(id).ok()),
        serde_json::Value::Array(ids) => ids
            .first()
            .and_then(|id| id.as_u64())
            .and_then(|id| u32::try_from(id).ok()),
        _ => None,
    }
}

/// Every token id a field states, in the order it writes them; a single number is a list of one.
pub(super) fn token_ids(value: &serde_json::Value) -> Vec<u32> {
    match value {
        serde_json::Value::Number(number) => number
            .as_u64()
            .and_then(|id| u32::try_from(id).ok())
            .into_iter()
            .collect(),
        serde_json::Value::Array(ids) => ids
            .iter()
            .filter_map(|id| id.as_u64())
            .filter_map(|id| u32::try_from(id).ok())
            .collect(),
        _ => Vec::new(),
    }
}

/// The id a tokenizer gives the special token whose text is `content`.
///
/// The tokenizer states its added tokens as a map from id to token in `added_tokens_decoder`, and
/// older checkpoints state them as a list of `{id, content}` in `added_tokens`; both are read, so a
/// checkpoint of either shape resolves the same token.
pub(super) fn added_token(tokenizer: &serde_json::Value, content: &str) -> Option<u32> {
    if let Some(decoder) = tokenizer
        .get("added_tokens_decoder")
        .and_then(serde_json::Value::as_object)
    {
        for (id, token) in decoder {
            if token.get("content").and_then(serde_json::Value::as_str) == Some(content) {
                return id.parse::<u32>().ok();
            }
        }
    }
    tokenizer
        .get("added_tokens")
        .and_then(serde_json::Value::as_array)?
        .iter()
        .find(|token| token.get("content").and_then(serde_json::Value::as_str) == Some(content))
        .and_then(|token| token.get("id"))
        .and_then(serde_json::Value::as_u64)
        .and_then(|id| u32::try_from(id).ok())
}

/// What one request may be given to generate on a text model.
///
/// This model understands a prompt and generates tokens, and it encodes no vision features and
/// denoises no latents, so the served feature is `GenerationFeatures::UNDERSTANDING` and every
/// **capacity** is zero: `max_latent_units`, `latent_bytes_per_unit`, `max_vae_grid_tokens`,
/// `max_vit_grid_tokens`, `max_latent_feature_bytes`, `max_vision_feature_bytes`,
/// `commit_marker_tokens` and `encoder_cache_entries`. A capacity of zero declares no capacity at
/// all, which is what a model that encodes and denoises nothing has.
///
/// The two nonzero figures are **lower bounds that must be at least one**, not capacities, and they
/// are the Qwen3 profile's own (`crates/frontend/src/model/generation.rs`, its `Qwen3` arm:
/// `GenerationLimits { latent_downsample: 1, max_cfg_branches: 1, ..Default::default() }`):
/// `latent_downsample` is a divisor, so zero would divide a pixel grid by it, and
/// `max_cfg_branches` counts the branches a request may run, so one branch is the minimum a
/// deployment admits rather than a media fact about this model.
pub(super) fn understanding_limits() -> GenerationLimits {
    GenerationLimits {
        features: GenerationFeatures::UNDERSTANDING,
        latent_downsample: 1,
        max_cfg_branches: 1,
        ..GenerationLimits::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A checkpoint directory holding the files a test states.
    fn checkpoint(files: &[(&str, &str)]) -> tempfile::TempDir {
        let directory = tempfile::tempdir().expect("a temporary checkpoint directory");
        for (name, body) in files {
            std::fs::write(directory.path().join(name), body).expect("writing a checkpoint file");
        }
        directory
    }

    #[test]
    fn a_checkpoint_without_a_beginning_begins_with_its_end_of_text() {
        let directory = checkpoint(&[
            (
                "config.json",
                r#"{"bos_token_id": null, "eos_token_id": 151645}"#,
            ),
            (
                "tokenizer_config.json",
                r#"{"added_tokens_decoder": {
                    "151643": {"content": "<|endoftext|>"},
                    "151645": {"content": "<|im_end|>"}
                }}"#,
            ),
        ]);

        let control = checkpoint_tokens(directory.path(), Some("<|endoftext|>"))
            .expect("the checkpoint's control tokens");

        // The architecture states no beginning, so the tokenizer's own end of text is the
        // beginning a text model has; the stated end of generation comes first and the vocabulary's
        // end of text follows it, because a completion that emits it must stop.
        assert_eq!(control.bos, 151643);
        assert_eq!(control.eos, vec![151645, 151643]);
    }

    #[test]
    fn the_stated_beginning_wins_and_the_generation_default_orders_the_ends() {
        let directory = checkpoint(&[
            (
                "config.json",
                r#"{"bos_token_id": 5, "eos_token_id": [3, 2]}"#,
            ),
            ("generation_config.json", r#"{"eos_token_id": 2}"#),
        ]);

        let control = checkpoint_tokens(directory.path(), Some("<|endoftext|>"))
            .expect("the checkpoint's control tokens");

        assert_eq!(
            control.bos, 5,
            "the architecture's own beginning is the one it means"
        );
        // `eos[0]` is what the engine gives a completion that commits none, so the generation
        // default leads and the architecture's remaining ids follow, each stated once.
        assert_eq!(control.eos, vec![2, 3]);
    }

    #[test]
    fn a_checkpoint_that_states_no_end_of_generation_is_refused_by_name() {
        let directory = checkpoint(&[("config.json", r#"{"bos_token_id": 5}"#)]);

        let error = checkpoint_tokens(directory.path(), Some("<|endoftext|>"))
            .expect_err("an unstated eos token");
        let message = format!("{error:#}");

        assert!(
            message.contains("states no end-of-generation token"),
            "the refusal names what is missing: {message}"
        );
        assert!(
            message.contains("config.json"),
            "the refusal names the file that was read: {message}"
        );
    }
}
