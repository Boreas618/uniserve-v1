//! The Qwen3-VL text encoder as H3 conditions on it.
//!
//! The text encoder's configuration is the Qwen3-VL checkpoint's own ([`crate::qwen3::Config`]); what
//! H3 adds is how much of it conditions a request. H3 reads the hidden state after the first
//! [`RETAINED_LAYERS`] decoder layers of a deeper checkpoint, a depth the checkpoint's
//! configuration does not state, and adds DeepStack feature `j` after retained layer `j`.

use nn::config::Fields;
use serde_json::Value;

use crate::minimax_h3::{Error, Result};

/// The decoder layers of the text encoder H3 conditions on: the hidden state after this many
/// layers is the prompt's features, with no final norm.
pub const RETAINED_LAYERS: u32 = 50;

/// The text encoder's configuration, from `text_encoder/config.json`.
///
/// # Errors
///
/// A checkpoint shallower than the retained layers, without a vision tower, or with more
/// DeepStack features than retained layers.
pub(super) fn text_encoder(metadata: &Value) -> Result<crate::qwen3::Config> {
    let config = crate::qwen3::Config::parse(&Fields::of("text_encoder/config.json", metadata)?)?;

    if RETAINED_LAYERS > config.text.num_hidden_layers {
        return Err(Error::config(format!(
            "H3 keeps {RETAINED_LAYERS} of the text encoder's {} layers",
            config.text.num_hidden_layers
        )));
    }
    let vision = config.vision.as_ref().ok_or_else(|| {
        Error::config("H3's text encoder is a Qwen3-VL model with a vision tower".to_owned())
    })?;

    let features = vision.deepstack_visual_indexes.len();
    if features > RETAINED_LAYERS as usize {
        return Err(Error::config(format!(
            "{features} DeepStack features, but H3 retains {RETAINED_LAYERS} layers"
        )));
    }
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_released_text_encoder_holds_the_retained_layers() {
        let released = crate::minimax_h3::config::tests_support::text_encoder();
        let config = text_encoder(&released).unwrap();
        assert!(config.text.num_hidden_layers >= RETAINED_LAYERS);

        let mut shallow = released;
        shallow["text_config"]["num_hidden_layers"] = serde_json::json!(RETAINED_LAYERS - 1);
        let error = text_encoder(&shallow).unwrap_err();
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("keeps")),
            "{error}"
        );
    }
}
