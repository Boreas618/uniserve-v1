//! The Qwen3-VL text encoder as H3 conditions on it.
//!
//! The text encoder's configuration is the Qwen3-VL checkpoint's own ([`qwen3::Config`]); what
//! H3 adds is how much of it conditions a request. H3 reads the hidden state after the first
//! [`RETAINED_LAYERS`] decoder layers of a deeper checkpoint, a depth the checkpoint's
//! configuration does not state, and adds DeepStack feature `j` after retained layer `j`.

use nn::config::Fields;
use serde_json::Value;

use crate::{Error, Result};

/// The decoder layers whose hidden state H3 conditions on.
///
/// The text encoder checkpoint is a 64-layer Qwen3-VL; H3 reads the state after layer 50. The
/// checkpoint's configuration does not state how many layers a consumer keeps.
pub const RETAINED_LAYERS: u32 = 50;

/// Read `text_encoder/config.json` and check that H3's truncation fits it: the checkpoint holds
/// the retained layers, and every DeepStack feature has a retained layer to follow.
pub(super) fn text_encoder(metadata: &Value) -> Result<qwen3::Config> {
    let config = qwen3::Config::parse(&Fields::of("text_encoder/config.json", metadata)?)?;
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
        let config =
            text_encoder(&crate::config::tests_support::text_encoder()).expect("the released one");
        assert!(config.text.num_hidden_layers >= RETAINED_LAYERS);

        let mut shallow = crate::config::tests_support::text_encoder();
        shallow["text_config"]["num_hidden_layers"] = serde_json::json!(RETAINED_LAYERS - 1);
        let error = text_encoder(&shallow).expect_err("too few layers");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("keeps")),
            "{error}"
        );
    }
}
