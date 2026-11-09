//! The Qwen3-VL text encoder as H3 conditions on it, and the pixel normalization its processor
//! applies.
//!
//! The text encoder's configuration is the Qwen3-VL checkpoint's own ([`qwen3::VlConfig`]); what
//! H3 adds is how much of it conditions a request. H3 reads the hidden state after the first
//! [`RETAINED_LAYERS`] decoder layers of a deeper checkpoint, a depth the checkpoint's
//! configuration does not state, and adds DeepStack feature `j` after retained layer `j`.

use nn::config::Fields;
use qwen3::VlConfig;
use serde_json::Value;

use crate::{Error, Result};

/// The decoder layers whose hidden state H3 conditions on.
///
/// The text encoder checkpoint is a 64-layer Qwen3-VL; H3 reads the state after layer 50. The
/// checkpoint's configuration does not state how many layers a consumer keeps.
pub const RETAINED_LAYERS: u32 = 50;

/// How the checkpoint's processor normalizes a pixel before the vision tower patches it.
///
/// Each RGB channel maps `x` in `[0, 255]` to `(x / 255 - mean) / std`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PixelConfig {
    /// Per-channel means.
    pub mean: [f32; 3],
    /// Per-channel standard deviations.
    pub std: [f32; 3],
}

impl PixelConfig {
    /// Read a processor's `image_mean` and `image_std`.
    pub fn parse(values: &Fields<'_>) -> Result<Self> {
        let owner = values.owner();
        let mean = values.fixed_numbers("image_mean", 3)?;
        let std = values.fixed_numbers("image_std", 3)?;
        let mean: [f32; 3] = [mean[0] as f32, mean[1] as f32, mean[2] as f32];
        let std: [f32; 3] = [std[0] as f32, std[1] as f32, std[2] as f32];
        if std.iter().any(|value| *value <= 0.0) {
            return Err(Error::config(format!(
                "{owner}.image_std must be positive, got {std:?}"
            )));
        }
        Ok(Self { mean, std })
    }
}

/// Read `text_encoder/config.json` and check that H3's truncation fits it: the checkpoint holds
/// the retained layers, and every DeepStack feature has a retained layer to follow.
pub(super) fn text_encoder(metadata: &Value) -> Result<VlConfig> {
    let config = VlConfig::parse(&Fields::of("text_encoder/config.json", metadata)?)?;
    if RETAINED_LAYERS > config.text.num_hidden_layers {
        return Err(Error::config(format!(
            "H3 keeps {RETAINED_LAYERS} of the text encoder's {} layers",
            config.text.num_hidden_layers
        )));
    }
    let features = config.vision.deepstack_visual_indexes.len();
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

    #[test]
    fn a_processor_states_three_positive_deviations() {
        let mut negative = crate::config::tests_support::processor();
        negative["image_std"] = serde_json::json!([0.5, 0.0, 0.5]);
        let error = PixelConfig::parse(
            &Fields::of("processor/preprocessor_config.json", &negative).expect("an object"),
        )
        .expect_err("a zero deviation is refused");
        assert!(
            matches!(error, Error::Config(ref problem) if problem.contains("image_std")),
            "{error}"
        );
    }
}
