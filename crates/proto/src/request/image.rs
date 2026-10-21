//! Image-generation parameters as a request's admission carries them.

use serde::{Deserialize, Serialize};

/// Image-generation parameters bounded for scheduler admission.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageParams {
    /// Number of denoising steps.
    pub steps: u16,
    /// Classifier-free guidance scale for text conditioning.
    pub cfg_text_scale: f32,
    /// Classifier-free guidance scale for image conditioning.
    pub cfg_img_scale: f32,
    /// Guidance renormalization strategy.
    pub cfg_renorm_type: CfgRenorm,
    /// Lower bound applied by guidance renormalization.
    pub cfg_renorm_min: f32,
    /// Inclusive normalized step interval where guidance is active.
    pub cfg_interval: (f32, f32),
    /// Shift applied to the diffusion timestep schedule.
    pub timestep_shift: f32,
    /// Output height in pixels.
    pub height: u32,
    /// Output width in pixels.
    pub width: u32,
    /// Optional deterministic diffusion seed.
    pub seed: Option<u64>,
}

impl ImageParams {
    /// Returns the classifier-free-guidance branch count implied by both axes.
    ///
    /// A scale of `1.0` (within a relative tolerance) disables its axis. The
    /// count is one conditional branch plus one per active axis, so it lies in
    /// `1..=3`.
    pub fn cfg_branch_count(&self) -> u8 {
        1 + u8::from(self.guides_text()) + u8::from(self.guides_image())
    }

    /// Whether text guidance is active: its text-unconditional branch is
    /// evaluated beside the conditioned one.
    pub fn guides_text(&self) -> bool {
        !scale_approx(self.cfg_text_scale, 1.0)
    }

    /// Whether image guidance is active: its image-unconditional branch is
    /// evaluated beside the conditioned one.
    pub fn guides_image(&self) -> bool {
        !scale_approx(self.cfg_img_scale, 1.0)
    }

    /// Text and image guidance scales whose [`Self::cfg_branch_count`] is
    /// `branches`: no guidance for one branch, text guidance for two, text
    /// and image guidance for three. `None` outside `1..=3`.
    pub fn guidance_scales(branches: u8) -> Option<(f32, f32)> {
        match branches {
            1 => Some((1.0, 1.0)),
            2 => Some((4.0, 1.0)),
            3 => Some((4.0, 2.0)),
            _ => None,
        }
    }
}

/// Returns whether two guidance scales are equal within relative tolerance.
fn scale_approx(left: f32, right: f32) -> bool {
    (left - right).abs() <= 1.0e-6_f32 * left.abs().max(right.abs()).max(1.0)
}

/// Validation failure returned by [`ImageParams::validate`].
///
/// Each variant names the offending field and the bound it violated so the
/// caller can report a precise input error before scheduling diffusion work.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ImageParamsError {
    /// Denoising step count is zero.
    #[error("steps must be positive")]
    Steps,
    /// Image height is zero or not a multiple of the grid.
    #[error("height must be a non-zero multiple of {multiple}, got {got}")]
    Height {
        /// Supplied height.
        got: u32,
        /// Required height alignment.
        multiple: u32,
    },
    /// Image width is zero or not a multiple of the grid.
    #[error("width must be a non-zero multiple of {multiple}, got {got}")]
    Width {
        /// Supplied width.
        got: u32,
        /// Required width alignment.
        multiple: u32,
    },
    /// A guidance scale is non-finite or negative.
    #[error("{field} must be finite and non-negative, got {got}")]
    CfgScale {
        /// Name of the invalid scale.
        field: &'static str,
        /// Supplied scale.
        got: f32,
    },
    /// A floating-point parameter is NaN or infinite.
    #[error("{field} must be finite, got {got}")]
    NonFinite {
        /// Name of the invalid parameter.
        field: &'static str,
        /// Supplied non-finite value.
        got: f32,
    },
    /// Guidance interval endpoints are descending.
    #[error("cfg_interval must be an ordered pair, got ({lo}, {hi})")]
    CfgIntervalOrder {
        /// Supplied lower endpoint.
        lo: f32,
        /// Supplied upper endpoint.
        hi: f32,
    },
}

/// Classifier-free guidance renormalization strategy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CfgRenorm {
    /// Leaves classifier-free guidance unnormalized.
    None,
    /// Renormalizes the combined guidance prediction globally.
    #[default]
    Global,
    /// Renormalizes against the text-conditioned prediction.
    TextChannel,
}

impl CfgRenorm {
    /// Returns the stable wire name for this strategy.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Global => "global",
            Self::TextChannel => "text_channel",
        }
    }
}

impl std::str::FromStr for CfgRenorm {
    type Err = CfgRenormParseError;

    /// Parses a stable guidance-renormalization wire name.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "none" => Ok(Self::None),
            "global" => Ok(Self::Global),
            "text_channel" => Ok(Self::TextChannel),
            _ => Err(CfgRenormParseError(value.to_owned())),
        }
    }
}

impl std::fmt::Display for CfgRenorm {
    /// Writes the stable guidance-renormalization wire name.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Error returned for an unsupported guidance renormalization string.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unsupported CFG renormalization mode {0:?}")]
pub struct CfgRenormParseError(String);

impl ImageParams {
    /// Required divisibility for each image dimension.
    pub const DIM_MULTIPLE: u32 = 16;

    /// Validates diffusion parameters before they reach the worker.
    ///
    /// Callers include `GenerationRequest::validate` and the worker IPC
    /// `NewRequest::validate`. `height` and `width` must each be non-zero
    /// and a multiple of `DIM_MULTIPLE`. The model's own limits bound the
    /// image's extent and the schedule's length at admission. The first
    /// violation found is returned.
    pub fn validate(&self) -> Result<(), ImageParamsError> {
        if self.steps == 0 {
            return Err(ImageParamsError::Steps);
        }

        // Both dimensions obey the same positive, aligned grid contract.
        let check_dim = |got: u32| -> bool { got != 0 && got.is_multiple_of(Self::DIM_MULTIPLE) };
        if !check_dim(self.height) {
            return Err(ImageParamsError::Height {
                got: self.height,
                multiple: Self::DIM_MULTIPLE,
            });
        }
        if !check_dim(self.width) {
            return Err(ImageParamsError::Width {
                got: self.width,
                multiple: Self::DIM_MULTIPLE,
            });
        }

        // Each guidance axis accepts finite non-negative scaling.
        for (field, scale) in [
            ("cfg_text_scale", self.cfg_text_scale),
            ("cfg_img_scale", self.cfg_img_scale),
        ] {
            if !scale.is_finite() || scale < 0.0 {
                return Err(ImageParamsError::CfgScale { field, got: scale });
            }
        }

        // Remaining continuous parameters must be finite before interval order
        // is evaluated.
        for (field, value) in [
            ("cfg_renorm_min", self.cfg_renorm_min),
            ("cfg_interval.0", self.cfg_interval.0),
            ("cfg_interval.1", self.cfg_interval.1),
            ("timestep_shift", self.timestep_shift),
        ] {
            if !value.is_finite() {
                return Err(ImageParamsError::NonFinite { field, got: value });
            }
        }

        if self.cfg_interval.0 > self.cfg_interval.1 {
            return Err(ImageParamsError::CfgIntervalOrder {
                lo: self.cfg_interval.0,
                hi: self.cfg_interval.1,
            });
        }
        Ok(())
    }
}

impl Default for ImageParams {
    /// Returns a bounded single-image diffusion configuration.
    fn default() -> Self {
        Self {
            steps: 50,
            cfg_text_scale: 4.0,
            cfg_img_scale: 1.0,
            cfg_renorm_type: CfgRenorm::Global,
            cfg_renorm_min: 0.0,
            cfg_interval: (0.0, 1.0),
            timestep_shift: 1.0,
            height: 512,
            width: 512,
            seed: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_params_default_is_valid() {
        assert!(ImageParams::default().validate().is_ok());
    }

    /// The scales of a branch count evaluate that many guidance branches.
    #[test]
    fn guidance_scales_realize_their_branch_count() {
        for branches in 1..=3 {
            let (cfg_text_scale, cfg_img_scale) = ImageParams::guidance_scales(branches).unwrap();
            let image = ImageParams {
                cfg_text_scale,
                cfg_img_scale,
                ..ImageParams::default()
            };
            assert_eq!(image.cfg_branch_count(), branches);
            assert!(image.validate().is_ok());
        }
        assert_eq!(ImageParams::guidance_scales(0), None);
        assert_eq!(ImageParams::guidance_scales(4), None);
    }

    #[test]
    fn image_params_validate_rejects_out_of_bounds() {
        let bad_steps = ImageParams {
            steps: 0,
            ..Default::default()
        };
        assert!(matches!(bad_steps.validate(), Err(ImageParamsError::Steps)));

        let zero_height = ImageParams {
            height: 0,
            ..Default::default()
        };
        assert!(matches!(
            zero_height.validate(),
            Err(ImageParamsError::Height { .. })
        ));

        let unaligned_width = ImageParams {
            width: 513, // not a multiple of DIM_MULTIPLE
            ..Default::default()
        };
        assert!(matches!(
            unaligned_width.validate(),
            Err(ImageParamsError::Width { .. })
        ));

        let nan_scale = ImageParams {
            cfg_text_scale: f32::NAN,
            ..Default::default()
        };
        assert!(matches!(
            nan_scale.validate(),
            Err(ImageParamsError::CfgScale { .. })
        ));

        let neg_scale = ImageParams {
            cfg_img_scale: -1.0,
            ..Default::default()
        };
        assert!(matches!(
            neg_scale.validate(),
            Err(ImageParamsError::CfgScale { .. })
        ));
    }
}
