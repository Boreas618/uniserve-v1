//! Named output resolutions.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
/// Named output canvas accepted by multimodal profiles.
pub enum ResolutionName {
    /// Square 1:1 canvas.
    #[default]
    #[serde(rename = "1:1")]
    Square,
    /// Landscape 16:9 canvas.
    #[serde(rename = "16:9")]
    Landscape16x9,
    /// Model-specific 1.5K canvas preset.
    #[serde(rename = "1.5K")]
    OnePointFiveK,
    /// Portrait 9:16 canvas.
    #[serde(rename = "9:16")]
    Portrait9x16,
    /// Landscape 3:2 canvas.
    #[serde(rename = "3:2")]
    Landscape3x2,
    /// Portrait 2:3 canvas.
    #[serde(rename = "2:3")]
    Portrait2x3,
    /// Landscape 4:3 canvas.
    #[serde(rename = "4:3")]
    Landscape4x3,
    /// Portrait 3:4 canvas.
    #[serde(rename = "3:4")]
    Portrait3x4,
    /// Portrait 1:2 canvas.
    #[serde(rename = "1:2")]
    Portrait1x2,
    /// Landscape 2:1 canvas.
    #[serde(rename = "2:1")]
    Landscape2x1,
    /// Portrait 1:3 canvas.
    #[serde(rename = "1:3")]
    Portrait1x3,
    /// Landscape 3:1 canvas.
    #[serde(rename = "3:1")]
    Landscape3x1,
    /// Landscape 21:9 canvas.
    #[serde(rename = "21:9")]
    Landscape21x9,
}

impl std::str::FromStr for ResolutionName {
    type Err = ResolutionError;

    /// Parses a resolution's wire name, its serialized form, also accepting
    /// the lowercase spelling `1.5k`.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "1:1" => Ok(Self::Square),
            "16:9" => Ok(Self::Landscape16x9),
            "1.5K" | "1.5k" => Ok(Self::OnePointFiveK),
            "9:16" => Ok(Self::Portrait9x16),
            "3:2" => Ok(Self::Landscape3x2),
            "2:3" => Ok(Self::Portrait2x3),
            "4:3" => Ok(Self::Landscape4x3),
            "3:4" => Ok(Self::Portrait3x4),
            "1:2" => Ok(Self::Portrait1x2),
            "2:1" => Ok(Self::Landscape2x1),
            "1:3" => Ok(Self::Portrait1x3),
            "3:1" => Ok(Self::Landscape3x1),
            "21:9" => Ok(Self::Landscape21x9),
            _ => Err(ResolutionError::UnsupportedName(value.to_owned())),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
/// An output-resolution name that names no resolution.
pub enum ResolutionError {
    /// A name that is not a known resolution.
    #[error("unsupported image resolution {0:?}")]
    UnsupportedName(String),
}
