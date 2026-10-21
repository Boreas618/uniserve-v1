//! The numeric data type a deployment loads its model in.

use serde::{Deserialize, Serialize};

/// Effective model dtype reported after runtime configuration resolves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelDtype {
    /// IEEE 754 half precision.
    #[serde(rename = "float16")]
    Float16,
    /// Brain floating-point half precision.
    #[serde(rename = "bfloat16")]
    BFloat16,
    /// IEEE 754 single precision.
    #[serde(rename = "float32")]
    Float32,
}

impl ModelDtype {
    /// Returns the stable wire name for this dtype.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Float16 => "float16",
            Self::BFloat16 => "bfloat16",
            Self::Float32 => "float32",
        }
    }

    /// Parses a supported wire name without allocating an error.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "float16" => Some(Self::Float16),
            "bfloat16" => Some(Self::BFloat16),
            "float32" => Some(Self::Float32),
            _ => None,
        }
    }
}

impl std::str::FromStr for ModelDtype {
    type Err = ModelDtypeParseError;

    /// Parses a stable model-dtype wire name.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value).ok_or_else(|| ModelDtypeParseError(value.to_owned()))
    }
}

impl std::fmt::Display for ModelDtype {
    /// Writes the stable model-dtype wire name.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Error returned for an unsupported model dtype string.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unsupported model dtype {0:?}")]
pub struct ModelDtypeParseError(String);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_dtype_uses_canonical_strings() {
        assert_eq!(
            serde_json::to_value(ModelDtype::Float16).unwrap(),
            serde_json::json!("float16")
        );
        assert_eq!(ModelDtype::parse("bfloat16"), Some(ModelDtype::BFloat16));
        assert_eq!(ModelDtype::parse("float32"), Some(ModelDtype::Float32));
    }
}
