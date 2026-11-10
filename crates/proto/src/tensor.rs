//! The element type a tensor crosses a process boundary with.
//!
//! A product the scheduler declares (`ipc::TensorRef`), a relay column and a transfer locator
//! state their element type as [`DType`]. The kernel library's element type (`ops::DType`) is the
//! same vocabulary with the formats only a kernel operand has (FP8, NVFP4), and converts from this
//! one; the variants this one names carry the same numbers there, so the two never disagree on
//! what a number means.

use serde::{Deserialize, Serialize};

use crate::request::ModelDtype;

/// Element type of a tensor's backing storage.
///
/// The numbering is the kernel library's ABI numbering (`ops::DType`); a variant this type lacks
/// keeps its number there, so the numbers here have gaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(u8)]
pub enum DType {
    /// Unsigned 8-bit integer.
    U8 = 0,
    /// Signed 32-bit integer.
    I32 = 1,
    /// Signed 64-bit integer.
    I64 = 2,
    /// IEEE 754 half precision.
    F16 = 3,
    /// Brain floating-point half precision.
    #[serde(rename = "bf16")]
    BF16 = 4,
    /// IEEE 754 single precision.
    F32 = 5,
    /// Signed 16-bit integer.
    I16 = 8,
}

impl DType {
    /// Storage width of one element in bytes.
    pub const fn element_bytes(self) -> u64 {
        match self {
            Self::U8 => 1,
            Self::I16 | Self::F16 | Self::BF16 => 2,
            Self::I32 | Self::F32 => 4,
            Self::I64 => 8,
        }
    }

    /// The name a transfer locator spells the type with: the torch dtype name without the
    /// `torch.` prefix, as the worker's DLPack dtype writes it.
    pub const fn torch_name(self) -> &'static str {
        match self {
            Self::U8 => "uint8",
            Self::I16 => "int16",
            Self::I32 => "int32",
            Self::I64 => "int64",
            Self::F16 => "float16",
            Self::BF16 => "bfloat16",
            Self::F32 => "float32",
        }
    }
}

impl From<ModelDtype> for DType {
    /// The element type of a model's parameters and activations.
    fn from(dtype: ModelDtype) -> Self {
        match dtype {
            ModelDtype::Float16 => Self::F16,
            ModelDtype::BFloat16 => Self::BF16,
            ModelDtype::Float32 => Self::F32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_model_dtype_is_the_element_type_of_its_name() {
        for dtype in [
            ModelDtype::Float16,
            ModelDtype::BFloat16,
            ModelDtype::Float32,
        ] {
            assert_eq!(DType::from(dtype).torch_name(), dtype.as_str());
        }
    }
}
