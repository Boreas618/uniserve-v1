//! The element types the interfaces name, and the one table of their names.
//!
//! A deployment names element types in its configuration (`dtype`, `kv_cache_config.dtype`) and
//! the interfaces carry them by the same names: the PyTorch dtype names without the `torch.`
//! prefix, which the checkpoints and the reference implementations write. [`DType`] is that
//! vocabulary, with one name per element type ([`DType::name`]). An option that admits only some
//! element types is a subset of it whose values are validated element types
//! (`request::ModelDtype`, `request::PrefixCacheDtype`): each converts from a [`DType`] and
//! refuses the ones it does not admit, and spells itself through this table.
//!
//! A kernel operand's element type is the tensor library's, which also has the formats only an
//! operand has; the kernel library's package ABI numbers those (`ops::abi`).

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// An element type an interface names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DType {
    /// IEEE 754 half precision.
    F16,
    /// Brain floating-point half precision.
    BF16,
    /// IEEE 754 single precision.
    F32,
    /// Finite-only E4M3 8-bit floating point.
    F8E4M3,
}

impl DType {
    /// Every element type, in the order of [`DType`]'s variants.
    pub const ALL: [Self; 4] = [Self::F16, Self::BF16, Self::F32, Self::F8E4M3];

    /// The element type's name: the PyTorch dtype name without the `torch.` prefix.
    pub const fn name(self) -> &'static str {
        match self {
            Self::F16 => "float16",
            Self::BF16 => "bfloat16",
            Self::F32 => "float32",
            Self::F8E4M3 => "float8_e4m3fn",
        }
    }

    /// The element type named `name`; `None` for a name the vocabulary lacks.
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|dtype| dtype.name() == name)
    }

    /// Storage width of one element in bytes.
    pub const fn element_bytes(self) -> u64 {
        match self {
            Self::F8E4M3 => 1,
            Self::F16 | Self::BF16 => 2,
            Self::F32 => 4,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

/// A name the element-type vocabulary lacks, or an element type an option does not admit.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("unsupported {what} {name:?}; expected one of: {expected}")]
pub struct DTypeError {
    /// What was being read, such as `model dtype`.
    pub what: &'static str,
    /// The name that was read.
    pub name: String,
    /// The names admitted, comma-separated.
    pub expected: String,
}

impl DTypeError {
    /// The error of reading `name` as one of `admitted`, described as `what`.
    pub fn new(what: &'static str, name: &str, admitted: impl IntoIterator<Item = DType>) -> Self {
        Self {
            what,
            name: name.to_owned(),
            expected: admitted
                .into_iter()
                .map(DType::name)
                .collect::<Vec<_>>()
                .join(", "),
        }
    }
}

impl FromStr for DType {
    type Err = DTypeError;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Self::parse(name).ok_or_else(|| DTypeError::new("element type", name, Self::ALL))
    }
}

impl Serialize for DType {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.name())
    }
}

impl<'de> Deserialize<'de> for DType {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

/// Declares an option's subset of [`DType`]: an enum of the admitted element types, each the
/// [`DType`] it names, read from a name by validating the element type it names and written as
/// that element type's name.
macro_rules! dtype_subset {
    (
        $(#[$meta:meta])*
        $subset:ident, $what:literal { $($(#[$variant_meta:meta])* $variant:ident = $dtype:ident),+ $(,)? }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
        #[serde(try_from = "crate::tensor::DType", into = "crate::tensor::DType")]
        pub enum $subset {
            $($(#[$variant_meta])* $variant),+
        }

        impl $subset {
            /// Every admitted element type.
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];

            /// The element type this value is.
            pub const fn dtype(self) -> crate::tensor::DType {
                match self {
                    $(Self::$variant => crate::tensor::DType::$dtype),+
                }
            }

            /// The element type's name ([`crate::tensor::DType::name`]).
            pub const fn as_str(self) -> &'static str {
                self.dtype().name()
            }
        }

        impl TryFrom<crate::tensor::DType> for $subset {
            type Error = crate::tensor::DTypeError;

            fn try_from(dtype: crate::tensor::DType) -> Result<Self, Self::Error> {
                Self::ALL
                    .iter()
                    .copied()
                    .find(|admitted| admitted.dtype() == dtype)
                    .ok_or_else(|| {
                        crate::tensor::DTypeError::new(
                            $what,
                            dtype.name(),
                            Self::ALL.iter().map(|admitted| admitted.dtype()),
                        )
                    })
            }
        }

        impl From<$subset> for crate::tensor::DType {
            fn from(value: $subset) -> Self {
                value.dtype()
            }
        }

        impl std::str::FromStr for $subset {
            type Err = crate::tensor::DTypeError;

            fn from_str(name: &str) -> Result<Self, Self::Err> {
                let refused = || {
                    crate::tensor::DTypeError::new(
                        $what,
                        name,
                        Self::ALL.iter().map(|admitted| admitted.dtype()),
                    )
                };
                let dtype = crate::tensor::DType::parse(name).ok_or_else(refused)?;
                Self::try_from(dtype).map_err(|_| refused())
            }
        }

        impl std::fmt::Display for $subset {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str(self.as_str())
            }
        }
    };
}

pub(crate) use dtype_subset;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::{ModelDtype, PrefixCacheDtype};

    /// Every element type reads back from its name, and an option's subset reads the names of the
    /// element types it admits, refusing the others by name.
    #[test]
    fn names_read_back_and_subsets_refuse_what_they_do_not_admit() {
        for dtype in DType::ALL {
            assert_eq!(dtype.name().parse::<DType>(), Ok(dtype));
        }
        assert!("bf16".parse::<DType>().is_err());

        assert_eq!("bfloat16".parse::<ModelDtype>(), Ok(ModelDtype::BFloat16));
        let refused = "float8_e4m3fn".parse::<ModelDtype>().unwrap_err();
        assert!(refused.to_string().contains("model dtype"), "{refused}");
        assert_eq!(
            "float8_e4m3fn".parse::<PrefixCacheDtype>(),
            Ok(PrefixCacheDtype::Float8E4m3Fn)
        );

        assert_eq!(
            serde_json::to_value(ModelDtype::Float16).unwrap(),
            serde_json::json!("float16")
        );
        assert!(serde_json::from_value::<ModelDtype>(serde_json::json!("float8_e4m3fn")).is_err());
    }
}
