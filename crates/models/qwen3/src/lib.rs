//! The Qwen3 family: its checkpoint configurations read once at the loading boundary, the decoder
//! layer and stack every Qwen3 model composes, and the models built from them.
//!
//! - [`config`]: [`TextConfig`], the decoder stack's fields, and [`MoeConfig`], the routed experts
//!   of `Qwen3MoeForCausalLM`;
//! - [`decoder`]: [`DecoderLayer`] and [`DecoderStack`], one implementation whatever the
//!   consumer's attention, rounding and rotary precision;
//! - [`moe`]: `Qwen3MoeForCausalLM`, implementing [`nn::CausalLm`];
//! - [`vl`]: the Qwen3-VL language model, vision tower, patch mergers and their host position
//!   tables, which [`VlConfig`] describes;
//! - [`family`] and [`processing`]: the family as execution loads it and as request processing
//!   serves it.
//!
//! Nothing here owns a runtime object: a model receives borrowed [`nn::Forward`] resources and
//! tensor views, and every device operation is a call to a layer.

pub mod config;
pub mod decoder;
pub mod family;
pub mod moe;
pub mod processing;
pub mod vl;

pub use config::{MoeConfig, TextConfig, VisionConfig, VlConfig};
pub use decoder::{DecoderLayer, DecoderStack};

/// What this crate can fail with.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The checkpoint metadata is valid JSON but not a supported Qwen3 configuration, or a
    /// deployment's split does not divide what it shards.
    #[error("unsupported qwen3 checkpoint: {0}")]
    Config(String),
    /// A layer, an operator or the weight loader refused the call.
    #[error(transparent)]
    Nn(#[from] nn::Error),
}

impl From<ops::Error> for Error {
    /// An operator-level refusal of a view (a narrow) as the shared library reports it.
    fn from(error: ops::Error) -> Self {
        Self::Nn(error.into())
    }
}

/// The crate's result type.
pub type Result<T> = std::result::Result<T, Error>;

/// The family's name in a refusal it reports through the shared library.
const MODEL: &str = "qwen3";

impl From<Error> for nn::Error {
    /// A failure as the capability and family contracts report it: a shared-library failure is
    /// already one and passes through unchanged, and every refusal of this crate's own is
    /// [`nn::Error::Model`] naming the family.
    fn from(error: Error) -> Self {
        match error {
            Error::Nn(error) => error,
            other => nn::Error::model(MODEL, other),
        }
    }
}
