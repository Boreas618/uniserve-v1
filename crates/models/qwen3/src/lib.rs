//! The Qwen3 family: its checkpoint configuration, the decoder layer every Qwen3 model composes,
//! and the models built from it.
//!
//! - [`config`]: [`Config`], read once from a checkpoint's sidecars;
//! - [`decoder`]: [`DecoderLayer`], Hugging Face's `Qwen3DecoderLayer` with the feed-forward
//!   block of `Qwen3MoeDecoderLayer` and the token mixers of `Qwen3_5MoeDecoderLayer`;
//! - [`causal_lm`]: [`Qwen3ForCausalLm`], `Qwen3MoeForCausalLM` and the language model of
//!   `Qwen3_5MoeForConditionalGeneration`, implementing [`nn::CausalLm`];
//! - [`mtp`]: [`Mtp`], the Qwen3.5 multi-token-prediction drafter ([`nn::Drafter`]);
//! - [`vision`]: [`VisionTower`], the Qwen3-VL vision tower ([`nn::Vision`]);
//! - [`vl`]: [`Qwen3VlModel`], `Qwen3VLModel` as an encoder over a prefix of its layers
//!   ([`nn::Encoder`]);
//! - [`positions`]: the M-RoPE coordinates of a prompt and the tower's position tables, as CPU
//!   tensor expressions;
//! - [`family`]: the family values execution loads and request processing serves.
//!
//! The Qwen-VL processor rules that size media to the tower's patch grid (`smart_resize`, frame
//! sampling) are request processing's ([`processing::qwen_vl`]).
//!
//! A model computes on tensors and calls `nn`: a module here is a struct of `nn` modules and
//! weight tensors with a `load` over a [`nn::VarBuilder`] and a `forward` over tensors. The
//! values a forward needs beyond its inputs (positions, row indices, sequence offsets) are CPU
//! tensors moved to the device of the tensor they meet, their shapes following from the batch's
//! extents alone.

pub mod causal_lm;
pub mod config;
pub mod decoder;
pub mod family;
pub mod mtp;
pub mod positions;
pub mod vision;
pub mod vl;

pub use causal_lm::Qwen3ForCausalLm;
pub use config::{Config, MoeConfig, TextConfig, VisionConfig};
pub use decoder::DecoderLayer;
pub use mtp::Mtp;
pub use vision::VisionTower;
pub use vl::Qwen3VlModel;

/// What this crate can fail with.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The checkpoint's configuration is not a Qwen3 configuration this crate builds, a
    /// deployment's split does not divide what it shards, or a call's inputs do not describe
    /// the model.
    #[error("qwen3: {0}")]
    Config(String),
    /// A module, a function or the weight loader refused the call.
    #[error(transparent)]
    Nn(#[from] nn::Error),
}

impl From<uniserve_tensor::Error> for Error {
    fn from(error: uniserve_tensor::Error) -> Self {
        Self::Nn(error.into())
    }
}

/// The crate's result type.
pub type Result<T> = std::result::Result<T, Error>;

impl From<Error> for nn::Error {
    /// A failure of the shared library passes through unchanged; a refusal of this crate's own
    /// is [`nn::Error::Model`] naming the family.
    fn from(error: Error) -> Self {
        match error {
            Error::Nn(error) => error,
            other => nn::Error::model("qwen3", other),
        }
    }
}
