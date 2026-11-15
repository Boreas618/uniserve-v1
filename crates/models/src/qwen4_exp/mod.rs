//! Qwen3.8 Flash Next (`Qwen4ExpForConditionalGeneration`, `model_type` `qwen4_exp`): its
//! checkpoint configuration, its modules and the model built from them.
//!
//! The arithmetic and the structure follow vLLM's implementation of the family
//! (`refs/vllm/vllm/models/qwen4_exp/`); where SGLang differs, vLLM's choice is taken.
//!
//! - [`config`]: [`Config`], read once from a checkpoint's sidecars;
//! - [`hyper_connection`]: [`HyperConnection`], the gated read and write of several residual
//!   streams around a block (`nvidia/hyperconnection.py`);
//! - [`ngram`]: [`NgramEmbedding`], the hashed bigram and trigram table, and [`PleLayer`],
//!   which gates its rows into the streams through a dilated causal convolution
//!   (`nvidia/ngram_embedding.py`, `nvidia/ple_layer.py`);
//! - [`attention`]: [`IndexedAttention`], gated attention over the tokens an indexer selects
//!   from compressed keys (`nvidia/qsa.py`, `nvidia/qsa_indexer.py`);
//! - [`decoder`]: [`DecoderLayer`], a gated delta network or indexed attention, then routed
//!   experts with a gated shared expert, each between two hyper-connections
//!   (`nvidia/model.py:295-368`);
//! - [`causal_lm`]: [`Qwen4ExpForCausalLm`], implementing [`nn::CausalLm`];
//! - [`mtp`]: [`Mtp`], the multi-token-prediction drafter over streams ([`nn::Drafter`]);
//! - [`family`]: the family value execution loads and request processing serves.
//!
//! The vision tower and the M-RoPE coordinates of a prompt are Qwen3.6's, one implementation
//! in the `qwen3` module ([`crate::qwen3::VisionTower`]).
//!
//! A model computes on tensors and calls `nn`: a module here is a struct of `nn` modules and
//! weight tensors with a `load` over a [`nn::VarBuilder`] and a `forward` over tensors. The
//! values a forward needs beyond its inputs (each token's row, position and state slot) are
//! tensor expressions over the batch's columns, so their shapes follow from the batch's
//! extents alone.

pub mod attention;
pub mod causal_lm;
pub mod config;
pub mod decoder;
pub mod family;
pub mod hyper_connection;
pub mod mtp;
pub mod ngram;

pub use attention::IndexedAttention;
pub use causal_lm::Qwen4ExpForCausalLm;
pub use config::{Config, HyperConnectionConfig, IndexerConfig, NgramEmbeddingConfig, TextConfig};
pub use decoder::DecoderLayer;
pub use hyper_connection::HyperConnection;
pub use mtp::Mtp;
pub use ngram::{NgramEmbedding, PleLayer};

/// What this family can fail with.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The checkpoint's configuration is not a configuration this family builds, a deployment's
    /// split does not divide what it shards, or a call's inputs do not describe the model.
    #[error("qwen4_exp: {0}")]
    Config(String),
    /// A module, a function or the weight loader refused the call.
    #[error(transparent)]
    Nn(#[from] nn::Error),
}

impl From<tensor::Error> for Error {
    fn from(error: tensor::Error) -> Self {
        Self::Nn(error.into())
    }
}

impl From<crate::qwen3::Error> for Error {
    /// A refusal of the shared readers and modules is this family's.
    fn from(error: crate::qwen3::Error) -> Self {
        match error {
            crate::qwen3::Error::Nn(error) => Self::Nn(error),
            crate::qwen3::Error::Config(problem) => Self::Config(problem),
        }
    }
}

/// The family's result type.
pub type Result<T> = std::result::Result<T, Error>;

impl From<Error> for nn::Error {
    /// A failure of the shared library passes through unchanged; a refusal of this family's own
    /// is [`nn::Error::Model`] naming the family.
    fn from(error: Error) -> Self {
        match error {
            Error::Nn(error) => error,
            other => nn::Error::model("qwen4_exp", other),
        }
    }
}
