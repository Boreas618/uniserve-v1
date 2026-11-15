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
#[cfg(test)]
pub(crate) mod fixtures;
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

/// The id this family's refusals name it by ([`nn::Error::Model`]): its registration's.
pub(crate) const ID: nn::ModelId = nn::ModelId(processing::qwen::QWEN4_EXP.id);
