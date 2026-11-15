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
#[cfg(test)]
pub(crate) mod fixtures;
pub mod mtp;
pub mod positions;
pub mod vision;
pub mod vl;

pub use causal_lm::Qwen3ForCausalLm;
pub use config::{Config, TextConfig, VisionConfig};
pub use decoder::DecoderLayer;
pub use mtp::Mtp;
pub use vision::VisionTower;
pub use vl::Qwen3VlModel;

/// The id this family's refusals name it by ([`nn::Error::Model`]): its registration's.
pub(crate) const ID: nn::ModelId = nn::ModelId(processing::qwen::QWEN3.id);
