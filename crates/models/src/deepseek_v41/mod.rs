//! DeepSeek-V4.1 (`DeepseekV41ForCausalLM`, `model_type` `deepseek_v41`): what its decoder's
//! residual streams and its n-gram memories read from a checkpoint.
//!
//! The decoder carries its residual as `hc_mult` hyper-connected streams, every attention and
//! feed-forward sub-layer between two manifold-constrained connections
//! ([`nn::modules::MhcResidual`], [`nn::modules::MhcConnection`]); layer `l`'s connections are
//! `layers.<l>.hc_attn_{fn,scale,base}` with `layers.<l>.attn_norm`, and
//! `layers.<l>.hc_ffn_{fn,scale,base}` with `layers.<l>.ffn_norm`, and the stack collapses into
//! `norm` (`inference/model.py:909-1019`). The layers `engram_layer_ids` add a hashed n-gram
//! memory into their streams before their attention ([`nn::modules::NgramMemory`]), whose
//! hashing this module derives as the reference does ([`engram`]).
//!
//! - [`config`]: [`Config`], the fields of the checkpoint's `text_config` these read;
//! - [`engram`]: [`Hashing`], the normalized vocabulary, the multipliers and the prime ranges of
//!   every n-gram layer.
//!
//! ## References
//!
//! A citation `path:lines` names a file of the checkpoint's own reference implementation,
//! `inference/` of <https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017> at commit
//! `1bae0e1088993b9b82de08f59de5be5a34e42170`.

pub mod config;
pub mod engram;

pub use config::{Config, EngramConfig};
pub use engram::Hashing;

/// The family's refusals are raised under this id.
pub(crate) const ID: nn::ModelId = nn::ModelId("deepseek_v41");

/// The class a checkpoint of the family declares in its `config.json`.
pub const CLASS: &str = "DeepseekV41ForCausalLM";
