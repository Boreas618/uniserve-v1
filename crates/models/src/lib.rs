//! The model families, one module each, as vLLM's or candle's model files hold them: the
//! configuration read from a checkpoint, a module tree whose `forward`s compose tensor operations
//! and [`nn`] modules, the checkpoint's names, and the family values execution loads and request
//! processing serves.
//!
//! - [`qwen3`]: the Qwen3 decoder, Qwen3-MoE and the Qwen3.5-MoE language model with its drafter,
//!   and the Qwen3-VL encoder and vision tower;
//! - [`qwen4_exp`]: Qwen3.8 Flash Next (`Qwen4ExpForConditionalGeneration`), which shares the
//!   vision tower and the configuration readers of [`qwen3`];
//! - [`minimax_h3`]: MiniMax-H3 and FastH3, whose text encoder is [`qwen3`]'s Qwen3-VL model.
//!
//! A family owns no device, stream, buffer, scheduler or request state (`AGENTS.md`, Model
//! Library Boundary): the crate depends on `nn`, the tensor library and `processing` only.

pub mod minimax_h3;
pub mod qwen3;
pub mod qwen4_exp;
