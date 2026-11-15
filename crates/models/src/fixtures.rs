//! The tiny checkpoints this crate's tests hold, for the tests of the crates that serve them.
//!
//! Each is a family's checkpoint at its smallest, a configuration and one FP32 safetensors shard
//! of seeded random weights, which `tests/it/qwen3/generate_reference.py` writes beside the
//! reference outputs this crate's CPU tests compare against. A test of another crate (the
//! engine's start and its ranks' loads) names a checkpoint through this module rather than
//! walking to this crate's directory.
//!
//! The paths name this crate's source tree, so the module is the development feature
//! `fixtures`: a serving build holds none of them.

use std::path::{Path, PathBuf};

/// The directory of the Qwen3 fixtures, each a directory of its own holding `model/` and the
/// reference outputs.
fn qwen3() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/it/qwen3/data"))
}

/// The Qwen3-MoE checkpoint (`Qwen3MoeForCausalLM`), saved with one module per expert as the
/// released Qwen3-30B-A3B stores them.
pub fn qwen3_moe() -> PathBuf {
    qwen3().join("qwen3_moe/model")
}

/// The Qwen3.5-MoE checkpoint (`Qwen3_5MoeForConditionalGeneration`): gated delta networks,
/// gated attention, a shared expert, the vision tower and the multi-token-prediction drafter.
pub fn qwen3_5_moe() -> PathBuf {
    qwen3().join("qwen3_5_moe/model")
}
