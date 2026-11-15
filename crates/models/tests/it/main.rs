//! The model families' integration tests: each family loaded through its own loading path on the
//! CPU device against the outputs of its reference implementation, and the profile runs on the
//! meta device over the released checkpoints' headers, which are ignored by default.

#![cfg(test)]

mod minimax_h3;
mod qwen3;
mod qwen4_exp;
mod releases;
