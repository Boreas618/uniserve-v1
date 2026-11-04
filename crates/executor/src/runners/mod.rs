//! The runners the device executor holds: one per block a deployment places on the device.
//!
//! `specs/worker.md` §4.5 names one runner per capability the model exposes: `EncoderRunner`,
//! `DenoiserRunner`, `DecoderRunner` and `CausalLmRunner`, plus the host-side `MediaRunner`. They
//! share the graph tables, the pools and the staging rules and differ in the capability they call
//! and in what a bucket of theirs means.
//!
//! This slice serves `CausalLmRunner` (Qwen3-MoE BF16, §11 step 2). The other three are added with
//! the model that needs them (H3, §11 step 3); the trait, the graph table and the arena they build
//! on are the same.

pub mod causal_lm;

pub use causal_lm::CausalLmRunner;
pub mod resident;
