//! The runners: one implementation of [`Runner`](crate::Runner) per capability a model exposes.
//!
//! A runner holds its model through a capability trait of `nn` and dispatches on nothing else.
//! What the implementations share (the profile run, capture, replay, the input buffer and the
//! generations) is [`Graphs`](crate::Graphs)'s.

pub mod causal_lm;

pub use causal_lm::CausalLmRunner;
