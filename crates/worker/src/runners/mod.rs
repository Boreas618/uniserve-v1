//! The runners: one implementation of [`Runner`](crate::Runner) per capability a model exposes.
//!
//! A runner holds its model through a capability trait of `nn` and dispatches on nothing else.
//! What the implementations share (the profile run, capture, replay, the input buffer and the
//! generations) is [`Graphs`](crate::Graphs)'s.
//!
//! | Runner | Capability |
//! | --- | --- |
//! | [`CausalLmRunner`] | `nn::family::TextModel`: every causal language model |
//! | [`DecoderRunner`] | `nn::Decoder`, with `nn::Splicer` for a video: the clips of a timeline and an audio track, from the latents in the rank's media products |

pub mod causal_lm;
pub mod decoder;

pub use causal_lm::CausalLmRunner;
pub use decoder::{DecoderConfig, DecoderRunner};
