//! The request semantics of UniServe's model families.
//!
//! Each module states the rules one family applies to its requests, as pure
//! functions over plain values: checkpoint facts, input rules and planning.
//! The daemon plans and presents requests with them before submission, and
//! the worker derives what it admits from the same rules, so each rule is
//! written once. Nothing here reads files, opens connections or knows the
//! serving loop: what a rule needs of the model's tokenizer comes through
//! [`Vocabulary`], and its refusals are [`FactsError`]s.
//!
//! - [`ModelDescription`]: the served families and how a checkpoint names
//!   its family;
//! - `omni`: the multimodal generation families (SenseNova, Bagel): prompt
//!   framing, image slots, encoder sizing and output filters;
//! - `diffusion_gemma`: DiffusionGemma's block-diffusion canvases, control
//!   tokens and image patch budget;
//! - `video`: MiniMax-H3 video requests.

mod description;
pub mod diffusion_gemma;
mod facts;
pub mod omni;
pub mod video;
mod vocabulary;

pub use description::ModelDescription;
pub use facts::{FactsError, Result};
pub use vocabulary::{Vocabulary, VocabularyError};
