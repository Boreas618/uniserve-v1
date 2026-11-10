//! The blocks a transformer is built from, composed of [`crate::layers`].
//!
//! A block owns the layers of one established sub-module - a gated MLP, an attention with its
//! projections, a diffusion block with its modulated norms - loads them from its scope, and runs
//! them in the order its reference does. Each block keeps the arithmetic the shipped baselines use
//! as named fields of its spec (the gated product's rounding, the q/k normalization's form), so two
//! consumers whose references round differently share one block without sharing a rounding.
//!
//! A block's intermediates are dead once its last launch is enqueued, so a block reserves them from
//! the forward's arena under a mark it releases before it returns: the arena's peak is one block's
//! intermediates, and a captured graph replays over the same addresses.

mod adaln;
mod attention;
mod delta_net;
mod mlp;
mod prenorm;

pub use adaln::{AdaLnBlock, AdaLnOut, AdaLnSpec, ModulationTable, Stage};
pub use attention::{
    AttentionKind, Keys, QkNormNames, QkvAttention, QkvAttentionSpec, RopeFactors,
};
pub use delta_net::{GatedDeltaNet, GatedDeltaNetSpec};
pub use mlp::{GateUp, GatedMlp, GatedMlpSpec, SharedExpert, SharedExpertSpec};
pub use prenorm::{Pending, PreNormBlock, PreNormSpec, ResidualForm};
