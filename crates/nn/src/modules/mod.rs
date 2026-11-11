//! The modules a model composes: structs that hold weights and compute with them.
//!
//! Each module follows the crate's conventions (crate documentation): a `Config` of the widths
//! and names its checkpoint module states, `load(config, vb)` reading its weights under the
//! builder's path, and `forward` over tensors. A module holds weights and sub-modules and
//! nothing of execution: no context, stream or buffer.
//!
//! A module of one input also implements the tensor library's [`Module`](crate::Module), so it
//! composes with `Tensor::apply`.

mod embedding;
mod head;
mod linear;
mod mlp;
mod norm;

pub use embedding::{Embedding, EmbeddingConfig};
pub use head::{LmHead, LmHeadConfig};
pub use linear::{Linear, LinearConfig, Parallel, QkvBias, QkvLinear, QkvLinearConfig, QkvNames};
pub use mlp::{GateUp, Mlp, MlpConfig};
pub use norm::{
    GatedRmsNorm, GatedRmsNormConfig, LayerNorm, LayerNormConfig, RmsNorm, RmsNormConfig,
};
