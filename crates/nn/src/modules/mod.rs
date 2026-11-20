//! The modules a model composes: structs that hold weights and compute with them.
//!
//! Each module follows the crate's conventions (crate documentation): a `Config` of the widths
//! and names its checkpoint module states, `load(config, vb)` reading its weights under the
//! builder's path, and `forward` over tensors. A module holds weights and sub-modules and
//! nothing of execution: no context, stream or buffer.
//!
//! A module of one input also implements the tensor library's [`Module`](crate::Module), so it
//! composes with `Tensor::apply`.

mod adaln;
mod attention;
mod codec;
mod conv;
mod embedding;
mod frame;
mod gated_delta_net;
mod head;
mod hyper;
mod latent;
mod linear;
mod mlp;
mod moe;
mod ngram;
mod norm;
mod rotary;

pub use attention::{Attention, AttentionConfig, QkNorm};
pub use embedding::{Embedding, EmbeddingConfig};
pub use gated_delta_net::{GatedDeltaNet, GatedDeltaNetConfig};
pub use latent::{
    Candidates, Compressed, Compressor, CompressorConfig, Indexer, IndexerConfig, LatentAttention,
    LatentAttentionConfig, LatentBatch, Selection,
};
mod timestep;
pub mod ulysses;
mod vocoder;
mod vsa;
pub use adaln::{AdaLnBlock, AdaLnBlockConfig, AdaLnOut, AdaLnOutConfig};
pub use codec::{DacEncoder, DacEncoderConfig};
pub use conv::{Conv1d, Conv3d, ConvConfig, ConvTranspose1d};
pub use frame::{FrameNorm, FrameNormConfig, ResnetBlock3d, ResnetBlock3dConfig};
pub use head::{LmHead, LmHeadConfig};
pub use hyper::{MhcConfig, MhcConnection, MhcResidual};
pub use linear::{
    GroupedLinear, Linear, LinearConfig, Parallel, QkvBias, QkvLinear, QkvLinearConfig, QkvNames,
};
pub use mlp::{GateUp, Mlp, MlpConfig};
pub use moe::{ExpertLayout, MoE, MoEConfig, SharedExpert, SharedExpertConfig};
pub use ngram::{
    NgramConfig, NgramHashing, NgramMemory, NgramTable, TableShard, Tokens, ngram_context,
};
pub use norm::{
    GatedRmsNorm, GatedRmsNormConfig, LayerNorm, LayerNormConfig, RmsNorm, RmsNormConfig,
};
pub use rotary::{Precision, RopeParameters, Rotary, RotaryConfig, Spectrum, Yarn};
pub use timestep::{
    TimestepEmbedding, TimestepEmbeddingConfig, TimestepTable, TimestepTableConfig,
};
pub use ulysses::{UlyssesAttention, UlyssesAttentionConfig};
pub use vocoder::{
    Activation1d, Activation1dConfig, AmpBlock, AmpBlockConfig, BigVgan, BigVganConfig,
};
pub use vsa::{VsaAttention, VsaAttentionConfig};
