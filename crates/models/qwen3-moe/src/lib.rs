//! `Qwen3MoeForCausalLM`: the checkpoint configuration read once at the loading boundary, the
//! mathematical partition of a deployment, and the causal-LM model itself, composed of `nn`
//! layers and implementing [`nn::CausalLm`].
//!
//! The model follows specs/worker.md 9's `Qwen3MoeForCausalLM` rows and the layer equations of
//! `uniserve/models/qwen3/transformer.py`:
//!
//! - the embedding is vocabulary-parallel over the tensor ranks, zero-filled outside a rank's shard
//!   and summed with an all-reduce;
//! - each layer is pre-norm: `hidden, residual = add_rms_norm(hidden, residual)` around attention and
//!   around the expert block, with the residual stream carried separately and summed in FP32 once;
//! - attention applies per-head RMSNorm to Q and K, a single-axis rotary embedding, paged KV
//!   attention over the prefix cache, and a row-parallel output projection;
//! - the expert block routes each token to its top-k experts of a replicated router, renormalizes
//!   the FP32 weights when the checkpoint says so, and combines the experts' outputs in FP32;
//! - the final RMSNorm feeds the vocabulary-parallel head.
//!
//! Nothing here owns a runtime object: the model receives borrowed [`nn::Forward`] resources and
//! tensor views, and every device operation is a call to a layer.

pub mod config;
pub mod load;
pub mod model;
pub mod partition;
pub mod weights;

pub use config::{Activation, Config, RopeScaling, YaRn};
pub use load::load;
pub use model::{LoadOptions, Plan, Qwen3Moe};
pub use partition::Partition;

/// What this crate can fail with.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The checkpoint's metadata directory or one of its files could not be read.
    #[error("reading {path}: {source}")]
    Io {
        /// The path that failed.
        path: String,
        /// The underlying filesystem error.
        source: std::io::Error,
    },
    /// The checkpoint's `config.json` is not valid JSON.
    #[error("parsing {path}: {source}")]
    Json {
        /// The path that failed.
        path: String,
        /// The underlying JSON error.
        source: serde_json::Error,
    },
    /// The checkpoint metadata is valid JSON but not a supported Qwen3-MoE configuration.
    #[error("unsupported qwen3-moe checkpoint: {0}")]
    Config(String),
    /// A checkpoint feature or deployment shape this crate does not implement yet.
    #[error("unsupported: {0}")]
    Unsupported(String),
    /// A layer, an operator or the weight loader refused the call.
    #[error(transparent)]
    Nn(#[from] nn::Error),
}

/// The crate's result type.
pub type Result<T> = std::result::Result<T, Error>;
