//! The layers a model composes.
//!
//! Each layer holds borrowed weight views and the geometry it was built for, and its forward calls
//! operators (specs/worker.md 4.3): no layer allocates, synchronizes, captures, looks up a program
//! or names a device. Activations a layer needs beyond its arguments come from
//! [`Forward::arena`](crate::Forward::arena), which the executor sizes per captured bucket.
//!
//! ## Tuning constants
//!
//! A few operator descriptors carry a constant that belongs to the kernel program rather than to
//! the model's geometry: the row tile a normalization loads, the tokens one embedding program
//! handles, the launch variant. The design gives those constants to the program package (specs/
//! worker.md 5.2, 5.4: "tile 常数" are package content, and a program's arguments come from the
//! epoch's program record), while `load_model` here has no program record to read (specs/worker.md
//! 4.3). Following a uniform rule per operator keeps the descriptor canonical and every launch
//! path the same until the program arguments are threaded through: a tile is the extent it covers
//! rounded up to a power of two, and the programs compiled here are the no-PDL variants. The
//! constants are declared once, in [`launch`], so the step that reads them from a program changes
//! one module.

mod activation;
mod attention;
mod conv;
mod embedding;
mod exchange;
mod frame;
mod head;
mod linear;
mod modulation;
mod moe;
mod noise;
mod norm;
mod residual;
mod rope;
mod sampler;

pub use activation::{GatedActivation, PackedSwiGlu, SwiGlu};
pub use attention::{Attention, AttentionBackend, TRTLLM_WORKSPACE_BYTES};
pub use conv::{Conv1d, Conv3d, ConvGeometry, ConvTranspose1d};
pub use embedding::Embedding;
pub use exchange::UlyssesExchange;
pub use frame::{FrameMoments, FrameNormPad, FramePad};
pub use head::{ParallelLMHead, gather_rows};
pub use linear::Linear;
pub use modulation::ModulatedRmsNorm;
pub use moe::Moe;
pub use noise::Noise;
pub use norm::{LayerNorm, RmsNorm};
pub use ops::rope::Rotation;
pub use residual::{
    ScaledResidual, ScaledResidualLayerNorm, ScaledResidualRmsNorm, WeightedRmsNorm,
};
pub use rope::Rotary;
pub use sampler::{Candidates, Sampler, SamplerState, Samples};

/// The constants a layer's operator descriptor carries and the program package owns.
///
/// See the module documentation: one place, one rule, until the program arguments reach loading.
pub(crate) mod launch {
    /// Whether the descriptors this crate builds are the programmatic-dependent-launch variants.
    pub(crate) const PDL: bool = false;

    /// Tokens one `embedding` program handles.
    pub(crate) const EMBEDDING_TOKENS: u32 = 8;

    /// Elements one `rotary_factors` program handles.
    pub(crate) const ROTARY_BLOCK: u32 = 256;

    /// Tokens one `qk_norm_rope` program handles.
    pub(crate) const QK_NORM_TOKENS: u32 = 1;

    /// The row width a row-wise kernel loads at a time: the width rounded up to a power of two.
    pub(crate) fn row_tile(columns: u32) -> u32 {
        columns.max(1).next_power_of_two()
    }

    /// The element count of one cache row a `kv_write` program covers.
    pub(crate) fn kv_tile(head_dim: u32) -> u32 {
        head_dim.clamp(1, 128).next_power_of_two().min(128)
    }
}

/// The accumulation element type of every floating-point operator here.
pub(crate) const ACCUMULATE: ops::DType = ops::DType::F32;
