//! The layers a model composes.
//!
//! Each layer holds borrowed weight views and the geometry it was built for, and its forward calls
//! operators: no layer allocates, synchronizes, captures, looks up a program
//! or names a device. Activations a layer needs beyond its arguments come from
//! [`Forward::arena`](crate::Forward::arena), which the worker sizes per captured bucket.
//!
//! ## Tuning constants
//!
//! A few operator descriptors carry a constant that belongs to the kernel program rather than to
//! the model's geometry: the row tile a normalization loads, the tokens one embedding program
//! handles, the launch variant. The design gives those constants to the program package (tile
//! constants are package content, and a program's arguments come from the epoch's program record),
//! while `load_model` here has no program record to read. Following a uniform rule per operator
//! keeps the descriptor canonical and every launch path the same until the program arguments are
//! threaded through: a tile is the extent it covers rounded up to a power of two, and the programs
//! compiled here are the no-PDL variants. The constants are declared once, in [`launch`], so the
//! step that reads them from a program changes one module.

mod activation;
mod affine;
mod attention;
mod conv;
mod embed;
mod embedding;
mod head;
mod indexed;
mod layout;
mod linear;
mod moe;
mod noise;
mod norm;
mod pool;
mod posterior;
mod residual;
mod rope;
mod rows;
mod transpose;
mod ulysses;
mod vsa;

pub use activation::{GateBias, GatedActivation, Gelu, Halves, Rounding, Silu, Snake};
pub use affine::ChannelAffine;
pub use attention::{DenseAttention, DenseSpec, PagedAttention, PagedSpec, Varlen};
pub use conv::{
    CausalPad, Conv1d, Conv3d, ConvSpec, ConvTranspose1d, Padding, ResnetBlock3d, ResnetSpec,
};
pub use embed::{PatchEmbed, PatchEmbedSpec};
pub use embedding::{VocabParallelEmbedding, vocabulary_shard};
pub use head::ParallelLMHead;
pub use indexed::StepIndexed;
pub use layout::{Access, Copy, Patchify};
pub use linear::{
    Bias, ENCODE_CHUNK, Linear, MergedColumnLinear, Parallel, QkvBias, QkvLinear, QkvNames,
    QkvSpec, encode_block,
};
pub use moe::{DispatchViews, ExpertNames, ExpertSlots, Moe, MoeSpec, WEIGHT_SCALE_LAYOUT};
pub use noise::{Key as NoiseKey, Noise};
pub use norm::{Epilogue, GroupNorm, LayerNorm, Modulation, ResidualStream, RmsNorm};
pub use ops::activation::Activation;
pub use ops::gemm::WeightOrder;
pub use pool::AdaptiveAvgPool2d;
pub use posterior::Posterior;
pub use residual::{Axpby, Cast, Residual, Scale};
pub use rope::{
    Gains, Normalization, Precision, QkNorm, Rotary, RotarySpec, Sections, Spectrum, Yarn,
    grid_positions,
};
pub use rows::IndexRows;
pub use transpose::Transpose;
pub use ulysses::{Interval, UlyssesAttention, UlyssesSpec};
pub use vsa::{VsaAttention, VsaScratch, VsaSpec, VsaTables};

/// The constants a layer's operator descriptor carries and the program package owns.
///
/// See the module documentation: one place, one rule, until the program arguments reach loading.
pub mod launch {
    /// Whether the descriptors this crate builds are the programmatic-dependent-launch variants.
    pub const PDL: bool = false;

    /// Tokens one `embedding` program handles.
    pub const EMBEDDING_TOKENS: u32 = 8;

    /// Elements one `rotary_factors` program handles.
    pub const ROTARY_BLOCK: u32 = 256;

    /// Head rows one `qk_norm_rope` program handles: a token holds one per head, so a step of few
    /// tokens still spreads over the device.
    pub const QK_NORM_ROWS: u32 = 16;

    /// The row width a row-wise kernel loads at a time: the width rounded up to a power of two.
    pub fn row_tile(columns: u32) -> u32 {
        columns.max(1).next_power_of_two()
    }

    /// The element count of one cache row a `kv_write` program covers.
    pub(crate) fn kv_tile(head_dim: u32) -> u32 {
        head_dim.clamp(1, 128).next_power_of_two().min(128)
    }
}

/// The accumulation element type of every floating-point operator here.
pub(crate) const ACCUMULATE: ops::DType = ops::DType::F32;
