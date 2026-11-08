//! The layers a model composes.
//!
//! Each layer holds borrowed weight views and the geometry it was built for, and its forward calls
//! operators (specs/worker.md 4.3): no layer allocates, synchronizes, captures, looks up a program
//! or names a device. Activations a layer needs beyond its arguments come from
//! [`Forward::arena`](crate::Forward::arena), which the worker sizes per captured bucket.
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
mod audio;
mod bias;
mod conv;
mod copy_block;
mod cross_fade;
mod embedding;
mod exchange;
mod frame;
mod head;
mod linear;
mod modulation;
mod moe;
mod noise;
mod norm;
mod patch;
mod pcm;
mod pool;
mod position;
mod posterior;
mod residual;
mod rgb;
mod rope;
mod rotate;
mod rows;
mod sampler;
mod segment;
mod segments;
mod tile_stitch;
mod transpose;
mod unpatchify;

pub use activation::{GatedActivation, Gelu, PackedSwiGlu, Silu, Snake, SwiGlu};
pub use attention::{Attention, AttentionBackend, TRTLLM_WORKSPACE_BYTES};
pub use audio::AudioActivation;
pub use bias::BiasAdd;
pub use conv::{Conv1d, Conv3d, ConvGeometry, ConvTranspose1d};
pub use copy_block::CopyBlock;
pub use cross_fade::CrossFade;
pub use embedding::Embedding;
pub use exchange::UlyssesExchange;
pub use frame::{FrameMoments, FrameNormPad, FramePad};
pub use head::{ParallelLMHead, gather_rows, gather_rows_descriptor};
pub use linear::{Linear, Nvfp4Linear, PACKED_M_CLASSES, packed_rows};
pub use modulation::{GatedResidual, ModulatedRmsNorm};
pub use moe::{
    Buffer, ENCODE_CHUNK, EncodedStep, FusedTables, Group, Launch, Moe, MoeGeometry, TILE_ROWS,
    dense_bytes, encoded_bytes, exchanged_bytes, exchanged_rows, router_bytes, routing_bytes,
    sort_bytes, step_bytes,
};
pub use noise::{Key as NoiseKey, Noise};
pub use norm::{LayerNorm, RmsNorm};
pub use ops::activation::Activation;
pub use ops::gemm::WeightOrder;
pub use ops::rope::Rotation;
pub use patch::Patch;
pub use pcm::PcmInterleave;
pub use pool::AdaptivePool;
pub use position::PositionInterpolate;
pub use posterior::Posterior;
pub use residual::{
    ScaledAdd, ScaledResidual, ScaledResidualLayerNorm, ScaledResidualRmsNorm, WeightedRmsNorm,
};
pub use rgb::RgbBytes;
pub use rope::{Rotary, Sections};
pub use rotate::Rotate;
pub use rows::Rows;
pub use sampler::{Candidates, Sampler, SamplerState, Samples};
pub use segment::{SegmentAttention, SegmentTables};
pub use segments::{SELECT_FLOOR, VsaTiles};
pub use tile_stitch::{TileGrid, TileStitch};
pub use transpose::Transpose;
pub use unpatchify::UnpatchifyVideo;

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

/// The tiling constants a stored program's descriptor key carries.
///
/// A key names the block a program was compiled for, so the number has one owner: the layer that
/// launches it. Re-exported here for a fixture generator that sits above this crate - it must read
/// what the layers launch rather than restate it, or `generator == fixture` proves only that two
/// statements agree with each other.
pub mod tiles {
    pub use super::activation::BLOCK as ELEMENTWISE_BLOCK;
    pub use super::bias::BLOCK as BIAS_BLOCK;
    pub use super::head::BLOCK as LOGITS_BLOCK;
    pub use super::launch::{EMBEDDING_TOKENS, PDL, QK_NORM_ROWS, ROTARY_BLOCK, row_tile};
    pub use super::pool::BLOCK as POOL_BLOCK;
    pub use super::position::BLOCK as POSITION_BLOCK;
    pub use super::rotate::{BLOCK as ROTATE_BLOCK, ROWS as ROTATE_ROWS};
    pub use super::rows::BLOCK as ROWS_BLOCK;
    pub use super::unpatchify::BLOCK as UNPATCHIFY_BLOCK;
}

/// The accumulation element type of every floating-point operator here.
pub(crate) const ACCUMULATE: ops::DType = ops::DType::F32;
