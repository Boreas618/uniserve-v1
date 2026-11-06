//! The shared computation library: tensor views, numerical layers, capability traits, a loaded
//! model's block structure, and the checkpoint and weight-loading boundary.
//!
//! ## Where this crate sits
//!
//! The call chain of one forward is `runner → model → layer → operator → kernel` (specs/worker.md
//! 4). A model in `crates/models/<model>/` composes the layers here; a layer calls the operators of
//! [`ops`] and nothing below them; the operator's launch code reaches the loaded kernel. `nn` owns
//! no runtime object: it never allocates device memory, creates a stream, captures a graph or
//! selects a program. Everything a forward needs arrives borrowed in [`Forward`].
//!
//! ## The tensors
//!
//! [`TensorView`], [`Shape`], [`Strides`], [`DType`] and [`DevicePtr`] are [`ops`] types: they are
//! the ABI of an operator's arguments, and `nn` re-exports them rather than defining a second
//! representation. A [`Weight`] pairs a view with the [`Backing`] the executor's retained storage
//! gives it, so a layer's weights are plainly borrowed and plainly retained.
//!
//! ## The layers
//!
//! [`layers`] holds what a model composes: [`Linear`],
//! [`RmsNorm`], [`LayerNorm`], [`Rotary`],
//! [`Embedding`], [`Attention`], [`Moe`],
//! [`Conv3d`], [`Conv1d`],
//! [`ConvTranspose1d`], [`UlyssesExchange`],
//! [`Sampler`] and [`ParallelLMHead`]; the diffusion and
//! video models add [`Patch`], [`UnpatchifyVideo`](layers::UnpatchifyVideo),
//! [`CrossFade`], [`RgbBytes`],
//! [`Transpose`](layers::Transpose) and [`SegmentAttention`], and the
//! audio path adds
//! [`Snake`], [`AdaptivePool`] and
//! [`BiasAdd`](layers::BiasAdd). Each holds weight views and geometry, and its forward calls
//! operators.
//!
//! ## The capabilities
//!
//! A model hands the executor one [`Block`] per named capability it implements: [`Encoder`],
//! [`Denoiser`], [`Decoder`] or [`CausalLm`] ([`capability`]). The executor dispatches by the
//! capability, never by the concrete model.
//!
//! ## Checkpoints and weights
//!
//! [`Checkpoint`] is a set of safetensors shards with the identity of every file;
//! [`WeightLoader`] converts a tensor once into the layout its operator declares and publishes it
//! to retained storage, keyed by the identity rules of specs/worker.md 8.

pub mod capability;
pub mod checkpoint;
pub mod error;
pub mod exchange;
pub mod forward;
pub mod group;
pub mod layers;
pub mod model;
pub mod tensor;
pub mod weights;

pub use capability::{
    BatchRow, BatchRows, CausalLm, ConditionPiece, ConditionRole, ConditionShape, Conditioning,
    DecodedMedia, Decoder, Denoiser, Encoder, EncoderInputs, EncoderLayout, EncoderOutputs,
    EncoderPrompt, ImageInput, ImageShape, KvGroup, KvPlanes, KvView, Latents, Layout, MediaFormat,
    RowKind, Seed, StateTable, Step, TokenBatch, VideoRequest, VideoTask,
};
pub use checkpoint::{Checkpoint, CheckpointFile, FileIdentity, Reading, TensorLocation};
pub use error::{Error, Result};
pub use exchange::ExchangeViews;
pub use forward::{Arena, AttentionProbe, Forward};
pub use group::Group;
pub use layers::{
    Activation, AdaptivePool, Attention, AttentionBackend, AudioActivation, Candidates, Conv1d,
    Conv3d, ConvGeometry, ConvTranspose1d, CrossFade, Embedding, FrameMoments, FrameNormPad,
    FramePad, GatedActivation, GatedResidual, Gelu, LayerNorm, Linear, ModulatedRmsNorm, Moe,
    Noise, NoiseKey, PackedSwiGlu, ParallelLMHead, Patch, PcmInterleave, RgbBytes, RmsNorm, Rotary,
    Rotate, Rotation, Sampler, SamplerState, Samples, ScaledAdd, ScaledResidual,
    ScaledResidualLayerNorm, ScaledResidualRmsNorm, Sections, SegmentAttention, SegmentTables,
    Silu, Snake, SwiGlu, TRTLLM_WORKSPACE_BYTES, UlyssesExchange, VsaTiles, WeightedRmsNorm,
    gather_rows,
};
pub use model::{Block, BlockName, Model, PhaseId};
pub use ops::PhaseKind;
pub use ops::{DType, DevicePtr, MAX_RANK, Shape, Strides, TensorView};
pub use tensor::Backing;
pub use weights::{
    HostStore, Quantization, QuantizedStack, QuantizedWeight, RetainedWeights, ValueLayout, Weight,
    WeightKey, WeightLayout, WeightLoader,
};
