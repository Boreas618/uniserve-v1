//! The shared computation library: tensor views, numerical layers, capability traits, the model
//! families execution loads, and the checkpoint and weight-loading boundary.
//!
//! ## Where this crate sits
//!
//! The call chain of one forward is `runner → model → layer → operator → kernel`. A
//! model package composes the layers here; a layer calls the operators of
//! [`ops`] and nothing below them; the operator's launch code reaches the loaded kernel. `nn` owns
//! no runtime object: it never allocates device memory, creates a stream, captures a graph or
//! selects a program. Everything a forward needs arrives borrowed in [`Forward`].
//!
//! ## The tensors
//!
//! [`TensorView`], [`Shape`], [`Strides`], [`DType`] and [`DevicePtr`] are [`ops`] types: they are
//! the ABI of an operator's arguments, and `nn` re-exports them rather than defining a second
//! representation. A [`Weight`] pairs a view with the [`Backing`] the worker's retained storage
//! gives it, so a layer's weights are plainly borrowed and plainly retained.
//!
//! ## The layers
//!
//! [`layers`] holds what a model composes: [`Linear`],
//! [`RmsNorm`], [`LayerNorm`], [`Rotary`],
//! [`VocabParallelEmbedding`](layers::VocabParallelEmbedding), [`DenseAttention`](layers::DenseAttention),
//! [`PagedAttention`](layers::PagedAttention), [`Moe`],
//! [`Conv3d`], [`Conv1d`],
//! [`ConvTranspose1d`], [`UlyssesAttention`](layers::UlyssesAttention),
//! and [`ParallelLMHead`]; the diffusion and
//! video models add [`Patchify`](layers::Patchify), [`Copy`](layers::Copy),
//! [`Transpose`](layers::Transpose) and [`VsaAttention`](layers::VsaAttention), and the
//! audio path adds
//! [`Snake`], [`AdaptiveAvgPool2d`] and
//! [`ChannelAffine`](layers::ChannelAffine). Each holds weight views and geometry, and its forward calls
//! operators.
//!
//! Beside the layers, [`diffusion`] holds the flow-matching schedules and the Euler step, [`tiling`]
//! the tiled VAE grid, blends and encode and decode drivers, [`vocoder`] the BigVGAN generator and
//! [`codec`] the DAC encoder.
//!
//! ## The capabilities
//!
//! A model's blocks implement the capabilities execution calls: [`Encoder`], [`Denoiser`],
//! [`Decoder`] or [`CausalLm`] ([`capability`]). A model crate exports its [`family::Family`], the
//! one value execution resolves a checkpoint with and receives the loaded model's blocks from; it
//! dispatches by the capability, never by the concrete model.
//!
//! ## Checkpoints and weights
//!
//! [`Checkpoint`] is a set of safetensors shards with the identity of every file;
//! [`WeightLoader`] converts a tensor once into the layout its operator declares and publishes it
//! to retained storage, keyed by the identity of the files it was read from, its converted element
//! type, its layout and its region.

pub mod adapter;
pub mod blocks;
pub mod capability;
pub mod checkpoint;
pub mod codec;
pub mod config;
pub mod diffusion;
pub mod error;
pub mod family;
pub mod forward;
pub mod group;
pub mod layers;
pub mod quantization;
pub mod tensor;
pub mod tiling;
pub mod timestep;
pub mod vocoder;
pub mod weights;

pub use adapter::{Adapter, Fusion};
pub use capability::{
    CausalLm, ConditionPiece, ConditionRole, ConditionShape, Conditioning, DecodedMedia, Decoder,
    Denoiser, Encoder, EncoderInputs, EncoderLayout, EncoderOutputs, EncoderPrompt, ImageInput,
    ImageShape, KvGroup, KvPlanes, KvView, Latents, MediaFormat, PatchFormat, PixelNorm, Placed,
    Seed, Splicer, StateTable, Step, TableValues, TokenBatch, VideoRequest, WarmPrompt,
    WarmRequest,
};
pub use checkpoint::{Checkpoint, CheckpointFile, FileIdentity, Reading, TensorLocation};
pub use error::{Error, Result};
pub use forward::{Arena, Fence, Forward, Split};
pub use group::{Group, Transfer};
pub use layers::{
    Activation, AdaptiveAvgPool2d, Axpby, Cast, CausalPad, Conv1d, Conv3d, ConvSpec,
    ConvTranspose1d, Epilogue, GateBias, GatedActivation, Gelu, GroupNorm, Halves, IndexRows,
    LayerNorm, Linear, Modulation, Moe, Noise, NoiseKey, Padding, ParallelLMHead, Residual,
    ResidualStream, RmsNorm, Rotary, Rounding, Scale, Silu, Snake,
};
pub use ops::{DType, DevicePtr, MAX_RANK, Shape, Strides, TensorView};
pub use proto::program::PhaseKind;
pub use tensor::Backing;
pub use weights::{
    Quantization, QuantizedStack, QuantizedWeight, RetainedWeights, Scope, ValueLayout, Weight,
    WeightKey, WeightLayout, WeightLoader,
};
