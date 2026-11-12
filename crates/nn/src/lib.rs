//! The neural-network library: modules, fused operators, weight loading and the capability traits
//! execution serves a model through, over the tensor library.
//!
//! ## Where this crate sits
//!
//! Three layers, as in PyTorch: the tensor library (`uniserve-tensor`: tensors, views, the CPU
//! device), the device backend (`backend`: the CUDA and meta devices, the execution context, the
//! one dispatch point of every kernel) and this crate, the `torch.nn` of the tree. A model crate
//! depends on this crate, the tensor library and `processing` only; what it may name of the
//! layers below is re-exported here ([`Tensor`], [`DType`], [`Device`], [`Shape`], [`D`],
//! [`IndexOp`], [`Module`], [`collective`], [`profiler`]).
//!
//! `nn` owns no runtime object: it never creates a context, a stream or a graph, and it
//! allocates only through the tensor operations and fused operators it computes with. Which
//! stream, kernels and memory a forward uses is the execution context installed on the thread
//! by execution, which a model never names.
//!
//! ## Conventions
//!
//! Every module, here and in a model crate, has one shape:
//!
//! - A **module** is a struct that holds its weights, as [`Tensor`]s, and its sub-modules. It
//!   holds nothing of execution.
//! - Its **config** is an immutable typed struct, named `<Module>Config`, of the widths, names
//!   and arithmetic choices the checkpoint states for it and of this rank's share
//!   ([`collective::Split`]); it derives `serde` where it is read from a checkpoint's JSON. A
//!   module's constructor consumes typed fields and reparses nothing.
//! - It is built by `Module::load(config: &Config, vb: VarBuilder) -> Result<Self>`, or
//!   `load(vb)` when it has no config. The builder names the module's tensors under its path
//!   ([`VarBuilder::pp`], [`VarBuilder::get`]), so one `load` serves a device, the CPU and a
//!   profile run: what a load is follows from the [`WeightLoader`] the builder is over.
//! - It is computed by `forward`, which takes and returns tensors and returns this crate's
//!   [`Result`]. A module of one input also implements the tensor library's [`Module`], so it
//!   composes with `Tensor::apply`; a module of more inputs takes them as arguments.
//! - A function of [`functional`] takes and returns tensors and holds no state: a fused
//!   operator with its CPU formula, or a stateless function.
//! - Nothing takes a context, a stream, an arena or a kernel argument. A destination argument
//!   exists only for memory execution owns and lends: key/value pages, recurrent state,
//!   latents and product buffers ([`capability`]).
//! - A module states its forward as a profiler range ([`profiler::range`]) where a deployment
//!   times it, with its class and the path its builder loaded it under
//!   ([`VarBuilder::path`]).
//! - A failure is this crate's [`Error`], which holds the tensor library's and the backend's
//!   as typed causes; a model crate converts it into its own.
//!
//! ## What a forward does
//!
//! There are no execution modes. On CPU tensors a function computes its formula with tensor
//! operations; on tensors of a GPU it launches kernels through the context; on meta tensors,
//! under a meta context, it checks its arguments, states the kernel it would launch and
//! computes nothing, which is a profile run. The same `forward` is all three.
//!
//! ## The parts
//!
//! | Module | Holds |
//! | --- | --- |
//! | [`var_builder`], [`weights`], [`checkpoint`], [`quantization`], [`adapter`] | [`VarBuilder`], [`WeightLoader`] and retained storage, safetensors shards, the quantization a checkpoint declares and its packed representations, adapter fusion |
//! | [`config`] | The strict reader of a checkpoint's JSON configuration |
//! | [`functional`] | The fused operators and stateless functions |
//! | [`modules`] | `Linear`, `Embedding`, `LmHead`, `RmsNorm`, `LayerNorm`, `Mlp`, `MoE`, `SharedExpert` and their parallel forms |
//! | [`capability`] | The traits execution calls and the input values of their methods |
//! | [`diffusion`] | Flow-matching schedules, the solver formulas and the sinusoidal time embedding |
//! | [`vae`] | The tiles of a raster, their blend and stitch, the tiled encode and the temporal clips |
//! | [`family`] | The family traits, the registry and the shapes execution sizes a model by |
//! | [`collective`], [`profiler`] | The model-facing forms of the backend's collectives and profiler ranges |

pub mod adapter;
pub mod capability;
pub mod checkpoint;
pub mod collective;
pub mod config;
pub mod diffusion;
pub mod error;
pub mod family;
pub mod functional;
pub mod modules;
pub mod profiler;
pub mod quantization;
pub mod vae;
pub mod var_builder;
pub mod weights;

pub use adapter::{Adapter, Fusion};
pub use capability::{
    CausalLm, Clip, ConditionPiece, ConditionShape, Conditioning, Decoder, Denoiser, Drafter,
    Encoder, EncoderLayout, ExpertExchange, KvCache, KvPlanes, MediaFormat, MediaInput, NO_SLOT,
    PatchFormat, PixelNorm, RecurrentState, Splice, Splicer, Step, TokenBatch, VerifySlots,
    VideoRequest, Vision,
};
pub use checkpoint::{Checkpoint, CheckpointFile, FileIdentity, TensorLocation};
pub use collective::Split;
pub use error::{Error, Result};
pub use proto::module::PhaseKind;
pub use proto::video::{ConditionRole, LatentShape, Layout, VisionGrid};
pub use uniserve_tensor::{D, DType, Device, IndexOp, Module, Shape, Tensor};
pub use var_builder::VarBuilder;
pub use weights::{
    CpuWeights, PartBlock, Provenance, Quantization, RetainedWeights, WeightKey, WeightLayout,
    WeightLoader, WeightSource,
};
