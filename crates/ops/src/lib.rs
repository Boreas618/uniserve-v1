//! The interface between Rust and kernel programs: operators and their launch code, the kernels a
//! program compiles into, the generation of kernels a runner serves, and the shadow buffers and
//! tolerances that decide whether a new implementation may replace an old one.
//!
//! ## The chain this crate sits in
//!
//! `runner → model → layer → operator → kernel` (specs/worker.md 4). Each operator here is one Rust
//! type with a name, a descriptor, and an argument struct; a kernel program — a Triton package, a
//! CuTe DSL package, a CUDA C++ package or a native library — fills that operator, and
//! `uniserve engine` resolves the programs the epoch record names and hands the modules to
//! [`Kernel`]. `ops` depends on `platform` alone: it launches what it is given, allocates nothing
//! that outlives a launch, owns no model and knows no layer.
//!
//! ## The operators
//!
//! Dense and layer math:
//!
//! | Operator | `NAME` | Module |
//! | --- | --- | --- |
//! | [`gemm::Gemm`] | `gemm` | [`gemm`] |
//! | [`norm::RmsNorm`] | `rms_norm` | [`norm`] |
//! | [`norm::AddRmsNorm`] | `add_rms_norm` | [`norm`] |
//! | [`norm::LayerNorm`] | `layer_norm` | [`norm`] |
//! | [`norm::Softmax`] | `softmax` | [`norm`] |
//! | [`embedding::Embedding`] | `embedding` | [`embedding`] |
//!
//! Position and cache:
//!
//! | Operator | `NAME` | Module |
//! | --- | --- | --- |
//! | [`rope::RotaryFactors`] | `rotary_factors` | [`rope`] |
//! | [`rope::QkNormRope`] | `qk_norm_rope` | [`rope`] |
//! | [`kv::KvWrite`] | `kv_write` | [`kv`] |
//!
//! Logits and sampling:
//!
//! | Operator | `NAME` | Module |
//! | --- | --- | --- |
//! | [`logits::LogitsGather`] | `logits_gather` | [`logits`] |
//! | [`logits::VocabUnpad`] | `vocab_unpad` | [`logits`] |
//! | [`sampler::LogitsShape`] | `logits_shape` | [`sampler`] |
//! | [`sampler::TopK`] | `top_k` | [`sampler`] |
//! | [`sampler::FilterCandidates`] | `filter_candidates` | [`sampler`] |
//! | [`sampler::InverseCdf`] | `inverse_cdf` | [`sampler`] |
//! | [`sampler::ArgMax`] | `argmax` | [`sampler`] |
//! | [`sampler::LogSoftmax`] | `log_softmax` | [`sampler`] |
//! | [`sampler::LogprobGather`] | `logprob_gather` | [`sampler`] |
//!
//! Convolutions and collectives:
//!
//! | Operator | `NAME` | Module |
//! | --- | --- | --- |
//! | [`conv::Conv3d`] | `conv3d` | [`conv`] |
//! | [`conv::Conv1d`] | `conv1d` | [`conv`] |
//! | [`conv::ConvTranspose1d`] | `conv_transpose1d` | [`conv`] |
//! | [`exchange::scatter`], [`exchange::gather`] | `ulysses_exchange` | [`exchange`] |
//!
//! The attention and mixture-of-experts operators are `attention` and `moe`, beside these.
//!
//! ## Descriptor keys
//!
//! A descriptor's `Display` is its key, and the key is the last segment of the module's path
//! (`<operator>/<backend>/<revision>/<target>/<key>`, specs/worker.md 5.3). Keys are canonical:
//! fields in declaration order, `<name>_<value>` joined by `_`, and a floating-point value as its
//! bit pattern ([`descriptor::Float`]). Nothing about a launch may enter a key and nothing a
//! compilation bakes in may stay out of it.
//!
//! ## Tolerances
//!
//! [`Tolerance`] states how far two correct implementations of an operator may be; the bound is
//! derived from the reduction length, the accumulation dtype and the output dtype, operator by
//! operator, and never from a measurement of the pair being compared.

pub mod activation;
pub mod attention;
pub mod audio_resample;
pub mod bias;
pub mod block_copy;
pub mod capture;
pub mod commit;
pub mod conv;
pub mod descriptor;
pub mod embedding;
pub mod error;
pub mod exchange;
pub mod fp4;
pub mod frame;
pub mod gemm;
pub mod generation;
pub mod install;
pub mod kernel;
pub mod kv;
pub mod logits;
pub mod metadata;
pub mod modulation;
pub mod moe;
pub mod noise;
pub mod norm;
pub mod op;
pub mod phase;
pub mod platform;
pub mod prefix_cache;
pub mod replicate_pad;
pub mod residual;
pub mod resize;
pub mod rope;
pub mod sampler;
pub mod segments;
pub mod snake;
pub mod snake_beta;
pub mod tensor;
pub mod tolerance;
pub mod twins;

pub use descriptor::{Float, Key};
pub use error::{Error, Result};
pub use fp4::{Nvfp4Gemm, Nvfp4Quantize, ScaleLayout};
pub use generation::{Generation, KernelEntry};
pub use install::{Module, install, operator_name, operators};
pub use kernel::{Arguments, CubinMetadata, Entry, Kernel, LaunchGrid, NativeCall, NativeEntry};
pub use op::Op;
pub use phase::PhaseKind;
pub use platform::{
    CollectiveType, Communicator, DeviceRange, PlatformDevicePtr, ReduceOp, Stream, collective_type,
};
pub use tensor::{DType, DevicePtr, MAX_RANK, Shape, Strides, TensorView};
pub use tolerance::Tolerance;
pub use twins::{Twin, Twins};
