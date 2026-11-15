//! The device side of the tensor library: the interface between Rust and kernel packages, and the
//! CUDA and meta devices the tensor library computes on through it.
//!
//! An operator's launch code, the kernels a program compiles into, the generation of kernels a
//! runner serves, and the shadow buffers and tolerances that decide whether a new implementation
//! may replace an old one live here, beside the device backend that launches them: the execution
//! context a forward computes under, the storage of a device tensor and the allocator of its
//! memory, the generic primitives, host-to-device copies and collectives.
//!
//! ## The chain this crate sits in
//!
//! `runner → model → layer → operator → kernel`. Each operator here is one Rust
//! type with a name, a descriptor, and an argument struct; a kernel package — a Triton package, a
//! CuTe DSL package, a CUDA C++ package or a native library — fills that operator, and
//! `uniserve engine` resolves the programs the epoch record names and hands the modules to
//! [`Kernel`]. `ops` depends on `platform`, which makes every driver call, on the module
//! vocabulary of `proto` (operator names and phases) and on the tensor library, whose device
//! backend it is and whose element type an operand carries, with the number and spelling in the
//! package ABI [`abi`] states. It owns no model and knows no layer.
//!
//! ## What a model sees
//!
//! Tensors. A model computes with the tensor library's operations on tensors whose device is
//! [`cuda`] or [`meta`], and each operation is one launch of a generic primitive operator with an
//! output from the context's allocator. A tensor is freed when its last reference drops, which
//! returns its block to the allocator it came from.
//!
//! ## What execution does
//!
//! The context is the thread's: execution builds one [`Context`] per runner and installs it
//! around a model call ([`Context::enter`]), and a model never holds, receives or names it. What
//! a forward does follows from the context and from the capture state of its stream:
//!
//! - under a context that launches, operations issue kernels on its stream, and while the stream
//!   is being captured they become the nodes of the graph;
//! - under a **meta context** nothing is launched. Every operation checks its operands, states
//!   the key of the kernel it would launch and takes its output from the meta context's
//!   allocator, so a profile run yields a forward's launch table and memory peak, and the
//!   forward of `prepare_inputs` writes a replay's host values into the input buffer at the
//!   offsets the capture recorded.
//!
//! Execution makes the tensors a forward starts from: [`retained`] over a weight, [`external`]
//! over memory it lends for the call (KV pages, state, latents, a graph's inputs), and
//! [`empty`] for a new tensor of the installed context. A tensor that has no data, the input of
//! a profile run or of a `prepare_inputs` forward, is [`retained`] or [`external`] over the
//! [`meta`] device: it takes no block of any allocator and its address is never read.
//!
//! ## What a fused operator does
//!
//! Every kernel reaches the device through [`launch()`], the one dispatch of an operator, so
//! kernel updates, profile runs, graph rebinding and shadow comparison cover every launch alike.
//! A fused operator of `nn` allocates its output with [`empty`], states its arguments with
//! [`operand()`] and calls [`launch()`]. A partitioned forward meets the other ranks of its group
//! through [`collective`], a module states the range its forward runs in through
//! [`profiler::range`], and a forward states host work between two of its launches through
//! [`graph_break()`].
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
//! Encoded products and their factors:
//!
//! | Operator | `NAME` | Module |
//! | --- | --- | --- |
//! | [`fp8::Fp8Quantize`] | `fp8_quantize` | [`fp8`] |
//! | [`fp8::Fp8Gemm`] | `fp8_gemm` | [`fp8`] |
//! | [`fp4::Nvfp4Quantize`] | `nvfp4_quantize` | [`fp4`] |
//! | [`fp4::Nvfp4Gemm`] | `nvfp4_gemm` | [`fp4`] |
//! | [`dynamic_scale::DynamicScale`] | `dynamic_scale` | [`dynamic_scale`] |
//!
//! Position and cache:
//!
//! | Operator | `NAME` | Module |
//! | --- | --- | --- |
//! | [`rope::RotaryFactors`] | `rotary_factors` | [`rope`] |
//! | [`rope::QkNormRope`] | `qk_norm_rope` | [`rope`] |
//! | [`kv::KvWrite`] | `kv_write` | [`kv`] |
//! | [`kv::KvWriteScaled`] | `kv_write_scaled` | [`kv`] |
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
//! | [`conv::Conv`] | `conv` | [`conv`] |
//! | [`transpose::Transpose`] | `transpose` | [`transpose`] |
//!
//! The attention and mixture-of-experts operators are `attention` and `moe`, beside these.
//!
//! Tensor primitives, which read strided operands of any shape ([`strided`]):
//!
//! | Operator | `NAME` | Module |
//! | --- | --- | --- |
//! | [`elementwise::ElementwiseUnary`] | `elementwise_unary` | [`elementwise`] |
//! | [`elementwise::ElementwiseBinary`] | `elementwise_binary` | [`elementwise`] |
//! | [`elementwise::ElementwiseWhere`] | `elementwise_where` | [`elementwise`] |
//! | [`cast::Cast`] | `cast` | [`cast`] |
//! | [`copy_strided::CopyStrided`] | `copy_strided` | [`copy_strided`] |
//! | [`fill::Fill`] | `fill` | [`fill`] |
//! | [`reduce::Reduce`] | `reduce` | [`reduce`] |
//! | [`index::Index`] | `index` | [`index`] |
//!
//! ## Descriptor keys
//!
//! A descriptor's `Display` is its key, and the key is the last segment of the module's path
//! (`<operator>/<backend>/<revision>/<target>/<key>`). Keys are canonical:
//! fields in declaration order, `<name>_<value>` joined by `_`, and a floating-point value as its
//! bit pattern ([`descriptor::Float`]). Nothing about a launch may enter a key and nothing a
//! compilation bakes in may stay out of it.
//!
//! ## Tolerances
//!
//! [`Tolerance`] states how far two correct implementations of an operator may be; the bound is
//! derived from the reduction length, the accumulation dtype and the output dtype, operator by
//! operator, and never from a measurement of the pair being compared.

pub mod abi;
pub mod activation;
pub mod activation1d;
pub mod adaptive_pool;
pub mod allocator;
pub mod attention;
pub mod block_copy;
pub mod breaks;
pub mod cast;
pub mod channel_affine;
pub mod collective;
pub mod commit;
pub mod context;
pub mod conv;
pub mod copy_block;
pub mod copy_strided;
pub mod cross_fade;
pub mod descriptor;
mod device;
pub mod dynamic_scale;
pub mod elementwise;
pub mod embedding;
pub mod error;
pub mod fill;
pub mod fp4;
pub mod fp8;
pub mod frame;
pub mod gelu;
pub mod gemm;
pub mod generation;
pub mod index;
mod input;
pub mod install;
pub mod kernel;
pub mod kv;
mod launch;
pub mod logits;
pub mod modulation;
pub mod moe;
pub mod noise;
pub mod norm;
pub mod op;
mod operand;
pub mod patch;
pub mod pcm;
pub mod position;
pub mod posterior;
pub mod prefix_cache;
mod primitives;
pub mod profiler;
pub mod recurrent;
pub mod reduce;
pub mod residual;
pub mod rgb;
pub mod rope;
pub mod rows;
pub mod sampler;
pub mod snake;
mod storage;
pub mod strided;
pub mod tile_stitch;
pub mod tolerance;
pub mod transpose;
pub mod twins;
pub mod unpatchify;
pub mod values;
pub mod vsa;

pub use allocator::Allocator;
pub use breaks::graph_break;
pub use collective::Group;
pub use context::{Context, ContextGuard};
pub use descriptor::{Float, Key};
pub use device::{cuda, meta};
pub use error::{Error, Result};
pub use fp4::{Nvfp4Gemm, Nvfp4Quantize, ScaleLayout};
pub use generation::{Generation, KernelEntry};
pub use install::{LOAD_ENTRY, Module, install};
pub use kernel::{
    Arguments, CubinMetadata, Entry, Kernel, LaunchGrid, LoadCall, NativeCall, NativeEntry,
};
pub use launch::launch;
pub use op::{Call, Checked, Op, Route};
pub use operand::Operand;
pub use profiler::Profiler;
pub use storage::{empty, external, operand, retained};
pub use tolerance::Tolerance;
pub use twins::{Twin, Twins};
