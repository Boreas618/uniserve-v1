//! The neural-network library: modules, fused operators, weight loading and the capability traits
//! execution serves a model through.

pub mod adapter;
pub mod checkpoint;
pub mod collective;
pub mod config;
pub mod error;
pub mod profiler;
pub mod quantization;
pub mod var_builder;
pub mod weights;

pub use adapter::{Adapter, Fusion};
pub use checkpoint::{Checkpoint, CheckpointFile, FileIdentity, TensorLocation};
pub use error::{Error, Result};
pub use proto::program::PhaseKind;
pub use uniserve_tensor::{D, DType, Device, IndexOp, Module, Shape, Tensor};
pub use var_builder::VarBuilder;
pub use weights::{
    CpuWeights, PartBlock, Provenance, Quantization, RetainedWeights, WeightKey, WeightLayout,
    WeightLoader, WeightSource,
};
