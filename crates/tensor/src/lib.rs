//! Tensors for UniServe: n-dimensional arrays of one element type on a device, their views and
//! the operations on them.
//!
//! ```rust
//! use tensor::{Tensor, DType, Device};
//! # use tensor::Error;
//! # fn main() -> Result<(), Error>{
//!
//! let a = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
//! let b = Tensor::arange(0f32, 12f32, &Device::Cpu)?.reshape((3, 4))?;
//! let c = a.matmul(&b)?;
//!
//! # Ok(())}
//! ```
//!
//! The crate serves inference only: a tensor is a value, it records no computation graph and
//! there are no gradients.
//!
//! ## Devices
//!
//! A [`Device`] is the CPU, whose backend is built into this crate, or a backend device
//! implemented outside of it through the two object-safe traits of [`backend`]:
//! [`backend::BackendDevice`] and [`backend::BackendStorage`]. Tensor operations are the same on
//! every device; each one checks its operands and calls one primitive of the storage's backend.
//! The operands of one operation are on the same device, and [`Tensor::to_device`] moves a
//! tensor between the CPU and a backend device.
//!
//! ## Storage access
//!
//! [`Tensor::from_storage`], [`Tensor::storage_and_layout`] and
//! [`Tensor::storage_mut_and_layout`] give a backend access to the storage under a tensor.
//!
//! ## Origin
//!
//! The crate is derived from `candle-core` of
//! [huggingface/candle](https://github.com/huggingface/candle); see the repository's README.

pub mod backend;
pub mod conv;
mod convert;
pub mod cpu;
pub mod cpu_backend;
mod device;
pub mod display;
mod dtype;
pub mod error;
mod indexer;
pub mod layout;
pub mod nditer;
pub mod op;
pub mod scalar;
pub mod shape;
mod sort;
mod storage;
mod strided_index;
mod tensor;
mod tensor_cat;

pub use cpu_backend::{CpuStorage, CpuStorageRef};
pub use device::{Device, DeviceLocation, NdArray};
pub use dtype::{DType, DTypeParseError, FloatDType, IntDType, WithDType};
pub use error::{Context, Error, Result};
pub use indexer::{IndexOp, TensorIndexer};
pub use layout::{Layout, LayoutRelation};
pub use nditer::NdIter;
pub use shape::{D, Shape};
pub use storage::{Storage, StorageMutRef, StorageRef};
pub use strided_index::{StridedBlocks, StridedIndex};
pub use tensor::{Tensor, TensorId};

pub trait ToUsize2 {
    fn to_usize2(self) -> (usize, usize);
}

impl ToUsize2 for usize {
    fn to_usize2(self) -> (usize, usize) {
        (self, self)
    }
}

impl ToUsize2 for (usize, usize) {
    fn to_usize2(self) -> (usize, usize) {
        self
    }
}

/// Defining a module with forward method using a single argument.
pub trait Module {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

impl<T: Fn(&Tensor) -> Result<Tensor>> Module for T {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self(xs)
    }
}

impl<M: Module> Module for Option<&M> {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            None => Ok(xs.clone()),
            Some(m) => m.forward(xs),
        }
    }
}
