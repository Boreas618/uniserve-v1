//! Configuration read before a model is loaded: the devices a rank computes on.
//!
//! The values here are read once from an engine's launch and are immutable afterwards. Options
//! of the numerical backend stay with the backend, and the sizes device memory is fitted against
//! stay with the owners of that memory.

mod device;

pub use device::Device;
