//! Configuration resolved before model loading and physical capacity fitting.
//!
//! What the design calls the build config (`specs/worker.md` §7.1) is resolved
//! here, before a main process loads a model: the deployment's placement of
//! blocks on devices, the devices themselves, and the values physical capacity
//! is fitted against. Numerical backend options stay with their program.
//!
//! Only the device grammar the executor needs so far lives here; the rest of
//! the build config arrives with the main process's launch.

mod device;

pub use device::Device;

/// A malformed launch descriptor or configuration value.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct DescriptorError(String);

impl DescriptorError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}
