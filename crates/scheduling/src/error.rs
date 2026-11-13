//! Why a policy module could not be compiled or loaded.

/// A failure of [`crate::Engine`]: the host has no engine for policy modules, or the bytes are
/// not a module it compiles or loads.
#[derive(Debug, thiserror::Error)]
pub enum PolicyError {
    /// The module bytes were rejected: not a wasm32 module, a component, an
    /// artifact precompiled for another engine, or a package that does not
    /// compile; or this host has no engine to compile them with.
    #[error("the policy module was rejected: {message}")]
    Load { message: String },
}
