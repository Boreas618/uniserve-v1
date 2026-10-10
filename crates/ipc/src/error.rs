//! Failures at the channel boundary.

use crate::ValidationError;

/// Result type returned by channel and frame calls.
pub type IpcResult<T> = std::result::Result<T, IpcError>;

/// Encoding, validation and transport failures at the channel boundary.
#[derive(Debug, thiserror::Error)]
pub enum IpcError {
    /// A message was not accepted because the bounded send queue is full.
    #[error("channel send queue is full")]
    WouldBlock,
    /// A message could not be encoded, or received bytes are not a frame of
    /// the message expected.
    #[error("worker message encoding error: {0}")]
    Encoding(String),
    /// A message violates the worker protocol contract.
    #[error(transparent)]
    Validation(#[from] ValidationError),
    /// A channel's setup or one of its calls failed, or its peer closed it.
    #[error("worker IPC error: {0}")]
    Transport(String),
}
