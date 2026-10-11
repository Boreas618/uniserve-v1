//! Failures at the channel boundary.

/// Result type returned by channel and frame calls.
pub type IpcResult<T> = std::result::Result<T, IpcError>;

/// Encoding and transport failures at the channel boundary.
#[derive(Debug, thiserror::Error)]
pub enum IpcError {
    /// A message could not be encoded, or received bytes are not a frame of
    /// the message expected.
    #[error("worker message encoding error: {0}")]
    Encoding(String),
    /// A channel's setup or one of its calls failed, or its peer closed it.
    #[error("worker IPC error: {0}")]
    Transport(String),
}
