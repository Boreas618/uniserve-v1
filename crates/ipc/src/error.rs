//! Failures at the channel boundary.

use std::borrow::Cow;
use std::io;
use std::sync::Arc;

/// Result type returned by channel and frame calls.
pub type IpcResult<T> = std::result::Result<T, IpcError>;

/// Encoding and transport failures at the channel boundary.
#[derive(Debug, thiserror::Error)]
pub enum IpcError {
    /// A message could not be encoded, or received bytes are not a frame of
    /// the message expected.
    #[error(transparent)]
    Encoding(EncodingError),
    /// A channel's setup or one of its calls failed, or its peer closed it.
    #[error(transparent)]
    Transport(TransportError),
}

/// A message that could not be framed, or a frame that does not decode.
#[derive(Debug, thiserror::Error)]
pub enum EncodingError {
    /// MessagePack could not encode the message.
    #[error("worker message encoding error: encoding a message")]
    Encode(#[source] rmp_serde::encode::Error),
    /// The frame's bytes are not an encoding of the message expected, for
    /// example a message of another type or of another build.
    #[error("worker message encoding error: decoding a message")]
    Decode(#[source] rmp_serde::decode::Error),
    /// The encoded message is longer than a frame's length field can state.
    #[error(
        "worker message encoding error: a message of {length} bytes exceeds the largest frame of {limit} bytes"
    )]
    Oversized {
        /// Bytes of the encoded message.
        length: usize,
        /// The largest message a frame carries: the length field's range.
        limit: usize,
    },
}

/// A channel's failed setup or call, or its peer's close.
///
/// A channel reports its terminal failure again on every later call, so the
/// error is cloneable: an operating-system failure is shared rather than
/// rendered into the message.
#[derive(Debug, Clone, thiserror::Error)]
#[error("worker IPC error: {what}")]
pub struct TransportError {
    /// What failed: the step the channel was taking, or what it observed.
    what: Cow<'static, str>,
    /// The operating system's failure of that step, if there was one.
    #[source]
    source: Option<Arc<io::Error>>,
}

impl TransportError {
    /// `step` failed with `source`.
    pub fn io(step: impl Into<Cow<'static, str>>, source: io::Error) -> Self {
        Self {
            what: step.into(),
            source: Some(Arc::new(source)),
        }
    }

    /// The channel observed `what`, such as its peer's close, without an
    /// operating-system failure.
    pub fn closed(what: impl Into<Cow<'static, str>>) -> Self {
        Self {
            what: what.into(),
            source: None,
        }
    }
}

impl From<EncodingError> for IpcError {
    fn from(error: EncodingError) -> Self {
        Self::Encoding(error)
    }
}

impl From<TransportError> for IpcError {
    fn from(error: TransportError) -> Self {
        Self::Transport(error)
    }
}
