//! Failures of media codecs, containers and media reading.

use std::path::PathBuf;

/// A media operation's failure.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The plan and the media disagree: a decoded extent, rate or sample
    /// count is not the one the request was planned with, or a caller passed
    /// data of the wrong shape.
    #[error("{0}")]
    Plan(String),

    /// A codec, the muxer or a decoder failed.
    #[error("{0}")]
    Codec(String),

    /// The serving encoders are not available in the FFmpeg runtime this
    /// build links against.
    #[error("media output requires the codec {0}, which the FFmpeg runtime does not provide")]
    MissingCodec(String),

    /// A codec or reference-video process failed at the operating system.
    #[error("{context}: {source}")]
    Io {
        /// What the operation was doing.
        context: String,
        /// The operating system's failure.
        #[source]
        source: std::io::Error,
    },

    /// The reference video's FFmpeg process exited in failure.
    #[error("decoding a reference video with {ffmpeg} failed: {message}")]
    ReferenceVideo {
        /// The executable that failed.
        ffmpeg: PathBuf,
        /// Its standard error, trimmed.
        message: String,
    },
}

impl Error {
    /// A plan violation with a formatted reason.
    pub(crate) fn plan(reason: impl Into<String>) -> Self {
        Self::Plan(reason.into())
    }

    /// An operating-system failure with its context.
    pub(crate) fn io(context: impl Into<String>, source: std::io::Error) -> Self {
        Self::Io {
            context: context.into(),
            source,
        }
    }
}

/// A media operation's result.
pub type Result<T> = std::result::Result<T, Error>;
