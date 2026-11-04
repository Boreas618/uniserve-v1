//! Where the external media executables of one deployment live.

use std::path::PathBuf;

/// The media configuration of a main process.
///
/// The codecs are linked into the executable (through the crate's FFmpeg
/// shim), so the only external program media work starts is the reference
/// video decoder, which stays an external `ffmpeg` process by design
/// (`specs/worker.md` §9). The main process takes its value from the build
/// configuration; [`MediaConfig::default`] resolves it from the environment
/// for tests and tools.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MediaConfig {
    /// The `ffmpeg` executable that decodes reference videos.
    pub ffmpeg: PathBuf,
}

impl MediaConfig {
    /// The configuration named by `UNISERVE_FFMPEG`, or `ffmpeg` from the
    /// loader's search path.
    pub fn from_env() -> Self {
        Self {
            ffmpeg: std::env::var_os("UNISERVE_FFMPEG")
                .map_or_else(|| PathBuf::from("ffmpeg"), PathBuf::from),
        }
    }
}

impl Default for MediaConfig {
    fn default() -> Self {
        Self::from_env()
    }
}
