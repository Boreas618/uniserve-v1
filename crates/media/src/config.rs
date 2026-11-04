//! Where the external media executables of one deployment live.

use std::path::{Path, PathBuf};

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
    /// The configuration named by `UNISERVE_FFMPEG`, else the `ffmpeg` of the
    /// runtime the crate links (the installation's `bin/ffmpeg`), else
    /// `ffmpeg` from the loader's search path.
    pub fn from_env() -> Self {
        Self {
            ffmpeg: ffmpeg_executable(),
        }
    }
}

/// The reference-video decoder of this deployment.
///
/// The linked runtime usually ships its own `ffmpeg`, and it is the natural
/// decoder because it is the build whose codecs the crate was compiled
/// against; a deployment that provisions its decoder elsewhere names it with
/// `UNISERVE_FFMPEG`.
fn ffmpeg_executable() -> PathBuf {
    if let Some(path) = std::env::var_os("UNISERVE_FFMPEG") {
        return PathBuf::from(path);
    }
    let runtime = Path::new(env!("UNISERVE_MEDIA_RUNTIME"))
        .join("bin")
        .join("ffmpeg");
    if runtime.is_file() {
        return runtime;
    }
    PathBuf::from("ffmpeg")
}

impl Default for MediaConfig {
    fn default() -> Self {
        Self::from_env()
    }
}
