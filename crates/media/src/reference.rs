//! The reference fixtures the readers and resamplers are compared with.
//!
//! A fixture generator under `tests/fixtures/` runs the reference
//! implementations (Pillow, torch, torchvision, torchaudio) in the Python
//! interpreter `UNISERVE_REFERENCE_PYTHON` names and writes their outputs into
//! a directory of the test process's own. The tests that compare with them are
//! `#[ignore]`d with that requirement, so a run that asks for them without the
//! interpreter fails instead of passing without a comparison.

use std::path::Path;

/// Runs the fixture generator `script` (relative to this crate) into a fresh
/// directory and returns it.
///
/// The generator decodes and encodes its reference video with the `ffmpeg`
/// this crate's [`crate::MediaConfig`] resolves, passed as `UNISERVE_FFMPEG`,
/// so the reference and the reader use the same FFmpeg build.
///
/// # Panics
///
/// When `UNISERVE_REFERENCE_PYTHON` is unset, or the generator fails.
pub(crate) fn generate(script: &str) -> tempfile::TempDir {
    let python = std::env::var_os("UNISERVE_REFERENCE_PYTHON").unwrap_or_else(|| {
        panic!(
            "UNISERVE_REFERENCE_PYTHON must name an interpreter with the reference \
             implementations that {script} imports"
        )
    });
    let directory = tempfile::tempdir().expect("a fixture directory");
    let status = std::process::Command::new(python)
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join(script))
        .arg(directory.path())
        .env("UNISERVE_FFMPEG", crate::MediaConfig::from_env().ffmpeg)
        .status()
        .expect("running the fixture generator");
    assert!(status.success(), "{script} failed: {status}");
    directory
}
