//! The run path of this package's test executables to the FFmpeg libraries they link through the
//! media probe.
//!
//! `crates/media` states, as its `links = "ffmpeg"` metadata, the directories that hold the FFmpeg
//! libraries when they are not on the default loader path (`DEP_FFMPEG_LIB`). Cargo passes a
//! dependency's search path and library names on to this package but not its link arguments, so
//! the package records those directories in its own executables' run path.

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    if let Some(directories) = std::env::var_os("DEP_FFMPEG_LIB") {
        for directory in std::env::split_paths(&directories) {
            println!("cargo::rustc-link-arg=-Wl,-rpath,{}", directory.display());
        }
    }
}
