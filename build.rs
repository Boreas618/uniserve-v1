//! The run path of this package's executables to the FFmpeg libraries they link.
//!
//! `crates/media` links the FFmpeg libraries and states, as its `links = "ffmpeg"` metadata,
//! the directories that hold them when they are not on the default loader path
//! (`DEP_FFMPEG_LIB`). Cargo passes a dependency's search path and library names on to this
//! package but not its link arguments, so the package records those directories in its own
//! executables' run path: the loader then finds the libraries without any environment, and the
//! linker resolves the libraries they need in turn.

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    if let Some(directories) = std::env::var_os("DEP_FFMPEG_LIB") {
        for directory in std::env::split_paths(&directories) {
            println!("cargo::rustc-link-arg=-Wl,-rpath,{}", directory.display());
        }
    }
}
