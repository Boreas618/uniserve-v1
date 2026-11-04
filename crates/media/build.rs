//! Locates the FFmpeg 8.1.2 runtime the crate's codecs call and compiles the
//! C shim over its headers.
//!
//! The shim (`src/ffi/shim.c`) includes the real libav* headers, so this crate
//! keeps up with the ABI instead of mirroring FFmpeg's structs in Rust. It is
//! linked against the shared libraries of one installation, named by
//! `UNISERVE_FFMPEG_DIR` or, when unset, the workspace's provisioned runtime
//! under `artifacts/native/prefix` (see `crates/media/README.md`).
//!
//! The runtime path is recorded in every executable this package builds, so a
//! test binary runs without a wrapper script.

use std::path::{Path, PathBuf};

/// Libraries the shim calls, in link order.
const LIBRARIES: [&str; 5] = ["avformat", "avcodec", "swscale", "swresample", "avutil"];

fn main() {
    println!("cargo:rerun-if-env-changed=UNISERVE_FFMPEG_DIR");
    println!("cargo:rerun-if-changed=src/ffi/shim.c");

    let prefix = ffmpeg_prefix();
    let include = prefix.join("include");
    let lib = prefix.join("lib");
    if !include.join("libavcodec").join("avcodec.h").is_file() {
        panic!(
            "FFmpeg 8.1.2 headers are missing from {}; set UNISERVE_FFMPEG_DIR to an \
             installation's prefix (crates/media/README.md describes how one is provisioned)",
            prefix.display()
        );
    }
    if !has_library(&lib) {
        panic!(
            "FFmpeg shared libraries are missing from {}; the media crate loads no static \
             FFmpeg (crates/media/README.md describes how the runtime is provisioned)",
            lib.display()
        );
    }

    cc::Build::new()
        .file("src/ffi/shim.c")
        .include(&include)
        .warnings(true)
        .compile("uniserve_media_shim");

    println!("cargo:rustc-link-search=native={}", lib.display());
    for name in LIBRARIES {
        println!("cargo:rustc-link-lib=dylib={name}");
    }
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    // The runtime's own `bin/ffmpeg` is the default reference-video decoder:
    // the same installation that provides the codecs provides the external
    // program, so a deployment needs no other media tooling on its path.
    println!(
        "cargo:rustc-env=UNISERVE_MEDIA_RUNTIME={}",
        prefix.display()
    );
}

/// The installation the crate is built against.
fn ffmpeg_prefix() -> PathBuf {
    if let Some(dir) = std::env::var_os("UNISERVE_FFMPEG_DIR") {
        return PathBuf::from(dir);
    }
    let manifest = match std::env::var_os("CARGO_MANIFEST_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => panic!("Cargo sets CARGO_MANIFEST_DIR for a build script"),
    };
    let workspace = manifest.join("../..");
    workspace.join("artifacts/native/prefix")
}

/// Whether `lib` holds a shared libav* library.
fn has_library(lib: &Path) -> bool {
    LIBRARIES.iter().any(|name| {
        ["so", "dylib"].iter().any(|extension| {
            let file = format!("lib{name}.{extension}");
            lib.join(&file).is_file()
                || std::fs::read_dir(lib).is_ok_and(|mut entries| {
                    entries.any(|entry| {
                        entry.is_ok_and(|entry| {
                            entry
                                .file_name()
                                .to_string_lossy()
                                .starts_with(&format!("{file}."))
                        })
                    })
                })
        })
    })
}
