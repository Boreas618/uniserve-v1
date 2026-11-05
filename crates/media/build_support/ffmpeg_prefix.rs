// Where the FFmpeg runtime this workspace links lives, resolved once for every build script that
// needs it.
//
// `crates/media` compiles the C shim over FFmpeg's headers and links the `libav*` shared
// libraries; the package that produces an executable must also record where those libraries live,
// because Cargo passes a dependency's `rustc-link-search` and `rustc-link-lib` on to dependents
// but not its `rustc-link-arg`. Both build scripts therefore `include!` this file, so the rule has
// one statement rather than two copies.
//
// `UNISERVE_FFMPEG_DIR` names one installation's prefix. Unset means the workspace's provisioned
// runtime under `artifacts/native/prefix`, which `crates/media/native/build.sh` builds and
// `crates/media/README.md` describes.

use std::path::PathBuf;

/// The prefix of the FFmpeg installation a build links against.
fn ffmpeg_prefix() -> PathBuf {
    if let Some(dir) = std::env::var_os("UNISERVE_FFMPEG_DIR") {
        return PathBuf::from(dir);
    }
    let manifest = std::env::var_os("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("Cargo sets CARGO_MANIFEST_DIR for a build script"));

    // A build script runs in the package it belongs to, which may be the workspace root or any
    // depth below it, so the workspace's provisioning directory is found by walking up to the
    // manifest that declares the workspace.
    let mut dir = manifest.as_path();
    loop {
        if let Ok(text) = std::fs::read_to_string(dir.join("Cargo.toml")) {
            if text.contains("[workspace]") {
                return dir.join("artifacts/native/prefix");
            }
        }
        match dir.parent() {
            Some(parent) => dir = parent,
            None => return manifest.join("artifacts/native/prefix"),
        }
    }
}

/// Whether `prefix` holds a shared `libavcodec`, which is what a link needs.
fn has_ffmpeg(prefix: &std::path::Path) -> bool {
    let lib = prefix.join("lib");
    let Ok(entries) = std::fs::read_dir(&lib) else {
        return false;
    };
    entries.flatten().any(|entry| {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        name == "libavcodec.so" || name.starts_with("libavcodec.so.")
    })
}
