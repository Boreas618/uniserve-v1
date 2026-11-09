// Where the FFmpeg runtime this workspace links lives, resolved once for every build script that
// needs it.
//
// `crates/media` compiles the C shim over FFmpeg's headers and links the `libav*` shared
// libraries; the package that produces an executable must also record where those libraries live,
// because Cargo passes a dependency's `rustc-link-search` and `rustc-link-lib` on to dependents
// but not its `rustc-link-arg`. Both build scripts therefore `include!` this file, so the rule has
// one statement rather than two copies.
//
// `UNISERVE_FFMPEG_DIR` names the installation's prefix; a build without it stops and says so.

use std::path::PathBuf;

/// The prefix of the FFmpeg installation a build links against.
fn ffmpeg_prefix() -> PathBuf {
    match std::env::var_os("UNISERVE_FFMPEG_DIR") {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => panic!(
            "UNISERVE_FFMPEG_DIR names the FFmpeg prefix this workspace links against; \
             crates/media/native/build.sh builds one"
        ),
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
