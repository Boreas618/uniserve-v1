//! The FFmpeg runtime paths this workspace's executables carry.
//!
//! `crates/media` states the FFmpeg recipe - the headers, the C shim, `-L` and `-l` - and Cargo
//! passes the search path and the library names on to every dependent. It does **not** pass a
//! dependency's `rustc-link-arg`, so an executable that links the shim has no run path of its own
//! unless its own package states one.
//!
//! That matters at the link and at the run:
//!
//! - The linker resolves the libraries `libavcodec` itself needs (`libx264`, `libswresample`,
//!   `libavutil`) only through `-rpath-link` or `-rpath` on its own command line, so without them a
//!   clean host fails with `libx264.so.165, needed by libavcodec.so, not found`.
//! - The runtime loader resolves the same names with the *requiring* library's run path, so the
//!   executable's run path is what locates `libavcodec` itself.
//!
//! `crates/media/build_support/ffmpeg_prefix.rs` holds which installation this is, once, for this script and
//! `crates/media`'s.

include!("crates/media/build_support/ffmpeg_prefix.rs");

fn main() {
    println!("cargo:rerun-if-env-changed=UNISERVE_FFMPEG_DIR");
    println!("cargo:rerun-if-changed=crates/media/build_support/ffmpeg_prefix.rs");

    let prefix = ffmpeg_prefix();
    if !has_ffmpeg(&prefix) {
        // A tree built without a provisioned runtime links no shim and needs no run path.
        // `crates/media`'s build script names the missing prefix when the media crate is built.
        return;
    }
    let lib = prefix.join("lib");
    println!("cargo:rustc-link-arg=-Wl,-rpath-link,{}", lib.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
}
