//! The FFmpeg runtime paths this package's test executables carry.
//!
//! The doubles of the engine's scheduler link the engine's library, which links the media shim
//! through its workers. Cargo passes `crates/media`'s search path and library names on to every
//! dependent, but not its `rustc-link-arg`: an executable has a run path only when its own
//! package states one. The linker resolves what `libavcodec` itself needs through `-rpath-link`,
//! and the runtime loader finds `libavcodec` through the executable's run path, so both are
//! stated here as the engine's package states them for its executable.
//!
//! `crates/media/build_support/ffmpeg_prefix.rs` holds which installation this is, once, for
//! this script, the root package's, the engine's, `scheduler`'s and `crates/media`'s.

include!("../media/build_support/ffmpeg_prefix.rs");

fn main() {
    println!("cargo:rerun-if-env-changed=UNISERVE_FFMPEG_DIR");
    println!("cargo:rerun-if-changed=../media/build_support/ffmpeg_prefix.rs");

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
