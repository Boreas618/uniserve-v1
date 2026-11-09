// The FFmpeg run path an executable carries, stated once for every package whose own targets link
// the media shim.
//
// `crates/media` states the FFmpeg recipe - the headers, the C shim, `-L` and `-l` - and Cargo
// passes the search path and the library names on to every dependent. It does **not** pass a
// dependency's `rustc-link-arg`, so an executable that links the shim has no run path of its own
// unless its own package states one. The workspace root states it for `uniserve`, and `worker` for
// its unit tests, which call the media crate.
//
// That matters at the link and at the run:
//
// - The linker resolves the libraries `libavcodec` itself needs (`libx264`, `libswresample`,
//   `libavutil`) only through `-rpath-link` or `-rpath` on its own command line, so without them a
//   clean host fails with `libx264.so.165, needed by libavcodec.so, not found`.
// - The runtime loader resolves the same names with the *requiring* library's run path, so the
//   executable's run path is what locates `libavcodec` itself. Without it a test binary stops with
//   `libavformat.so.62: cannot open shared object file`.
//
// A build script includes `ffmpeg_prefix.rs` and then this file, and calls `state_run_path` with the
// path of this directory relative to its package.

/// Gives this package's executables the run path of the FFmpeg installation the workspace links.
fn state_run_path(support: &str) {
    println!("cargo:rerun-if-env-changed=UNISERVE_FFMPEG_DIR");
    println!("cargo:rerun-if-changed={support}/ffmpeg_prefix.rs");
    println!("cargo:rerun-if-changed={support}/run_path.rs");

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
