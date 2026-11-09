# media

Host media of the main process: H.264/AAC output encoding and MP4 muxing, conditional media reading, and the host resamplers model conditioning uses. The crate is called by `crates/worker`'s media runner on the main process's blocking thread pool.

## Native runtime

The codec calls go through one C shim (`src/ffi/shim.c`) over FFmpeg 8.1.2's headers. The crate builds against a **shared** FFmpeg 8.1.2 that has libx264: a static FFmpeg configured without external libraries has no H.264 encoder. The runtime the crate links is provisioned by the committed script `crates/media/native/build.sh`, which builds libx264 (pinned by SHA-256) and then FFmpeg 8.1.2 (`--enable-shared --enable-rpath --enable-libx264 --enable-version3`) from the FFmpeg source `FFMPEG_SOURCE` names into the prefix its argument names. The script then sets every shared library's run path to `$ORIGIN`, so the prefix is relocatable: `bin/ffmpeg` and `bin/ffprobe` run without `LD_LIBRARY_PATH`, a binary that names the prefix's `lib` in its own run path resolves `libx264` beside `libavcodec`, and a prefix built in one worktree and provisioned elsewhere still works. Build one with:

```sh
FFMPEG_SOURCE=<FFmpeg 8.1.2 source> crates/media/native/build.sh <prefix>
```

The build script finds the runtime in `UNISERVE_FFMPEG_DIR` and stops when it is unset; `crates/media/build_support/ffmpeg_prefix.rs` holds that rule once, for this crate's build script and for the workspace's (which records the prefix's `lib` in the executable's run path, since Cargo passes a dependency's link search path and library names on but not its link arguments). Headers and shared libraries must both be present. An `ffmpeg` and an `ffprobe` stay in use as external programs: the reference video decoder is an external `ffmpeg` process by design, and tests validate MP4s with `ffprobe`.

## External programs

`MediaConfig::ffmpeg` names the `ffmpeg` executable used to decode reference videos. `MediaConfig::from_env` takes it from `UNISERVE_FFMPEG`, else uses the linked runtime's own `bin/ffmpeg`, else `ffmpeg` from the loader's search path. Either binary must be able to decode H.264 and AAC; the runtime prefix's own `ffmpeg` is the natural choice and needs no environment beyond its baked rpath.

## Tests

`cargo test -p media` builds with `UNISERVE_FFMPEG_DIR` naming the runtime prefix, like every build of the crate, and runs every test that needs nothing more: the encoder, muxing and MP4 contract tests and the readers' refusals, which use the linked runtime's `bin/ffmpeg` and `bin/ffprobe` (`UNISERVE_FFMPEG` and `UNISERVE_FFPROBE` select others) and media the tests generate themselves.

The comparisons with the reference implementations are ignored by default because they need `UNISERVE_REFERENCE_PYTHON` to name an interpreter with numpy, Pillow, torch, torchvision and torchaudio. The `generate.py` scripts under `tests/fixtures/` produce their fixtures in each test process. Run them with `UNISERVE_REFERENCE_PYTHON=<python> cargo test -p media -- --ignored`; without the variable they fail rather than pass.
