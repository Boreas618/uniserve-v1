# media

Host media of the main process: H.264/AAC output encoding and MP4 muxing,
conditional media reading, and the host resamplers model conditioning uses.
The crate is the implementation of `specs/worker.md` §3's `media/` row and §9's
media replacements; it is called by `crates/executor`'s media runner on the
main process's blocking thread pool.

## Native runtime

The codec calls go through one C shim (`src/ffi/shim.c`) over FFmpeg 8.1.2's
headers. The crate builds against a **shared** FFmpeg 8.1.2 that has libx264:
the environment's installation at `/opt/uv/ffmpeg-8.1.2` is static and was
configured without external libraries, so it has no H.264 encoder. The
runtime the crate links is provisioned by `artifacts/native/build.sh` in this
worktree, which builds libx264 and then FFmpeg 8.1.2
(`--enable-shared --enable-libx264 --enable-version3`) from the environment's
FFmpeg source into `artifacts/native/prefix`. That directory is git-ignored;
rebuild it with:

```sh
artifacts/native/build.sh
```

The build script finds the runtime in `UNISERVE_FFMPEG_DIR` when set, and in
`<workspace>/artifacts/native/prefix` otherwise. Headers and shared libraries
must both be present. The static `/opt/uv/ffmpeg-8.1.2/bin/ffmpeg` and
`ffprobe` stay in use as external programs: the reference video decoder is an
external `ffmpeg` process by design, and tests validate MP4s with `ffprobe`.

## External programs

`MediaConfig::ffmpeg` names the `ffmpeg` executable used to decode reference
videos; `MediaConfig::from_env` takes it from `UNISERVE_FFMPEG` and otherwise
uses `ffmpeg` from the loader's search path.

## Tests

All tests are run by the caller (`specs/WORK.md`): `cargo test -p media` from
a worktree whose `artifacts/native/prefix` exists. The tests use the
environment's `/opt/uv/ffmpeg-8.1.2/bin/ffprobe` through `MediaConfig::ffmpeg`
or the default path, and the reference fixtures under `tests/fixtures/`.
