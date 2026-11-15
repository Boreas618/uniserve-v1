# media

Host media of the main process: H.264/AAC output encoding and MP4 muxing, conditional media reading (images, audio tracks and reference videos), and the host resamplers model conditioning uses. The crate is called by the engine's media runner (`crates/engine/src/runner/media.rs`) on the main process's blocking thread pool, and its probe by the server's condition checks.

## Native dependency

Every codec, demuxer and filter call goes through one C shim (`src/ffi/shim.c`) over the headers of the FFmpeg 8.1 libraries: `libavformat`, `libavfilter`, `libavcodec`, `libswscale`, `libswresample` and `libavutil`, linked as shared libraries, with `libavcodec` built with libx264 (the output video encoder). The build script (`build.rs`) requires exactly the library versions of FFmpeg 8.1 and finds them in one of two ways:

- **pkg-config** (the default): the libraries of a distribution package, or of an installation whose `lib/pkgconfig` the host registers with pkg-config. A host that keeps FFmpeg 8.1 under its own prefix adds that prefix's `lib/pkgconfig` to `PKG_CONFIG_PATH`, the standard pkg-config search path, for the build.
- **the `vendored` feature**: the build fetches the pinned FFmpeg 8.1.2 release and libx264 (each archive verified by its SHA-256), configures and builds them as shared libraries into its output directory, and finds that installation through its own pkg-config files (the `vendored` module of `build.rs`). It needs `curl`, `tar` with `xz`, `make` and a C compiler on the build host, and `nasm` on x86-64; it builds for the host only, and its parallelism is the jobs Cargo grants the build. Each FFmpeg library is linked with `$ORIGIN` as its run path, so the installation resolves its own libraries wherever it is. Select it with `--features media/vendored`.

The package declares `links = "ffmpeg"`. Its build states the directories of the libraries, when they are outside the default loader path, as `DEP_FFMPEG_LIB` to the build scripts of the packages that depend on it directly, and the version of `libavcodec` as `DEP_FFMPEG_VERSION`. Every package that produces an executable linking this crate (the workspace's `uniserve`, `crates/engine`, and the test executables of `crates/daemon` and `crates/testkit`) records `DEP_FFMPEG_LIB` in its executables' run path from its own build script, because Cargo passes a dependency's search path and library names on to dependents but not its link arguments. An executable therefore runs with no loader environment. A run path into a build's output directory (the `vendored` installation) lasts as long as that directory; a deployment host provides FFmpeg 8.1 where the loader finds it.

`codec::require_codecs` opens both output encoders with their declared options, so a runtime that lacks libx264 or the AAC encoder fails at start, by name.

## Encoders

The output encoders are declared once, in `codec::VIDEO_ENCODER` and `codec::AUDIO_ENCODER`: libx264 at `preset=ultrafast` and `tune=zerolatency` with eight slice threads under its own rate control, fed yuv420p through swscale's bilinear filter, and FFmpeg's AAC encoder with `aac_coder=fast` at 144 kb/s. The shim receives them with every call and holds no encoder name or option of its own.

## Reference videos

`read::read_video` decodes a reference video's clip through the linked libraries: its first video stream that is not cover art, put upright by its display matrix, resampled to the model's frame rate with FFmpeg's `fps` filter, its first `start_frame` resampled frames skipped, and every kept frame scaled with LANCZOS to the clip's canvas and converted to RGB24. These are the frames FFmpeg's command-line tool produces from the file with `-vf fps=R,trim=start_frame=S,scale=W:H:flags=lanczos,setsar=1 -pix_fmt rgb24`. The file is read through its descriptor without moving its offset, and a demuxer's request to open another file or URL is refused for every input the crate reads.

## Tests

`cargo test -p media` needs nothing beyond the build: the encoder, muxing and MP4 output tests read their output back through the crate's own probe and decoders (`tests/codec.rs`), the readers' and resamplers' refusals use media the tests make, and the comparisons with the libraries whose contracts the readers and resamplers implement read checked-in fixtures (`tests/read.rs`, `tests/resample.rs`).

The fixtures under `tests/fixtures/read/` and `tests/fixtures/resample/` were written by the `generate.py` beside them, scripts whose inline metadata pins the reference libraries (Pillow, PyAV, torch, torchvision and torchaudio). The reading fixtures' reference videos are encoded, and their expected frames decoded, by the `ffmpeg` program of FFmpeg 8.1 with libx264 on the search path. Regenerate a set with `uv run generate.py .` from its directory and commit the result.
