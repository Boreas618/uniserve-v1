#!/bin/bash
# Builds the shared FFmpeg 8.1.2 + libx264 runtime the media crate's shim
# links (see README.md). A static FFmpeg configured --disable-autodetect
# carries no H.264 encoder; the crate needs a shared build of FFmpeg 8.1.2
# with libx264.
#
# Usage:
#   FFMPEG_SOURCE=<FFmpeg 8.1.2 source> crates/media/native/build.sh <PREFIX>
#
# PREFIX is the directory the runtime is installed into, which a build then
# names with UNISERVE_FFMPEG_DIR. The build is idempotent: an existing PREFIX
# with both libraries is left alone.
#
# Requirements: gcc, make, pkg-config, zlib and the FFmpeg 8.1.2 source tree
# FFMPEG_SOURCE names. The x264 tarball is pinned by its SHA-256; if upstream's master archive has
# moved, download the archive, verify it is a working x264, and update
# X264_SHA256 below.
set -euo pipefail

PREFIX=${1:?usage: build.sh <PREFIX>, the directory the runtime is installed into}
JOBS=${JOBS:-$(nproc)}
FFMPEG_SOURCE=${FFMPEG_SOURCE:?names the FFmpeg 8.1.2 source tree}

# The x264 master archive the provisioned runtime was built from, x264 build
# 165 (point version 0.165.x).
X264_URL=https://code.videolan.org/videolan/x264/-/archive/master/x264-master.tar.gz
X264_SHA256=0cfddd004b09be7bfcfa2d68e0b8cf1afad011bc531325915cf56d92848b789b

if [ ! -f "$PREFIX/lib/libavcodec.so" ] || [ ! -f "$PREFIX/lib/libx264.so" ]; then
  build=$(mktemp -d)
  trap 'rm -rf "$build"' EXIT
  mkdir -p "$PREFIX"

  if [ ! -f "$PREFIX/lib/libx264.so" ]; then
    echo "building x264 into $PREFIX"
    curl -L --fail -o "$build/x264.tar.gz" "$X264_URL"
    echo "$X264_SHA256  $build/x264.tar.gz" | sha256sum -c -
    tar xzf "$build/x264.tar.gz" -C "$build"
    (
      cd "$build/x264-master"
      ./configure --prefix="$PREFIX" --enable-shared --enable-pic --disable-cli
      make -j"$JOBS"
      make install
    )
  fi

  if [ ! -f "$PREFIX/lib/libavcodec.so" ]; then
    echo "building FFmpeg 8.1.2 with libx264 into $PREFIX"
    if [ ! -d "$FFMPEG_SOURCE" ]; then
      echo "FFmpeg 8.1.2 source not found at $FFMPEG_SOURCE" >&2
      exit 1
    fi
    cp -a "$FFMPEG_SOURCE" "$build/ffmpeg"
    (
      cd "$build/ffmpeg"
      PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig" ./configure \
        --prefix="$PREFIX" \
        --enable-shared --disable-static \
        --enable-gpl --enable-libx264 --enable-version3 \
        --disable-doc --disable-debug --disable-ffplay --disable-autodetect \
        --enable-zlib
      make -j"$JOBS"
      make install
    )
  fi
fi

# Every shared library records `$ORIGIN` as its run path rather than the prefix it was built in.
# FFmpeg's `--enable-rpath` writes the prefix as an absolute path into each library, so a prefix
# built in one worktree and provisioned elsewhere leaves `libavcodec` looking for `libx264` in a
# directory that does not exist - and the loader resolves a library's own dependency with the
# requiring library's run path, so no path on the executable can make up for it. `$ORIGIN` is the
# library's own directory, which makes the prefix relocatable: a binary that names its `lib` runs
# with no `LD_LIBRARY_PATH`. This runs for an existing prefix too, so a provisioned copy is
# repaired rather than rebuilt.
if ! command -v patchelf > /dev/null; then
  echo "patchelf is needed to make $PREFIX relocatable: without it every library keeps the path it" >&2
  echo "was built in, and a copy of the prefix cannot resolve its own dependencies" >&2
  exit 1
fi
for library in "$PREFIX"/lib/libav*.so.* "$PREFIX"/lib/libsw*.so.* "$PREFIX"/lib/libx264.so.*; do
  if [ -f "$library" ]; then
    patchelf --set-rpath '$ORIGIN' "$library"
  fi
done
# `ffmpeg` and `ffprobe` are the second kind of binary here: the media stage spawns `ffmpeg` at
# the encode, so it must resolve its own libraries too. `--enable-rpath` writes the build
# directory into them, and a library's own dependencies are resolved with the requiring
# library's run path, so a provisioned copy starts only once each of these names `$ORIGIN/../lib`.
for binary in "$PREFIX"/bin/ffmpeg "$PREFIX"/bin/ffprobe; do
  if [ -f "$binary" ]; then
    patchelf --set-rpath '$ORIGIN/../lib' "$binary"
  fi
done

echo "media native runtime: $PREFIX"
