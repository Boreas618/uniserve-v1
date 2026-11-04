#!/bin/bash
# Builds the shared FFmpeg 8.1.2 + libx264 runtime the media crate's shim
# links (see README.md). The environment's FFmpeg installation at
# /opt/uv/ffmpeg-8.1.2 is static and was configured --disable-autodetect, so
# it carries no H.264 encoder; the crate needs a shared build of the same
# FFmpeg version with libx264.
#
# Usage:
#   crates/media/native/build.sh [PREFIX]
#
# PREFIX defaults to <workspace>/artifacts/native/prefix, which is where
# crates/media/build.rs looks when UNISERVE_FFMPEG_DIR is unset. The build is
# idempotent: an existing PREFIX with both libraries is left alone.
#
# Requirements: gcc, make, pkg-config, zlib and the FFmpeg 8.1.2 source at
# /opt/uv/ffmpeg-8.1.2/src/ffmpeg-8.1.2 (the environment provides it). The
# x264 tarball is pinned by its SHA-256; if upstream's master archive has
# moved, download the archive, verify it is a working x264, and update
# X264_SHA256 below.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PREFIX=${1:-${PREFIX:-$ROOT/artifacts/native/prefix}}
JOBS=${JOBS:-$(nproc)}
FFMPEG_SOURCE=${FFMPEG_SOURCE:-/opt/uv/ffmpeg-8.1.2/src/ffmpeg-8.1.2}

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

echo "media native runtime: $PREFIX"
