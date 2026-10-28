# NIXL transfers

UniServe loads NIXL's C API at runtime for RDMA transfers between Storage Servers. The bindings target NIXL 1.5.1 with the UCX backend. Read-only host storage requires both patches in `crates/nixl/patches/`: NIXL passes the requested read access to UCX, and UCX registers the mapping for remote reads without requesting write access. The patches apply to the revisions below.

The build requires Git, a C/C++ compiler, a CUDA toolkit, Autoconf, Automake, Libtool, pkg-config, Meson, Ninja, patchelf, and the development headers and libraries for verbs, RDMA CM and GDRCopy. Put the CUDA toolkit's `bin` directory on `PATH`. Run these commands from the repository root; sources, build outputs and installed libraries stay under `artifacts/native`.

```bash
export MAX_JOBS=8
native_root="$PWD/artifacts/native"
patch_root="$PWD/crates/nixl/patches"
mkdir -p "$native_root/src"

git clone https://github.com/openucx/ucx.git "$native_root/src/ucx"
git -C "$native_root/src/ucx" checkout --detach 65e342a82dd2609a28c8c2b8c45ed7fe2cb33fa0
git -C "$native_root/src/ucx" apply "$patch_root/ucx-read-only.patch"
(
    cd "$native_root/src/ucx"
    ./autogen.sh
    mkdir build
    cd build
    ../contrib/configure-release --prefix="$native_root/ucx" \
        --enable-mt --enable-shared --disable-static --disable-doxygen-doc \
        --enable-optimizations --without-avx --enable-cma --enable-devel-headers \
        --with-cuda --with-verbs --with-dm --with-gdrcopy=/usr
    make -j "$MAX_JOBS"
    make install
)

git clone --branch v1.5.1 https://github.com/ai-dynamo/nixl.git "$native_root/src/nixl"
git -C "$native_root/src/nixl" checkout --detach b5d3c3e7f6c7bf08896df02f65cd96596f0a6bf1
git -C "$native_root/src/nixl" apply "$patch_root/nixl-read-only.patch"
meson setup "$native_root/src/nixl/build" "$native_root/src/nixl" \
    --prefix="$native_root/nixl" --libdir=lib --buildtype=release \
    -Ducx_path="$native_root/ucx" -Drust=true -Denable_plugins=UCX \
    -Dbuild_tests=false -Dbuild_examples=false -Dforce_fallback_for=abseil-cpp
ninja -C "$native_root/src/nixl/build" -j "$MAX_JOBS"
ninja -C "$native_root/src/nixl/build" install
patchelf --set-rpath "$native_root/nixl/lib:$native_root/ucx/lib" \
    "$native_root/nixl/lib/libnixl_capi.so"
```

Set `LD_LIBRARY_PATH` to include `artifacts/native/nixl/lib` before starting serving. A separately launched Storage Server can select the library explicitly with `uniserve storage serve --nixl-library /absolute/path/to/libnixl_capi.so`. NIXL finds its UCX plugin beside `libnixl.so`; `UCX_NET_DEVICES` can select the host's RDMA ports.

The NIXL backend uses pinned registration and disables its registration cache (`REG_NONBLOCK_MEM_TYPES=` and `RCACHE_ENABLE=n`). A released CUDA mapping can be reused at the same virtual address, so registrations must follow the backing allocation's lifetime. On hosts in the same NVLink fabric, the Transfer Server uses imported CUDA fabric handles when available and RDMA otherwise.
