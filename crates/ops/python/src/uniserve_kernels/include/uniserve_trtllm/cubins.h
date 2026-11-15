// SPDX-License-Identifier: Apache-2.0

// The cubin loader of a module that launches trtllm-gen kernels, and its load entry.
//
// trtllm-gen's launch code loads a kernel's cubin through
// `flashinfer::trtllm_cubin_loader::getCubin(name, sha256)`, which the module defines. `name` is
// the cubin's file name prefixed by the directory the library's cubin-path definition states
// (`TLLM_GEN_FMHA_CUBIN_PATH`, `TLLM_GEN_GEMM_CUBIN_PATH`): the module's `cubins` directory, where
// the build published the cubins its descriptor can select (`uniserve_build_server.trtllm`).
//
// Where the module's files are is the loader's to state, not the module's to guess: `ops` calls the
// module's load entry, `uniserve_trtllm_load` (the `load` entry of the package's manifest), once
// it has opened the library and before any launch, with the directory the module's files were
// published in, and every cubin is read from there.
//
// `sha256` is the hash the library's table states for the cubin. The build publishes a cubin only
// after verifying it against that hash (`uniserve_build_server.trtllm.publish`), and the module
// store publishes a module's files together and unchanged, so the loader reads the file as the
// build published it.
//
// The header defines the loader and the load entry: exactly one translation unit of a module, its
// entry unit, includes it.

#pragma once

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>

// The load entry's call, where the module's files are: `uniserve_load_call`, generated from the
// operator crate.
#include <uniserve_load.h>

namespace uniserve::trtllm {
namespace {

// The directory of the module's files, as the load entry was handed it. Every device of the process
// installs the module, each from its own thread, and each hands the same directory; a kernel's
// cubin is read at its first launch on a device, from that device's thread.
//
// It has internal linkage: every module that includes this header holds its own. A static of an
// inline function would be one object for the whole process instead, which the compiler binds
// uniquely (`STB_GNU_UNIQUE`) whatever library opened it, and every trtllm-gen module would then
// read its cubins from the directory the last one was handed.
struct Directory {
  std::mutex mutex;
  std::string path;
};

Directory& directory() {
  static Directory stated;
  return stated;
}

}  // namespace
}  // namespace uniserve::trtllm

namespace flashinfer::trtllm_cubin_loader {

// The cubin `name` names, read from the module's directory. The table's hash is the build's to
// verify, which it did before it published the file (see above).
std::string getCubin(const std::string& name, const std::string& /* sha256 */) {
  std::string path;
  {
    auto& stated = uniserve::trtllm::directory();
    std::lock_guard<std::mutex> held(stated.mutex);
    path = stated.path;
  }
  if (path.empty()) {
    throw std::runtime_error(
        "the module was handed no directory: its loader calls uniserve_trtllm_load before a "
        "launch");
  }
  const std::string cubin = path + "/" + name;
  std::ifstream file(cubin, std::ios::binary);
  if (!file) {
    throw std::runtime_error("the module has no cubin " + cubin);
  }
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

}  // namespace flashinfer::trtllm_cubin_loader

// The load entry: records the directory of the module's files. It issues no device work, so
// `stream` is unused; it returns 0, or 1 with a message in `call->error`.
extern "C" int32_t uniserve_trtllm_load(uniserve_load_call* call, void* /* stream */) {
  if (call == nullptr) {
    return 1;
  }
  if (call->directory == nullptr || call->directory[0] == '\0') {
    std::snprintf(call->error, sizeof(call->error), "%s",
                  "uniserve_trtllm_load needs the directory of the module's files");
    return 1;
  }
  auto& stated = uniserve::trtllm::directory();
  std::lock_guard<std::mutex> held(stated.mutex);
  stated.path = call->directory;
  return 0;
}
