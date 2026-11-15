// SPDX-License-Identifier: Apache-2.0

// The entry of a routed, block-scaled NVFP4 product of trtllm-gen's batched-GEMM library: the
// operator's call, checked against the geometry the module was compiled for, handed to the
// library's runner.
//
// What the entry owns: the checks that turn a misplaced operand into a message (the compiled
// geometry, the operand dtypes, the scratch floor), the per-call tactic choice, and the device word
// the GEMM bounds its M by -- the padded route count. What it does not own: any arithmetic. The
// product is the library's, and everything the call states is either its geometry or an operand.
//
// The routed products (`gather_gemm_nvfp4`'s first matrix, `route_gemm_nvfp4`'s second) differ
// only in their runner, the rows the runner sizes its tactics by, what the call's counts must
// satisfy and the operands the runner takes. A package's entry unit includes its operator's header
// (its call structure and `uniserve_dtype_bf16`), then this one, states those differences as a
// `Product` (`using Runner`, `runner()`, `rows(call)`, `counts(call)`, `run(...)`) and defines its
// C entry as `uniserve::trtllm::routed::entry<Product>(call, stream)`.
//
// The build states the geometry as definitions (`uniserve_build_server.trtllm.geometry`).

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>

#include "flashinfer/trtllm/fused_moe/runner.h"

#include <uniserve_trtllm/cubins.h>

#ifndef UNISERVE_HIDDEN
#error "UNISERVE_HIDDEN is not defined; the build states the descriptor's geometry"
#endif
#ifndef UNISERVE_INTERMEDIATE
#error "UNISERVE_INTERMEDIATE is not defined"
#endif
#ifndef UNISERVE_EXPERTS
#error "UNISERVE_EXPERTS is not defined"
#endif
#ifndef UNISERVE_TOP
#error "UNISERVE_TOP is not defined"
#endif
#ifndef UNISERVE_TILE_TOKENS
#error "UNISERVE_TILE_TOKENS is not defined"
#endif
#ifndef UNISERVE_TILE_COLUMNS
#error "UNISERVE_TILE_COLUMNS is not defined"
#endif
#ifndef UNISERVE_CLUSTER_ROWS
#error "UNISERVE_CLUSTER_ROWS is not defined"
#endif
#ifndef UNISERVE_CLUSTER_COLUMNS
#error "UNISERVE_CLUSTER_COLUMNS is not defined"
#endif

namespace uniserve::trtllm::routed {
namespace {

namespace btg = batchedGemm::trtllm::gen;

constexpr int32_t kHidden = UNISERVE_HIDDEN;
constexpr int32_t kIntermediate = UNISERVE_INTERMEDIATE;
constexpr int32_t kExperts = UNISERVE_EXPERTS;
constexpr int32_t kTop = UNISERVE_TOP;
// The MMA tactic the module's cubins were selected for: the descriptor's tile and cluster.
constexpr int32_t kTileTokens = UNISERVE_TILE_TOKENS;
constexpr int32_t kTileColumns = UNISERVE_TILE_COLUMNS;
constexpr int32_t kClusterRows = UNISERVE_CLUSTER_ROWS;
constexpr int32_t kClusterColumns = UNISERVE_CLUSTER_COLUMNS;

// The weights' block scales: the module serves the library's variants whose *weight* scales are
// the plain `[experts, rows, columns/16]` table. The 8x4-tiled variants are a different shard
// layout and a different descriptor, and a module built for one may not read the other.
constexpr batchedGemm::gemm::MatrixLayout kWeightLayout = batchedGemm::gemm::MatrixLayout::MajorK;

// The scratch the entry owns ahead of the tactic's own, in bytes: the padded route count (one
// i32), with the tactic's workspace starting on a 16-byte boundary.
constexpr size_t kCounterBytes = 16;

// The problem a call states, as the runner takes it.
struct Problem {
  int32_t top;
  int32_t hidden;
  int32_t intermediate;
  int32_t experts;
  // The rows the runner sizes its tactics by: the product's own (`Product::rows`).
  int32_t rows;
};

template <typename Call>
int refuse(Call* call, char const* message) {
  std::snprintf(call->error, sizeof(call->error), "%s", message);
  return 1;
}

// Whether the call's geometry is the one the module was compiled for, and its counts are what the
// product needs. The tactic is compiled in: the tiles, the cluster, the operand types and the
// epilogue belong to the cubins the module published, so a call of another shape would be read by
// a kernel addressing rows and columns it does not have.
template <typename Product, typename Call>
bool compiled_for(Call const* call) {
  if (call->hidden_size != kHidden || call->intermediate != kIntermediate ||
      call->experts != kExperts) {
    return false;
  }
  if (call->top > kTop || call->top <= 0) {
    return false;
  }
  return Product::counts(call);
}

// The message for a refused geometry.
template <typename Call>
int refuse_geometry(Call* call) {
  char message[512];
  std::snprintf(message, sizeof(message),
                "this module is compiled for hidden %d, intermediate %d, experts %d, top up to %d "
                "and was handed hidden %lld, intermediate %lld, experts %lld, top %lld",
                kHidden, kIntermediate, kExperts, kTop, static_cast<long long>(call->hidden_size),
                static_cast<long long>(call->intermediate), static_cast<long long>(call->experts),
                static_cast<long long>(call->top));
  return refuse(call, message);
}

// The launch itself: every check the entry makes, then the library's product.
//
// It answers 0 or a refusal with `call->error` filled, and it may throw: the library's runner
// construction and its own `TLLM_CHECK`s raise `TllmException`, which `entry` turns into a message.
template <typename Product, typename Call>
int launch(Call* call, void* stream) {
  std::memset(call->error, 0, sizeof(call->error));
  if (call->dtype != uniserve_dtype_bf16) {
    return refuse(call, "the product's output is compiled as bf16 and this module is the bf16 one");
  }
  if (!compiled_for<Product>(call)) {
    return refuse_geometry(call);
  }
  // The tile and cluster the descriptor states are the ones the published cubins carry; a call of
  // another tactic belongs to another module, and this is the refusal that says so rather than a
  // launch into a shape mismatch.
  if (call->tile_m != kTileTokens || call->tile_n != kTileColumns ||
      call->cluster_m != kClusterRows || call->cluster_n != kClusterColumns) {
    char message[512];
    std::snprintf(message, sizeof(message),
                  "this module's cubins are tile %dx%d, cluster %dx%d, and the call states tile "
                  "%lldx%lld, cluster %lldx%lld",
                  kTileTokens, kTileColumns, kClusterRows, kClusterColumns,
                  static_cast<long long>(call->tile_m), static_cast<long long>(call->tile_n),
                  static_cast<long long>(call->cluster_m), static_cast<long long>(call->cluster_n));
    return refuse(call, message);
  }

  Problem const problem{
      static_cast<int32_t>(call->top),          static_cast<int32_t>(call->hidden_size),
      static_cast<int32_t>(call->intermediate), static_cast<int32_t>(call->experts),
      Product::rows(call),
  };

  auto& runner = Product::runner();
  int32_t const config = runner.getDefaultValidConfigIndex(
      problem.top, problem.hidden, problem.intermediate, problem.experts, problem.rows);
  if (config < 0) {
    return refuse(call, "no compiled tactic serves this problem's shape");
  }
  // The arming: the library's own requirement for the chosen tactic, plus the device word this
  // entry owns. A buffer shorter than that is refused with both numbers -- the caller sizes it
  // from the bucket's route count (`nn::functional::moe`), and a step that grows one is a host
  // allocation inside a captured graph. No bias table: the kernels index one per expert, the
  // operator has none, and a null pointer is what the runner takes for that.
  size_t const required = runner.getWorkspaceSizeInBytes(
      problem.top, problem.hidden, problem.intermediate, problem.experts, problem.rows, config);
  size_t const needed = kCounterBytes + required;
  if (static_cast<size_t>(call->workspace_bytes) < needed) {
    char message[512];
    std::snprintf(message, sizeof(message),
                  "the product's tactic needs %zu bytes of scratch and the step provided %lld: the "
                  "caller sizes this buffer, and the entry never grows one",
                  needed, static_cast<long long>(call->workspace_bytes));
    return refuse(call, message);
  }

  auto* seat = static_cast<unsigned char*>(call->workspace);
  auto* padded_routes = reinterpret_cast<int32_t*>(seat);
  void* workspace = static_cast<void*>(seat + kCounterBytes);
  cudaStream_t const order = reinterpret_cast<cudaStream_t>(stream);
  // The GEMM bounds its M by the padded route count, which is this call's own value: it is written
  // on the stream, before the product, so the ordering is the stream's.
  if (cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(padded_routes),
                       static_cast<unsigned int>(call->routes), 1,
                       reinterpret_cast<CUstream>(order)) != CUDA_SUCCESS) {
    return refuse(call, "the padded route count could not be written on the call's stream");
  }

  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return refuse(call, "the entry could not read the device the call runs on");
  }

  Product::run(runner, call, problem, padded_routes, workspace, device, order, config);
  return 0;
}

// The operator's entry: the launch, and the library's exceptions turned into messages.
//
// An exception raised inside the library's runner would otherwise cross the C boundary, where
// nothing may unwind: the process ends and the only message a log holds is the runtime's own
// (`internal error: entered unreachable code`), which names neither the operator nor the reason
// the library refused. A thrown `TllmException` states one, so it goes into the call's error buffer
// and the entry answers a refusal like every other one it makes.
template <typename Product, typename Call>
int entry(Call* call, void* stream) {
  if (call == nullptr) {
    return 1;
  }
  try {
    return launch<Product>(call, stream);
  } catch (std::exception const& thrown) {
    return refuse(call, thrown.what());
  } catch (...) {
    return refuse(call, "the library's launch failed without stating why");
  }
}

}  // namespace
}  // namespace uniserve::trtllm::routed
