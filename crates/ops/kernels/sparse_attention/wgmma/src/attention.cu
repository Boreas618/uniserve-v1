// SPDX-License-Identifier: Apache-2.0

// The sparse_attention module's sm90 backend: the operator's call turned into the VSA
// block-sparse SM90a forward pass.
//
// The kernel is the warpgroup-pipelined SM90a block-sparse attention carried verbatim in
// this package's sources: one CTA per (head, 128-row query tile), a TMA producer
// warpgroup and two consumer warpgroups, GMMA products with register-sourced A operands,
// an online softmax in FP32 registers, and the sparse block map read on the device.
//
// This unit owns the operator's side of the trade: the geometry checks the kernel's TMA
// copies need (16-byte aligned bases, strides whole in 16-byte units, contiguous
// 128-element channels), the 32-bit bounds of its metadata addressing, and the two things
// this backend does not do — write the softmax normalizer, and read a per-tile query
// valid-row count, which the operator's call does not carry and the kernel takes as null.
//
// The kernel's tile is fixed at 128 rows in both directions, so this backend serves the
// descriptor's block 128 and only that.

#include "block_sparse_sm90a.cuh"

#include "uniserve_sparse_attention.h"

#include <cmath>
#include <cstdio>
#include <limits>

namespace {

//: The kernel's own block and head geometry; a call with others is a compilation this
//: module lacks.
constexpr int64_t kBlock = uniserve::vsa_sm90::kBlockM;
constexpr int64_t kHeadDim = uniserve::vsa_sm90::kHeadDim;

// The message an entry writes before it returns a nonzero status.
int32_t refuse(const uniserve_sparse_attention_call* call, const char* message) {
  std::snprintf(const_cast<uniserve_sparse_attention_call*>(call)->error, sizeof(call->error), "%s",
                message);
  return 1;
}

// A TMA tensor map needs 16-byte aligned bases and strides that are whole 16-byte units in
// bytes; for bf16 that is 8 elements.
bool aligned(const void* base, int64_t row_stride, int64_t head_stride) {
  return base != nullptr && reinterpret_cast<uintptr_t>(base) % 16 == 0 && row_stride > 0 &&
         head_stride > 0 && row_stride % 8 == 0 && head_stride % 8 == 0 &&
         row_stride <= std::numeric_limits<int32_t>::max() &&
         head_stride <= std::numeric_limits<int32_t>::max();
}

}  // namespace

extern "C" int32_t uniserve_sparse_attention(const uniserve_sparse_attention_call* call,
                                             void* stream) {
  if (call == nullptr) {
    return 1;
  }
  if (call->dtype != UNISERVE_SPARSE_BF16 || call->output_dtype != UNISERVE_SPARSE_BF16) {
    return refuse(call, "the sm90 module serves bf16 queries, keys, values and output");
  }
  if (call->head_dim != kHeadDim) {
    return refuse(call, "the sm90 module's kernel is compiled for 128 elements per head");
  }
  if (call->lse != nullptr) {
    return refuse(call, "the sm90 module's kernel writes no softmax normalizer");
  }
  if (call->heads <= 0 || call->query_rows <= 0 || call->key_rows <= 0) {
    return refuse(call, "sparse_attention needs at least one head and one query tile and key block");
  }
  if (call->query_rows % kBlock != 0 || call->key_rows % kBlock != 0) {
    return refuse(call,
                  "query and key rows must be whole 128-row tiles; a partial tile is the "
                  "caller's padding, addressed by the valid-size array");
  }
  if (call->selected <= 0) {
    return refuse(call, "sparse_attention needs a positive selection width");
  }
  if (!(call->scale > 0.0f) || !std::isfinite(call->scale)) {
    return refuse(call, "sparse_attention needs a positive, finite softmax scale");
  }
  if (!aligned(call->query, call->q_row_stride, call->q_head_stride) ||
      !aligned(call->out, call->o_row_stride, call->o_head_stride) ||
      !aligned(call->key, call->k_row_stride, call->k_head_stride) ||
      !aligned(call->value, call->v_row_stride, call->v_head_stride)) {
    return refuse(call,
                  "query, key, value and out need 16-byte aligned bases and row and head strides "
                  "that are multiples of 8 elements");
  }
  if (call->indices == nullptr || call->counts == nullptr || call->valid == nullptr) {
    return refuse(call, "sparse_attention reads the selection from the device; it needs indices, "
                        "counts and the valid-size array");
  }
  // The metadata arrays are read by the kernel only, never on the host, so this module
  // checks their presence and nothing of their contents.
  const int64_t query_tiles = call->query_rows / kBlock;
  if (call->heads > std::numeric_limits<int32_t>::max() ||
      call->query_rows > std::numeric_limits<int32_t>::max() ||
      call->key_rows > std::numeric_limits<int32_t>::max() ||
      call->selected > std::numeric_limits<int32_t>::max() ||
      call->heads * query_tiles > std::numeric_limits<int32_t>::max()) {
    return refuse(call, "the sm90 kernel's geometry and grid are 32-bit signed; this call does not "
                        "fit");
  }
  if (query_tiles * call->selected > std::numeric_limits<int32_t>::max()) {
    return refuse(call, "the sm90 kernel's metadata addressing is 32-bit signed; this call does "
                        "not fit");
  }

  const uniserve::vsa_sm90::Arguments arguments{
      static_cast<const uniserve::vsa_sm90::Element*>(call->query),
      static_cast<const uniserve::vsa_sm90::Element*>(call->key),
      static_cast<const uniserve::vsa_sm90::Element*>(call->value),
      static_cast<uniserve::vsa_sm90::Element*>(call->out),
      call->q_row_stride,
      call->q_head_stride,
      call->k_row_stride,
      call->k_head_stride,
      call->v_row_stride,
      call->v_head_stride,
      call->o_row_stride,
      call->o_head_stride,
      static_cast<int32_t>(call->query_rows),
      static_cast<int32_t>(call->key_rows),
      static_cast<int32_t>(call->heads),
      static_cast<const int32_t*>(call->indices),
      static_cast<const int32_t*>(call->counts),
      static_cast<const int32_t*>(call->valid),
      static_cast<int32_t>(call->selected),
      // The operator's call carries no per-tile query validity: every query row is
      // computed, as the kernel does when the array is absent.
      nullptr,
      call->scale,
  };
  const cudaError_t status =
      uniserve::vsa_sm90::launch(arguments, reinterpret_cast<cudaStream_t>(stream));
  if (status != cudaSuccess) {
    std::snprintf(const_cast<uniserve_sparse_attention_call*>(call)->error, sizeof(call->error),
                  "the SM90 sparse attention kernel refused this launch: %s",
                  cudaGetErrorString(status));
    return 1;
  }
  return 0;
}
