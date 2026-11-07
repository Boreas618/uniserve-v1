// SPDX-License-Identifier: Apache-2.0
//
// The C header of the expert_exchange operator's packages: the operator's one
// call structure (`crates/ops/src/moe/exchange.rs`, CALL_LAYOUT), the transport's
// lifecycle structures, the group vtable they are handed, and the entries a
// package exports.
//
// Every backend of the operator includes this header unchanged, so the executor
// drives any backend through the same seven entries.

#pragma once

#include <cstddef>
#include <cstdint>

// The directions of one launch, as the operator numbers them.
enum uniserve_exchange_mode : int64_t {
  UNISERVE_EXCHANGE_DISPATCH = 0,
  UNISERVE_EXCHANGE_COMBINE = 1,
  UNISERVE_EXCHANGE_FUSED = 2,
};

// What a fused launch reads of one layer: its experts, as the loader laid them out.
// Each weight is `[local experts, rows, columns / 2]` packed values, read K-major,
// with its block scales in a table of `scale_bytes` per expert.
struct uniserve_fused_layer {
  const void* up_gate;        // [local, 2 * intermediate, hidden / 2]
  const void* up_gate_scales;
  const void* down;           // [local, hidden, intermediate / 2]
  const void* down_scales;
  const float* up_gate_alpha;  // [local] dequantizes the first product
  const float* down_alpha;     // [local] dequantizes the second product
  const float* gated_scale;    // [local] encodes the gated rows
  int64_t up_gate_scale_bytes;
  int64_t down_scale_bytes;
};

// One launch of the exchange.
//
// Dispatch sends this rank's rows and leaves what the rank received in the
// transport's views. Combine returns the rows the rank's experts computed, which
// it wrote into the transport's `combined` view, and sums each token's partial
// results into `out`.
struct uniserve_exchange_call {
  void* transport;
  const void* states;      // dispatch: [tokens, hidden] rows to send
  const int32_t* ids;      // dispatch: [tokens, top] expert ids
  const float* weights;    // dispatch: [tokens, top] route weights
  const int32_t* live;     // [1] device count of live rows, or null for all `tokens`
  void* out;               // combine: [tokens, hidden] summed rows, written
  const uniserve_fused_layer* layer;  // fused: this layer's expert operands
  int64_t mode;            // uniserve_exchange_mode
  int64_t tokens;          // rows of this launch's extent, the captured bucket's
  int64_t top;
  int64_t hidden;
  int64_t experts;
  int64_t max_tokens;
  int64_t ranks;
  int64_t dtype;
  char error[512];
};

static_assert(offsetof(uniserve_exchange_call, error) == 120,
              "the call's error must sit where the operator's layout puts it");
static_assert(sizeof(uniserve_exchange_call) == 632,
              "the call must be the size of the operator's layout");

// The executor's host channel to the group, as a package sees it. One instance
// per group: the executor creates it before the first transport and destroys it
// after the last one is released. A package may cache neither the pointer nor a
// copy of what it points at, and must never free it.
//
// Every function is collective over the group and is called from a lifecycle
// entry only, never from a launch. The ranks of a host are threads of one
// process, so a function blocks its caller's thread and no other's.
struct UniserveGroup {
  // What the functions below take as their first argument.
  void* handle;
  // The group's NCCL communicator (`ncclComm_t`), for a package that drives NCCL
  // itself, or null when the deployment opened none.
  void* communicator;
  int64_t ranks;   // ranks of the group
  int64_t rank;    // this rank's index in it
  int64_t device;  // the CUDA device this rank's transport lives on

  // Gathers each rank's `bytes` bytes of `mine` into `peers`, `[ranks][bytes]`
  // in rank order. Peer handles and addresses travel through this.
  int32_t (*all_gather)(void* handle, const void* mine, void* peers, int64_t bytes);

  // Returns once every rank of the group has called it.
  int32_t (*barrier)(void* handle);

  // Why the last call failed; empty after a successful one.
  char error[512];
};

static_assert(offsetof(UniserveGroup, error) == 56,
              "the group vtable's error message must sit where the operator's mirror puts it");
static_assert(sizeof(UniserveGroup) == 568,
              "the group vtable must be the size of the operator's mirror");

// The transport's lifecycle, as the executor drives it.
struct uniserve_exchange_lifecycle {
  void* transport;
  // The executor's channel to the group, alive for the whole package life.
  const UniserveGroup* group;
  int64_t ranks;
  int64_t rank;
  int64_t experts;       // experts in the router's vocabulary
  int64_t hidden;
  int64_t max_tokens;    // rows one rank sends in one step, at most
  int64_t top;
  int64_t dtype;
  int64_t device;
  int64_t intermediate;  // an expert's gated width, for a backend that runs the experts
  int64_t capacity;      // step entry: the rows per rank the group agreed for this step
  int64_t attention_ranks;  // leading ranks that hold no experts; zero when every rank holds some
  char error[512];
};

static_assert(offsetof(uniserve_exchange_lifecycle, error) == 104,
              "the lifecycle's error message must sit where the operator's mirror puts it");
static_assert(sizeof(uniserve_exchange_lifecycle) == 616,
              "the lifecycle must be the size of the operator's mirror");

// The views the transport holds, asked for once per transport and again after a
// restore, whose addresses it must agree with.
struct uniserve_exchange_info {
  void* transport;
  void* states;    // [ranks * capacity, hidden] rows received, in the activation's type
  void* ids;       // [ranks * capacity, top] i32
  void* weights;   // [ranks * capacity, top] f32
  void* combined;  // [ranks * capacity, hidden] rows the experts computed
  void* live;      // [1] i32: how many leading rows of those views this step holds
  void* packed;    // fused: [capacity, hidden / 2] the encoded rows a rank sends
  void* packed_scales;  // fused: [capacity, scale columns] their block scales
  int64_t ranks;
  int64_t rank;
  int64_t capacity;
  int64_t hidden;
  int64_t top;
  int64_t dtype;
  int64_t device;
  int64_t fused;   // nonzero when one launch runs the whole expert layer
  char error[512];
};

static_assert(offsetof(uniserve_exchange_info, error) == 128,
              "the info's error message must sit where the operator's mirror puts it");
static_assert(sizeof(uniserve_exchange_info) == 640,
              "the info must be the size of the operator's mirror");

// The operator crate's `DType::as_i64()` numbering.
enum uniserve_dtype : int64_t {
  UNISERVE_DTYPE_U8 = 0,
  UNISERVE_DTYPE_I32 = 1,
  UNISERVE_DTYPE_I64 = 2,
  UNISERVE_DTYPE_F16 = 3,
  UNISERVE_DTYPE_BF16 = 4,
  UNISERVE_DTYPE_F32 = 5,
  UNISERVE_DTYPE_F8E4M3 = 6,
  UNISERVE_DTYPE_F4E2M1 = 7,
  UNISERVE_DTYPE_I16 = 8,
};

extern "C" int32_t uniserve_expert_exchange_create(uniserve_exchange_lifecycle* call,
                                                   void* stream);
extern "C" int32_t uniserve_expert_exchange_step(uniserve_exchange_lifecycle* call, void* stream);
extern "C" int32_t uniserve_expert_exchange_suspend(uniserve_exchange_lifecycle* call,
                                                    void* stream);
extern "C" int32_t uniserve_expert_exchange_restore(uniserve_exchange_lifecycle* call,
                                                    void* stream);
extern "C" int32_t uniserve_expert_exchange_release(uniserve_exchange_lifecycle* call,
                                                    void* stream);
extern "C" int32_t uniserve_expert_exchange_info(uniserve_exchange_info* call, void* stream);
extern "C" int32_t uniserve_expert_exchange_launch(uniserve_exchange_call* call, void* stream);
