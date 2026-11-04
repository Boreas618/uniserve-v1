// SPDX-License-Identifier: Apache-2.0
//
// The C header of the expert_exchange module: the operator's two call
// structures (`crates/ops/src/moe/exchange.rs`, DISPATCH_CALL_LAYOUT and
// COMBINE_CALL_LAYOUT), the transport's lifecycle structures and the group
// vtable they are handed (`notes/programs/attention_moe.md`, "Transport
// lifecycle"), and the entries the package declares.

#pragma once

#include <cstddef>
#include <cstdint>

// A dispatch launch: the rank's live rows and the group's transport.
struct uniserve_exchange_dispatch_call {
  void* transport;
  const void* states;
  const int32_t* ids;
  const float* weights;
  int32_t* counts;
  int64_t tokens;
  int64_t top;
  int64_t hidden;
  int64_t experts;
  int64_t max_tokens;
  int64_t ranks;
  int64_t dtype;
  char error[512];
};

// A combine launch: the rows this rank received and processed, summed back into
// its own rows.
struct uniserve_exchange_combine_call {
  void* transport;
  const void* partials;
  const int32_t* counts;
  void* out;
  int64_t tokens;
  int64_t top;
  int64_t hidden;
  int64_t experts;
  int64_t max_tokens;
  int64_t ranks;
  int64_t dtype;
  char error[512];
};

// The executor's transport to the expert-parallel group, as a package sees it.
// One instance per group: the executor creates it before the first transport
// and destroys it after the last one is released. A package may cache neither
// the pointer nor a copy of what it points at, and must never free it.
//
// Every entry is collective over the group and may be called from `create`
// only, never from a launch: a collective issued inside a captured bucket would
// put a second collective in the graph, and the exchange's own all-to-all is
// already there.
struct UniserveGroup {
  // The group's communicator (`ncclComm_t`), for a package that must call NCCL
  // directly, and for nothing else.
  void* handle;
  int64_t ranks;   // ranks of the group
  int64_t rank;    // this rank's index in it
  int64_t device;  // the CUDA device this rank's transport lives on

  // All-gather each rank's `mine` (`bytes` bytes) into `peers`
  // (`[ranks][bytes]`, rank order), on `stream`. The exchange of peer handles
  // goes through this: a fabric handle is 64 bytes, a POSIX-fd handle 4, and
  // the caller states which by its byte count.
  int32_t (*all_gather)(void* handle, const void* mine, void* peers, int64_t bytes,
                        void* stream);

  // Sum `elements` f32 of `data` across the group, in place, on `stream`.
  int32_t (*all_reduce_sum_f32)(void* handle, void* data, int64_t elements, void* stream);

  // `mine` from every rank, in rank order, into `out[ranks]`. Sizing exchanges
  // and flags.
  int32_t (*all_gather_i32)(void* handle, int32_t mine, int32_t* out, void* stream);

  // Every rank reached this point. The executor implements it as a zero-byte
  // collective on the group's communicator, so it costs one launch and works
  // wherever all_gather does.
  int32_t (*barrier)(void* handle, void* stream);

  // Why the last call failed; empty after a successful one.
  char error[512];
};

// The transport's lifecycle, as the executor drives it.
struct uniserve_exchange_lifecycle {
  void* transport;
  // The executor's transport to the group, created before this structure and
  // alive for the whole package life.
  const UniserveGroup* group;
  int64_t ranks;
  int64_t rank;
  int64_t experts;
  int64_t hidden;
  int64_t max_tokens;
  int64_t top;
  int64_t dtype;
  int64_t device;
  char error[512];
};

// The views the transport holds, asked for once per transport (and again after
// a restore, whose addresses it must agree with).
struct uniserve_exchange_info {
  void* transport;
  void* states;
  void* ids;
  void* weights;
  void* counts;
  int64_t ranks;
  int64_t rank;
  int64_t capacity;
  int64_t hidden;
  int64_t top;
  int64_t dtype;
  int64_t device;
  char error[512];
};

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

static_assert(offsetof(UniserveGroup, error) == 64,
              "the group vtable's error message must sit where the executor's mirror puts it");
static_assert(sizeof(UniserveGroup) == 576,
              "the group vtable must be the size of the executor's mirror");
static_assert(offsetof(uniserve_exchange_dispatch_call, error) == 96,
              "the dispatch call's error message must sit where the operator's layout puts it");
static_assert(sizeof(uniserve_exchange_dispatch_call) == 608,
              "the dispatch call must be the size of the operator's layout");
static_assert(offsetof(uniserve_exchange_combine_call, error) == 88,
              "the combine call's error message must sit where the operator's layout puts it");
static_assert(sizeof(uniserve_exchange_combine_call) == 600,
              "the combine call must be the size of the operator's layout");

extern "C" int32_t uniserve_expert_exchange_create(uniserve_exchange_lifecycle* call,
                                                   void* stream);
extern "C" int32_t uniserve_expert_exchange_suspend(uniserve_exchange_lifecycle* call,
                                                    void* stream);
extern "C" int32_t uniserve_expert_exchange_restore(uniserve_exchange_lifecycle* call,
                                                    void* stream);
extern "C" int32_t uniserve_expert_exchange_release(uniserve_exchange_lifecycle* call,
                                                    void* stream);
extern "C" int32_t uniserve_expert_exchange_info(uniserve_exchange_info* call, void* stream);
extern "C" int32_t uniserve_expert_exchange_dispatch(const uniserve_exchange_dispatch_call* call,
                                                     void* stream);
extern "C" int32_t uniserve_expert_exchange_combine(const uniserve_exchange_combine_call* call,
                                                    void* stream);
