// SPDX-License-Identifier: Apache-2.0
//
// The moe_sort module's C entry: validates the operator's call and starts the
// routing pipeline in `routing.cu` (derived from TensorRT-LLM's routing
// kernels; see that file's notice).

#include "uniserve_moe_sort.h"

#include <cstdio>
#include <exception>

/// Starts the routing pipeline; 0 on success, or the stage that failed.
int32_t uniserve_moe_sort_launch(const uniserve_moe_sort_call* call, void* stream);

namespace {

// The message an entry writes before it returns a nonzero status.
int32_t refuse(const uniserve_moe_sort_call* call, const char* message) {
  std::snprintf(const_cast<uniserve_moe_sort_call*>(call)->error,
                sizeof(call->error), "%s", message);
  return 1;
}

}  // namespace

int32_t launch(const uniserve_moe_sort_call* call, void* stream) {
  if (call == nullptr) {
    return 1;
  }
  if (call->ids == nullptr || call->tile_to_expert == nullptr || call->tile_to_limit == nullptr ||
      call->expanded_to_permuted == nullptr || call->permuted_to_expanded == nullptr ||
      call->permuted_to_token == nullptr || call->total_padded_tokens == nullptr ||
      call->num_non_exiting_tiles == nullptr) {
    return refuse(call, "moe_sort needs the router's ids and all seven metadata buffers");
  }
  // The sort writes indices alone, but the weights it is handed must still be
  // the route weights the descriptor names: a call whose scales are another
  // type is not the sort this module was compiled for.
  if (call->scales_dtype != UNISERVE_DTYPE_F32 && call->scales_dtype != UNISERVE_DTYPE_BF16 &&
      call->scales_dtype != UNISERVE_DTYPE_F16) {
    return refuse(call, "moe_sort serves f32, bf16 or fp16 route weights");
  }
  if (call->tokens < 0 || call->top < 0 || call->experts <= 0 || call->local_experts <= 0 ||
      call->tile_tokens <= 0) {
    return refuse(call, "moe_sort needs a non-negative route count and positive experts, local experts and tile");
  }
  if (call->local_offset < 0 || call->local_offset + call->local_experts > call->experts) {
    return refuse(call, "moe_sort's local expert window lies outside the router's experts");
  }
  // Expanded routes and permuted rows are indexed by int32 in the metadata the
  // GEMMs read.
  if (call->tokens * call->top > 0x7fffffffLL) {
    return refuse(call, "moe_sort's expanded routes do not fit the metadata's int32 indexes");
  }
  switch (uniserve_moe_sort_launch(call, stream)) {
    case 0:
      return 0;
    case 2:
      return refuse(call, "moe_sort needs positive local experts and tile");
    case 3:
      return refuse(call, "moe_sort's expert count outgrows one block's shared memory");
    case 4:
      return refuse(call, "moe_sort could not start its metadata kernel");
    case 5:
      return refuse(call, "moe_sort could not start its permutation kernel");
    default:
      return refuse(call, "moe_sort refused the call");
  }
}

/// The operator's entry: the launch, and an exception turned into a message.
///
/// An exception raised inside the kernel's own code would otherwise cross this C boundary,
/// where nothing may unwind: the process ends and the only message a log holds is the runtime's
/// own (`internal error: entered unreachable code`), which names neither the operator nor the
/// reason. A thrown exception states one, so it goes into the call's error buffer and the entry
/// answers a refusal like every other one it makes.
extern "C" int32_t uniserve_moe_sort(const uniserve_moe_sort_call* call, void* stream) {
  if (call == nullptr) {
    return 1;
  }
  try {
    return launch(const_cast<uniserve_moe_sort_call*>(call), stream);
  } catch (std::exception const& thrown) {
    return refuse(const_cast<uniserve_moe_sort_call*>(call), thrown.what());
  } catch (...) {
    return refuse(const_cast<uniserve_moe_sort_call*>(call), "the kernel's launch failed without stating why");
  }
}
