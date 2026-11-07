// SPDX-License-Identifier: Apache-2.0
//
// The device half of the megamoe module's staging: the routes the fused kernel reads.
//
// The kernel reads 64-bit expert ids and masks a row by the id -1. A captured launch
// covers a bucket's extent, of which only the live rows are the batch's, so the rows
// past the live count are masked here.

#include <cstdint>

#include <cuda_runtime.h>

namespace {

__global__ void stage_routes_kernel(int64_t* routes, float* weights, const int32_t* ids,
                                    const float* route_weights, const int32_t* live,
                                    int32_t tokens, int32_t extent, int32_t top,
                                    int32_t experts) {
  const int32_t count = live != nullptr ? min(*live, tokens) : tokens;
  const int64_t all = static_cast<int64_t>(extent) * top;
  for (int64_t at = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; at < all;
       at += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    // A route of a row past the step, or one whose id names no expert, goes nowhere: the router
    // states such an id for a row whose scores are no token's, as a warm pose's are.
    const int32_t id = at / top < count ? ids[at] : -1;
    const bool used = id >= 0 && id < experts;
    routes[at] = used ? static_cast<int64_t>(id) : -1;
    weights[at] = used ? route_weights[at] : 0.0f;
  }
}

}  // namespace

extern "C" int32_t uniserve_megamoe_stage_routes(int64_t* routes, float* weights,
                                                 const int32_t* ids, const float* route_weights,
                                                 const int32_t* live, int32_t tokens,
                                                 int32_t extent, int32_t top, int32_t experts,
                                                 void* stream) {
  const int64_t all = static_cast<int64_t>(extent) * top;
  const int blocks = static_cast<int>((all + 255) / 256 < 64 ? (all + 255) / 256 : 64);
  stage_routes_kernel<<<blocks > 0 ? blocks : 1, 256, 0, static_cast<cudaStream_t>(stream)>>>(
      routes, weights, ids, route_weights, live, tokens, extent, top, experts);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
