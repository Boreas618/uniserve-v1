// SPDX-License-Identifier: Apache-2.0
//
// The deepgemm module's staging: what a rank writes into its own window before the fused launch
// pulls it.
//
// - Each BF16 row of the launch is encoded into MXFP8, the rule `mxfp8_quantize` and the
//   reference's `act_quant` (`scale_fmt = "ue8m0"`) state: per 32 elements, `amax = max(|x|,
//   1e-4)` in FP32, the scale the least power of two at or above `amax * fl(1/448)` (its exponent
//   read from the product's bits), the codes `e4m3(x / scale)`, round to nearest even. The codes
//   go to the window's `[rows, hidden]` input; the scale bytes, four a 32-bit word (block `4g + i`
//   in byte `i`), to its row-major `[rows, hidden / 128]` scales.
// - With shared experts, the same words also go to the shared experts' scale table, which their
//   first product reads MN-major in pages of the tile height `block_m`: row `t` of the rank at
//   page `t / block_m`, each page padded to whole 128-row scale copies and its rows in the copy's
//   order (`32i + j` at `4j + i` within each 128 rows), column `g` of `hidden / 128` at
//   `g * shared_rows` (vLLM's `prepare_megamoe_inputs`, `_shared_sf_row`).
// - The routes go in as 64-bit expert ids with their FP32 weights. A captured launch covers a
//   bucket's extent, of which the live count is the batch's: a row past it is staged as zeros
//   (no shared expert output) and routed nowhere (id -1), as is a route whose id names no expert.
//
// The encoding reads every row once and is bound by memory: a warp encodes two 256-element chunks
// of a row, each lane eight elements of each in one 16-byte load, both chunks' loads issued before
// either is encoded; four lanes hold a 32-element block and sixteen a 128-element scale word.
//
// The fused kernel is launched as this kernel's programmatic dependent (`kernel.cu`): it sets up
// its barriers and tensor memory while this kernel runs and waits for its completion before it
// reads the window, so this kernel lets it start at once.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int kChunkElements = 256;
constexpr int kLaneElements = 8;
constexpr int kBlockLanes = 32 / kLaneElements;  // lanes of one 32-element block
constexpr int kWordLanes = 128 / kLaneElements;  // lanes of one 128-element scale word
constexpr int kChunksPerWarp = 2;
constexpr int kWarpsPerBlock = 8;

// Where row `token` of the shared experts' scale table lies (`_shared_sf_row`). Indices are 32-bit:
// a rank's rows and a scale column's rows stay far below 2^31, and a 64-bit division costs the
// encoding more than its memory traffic.
__device__ uint32_t shared_scale_row(uint32_t token, uint32_t block_m) {
  const uint32_t page = token / block_m;
  const uint32_t within = token - page * block_m;
  const uint32_t padded = (block_m + 127) / 128 * 128;
  const uint32_t ordered = within / 128 * 128 + within % 32 * 4 + within % 128 / 32;
  return page * padded + ordered;
}

// Encodes one lane's eight elements of a chunk, writes their codes, and answers the scale byte of
// the lane's block.
__device__ uint32_t encode(const float (&values)[kLaneElements], uint8_t* codes) {
  float amax = 0.0f;
  for (int i = 0; i < kLaneElements; ++i) {
    amax = fmaxf(amax, fabsf(values[i]));
  }
  for (int offset = 1; offset < kBlockLanes; offset *= 2) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, offset));
  }
  amax = fmaxf(amax, 1e-4f);

  // The least power of two at or above amax * fl(1/448): the product's exponent, plus one unless
  // its mantissa is zero. A division by a power of two is exact, so the cast is the one rounding.
  const uint32_t ratio = __float_as_uint(__fmul_rn(amax, 1.0f / 448.0f));
  const uint32_t biased = ((ratio >> 23) & 0xff) + ((ratio & 0x7fffff) != 0 ? 1 : 0);
  const float inverse = __uint_as_float((254u - biased) << 23);
  uint32_t words[2] = {0, 0};
  for (int i = 0; i < kLaneElements; i += 2) {
    const float2 scaled = make_float2(fminf(fmaxf(values[i] * inverse, -448.0f), 448.0f),
                                      fminf(fmaxf(values[i + 1] * inverse, -448.0f), 448.0f));
    const __nv_fp8x2_storage_t pair = __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
    words[i / 4] |= static_cast<uint32_t>(pair) << (8 * (i % 4));
  }
  *reinterpret_cast<uint2*>(codes) = make_uint2(words[0], words[1]);
  return biased;
}

__global__ void stage_kernel(const __nv_bfloat16* states, const int32_t* ids, const float* weights,
                             const int32_t* live, uint8_t* codes, uint32_t* scales,
                             int64_t* routes, float* route_weights, uint32_t* shared_scales,
                             uint32_t tokens, uint32_t hidden, uint32_t top, int32_t experts,
                             uint32_t block_m, uint32_t shared_rows) {
  // The dependent fused kernel waits for this grid's completion before it reads anything.
  cudaTriggerProgrammaticLaunchCompletion();

  const uint32_t chunks = hidden / kChunkElements;
  const uint32_t all = tokens * chunks;
  const uint32_t first = (blockIdx.x * kWarpsPerBlock + threadIdx.x / 32) * kChunksPerWarp;
  if (first >= all) {
    return;
  }
  const uint32_t lane = threadIdx.x % 32;
  const uint32_t count = live != nullptr ? min(static_cast<uint32_t>(*live), tokens) : tokens;
  const uint32_t words = hidden / 128;

  // Both chunks' loads first; a row past the live count, or a chunk past the last, reads zeros.
  float values[kChunksPerWarp][kLaneElements];
  for (int c = 0; c < kChunksPerWarp; ++c) {
    const uint32_t chunk = first + c;
    uint4 packed = make_uint4(0, 0, 0, 0);
    if (chunk < all && chunk / chunks < count) {
      const size_t at = static_cast<size_t>(chunk) * kChunkElements + lane * kLaneElements;
      packed = *reinterpret_cast<const uint4*>(states + at);
    }
    const uint32_t parts[4] = {packed.x, packed.y, packed.z, packed.w};
    for (int i = 0; i < 4; ++i) {
      const __nv_bfloat162 pair = *reinterpret_cast<const __nv_bfloat162*>(&parts[i]);
      values[c][2 * i] = __low2float(pair);
      values[c][2 * i + 1] = __high2float(pair);
    }
  }

  for (int c = 0; c < kChunksPerWarp; ++c) {
    const uint32_t chunk = first + c;
    if (chunk >= all) {
      break;
    }
    const uint32_t row = chunk / chunks;
    const uint32_t column = chunk - row * chunks;
    const uint32_t biased = encode(
        values[c], codes + static_cast<size_t>(chunk) * kChunkElements + lane * kLaneElements);

    // The four blocks' scale bytes of each of the chunk's two words, block `i` in byte `i`.
    const uint32_t half = lane / kWordLanes;
    uint32_t scale_word = 0;
    for (int i = 0; i < 4; ++i) {
      const uint32_t byte =
          __shfl_sync(0xffffffff, biased, half * kWordLanes + i * kBlockLanes);
      scale_word |= byte << (8 * i);
    }
    if (lane % kWordLanes == 0) {
      const uint32_t word = column * 2 + half;
      scales[static_cast<size_t>(row) * words + word] = scale_word;
      if (shared_scales != nullptr) {
        shared_scales[static_cast<size_t>(word) * shared_rows + shared_scale_row(row, block_m)] =
            scale_word;
      }
    }

    // The routes, once a row.
    if (column == 0 && lane < top) {
      const size_t at = static_cast<size_t>(row) * top + lane;
      const int32_t id = row < count ? ids[at] : -1;
      const bool routed = id >= 0 && id < experts;
      routes[at] = routed ? static_cast<int64_t>(id) : -1;
      route_weights[at] = routed ? weights[at] : 0.0f;
    }
  }
}

}  // namespace

// Stages `tokens` rows of `states` (`[tokens, hidden]` BF16, `hidden` whole 256-element chunks)
// and their `[tokens, top]` routes (`ids` I32 slots, `weights` FP32) into a rank's window,
// masking the rows at and past `*live` (every row when `live` is null). `shared_scales` is null
// for a layer without shared experts. Answers the runtime's error code, zero on success.
extern "C" int32_t uniserve_deepgemm_stage(const void* states, const int32_t* ids,
                                           const float* weights, const int32_t* live,
                                           void* codes, void* scales, int64_t* routes,
                                           float* route_weights, void* shared_scales,
                                           int32_t tokens, int32_t hidden, int32_t top,
                                           int32_t experts, int32_t block_m, int64_t shared_rows,
                                           void* stream) {
  if (tokens == 0) {
    return 0;
  }
  const int64_t chunks = static_cast<int64_t>(tokens) * (hidden / kChunkElements);
  const int64_t per_block = kWarpsPerBlock * kChunksPerWarp;
  const auto blocks = static_cast<unsigned>((chunks + per_block - 1) / per_block);
  stage_kernel<<<blocks, kWarpsPerBlock * 32, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(states), ids, weights, live,
      static_cast<uint8_t*>(codes), static_cast<uint32_t*>(scales), routes, route_weights,
      static_cast<uint32_t*>(shared_scales), static_cast<uint32_t>(tokens),
      static_cast<uint32_t>(hidden), static_cast<uint32_t>(top), experts,
      static_cast<uint32_t>(block_m), static_cast<uint32_t>(shared_rows));
  return static_cast<int32_t>(cudaGetLastError());
}
