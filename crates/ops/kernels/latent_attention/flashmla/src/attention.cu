// SPDX-License-Identifier: Apache-2.0

// The latent_attention module's FlashMLA backend: the operator's call launched as FlashMLA's
// SM100 sparse decoding kernel for DeepSeek-V4.1.
//
// The kernel is FlashMLA's own (`sm100::decode::sparse::head64`, the `flashmla` pin), and it
// owns the attention: TMA gathers of the entries a query names, their dequantization into BF16
// tiles, the tcgen05 products of both GEMMs, the online softmax in log-2 units with the head's
// sink in its final denominator, and the BF16 store of the attended latents. It reads the
// entries of one query by their *slot* in a plane -- `page * entries + offset`, `-1` for none --
// a block of 64 at a time, and takes one valid-prefix length per query.
//
// This unit owns the operator's side:
//
// * `slots_kernel`, which writes each query's entry slots into the call's workspace from the
//   page tables, the positions and the selection the operator passes: the window entries at
//   positions `max(start, p - W + 1) ..= p` and then the entries the selection names, each
//   through its table row (a window table narrower than the request is a ring of pages);
// * the kernel's parameters: the extents and the byte strides of both planes, the queries' and
//   the output's element strides, and the scale in the log-2 units the kernel's softmax keeps;
// * the guards the vendor's assertions would otherwise raise as exceptions, so a call the
//   module cannot serve is refused with the condition it fails.
//
// Every query token of a call is one request of the kernel's batch (`b = tokens`, `s_q = 1`):
// a decode row, the six rows of a verification step and the tokens of a prefill chunk all
// attend their own entries, so one launch serves every phase and the cost of a call follows the
// entries its queries attend rather than the history behind them. Split-KV scheduling is off:
// the entries of one query are the window and the selection (640 of them at the model's shapes,
// under the 640 FlashMLA itself stops splitting at), one query is one block of the grid, and an
// unsplit launch needs neither the scheduler's metadata nor a reduction pass, and is invariant
// to how a batch is divided.

#include <uniserve_latent_attention.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>

#include "cuda_kernels/sm100/decode/sparse/head64/kernel.cuh"

#if !defined(UNISERVE_EXTRA_FORMAT) || !defined(UNISERVE_COMPRESSED)
#error "the build defines the descriptor's compressed side (src/build.py)"
#endif

namespace {

// The record formats of the descriptor: MXFP8 window records, and the compressed side's.
constexpr ModelType kWindowFormat = ModelType::V41;
constexpr ModelType kExtraFormat = ModelType::UNISERVE_EXTRA_FORMAT;
constexpr bool kCompressed = UNISERVE_COMPRESSED;

// The kernel, with the pair of formats the descriptor fixes and no split-KV scheduling.
using Kernel = sm100::decode::sparse::head64::Config;
constexpr Kernel kKernel = Kernel{kWindowFormat, kExtraFormat, /*ENABLE_SPLITKV=*/false};

// The entries one tile load gathers, which a workspace holds whole rows of (`B_TOPK`), and the
// query heads one MMA holds (`MMA_M`).
constexpr int64_t kSlotBlock = 64;
constexpr int64_t kMaxHeads = 64;

// The latent the kernel attends: 512 values, key and value at once.
constexpr int64_t kHeadDim = 512;

// `log2(e)`, the factor the kernel's softmax keeps its scores in log-2 units by (FlashMLA's
// `LOG_2_E`).
constexpr float kLog2E = 1.44269504f;

// The alignment of every region of the workspace (`ops::attention::latent`).
constexpr int64_t kWorkspaceAlignment = 256;

// The threads one block of the slot program writes a query's slots with.
constexpr int kSlotThreads = 256;

// The largest grid extent along y, which the attention's blocks of queries take.
constexpr int64_t kMaxGridY = 65535;

int64_t aligned(int64_t value, int64_t granule) {
  return (value + granule - 1) / granule * granule;
}

int refuse(uniserve_latent_attention_call* call, const char* message) {
  std::snprintf(call->error, sizeof(call->error), "%s", message);
  return 1;
}

// One token's entry slots. Grid: `(tokens)`, one block a query token.
//
// The window's valid slots are a prefix of `window_pad` columns and its length is the entries
// the token's row holds of its window; the selection's slots keep its columns, a negative entry
// staying negative, and its length bounds the valid ones (the column after the last of them),
// which is what the kernel reads of a row.
__global__ void slots_kernel(const int32_t* __restrict__ window_table,
                             const int32_t* __restrict__ window_starts,
                             const int32_t* __restrict__ rows,
                             const int32_t* __restrict__ positions,
                             const int32_t* __restrict__ selection,
                             const int32_t* __restrict__ compressed_table,
                             int32_t* __restrict__ window_slots,
                             int32_t* __restrict__ window_lengths,
                             int32_t* __restrict__ compressed_slots,
                             int32_t* __restrict__ compressed_lengths, int window, int page,
                             int window_pages, int window_pad, int entries, int columns,
                             int compressed_pages, int columns_pad) {
  const int token = blockIdx.x;
  const int row = rows[token];
  const int position = positions[token];

  // The oldest window position the query attends: its window, from the position its row holds
  // its window from. A padding token's negative position attends none.
  const int held = window_starts[row];
  const int first = max(position - window + 1, held);
  const int length = position < 0 ? 0 : position - first + 1;

  for (int column = threadIdx.x; column < window_pad; column += blockDim.x) {
    int slot = -1;
    if (column < length) {
      const int entry = first + column;
      const int page_id = window_table[row * window_pages + (entry / page) % window_pages];
      slot = page_id * page + entry % page;
    }
    window_slots[(int64_t)token * window_pad + column] = slot;
  }
  if (threadIdx.x == 0) {
    window_lengths[token] = length;
  }

  if constexpr (kCompressed) {
    // The column after the last entry the selection names, reduced over the block.
    __shared__ int named;
    if (threadIdx.x == 0) {
      named = 0;
    }
    __syncthreads();

    int last = 0;
    for (int column = threadIdx.x; column < columns_pad; column += blockDim.x) {
      const int entry = column < columns ? selection[(int64_t)token * columns + column] : -1;
      int slot = -1;
      if (entry >= 0) {
        const int page_id =
            compressed_table[row * compressed_pages + (entry / entries) % compressed_pages];
        slot = page_id * entries + entry % entries;
        last = column + 1;
      }
      compressed_slots[(int64_t)token * columns_pad + column] = slot;
    }
    atomicMax(&named, last);
    __syncthreads();

    if (threadIdx.x == 0) {
      compressed_lengths[token] = named;
    }
  }
}

// The workspace regions: the window's slots and lengths, the selection's when the compilation
// has a compressed side, and the normalizer the kernel writes one of per query head. The bytes
// are `ops::attention::latent::workspace_bytes`, which the caller sizes the workspace by.
struct Workspace {
  int32_t* window_slots;
  int32_t* window_lengths;
  int32_t* compressed_slots;
  int32_t* compressed_lengths;
  float* lse;
  int64_t bytes;
};

Workspace carved(const uniserve_latent_attention_call* call, int64_t window_pad,
                 int64_t columns_pad) {
  constexpr int64_t kWord = 4;
  auto* at = static_cast<uint8_t*>(call->workspace);
  int64_t taken = 0;

  auto region = [&](int64_t bytes) {
    uint8_t* start = at + taken;
    taken += aligned(bytes, kWorkspaceAlignment);
    return start;
  };

  Workspace workspace = {};
  workspace.window_slots = reinterpret_cast<int32_t*>(region(call->tokens * window_pad * kWord));
  workspace.window_lengths = reinterpret_cast<int32_t*>(region(call->tokens * kWord));
  if (kCompressed) {
    workspace.compressed_slots =
        reinterpret_cast<int32_t*>(region(call->tokens * columns_pad * kWord));
    workspace.compressed_lengths = reinterpret_cast<int32_t*>(region(call->tokens * kWord));
  }
  workspace.lse = reinterpret_cast<float*>(region(call->tokens * call->heads * kWord));
  workspace.bytes = taken;
  return workspace;
}

// The slots of every query, then the attention over them.
void launch(const uniserve_latent_attention_call* call, const Workspace& workspace,
            int64_t window_pad, int64_t columns_pad, cudaStream_t stream) {
  slots_kernel<<<static_cast<unsigned>(call->tokens), kSlotThreads, 0, stream>>>(
      static_cast<const int32_t*>(call->window_table),
      static_cast<const int32_t*>(call->window_starts), static_cast<const int32_t*>(call->rows),
      static_cast<const int32_t*>(call->positions),
      static_cast<const int32_t*>(call->selection),
      static_cast<const int32_t*>(call->compressed_table), workspace.window_slots,
      workspace.window_lengths, workspace.compressed_slots, workspace.compressed_lengths,
      static_cast<int>(call->window_size), static_cast<int>(call->page),
      static_cast<int>(call->window_pages), static_cast<int>(window_pad),
      static_cast<int>(call->entries), static_cast<int>(call->columns),
      static_cast<int>(call->compressed_pages), static_cast<int>(columns_pad));
  KU_CHECK_KERNEL_LAUNCH();

  const int64_t query_row = call->heads * kHeadDim;
  SparseAttnDecodeParams params = {};
  params.b = static_cast<int>(call->tokens);
  params.s_q = 1;
  params.h_q = static_cast<int>(call->heads);
  params.h_kv = 1;
  params.d_qk = static_cast<int>(kHeadDim);
  params.d_v = static_cast<int>(kHeadDim);
  params.sm_scale = call->scale;
  params.sm_scale_div_log2 = call->scale * kLog2E;
  params.num_blocks = static_cast<int>(call->window_plane_pages);
  params.page_block_size = static_cast<int>(call->page);
  params.topk = static_cast<int>(window_pad);
  params.model_type = kWindowFormat;
  params.extra_model_type = kExtraFormat;

  params.q = static_cast<bf16*>(const_cast<void*>(call->query));
  params.kv = static_cast<bf16*>(const_cast<void*>(call->window));
  params.indices = workspace.window_slots;
  params.topk_length = workspace.window_lengths;
  params.attn_sink = static_cast<float*>(const_cast<void*>(call->sink));
  params.lse = workspace.lse;
  params.out = static_cast<bf16*>(call->out);

  if (kCompressed) {
    params.extra_num_blocks = static_cast<int>(call->compressed_plane_pages);
    params.extra_page_block_size = static_cast<int>(call->entries);
    params.extra_topk = static_cast<int>(columns_pad);
    params.extra_kv = static_cast<bf16*>(const_cast<void*>(call->compressed));
    params.extra_indices = workspace.compressed_slots;
    params.extra_topk_length = workspace.compressed_lengths;
  }

  // The queries, the output and the normalizer are one row per query token; both planes stride
  // by bytes, a page of records at a time.
  params.stride_q_b = static_cast<int>(query_row);
  params.stride_q_s_q = static_cast<int>(query_row);
  params.stride_q_h_q = static_cast<int>(kHeadDim);
  params.stride_kv_block = static_cast<int>(call->page * call->window_record);
  params.stride_kv_row = static_cast<int>(call->window_record);
  params.stride_indices_b = static_cast<int>(window_pad);
  params.stride_indices_s_q = static_cast<int>(window_pad);
  params.stride_lse_b = static_cast<int>(call->heads);
  params.stride_lse_s_q = static_cast<int>(call->heads);
  params.stride_o_b = static_cast<int>(query_row);
  params.stride_o_s_q = static_cast<int>(query_row);
  params.stride_o_h_q = static_cast<int>(kHeadDim);
  if (kCompressed) {
    params.stride_extra_kv_block = static_cast<int>(call->entries * call->compressed_record);
    params.stride_extra_kv_row = static_cast<int>(call->compressed_record);
    params.stride_extra_indices_b = static_cast<int>(columns_pad);
    params.stride_extra_indices_s_q = static_cast<int>(columns_pad);
  }

  params.num_sm = static_cast<int>(call->multiprocessors);
  params.stream = stream;
  params.enable_split_kv = false;

  sm100::decode::sparse::head64::run_flash_splitkv_mla_fp8_sparse_kernel<kKernel>(params);
}

}  // namespace

extern "C" int uniserve_latent_attention(uniserve_latent_attention_call* call, void* stream) {
  if (call == nullptr) {
    return 1;
  }
  std::memset(call->error, 0, sizeof(call->error));

  if (call->head_dim != kHeadDim) {
    return refuse(call, "the FlashMLA module attends latents of 512 values");
  }
  if (call->heads < 1 || call->heads > kMaxHeads) {
    char message[256];
    std::snprintf(message, sizeof(message),
                  "the FlashMLA module attends 1 to %lld query heads of a token and was handed "
                  "%lld",
                  static_cast<long long>(kMaxHeads), static_cast<long long>(call->heads));
    return refuse(call, message);
  }
  if (call->window_record != 528) {
    return refuse(call, "the FlashMLA module reads MXFP8 window records of 528 bytes");
  }
  if (kCompressed) {
    if (call->compressed == nullptr) {
      return refuse(call, "a compilation with a compressed side called without its plane");
    }
    if (call->compressed_record != 288) {
      return refuse(call, "the FlashMLA module reads NVFP4 compressed records of 288 bytes");
    }
  } else if (call->compressed != nullptr) {
    return refuse(call, "a window-only compilation called with a compressed plane");
  }
  if (call->tokens > kMaxGridY) {
    char message[256];
    std::snprintf(message, sizeof(message),
                  "the attention takes one block of its grid per query token, and a grid holds "
                  "%lld of them, not %lld",
                  static_cast<long long>(kMaxGridY), static_cast<long long>(call->tokens));
    return refuse(call, message);
  }
  if (call->tokens == 0) {
    return 0;
  }

  // The slots of a query are whole blocks of the entries one tile load gathers.
  const int64_t window_pad = aligned(call->window_size, kSlotBlock);
  const int64_t columns_pad = kCompressed ? aligned(call->columns, kSlotBlock) : 0;
  const Workspace workspace = carved(call, window_pad, columns_pad);
  if (workspace.bytes > call->workspace_bytes) {
    char message[256];
    std::snprintf(message, sizeof(message),
                  "the module writes %lld bytes of slots, lengths and normalizers for %lld query "
                  "tokens, and the workspace holds %lld",
                  static_cast<long long>(workspace.bytes), static_cast<long long>(call->tokens),
                  static_cast<long long>(call->workspace_bytes));
    return refuse(call, message);
  }

  // The kernel's host side raises its own refusals as exceptions (`KU_ASSERT`, `KU_CUDA_CHECK`
  // of the pin's `kerutils`); a serving process takes them as this call's error.
  try {
    launch(call, workspace, window_pad, columns_pad, static_cast<cudaStream_t>(stream));
  } catch (const std::exception& error) {
    return refuse(call, error.what());
  }
  return 0;
}
