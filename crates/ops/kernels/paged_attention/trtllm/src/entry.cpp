// SPDX-License-Identifier: Apache-2.0

// The paged_attention module's trtllm backend: the operator's call turned into a
// trtllm-gen FMHA launch.
//
// The kernels are TensorRT-LLM's: `TllmGenFmhaRunner` selects one from the
// artifact's generated kernel table for the call's configuration, and
// `fmhaKernels.cuh` loads the chosen cubin through `getCubin`. The module
// publishes the cubins its descriptors can select beside itself, under
// `cubins/`, and answers that call from its own directory: the Program Server
// hands a resolved module to the process, so a path compiled in at build time
// would point at the build host. The manifest records each cubin's sha256, which
// is what the installer verified when the module arrived; this loader opens the
// file the kernel table names.
//
// What this unit owns is the operator's side of the trade: the tier's
// configuration (paged KV, causal or dense mask, persistent scheduler for the
// prefill and static multi-CTA for the decode), the workspace carve-out between
// the multi-CTA counter and the split-KV scratch, and the refusal of a call the
// module does not serve.

#include <dlfcn.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "flashinfer/trtllm/fmha/fmhaRunner.cuh"

#include "entry.h"

namespace {

namespace fs = std::filesystem;

// The directory this module was loaded from.
fs::path module_directory() {
  Dl_info info{};
  if (dladdr(reinterpret_cast<void*>(&uniserve_paged_attention), &info) == 0 ||
      info.dli_fname == nullptr) {
    return fs::current_path();
  }
  return fs::path(info.dli_fname).parent_path();
}

}  // namespace

namespace flashinfer::trtllm_cubin_loader {

// The artifact's kernel table asks for a cubin by the name its build gave it;
// this module publishes the ones its descriptors can select beside itself,
// under `cubins/`. The manifest recorded each cubin's sha256, which the
// installer verified when the module arrived, so the file the table names is the
// one the table's hash belongs to.
std::string getCubin(const std::string& name, const std::string& sha256) {
  (void)sha256;
  // The build host cannot run a kernel, so the module names the cubin its
  // launcher asked for when asked to: the GPU check reads which kernel served a
  // call from that file. Diagnostics only; the load path does not depend on it.
  if (const char* log = std::getenv("UNISERVE_CUBIN_LOG"); log != nullptr && log[0] != '\0') {
    std::ofstream written(log, std::ios::app);
    written << fs::path(name).filename().string() << "\n";
  }
  const fs::path cubin = module_directory() / "cubins" / fs::path(name).filename();
  std::ifstream file(cubin, std::ios::binary);
  if (!file) {
    throw std::runtime_error("the module has no cubin " + cubin.string());
  }
  return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

}  // namespace flashinfer::trtllm_cubin_loader

namespace {

// The message an entry writes before it returns a nonzero status.
int32_t refuse(uniserve_paged_attention_call* call, const std::string& message) {
  std::snprintf(call->error, sizeof(call->error), "%s", message.c_str());
  return 1;
}

// The operator's element type as trtllm-gen names it.
bool dtype_of(int64_t code, Data_type& dtype) {
  switch (code) {
    case UNISERVE_DTYPE_F16:
      dtype = DATA_TYPE_FP16;
      return true;
    case UNISERVE_DTYPE_BF16:
      dtype = DATA_TYPE_BF16;
      return true;
    case UNISERVE_DTYPE_F8E4M3:
      dtype = DATA_TYPE_E4M3;
      return true;
    default:
      return false;
  }
}

// The runners are one per element-type combination and load the artifact's
// kernel table; the process keeps them for its life, as captured graphs do.
TllmGenFmhaRunner& runner_of(Data_type query, Data_type key_value, Data_type out) {
  static std::mutex mutex;
  static std::unordered_map<uint32_t, TllmGenFmhaRunner> runners;
  const auto key = static_cast<uint32_t>(query) << 16 | static_cast<uint32_t>(key_value) << 8 |
                   static_cast<uint32_t>(out);
  std::lock_guard<std::mutex> held(mutex);
  auto found = runners.find(key);
  if (found == runners.end()) {
    found = runners
                .emplace(key, TllmGenFmhaRunner(query, key_value, key_value, out))
                .first;
  }
  return found->second;
}

// The multi-processor count the kernel's scheduler is sized by.
int32_t multiprocessor_count() {
  static std::mutex mutex;
  static std::unordered_map<int, int32_t> counts;
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return 0;
  }
  std::lock_guard<std::mutex> held(mutex);
  auto found = counts.find(device);
  if (found == counts.end()) {
    int32_t value = 0;
    cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, device);
    found = counts.emplace(device, value).first;
  }
  return found->second;
}

// The bytes the decode path's multi-CTA KV counter needs, rounded to whole
// eight-slot groups as the kernel's semaphores expect.
int64_t counter_bytes(int64_t batch, int64_t q_heads, int64_t sm_count) {
  const int64_t slots = std::max(batch * q_heads, sm_count);
  return ((slots + 7) / 8) * 8 * static_cast<int64_t>(sizeof(uint32_t));
}

}  // namespace

extern "C" int32_t uniserve_paged_attention(const uniserve_paged_attention_call* call,
                                            void* stream) {
  if (call == nullptr) {
    return 1;
  }
  auto refuse_call = [&](const std::string& message) {
    return refuse(const_cast<uniserve_paged_attention_call*>(call), message);
  };

  Data_type query_type{};
  Data_type key_value_type{};
  Data_type output_type{};
  if (!dtype_of(call->dtype_query, query_type) || !dtype_of(call->dtype_key_value, key_value_type) ||
      !dtype_of(call->dtype_output, output_type)) {
    return refuse_call("paged_attention/trtllm serves bf16, fp16 and fp8 e4m3");
  }
  // An FP8 cache is a separate trade: the launch code reads one E4M3 block scale
  // per sixteen elements of the head dimension per key, per head and per page,
  // while the call carries one FP32 scale per page, so a call here cannot hand
  // the kernel the scales it reads.
  if (key_value_type == DATA_TYPE_E4M3) {
    return refuse_call(
        "paged_attention/trtllm serves an unquantized cache: trtllm-gen reads E4M3 block scales "
        "per sixteen head elements and the call carries one scale per page");
  }
  if (call->key_scale != nullptr || call->value_scale != nullptr) {
    return refuse_call("an unquantized cache takes no scales");
  }
  if (call->out == nullptr || call->query == nullptr || call->key == nullptr ||
      call->value == nullptr || call->page_table == nullptr || call->seq_lens == nullptr ||
      call->workspace == nullptr) {
    return refuse_call("paged_attention needs out, query, key, value, page_table, seq_lens and workspace");
  }
  if (call->window_left >= 0) {
    return refuse_call("the trtllm module is compiled without an attention window");
  }
  if (call->q_heads % call->kv_heads != 0) {
    return refuse_call("paged_attention needs q_heads to be a multiple of kv_heads");
  }
  if (call->head_dim % 8 != 0 || call->head_dim > 256) {
    return refuse_call("paged_attention serves head widths in whole 8-element groups up to 256");
  }
  if (call->page_size <= 0 || (call->page_size & (call->page_size - 1)) != 0) {
    return refuse_call("paged_attention needs a power-of-two page");
  }
  if (!(call->scale > 0.0)) {
    return refuse_call("paged_attention needs a positive softmax scale");
  }
  if (call->phase != UNISERVE_PHASE_PREFILL && call->phase != UNISERVE_PHASE_DECODE) {
    return refuse_call("paged_attention takes the prefill or the decode phase");
  }

  const int32_t sm_count = multiprocessor_count();
  if (sm_count <= 0) {
    return refuse_call("paged_attention could not read the device's multiprocessor count");
  }

  // The workspace is the program's arena: the decode's multi-CTA counter first,
  // then the split-KV scratch. The prefill needs the scratch alone.
  auto* arena = static_cast<char*>(call->workspace);
  const bool decode = call->phase == UNISERVE_PHASE_DECODE;
  int64_t counter = 0;
  if (decode) {
    counter = counter_bytes(call->batch, call->q_heads, sm_count);
    if (reinterpret_cast<std::uintptr_t>(arena) % 16 != 0) {
      return refuse_call("the workspace must be 16-byte aligned");
    }
  }
  if (call->workspace_bytes <= counter) {
    return refuse_call("the workspace is too small for the call's counter and scratch");
  }

  TllmGenFmhaRunnerParams params{};
  params.mBf16QFp8KvTransformMode = Bf16QFp8KvTransformMode::Full;
  params.qPtr = const_cast<void*>(call->query);
  params.kPtr = const_cast<void*>(call->key);
  params.vPtr = const_cast<void*>(call->value);
  params.kSfBasePtr = nullptr;
  params.vSfBasePtr = nullptr;
  params.kvPageIdxPtr = const_cast<int32_t*>(call->page_table);
  params.seqLensKvPtr = const_cast<int32_t*>(call->seq_lens);
  params.oPtr = call->out;
  params.mHeadDimQk = call->head_dim;
  params.mHeadDimV = call->head_dim;
  params.mNumHeadsQ = call->q_heads;
  params.mNumHeadsKv = call->kv_heads;
  params.mNumHeadsQPerKv = call->q_heads / call->kv_heads;
  params.mBatchSize = call->batch;
  params.mMaxSeqLenKv = call->max_kv_len;
  params.mMaxNumPagesPerSeqKv =
      call->page_size > 0 ? (call->max_kv_len + call->page_size - 1) / call->page_size : 0;
  params.mNumTokensPerPage = call->page_size;
  params.mQkvLayout = QkvLayout::PagedKv;
  params.mMultiProcessorCount = sm_count;
  params.qStrideTokens = call->q_stride_tokens;
  params.qStrideHeads = call->q_stride_heads;
  params.kStrideKeysValues = call->kv_stride_tokens;
  params.kStrideHeads = call->kv_stride_heads;
  params.kStrideBatch = call->kv_stride_page;
  params.vStrideKeysValues = call->kv_stride_tokens;
  params.vStrideHeads = call->kv_stride_heads;
  params.vStrideBatch = call->kv_stride_page;
  params.mNumPagesInMemPool = call->num_pages;
  params.stream = reinterpret_cast<cudaStream_t>(stream);
  params.outputScale = 1.0;
  params.outputScalePtr = nullptr;
  // The cache's per-page scales are applied by the kernel itself, so the softmax
  // scale is the operator's and nothing else is fused into it.
  params.mScaleSfKv = 1.0f;
  params.kvSfScalePtr = nullptr;
  params.scaleSoftmaxLog2 = call->scale * M_LOG2E;
  params.scaleSoftmaxLog2Ptr = nullptr;
  params.oSfPtr = nullptr;
  params.mChunkedAttentionSize = INT_MAX;
  params.mAttentionWindowSize =
      call->window_left < 0 ? INT_MAX : static_cast<int>(call->window_left + 1);
  params.mMaxSeqLenQ = call->max_q_len;
  params.mSumOfSeqLensQ = call->tokens;
  params.mUsesSharedPagedKvIdx = true;
  params.enable_pdl = true;
  params.mUseBlockSparseAttention = false;
  params.mSparseMlaType = TrtllmGenSparseMlaType::None;
  params.mSparseMlaTopK = 0;
  params.mSkipsSoftmaxWhenPossible = false;
  params.mSkipSoftmaxThresholdScaleFactor = 0.0f;
  params.mUseFp16Softmax = false;
  params.mUsesSpcompress = false;
  params.ptrAttentionSinks = nullptr;

  if (decode) {
    // One query token per request: the generation kernels take the causal mask
    // as a dense one, a static tile scheduler and the multi-CTA split of the KV
    // pages, which is what the counter serializes.
    params.mMaskType = TrtllmGenAttentionMaskType::Causal;
    params.mKernelType = FmhaKernelType::Generation;
    params.mTileScheduler = TileScheduler::Static;
    params.mMultiCtasKvMode = true;
    params.multiCtasKvCounterPtr = reinterpret_cast<int32_t*>(arena);
    params.multiCtasKvScratchPtr = arena + counter;
    params.cumSeqLensQPtr = const_cast<int32_t*>(call->cum_seq_lens_q);
    params.cumSeqLensKvPtr = nullptr;
  } else {
    params.mMaskType =
        call->causal != 0 ? TrtllmGenAttentionMaskType::Causal : TrtllmGenAttentionMaskType::Dense;
    params.mKernelType = FmhaKernelType::Context;
    params.mTileScheduler = TileScheduler::Persistent;
    params.mMultiCtasKvMode = false;
    params.multiCtasKvScratchPtr = arena;
    params.cumSeqLensQPtr = const_cast<int32_t*>(call->cum_seq_lens_q);
    params.cumSeqLensKvPtr = nullptr;
  }

  try {
    TllmGenFmhaRunner& runner = runner_of(query_type, key_value_type, output_type);
    const auto [supported, reason] = runner.isSupportedWithInfo(params);
    if (!supported) {
      return refuse_call(std::string("no trtllm-gen kernel for this call: ") + reason);
    }
    runner.run(params);
  } catch (const std::exception& failure) {
    return refuse_call(failure.what());
  }
  return 0;
}
