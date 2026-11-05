// SPDX-License-Identifier: Apache-2.0

// The route_gemm_nvfp4 entry: the operator's call, checked against the geometry this
// module was compiled for, handed to trtllm-gen's routed block-scaled second matrix.
//
// What the entry owns: the checks that turn a misplaced operand into a message (the
// compiled geometry, the operand dtypes, the scratch floor), the per-call tactic
// choice, the per-column bias the artifact's kernels take (zeroed, because this
// operator has none), and the device word the GEMM bounds its M by -- the padded route
// count. What it does not own: any arithmetic. The product is the artifact's, and
// everything the call states is either its geometry or an operand.

#include "entry.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "flashinfer/trtllm/fused_moe/runner.h"
#include <exception>

namespace {

namespace fs = std::filesystem;

// The directory this module was loaded from.
fs::path module_directory() {
  Dl_info info{};
  if (dladdr(reinterpret_cast<void*>(uniserve_route_gemm_nvfp4), &info) == 0 ||
      info.dli_fname == nullptr) {
    return fs::current_path();
  }
  return fs::path(info.dli_fname).parent_path();
}

}  // namespace

namespace flashinfer::trtllm_cubin_loader {

// The artifact's kernel table asks for a cubin by the name its build gave it; this module
// publishes the ones its descriptors can select beside itself, under `cubins/`. The
// manifest recorded each cubin's sha256, which the installer verified when the module
// arrived, so the file the table names is the one the table's hash belongs to.
std::string getCubin(const std::string& name, const std::string& sha256) {
  (void)sha256;
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


// The compiled geometry, stated by the build from the descriptor's own values.
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

namespace {

namespace btg = batchedGemm::trtllm::gen;
using Product = tensorrt_llm::kernels::trtllmgen_moe::Gemm2::Runner;

constexpr int32_t kHidden = UNISERVE_HIDDEN;
constexpr int32_t kIntermediate = UNISERVE_INTERMEDIATE;
constexpr int32_t kExperts = UNISERVE_EXPERTS;
constexpr int32_t kTop = UNISERVE_TOP;
constexpr int32_t kTileTokens = UNISERVE_TILE_TOKENS;

//: The dense half of the E2M1 kernels' block scales: this module serves the
//: artifact's variants whose *weight* scales are the plain `[experts, rows, columns/16]`
//: table. The 8x4-tiled variants are a different shard layout and a different
//: descriptor, and a module built for one may not read the other.
constexpr batchedGemm::gemm::MatrixLayout kWeightLayout =
    batchedGemm::gemm::MatrixLayout::MajorK;

/// The runner, built once per process from the compiled geometry.
///
/// The kernels must be read in their block-swizzled form (`useShuffledMatrix`): the
/// loader states the shard's layout and refuses a plain one, so by the time a call
/// arrives the operand is the layout the kernels index.
Product& runner() {
  static Product one(
      /*dtypeAct=*/btg::Dtype::E2m1,
      /*dtypeWeights=*/btg::Dtype::E2m1,
      /*dtypeOutput=*/btg::Dtype::Bfloat16,
      /*useDeepSeekFp8=*/false,
      /*tileTokensDim=*/kTileTokens,
      /*useShuffledMatrix=*/true,
      /*weight_layout=*/kWeightLayout,
      // The artifact's FP4 kernels read the encoder's per-token factor beside the block scales.
      /*usePerTokenScaling=*/true,
      /*usePerChannelScaling=*/false);
  return one;
}

int refuse(uniserve_route_gemm_nvfp4_call* call, char const* message) {
  std::snprintf(call->error, sizeof(call->error), "%s", message);
  return 1;
}

/// Refuse a call whose geometry is not the one this module was compiled for.
///
/// The tactic is compiled in: the tiles, the cluster and the operand types belong to the
/// cubins this module published, so a call of another shape would be read by a kernel
/// addressing rows and columns it does not have.
int check_geometry(uniserve_route_gemm_nvfp4_call const* call) {
  struct Expected {
    char const* name;
    int64_t found;
    int64_t expected;
  };
  for (Expected const& one :
       {Expected{"hidden", call->hidden_size, kHidden},
        Expected{"intermediate", call->intermediate, kIntermediate},
        Expected{"experts", call->experts, kExperts}}) {
    if (one.found != one.expected) {
      return 1;
    }
  }
  if (call->top > kTop || call->top <= 0) {
    return 1;
  }
  if (call->routes < call->top) {
    return 1;
  }
  return 0;
}

/// The message for a refused geometry, filled only when there is one.
int refuse_geometry(uniserve_route_gemm_nvfp4_call* call) {
  char message[512];
  std::snprintf(message, sizeof(message),
                "this module is compiled for hidden %d, intermediate %d, experts %d, top up to %d "
                "and was handed hidden %lld, intermediate %lld, experts %lld, top %lld",
                kHidden, kIntermediate, kExperts, kTop, static_cast<long long>(call->hidden_size),
                static_cast<long long>(call->intermediate), static_cast<long long>(call->experts),
                static_cast<long long>(call->top));
  return refuse(call, message);
}

}  // namespace

/// The launch itself: every check this entry makes, then the artifact's product.
///
/// It answers 0 or a refusal with `call->error` filled, and it may throw: the artifact's
/// runner construction and its own `TLLM_CHECK`s raise `TllmException`, which is what the
/// entry below turns into a message.
int launch(uniserve_route_gemm_nvfp4_call* call, void* stream) {
  if (call == nullptr) {
    return 1;
  }
  std::memset(call->error, 0, sizeof(call->error));
  if (call == nullptr) {
    return 1;
  }
  std::memset(call->error, 0, sizeof(call->error));
  if (call->dtype != uniserve_dtype_bf16) {
    return refuse(call, "the product's output is compiled as bf16 and this module is the bf16 one");
  }
  if (check_geometry(call) != 0) {
    return refuse_geometry(call);
  }
  // The tiles the descriptor states are the ones the published cubins carry; a module
  // whose tactic is another row/column tile is another module, and this is the refusal
  // that says so rather than a launch into a shape mismatch.
  if (call->tile_m != 128 || call->tile_n <= 0 || call->cluster_m <= 0 || call->cluster_n <= 0) {
    return refuse(call, "the compiled tile is 128 rows by the descriptor's columns at a positive "
                        "cluster; the call states another");
  }

  auto const top_k = static_cast<int32_t>(call->top);
  auto const hidden = static_cast<int32_t>(call->hidden_size);
  auto const intermediate = static_cast<int32_t>(call->intermediate);
  auto const experts = static_cast<int32_t>(call->experts);
  auto const routes = static_cast<int32_t>(call->routes);

  Product& product = runner();
  int32_t const config = product.getDefaultValidConfigIndex(top_k, hidden, intermediate, experts,
                                                            routes);
  if (config < 0) {
    return refuse(call, "no compiled tactic serves this problem's shape");
  }
  // The arming: the artifact's own requirement for the chosen tactic, plus the two device
  // words this entry owns (the padded route count and the zeroed per-column bias). A
  // buffer shorter than that is refused with both numbers -- the plan sizes it, and a step
  // that grows one is a host allocation inside a captured graph.
  size_t const required = product.getWorkspaceSizeInBytes(top_k, hidden, intermediate, experts,
                                                          routes, config);
  // The scratch the entry owns: the padded route count (one i32), then the zeroed bias the
  // artifact's kernels take (one f32 per output column), then the tactic's own.
  size_t const bias_bytes = sizeof(float) * static_cast<size_t>(hidden);
  size_t const bias_at = 16;
  size_t const needed = bias_at + bias_bytes + required;
  if (static_cast<size_t>(call->workspace_bytes) < needed) {
    char message[512];
    std::snprintf(message, sizeof(message),
                  "the product's tactic needs %zu bytes of scratch and the step provided %lld: the "
                  "plan sizes this buffer, and the entry never grows one",
                  needed, static_cast<long long>(call->workspace_bytes));
    return refuse(call, message);
  }

  auto* seat = static_cast<unsigned char*>(call->workspace);
  auto* total_num_padded_tokens = reinterpret_cast<int32_t*>(seat);
  auto* bias = reinterpret_cast<float*>(seat + bias_at);
  void* bmm_workspace = static_cast<void*>(seat + bias_at + bias_bytes);
  cudaStream_t const order = reinterpret_cast<cudaStream_t>(stream);
  // The GEMM bounds its M by the padded route count, which is this call's own value: it is
  // written on the stream, before the product, so the ordering is the stream's.
  if (cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(total_num_padded_tokens),
                       static_cast<unsigned int>(call->routes), 1,
                       reinterpret_cast<CUstream>(order)) != CUDA_SUCCESS) {
    return refuse(call, "the padded route count could not be written on the call's stream");
  }

  // The bias is zero, on the same stream and before the product: the operator has none, and
  // the kernel's slot takes the additive identity rather than a value the model never
  // stated.
  if (cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(bias), 0,
                       static_cast<size_t>(bias_bytes / sizeof(float)),
                       reinterpret_cast<CUstream>(order)) != CUDA_SUCCESS) {
    return refuse(call, "the zeroed bias could not be written on the call's stream");
  }

  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return refuse(call, "the entry could not read the device the call runs on");
  }

  product.run(const_cast<void*>(call->hidden), const_cast<void*>(call->hidden_scale),
              const_cast<void*>(call->weights), const_cast<void*>(call->scales),
              const_cast<void*>(call->hidden_token_scale), /*perChannelScales=*/nullptr,
              const_cast<float*>(static_cast<float const*>(call->global_scale)), bias, call->out,
              /*outputScale=*/nullptr, top_k, hidden, intermediate, experts, routes,
              const_cast<int32_t*>(call->num_non_exiting_tiles),
              total_num_padded_tokens, const_cast<int32_t*>(call->tile_to_expert),
              const_cast<int32_t*>(call->tile_to_limit), bmm_workspace, device, order, config,
              /*enable_pdl=*/false);
  return 0;
}

/// The operator's entry: the launch, and the artifact's exceptions turned into messages.
///
/// An exception raised inside the artifact's runner would otherwise cross this C boundary,
/// where nothing may unwind: the process ends and the only message a log holds is the runtime's
/// own (`internal error: entered unreachable code`), which names neither the operator nor the
/// reason the artifact refused. A thrown `TllmException` states one, so it goes into the call's
/// error buffer and the launch answers a refusal like every other one this entry makes.
extern "C" int uniserve_route_gemm_nvfp4(uniserve_route_gemm_nvfp4_call* call, void* stream) {
  if (call == nullptr) {
    return 1;
  }
  try {
    return launch(call, stream);
  } catch (std::exception const& thrown) {
    return refuse(call, thrown.what());
  } catch (...) {
    return refuse(call, "the artifact's launch failed without stating why");
  }
}
