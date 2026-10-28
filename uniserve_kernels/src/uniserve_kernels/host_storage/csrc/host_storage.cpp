// SPDX-License-Identifier: Apache-2.0

// Lifetime tracking of PyTorch pinned host storage that native code reads
// with its own asynchronous DMA. The Python package
// uniserve_kernels.host_storage compiles this file on first use.

#include <torch/extension.h>
#include <ATen/core/CachingHostAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>

namespace {

// Records `stream` on the PyTorch pinned host allocator block backing `tensor`,
// so the allocator does not reuse the block until work already enqueued on the
// stream completes. Fails unless `tensor` is pinned host storage from that
// allocator.
void record_host_usage(torch::Tensor tensor, int64_t device, uint64_t stream_handle) {
  TORCH_CHECK(tensor.is_cpu() && tensor.is_pinned(),
              "native DMA lifetime tracking requires pinned host storage");
  const c10::cuda::CUDAGuard guard(static_cast<c10::DeviceIndex>(device));
  const auto stream = c10::cuda::getStreamFromExternal(
      reinterpret_cast<cudaStream_t>(stream_handle), static_cast<c10::DeviceIndex>(device));
  const auto& allocation = tensor.storage().data_ptr();
  TORCH_CHECK(at::getHostAllocator(at::kCUDA)->record_event(
                  tensor.data_ptr(), allocation.get_context(), stream.unwrap()),
              "native DMA source must belong to the PyTorch pinned allocator");
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, binding) {
  binding.def("record_host_usage", &record_host_usage);
}
