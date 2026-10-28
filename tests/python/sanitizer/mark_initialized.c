/* Mark device ranges initialized for compute-sanitizer's initcheck through
 * the NVTX memory API (nvtxMemCudaMarkInitialized): a diagnostic that
 * stands in for writes initcheck does not record. Outside the sanitizer
 * the NVTX calls are no-ops.
 *
 *   int mark_initialized(void *stream, size_t count,
 *                        const uintptr_t *pointers, const size_t *sizes)
 */
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExtMemCudaRt.h>

static nvtxDomainHandle_t domain;

int mark_initialized(void *stream, size_t count, const uintptr_t *pointers,
                     const size_t *sizes) {
  if (!domain) domain = nvtxDomainCreateA("uniserve-initcheck");
  nvtxMemVirtualRangeDesc_t *ranges = calloc(count, sizeof(*ranges));
  if (!ranges) return -1;
  for (size_t i = 0; i < count; ++i) {
    ranges[i].ptr = (void *)pointers[i];
    ranges[i].size = sizes[i];
  }
  nvtxMemMarkInitializedBatch_t batch = {0};
  batch.extCompatID = NVTX_EXT_COMPATID_MEM;
  batch.structSize = sizeof(batch);
  batch.regionType = NVTX_MEM_TYPE_VIRTUAL_ADDRESS;
  batch.regionDescCount = count;
  batch.regionDescElementSize = sizeof(*ranges);
  batch.regionDescElements = ranges;
  nvtxMemCudaMarkInitialized(domain, (cudaStream_t)stream, 0, &batch);
  free(ranges);
  return 0;
}
