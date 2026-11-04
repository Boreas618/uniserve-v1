// A block's reduction of one value, as PyTorch's CUDA group norm carries it.
//
// This is `at::native::cuda_utils::WarpReduce` and `BlockReduce` from
// `ATen/native/cuda/block_reduce.cuh` with the two constants that fix their tree —
// the warp size and the 512-thread block the reductions launch with — and no c10
// headers. The tree is the point: a warp reduces by halving its lanes, and a block
// reduces the warps' first lanes by the same halving, so the reference's mean and
// second moment associate exactly as the module that carried this code did.

#ifndef UNISERVE_ATEN_BLOCK_REDUCE_CUH
#define UNISERVE_ATEN_BLOCK_REDUCE_CUH

#include "aten_welford.cuh"

namespace uniserve {

/// PyTorch's warp size, and the thread count its block reductions launch with.
constexpr int kCudaWarpSize = 32;
constexpr int kCudaBlockReduceNumThreads = 512;
/// The largest block whose warps still fit one warp's reduction.
constexpr int kCudaWarpSizeUpperBound = kCudaBlockReduceNumThreads / kCudaWarpSize;

/// Reduces one value across a warp's lanes.
template <typename T, typename ReduceOp>
__inline__ __device__ T warp_reduce(T val, const ReduceOp& op) {
#pragma unroll
  for (int offset = (kCudaWarpSize >> 1); offset > 0; offset >>= 1) {
    val = op.combine(val, op.warp_shfl_down(val, offset));
  }
  return val;
}

/// Reduces one value across a block's threads; valid in thread 0.
///
/// The block must hold `kCudaWarpSize` threads or a multiple of them, and `shared` must hold one
/// value per warp.
template <typename T, typename ReduceOp>
__inline__ __device__ T block_reduce(T val, const ReduceOp& op, T identity_element, T* shared) {
  const int tid = threadIdx.x + threadIdx.y * blockDim.x;
  const int lid = tid % kCudaWarpSize;
  const int wid = tid / kCudaWarpSize;
  const int warps = (blockDim.x * blockDim.y) / kCudaWarpSize;
  val = warp_reduce(val, op);
  __syncthreads();  // prevent races when the block reduces twice in a row
  if (lid == 0) {
    shared[wid] = val;
  }
  __syncthreads();
  val = (tid < warps) ? shared[lid] : identity_element;
  if (wid == 0) {
    val = warp_reduce(val, op);
  }
  return val;
}

}  // namespace uniserve

#endif  // UNISERVE_ATEN_BLOCK_REDUCE_CUH
