// Welford's accumulation, as PyTorch's CUDA group norm carries it.
//
// This is `at::native::WelfordData` and `at::native::WelfordOps` from
// `ATen/native/SharedReduceOps.h`, device parts only and under this module's own
// namespace: the reference kernel reuses them so its mean and its second central
// moment associate exactly as `torch.var_mean` does, and a module that the Program
// Server `dlopen`s cannot link torch. The arithmetic is unchanged, operation for
// operation, because the point of carrying it is the order of its additions.

#ifndef UNISERVE_ATEN_WELFORD_CUH
#define UNISERVE_ATEN_WELFORD_CUH

#include <cstdint>

namespace uniserve {

#define UNISERVE_HOST_DEVICE __host__ __device__

/// A projection's two values: the variance and the mean. PyTorch names it
/// `std::pair<float, float>`, which is a host type and cannot be returned from a
/// device function; this carries the same two values in the same order.
struct variance_and_mean {
  float variance;
  float mean;
};

/// One warp's exchange of a value with the lane `delta` above it. The name is not
/// `warp_shfl_down`: that is a member of `welford_ops`, which would hide the free function
/// inside it.
template <typename T>
__device__ __forceinline__ T lane_shfl_down(T value, unsigned int delta, int width = 32,
                                            unsigned int mask = 0xffffffff) {
  return __shfl_down_sync(mask, value, delta, width);
}

/// A 64-bit lane count travels as its two halves.
template <>
__device__ __forceinline__ int64_t lane_shfl_down<int64_t>(int64_t value, unsigned int delta,
                                                           int width, unsigned int mask) {
  int low = __shfl_down_sync(mask, static_cast<int>(value), delta, width);
  int high = __shfl_down_sync(mask, static_cast<int>(value >> 32), delta, width);
  return static_cast<int64_t>((static_cast<uint64_t>(static_cast<uint32_t>(high)) << 32) |
                              static_cast<uint32_t>(low));
}

/// The count, the mean and the sum of squared deviations of a stream of values.
template <typename scalar_t, typename index_t>
struct welford_data {
  scalar_t mean;
  scalar_t m2;
  index_t n;
  scalar_t nf;

  UNISERVE_HOST_DEVICE welford_data() : mean(0), m2(0), n(0), nf(0) {}

  UNISERVE_HOST_DEVICE welford_data(scalar_t mean, scalar_t m2, index_t n, scalar_t nf)
      : mean(mean), m2(m2), n(n), nf(nf) {}
};

/// The reduce/combine/project triple PyTorch's reductions are parameterised by.
template <typename scalar_t, typename acc_scalar_t, typename index_t, typename res_t>
struct welford_ops {
  acc_scalar_t correction;
  bool take_sqrt;

  using acc_t = welford_data<acc_scalar_t, index_t>;

  UNISERVE_HOST_DEVICE welford_ops(acc_scalar_t correction, bool take_sqrt)
      : correction(correction), take_sqrt(take_sqrt) {}

  inline __device__ acc_t reduce(acc_t acc, scalar_t data, index_t /*idx*/) const {
    // The count accumulates in `index_t` so it cannot lose its last bits to rounding, while `nf`
    // carries it as a float for the combine, where an int32 could overflow.
    index_t new_n = acc.n + 1;
    acc_scalar_t new_nf = static_cast<acc_scalar_t>(new_n);
    acc_scalar_t delta = data - acc.mean;
    acc_scalar_t new_mean = acc.mean + delta / new_nf;
    acc_scalar_t new_delta = data - new_mean;
    return {new_mean, acc.m2 + delta * new_delta, new_n, new_nf};
  }

  inline __device__ acc_t combine(acc_t a, acc_t b) const {
    if (a.nf == 0) {
      return b;
    }
    if (b.nf == 0) {
      return a;
    }
    acc_scalar_t delta = b.mean - a.mean;
    acc_scalar_t new_count = a.nf + b.nf;
    acc_scalar_t nb_over_n = b.nf / new_count;
    // The count is -1 after a combine: `n` cannot represent the merged count, and nothing reads it
    // once `nf` carries it.
    return {a.mean + delta * nb_over_n, a.m2 + b.m2 + delta * delta * a.nf * nb_over_n, -1,
            new_count};
  }

  inline __device__ res_t project(acc_t acc) const {
    const auto mean = static_cast<scalar_t>(acc.mean);
    const auto divisor = acc.nf > correction ? acc.nf - correction : 0;
    const auto var = acc.m2 / divisor;
    return res_t{take_sqrt ? device_sqrt(var) : var, mean};
  }

  static inline __device__ acc_t translate_idx(acc_t acc, int64_t /*base_idx*/) { return acc; }

  inline __device__ acc_t warp_shfl_down(acc_t acc, int offset) const {
    return {uniserve::lane_shfl_down(acc.mean, offset), uniserve::lane_shfl_down(acc.m2, offset),
            uniserve::lane_shfl_down(acc.n, offset), uniserve::lane_shfl_down(acc.nf, offset)};
  }

 private:
  /// `sqrt` of an accumulator, as `c10::cuda::compat::sqrt` resolves it.
  static inline __device__ acc_scalar_t device_sqrt(acc_scalar_t value) { return ::sqrt(value); }
};

}  // namespace uniserve

#endif  // UNISERVE_ATEN_WELFORD_CUH
