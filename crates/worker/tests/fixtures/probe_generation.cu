// Three program generations of one operator, and the deviation the test reads.
//
// The operator adds `value` to each word of the activation, in the word's own integer domain.
// `probe_exact` and `probe_equivalent` compute the same number - the same addition with the
// operands in the other order - so a paired forward of the two must agree exactly, which is what
// this probe's `Op::tolerance` states (atol 0, rtol 0: an integer addition has no rounding).
//
// `probe_wrong` adds one more than either, so the paired forward's two results differ by exactly
// 1 in every element: the test asserts the refusal names element 0 with reference 14 and candidate
// 15, which is 7 + 7 and 7 + 7 + 1 for the value 7 the test passes in. The deviation is stated here
// and read from the kernel's own arithmetic, never fitted to a tolerance chosen after a failure.

extern "C" __global__ void probe_exact(unsigned int *data, unsigned int count, unsigned int value) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        data[i] = data[i] + value;
    }
}

extern "C" __global__ void probe_equivalent(unsigned int *data, unsigned int count,
                                            unsigned int value) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        data[i] = value + data[i];
    }
}

extern "C" __global__ void probe_wrong(unsigned int *data, unsigned int count, unsigned int value) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        data[i] = data[i] + value + 1u;
    }
}
