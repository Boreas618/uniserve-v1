// Two generations of one kernel: the same launch shape, different arithmetic.
//
// The probe of design section 12's open question captures one graph per
// generation over the same activation buffers and replays them alternately, so
// the arithmetic has to differ: generation 0 adds `value` once, generation 1
// adds it twice, and a replay of the wrong executable is visible in the result.

extern "C" __global__ void generation_0(unsigned int *data, int count, unsigned int value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        data[i] = data[i] + value;
    }
}

extern "C" __global__ void generation_1(unsigned int *data, int count, unsigned int value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        data[i] = data[i] + 2u * value;
    }
}
