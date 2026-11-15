// The kernel of the library-load tests, which compile it with NVRTC for the
// compute capability of the device they load it on: the cubin NVRTC produces
// is a binary image, NUL bytes included, which the driver reads by its own
// headers rather than as PTX text.
extern "C" __global__ void add_constant(unsigned int *out, unsigned int value) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    out[index] += value;
}
