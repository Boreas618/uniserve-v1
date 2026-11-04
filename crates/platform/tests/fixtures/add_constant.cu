// Source of add_constant.fatbin, which the library-load test loads: a binary
// image contains NUL bytes and so could not pass the CString path the crate
// used for PTX text.
//
// Regenerate it with the CUDA toolkit of the branch's target (CUDA 13):
//
//     nvcc -gencode arch=compute_90,code=[sm_90,compute_90] \
//         -fatbin add_constant.cu -o add_constant.fatbin
//
// The embedded PTX makes the image load on every device from sm_90 on, and the
// SASS section keeps it honest for sm_90 itself.
extern "C" __global__ void add_constant(unsigned int *out, unsigned int value) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    out[index] += value;
}
