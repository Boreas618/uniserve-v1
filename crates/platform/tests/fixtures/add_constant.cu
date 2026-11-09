// The kernel of the library-load tests, which compile it to a fatbin with the
// toolkit CUDA_HOME names: a binary image contains NUL bytes and so could not
// pass the CString path the crate used for PTX text.
//
// The tests compile it as
//
//     $CUDA_HOME/bin/nvcc -gencode arch=compute_90,code=[sm_90,compute_90] \
//         -fatbin add_constant.cu -o <their own directory>/add_constant.fatbin
//
// The embedded PTX makes the image load on every device from sm_90 on, and the
// SASS section keeps it honest for sm_90 itself.
extern "C" __global__ void add_constant(unsigned int *out, unsigned int value) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    out[index] += value;
}
