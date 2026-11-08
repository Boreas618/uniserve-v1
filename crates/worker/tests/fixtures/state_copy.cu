// The state block copy's two directions, as the probe drives them: one row of the snapshot per
// block index, `width` bytes of a block's field. The kernels take the operand order
// (destination, index vector, source), so a probe reads its launch as what it does.
//
// `blocks` names field blocks, one per snapshot row: a negative index writes nothing (the inactive
// pair's placeholder), the same reading `ops::block_copy` gives it.

extern "C" __global__ void state_snapshot(
    char* snapshot,
    const long long* blocks,
    const char* field,
    int rows,
    int width)
{
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    long long source = blocks[row];
    if (source < 0) {
        return;
    }
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        snapshot[(long long)row * width + i] = field[source * width + i];
    }
}

extern "C" __global__ void state_scatter(
    char* field,
    const long long* blocks,
    const char* snapshot,
    int rows,
    int width)
{
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    long long target = blocks[row];
    if (target < 0) {
        return;
    }
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        field[target * width + i] = snapshot[(long long)row * width + i];
    }
}
