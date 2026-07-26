// zealot/src/attention/kernels/transpose.cu
// QKV transpose: convert between interleaved [num_heads, seq_len, head_dim]
// and contiguous [seq_len, num_heads, head_dim] layouts.
//
// direction=0: interleaved → contiguous (QKV projection output → attention input)
// direction=1: contiguous → interleaved (attention output → projection input)

extern "C" __global__ void qkv_transpose_kernel(
    const float* src,
    float* dst,
    int seq_len,
    int num_heads,
    int head_dim,
    int direction
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * num_heads * head_dim;
    if (idx >= total) return;

    int dim = idx % head_dim;
    int remaining = idx / head_dim;
    int pos = remaining % seq_len;
    int head = remaining / seq_len;

    if (direction == 0) {
        // src: [seq_len, num_heads, head_dim] → dst: [num_heads, seq_len, head_dim]
        dst[head * seq_len * head_dim + pos * head_dim + dim] =
            src[pos * num_heads * head_dim + head * head_dim + dim];
    } else {
        // src: [num_heads, seq_len, head_dim] → dst: [seq_len, num_heads, head_dim]
        dst[pos * num_heads * head_dim + head * head_dim + dim] =
            src[head * seq_len * head_dim + pos * head_dim + dim];
    }
}
