// zealot/src/attention/kernels/embedding.cu
// Embedding gather: out[token_idx][dim_offset:] = embedding_table[token_ids[token_idx]][dim_offset:]

extern "C" __global__ void embedding_gather_kernel(
    const long long* token_ids,    // [seq_len]
    const float* embedding_table,  // [vocab_size, hidden_dim]
    float* out,                    // [seq_len, hidden_dim]
    int seq_len,
    int hidden_dim
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int dim_offset = blockIdx.y * blockDim.y;
    if (token_idx >= seq_len) return;

    long long token = token_ids[token_idx];
    const float* src = embedding_table + token * hidden_dim + dim_offset;
    float* dst = out + token_idx * hidden_dim + dim_offset;

    for (int i = 0; i < blockDim.y && (dim_offset + i) < hidden_dim; i++) {
        dst[i] = src[i];
    }
}
