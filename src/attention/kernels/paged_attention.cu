// zealot/src/attention/kernels/paged_attention.cu
// PagedAttention: online-softmax attention over KV cache pages.
// One CUDA block per (seq, head) pair.  Q is a single vector per (seq, head)
// — this is the decode-phase kernel (one query token at the end of the
// sequence attending to all cached K/V blocks).
//
// Supports GQA: kv_head = head / (num_heads / num_kv_heads).
// Sentinel: block_table entry < 0 signals end-of-sequence.

#define WARP 32

extern "C" __global__ void paged_attention_kernel(
    const float* Q,            // [num_seqs, num_heads, head_dim]
    const float* K_pool,       // flat K cache: [num_blocks * block_size * num_kv_heads * head_dim]
    const float* V_pool,       // flat V cache (same layout as K_pool)
    float* O,                  // [num_seqs, num_heads, head_dim]
    const int* block_table,    // [num_seqs, max_blocks_per_seq] — sentinel < 0
    const int* seq_lens,       // [num_seqs]
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_blocks_per_seq,
    int block_size,
    float scale
) {
    int seq = blockIdx.x;
    int head = blockIdx.y;
    if (seq >= num_seqs || head >= num_heads) return;

    int kv_head = head / (num_heads / num_kv_heads);
    int seq_len = seq_lens[seq];
    int num_tiles = (seq_len + block_size - 1) / block_size;

    // ── Shared memory ─────────────────────────────────────────────────
    extern __shared__ float smem[];
    float* K_smem = smem;
    float* V_smem = K_smem + block_size * head_dim;
    float* O_smem = V_smem + block_size * head_dim;  // head_dim floats

    // ── Q in registers (thread 0) ─────────────────────────────────────
    float Q_reg[128];
    if (threadIdx.x == 0) {
        int q_base = seq * num_heads * head_dim + head * head_dim;
        for (int d = 0; d < head_dim; d++) {
            Q_reg[d] = Q[q_base + d];
        }
    }

    // ── Online softmax accumulators (thread 0) ────────────────────────
    float m = -1e30f;
    float l = 0.0f;

    // Init O_smem = 0
    if (threadIdx.x == 0) {
        for (int d = 0; d < head_dim; d++) {
            O_smem[d] = 0.0f;
        }
    }
    __syncthreads();

    int kv_block_stride = block_size * num_kv_heads * head_dim;

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int block_id = block_table[seq * max_blocks_per_seq + tile_idx];
        if (block_id < 0) break;

        int n_tokens = block_size;
        if (tile_idx == num_tiles - 1) {
            int remaining = seq_len - tile_idx * block_size;
            if (remaining < block_size) n_tokens = remaining;
        }

        // ── Load K and V tile into shared memory ──────────────────
        int block_base = block_id * kv_block_stride + kv_head * block_size * head_dim;
        for (int i = threadIdx.x; i < n_tokens * head_dim; i += blockDim.x) {
            int token = i / head_dim;
            int dim = i % head_dim;
            int off = block_base + token * head_dim + dim;
            K_smem[token * head_dim + dim] = K_pool[off];
            V_smem[token * head_dim + dim] = V_pool[off];
        }
        __syncthreads();

        // ── Thread 0: compute S = Q @ K^T, online softmax ─────────
        if (threadIdx.x == 0) {
            // Compute S = Q @ K^T → n_tokens scores
            float S[16];  // max block_size
            for (int t = 0; t < n_tokens; t++) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    dot += Q_reg[d] * K_smem[t * head_dim + d];
                }
                S[t] = dot * scale;
            }

            // Find max
            float m_new = m;
            for (int t = 0; t < n_tokens; t++) {
                if (S[t] > m_new) m_new = S[t];
            }

            // Exp + softmax denominator update
            float l_new = 0.0f;
            float P[16];
            for (int t = 0; t < n_tokens; t++) {
                float p = expf(S[t] - m_new);
                P[t] = p;
                l_new += p;
            }

            // Rescale O accumulator
            float alpha = expf(m - m_new);
            for (int d = 0; d < head_dim; d++) {
                O_smem[d] *= alpha;
            }

            // Accumulate P @ V
            for (int t = 0; t < n_tokens; t++) {
                float p = P[t];
                for (int d = 0; d < head_dim; d++) {
                    O_smem[d] += p * V_smem[t * head_dim + d];
                }
            }

            l = l * alpha + l_new;
            m = m_new;
        }
        __syncthreads();
    }

    // ── Normalize and write O ─────────────────────────────────────────
    if (threadIdx.x == 0) {
        float inv_l = 1.0f / l;
        int o_base = seq * num_heads * head_dim + head * head_dim;
        for (int d = 0; d < head_dim; d++) {
            O[o_base + d] = O_smem[d] * inv_l;
        }
    }
}
