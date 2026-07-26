// zealot/src/attention/kernels/attention.cu
// Tiled FlashAttention with online softmax and optional causal mask.
// Q, K, V, O are interleaved: [num_heads * seq_len * head_dim] fp32.

#define Br 64
#define Bc 64
#define WARP 32

extern "C" __global__ void tiled_attention_kernel(
    const float* Q,     // [num_heads * seq_len * head_dim] interleaved
    const float* K,     // same layout
    const float* V,     // same layout
    float* O,           // same layout
    int seq_len,
    int num_heads,
    int head_dim,
    float scale,
    int causal          // bool (0 or 1), passed as int for C ABI
) {
    int head = blockIdx.x;
    int q_tile_idx = blockIdx.y;
    int q_start = q_tile_idx * Br;
    if (q_start >= seq_len) return;

    int q_actual = Br;
    if (q_start + Br > seq_len) q_actual = seq_len - q_start;

    // ── Shared memory ──
    extern __shared__ float smem[];
    float* Q_smem = smem;
    float* K_smem = Q_smem + Br * head_dim;
    float* V_smem = K_smem + Bc * head_dim;
    float* S_smem = V_smem + Bc * head_dim;  // [Br * Bc]

    // ── Per-row accumulators in registers ──
    // Each thread handles Br/blockDim.x Q rows (typically 1 for Br=64, BLOCK=256)
    int thread_rows = (Br + blockDim.x - 1) / blockDim.x;
    float m_i[8];   // max 8 rows per thread
    float l_i[8];

    // Init accumulators
    for (int r = 0; r < thread_rows; r++) {
        int q_row = r * blockDim.x + threadIdx.x;
        if (q_row < Br) {
            m_i[r] = -1e30f;
            l_i[r] = 0.0f;
        }
    }

    // ── Load Q tile ──
    for (int i = threadIdx.x; i < Br * head_dim; i += blockDim.x) {
        int row = i / head_dim;
        int col = i % head_dim;
        int global_row = q_start + row;
        if (global_row < seq_len) {
            Q_smem[i] = Q[head * seq_len * head_dim + global_row * head_dim + col];
        } else {
            Q_smem[i] = 0.0f;
        }
    }
    __syncthreads();

    // O accumulator in registers (per-thread, per-assigned-row).
    // For Br=64, BLOCK=256: each thread handles 1 row with up to 128 head_dim = 128 floats.
    float O_reg[256];  // fixed size; access O_reg[q_local * head_dim + d]
    int my_rows = (Br + blockDim.x - 1) / blockDim.x;
    for (int r = 0; r < my_rows; r++) {
        int q_row = r * blockDim.x + threadIdx.x;
        if (q_row < Br) {
            for (int d = 0; d < head_dim; d++) {
                O_reg[r * head_dim + d] = 0.0f;
            }
        }
    }

    // ── Main loop over K/V tiles ──
    int num_k_tiles = (seq_len + Bc - 1) / Bc;
    for (int k_tile_idx = 0; k_tile_idx < num_k_tiles; k_tile_idx++) {
        int k_start = k_tile_idx * Bc;
        int k_actual = Bc;
        if (k_start + Bc > seq_len) k_actual = seq_len - k_start;

        // Causal early-skip
        int q_end = q_start + q_actual - 1;
        if (causal && k_start > q_end) continue;

        // Load K_tile and V_tile
        for (int i = threadIdx.x; i < Bc * head_dim; i += blockDim.x) {
            int row = i / head_dim;
            int col = i % head_dim;
            int global_row = k_start + row;
            if (global_row < seq_len) {
                K_smem[i] = K[head * seq_len * head_dim + global_row * head_dim + col];
                V_smem[i] = V[head * seq_len * head_dim + global_row * head_dim + col];
            } else {
                K_smem[i] = 0.0f;
                V_smem[i] = 0.0f;
            }
        }
        __syncthreads();

        // Compute S = Q @ K^T (thread-parallel over S elements)
        for (int i = threadIdx.x; i < Br * Bc; i += blockDim.x) {
            int q_row = i / Bc;
            int k_row = i % Bc;
            if (q_row >= q_actual || k_row >= k_actual) {
                S_smem[i] = -1e30f;
                continue;
            }
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += Q_smem[q_row * head_dim + d] * K_smem[k_row * head_dim + d];
            }
            S_smem[i] = dot * scale;

            // Causal mask per-element
            int q_pos = q_start + q_row;
            int k_pos = k_start + k_row;
            if (causal && k_pos > q_pos) {
                S_smem[i] = -1e30f;
            }
        }
        __syncthreads();

        // Online softmax update
        for (int r = 0; r < thread_rows; r++) {
            int q_row = r * blockDim.x + threadIdx.x;
            if (q_row >= q_actual) continue;

            // Find max in this tile's row
            float m_old = m_i[r];
            float m_new = -1e30f;
            for (int j = 0; j < k_actual; j++) {
                float s = S_smem[q_row * Bc + j];
                if (s > m_new) m_new = s;
            }

            // Compute P = exp(S - m_new) and row sum
            float l_new = 0.0f;
            for (int j = 0; j < k_actual; j++) {
                float p = expf(S_smem[q_row * Bc + j] - m_new);
                S_smem[q_row * Bc + j] = p;  // reuse S_smem for P
                l_new += p;
            }

            // Rescale O accumulator
            float alpha = expf(m_old - m_new);
            for (int d = 0; d < head_dim; d++) {
                O_reg[r * head_dim + d] *= alpha;
            }

            // Accumulate P @ V
            for (int j = 0; j < k_actual; j++) {
                float p = S_smem[q_row * Bc + j];
                for (int d = 0; d < head_dim; d++) {
                    O_reg[r * head_dim + d] += p * V_smem[j * head_dim + d];
                }
            }

            l_i[r] = l_i[r] * alpha + l_new;
            m_i[r] = m_new;
        }
        __syncthreads();
    }

    // ── Final normalize and write back ──
    // Store l_i from registers to shared memory for cross-thread access
    __shared__ float L_smem[Br];
    for (int r = 0; r < my_rows; r++) {
        int q_row = r * blockDim.x + threadIdx.x;
        if (q_row < q_actual) {
            L_smem[q_row] = l_i[r];
        }
    }
    __syncthreads();

    // Per-thread write-back: each thread writes its own assigned rows
    for (int r = 0; r < my_rows; r++) {
        int q_row = r * blockDim.x + threadIdx.x;
        if (q_row >= q_actual) continue;
        float inv_l = 1.0f / L_smem[q_row];
        for (int d = 0; d < head_dim; d++) {
            O[head * seq_len * head_dim + (q_start + q_row) * head_dim + d] =
                O_reg[r * head_dim + d] * inv_l;
        }
    }
}
