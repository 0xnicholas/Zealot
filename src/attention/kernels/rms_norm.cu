// zealot/src/attention/kernels/rms_norm.cu
// RMS Norm: out[i] = x[i] * rsqrt(mean(x^2) + eps) * weight

extern "C" __global__ void rms_norm_kernel(
    const float* x,       // [rows, dim]
    const float* weight,  // [dim]
    float* out,           // [rows, dim]
    int rows,
    int dim,
    float eps
) {
    __shared__ float warp_sums[32];  // max 32 warps

    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x_row = x + row * dim;
    float* out_row = out + row * dim;

    // Compute sum of squares via warp reduction
    float var = 0.0f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = x_row[i];
        var += v * v;
    }

    // Warp-level reduction, then cross-warp aggregation via shared memory
    for (int offset = 16; offset > 0; offset >>= 1) {
        var += __shfl_down_sync(0xffffffff, var, offset);
    }
    // Write warp sums to shared memory
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) {
        warp_sums[warp_id] = var;
    }
    __syncthreads();
    // Warp 0 lane 0 aggregates warp sums (no redundant second shuffle)
    if (warp_id == 0 && lane_id == 0) {
        var = 0.0f;
        int num_warps = (blockDim.x + 31) / 32;
        for (int w = 0; w < num_warps; w++) {
            var += warp_sums[w];
        }
        warp_sums[0] = var;
    }
    __syncthreads();
    var = warp_sums[0];

    float rms = rsqrtf(var / (float)dim + eps);

    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        out_row[i] = x_row[i] * rms * weight[i];
    }
}
