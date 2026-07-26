// zealot/src/attention/kernels/activation.cu
// Activation kernels: SiLU gate-mul and element-wise add.

extern "C" __global__ void silu_gate_mul_kernel(
    const float* gate, const float* up, float* out,
    int n, int inter_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * inter_dim) return;
    float g = gate[idx];
    out[idx] = (g / (1.0f + expf(-g))) * up[idx];
}

extern "C" __global__ void add_kernel(
    const float* a, const float* b, float* out, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = a[idx] + b[idx];
}
