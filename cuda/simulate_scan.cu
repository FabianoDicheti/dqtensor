extern "C" __global__
void simulate_scan(
    const float* A,  // [L, N, N]
    const float* B,  // [L, N]
    const float* x,  // [L]
    float* h,        // [L, N]
    int L,
    int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= L) return;

    for (int j = 0; j < N; ++j) {
        float sum = 0.0;
        for (int k = 0; k < N; ++k) {
            if (i > 0)
                sum += A[i * N * N + j * N + k] * h[(i - 1) * N + k];
        }
        sum += B[i * N + j] * x[i];
        h[i * N + j] = sum;
    }
}