extern "C" __global__
void ssm_kernel_conv(
    const float* input,     // [B, L]
    const float* kernel,    // [K]
    float* output,          // [B, L - K + 1]
    int B, int L, int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * (L - K + 1);
    if (idx >= total) return;

    int b = idx / (L - K + 1);
    int t = idx % (L - K + 1);

    float acc = 0.0f;
    for (int i = 0; i < K; ++i) {
        acc += kernel[i] * input[b * L + t + i];
    }

    output[b * (L - K + 1) + t] = acc;
}