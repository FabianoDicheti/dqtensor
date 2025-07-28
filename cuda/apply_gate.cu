__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

extern "C" __global__
void apply_gate(
    const float* gate,
    const float* x,
    const float* y,
    float* out,
    int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float g = sigmoid(gate[i]);
        out[i] = g * x[i] + (1.0f - g) * y[i];
    }
}