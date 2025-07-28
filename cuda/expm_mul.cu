// Versão simplificada: aplica (I + ΔA + (ΔA)^2 / 2!) h0

extern "C" __global__
void expm_mul(const float* A, const float* h, float* out, float delta, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float acc = h[i];  // I * h
        float temp = 0.0;

        // 1st order: ΔA * h
        for (int j = 0; j < N; ++j) {
            temp += A[i * N + j] * h[j];
        }
        acc += delta * temp;

        // 2nd order: ΔA^2 * h / 2
        float second_order = 0.0;
        for (int j = 0; j < N; ++j) {
            float inner = 0.0;
            for (int k = 0; k < N; ++k) {
                inner += A[i * N + k] * A[k * N + j];
            }
            second_order += inner * h[j];
        }
        acc += (delta * delta * 0.5) * second_order;

        out[i] = acc;
    }
}