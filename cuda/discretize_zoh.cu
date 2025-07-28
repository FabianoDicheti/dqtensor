extern "C" __global__
void discretize_zoh(
    const float* A, const float* B,
    float* A_out, float* B_out,
    float delta, int N
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        for (int j = 0; j < N; ++j) {
            int idx = i * N + j;

            // 1st-order approximation of exp(ΔA): I + ΔA + Δ²A²/2
            float a_val = (i == j ? 1.0f : 0.0f) + delta * A[idx];

            // Optionally include second-order
            float second_order = 0.0f;
            for (int k = 0; k < N; ++k) {
                second_order += A[i * N + k] * A[k * N + j];
            }
            a_val += (delta * delta * 0.5f) * second_order;
            A_out[idx] = a_val;
        }

        // Discretize B: B_d = Δ * B (simple for now)
        B_out[i] = delta * B[i];
    }
}