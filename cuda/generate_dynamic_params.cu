__device__ float softplus(float x) {
    return logf(1.0f + expf(x));
}

extern "C" __global__
void generate_dynamic_params(
    const float* x,
    const float* Wb, const float* bb,
    const float* Wc, const float* bc,
    const float* Wd, const float* bd,
    float* out_b,
    float* out_c,
    float* out_delta,
    int D_in, int D_out
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < D_out) {
        float b_val = bb[i];
        float c_val = bc[i];
        float d_val = bd[i];
        for (int j = 0; j < D_in; ++j) {
            b_val += Wb[i * D_in + j] * x[j];
            c_val += Wc[i * D_in + j] * x[j];
            d_val += Wd[i * D_in + j] * x[j];
        }
        out_b[i] = b_val;
        out_c[i] = c_val;
        out_delta[i] = softplus(d_val);
    }
}
