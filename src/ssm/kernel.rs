use super::parametric::DynamicParamGenerator;

/// Gera o kernel dinâmico por tempo: K[t][k]
pub fn generate_dynamic_kernel(
    param_gen: &DynamicParamGenerator,
    inputs: &Vec<Vec<f64>>,   // sequência: x_0, x_1, ..., x_T-1
    C: &Vec<f64>
) -> Vec<Vec<f64>> {
    let T = inputs.len();
    let d = C.len();
    let mut kernels = vec![vec![0.0; T]; T];

    for t in 0..T {
        let x_t = &inputs[t];
        let (A_t, B_t, Delta_t) = param_gen.generate(x_t);

        let mut A_bar = vec![0.0; d];
        let mut B_bar = vec![0.0; d];

        for i in 0..d {
            let a_dt = A_t[i] * Delta_t[i];
            A_bar[i] = a_dt.exp();

            if A_t[i].abs() > 1e-8 {
                B_bar[i] = ((A_bar[i] - 1.0) / A_t[i]) * B_t[i];
            } else {
                B_bar[i] = B_t[i] * Delta_t[i];
            }
        }

        for k in 0..=t {
            let mut sum = 0.0;
            for i in 0..d {
                sum += C[i] * A_bar[i].powi(k as i32) * B_bar[i];
            }
            kernels[t][k] = sum;
        }
    }

    kernels
}
