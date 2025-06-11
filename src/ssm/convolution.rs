/// Gera o kernel causal de convolução K para um SSM diagonal
pub fn generate_kernel(
    A: &Vec<f64>,
    B: &Vec<f64>,
    C: &Vec<f64>,
    length: usize
) -> Vec<f64> {
    let d = A.len();
    let mut kernel = vec![0.0; length];

    for k in 0..length {
        let mut sum = 0.0;
        for i in 0..d {
            let Ak = A[i].powi(k as i32);  // A^k
            sum += C[i] * Ak * B[i];       // K_k[i]
        }
        kernel[k] = sum;
    }

    kernel
}


/// Aplica convolução causal com kernel dinâmico por tempo:
/// y[t] = sum_k K[t][k] * u[t-k]
pub fn dynamic_causal_convolution(
    kernels: &Vec<Vec<f64>>,  // K[t][k]
    inputs: &Vec<f64>         // u[t]
) -> Vec<f64> {
    let T = inputs.len();
    let mut outputs = vec![0.0; T];

    for t in 0..T {
        let mut acc = 0.0;
        for k in 0..=t {
            acc += kernels[t][k] * inputs[t - k];
        }
        outputs[t] = acc;
    }

    outputs
}
