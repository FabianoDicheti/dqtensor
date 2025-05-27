/// Calcula o Softmax de um vetor de entrada
pub fn softmax(inputs: &[f64]) -> Vec<f64> {
    let max = inputs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = inputs.iter().map(|&x| (x - max).exp()).collect();
    let sum_exps: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum_exps).collect()
}

/// Retorna a matriz Jacobiana do vetor softmax
pub fn softmax_jacobian(softmax_output: &[f64]) -> Vec<Vec<f64>> {
    let len = softmax_output.len();
    let mut jacobian = vec![vec![0.0; len]; len];

    for i in 0..len {
        for j in 0..len {
            if i == j {
                jacobian[i][j] = softmax_output[i] * (1.0 - softmax_output[i]);
            } else {
                jacobian[i][j] = -softmax_output[i] * softmax_output[j];
            }
        }
    }
    jacobian
}
