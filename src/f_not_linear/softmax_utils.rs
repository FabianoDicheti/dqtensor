/// Aplica softmax no vetor de entrada
pub fn softmax(input: &Vec<f64>) -> Vec<f64> {
    let max = input.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = input.iter().map(|&x| (x - max).exp()).collect();
    let sum_exps: f64 = exps.iter().sum();
    exps.iter().map(|&x| x / sum_exps).collect()
}

/// Calcula a Jacobiana da softmax
/// Retorna uma matriz NxN onde N é o tamanho do vetor de entrada
/// Jacobiana[i][j] = Softmax[i] * (δ[i==j] - Softmax[j])
pub fn softmax_jacobian(softmax_output: &Vec<f64>) -> Vec<Vec<f64>> {
    let n = softmax_output.len();
    let mut jacobian = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                jacobian[i][j] = softmax_output[i] * (1.0 - softmax_output[i]);
            } else {
                jacobian[i][j] = -softmax_output[i] * softmax_output[j];
            }
        }
    }

    jacobian
}
