pub fn cross_entropy(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let epsilon = 1e-15; // Evita log(0)
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| {
            let p = p.clamp(epsilon, 1.0 - epsilon);
            -t * p.ln()
        })
        .sum()
}

pub fn cross_entropy_derivative(y_true: &[f64], y_pred: &[f64]) -> Vec<f64> {
    let epsilon = 1e-15; // Protege contra divisão por zero
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| {
            let p = p.clamp(epsilon, 1.0 - epsilon);
            -t / p
        })
        .collect()
}

pub fn mse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum::<f64>() / y_true.len() as f64
}

pub fn mse_derivative(y_true: &[f64], y_pred: &[f64]) -> Vec<f64> {
    let n = y_true.len() as f64;
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| 2.0 * (p - t) / n)
        .collect()
}
