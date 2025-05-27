pub fn mse_derivative(predictions: &[f64], targets: &[f64]) -> Vec<f64> {
    let n = predictions.len() as f64;
    predictions.iter()
        .zip(targets.iter())
        .map(|(&p, &t)| 2.0 * (p - t) / n)
        .collect()
}

pub fn mae_derivative(predictions: &[f64], targets: &[f64]) -> Vec<f64> {
    predictions.iter()
        .zip(targets.iter())
        .map(|(&p, &t)| {
            if p - t > 0.0 {
                1.0
            } else if p - t < 0.0 {
                -1.0
            } else {
                0.0 // ponto não diferenciável
            }
        })
        .collect()
}

pub fn huber_loss_derivative(predictions: &[f64], targets: &[f64], delta: f64) -> Vec<f64> {
    predictions.iter()
        .zip(targets.iter())
        .map(|(&p, &t)| {
            let error = p - t;
            if error.abs() <= delta {
                error
            } else {
                delta * error.signum()
            }
        })
        .collect()
}

pub fn cross_entropy_derivative(predictions: &[f64], targets: &[f64]) -> Vec<f64> {
    predictions.iter()
        .zip(targets.iter())
        .map(|(&p, &t)| p - t)
        .collect()
}
