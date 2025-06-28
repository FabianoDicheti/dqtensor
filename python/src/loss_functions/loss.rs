pub fn mse(y_true: &[f64], y_pred: &[f64]) -> f64 {
    y_true.iter().zip(y_pred.iter())
        .map(|(t, p)| (t - p).powi(2))
        .sum::<f64>() / y_true.len() as f64
}

pub fn mse_derivative(y_true: &[f64], y_pred: &[f64]) -> Vec<f64> {
    let n = y_true.len() as f64;
    y_true.iter().zip(y_pred.iter())
        .map(|(t, p)| 2.0 * (p - t) / n)
        .collect()
}

pub fn mae(y_true: &[f64], y_pred: &[f64]) -> f64 {
    y_true.iter().zip(y_pred.iter())
        .map(|(t, p)| (t - p).abs())
        .sum::<f64>() / y_true.len() as f64
}

pub fn mae_derivative(y_true: &[f64], y_pred: &[f64]) -> Vec<f64> {
    y_true.iter().zip(y_pred.iter())
        .map(|(t, p)| {
            if p - t > 0.0 {
                1.0
            } else if p - t < 0.0 {
                -1.0
            } else {
                0.0
            }
        })
        .collect()
}

///  p/ outliers
pub fn huber_loss(y_true: &[f64], y_pred: &[f64], delta: f64) -> f64 {
    y_true.iter().zip(y_pred.iter())
        .map(|(t, p)| {
            let error = p - t;
            if error.abs() <= delta {
                0.5 * error.powi(2)
            } else {
                delta * (error.abs() - 0.5 * delta)
            }
        })
        .sum::<f64>() / y_true.len() as f64
}

pub fn huber_loss_derivative(y_true: &[f64], y_pred: &[f64], delta: f64) -> Vec<f64> {
    y_true.iter().zip(y_pred.iter())
        .map(|(t, p)| {
            let error = p - t;
            if error.abs() <= delta {
                error
            } else {
                delta * error.signum()
            }
        })
        .collect()
}

pub fn cross_entropy(y_true: &[f64], y_pred: &[f64]) -> f64 {
    let epsilon = 1e-15;
    y_true.iter().zip(y_pred.iter())
        .map(|(t, p)| {
            let p = p.clamp(epsilon, 1.0 - epsilon);
            -t * p.ln()
        })
        .sum()
}

pub fn cross_entropy_derivative(y_true: &[f64], y_pred: &[f64]) -> Vec<f64> {
    let epsilon = 1e-15;
    y_true.iter().zip(y_pred.iter())
        .map(|(t, p)| {
            let p = p.clamp(epsilon, 1.0 - epsilon);
            -t / p
        })
        .collect()
}
