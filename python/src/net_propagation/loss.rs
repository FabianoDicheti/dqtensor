pub fn mse(y_pred: &Vec<f64>, y_true: &Vec<f64>) -> f64 {
    y_pred.iter()
        .zip(y_true.iter())
        .map(|(y, t)| (y - t).powi(2))
        .sum::<f64>() / y_pred.len() as f64
}

pub fn mse_derivative(y_pred: &Vec<f64>, y_true: &Vec<f64>) -> Vec<f64> {
    y_pred.iter()
        .zip(y_true.iter())
        .map(|(y, t)| 2.0 * (y - t) / y_pred.len() as f64)
        .collect()
}
