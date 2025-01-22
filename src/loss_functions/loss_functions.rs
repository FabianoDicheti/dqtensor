pub enum LossFuncion {
    MeanSquaredError,
    CrossEntropy,
    MeanAbsoluteError,
}

impl LossFuncion {

    pub fn calculate( &self, predictions: &[f64], targets: &[f64]) -> f64 {
        match self {
            LossFuncion::MeanSquaredError => Self::mean_squared_error(predictions, targets),
            LossFuncion::CrossEntropy => Self::cross_entropy(predictions, targets),
            LossFuncion::MeanAbsoluteError => Self::mean_absolute_error(predictions, targets),
        }
    }

    fn mean_squared_error(predictions: &[f64], targets: &[f64]) -> f64 {
        let n = predictions.len();
        predictions.iter()
            .zip(targets.iter())
            .map(|(&p, &t)| (p-t).powi(2))
            .sum::<f64>()
            / n as f64
    }

    fn cross_entropy(predictions: &[f64], targets: &[f64]) -> f64 {
        predictions.iter()
            .zip(targets.iter())
            .map(|(&p, &t)| {
                if t == 1.0 {
                    -p.ln()
                } else if t == 0.0 {
                    -(1.0 -p).ln()
                } else {
                    panic!("Targets for Cross-Entropy must be (0 or 1)!");
                }
            })
            .sum()
    }

    fn mean_absolute_error(predictions: &[f64], targets: &[f64]) -> f64 {
        let n = predictions.len();
        predictions.iter()
            .zip(targets.iter())
            .map(|(&p, &t)| (p-t).abs())
            .sum::<f64>()
            / n as f64
    }
}