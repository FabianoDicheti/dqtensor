use crate::f_not_linear::activation::ActivationFunction;

pub trait ActivationDerivatives {
    fn derivative(&self, x: f64) -> f64;
}

impl ActivationDerivatives for ActivationFunction {
    fn derivative(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Sigmoid => {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                sigmoid * (1.0 - sigmoid)
            }
            ActivationFunction::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            ActivationFunction::Tanh => {
                1.0 - x.tanh().powi(2)
            }
            ActivationFunction::Softplus => {
                1.0 / (1.0 + (-x).exp())
            }
            ActivationFunction::Softmax(_) => {
                // (Jacobian). Placeholder.
                1.0
            }
            ActivationFunction::Linear => {
                1.0
            }
        }
    }
}
