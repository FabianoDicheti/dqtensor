use crate::f_not_linear::activation::ActivationFunction;

/// Trait para calcular a derivada das funções de ativação
pub trait ActivationDerivatives {
    fn derivative(&self, x: f64) -> f64;
}

impl ActivationDerivatives for ActivationFunction {
    fn derivative(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::ReLU => {
                if x > 0.0 { 1.0 } else { 0.0 }
            }
            ActivationFunction::LeakyReLU(alpha) => {
                if x > 0.0 { 1.0 } else { *alpha }
            }
            ActivationFunction::ReLUN(n) => {
                if x > 0.0 { 1.0 } else { *n }
            }
            ActivationFunction::StarReLU(s, t) => {
                if x > *t { 1.0 } else { *s }
            }
            ActivationFunction::ShiLU(alpha, beta) => {
                if x >= *beta { 1.0 } else { *alpha }
            }
            ActivationFunction::ParametricReLU(a) => {
                if x > 0.0 { 1.0 } else { *a }
            }
            ActivationFunction::ELU(alpha) => {
                if x > 0.0 { 1.0 } else { alpha * x.exp() }
            }
            ActivationFunction::GELU => {
                let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
                let cdf = 0.5 * (1.0 + (x / sqrt_2_over_pi).tanh());
                let pdf = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
                0.5 * (1.0 + cdf + x * pdf)
            }
            ActivationFunction::Sigmoid => {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                sigmoid * (1.0 - sigmoid)
            }
            ActivationFunction::Tanh => {
                1.0 - x.tanh().powi(2)
            }
            ActivationFunction::Swish => {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                sigmoid + x * sigmoid * (1.0 - sigmoid)
            }
            ActivationFunction::Softsign => {
                1.0 / (1.0 + x.abs()).powi(2)
            }
            ActivationFunction::Softmax(_) => {
                panic!("A derivada da Softmax não é escalar. É uma matriz Jacobiana e deve ser calculada sobre um vetor.")
            }
        }
    }
}
