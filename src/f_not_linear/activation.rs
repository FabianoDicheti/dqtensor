use std::f64::consts::E;

#[derive(Clone, Debug)]
pub enum ActivationFunction {
    Sigmoid,
    Tanh,
    Relu,
    Linear,
    ELU,
    Mish,
    Gaussian,
}

impl ActivationFunction {
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::Relu => if x > 0.0 { x } else { 0.0 },
            ActivationFunction::Linear => x,
            ActivationFunction::ELU => if x >= 0.0 { x } else { x.exp() - 1.0 },

            ActivationFunction::Mish => x * (1.0 + x.exp()).ln().tanh(),
            ActivationFunction::Gaussian => (-x.powi(2)).exp(),
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            ActivationFunction::Tanh => 1.0 - x.tanh().powi(2),
            ActivationFunction::Relu => if x > 0.0 { 1.0 } else { 0.0 },
            ActivationFunction::Linear => 1.0,
            ActivationFunction::ELU => if x >= 0.0 { 1.0 } else { x.exp() },
            ActivationFunction::Mish => {
                let sp = (1.0 + x.exp()).ln();
                let tanh_sp = sp.tanh();
                let sech_sp_sq = 1.0 - tanh_sp.powi(2);
                tanh_sp + x * sech_sp_sq / (1.0 + x.exp())
            }
            ActivationFunction::Gaussian => -2.0 * x * (-x.powi(2)).exp(),
        }
    }
}


pub fn apply_activation(activation: &ActivationFunction, x: f64) -> f64 {
    activation.apply(x)

}


pub fn apply_activation_derivative(activation: &ActivationFunction, x: f64) -> f64 {
    activation.derivative(x)
}
