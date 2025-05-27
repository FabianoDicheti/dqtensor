#[derive(Clone, Debug)]
pub enum ActivationFunction {
    Sigmoid,
    ReLU,
    Tanh,
    Softplus,
    Softmax(usize), // Softmax geralmente aplicada no vetor completo
    Linear,
}

///  aplicar e derivar funções de ativação
pub trait ActivationFunctionTrait {
    fn activate(&self, x: f64) -> f64;
    fn derivative(&self, x: f64) -> f64;
}

impl ActivationFunctionTrait for ActivationFunction {
    fn activate(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::ReLU => if x > 0.0 { x } else { 0.0 },
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::Softplus => (1.0 + x.exp()).ln(),
            ActivationFunction::Softmax(_) => x, // Softmax é aplicada no vetor, não aqui
            ActivationFunction::Linear => x,
        }
    }

    fn derivative(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Sigmoid => {
                let sig = 1.0 / (1.0 + (-x).exp());
                sig * (1.0 - sig)
            }
            ActivationFunction::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            ActivationFunction::Tanh => 1.0 - x.tanh().powi(2),
            ActivationFunction::Softplus => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::Softmax(_) => 1.0, // Derivada simplificada, o correto seria Jacobiana
            ActivationFunction::Linear => 1.0,
        }
    }
}
