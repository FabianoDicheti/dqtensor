#[derive(Debug, Clone)]


pub enum ActivationFunction {
    ReLU,
    LeakyReLU(f64),
    ReLUN(f64),
    StarReLU(f64, f64),
    ShiLU(f64, f64),
    ParametricReLU(f64),
    ELU(f64),
    GELU,
    Sigmoid,
    Softsign,
    Tanh,
    Swish,
    Softmax(usize), // Define o número de classes para Softmax
}

impl ActivationFunction {
    /// Aplica a função de ativação para um único valor.
    /// Softmax NÃO é tratado aqui porque requer um vetor.
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::ReLU => Self::relu(x),
            ActivationFunction::LeakyReLU(alpha) => Self::leaky_relu(x, *alpha),
            ActivationFunction::ReLUN(n) => Self::relun(x, *n),
            ActivationFunction::StarReLU(s, t) => Self::star_relu(x, *s, *t),
            ActivationFunction::ShiLU(alpha, beta) => Self::shilu(x, *alpha, *beta),
            ActivationFunction::ParametricReLU(a) => Self::parametric_relu(x, *a),
            ActivationFunction::ELU(alpha) => Self::elu(x, *alpha),
            ActivationFunction::GELU => Self::gelu(x),
            ActivationFunction::Sigmoid => Self::sigmoid(x),
            ActivationFunction::Tanh => Self::tanh(x),
            ActivationFunction::Swish => Self::swish(x),
            ActivationFunction::Softsign => Self::softsign(x), // <-- Adicionado aqui!
            ActivationFunction::Softmax(_) => {
                panic!("Softmax requer um vetor de entrada, use `apply_softmax()`");
            }
        }
    }

    /// Método separado para Softmax
    pub fn apply_softmax(inputs: &[f64]) -> Vec<f64> {
        let max_input = inputs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exps: Vec<f64> = inputs.iter().map(|&x| (x - max_input).exp()).collect();
        let sum_exps: f64 = exps.iter().sum();
        exps.iter().map(|&x| x / sum_exps).collect()
    }

    // Implementações das funções de ativação individuais
    fn relu(x: f64) -> f64 {
        x.max(0.0)
    }

    fn leaky_relu(x: f64, alpha: f64) -> f64 {
        if x > 0.0 { x } else { alpha * x }
    }

    fn relun(x: f64, n: f64) -> f64 {
        if x > 0.0 { x } else { n * x }
    }

    fn star_relu(x: f64, s: f64, t: f64) -> f64 {
        if x > t { x } else { s * x }
    }

    fn shilu(x: f64, alpha: f64, beta: f64) -> f64 {
        if x >= beta { x } else { alpha * (x - beta) }
    }

    fn parametric_relu(x: f64, a: f64) -> f64 {
        if x > 0.0 { x } else { a * x }
    }

    fn elu(x: f64, alpha: f64) -> f64 {
        if x > 0.0 { x } else { alpha * ((x).exp() - 1.0) }
    }

    fn gelu(x: f64) -> f64 {
        0.5 * x * (1.0 + (x / (2.0f64.sqrt())).tanh())
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn tanh(x: f64) -> f64 {
        x.tanh()
    }

    fn swish(x: f64) -> f64 {
        x * Self::sigmoid(x)
    }

    fn softsign(x: f64) -> f64 {
    x / (1.0 + x.abs())
    }

}
