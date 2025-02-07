#[derive(Clone)]
pub enum DerivativeFunction {
    ReLU,
    LeakyReLU(f64),
    ReLUN(f64),
    StarReLU(f64, f64),
    ShiLU(f64, f64),
    ParametricReLU(f64),
    ELU(f64),
    GELU,
    Sigmoid,
    Tanh,
    Swish,
    Softmax,
}

impl DerivativeFunction {
    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            DerivativeFunction::ReLU => Self::relu_derivative(x),
            DerivativeFunction::LeakyReLU(alpha) => Self::leaky_relu_derivative(x, *alpha),
            DerivativeFunction::ReLUN(n) => Self::relun_derivative(x, *n),
            DerivativeFunction::StarReLU(s, t) => Self::star_relu_derivative(x, *s, *t),
            DerivativeFunction::ShiLU(alpha, beta) => Self::shilu_derivative(x, *alpha, *beta),
            DerivativeFunction::ParametricReLU(a) => Self::parametric_relu_derivative(x, *a),
            DerivativeFunction::ELU(alpha) => Self::elu_derivative(x, *alpha),
            DerivativeFunction::GELU => Self::gelu_derivative(x),
            DerivativeFunction::Sigmoid => Self::sigmoid_derivative(x),
            DerivativeFunction::Tanh => Self::tanh_derivative(x),
            DerivativeFunction::Swish => Self::swish_derivative(x),
            DerivativeFunction::Softmax => panic!("Softmax não tem derivada para um único valor!"),
        }
    }

    fn relu_derivative(x: f64) -> f64 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }

    fn leaky_relu_derivative(x: f64, alpha: f64) -> f64 {
        if x > 0.0 { 1.0 } else { alpha }
    }

    fn relun_derivative(x: f64, n: f64) -> f64 {
        if x > 0.0 { 1.0 } else { n }
    }

    fn star_relu_derivative(x: f64, s: f64, t: f64) -> f64 {
        if x > t { 1.0 } else { s }
    }

    fn shilu_derivative(x: f64, alpha: f64, beta: f64) -> f64 {
        if x >= beta { 1.0 } else { alpha }
    }

    fn parametric_relu_derivative(x: f64, a: f64) -> f64 {
        if x > 0.0 { 1.0 } else { a }
    }

    fn elu_derivative(x: f64, alpha: f64) -> f64 {
        if x > 0.0 { 1.0 } else { alpha * x.exp() }
    }

    fn gelu_derivative(x: f64) -> f64 {
        let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
        let cdf = 0.5 * (1.0 + (x / sqrt_2_over_pi).tanh());
        let pdf = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
        0.5 * (1.0 + cdf + x * pdf)
    }

    fn sigmoid_derivative(x: f64) -> f64 {
        let sigmoid = 1.0 / (1.0 + (-x).exp());
        sigmoid * (1.0 - sigmoid)
    }

    fn tanh_derivative(x: f64) -> f64 {
        1.0 - x.tanh().powi(2)
    }

    fn swish_derivative(x: f64) -> f64 {
        let sigmoid = 1.0 / (1.0 + (-x).exp());
        sigmoid + x * sigmoid * (1.0 - sigmoid)
    }
}


//p/ Softmax não tem uma derivada direta para um único valor, pq opera em um vetor. 
//tem q ser calculada em pelo vetor de entradas e a saída é uma matriz Jacobiana.

// fazer uma implementação separada
