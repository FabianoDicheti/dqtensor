#[derive(Clone)]

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
    Tanh,
    Swish,
    Softmax,
}

impl ActivationFunction {
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
            ActivationFunction::Softmax => panic!("Softmax precisa ser aplicada a um vetor!"),
        }
    }

    pub fn apply_vector(&self, inputs: Vec<f64>) -> Vec<f64> {
        match self {
            ActivationFunction::Softmax => Self::softmax(inputs),
            _ => panic!("Apenas Softmax aceita um vetor de entrada."),
        }
    }

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

    fn softmax(inputs: Vec<f64>) -> Vec<f64> {
        let max_input = inputs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_values: Vec<f64> = inputs.iter().map(|&x| (x - max_input).exp()).collect();
        let sum_exp: f64 = exp_values.iter().sum();
        exp_values.iter().map(|&x| x / sum_exp).collect()
    }
}


// pub fn softmax(inputs: Vec<f64>) -> Vec<f64>{
//     let max_input = inputs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
//     //inputs.iter() -> cria um iterador sobre os elementos do vetor
//     //.cloned() clona os valores do iterador (copia os números não as referências)
//     // .fold(f64::NEG_INFINITY, f64::max) -> fold é uma função de redução, que percorre todo o vetor e aplica uma operação
//     // nesse caso fold está comparando todos os valores do vetor com o menor valor possível f64::NEG_INFINITY, para encontrar o maior valor do vetor
//     // esse maior valor vai ser subtraido dos outros na linha de baixo, pra evitar numeros gigantescos na hora de calcular o exponencial
//     let exp_values: Vec<f64> = inputs.iter().map(|&x| (x -max_input).exp()).collect();
//     // .map(|&x| (x -max_input).exp()) -> para cada referência de cada elemento |&x| pega o x e reduz o valor maximo, depois aplica o expoente,
//     // usa o collect pra coletar os valores e criar um novo vetor
//     let sum_exp: f64 = exp_values.iter().sum();
//     // soma todos os valores do vetor para criar o '100%'
//     return exp_values.iter().map(|&x| x / sum_exp).collect()
//     // para cada valor de x divide pela soma, de modo que normalize tudo e a soma dos valores resulte em 100%
// }