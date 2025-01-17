
pub fn relu(x: f64) -> f64 {
    return x.max(0.0)
}

pub fn leaky_relu(x: f64, alpha: f64) -> f64 {
    if x > 0.0 {
        return x
    } else {
        return alpha*x
    }
}

pub fn relun(x: f64, n: f64) -> f64 {
    if x > 0.0 {
        return x
    } else {
        return n*x
    }
}

pub fn star_relu(x: f64, s: f64, t: f64) -> f64{
    if x > t{
        return x
    } else {
        return s*x
    }
}

pub fn shilu(x: f64, alpha: f64, beta: f64) -> f64{
    if x >= beta{
        return x
    } else {
        return alpha*(x-beta)
    }
}

pub fn parametric_relu(x: f64, a: f64) -> f64{
    if x > 0.0{
        return x
    } else {
        return a * x
    }
}

pub fn elu(x: f64, alpha: f64) -> f64{
    if x > 0.0{
        return x
    } else {
        return alpha * ((x).exp() -1.0)
    }

}

pub fn gelu(x: f64) -> f64{
    // revisar
    return 0.5 * x * (1.0 + (x / (2.0f64.sqrt())).tanh())
}

pub fn sigmoid(x: f64) -> f64{
    return 1.0 / (1.0+(-x).exp())
}

pub fn tanh(x: f64) -> f64{
    return x.tanh()
}

pub fn swish(x: f64) -> f64{
    return x * sigmoid(x)
}

pub fn softmax(inputs: Vec<f64>) -> Vec<f64>{
    let max_input = inputs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    //inputs.iter() -> cria um iterador sobre os elementos do vetor
    //.cloned() clona os valores do iterador (copia os números não as referências)
    // .fold(f64::NEG_INFINITY, f64::max) -> fold é uma função de redução, que percorre todo o vetor e aplica uma operação
    // nesse caso fold está comparando todos os valores do vetor com o menor valor possível f64::NEG_INFINITY, para encontrar o maior valor do vetor
    // esse maior valor vai ser subtraido dos outros na linha de baixo, pra evitar numeros gigantescos na hora de calcular o exponencial
    let exp_values: Vec<f64> = inputs.iter().map(|&x| (x -max_input).exp()).collect();
    // .map(|&x| (x -max_input).exp()) -> para cada referência de cada elemento |&x| pega o x e reduz o valor maximo, depois aplica o expoente,
    // usa o collect pra coletar os valores e criar um novo vetor
    let sum_exp: f64 = exp_values.iter().sum();
    // soma todos os valores do vetor para criar o '100%'
    return exp_values.iter().map(|&x| x / sum_exp).collect()
    // para cada valor de x divide pela soma, de modo que normalize tudo e a soma dos valores resulte em 100%
}