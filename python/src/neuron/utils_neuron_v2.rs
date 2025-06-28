use rand::thread_rng;
use rand::distributions::{Distribution, Uniform};
use rand_distr::Normal;


/// Inicializador: Zera todos os pesos
pub fn zeros_initializer(size: usize) -> Vec<f64> {
    vec![0.0; size]
}

/// Inicializador: Bias zerado
pub fn zero_bias_initializer() -> f64 {
    0.0
}

/// Inicializador: Valor constante para todos os pesos
pub fn constant_initializer(size: usize, value: f64) -> Vec<f64> {
    vec![value; size]
}

/// Inicializador: Bias constante
pub fn constant_bias_initializer(value: f64) -> f64 {
    value
}

/// Inicializador uniforme: valores no intervalo [-limit, limit]
pub fn uniform_initializer(size: usize, limit: f64) -> Vec<f64> {
    let mut rng = thread_rng();
    let dist = Uniform::new_inclusive(-limit, limit);
    (0..size).map(|_| dist.sample(&mut rng)).collect()
}

/// Inicializador normal: média = 0, desvio padrão = std
pub fn normal_initializer(size: usize, std: f64) -> Vec<f64> {
    let mut rng = thread_rng();
    let dist = Normal::new(0.0, std).unwrap();
    (0..size).map(|_| dist.sample(&mut rng)).collect()
}

/// Inicializador Xavier (Glorot)
/// Usado para ativação tanh ou linear
pub fn xavier_initializer(size: usize) -> Vec<f64> {
    let bound = (6.0f64).sqrt() / (size as f64).sqrt();
    let between = Uniform::from(-bound..bound);
    let mut rng = rand::thread_rng();

    (0..size).map(|_| between.sample(&mut rng)).collect()
}

/// Inicializador He
/// Usado para Relu e variantes
pub fn he_initializer(size: usize) -> Vec<f64> {
    let std = (2.0 / size as f64).sqrt();
    normal_initializer(size, std)
}

/// Inicializador uniforme padrão: [-0.5, 0.5]
pub fn default_uniform_initializer(size: usize) -> Vec<f64> {
    uniform_initializer(size, 0.5)
}

/// Bias uniform
pub fn default_uniform_bias_initializer() -> f64 {
    let between = Uniform::from(-0.5..0.5);
    let mut rng = rand::thread_rng();
    between.sample(&mut rng)
}



pub fn clip_gradients(grads: &mut [f64], clip_value: f64) {
    for g in grads.iter_mut() {
        if *g > clip_value {
            *g = clip_value;
        } else if *g < -clip_value {
            *g = -clip_value;
        }
    }
}