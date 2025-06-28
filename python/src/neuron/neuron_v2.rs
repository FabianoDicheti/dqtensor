use crate::f_not_linear::activation::{ActivationFunction, apply_activation, apply_activation_derivative};
use crate::optimizers::bp_optimizers::Optimizer;

pub use crate::f_not_linear::activation::ActivationFunction;

#[derive(Debug, Clone)]
pub struct NeuronV2 {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub activation: ActivationFunction,
    pub weight_optimizer: Box<dyn Optimizer>,
    pub bias_optimizer: Box<dyn Optimizer>,
    pub last_input: Vec<f64>,
    pub last_z: f64,
    pub last_a: f64,
    pub grad_weights: Vec<f64>,
    pub grad_bias: f64,
}

impl NeuronV2 {
    pub fn new(
        input_size: usize,
        activation: ActivationFunction,
        weight_optimizer: Box<dyn Optimizer>,
        bias_optimizer: Box<dyn Optimizer>,
    ) -> Self {
        let weights = vec![0.01; input_size];
        let bias = 0.0;

        Self {
            weights,
            bias,
            activation,
            weight_optimizer,
            bias_optimizer,
            last_input: vec![0.0; input_size],
            last_z: 0.0,
            last_a: 0.0,
            grad_weights: vec![0.0; input_size],
            grad_bias: 0.0,
        }
    }

    pub fn new_with_initializer(
        input_size: usize,
        activation: ActivationFunction,
        weight_initializer: fn(usize) -> Vec<f64>,
        bias_initializer: fn() -> f64,
        weight_optimizer: Box<dyn Optimizer>,
        bias_optimizer: Box<dyn Optimizer>,
    ) -> Self {
        let weights = weight_initializer(input_size);
        let bias = bias_initializer();

        Self {
            weights: weights.clone(),
            bias,
            activation,
            weight_optimizer,
            bias_optimizer,
            last_input: vec![0.0; weights.len()],
            last_z: 0.0,
            last_a: 0.0,
            grad_weights: vec![0.0; weights.len()],
            grad_bias: 0.0,
        }
    }

    pub fn forward(&mut self, input: &Vec<f64>) -> f64 {
        if input.len() != self.weights.len() {
            println!(" Tamanho inesperado detectado!");
            println!(" input.len(): {}", input.len());
            println!(" weights.len(): {}", self.weights.len());
            println!(" input: {:?}", input);
            println!(" weights: {:?}", self.weights);
        }

        assert_eq!(
            input.len(),
            self.weights.len(),
            "Tamanho da entrada ({}) não corresponde ao número de pesos ({}) no neurônio.",
            input.len(),
            self.weights.len()
        );

        self.last_input = input.clone();
        self.last_z = input.iter()
            .zip(&self.weights)
            .map(|(i, w)| i * w)
            .sum::<f64>() + self.bias;

        self.last_a = apply_activation(&self.activation, self.last_z);
        self.last_a
    }


    pub fn backward(&mut self, grad_output: f64) -> Vec<f64> {
        assert_eq!(
            self.last_input.len(),
            self.weights.len(),
            "Inconsistência em backward(): entrada ({}) e pesos ({}) com tamanhos diferentes.",
            self.last_input.len(),
            self.weights.len()
        );

        let d_activation = apply_activation_derivative(&self.activation, self.last_z);
        let delta = (grad_output * d_activation).clamp(-1.0, 1.0);

        for (gw, inp) in self.grad_weights.iter_mut().zip(&self.last_input) {
            *gw = delta * inp;
        }
        self.grad_bias = delta;

        self.weights.iter().map(|w| delta * w).collect()
    }

    pub fn update(&mut self) {
        self.weight_optimizer
            .update(&mut self.weights, &self.grad_weights);
        self.bias_optimizer
            .update(std::slice::from_mut(&mut self.bias), &[self.grad_bias]);
    }

    pub fn reset(&mut self) {
        self.grad_weights.fill(0.0);
        self.grad_bias = 0.0;
    }
}
