use crate::f_not_linear::activation::{ActivationFunction, ActivationFunctionTrait};
use crate::optimizers::bp_optimizers::Optimizer;
use rand::Rng;

#[derive(Clone, Debug)]
pub struct DenseLayer {
    pub input_size: usize,
    pub output_size: usize,
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub activations: Vec<ActivationFunction>,

    pub last_input: Vec<f64>,
    pub last_z: Vec<f64>,
    pub last_output: Vec<f64>,

    pub grad_weights: Vec<Vec<f64>>,
    pub grad_biases: Vec<f64>,

    pub weight_optim: Box<dyn Optimizer>,
    pub bias_optim: Box<dyn Optimizer>,
}

impl DenseLayer {
    pub fn new(
        input_size: usize,
        output_size: usize,
        activations: Vec<ActivationFunction>,
        weight_optim: Box<dyn Optimizer>,
        bias_optim: Box<dyn Optimizer>,
    ) -> Self {
        assert_eq!(activations.len(), output_size, "Cada neurônio precisa de uma função de ativação");

        let mut rng = rand::thread_rng();
        let weights = (0..output_size)
            .map(|_| {
                (0..input_size)
                    .map(|_| rng.gen_range(-1.0..1.0))
                    .collect()
            })
            .collect();

        let biases = vec![0.0; output_size];

        Self {
            input_size,
            output_size,
            weights,
            biases,
            activations,
            last_input: vec![0.0; input_size],
            last_z: vec![0.0; output_size],
            last_output: vec![0.0; output_size],
            grad_weights: vec![vec![0.0; input_size]; output_size],
            grad_biases: vec![0.0; output_size],
            weight_optim,
            bias_optim,
        }
    }

    pub fn forward(&mut self, input: &Vec<f64>) -> Vec<f64> {
        assert_eq!(input.len(), self.input_size);

        self.last_input = input.clone();

        let mut output = vec![0.0; self.output_size];
        let mut z_values = vec![0.0; self.output_size];

        for i in 0..self.output_size {
            let z = self.weights[i]
                .iter()
                .zip(input.iter())
                .map(|(w, x)| w * x)
                .sum::<f64>()
                + self.biases[i];

            let activated = self.activations[i].activate(z);

            output[i] = activated;
            z_values[i] = z;
        }

        self.last_output = output.clone();
        self.last_z = z_values;

        output
    }

    pub fn backward(&mut self, gradient: &Vec<f64>) -> Vec<f64> {
        assert_eq!(gradient.len(), self.output_size);

        let mut grad_input = vec![0.0; self.input_size];

        for i in 0..self.output_size {
            let dz = gradient[i] * self.activations[i].derivative(self.last_z[i]);

            self.grad_biases[i] = dz;
            for j in 0..self.input_size {
                self.grad_weights[i][j] = dz * self.last_input[j];
                grad_input[j] += self.weights[i][j] * dz;
            }
        }

        grad_input
    }

    pub fn update(&mut self, _learning_rate: f64) {
        for i in 0..self.output_size {
            self.weight_optim
                .update(&mut self.weights[i], &self.grad_weights[i]);
            self.bias_optim
                .update(&mut self.biases[i..=i], &self.grad_biases[i..=i]);
        }
    }
}
