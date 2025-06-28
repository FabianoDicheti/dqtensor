use crate::f_not_linear::activation::ActivationFunction;
use crate::neuron::neuron_v2::NeuronV2;
use crate::optimizers::bp_optimizers::Optimizer;

/// densa simples
pub struct DenseLayer {
    pub neurons: Vec<NeuronV2>,
}

impl DenseLayer {
    pub fn new(
        input_size: usize,
        num_neurons: usize,
        activation: ActivationFunction,
        weight_optimizer: Box<dyn Optimizer>,
        bias_optimizer: Box<dyn Optimizer>,
    ) -> Self {
        let neurons = (0..num_neurons)
            .map(|_| {
                NeuronV2::new(
                    input_size,
                    activation.clone(),
                    weight_optimizer.clone(),
                    bias_optimizer.clone(),
                )
            })
            .collect();

        Self { neurons }
    }

    /// diferentes funções de ativação por neurônio
    pub fn new_heterogeneous(
        input_size: usize,
        funcoes: Vec<ActivationFunction>,
        weight_optimizer: Box<dyn Optimizer>,
        bias_optimizer: Box<dyn Optimizer>,
    ) -> Self {
        let neurons = funcoes
            .into_iter()
            .map(|func| {
                NeuronV2::new(
                    input_size,
                    func,
                    weight_optimizer.clone(),
                    bias_optimizer.clone(),
                )
            })
            .collect();

        Self { neurons }
    }

    pub fn forward(&mut self, input: &Vec<f64>) -> Vec<f64> {
        self.neurons.iter_mut().map(|n| n.forward(input)).collect()
    }

    pub fn backward(&mut self, grad_outputs: &Vec<f64>) -> Vec<f64> {
        let mut input_grads = vec![0.0; self.neurons[0].weights.len()];
        for (neuron, grad_output) in self.neurons.iter_mut().zip(grad_outputs.iter()) {
            let grads = neuron.backward(*grad_output);
            for (i, grad) in grads.iter().enumerate() {
                input_grads[i] += grad;
            }
        }
        input_grads
    }

    pub fn update(&mut self) {
        for neuron in self.neurons.iter_mut() {
            neuron.weight_optimizer.update(&mut neuron.weights, &neuron.grad_weights);
            neuron.bias_optimizer.update(std::slice::from_mut(&mut neuron.bias), &[neuron.grad_bias]);
        }
    }

    pub fn reset(&mut self) {
        for neuron in self.neurons.iter_mut() {
            neuron.reset();
        }
    }

    pub fn debug(&self) {
        for (i, neuron) in self.neurons.iter().enumerate() {
            println!("Neuron {}: pesos = {:?}, bias = {}", i, neuron.weights, neuron.bias);
        }
    }
}
