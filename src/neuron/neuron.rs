use crate::f_not_linear::activation::ActivationFunction;
use crate::f_not_linear::derivatives::ActivationDerivatives;
use rand::random;

#[derive(Debug, Clone)]
pub struct Neuron {
    pub activation_func: ActivationFunction,
}

impl Neuron {
    pub fn new(activation_func: ActivationFunction) -> Self {
        Neuron { activation_func }
    }

    pub fn activate(&self, input: f64) -> f64 {
        self.activation_func.apply(input)
    }
}

#[derive(Debug, Clone)]
pub struct Layer {
    pub name: String,
    pub neurons: Vec<Neuron>,
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
}

impl Layer {
    pub fn new(name: String, neurons: Vec<Neuron>, input_size: usize) -> Self {
        let num_neurons = neurons.len();

        let weights = (0..num_neurons)
            .map(|_| (0..input_size).map(|_| random::<f64>()).collect())
            .collect();

        let biases = (0..num_neurons).map(|_| random::<f64>()).collect();

        Layer {
            name,
            neurons,
            weights,
            biases,
        }
    }

    /// Forward da camada
    pub fn forward(&self, input_vec: &Vec<f64>) -> Vec<f64> {
        self.neurons
            .iter()
            .enumerate()
            .map(|(i, neuron)| {
                let weighted_sum: f64 = input_vec
                    .iter()
                    .zip(self.weights[i].iter())
                    .map(|(x, w)| x * w)
                    .sum::<f64>()
                    + self.biases[i];

                neuron.activate(weighted_sum)
            })
            .collect()
    }

    /// Backward com atualização simples
    pub fn backward(
        &mut self,
        input_vec: &Vec<f64>,
        output: &Vec<f64>,
        d_loss_d_output: &Vec<f64>,
    ) -> Vec<f64> {
        assert_eq!(output.len(), self.neurons.len());
        assert_eq!(d_loss_d_output.len(), self.neurons.len());

        let mut d_loss_d_input = vec![0.0; self.weights[0].len()];

        let mut bias_grads = vec![0.0; self.neurons.len()];
        let mut weight_grads = vec![vec![0.0; self.weights[0].len()]; self.neurons.len()];

        for (i, neuron) in self.neurons.iter().enumerate() {
            let deriv_ativacao = neuron.activation_func.derivative(output[i]);
            let delta = d_loss_d_output[i] * deriv_ativacao;

            bias_grads[i] = delta;

            for j in 0..input_vec.len() {
                weight_grads[i][j] = delta * input_vec[j];
                d_loss_d_input[j] += self.weights[i][j] * delta;
            }
        }

        // Atualiza pesos e bias (pode ser substituído por otimizadores externos)
        let lr = 0.01;
        for i in 0..self.neurons.len() {
            for j in 0..input_vec.len() {
                self.weights[i][j] -= lr * weight_grads[i][j];
            }
            self.biases[i] -= lr * bias_grads[i];
        }

        d_loss_d_input
    }
}
