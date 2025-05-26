use crate::f_not_linear::activation::ActivationFunction;
use rand::random;

/// Estrutura para representar um Neurônio
#[derive(Debug, Clone)]
pub struct Neuron {
    pub activation_func: ActivationFunction,
}

impl Neuron {
    /// Construtor
    pub fn new(activation_func: ActivationFunction) -> Self {
        Neuron { activation_func }
    }

    /// Método para ativar o neurônio
    pub fn activate(&self, input: f64) -> f64 {
        self.activation_func.apply(input)
    }
}

/// Estrutura para representar uma camada de neurônios
#[derive(Debug, Clone)]
pub struct Layer {
    pub name: String,
    pub neurons: Vec<Neuron>,
    pub weights: Vec<Vec<f64>>, // Matriz de pesos (neurônio x entrada)
    pub biases: Vec<f64>,        // Vetor de biases (um por neurônio)
}

impl Layer {
    /// Cria uma camada
    pub fn new(name: String, neurons: Vec<Neuron>, input_size: usize) -> Self {
        let num_neurons = neurons.len();

        let weights = (0..num_neurons)
            .map(|_| (0..input_size).map(|_| random::<f64>()).collect())
            .collect();

        let biases = (0..num_neurons)
            .map(|_| random::<f64>())
            .collect();

        Layer { name, neurons, weights, biases }
    }

    /// Forward normal, retorna a saída dos neurônios
    pub fn forward(&self, input_vec: &[f64]) -> Vec<f64> {
        self.neurons.iter()
            .enumerate()
            .map(|(i, neuron)| {
                let weighted_sum = self.weighted_sum(i, input_vec);
                neuron.activate(weighted_sum)
            })
            .collect()
    }

    /// Calcula o somatório ponderado (wx + b) de um neurônio
    pub fn weighted_sum(&self, neuron_index: usize, input_vec: &[f64]) -> f64 {
        input_vec.iter()
            .zip(self.weights[neuron_index].iter())
            .map(|(&x, &w)| x * w)
            .sum::<f64>() + self.biases[neuron_index]
    }

    /// Atualiza pesos e biases (útil para otimizadores)
    pub fn update_weights(&mut self, delta_weights: &[Vec<f64>]) {
        for (i, delta_w) in delta_weights.iter().enumerate() {
            for (j, delta) in delta_w.iter().enumerate() {
                self.weights[i][j] += delta;
            }
        }
    }

    pub fn update_biases(&mut self, delta_biases: &[f64]) {
        for (i, delta) in delta_biases.iter().enumerate() {
            self.biases[i] += delta;
        }
    }
}
