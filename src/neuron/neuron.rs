use crate::f_not_linear::activation::ActivationFunction;
use rand::random;

/// Estrutura para representar um Neurônio
#[derive(Clone)]
pub struct Neuron {
    activation_func: ActivationFunction,
}

impl Neuron {
    /// Construtor para criar um neurônio apenas com a função de ativação
    pub fn new(activation_func: ActivationFunction) -> Self {
        Neuron { activation_func }
    }

    /// Método para ativar o neurônio
    pub fn activate(&self, input: f64) -> f64 {
        self.activation_func.apply(input)
    }
}

/// Estrutura para representar uma camada de neurônios
pub struct Layer {
    name: String,
    neurons: Vec<Neuron>,
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
}

impl Layer {
    /// Construtor para criar uma camada com neurônios e inicializar pesos e bias aleatoriamente
    pub fn new(name: String, neurons: Vec<Neuron>, input_size: usize) -> Self {
        let num_neurons = neurons.len();
        
        // Inicializando pesos aleatórios para cada neurônio
        let weights = (0..num_neurons)
            .map(|_| (0..input_size).map(|_| random::<f64>()).collect())
            .collect();

        // Inicializando bias aleatórios
        let biases = (0..num_neurons).map(|_| random::<f64>()).collect();

        Layer { name, neurons, weights, biases }
    }

    /// Método para processar uma entrada e retornar um vetor com as saídas dos neurônios
    pub fn forward(&self, input_vec: &[f64]) -> Vec<f64> {
        self.neurons.iter()
            .enumerate()
            .map(|(i, neuron)| {
                let weighted_sum: f64 = input_vec.iter()
                    .zip(self.weights[i].iter())
                    .map(|(&x, &w)| x * w)
                    .sum::<f64>() + self.biases[i];

                neuron.activate(weighted_sum)
            })
            .collect()
    }
}
