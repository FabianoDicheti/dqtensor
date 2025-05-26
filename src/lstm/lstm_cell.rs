use std::collections::HashMap;
use crate::f_not_linear::activation::ActivationFunction;
use crate::f_not_linear::derivatives::ActivationDerivatives;
use crate::neuron::neuron::{Layer, Neuron};
use crate::lstm::state::EstadoLSTM;
use crate::optimizers::bp_optimizers::Optimizer;

#[derive(Debug)]
pub struct LSTMCell {
    pub porta_forget: Layer,
    pub porta_input: Layer,
    pub porta_output: Layer,
    pub porta_candidato: Layer,

    pub estado: EstadoLSTM,

    // Otimizadores
    pub weight_optimizers: HashMap<String, Box<dyn Optimizer>>,
    pub bias_optimizers: HashMap<String, Box<dyn Optimizer>>,
}

impl LSTMCell {
    pub fn new(
        name: &str,
        num_neuronios: usize,
        input_size: usize,
        activation_funcs: (ActivationFunction, ActivationFunction),
        weight_optimizers: HashMap<String, Box<dyn Optimizer>>,
        bias_optimizers: HashMap<String, Box<dyn Optimizer>>,
    ) -> Self {
        let (sigmoid, tanh) = activation_funcs;

        Self {
            porta_forget: Layer::new(
                format!("{name}_forget"),
                vec![Neuron::new(sigmoid.clone()); num_neuronios],
                input_size + num_neuronios,
            ),
            porta_input: Layer::new(
                format!("{name}_input"),
                vec![Neuron::new(sigmoid.clone()); num_neuronios],
                input_size + num_neuronios,
            ),
            porta_output: Layer::new(
                format!("{name}_output"),
                vec![Neuron::new(sigmoid.clone()); num_neuronios],
                input_size + num_neuronios,
            ),
            porta_candidato: Layer::new(
                format!("{name}_candidato"),
                vec![Neuron::new(tanh.clone()); num_neuronios],
                input_size + num_neuronios,
            ),

            estado: EstadoLSTM::new(num_neuronios),

            weight_optimizers,
            bias_optimizers,
        }
    }

    pub fn forward(&mut self, entrada: &Vec<f64>) -> Vec<f64> {
        let entrada_completa = [&entrada[..], &self.estado.memoria_h[..]].concat();

        let forget = self.porta_forget.forward(&entrada_completa);
        let input = self.porta_input.forward(&entrada_completa);
        let output = self.porta_output.forward(&entrada_completa);
        let candidato = self.porta_candidato.forward(&entrada_completa);

        self.estado.memoria_c = forget
            .iter()
            .zip(&self.estado.memoria_c)
            .zip(&input)
            .zip(&candidato)
            .map(|(((f, c), i), g)| f * c + i * g)
            .collect();

        self.estado.memoria_h = output
            .iter()
            .zip(&self.estado.memoria_c)
            .map(|(o, c)| o * c.tanh())
            .collect();

        self.estado.memoria_h.clone()
    }

    /// Placeholder do Backpropagation
    pub fn backward(
        &mut self,
        entrada: &Vec<f64>,
        saida: &Vec<f64>,
        d_loss_d_output: &Vec<f64>,
    ) -> Vec<f64> {
        println!("Backward ainda precisa implementar toda a matemática.");

        let grad_para_entrada_anterior = vec![0.0; entrada.len()]; // Placeholder

        grad_para_entrada_anterior
    }
}
