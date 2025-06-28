use std::collections::HashMap;

use crate::f_not_linear::activation::ActivationFunction;
use crate::neuron::neuron_v2::NeuronV2;
use crate::neuron::utils_neuron_v2::*;
use crate::optimizers::bp_optimizers::Optimizer;

/// Estado interno da célula (h e c)
#[derive(Clone, Debug)]
pub struct LSTMState {
    pub h: f64,
    pub c: f64,
}

/// Identificadores das portas/gates
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum GateType {
    Candidate,
    Input,
    Forget,
    Output,
}

/// Trait geral para células recorrentes
pub trait RecurrentCell {
    fn forward(&mut self, input: &Vec<f64>) -> (f64, f64);
    fn backward(&mut self, input: &Vec<f64>, grad_h: f64, grad_c: f64) -> (Vec<f64>, f64);
    fn reset_state(&mut self);
    fn update(&mut self);
}

/// Implementação da célula LSTM
pub struct LSTMCell {
    pub neurons: HashMap<GateType, NeuronV2>,
    pub state: LSTMState,
    pub combination_fn: Option<Box<dyn Fn(f64, f64, f64, f64) -> f64>>,
}

impl LSTMCell {
    pub fn new(
        input_size: usize,
        activations: HashMap<GateType, ActivationFunction>,
        weight_initializer: fn(usize) -> Vec<f64>,
        bias_initializer: fn() -> f64,
        weight_optimizer: Box<dyn Optimizer>,
        bias_optimizer: Box<dyn Optimizer>,
        combination_fn: Option<Box<dyn Fn(f64, f64, f64, f64) -> f64>>,
    ) -> Self {
        let mut neurons = HashMap::new();

        for gate in [
            GateType::Candidate,
            GateType::Input,
            GateType::Forget,
            GateType::Output,
        ] {
            let activation = activations.get(&gate).unwrap().clone();
            neurons.insert(
                gate,
                NeuronV2::new_with_initializer(
                    input_size + 1,
                    activation,
                    weight_initializer,
                    bias_initializer,
                    weight_optimizer.clone(),
                    bias_optimizer.clone(),
                ),
            );
        }

        Self {
            neurons,
            state: LSTMState { h: 0.0, c: 0.0 },
            combination_fn,
        }
    }

    fn prepare_input(&self, input: &Vec<f64>) -> Vec<f64> {
        let mut full_input = input.clone();
        full_input.push(self.state.h);
        full_input
    }
}

impl RecurrentCell for LSTMCell {
    fn forward(&mut self, input: &Vec<f64>) -> (f64, f64) {
        let full_input = self.prepare_input(input);

        let candidate = self.neurons.get_mut(&GateType::Candidate).unwrap().forward(&full_input);
        let input_gate = self.neurons.get_mut(&GateType::Input).unwrap().forward(&full_input);
        let forget_gate = self.neurons.get_mut(&GateType::Forget).unwrap().forward(&full_input);
        let output_gate = self.neurons.get_mut(&GateType::Output).unwrap().forward(&full_input);

        let new_c = if let Some(comb) = &self.combination_fn {
            comb(forget_gate, self.state.c, input_gate, candidate)
        } else {
            forget_gate * self.state.c + input_gate * candidate
        };

        let new_h = output_gate * new_c.tanh();

        self.state.c = new_c;
        self.state.h = new_h;

        (new_h, new_c)
    }

    fn backward(
        &mut self,
        input: &Vec<f64>,
        grad_h: f64,
        grad_c: f64,
    ) -> (Vec<f64>, f64) {
        let full_input = self.prepare_input(input);

        let output_gate = self.neurons.get(&GateType::Output).unwrap().last_a;
        let forget_gate = self.neurons.get(&GateType::Forget).unwrap().last_a;
        let input_gate = self.neurons.get(&GateType::Input).unwrap().last_a;
        let candidate = self.neurons.get(&GateType::Candidate).unwrap().last_a;

        let tanh_c = self.state.c.tanh();
        let d_tanh_c = 1.0 - tanh_c.powi(2);

        let d_output = grad_h * tanh_c;
        let d_c = grad_h * output_gate * d_tanh_c + grad_c;

        let d_forget = d_c * self.state.c;
        let d_input = d_c * candidate;
        let d_candidate = d_c * input_gate;

        let grad_a = self.neurons.get_mut(&GateType::Candidate).unwrap().backward(d_candidate);
        let grad_b = self.neurons.get_mut(&GateType::Input).unwrap().backward(d_input);
        let grad_c = self.neurons.get_mut(&GateType::Forget).unwrap().backward(d_forget);
        let grad_d = self.neurons.get_mut(&GateType::Output).unwrap().backward(d_output);

        let grad_input: Vec<f64> = grad_a
            .iter()
            .zip(grad_b.iter())
            .zip(grad_c.iter())
            .zip(grad_d.iter())
            .map(|(((a, b), c), d)| a + b + c + d)
            .collect();

        let grad_h_prev = grad_input.last().copied().unwrap_or(0.0);
        let grad_input = grad_input[..grad_input.len() - 1].to_vec();

        let grad_c_prev = d_c * forget_gate;

        (grad_input, grad_c_prev + grad_h_prev)
    }

    fn reset_state(&mut self) {
        self.state = LSTMState { h: 0.0, c: 0.0 };
        for neuron in self.neurons.values_mut() {
            neuron.reset();
        }
    }

    fn update(&mut self) {
        for neuron in self.neurons.values_mut() {
            neuron.update();
        }
    }
}


impl std::fmt::Debug for LSTMCell {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LSTMCell")
            .field("neurons", &self.neurons)
            .field("state", &self.state)
            .finish()
    }
}

impl Clone for LSTMCell {
    fn clone(&self) -> Self {
        LSTMCell {
            neurons: self.neurons.clone(),
            state: self.state.clone(),
            combination_fn: None, //  Não clona a função customizada, usa None ou define manual depois
        }
    }
}