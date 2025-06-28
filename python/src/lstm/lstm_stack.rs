use crate::lstm::lstm_cell::RecurrentCell;
use crate::lstm::lstm_layer::LSTMLayer;
use std::collections::HashMap;

/// Representa uma pilha de camadas LSTM (recurrent stack)
#[derive(Debug)]
pub struct LSTMStack {
    pub layers: Vec<LSTMLayer>,
}

impl LSTMStack {
    /// nova pilha 
    pub fn new(layers: Vec<LSTMLayer>) -> Self {
        Self { layers }
    }

    /// forward completo pela pilha
    pub fn forward(
        &mut self,
        input_sequence: &Vec<Vec<f64>>,
    ) -> (
        Vec<Vec<HashMap<String, f64>>>, // hidden_states por camada
        Vec<Vec<HashMap<String, f64>>>, // cell_states por camada
    ) {
        let mut hidden_states = vec![];
        let mut cell_states = vec![];

        let mut current_input = input_sequence.clone();

        for layer in self.layers.iter_mut() {
            let layer_output = layer.forward(&current_input);
            hidden_states.push(layer_output.clone());
            cell_states.push(layer.cell_history.clone());

            // Prepara entrada da próxima camada
            current_input = layer_output
                .iter()
                .map(|map| map.values().cloned().collect())
                .collect();
        }

        (hidden_states, cell_states)
    }

    /// Executa backpropagation através do tempo e camadas (BPTT)
    pub fn backward(&mut self, grad_output: &Vec<f64>) {
        let mut current_grad = grad_output.clone();

        for layer in self.layers.iter_mut().rev() {
            current_grad = layer.backward(&current_grad);
        }
    }

    /// Atualiza todos os neurônios da pilha após o backward
    pub fn update(&mut self) {
        for layer in self.layers.iter_mut() {
            for cell in layer.cells.values_mut() {
                cell.update();
            }
        }
    }

    /// Reseta os estados internos das células
    pub fn reset(&mut self) {
        for layer in self.layers.iter_mut() {
            for cell in layer.cells.values_mut() {
                cell.reset_state();
            }
            layer.hidden_history.clear();
            layer.cell_history.clear();
        }
    }
}
