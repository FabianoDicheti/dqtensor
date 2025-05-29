use crate::lstm::lstm_cell::{LSTMCell, RecurrentCell};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct LSTMLayer {
    pub cells: HashMap<String, LSTMCell>,

    pub output_size: usize,

    pub hidden_history: Vec<HashMap<String, f64>>,
    pub cell_history: Vec<HashMap<String, f64>>,
}

impl LSTMLayer {
    pub fn new(cells: HashMap<String, LSTMCell>) -> Self {
        let output_size = cells.len();

        Self {
            cells,
            output_size,
            hidden_history: vec![],
            cell_history: vec![],
        }
    }

    pub fn forward(
        &mut self,
        input_sequence: &Vec<Vec<f64>>,
    ) -> Vec<HashMap<String, f64>> {
        let mut hidden_state: HashMap<String, f64> =
            self.cells.keys().map(|id| (id.clone(), 0.0)).collect();
        let mut cell_state: HashMap<String, f64> =
            self.cells.keys().map(|id| (id.clone(), 0.0)).collect();

        let mut hidden_outputs = vec![];

        for input in input_sequence {
            let mut current_hidden = HashMap::new();
            let mut current_cell = HashMap::new();

            for (id, cell) in self.cells.iter_mut() {
                let h_prev = *hidden_state.get(id).unwrap();
                let c_prev = *cell_state.get(id).unwrap();

                cell.state.h = h_prev;
                cell.state.c = c_prev;

                let (h, c) = cell.forward(input);

                current_hidden.insert(id.clone(), h);
                current_cell.insert(id.clone(), c);
            }

            hidden_outputs.push(current_hidden.clone());
            self.hidden_history.push(current_hidden.clone());
            self.cell_history.push(current_cell.clone());

            hidden_state = current_hidden;
            cell_state = current_cell;
        }

        hidden_outputs
    }

    pub fn backward(&mut self, grad_output: &Vec<f64>) -> Vec<f64> {
        let mut grad_h: HashMap<String, f64> = self
            .cells
            .keys()
            .enumerate()
            .map(|(i, id)| (id.clone(), grad_output.get(i).copied().unwrap_or(0.0)))
            .collect();

        let mut grad_c: HashMap<String, f64> =
            self.cells.keys().map(|id| (id.clone(), 0.0)).collect();

        let mut grad_inputs: Vec<Vec<f64>> = vec![];

        for timestep in (0..self.hidden_history.len()).rev() {
            let input = if timestep == 0 {
                vec![0.0; grad_output.len()]
            } else {
                self.hidden_history[timestep - 1]
                    .values()
                    .cloned()
                    .collect()
            };

            let mut grad_input_step = vec![0.0; input.len()];

            for (id, cell) in self.cells.iter_mut() {
                let h_prev = if timestep == 0 {
                    0.0
                } else {
                    *self.hidden_history[timestep - 1].get(id).unwrap()
                };

                let c_prev = if timestep == 0 {
                    0.0
                } else {
                    *self.cell_history[timestep - 1].get(id).unwrap()
                };

                cell.state.h = h_prev;
                cell.state.c = c_prev;

                let (grad_input, grad_c_prev) = cell.backward(
                    &input,
                    *grad_h.get(id).unwrap(),
                    *grad_c.get(id).unwrap(),
                );

                for (i, val) in grad_input.iter().enumerate() {
                    grad_input_step[i] += val;
                }

                *grad_c.get_mut(id).unwrap() = grad_c_prev;
            }

            grad_inputs.push(grad_input_step);
        }

        grad_inputs.reverse();

        grad_inputs
            .iter()
            .fold(vec![0.0; grad_inputs[0].len()], |mut acc, item| {
                for (i, v) in item.iter().enumerate() {
                    acc[i] += v;
                }
                acc
            })
    }

    pub fn update(&mut self) {
        for cell in self.cells.values_mut() {
            cell.update();
        }
    }

    pub fn reset(&mut self) {
        self.hidden_history.clear();
        self.cell_history.clear();
        for cell in self.cells.values_mut() {
            cell.reset_state();
        }
    }
}
