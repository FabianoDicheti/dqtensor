use crate::lstm::lstm_cell::LSTMCell;

#[derive(Debug)]
pub struct LSTMLayer {
    pub cells: Vec<LSTMCell>,
}

impl LSTMLayer {
    pub fn new(cells: Vec<LSTMCell>) -> Self {
        Self { cells }
    }

    pub fn forward(&mut self, sequencia: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut outputs = vec![];

        for entrada in sequencia {
            let mut saida_passo = vec![];
            for cell in self.cells.iter_mut() {
                let saida = cell.forward(entrada);
                saida_passo.extend(saida);
            }
            outputs.push(saida_passo);
        }

        outputs
    }

    pub fn backward(
        &mut self,
        sequencia: &Vec<Vec<f64>>,
        d_loss_d_output: &Vec<Vec<f64>>,
    ) {
        for (entrada, grad_saida) in sequencia.iter().zip(d_loss_d_output.iter()) {
            for (cell, grad_cell_saida) in self.cells.iter_mut().zip(grad_saida.chunks(grad_saida.len() / self.cells.len())) {
                let _ = cell.backward(entrada, &grad_cell_saida.to_vec());
            }
        }
    }
}
