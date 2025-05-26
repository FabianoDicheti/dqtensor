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
        println!("Backward da LSTMLayer ainda precisa ser implementado.");
    }
}
