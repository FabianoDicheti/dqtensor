use crate::lstm::lstm_stack::LSTMStack;
use crate::mlp::dense::DenseLayer;
use crate::net_propagation::loss::{mse, mse_derivative};

///  Uma etapa de treino com Backpropagation Through Time (BPTT)
pub fn train_step(
    lstm_stack: &mut LSTMStack,
    dense: &mut DenseLayer,
    input_sequence: &Vec<Vec<f64>>,
    target: &Vec<f64>,
) -> f64 {
    //  Forward na LSTM
    let (hidden_states, _) = lstm_stack.forward(input_sequence);

    //  Acessa a última camada
    let last_layer_hidden = hidden_states.last().expect("Hidden states está vazio");

    //  Acessa o último timestep dessa camada
    let last_hidden_map = last_layer_hidden.last().expect("Último timestep está vazio");

    // # Ordena as chaves para garantir consistência na ordem
    let mut keys: Vec<_> = last_hidden_map.keys().collect();
    keys.sort();

    // # Coleta os valores dos hidden states na mesma ordem das chaves
    let last_hidden = keys
        .iter()
        .map(|&k| *last_hidden_map.get(k).unwrap())
        .collect::<Vec<f64>>();

    // # Forward na Dense
    let output = dense.forward(&last_hidden);

    // # Cálculo da perda (loss)
    let loss = mse(&output, target);

    // # Gradiente da perda
    let grad_output = mse_derivative(&output, target);

    // Backward na Dense
    let grad_dense = dense.backward(&grad_output);
    dense.update();

    //  Backward na LSTM
    lstm_stack.backward(&grad_dense);
    lstm_stack.update();

    loss
}
