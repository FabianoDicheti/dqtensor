use dqtensor::data::ingestion::DataFrame;
use dqtensor::f_not_linear::activation::ActivationFunction;
use dqtensor::lstm::lstm_cell::{GateType, LSTMCell, RecurrentCell}; 
use dqtensor::lstm::lstm_layer::LSTMLayer;
use dqtensor::lstm::lstm_stack::LSTMStack;
use dqtensor::mlp::dense::DenseLayer; 
use dqtensor::optimizers::bp_optimizers::{Adam, Optimizer};

use std::collections::HashMap;
use std::error::Error;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;


fn mse_loss(predictions: &[f64], targets: &[f64]) -> f64 {
    if predictions.len() != targets.len() || predictions.is_empty() {
        return 0.0; 
    }
    let n = predictions.len() as f64;
    predictions.iter().zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>() / n
}

fn mse_loss_derivative(prediction: f64, target: f64) -> f64 {
    2.0 * (prediction - target)
}

// Weight Initializer: Random Uniform
fn random_uniform_init(size: usize) -> Vec<f64> {
    let mut rng = thread_rng();
    (0..size).map(|_| rng.gen_range(-0.1..0.1)).collect()
}

// Bias Initializer
fn zeros_init() -> f64 {
    0.0
}


fn train_model(
    model_name: &str,
    lstm_stack: &mut LSTMStack,
    dense_layer: &mut DenseLayer,
    sequences: &Vec<Vec<f64>>,
    targets: &Vec<f64>,
    epochs: usize,
) -> Result<f64, Box<dyn Error>> {
    println!("\nTraining {}...", model_name);
    let num_sequences = sequences.len();
    if num_sequences == 0 {
        return Err("No sequences provided for training.".into());
    }

    let mut indices: Vec<usize> = (0..num_sequences).collect();
    let mut final_avg_loss = 0.0;

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        indices.shuffle(&mut thread_rng());

        for &i in &indices {
            let input_sequence = &sequences[i];
            let target_value = targets[i];

            let reshaped_input: Vec<Vec<f64>> = input_sequence.iter().map(|&val| vec![val]).collect();

            // --- Forward  ---
            lstm_stack.reset(); // Reset state for each sequence
            let (hidden_states, _) = lstm_stack.forward(&reshaped_input);

            let last_layer_hidden_states = hidden_states.last()
                .ok_or_else(|| format!("[{}] No hidden states from LSTM", model_name))?;
            let final_timestep_hidden_state_map = last_layer_hidden_states.last()
                .ok_or_else(|| format!("[{}] Empty sequence output from LSTM", model_name))?;
            let lstm_output_vec: Vec<f64> = final_timestep_hidden_state_map.values().cloned().collect();

            if lstm_output_vec.is_empty() {
                return Err(format!("[{}] LSTM output vector is empty.", model_name).into());
            }

            let prediction_vec = dense_layer.forward(&lstm_output_vec);
            let prediction = prediction_vec.get(0).copied().unwrap_or(0.0);

            // manual tech debt
            let loss = (prediction - target_value).powi(2);
            epoch_loss += loss;

            // backward
            let grad_prediction = mse_loss_derivative(prediction, target_value);
            let grad_dense_input = dense_layer.backward(&vec![grad_prediction]);
            lstm_stack.backward(&grad_dense_input);

            // update weights
            dense_layer.update();
            lstm_stack.update();
        }

        let avg_loss = epoch_loss / num_sequences as f64;
        println!("Epoch {}/{}, Average Loss: {:.6}", epoch + 1, epochs, avg_loss);
        if epoch == epochs - 1 {
            final_avg_loss = avg_loss;
        }
    }
    Ok(final_avg_loss)
}


pub fn main() -> Result<(), Box<dyn Error>> {
    println!("Starting LSTM Test for Jena Climate dataset...");

    let data_path = "jena_climate.csv";
    let df = DataFrame::from_file(data_path)?;
    println!("Dataset loaded successfully.");

    let temp_col_index = df.df_cols.iter().position(|name| name == "Temp")
        .ok_or_else(|| Box::<dyn Error>::from("Column 'Temp' not found"))?;
    let temp_values_str: &Vec<String> = &df.columns[temp_col_index];
    let mut temp_values: Vec<f64> = temp_values_str.iter()
        .map(|s| s.parse::<f64>())
        .collect::<Result<Vec<f64>, _>>()?;
    println!("Extracted 'Temp' column with {} values.", temp_values.len());

    let (min_temp, max_temp) = normalize_min_max(&mut temp_values);
    println!("Temperature data normalized (Min: {}, Max: {}).", min_temp, max_temp);

    let look_back = 10;
    let (sequences, targets) = create_sequences(&temp_values, look_back);
    if sequences.is_empty() {
        return Err("Not enough data to create sequences.".into());
    }
    println!("Created {} sequences of length {}.\n", sequences.len(), look_back);

    // --- Model Definitions & Instantiation ---
    println!("Defining LSTM models...");
    let lstm_input_dim = 1;
    let hidden_dim = 50;
    let mlp_output_dim = 1;
    let learning_rate = 0.001;
    let epochs = 10;

    // --- Model 1: Traditional LSTM ---
    println!("Defining Traditional LSTM (Model 1)...");
    let mut traditional_activations = HashMap::new();
    traditional_activations.insert(GateType::Input, ActivationFunction::Sigmoid);
    traditional_activations.insert(GateType::Forget, ActivationFunction::Sigmoid);
    traditional_activations.insert(GateType::Output, ActivationFunction::Sigmoid);
    traditional_activations.insert(GateType::Candidate, ActivationFunction::Tanh);

    let mut traditional_cells = HashMap::new();
    for i in 0..hidden_dim {
        let neuron_input_size = lstm_input_dim + 1; 
        let weight_opt = Box::new(Adam::new(learning_rate, neuron_input_size)) as Box<dyn Optimizer>;
        let bias_opt = Box::new(Adam::new(learning_rate, 1)) as Box<dyn Optimizer>; 
        traditional_cells.insert(
            format!("cell_{}", i),
            LSTMCell::new(
                lstm_input_dim,
                traditional_activations.clone(),
                random_uniform_init,
                zeros_init, 
                weight_opt,
                bias_opt,
                None, 
            ),
        );
    }
    let mut traditional_lstm_stack = LSTMStack::new(vec![LSTMLayer::new(traditional_cells)]);
    let dense_input_size = hidden_dim;
    let dense_weight_size = dense_input_size * mlp_output_dim;
    let dense_bias_size = mlp_output_dim;
    let mut traditional_dense_layer = DenseLayer::new(
        dense_input_size,
        mlp_output_dim,
        ActivationFunction::Linear,
        Box::new(Adam::new(learning_rate, dense_weight_size)) as Box<dyn Optimizer>,
        Box::new(Adam::new(learning_rate, dense_bias_size)) as Box<dyn Optimizer>,
    );
    println!("Traditional LSTM defined.");

    // --- Model 2: Custom LSTM 1 (ReLU Gates) ---
    println!("Defining Custom LSTM 1 (Model 2 - ReLU Gates)...");
    let mut custom_activations1 = HashMap::new();
    custom_activations1.insert(GateType::Input, ActivationFunction::Relu); // *
    custom_activations1.insert(GateType::Forget, ActivationFunction::Sigmoid);
    custom_activations1.insert(GateType::Output, ActivationFunction::Relu); // *
    custom_activations1.insert(GateType::Candidate, ActivationFunction::Tanh);

    let mut custom_cells1 = HashMap::new();
    for i in 0..hidden_dim {
        let neuron_input_size = lstm_input_dim + 1;
        let weight_opt = Box::new(Adam::new(learning_rate, neuron_input_size)) as Box<dyn Optimizer>;
        let bias_opt = Box::new(Adam::new(learning_rate, 1)) as Box<dyn Optimizer>;
        custom_cells1.insert(
            format!("cell_{}", i),
            LSTMCell::new(
                lstm_input_dim,
                custom_activations1.clone(),
                random_uniform_init, zeros_init,
                weight_opt, bias_opt, None,
            ),
        );
    }
    let mut custom_lstm_stack1 = LSTMStack::new(vec![LSTMLayer::new(custom_cells1)]);
    let mut custom_dense_layer1 = DenseLayer::new(
        dense_input_size, mlp_output_dim, ActivationFunction::Linear,
        Box::new(Adam::new(learning_rate, dense_weight_size)) as Box<dyn Optimizer>,
        Box::new(Adam::new(learning_rate, dense_bias_size)) as Box<dyn Optimizer>,
    );
    println!("Custom LSTM 1 defined.");

    // --- Model 3: Custom LSTM 2 (Mixed Gates - Using ELU instead of LeakyReLU) ---
    println!("Defining Custom LSTM 2 (Model 3 - ELU/Sigmoid/ReLU Mix)...");
    let mut custom_activations2 = HashMap::new();
    custom_activations2.insert(GateType::Input, ActivationFunction::ELU); // *
    custom_activations2.insert(GateType::Forget, ActivationFunction::Sigmoid);
    custom_activations2.insert(GateType::Output, ActivationFunction::Sigmoid);
    custom_activations2.insert(GateType::Candidate, ActivationFunction::Relu);

    let mut custom_cells2 = HashMap::new();
    for i in 0..hidden_dim {
        let neuron_input_size = lstm_input_dim + 1;
        let weight_opt = Box::new(Adam::new(learning_rate, neuron_input_size)) as Box<dyn Optimizer>;
        let bias_opt = Box::new(Adam::new(learning_rate, 1)) as Box<dyn Optimizer>;
        custom_cells2.insert(
            format!("cell_{}", i),
            LSTMCell::new(
                lstm_input_dim, 
                custom_activations2.clone(),
                random_uniform_init, zeros_init,
                weight_opt, bias_opt, None,
            ),
        );
    }
    let mut custom_lstm_stack2 = LSTMStack::new(vec![LSTMLayer::new(custom_cells2)]);
    let mut custom_dense_layer2 = DenseLayer::new(
        dense_input_size, mlp_output_dim, ActivationFunction::Linear,
        Box::new(Adam::new(learning_rate, dense_weight_size)) as Box<dyn Optimizer>,
        Box::new(Adam::new(learning_rate, dense_bias_size)) as Box<dyn Optimizer>,
    );
    println!("Custom LSTM 2 defined.");

    // TRAINING
    let traditional_loss = train_model("Traditional LSTM", &mut traditional_lstm_stack, &mut traditional_dense_layer, &sequences, &targets, epochs)?;
    let custom1_loss = train_model("Custom LSTM 1 (ReLU)", &mut custom_lstm_stack1, &mut custom_dense_layer1, &sequences, &targets, epochs)?;
    let custom2_loss = train_model("Custom LSTM 2 (Mixed)", &mut custom_lstm_stack2, &mut custom_dense_layer2, &sequences, &targets, epochs)?;

  
    println!("\n--- Experiment Comparison ---");
    println!("{:<30} | {:<20}", "Model Configuration", "Final Avg Train MSE");
    println!("{:-<53}", "");
    println!("{:<30} | {:<20.6}", "Traditional LSTM", traditional_loss);
    println!("{:<30} | {:<20.6}", "Custom LSTM 1 (ReLU Gates)", custom1_loss);
    println!("{:<30} | {:<20.6}", "Custom LSTM 2 (Mixed Gates)", custom2_loss);
    println!("-----------------------------------------------------");

    println!("\nLSTM Test script finished.");

    Ok(())
}


//  Helper //

fn create_sequences(data: &[f64], look_back: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut sequences = Vec::new();
    let mut targets = Vec::new();
    if data.len() <= look_back {
        return (sequences, targets);
    }
    for i in 0..(data.len() - look_back) {
        let sequence = data[i..i + look_back].to_vec();
        let target = data[i + look_back];
        sequences.push(sequence);
        targets.push(target);
    }
    (sequences, targets)
}


fn normalize_min_max(data: &mut [f64]) -> (f64, f64) {
    if data.is_empty() {
        return (0.0, 1.0);
    }
    let mut min = data[0];
    let mut max = data[0];
    for &val in data.iter() {
        if val < min { min = val; }
        if val > max { max = val; }
    }
    let range = max - min;
    if range.abs() < 1e-10 {
        for val in data.iter_mut() {
            *val = 0.0;
        }
    } else {
        for val in data.iter_mut() {
            *val = (*val - min) / range;
        }
    }
    (min, max)
}
