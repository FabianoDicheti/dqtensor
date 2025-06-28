use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

// Imports do seu projeto Rust
use crate::lstm::lstm_stack::LSTMStack as RustLSTMStack;
use crate::lstm::lstm_layer::LSTMLayer;
use crate::lstm::lstm_cell::{LSTMCell, GateType, RecurrentCell};
use crate::mlp::dense::DenseLayer as RustDenseLayer;
use crate::optimizers::bp_optimizers::{Adam, Optimizer};
use crate::f_not_linear::activation::ActivationFunction;

// #########################
// 1. LSTM TRAINER (Classe Principal)
// #########################

#[pyclass]
pub struct LSTMTrainer {
    lstm_stack: RustLSTMStack,
    dense_layer: RustDenseLayer,
    input_dim: usize,
    hidden_dim: usize,
}

#[pymethods]
impl LSTMTrainer {
    #[new]
    fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        learning_rate: f64,
        activation_config: HashMap<String, String>
    ) -> PyResult<Self> {
        
        // Converter configurações de ativação
        let mut activations = HashMap::new();
        
        // Configuração padrão se não especificado
        let default_config = HashMap::from([
            ("input".to_string(), "sigmoid".to_string()),
            ("forget".to_string(), "sigmoid".to_string()),
            ("output".to_string(), "sigmoid".to_string()),
            ("candidate".to_string(), "tanh".to_string()),
        ]);
        
        let config = if activation_config.is_empty() { &default_config } else { &activation_config };
        
        for (gate_str, act_str) in config {
            let gate_type = match gate_str.as_str() {
                "input" => GateType::Input,
                "forget" => GateType::Forget,
                "output" => GateType::Output,
                "candidate" => GateType::Candidate,
                _ => continue,
            };
            
            let activation = match act_str.as_str() {
                "sigmoid" => ActivationFunction::Sigmoid,
                "tanh" => ActivationFunction::Tanh,
                "relu" => ActivationFunction::Relu,
                "linear" => ActivationFunction::Linear,
                "elu" => ActivationFunction::ELU,
                _ => ActivationFunction::Sigmoid,
            };
            
            activations.insert(gate_type, activation);
        }
        
        // Função de inicialização de pesos
        let random_uniform_init = |size: usize| -> Vec<f64> {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            (0..size).map(|_| rng.gen_range(-0.1..0.1)).collect()
        };
        
        let zeros_init = || -> f64 { 0.0 };
        
        // Criar células LSTM
        let mut cells = HashMap::new();
        for i in 0..hidden_dim {
            let neuron_input_size = input_dim + 1;
            let weight_opt = Box::new(Adam::new(learning_rate, neuron_input_size)) as Box<dyn Optimizer>;
            let bias_opt = Box::new(Adam::new(learning_rate, 1)) as Box<dyn Optimizer>;
            
            cells.insert(
                format!("cell_{}", i),
                LSTMCell::new(
                    input_dim,
                    activations.clone(),
                    random_uniform_init,
                    zeros_init,
                    weight_opt,
                    bias_opt,
                    None,
                ),
            );
        }
        
        // Criar LSTM Stack
        let lstm_layer = LSTMLayer::new(cells);
        let lstm_stack = RustLSTMStack::new(vec![lstm_layer]);
        
        // Criar Dense Layer
        let dense_weight_size = hidden_dim * output_dim;
        let dense_bias_size = output_dim;
        let dense_layer = RustDenseLayer::new(
            hidden_dim,
            output_dim,
            ActivationFunction::Linear,
            Box::new(Adam::new(learning_rate, dense_weight_size)) as Box<dyn Optimizer>,
            Box::new(Adam::new(learning_rate, dense_bias_size)) as Box<dyn Optimizer>,
        );
        
        Ok(LSTMTrainer {
            lstm_stack,
            dense_layer,
            input_dim,
            hidden_dim,
        })
    }
    
    fn train_epoch(&mut self, sequences: Vec<Vec<f64>>, targets: Vec<f64>) -> PyResult<f64> {
        if sequences.len() != targets.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Sequences and targets must have same length"));
        }
        
        let mut total_loss = 0.0;
        let num_sequences = sequences.len();
        
        // Embaralhar índices
        let mut indices: Vec<usize> = (0..num_sequences).collect();
        use rand::seq::SliceRandom;
        indices.shuffle(&mut rand::thread_rng());
        
        for &i in &indices {
            let input_sequence = &sequences[i];
            let target_value = targets[i];
            
            // Reshape input para LSTM (cada valor vira um vetor)
            let reshaped_input: Vec<Vec<f64>> = input_sequence.iter()
                .map(|&val| vec![val])
                .collect();
            
            // --- Forward Pass ---
            self.lstm_stack.reset();
            let (hidden_states, _) = self.lstm_stack.forward(&reshaped_input);
            
            // Pegar último estado da última camada
            let last_layer_hidden_states = hidden_states.last()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No hidden states from LSTM"))?;
            
            let final_timestep_hidden_state_map = last_layer_hidden_states.last()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Empty sequence output from LSTM"))?;
            
            // Converter HashMap para Vec
            let lstm_output_vec: Vec<f64> = final_timestep_hidden_state_map.values().cloned().collect();
            
            if lstm_output_vec.is_empty() {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("LSTM output vector is empty"));
            }
            
            // Dense layer forward
            let prediction_vec = self.dense_layer.forward(&lstm_output_vec);
            let prediction = prediction_vec.get(0).copied().unwrap_or(0.0);
            
            // Calcular loss (MSE)
            let loss = (prediction - target_value).powi(2);
            total_loss += loss;
            
            // --- Backward Pass ---
            let grad_prediction = 2.0 * (prediction - target_value); // MSE derivative
            let grad_dense_input = self.dense_layer.backward(&vec![grad_prediction]);
            self.lstm_stack.backward(&grad_dense_input);
            
            // --- Update Weights ---
            self.dense_layer.update();
            self.lstm_stack.update();
        }
        
        Ok(total_loss / num_sequences as f64)
    }
    
    fn train(&mut self, sequences: Vec<Vec<f64>>, targets: Vec<f64>, epochs: usize) -> PyResult<Vec<f64>> {
        let mut losses = Vec::new();
        
        println!(" Iniciando treinamento LSTM...");
        println!("   Sequências: {}", sequences.len());
        println!("   Épocas: {}", epochs);
        
        for epoch in 0..epochs {
            let avg_loss = self.train_epoch(sequences.clone(), targets.clone())?;
            losses.push(avg_loss);
            
            if epoch % 10 == 0 || epoch == epochs - 1 {
                println!("Época {}/{}, Loss: {:.6}", epoch + 1, epochs, avg_loss);
            }
        }
        
        println!(" Treinamento concluído!");
        Ok(losses)
    }
    
    fn predict(&mut self, sequence: Vec<f64>) -> PyResult<f64> {
        // Reshape input
        let reshaped_input: Vec<Vec<f64>> = sequence.iter()
            .map(|&val| vec![val])
            .collect();
        
        // Forward pass
        self.lstm_stack.reset();
        let (hidden_states, _) = self.lstm_stack.forward(&reshaped_input);
        
        // Pegar último estado
        let last_layer_hidden_states = hidden_states.last()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("No hidden states"))?;
        
        let final_timestep_hidden_state_map = last_layer_hidden_states.last()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Empty sequence output"))?;
        
        let lstm_output_vec: Vec<f64> = final_timestep_hidden_state_map.values().cloned().collect();
        
        // Dense layer
        let prediction_vec = self.dense_layer.forward(&lstm_output_vec);
        let prediction = prediction_vec.get(0).copied().unwrap_or(0.0);
        
        Ok(prediction)
    }
    
    fn get_info(&self) -> PyResult<HashMap<String, usize>> {
        let mut info = HashMap::new();
        info.insert("input_dim".to_string(), self.input_dim);
        info.insert("hidden_dim".to_string(), self.hidden_dim);
        Ok(info)
    }
}

// #########################
// 2. LSTM STACK (Acesso Direto)
// #########################

#[pyclass]
pub struct LSTMStack {
    inner: RustLSTMStack,
}

#[pymethods]
impl LSTMStack {
    #[new]
    fn new(
        input_dim: usize,
        hidden_dim: usize,
        learning_rate: f64,
        activation_config: Option<HashMap<String, String>>
    ) -> PyResult<Self> {
        
        let config = activation_config.unwrap_or_else(|| HashMap::from([
            ("input".to_string(), "sigmoid".to_string()),
            ("forget".to_string(), "sigmoid".to_string()),
            ("output".to_string(), "sigmoid".to_string()),
            ("candidate".to_string(), "tanh".to_string()),
        ]));
        
        let mut activations = HashMap::new();
        for (gate_str, act_str) in config {
            let gate_type = match gate_str.as_str() {
                "input" => GateType::Input,
                "forget" => GateType::Forget,
                "output" => GateType::Output,
                "candidate" => GateType::Candidate,
                _ => continue,
            };
            
            let activation = match act_str.as_str() {
                "sigmoid" => ActivationFunction::Sigmoid,
                "tanh" => ActivationFunction::Tanh,
                "relu" => ActivationFunction::Relu,
                "linear" => ActivationFunction::Linear,
                "elu" => ActivationFunction::ELU,
                _ => ActivationFunction::Sigmoid,
            };
            
            activations.insert(gate_type, activation);
        }
        
        // Criar células
        let random_uniform_init = |size: usize| -> Vec<f64> {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            (0..size).map(|_| rng.gen_range(-0.1..0.1)).collect()
        };
        
        let zeros_init = || -> f64 { 0.0 };
        
        let mut cells = HashMap::new();
        for i in 0..hidden_dim {
            let neuron_input_size = input_dim + 1;
            let weight_opt = Box::new(Adam::new(learning_rate, neuron_input_size)) as Box<dyn Optimizer>;
            let bias_opt = Box::new(Adam::new(learning_rate, 1)) as Box<dyn Optimizer>;
            
            cells.insert(
                format!("cell_{}", i),
                LSTMCell::new(
                    input_dim,
                    activations.clone(),
                    random_uniform_init,
                    zeros_init,
                    weight_opt,
                    bias_opt,
                    None,
                ),
            );
        }
        
        let lstm_layer = LSTMLayer::new(cells);
        let lstm_stack = RustLSTMStack::new(vec![lstm_layer]);
        
        Ok(LSTMStack { inner: lstm_stack })
    }
    
    fn forward(&mut self, sequences: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        let (hidden_states, _) = self.inner.forward(&sequences);
        
        // Simplificar output para Python
        let simplified_output: Vec<Vec<f64>> = hidden_states.into_iter()
            .map(|layer_states| {
                layer_states.into_iter()
                    .map(|timestep_states| {
                        timestep_states.values().cloned().collect::<Vec<f64>>()
                    })
                    .flatten()
                    .collect()
            })
            .flatten()
            .collect::<Vec<f64>>()
            .chunks(10) // Assumir chunks de 10 para simplificar
            .map(|chunk| chunk.to_vec())
            .collect();
        
        Ok(simplified_output)
    }
    
    fn backward(&mut self, gradients: Vec<f64>) -> PyResult<()> {
        self.inner.backward(&gradients);
        Ok(())
    }
    
    fn update(&mut self) -> PyResult<()> {
        self.inner.update();
        Ok(())
    }
    
    fn reset(&mut self) -> PyResult<()> {
        self.inner.reset();
        Ok(())
    }
}

// #########################
// 3. DENSE LAYER
// #########################

#[pyclass]
pub struct DenseLayer {
    inner: RustDenseLayer,
}

#[pymethods]
impl DenseLayer {
    #[new]
    fn new(
        input_size: usize,
        output_size: usize,
        activation: Option<String>,
        learning_rate: Option<f64>
    ) -> PyResult<Self> {
        
        let activation_fn = match activation.as_deref().unwrap_or("linear") {
            "sigmoid" => ActivationFunction::Sigmoid,
            "tanh" => ActivationFunction::Tanh,
            "relu" => ActivationFunction::Relu,
            "linear" => ActivationFunction::Linear,
            "elu" => ActivationFunction::ELU,
            _ => ActivationFunction::Linear,
        };
        
        let lr = learning_rate.unwrap_or(0.001);
        let weight_size = input_size * output_size;
        let bias_size = output_size;
        
        let weight_optimizer = Box::new(Adam::new(lr, weight_size)) as Box<dyn Optimizer>;
        let bias_optimizer = Box::new(Adam::new(lr, bias_size)) as Box<dyn Optimizer>;
        
        let dense = RustDenseLayer::new(
            input_size,
            output_size,
            activation_fn,
            weight_optimizer,
            bias_optimizer,
        );
        
        Ok(DenseLayer { inner: dense })
    }
    
    fn forward(&mut self, input: Vec<f64>) -> PyResult<Vec<f64>> {
        let output = self.inner.forward(&input);
        Ok(output)
    }
    
    fn backward(&mut self, gradients: Vec<f64>) -> PyResult<Vec<f64>> {
        let input_gradients = self.inner.backward(&gradients);
        Ok(input_gradients)
    }
    
    fn update(&mut self) -> PyResult<()> {
        self.inner.update();
        Ok(())
    }
}

// #########################
// 4. FUNÇÕES AUXILIARES
// #########################

#[pyfunction]
fn available_activations() -> Vec<String> {
    vec![
        "sigmoid".to_string(),
        "tanh".to_string(),
        "relu".to_string(),
        "linear".to_string(),
        "elu".to_string(),
    ]
}

#[pyfunction]
fn mse_loss(predictions: Vec<f64>, targets: Vec<f64>) -> f64 {
    if predictions.len() != targets.len() || predictions.is_empty() {
        return 0.0;
    }
    
    let n = predictions.len() as f64;
    predictions.iter().zip(targets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>() / n
}

#[pyfunction]
fn create_sequences(data: Vec<f64>, look_back: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
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

#[pyfunction]
fn normalize_min_max(data: Vec<f64>) -> (Vec<f64>, f64, f64) {
    if data.is_empty() {
        return (data, 0.0, 1.0);
    }
    
    let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let range = max_val - min_val;
    
    let normalized = if range.abs() < 1e-10 {
        vec![0.0; data.len()]
    } else {
        data.iter().map(|&val| (val - min_val) / range).collect()
    };
    
    (normalized, min_val, max_val)
}

// #########################
// 5. CLASSE SIMPLIFICADA (Compatibilidade)
// #########################

#[pyclass]
pub struct SimpleLSTM {
    trainer: LSTMTrainer,
    input_size: usize,
    activation: String,
}

#[pymethods]
impl SimpleLSTM {
    #[new]
    fn new(input_size: usize, activation: String) -> PyResult<Self> {
        let config = HashMap::from([
            ("input".to_string(), activation.clone()),
            ("forget".to_string(), "sigmoid".to_string()),
            ("output".to_string(), activation.clone()),
            ("candidate".to_string(), "tanh".to_string()),
        ]);
        
        let trainer = LSTMTrainer::new(1, input_size, 1, 0.001, config)?;
        
        Ok(SimpleLSTM {
            trainer,
            input_size,
            activation,
        })
    }
    
    fn forward(&mut self, input: Vec<f64>) -> PyResult<Vec<f64>> {
        // Usar como sequência de tamanho 1
        let prediction = self.trainer.predict(input)?;
        Ok(vec![prediction])
    }
    
    fn get_input_size(&self) -> usize {
        self.input_size
    }
    
    fn get_activation(&self) -> String {
        self.activation.clone()
    }
}

// #########################
// 6. MÓDULO PYTHON
// #########################

#[pymodule]
fn dqtensor(_py: Python, m: &PyModule) -> PyResult<()> {
    // Classes principais
    m.add_class::<LSTMTrainer>()?;
    m.add_class::<LSTMStack>()?;
    m.add_class::<DenseLayer>()?;
    m.add_class::<SimpleLSTM>()?;
    
    // Funções auxiliares
    m.add_function(wrap_pyfunction!(available_activations, m)?)?;
    m.add_function(wrap_pyfunction!(mse_loss, m)?)?;
    m.add_function(wrap_pyfunction!(create_sequences, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_min_max, m)?)?;
    
    Ok(())
}

