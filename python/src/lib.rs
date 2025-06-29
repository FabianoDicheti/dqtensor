use pyo3::prelude::*;
use std::collections::HashMap;

// #########################
// 1. LSTM TRAINER SIMPLES (Funcional)
// #########################

#[pyclass]
pub struct LSTMTrainer {
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    learning_rate: f64,
    activation_config: HashMap<String, String>,
    
    // Pesos simulados (para demonstração)
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    
    // Estado interno
    hidden_state: Vec<f64>,
    cell_state: Vec<f64>,
}

#[pymethods]
impl LSTMTrainer {
    #[new]
    fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        learning_rate: f64,
        activation_config: Option<HashMap<String, String>>
    ) -> PyResult<Self> {
        
        let config = activation_config.unwrap_or_else(|| {
            let mut default = HashMap::new();
            default.insert("input".to_string(), "sigmoid".to_string());
            default.insert("forget".to_string(), "sigmoid".to_string());
            default.insert("output".to_string(), "sigmoid".to_string());
            default.insert("candidate".to_string(), "tanh".to_string());
            default
        });
        
        // Inicializar pesos aleatórios
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let total_weights = (input_dim + hidden_dim + 1) * hidden_dim * 4 + hidden_dim * output_dim;
        let weights: Vec<Vec<f64>> = (0..4).map(|_| {
            (0..total_weights/4).map(|_| rng.gen_range(-0.1..0.1)).collect()
        }).collect();
        
        let biases: Vec<f64> = (0..hidden_dim + output_dim).map(|_| 0.0).collect();
        
        Ok(LSTMTrainer {
            input_dim,
            hidden_dim,
            output_dim,
            learning_rate,
            activation_config: config,
            weights,
            biases,
            hidden_state: vec![0.0; hidden_dim],
            cell_state: vec![0.0; hidden_dim],
        })
    }
    
    fn sigmoid(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    
    fn tanh(&self, x: f64) -> f64 {
        x.tanh()
    }
    
    fn relu(&self, x: f64) -> f64 {
        x.max(0.0)
    }
    
    fn elu(&self, x: f64) -> f64 {
        if x >= 0.0 { x } else { x.exp() - 1.0 }
    }
    
    fn apply_activation(&self, x: f64, activation: &str) -> f64 {
        match activation {
            "sigmoid" => self.sigmoid(x),
            "tanh" => self.tanh(x),
            "relu" => self.relu(x),
            "elu" => self.elu(x),
            "linear" => x,
            _ => self.sigmoid(x),
        }
    }
    
    fn forward_step(&mut self, input: &[f64]) -> Vec<f64> {
        // Simular forward pass LSTM
        let input_gate_act = self.activation_config.get("input").unwrap_or(&"sigmoid".to_string());
        let forget_gate_act = self.activation_config.get("forget").unwrap_or(&"sigmoid".to_string());
        let output_gate_act = self.activation_config.get("output").unwrap_or(&"sigmoid".to_string());
        let candidate_act = self.activation_config.get("candidate").unwrap_or(&"tanh".to_string());
        
        // Calcular gates (simplificado)
        let mut input_gate = 0.0;
        let mut forget_gate = 0.0;
        let mut output_gate = 0.0;
        let mut candidate = 0.0;
        
        // Combinar input e hidden state
        for i in 0..input.len().min(self.input_dim) {
            input_gate += input[i] * self.weights[0].get(i).unwrap_or(&0.1);
            forget_gate += input[i] * self.weights[1].get(i).unwrap_or(&0.1);
            output_gate += input[i] * self.weights[2].get(i).unwrap_or(&0.1);
            candidate += input[i] * self.weights[3].get(i).unwrap_or(&0.1);
        }
        
        for i in 0..self.hidden_dim {
            let h_val = self.hidden_state.get(i).unwrap_or(&0.0);
            input_gate += h_val * self.weights[0].get(self.input_dim + i).unwrap_or(&0.1);
            forget_gate += h_val * self.weights[1].get(self.input_dim + i).unwrap_or(&0.1);
            output_gate += h_val * self.weights[2].get(self.input_dim + i).unwrap_or(&0.1);
            candidate += h_val * self.weights[3].get(self.input_dim + i).unwrap_or(&0.1);
        }
        
        // Aplicar ativações
        input_gate = self.apply_activation(input_gate, input_gate_act);
        forget_gate = self.apply_activation(forget_gate, forget_gate_act);
        output_gate = self.apply_activation(output_gate, output_gate_act);
        candidate = self.apply_activation(candidate, candidate_act);
        
        // Atualizar cell state e hidden state
        for i in 0..self.hidden_dim {
            let old_cell = self.cell_state.get(i).unwrap_or(&0.0);
            let new_cell = forget_gate * old_cell + input_gate * candidate;
            self.cell_state[i] = new_cell;
            self.hidden_state[i] = output_gate * self.tanh(new_cell);
        }
        
        self.hidden_state.clone()
    }
    
    fn reset(&mut self) -> PyResult<()> {
        self.hidden_state = vec![0.0; self.hidden_dim];
        self.cell_state = vec![0.0; self.hidden_dim];
        Ok(())
    }
    
    fn predict(&mut self, sequence: Vec<f64>) -> PyResult<f64> {
        self.reset()?;
        
        let mut output = vec![0.0];
        
        // Processar sequência
        for &value in &sequence {
            output = self.forward_step(&[value]);
        }
        
        // Simular dense layer (média dos hidden states)
        let prediction = output.iter().sum::<f64>() / output.len() as f64;
        
        Ok(prediction)
    }
    
    fn train_epoch(&mut self, sequences: Vec<Vec<f64>>, targets: Vec<f64>) -> PyResult<f64> {
        if sequences.len() != targets.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Sequences and targets must have same length"));
        }
        
        let mut total_loss = 0.0;
        let num_sequences = sequences.len();
        
        for (sequence, target) in sequences.iter().zip(targets.iter()) {
            // Forward pass
            let prediction = self.predict(sequence.clone())?;
            
            // Calcular loss (MSE)
            let loss = (prediction - target).powi(2);
            total_loss += loss;
            
            // Simular backward pass (atualização simples dos pesos)
            let error = prediction - target;
            let learning_factor = self.learning_rate * error;
            
            // Atualizar pesos (simplificado)
            for weight_group in &mut self.weights {
                for weight in weight_group.iter_mut() {
                    *weight -= learning_factor * 0.01; // Gradiente simulado
                }
            }
        }
        
        Ok(total_loss / num_sequences as f64)
    }
    
    fn train(&mut self, sequences: Vec<Vec<f64>>, targets: Vec<f64>, epochs: usize) -> PyResult<Vec<f64>> {
        let mut losses = Vec::new();
        
        println!(" Iniciando treinamento LSTM...");
        println!("   Sequências: {}", sequences.len());
        println!("   Épocas: {}", epochs);
        println!("   Configuração: {:?}", self.activation_config);
        
        for epoch in 0..epochs {
            let avg_loss = self.train_epoch(sequences.clone(), targets.clone())?;
            losses.push(avg_loss);
            
            if epoch % 10 == 0 || epoch == epochs - 1 {
                println!("Época {}/{}, Loss: {:.6}", epoch + 1, epochs, avg_loss);
            }
        }
        
        println!("✅ Treinamento concluído!");
        Ok(losses)
    }
    
    fn get_info(&self) -> PyResult<HashMap<String, String>> {
        let mut info = HashMap::new();
        info.insert("input_dim".to_string(), self.input_dim.to_string());
        info.insert("hidden_dim".to_string(), self.hidden_dim.to_string());
        info.insert("output_dim".to_string(), self.output_dim.to_string());
        info.insert("learning_rate".to_string(), self.learning_rate.to_string());
        
        for (key, value) in &self.activation_config {
            info.insert(format!("activation_{}", key), value.clone());
        }
        
        Ok(info)
    }
}

// #########################
// 2. CLASSE SIMPLIFICADA (Compatibilidade)
// #########################

#[pyclass]
pub struct SimpleLSTM {
    input_size: usize,
    activation: String,
    weights: Vec<f64>,
}

#[pymethods]
impl SimpleLSTM {
    #[new]
    fn new(input_size: usize, activation: String) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let weights: Vec<f64> = (0..input_size * 4).map(|_| rng.gen_range(-0.1..0.1)).collect();
        
        Self {
            input_size,
            activation,
            weights,
        }
    }
    
    fn forward(&self, input: Vec<f64>) -> Vec<f64> {
        let mut result = input.clone();
        
        // Aplicar transformação com pesos
        for (i, val) in result.iter_mut().enumerate() {
            let weight = self.weights.get(i % self.weights.len()).unwrap_or(&1.0);
            *val *= weight;
        }
        
        // Aplicar função de ativação
        match self.activation.as_str() {
            "sigmoid" => {
                for val in &mut result {
                    *val = 1.0 / (1.0 + (-*val).exp());
                }
            },
            "tanh" => {
                for val in &mut result {
                    *val = val.tanh();
                }
            },
            "relu" => {
                for val in &mut result {
                    *val = val.max(0.0);
                }
            },
            "linear" => {
                // Manter valores como estão
            },
            "elu" => {
                for val in &mut result {
                    *val = if *val >= 0.0 { *val } else { (*val).exp() - 1.0 };
                }
            },
            "mish" => {
                for val in &mut result {
                    *val = *val * (1.0 + (*val).exp()).ln().tanh();
                }
            },
            "gaussian" => {
                for val in &mut result {
                    *val = (-(*val * *val)).exp();
                }
            },
            _ => {
                // Default: sigmoid
                for val in &mut result {
                    *val = 1.0 / (1.0 + (-*val).exp());
                }
            }
        }
        
        result
    }
    
    fn get_input_size(&self) -> usize {
        self.input_size
    }
    
    fn get_activation(&self) -> String {
        self.activation.clone()
    }
}

// #########################
// 3. FUNÇÕES AUXILIARES
// #########################

#[pyfunction]
fn available_activations() -> Vec<String> {
    vec![
        "sigmoid".to_string(),
        "tanh".to_string(),
        "relu".to_string(),
        "linear".to_string(),
        "elu".to_string(),
        "mish".to_string(),
        "gaussian".to_string(),
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
// 4. MÓDULO PYTHON
// #########################

#[pymodule]
fn dqtensor(_py: Python, m: &PyModule) -> PyResult<()> {
    // Classes principais
    m.add_class::<LSTMTrainer>()?;
    m.add_class::<SimpleLSTM>()?;
    
    // Funções auxiliares
    m.add_function(wrap_pyfunction!(available_activations, m)?)?;
    m.add_function(wrap_pyfunction!(mse_loss, m)?)?;
    m.add_function(wrap_pyfunction!(create_sequences, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_min_max, m)?)?;
    
    Ok(())
}

