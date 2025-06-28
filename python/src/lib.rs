use pyo3::prelude::*;

#[pyclass]
pub struct SimpleLSTM {
    input_size: usize,
    activation: String,
}

#[pymethods]
impl SimpleLSTM {
    #[new]
    pub fn new(input_size: usize, activation: String) -> Self {
        Self {
            input_size,
            activation,
        }
    }

    pub fn get_input_size(&self) -> usize {
        self.input_size
    }

    pub fn get_activation(&self) -> String {
        self.activation.clone()
    }

    pub fn forward(&self, input: Vec<f64>) -> Vec<f64> {
        // Implementação básica que apenas retorna o input
        input
    }
}

#[pyfunction]
pub fn available_activations() -> Vec<String> {
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

#[pymodule]
fn dqtensor(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SimpleLSTM>()?;
    m.add_function(wrap_pyfunction!(available_activations, m)?)?;
    Ok(())
}

