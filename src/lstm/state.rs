#[derive(Debug, Clone)]
pub struct EstadoLSTM {
    pub memoria_c: Vec<f64>,
    pub memoria_h: Vec<f64>,
}

impl EstadoLSTM {
    pub fn new(tamanho: usize) -> Self {
        Self {
            memoria_c: vec![0.0; tamanho],
            memoria_h: vec![0.0; tamanho],
        }
    }
}
