use super::parametric::DynamicParamGenerator;

/// Executa um SSM com parâmetros recalculados a cada passo a partir da entrada
pub struct DynamicSSMExecutor {
    pub param_generator: DynamicParamGenerator,
    pub C: Vec<f64>,  // vetor de saída
    pub state_dim: usize,
}

impl DynamicSSMExecutor {
    pub fn new(param_generator: DynamicParamGenerator, C: Vec<f64>) -> Self {
        let d = C.len();
        Self { param_generator, C, state_dim: d }
    }

    pub fn simulate(&self, inputs: &Vec<Vec<f64>>, h0: Vec<f64>) -> Vec<f64> {
        let mut h = h0.clone();
        let mut outputs = Vec::with_capacity(inputs.len());

        for x_t in inputs {
            let (A_t, B_t, Delta_t) = self.param_generator.generate(x_t);

            // Calcula A_bar e B_bar
            let mut A_bar = vec![0.0; self.state_dim];
            let mut B_bar = vec![0.0; self.state_dim];
            for i in 0..self.state_dim {
                let a_dt = A_t[i] * Delta_t[i];
                A_bar[i] = a_dt.exp();

                if A_t[i].abs() > 1e-8 {
                    B_bar[i] = ((A_bar[i] - 1.0) / A_t[i]) * B_t[i];
                } else {
                    B_bar[i] = B_t[i] * Delta_t[i];
                }
            }

            // Passo do SSM
            let mut h_next = vec![0.0; self.state_dim];
            for i in 0..self.state_dim {
                h_next[i] = A_bar[i] * h[i] + B_bar[i] * x_t.iter().copied().sum::<f64>(); // soma simplificada
            }

            // Saída
            let y_t: f64 = self.C.iter().zip(h.iter()).map(|(c, hi)| c * hi).sum();
            outputs.push(y_t);

            h = h_next;
        }

        outputs
    }
}
