/// Define um gerador dinâmico de parâmetros do SSM baseado na entrada x_t
/// Exemplo simplificado de um "MLP" — função linear por entrada
pub struct DynamicParamGenerator {
    pub W_a: Vec<Vec<f64>>, // Peso para gerar A_t (d_out x d_in)
    pub W_b: Vec<Vec<f64>>, // Peso para gerar B_t (d_out x d_in)
    pub W_dt: Vec<Vec<f64>>, // Peso para gerar Delta_t (d_out x d_in)
    pub bias_a: Vec<f64>,
    pub bias_b: Vec<f64>,
    pub bias_dt: Vec<f64>,
}

impl DynamicParamGenerator {
    pub fn new(input_dim: usize, state_dim: usize) -> Self {
        // Inicializa pesos com valores pequenos ou zero
        fn zeros(rows: usize, cols: usize) -> Vec<Vec<f64>> {
            vec![vec![0.0; cols]; rows]
        }

        fn zeros_1d(n: usize) -> Vec<f64> {
            vec![0.0; n]
        }

        Self {
            W_a: zeros(state_dim, input_dim),
            W_b: zeros(state_dim, input_dim),
            W_dt: zeros(state_dim, input_dim),
            bias_a: zeros_1d(state_dim),
            bias_b: zeros_1d(state_dim),
            bias_dt: zeros_1d(state_dim),
        }
    }

    /// Gera os vetores A_t, B_t, Delta_t a partir da entrada `x_t`
    pub fn generate(&self, x: &Vec<f64>) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let linear = |W: &Vec<Vec<f64>>, b: &Vec<f64>| {
            W.iter().enumerate().map(|(i, row)| {
                row.iter().zip(x).map(|(w, xi)| w * xi).sum::<f64>() + b[i]
            }).collect()
        };

        let A_t = linear(&self.W_a, &self.bias_a);
        let B_t = linear(&self.W_b, &self.bias_b);
        let Delta_t = linear(&self.W_dt, &self.bias_dt);

        (A_t, B_t, Delta_t)
    }
}
