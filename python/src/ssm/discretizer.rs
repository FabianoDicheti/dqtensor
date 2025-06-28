/// Representa um modelo de espaço de estado linear discreto do tipo:
///   h_{t+1} = A_bar * h_t + B_bar * u_t
///   y_t     = C * h_t
#[derive(Debug, Clone)]
pub struct DiscretizedSSM {
    pub A_bar: Vec<f64>,  // Vetor com os termos exp(A[i][i] * delta_t)
    pub B_bar: Vec<f64>,  // Vetor com os termos ajustados do B
    pub C: Vec<f64>,      // Vetor de saída (C)
    pub state_dim: usize, // Dimensão do estado (d)
}

impl DiscretizedSSM {
    /// Construtor do sistema discreto, usando A (diagonal), B, C e passo delta_t
    pub fn new_from_continuous_diagonal(A_diag: Vec<f64>, B: Vec<f64>, C: Vec<f64>, delta_t: f64) -> Self {
        let d = A_diag.len();
        assert!(B.len() == d && C.len() == d, "Todos os vetores devem ter dimensão d");

        let mut A_bar = vec![0.0; d];
        let mut B_bar = vec![0.0; d];

        for i in 0..d {
            A_bar[i] = (A_diag[i] * delta_t).exp(); // e^(A[i] * dt)

            // Evita divisão por zero se A[i] ≈ 0
            if A_diag[i].abs() > 1e-8 {
                B_bar[i] = ((A_bar[i] - 1.0) / A_diag[i]) * B[i];
            } else {
                // Caso limite para A[i] → 0: B_bar[i] ≈ B[i] * delta_t
                B_bar[i] = B[i] * delta_t;
            }
        }

        Self { A_bar, B_bar, C, state_dim: d }
    }

    /// Executa um passo de simulação discreta: h_{t+1} = A_bar * h_t + B_bar * u_t
    pub fn step(&self, h: &Vec<f64>, u: f64) -> (Vec<f64>, f64) {
        assert!(h.len() == self.state_dim, "Estado h inválido");

        let mut h_next = vec![0.0; self.state_dim];
        for i in 0..self.state_dim {
            h_next[i] = self.A_bar[i] * h[i] + self.B_bar[i] * u;
        }

        // Saída y_t = C * h
        let y = self.C.iter().zip(h).map(|(c, hi)| c * hi).sum();

        (h_next, y)
    }
}
