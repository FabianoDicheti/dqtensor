/// Representa um modelo contínuo de espaço de estado linear invariante no tempo (LTI)
/// do tipo:
///   dh/dt = A * h(t) + B * u(t)
///   y(t)  = C * h(t)
#[derive(Debug, Clone)]
pub struct ContinuousSSM {
    pub A: Vec<Vec<f64>>,  // Matriz A (dimensão d x d) — dinâmica do estado
    pub B: Vec<f64>,       // Vetor B (dimensão d) — entrada influencia o estado
    pub C: Vec<f64>,       // Vetor C (dimensão d) — estado influencia a saída
    pub state_dim: usize,  // d = dimensão do vetor de estado h(t)
}

impl ContinuousSSM {
    /// Construtor para o modelo contínuo
    /// Garante que A seja d x d, e que B e C tenham dimensão d
    pub fn new(A: Vec<Vec<f64>>, B: Vec<f64>, C: Vec<f64>) -> Self {
        let d = B.len();

        // Verificações de consistência dimensional
        assert!(
            A.len() == d && A[0].len() == d,
            "A deve ser uma matriz quadrada d x d"
        );
        assert!(
            C.len() == d,
            "C deve ter mesma dimensão de B (d)"
        );

        Self { A, B, C, state_dim: d }
    }
}

impl ContinuousSSM {
    /// Calcula a derivada do vetor de estado (dh/dt), dado o estado atual `h` e a entrada `u`
    ///
    /// Equação:
    ///   dh/dt = A * h + B * u
    pub fn compute_derivative(&self, h: &Vec<f64>, u: f64) -> Vec<f64> {
        assert!(
            h.len() == self.state_dim,
            "Estado h deve ter dimensão d"
        );

        // Inicializa vetor da derivada com zeros
        let mut dh_dt = vec![0.0; self.state_dim];

        // Para cada dimensão i do vetor de estado
        for i in 0..self.state_dim {
            let mut sum = 0.0;

            // Calcula o produto escalar entre a linha i da matriz A e o vetor h
            for j in 0..self.state_dim {
                sum += self.A[i][j] * h[j];
            }

            // Soma o termo B[i] * u à derivada
            dh_dt[i] = sum + self.B[i] * u;
        }

        dh_dt
    }

    /// Calcula a saída do sistema y(t), dado o vetor de estado `h`
    ///
    /// Equação:
    ///   y(t) = C * h(t)
    pub fn compute_output(&self, h: &Vec<f64>) -> f64 {
        assert!(
            h.len() == self.state_dim,
            "Estado h deve ter dimensão d"
        );

        // Produto escalar entre o vetor C e o vetor de estado h
        self.C.iter().zip(h).map(|(c, hi)| c * hi).sum()
    }
}
