use super::discretizer::DiscretizedSSM;

/// Simulador causal que aplica um SSM discreto sobre uma sequência de entradas
pub struct SSMCausalSimulator {
    pub model: DiscretizedSSM,
    pub h0: Vec<f64>,  // Estado inicial
}


impl SSMCausalSimulator {
    /// Cria um simulador com o modelo SSM e um estado inicial (pode ser zero)
    pub fn new(model: DiscretizedSSM, h0: Vec<f64>) -> Self {
        assert!(h0.len() == model.state_dim, "Estado inicial deve ter dimensão d");
        Self { model, h0 }
    }

    /// Executa a simulação causal sobre a sequência `inputs` e retorna as saídas
    ///
    /// Para cada passo t:
    ///   h_{t+1} = A_bar * h_t + B_bar * u_t
    ///   y_t     = C * h_t
    pub fn simulate(&self, inputs: &Vec<f64>) -> Vec<f64> {
        let mut h = self.h0.clone();
        let mut outputs = Vec::with_capacity(inputs.len());

        for &u in inputs.iter() {
            let (h_next, y_t) = self.model.step(&h, u);
            outputs.push(y_t);
            h = h_next;
        }

        outputs
    }

    /// Versão alternativa: também retorna a sequência completa de estados
    pub fn simulate_with_states(&self, inputs: &Vec<f64>) -> (Vec<f64>, Vec<Vec<f64>>) {
        let mut h = self.h0.clone();
        let mut outputs = Vec::with_capacity(inputs.len());
        let mut states = Vec::with_capacity(inputs.len() + 1);

        states.push(h.clone());  // salva h₀

        for &u in inputs.iter() {
            let (h_next, y_t) = self.model.step(&h, u);
            outputs.push(y_t);
            states.push(h_next.clone());
            h = h_next;
        }

        (outputs, states)
    }
}
