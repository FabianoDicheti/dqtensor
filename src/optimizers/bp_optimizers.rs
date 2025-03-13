pub trait Optimizer {
    /// Atualiza os parâmetros com base nos gradientes calculados.
    /// # Argumentos
    /// * `params` - Slice mutável dos parâmetros a serem atualizados.
    /// * `grads` - Slice dos gradientes correspondentes aos parâmetros.
    fn update(&mut self, params: &mut [f64], grads: &[f64]);
}

/// Implementação do otimizador Adam (Adaptive Moment Estimation).
/// # Campos
/// - `learning_rate`: Taxa de aprendizagem (η).
/// - `beta1`, `beta2`: Fatores de decaimento para os momentos.
/// - `epsilon`: Termo de estabilidade numérica.
/// - `time_step`: Contador de passos de atualização.
/// - `momentum_first`: Primeiro momento (média dos gradientes).
/// - `rmsprop_second`: Segundo momento (não centrado, da RMSprop).
pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    time_step: u64,
    momentum_first: Vec<f64>,
    rmsprop_second: Vec<f64>,
}


impl Adam {
    /// Cria uma nova instância do otimizador Adam.
    /// # Argumentos
    /// * `learning_rate` - Taxa de aprendizagem (η).
    /// * `beta1` - Decaimento do primeiro momento (ex: 0.9).
    /// * `beta2` - Decaimento do segundo momento (ex: 0.999).
    /// * `epsilon` - Termo de estabilidade (ex: 1e-8).
    /// * `param_size` - Número de parâmetros a serem otimizados.
    pub fn new(
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        param_size: usize,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            time_step: 0,
            momentum_first: vec![0.0; param_size],
            rmsprop_second: vec![0.0; param_size],
        }
    }
}

impl Optimizer for Adam {
    fn update(&mut self, params: &mut [f64], grads: &[f64]) {
        debug_assert_eq!(
            params.len(),
            grads.len(),
            "Params e grads devem ter o mesmo tamanho"
        );
        debug_assert_eq!(
            params.len(),
            self.momentum_first.len(),
            "Params e momentum_first devem ter o mesmo tamanho"
        );
        debug_assert_eq!(
            params.len(),
            self.rmsprop_second.len(),
            "Params e rmsprop_second devem ter o mesmo tamanho"
        );

        self.time_step += 1;
        let t = self.time_step;
        let beta1_t = self.beta1.powi(t as i32);
        let beta2_t = self.beta2.powi(t as i32);

        for i in 0..params.len() {
            // Atualiza estimativas dos momentos
            self.momentum_first[i] = self.beta1 * self.momentum_first[i] + (1.0 - self.beta1) * grads[i];
            self.rmsprop_second[i] = self.beta2 * self.rmsprop_second[i] + (1.0 - self.beta2) * grads[i].powi(2);

            // Aplica correção de viés
            let m_hat = self.momentum_first[i] / (1.0 - beta1_t);
            let v_hat = self.rmsprop_second[i] / (1.0 - beta2_t);

            // Atualiza parâmetros
            params[i] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Implementação do Nadam (Nesterov-accelerated Adam)
/// Incorpora a aceleração de Nesterov na estimativa do primeiro momento
pub struct Nadam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    time_step: u64,
    momentum_first: Vec<f64>,  // Primeiro momento (m)
    rmsprop_second: Vec<f64>, // Segundo momento (v)
}

impl Nadam {
    pub fn new(
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        param_size: usize,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            time_step: 0,
            momentum_first: vec![0.0; param_size],
            rmsprop_second: vec![0.0; param_size],
        }
    }
}

impl Optimizer for Nadam {
    fn update(&mut self, params: &mut [f64], grads: &[f64]) {
        debug_assert_eq!(params.len(), grads.len());
        debug_assert_eq!(params.len(), self.momentum_first.len());
        debug_assert_eq!(params.len(), self.rmsprop_second.len());

        self.time_step += 1;
        let t = self.time_step;

        // Cálculo das potências beta para correção de viés
        let beta1_pow_t = self.beta1.powi(t as i32);
        let beta1_pow_t_plus_1 = self.beta1 * beta1_pow_t; // β1^(t+1)
        let beta2_pow_t = self.beta2.powi(t as i32);

        for i in 0..params.len() {
            // Atualiza os momentos (igual ao Adam)
            self.momentum_first[i] = self.beta1 * self.momentum_first[i] + (1.0 - self.beta1) * grads[i];
            self.rmsprop_second[i] = self.beta2 * self.rmsprop_second[i] + (1.0 - self.beta2) * grads[i].powi(2);

            // Ajuste de Nesterov: combinação do momento atual e gradiente futuro
            let m_hat = (self.beta1 * self.momentum_first[i]) / (1.0 - beta1_pow_t_plus_1)
                + ((1.0 - self.beta1) * grads[i]) / (1.0 - beta1_pow_t);

            // Correção de viés do segundo momento (igual ao Adam)
            let v_hat = self.rmsprop_second[i] / (1.0 - beta2_pow_t);

            // Atualização dos parâmetros
            params[i] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }
}
