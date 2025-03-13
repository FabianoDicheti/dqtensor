pub trait Optimizer {
    /// Atualiza os parâmetros com base nos gradientes calculados.
    /// # Argumentos
    /// * `params` - Slice mutável dos parâmetros a serem atualizados.
    /// * `grads` - Slice dos gradientes correspondentes aos parâmetros.
    fn update(&mut self, params: &mut [f64], grads: &[f64]);
}


//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//


/// Implementação do Gradiente Descendente Simples (SGD)
/// Atualiza os parâmetros diretamente usando: param -= learning_rate * grad
pub struct SGD {
    learning_rate: f64,
}

impl SGD {
    /// Cria uma nova instância do SGD.
    /// # Argumento
    /// * `learning_rate` - Taxa de aprendizagem (η).
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for SGD {
    fn update(&mut self, params: &mut [f64], grads: &[f64]) {
        debug_assert_eq!(
            params.len(),
            grads.len(),
            "Params e grads devem ter o mesmo tamanho"
        );

        for i in 0..params.len() {
            params[i] -= self.learning_rate * grads[i];
        }
    }
}


//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------

/// Gradiente Descendente por Lotes (Batch Gradient Descent)
/// Atualização direta: params -= learning_rate * gradiente_médio_do_batch
pub struct BatchGradientDescent {
    learning_rate: f64,
}

impl BatchGradientDescent {
    /// Cria uma nova instância do otimizador.
    /// # Argumento
    /// * `learning_rate` - Taxa de aprendizagem (η).
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for BatchGradientDescent {
    fn update(&mut self, params: &mut [f64], grads: &[f64]) {
        debug_assert_eq!(
            params.len(),
            grads.len(),
            "Params e grads devem ter o mesmo tamanho"
        );

        for i in 0..params.len() {
            params[i] -= self.learning_rate * grads[i];
        }
    }
}


//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------

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

//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//

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

//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//


/// AdaGrad (Adaptive Gradient Algorithm)
/// Adapta a taxa de aprendizado para cada parâmetro com base no histórico quadrático dos gradientes.
/// Campos:
/// - `learning_rate`: Taxa de aprendizado base (η).
/// - `epsilon`: Termo de estabilidade numérica.
/// - `cache`: Acumulador de gradientes quadrados (G_t na literatura).
pub struct AdaGrad {
    learning_rate: f64,
    epsilon: f64,
    cache: Vec<f64>,
}

impl AdaGrad {
    /// Cria uma nova instância do AdaGrad.
    /// # Argumentos
    /// * `learning_rate` - Taxa base de aprendizado (ex: 0.01).
    /// * `epsilon` - Termo de estabilidade (ex: 1e-8).
    /// * `param_size` - Número de parâmetros a serem otimizados.
    pub fn new(learning_rate: f64, epsilon: f64, param_size: usize) -> Self {
        Self {
            learning_rate,
            epsilon,
            cache: vec![0.0; param_size],
        }
    }
}

impl Optimizer for AdaGrad {
    fn update(&mut self, params: &mut [f64], grads: &[f64]) {
        debug_assert_eq!(params.len(), grads.len(), "Params e grads devem ter o mesmo tamanho");
        debug_assert_eq!(params.len(), self.cache.len(), "Params e cache devem ter o mesmo tamanho");

        for i in 0..params.len() {
            // Acumula gradientes quadrados no cache
            self.cache[i] += grads[i].powi(2);
            
            // Calcula a taxa de aprendizado adaptativa
            let adapted_lr = self.learning_rate / (self.cache[i].sqrt() + self.epsilon);
            
            // Atualiza o parâmetro
            params[i] -= adapted_lr * grads[i];
        }
    }
}


//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//


/// Adamax (Adaptive Moment Estimation with L∞ norm)
/// Uma variante do Adam que usa a norma infinita para a estimativa do segundo momento.
pub struct Adamax {
    learning_rate: f64,
    beta1: f64,         // Decaimento do primeiro momento
    beta2: f64,         // Decaimento do segundo momento (norma L∞)
    epsilon: f64,       // Termo de estabilidade
    time_step: u64,     // Contador de iterações
    m: Vec<f64>,        // Primeiro momento (média)
    u: Vec<f64>,        // Segundo momento (norma L∞)
}

impl Adamax {
    /// Cria uma nova instância do Adamax
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
            m: vec![0.0; param_size],
            u: vec![0.0; param_size],
        }
    }
}

impl Optimizer for Adamax {
    fn update(&mut self, params: &mut [f64], grads: &[f64]) {
        debug_assert_eq!(params.len(), grads.len(), "Params e grads devem ter o mesmo tamanho");
        debug_assert_eq!(params.len(), self.m.len(), "Params e m devem ter o mesmo tamanho");
        debug_assert_eq!(params.len(), self.u.len(), "Params e u devem ter o mesmo tamanho");

        self.time_step += 1;
        let t = self.time_step as f64;

        for i in 0..params.len() {
            // Atualiza o primeiro momento
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i];
            
            // Atualiza o segundo momento (norma L∞)
            let grad_abs = grads[i].abs();
            self.u[i] = self.beta2 * self.u[i].max(grad_abs);
            
            // Correção de viés do primeiro momento
            let m_hat = self.m[i] / (1.0 - self.beta1.powi(self.time_step as i32));
            
            // Atualização dos parâmetros
            params[i] -= self.learning_rate * m_hat / (self.u[i] + self.epsilon);
        }
    }
}

//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//


/// AdaDelta (Adaptive Learning Rate Method)
/// Elimina a necessidade de uma taxa de aprendizado manual, adaptando-a automaticamente.
pub struct AdaDelta {
    rho: f64,            // Fator de decaimento para as médias móveis (0.95 é comum)
    epsilon: f64,         // Termo de estabilidade numérica
    avg_grad_sq: Vec<f64>, // Média móvel dos gradientes quadrados (E[g²])
    avg_delta_sq: Vec<f64>, // Média móvel dos deltas quadrados (E[Δx²])
}

impl AdaDelta {
    /// Cria uma nova instância do AdaDelta
    /// # Argumentos
    /// * `rho` - Fator de decaimento para as médias móveis (ex: 0.95)
    /// * `epsilon` - Termo de estabilidade (ex: 1e-6)
    /// * `param_size` - Número de parâmetros a serem otimizados
    pub fn new(rho: f64, epsilon: f64, param_size: usize) -> Self {
        Self {
            rho,
            epsilon,
            avg_grad_sq: vec![0.0; param_size],  // Inicializa acumuladores
            avg_delta_sq: vec![0.0; param_size],
        }
    }
}

impl Optimizer for AdaDelta {
    fn update(&mut self, params: &mut [f64], grads: &[f64]) {
        debug_assert_eq!(params.len(), grads.len(), "Params e grads devem ter o mesmo tamanho");
        debug_assert_eq!(params.len(), self.avg_grad_sq.len(), "Params e avg_grad_sq devem ter o mesmo tamanho");
        debug_assert_eq!(params.len(), self.avg_delta_sq.len(), "Params e avg_delta_sq devem ter o mesmo tamanho");

        for i in 0..params.len() {
            // Atualiza a média móvel dos gradientes quadrados
            self.avg_grad_sq[i] = self.rho * self.avg_grad_sq[i] + (1.0 - self.rho) * grads[i].powi(2);
            
            // Calcula o delta usando a raiz quadrada das médias móveis
            let delta_numerator = (self.avg_delta_sq[i] + self.epsilon).sqrt();
            let delta_denominator = (self.avg_grad_sq[i] + self.epsilon).sqrt();
            let delta = (delta_numerator / delta_denominator) * grads[i];
            
            // Atualiza o parâmetro
            params[i] -= delta;
            
            // Atualiza a média móvel dos deltas quadrados
            self.avg_delta_sq[i] = self.rho * self.avg_delta_sq[i] + (1.0 - self.rho) * delta.powi(2);
        }
    }
}


//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//

/// RMSProp (Root Mean Square Propagation)
/// Mantém uma média móvel exponencial dos quadrados dos gradientes para adaptar a taxa de aprendizado.
pub struct RMSProp {
    learning_rate: f64,
    gamma: f64,       // Fator de decaimento para a média móvel (tipicamente 0.9)
    epsilon: f64,     // Termo de estabilidade numérica
    cache: Vec<f64>,  // Média móvel dos gradientes quadrados (E[g²] na literatura)
}

impl RMSProp {
    /// Cria uma nova instância do RMSProp.
    /// # Argumentos
    /// * `learning_rate` - Taxa base de aprendizado (ex: 0.001).
    /// * `gamma` - Fator de decaimento da média móvel (ex: 0.9).
    /// * `epsilon` - Termo de estabilidade (ex: 1e-8).
    /// * `param_size` - Número de parâmetros a serem otimizados.
    pub fn new(learning_rate: f64, gamma: f64, epsilon: f64, param_size: usize) -> Self {
        Self {
            learning_rate,
            gamma,
            epsilon,
            cache: vec![0.0; param_size],
        }
    }
}

impl Optimizer for RMSProp {
    fn update(&mut self, params: &mut [f64], grads: &[f64]) {
        debug_assert_eq!(params.len(), grads.len(), "Params e grads devem ter o mesmo tamanho");
        debug_assert_eq!(params.len(), self.cache.len(), "Params e cache devem ter o mesmo tamanho");

        for i in 0..params.len() {
            // Atualiza a média móvel dos gradientes quadrados
            self.cache[i] = self.gamma * self.cache[i] + (1.0 - self.gamma) * grads[i].powi(2);
            
            // Calcula a taxa de aprendizado adaptativa
            let adapted_lr = self.learning_rate / (self.cache[i].sqrt() + self.epsilon);
            
            // Atualiza o parâmetro
            params[i] -= adapted_lr * grads[i];
        }
    }
}





//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//


/// Momentum (Polyak Momentum)
/// Acelera o treinamento em direções relevantes e reduz oscilações.
pub struct Momentum {
    learning_rate: f64,
    gamma: f64,         // Fator de momentum (0.9 é comum)
    velocity: Vec<f64>, // Velocidade acumulada (direção do movimento)
}

impl Momentum {
    /// Cria uma nova instância do Momentum
    /// # Argumentos
    /// * `learning_rate` - Taxa de aprendizado (η)
    /// * `gamma` - Fator de momentum (0.0 a 1.0)
    /// * `param_size` - Número de parâmetros
    pub fn new(learning_rate: f64, gamma: f64, param_size: usize) -> Self {
        Self {
            learning_rate,
            gamma,
            velocity: vec![0.0; param_size],
        }
    }
}

impl Optimizer for Momentum {
    fn update(&mut self, params: &mut [f64], grads: &[f64]) {
        debug_assert_eq!(
            params.len(),
            grads.len(),
            "Params e grads devem ter o mesmo tamanho"
        );
        debug_assert_eq!(
            params.len(),
            self.velocity.len(),
            "Params e velocity devem ter o mesmo tamanho"
        );

        for i in 0..params.len() {
            // Atualiza a velocidade: v = γ*v + η*grad
            self.velocity[i] = self.gamma * self.velocity[i] + self.learning_rate * grads[i];
            
            // Atualiza os parâmetros: θ = θ - v
            params[i] -= self.velocity[i];
        }
    }
}



//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//-------//






//

fn main() {
    let mut params = vec![1.0, 2.0, 3.0];
    let grads = vec![0.1, 0.2, 0.3];

    // SGD
    let mut sgd = SGD::new(0.01);
    sgd.update(&mut params, &grads);

    // Adam
    let mut adam = Adam::new(0.001, 0.9, 0.999, 1e-8, params.len());
    adam.update(&mut params, &grads);

    // Nadam
    let mut nadam = Nadam::new(0.001, 0.9, 0.999, 1e-8, params.len());
    nadam.update(&mut params, &grads);
}