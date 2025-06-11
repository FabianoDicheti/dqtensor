use super::{
    parametric::DynamicParamGenerator,
    kernel::generate_dynamic_kernel,
    convolution::dynamic_causal_convolution,
    gate::sigmoid_gate,
};


/// Bloco Mamba completo (sem dependência de frameworks)
pub struct MambaBlock {
    pub param_gen: DynamicParamGenerator,
    pub C: Vec<f64>,  // vetor de saída
    pub use_skip: bool,
}

impl MambaBlock {
    pub fn new(param_gen: DynamicParamGenerator, C: Vec<f64>, use_skip: bool) -> Self {
        Self { param_gen, C, use_skip }
    }

    /// Executa a sequência completa com gate e skip connection
    pub fn forward(&self, x_seq: &Vec<Vec<f64>>) -> Vec<f64> {
        let T = x_seq.len();
        let u: Vec<f64> = x_seq.iter()
            .map(|xt| xt.iter().copied().sum::<f64>())  // simplificação de u_t = soma(x_t)
            .collect();

        // 1. Kernel dinâmico por tempo
        let kernels = generate_dynamic_kernel(&self.param_gen, x_seq, &self.C);

        // 2. Convolução causal
        let ssm_outputs = dynamic_causal_convolution(&kernels, &u);

        // 3. Gating dinâmico
        let gates: Vec<f64> = x_seq.iter()
            .map(|xt| {
                let g = sigmoid_gate(xt);
                g.iter().sum::<f64>() / g.len() as f64  // média como fator escalar
            })
            .collect();

        // 4. Gate * SSM + skip
        let mut final_output = vec![0.0; T];
        for t in 0..T {
            let gated = gates[t] * ssm_outputs[t];
            if self.use_skip {
                final_output[t] = gated + u[t];  // skip connection
            } else {
                final_output[t] = gated;
            }
        }

        final_output
    }
}
