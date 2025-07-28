use crate::tensor::Tensor;

pub struct ParametricGPU;

impl ParametricGPU {
    /// Gera parâmetros A, B, C, Delta constantes com broadcast para sequência [B, L]
    pub fn generate(
        a: &Tensor,  // [N, N]
        b: &Tensor,  // [N]
        c: &Tensor,  // [N]
        delta: f32,  // escalar
        bsz: usize,
        seq_len: usize
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        let (n, _) = a.shape2().unwrap();

        let a_out = a.repeat(bsz * seq_len);        // [B, L, N, N]
        let b_out = b.repeat(bsz * seq_len);        // [B, L, N]
        let c_out = c.repeat(bsz * seq_len);        // [B, L, N]
        let d_out = Tensor::full(vec![bsz, seq_len, 1], delta);

        (a_out, b_out, c_out, d_out)
    }
}