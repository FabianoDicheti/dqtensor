use crate::tensor::Tensor;
use crate::ssm_gpu::{
    dynamic_gpu::DynamicGPU,
    discretizer_gpu::DiscretizerGPU,
    kernel_gpu::KernelGPU,
    gate_gpu::GateGPU
};
use std::error::Error;

pub struct MambaBlockGPU {
    pub dynamic: DynamicGPU,
    pub discretizer: DiscretizerGPU,
    pub kernel: KernelGPU,
    pub gate: GateGPU,
}

impl MambaBlockGPU {
    pub fn new(
        dynamic_ptx: &str,
        discretizer_ptx: &str,
        kernel_ptx: &str,
        gate_ptx: &str
    ) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            dynamic: DynamicGPU::new(dynamic_ptx)?,
            discretizer: DiscretizerGPU::new(discretizer_ptx)?,
            kernel: KernelGPU::new(kernel_ptx)?,
            gate: GateGPU::new(gate_ptx)?
        })
    }

    pub fn forward(
        &self,
        input: &Tensor,        // [B, L, D]
        w_a: &Tensor,
        w_b: &Tensor,
        w_c: &Tensor,
        w_d: &Tensor
    ) -> Result<Tensor, Box<dyn Error>> {
        // 1. Gera A, B, C, Delta
        let (a, b, c, delta) = self.dynamic.generate_params(input, w_a, w_b, w_c, w_d)?;

        // 2. Discretiza A e B
        let (a_d, b_d) = self.discretizer.discretize(&a.mean_axis(1)?, &b.mean_axis(1)?, delta.mean_scalar()?)?;

        // 3. Calcula saída do SSM: kernel convolution
        let kernel = self.kernel.convolve_kernel(&input.mean_axis(1)?, &b_d)?;

        // 4. Aplica gate: output = sigmoid(c) * kernel + (1 - sigmoid(c)) * input
        let output = self.gate.apply_gate(&c.mean_axis(1)?, &kernel, &input.mean_axis(1)?)?;

        Ok(output)
    }
}