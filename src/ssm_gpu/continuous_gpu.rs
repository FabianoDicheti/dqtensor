use cust::prelude::*;
use std::error::Error;
use crate::tensor::Tensor;

pub struct ContinuousGPU {
    context: Context,
    module: Module,
    stream: Stream,
}

impl ContinuousGPU {
    pub fn new(ptx_path: &str) -> Result<Self, Box<dyn Error>> {
        let context = cust::quick_init()?;
        let ptx = std::fs::read_to_string(ptx_path)?;
        let module = Module::from_ptx(ptx.trim(), &[])?;
        let stream = Stream::new(StreamFlags::DEFAULT, None)?;
        Ok(Self { context, module, stream })
    }

    /// h(t) = exp(ΔA) h₀ + ...
    pub fn expm_mul(
        &self,
        delta: f32,
        a_matrix: &Tensor, // shape [N, N]
        h0: &Tensor,        // shape [N]
    ) -> Result<Tensor, Box<dyn Error>> {
        let n = h0.len();
        let mut result = vec![0.0f32; n];

        let d_a = DeviceBuffer::from_slice(a_matrix.data().as_slice().unwrap())?;
        let d_h = DeviceBuffer::from_slice(h0.data().as_slice().unwrap())?;
        let mut d_out = DeviceBuffer::from_slice(&result)?;

        let func = self.module.get_function("expm_mul")?;
        unsafe {
            launch!(
                func<<<(n as u32 + 255) / 256, 256, 0, self.stream>>>(
                    d_a.as_device_ptr(),
                    d_h.as_device_ptr(),
                    d_out.as_device_ptr(),
                    delta,
                    n as u32
                )
            )?;
        }

        self.stream.synchronize()?;
        d_out.copy_to(&mut result)?;

        Ok(Tensor::from_vec(result, vec![n]))
    }
}
