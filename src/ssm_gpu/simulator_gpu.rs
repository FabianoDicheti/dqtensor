use cust::prelude::*;
use crate::tensor::Tensor;
use std::error::Error;

pub struct SimulatorGPU {
    context: Context,
    module: Module,
    stream: Stream,
}

impl SimulatorGPU {
    pub fn new(ptx_path: &str) -> Result<Self, Box<dyn Error>> {
        let context = cust::quick_init()?;
        let ptx = std::fs::read_to_string(ptx_path)?;
        let module = Module::from_ptx(ptx.trim(), &[])?;
        let stream = Stream::new(StreamFlags::DEFAULT, None)?;
        Ok(Self { context, module, stream })
    }

    /// Aplica simulação paralela (scan): h[t] = A[t] h[t-1] + B[t] x[t]
    pub fn simulate(
        &self,
        a_seq: &Tensor,  // [L, N, N]
        b_seq: &Tensor,  // [L, N]
        x_seq: &Tensor   // [L]
    ) -> Result<Tensor, Box<dyn Error>> {
        let l = x_seq.shape()[0];
        let n = b_seq.shape()[1];

        let mut h_out = vec![0.0f32; l * n];

        let d_a = DeviceBuffer::from_slice(a_seq.data().as_slice().unwrap())?;
        let d_b = DeviceBuffer::from_slice(b_seq.data().as_slice().unwrap())?;
        let d_x = DeviceBuffer::from_slice(x_seq.data().as_slice().unwrap())?;
        let mut d_h = DeviceBuffer::from_slice(&h_out)?;

        let func = self.module.get_function("simulate_scan")?;
        unsafe {
            launch!(
                func<<<(l as u32 + 255) / 256, 256, 0, self.stream>>>(
                    d_a.as_device_ptr(),
                    d_b.as_device_ptr(),
                    d_x.as_device_ptr(),
                    d_h.as_device_ptr(),
                    l as u32,
                    n as u32
                )
            )?;
        }

        self.stream.synchronize()?;
        d_h.copy_to(&mut h_out)?;
        Ok(Tensor::from_vec(h_out, vec![l, n]))
    }
}