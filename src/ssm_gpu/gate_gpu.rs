use cust::prelude::*;
use std::error::Error;
use crate::tensor::Tensor;

pub struct GateGPU {
    context: Context,
    module: Module,
    stream: Stream,
}

impl GateGPU {
    pub fn new(ptx_path: &str) -> Result<Self, Box<dyn Error>> {
        let context = cust::quick_init()?;
        let ptx = std::fs::read_to_string(ptx_path)?;
        let module = Module::from_ptx(ptx.trim(), &[])?;
        let stream = Stream::new(StreamFlags::DEFAULT, None)?;
        Ok(Self { context, module, stream })
    }

    /// Aplica gating: out = sigmoid(gate) * x + (1 - sigmoid(gate)) * y
    pub fn apply_gate(
        &self,
        gate: &Tensor,  // shape: [B, L, D]
        x: &Tensor,
        y: &Tensor
    ) -> Result<Tensor, Box<dyn Error>> {
        let n = gate.len();
        let mut output = vec![0.0f32; n];

        let d_gate = DeviceBuffer::from_slice(gate.data().as_slice().unwrap())?;
        let d_x = DeviceBuffer::from_slice(x.data().as_slice().unwrap())?;
        let d_y = DeviceBuffer::from_slice(y.data().as_slice().unwrap())?;
        let mut d_out = DeviceBuffer::from_slice(&output)?;

        let func = self.module.get_function("apply_gate")?;
        unsafe {
            launch!(
                func<<<(n as u32 + 255) / 256, 256, 0, self.stream>>>(
                    d_gate.as_device_ptr(),
                    d_x.as_device_ptr(),
                    d_y.as_device_ptr(),
                    d_out.as_device_ptr(),
                    n as u32
                )
            )?;
        }

        self.stream.synchronize()?;
        d_out.copy_to(&mut output)?;
        Ok(Tensor::from_vec(output, gate.shape().to_vec()))
    }
}