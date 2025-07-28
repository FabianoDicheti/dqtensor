use cust::prelude::*;
use std::error::Error;
use crate::tensor::Tensor;

pub struct KernelGPU {
    context: Context,
    module: Module,
    stream: Stream,
}

impl KernelGPU {
    pub fn new(ptx_path: &str) -> Result<Self, Box<dyn Error>> {
        let context = cust::quick_init()?;
        let ptx = std::fs::read_to_string(ptx_path)?;
        let module = Module::from_ptx(ptx.trim(), &[])?;
        let stream = Stream::new(StreamFlags::DEFAULT, None)?;
        Ok(Self { context, module, stream })
    }

    /// Aplica convolução 1D com kernel gerado: out[t] = sum_i (kernel[i] * input[t - i])
    pub fn convolve_kernel(
        &self,
        input: &Tensor,     // [B, L]
        kernel: &Tensor     // [K]
    ) -> Result<Tensor, Box<dyn Error>> {
        let b = input.shape()[0];
        let l = input.shape()[1];
        let k = kernel.len();
        let len_out = l - k + 1;
        let mut output = vec![0.0f32; b * len_out];

        let d_input = DeviceBuffer::from_slice(input.data().as_slice().unwrap())?;
        let d_kernel = DeviceBuffer::from_slice(kernel.data().as_slice().unwrap())?;
        let mut d_output = DeviceBuffer::from_slice(&output)?;

        let func = self.module.get_function("ssm_kernel_conv")?;
        unsafe {
            launch!(
                func<<<(b * len_out as usize + 255) / 256, 256, 0, self.stream>>>(
                    d_input.as_device_ptr(),
                    d_kernel.as_device_ptr(),
                    d_output.as_device_ptr(),
                    b as u32,
                    l as u32,
                    k as u32
                )
            )?;
        }

        self.stream.synchronize()?;
        d_output.copy_to(&mut output)?;
        Ok(Tensor::from_vec(output, vec![b, len_out]))
    }
}