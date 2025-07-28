use cust::prelude::*;
use std::error::Error;
use crate::tensor::Tensor;

pub struct GpuConvolution {
    context: Context,
    module: Module,
    stream: Stream,
}

impl GpuConvolution {
    pub fn new(ptx_path: &str) -> Result<Self, Box<dyn Error>> {
        let context = cust::quick_init()?;
        let ptx = std::fs::read_to_string(ptx_path)?;
        let module = Module::from_ptx(ptx.trim(), &[])?;
        let stream = Stream::new(StreamFlags::DEFAULT, None)?;
        Ok(Self { context, module, stream })
    }

    pub fn convolve(&self, input: &Tensor, kernel: &Tensor) -> Result<Tensor, Box<dyn Error>> {
        let input_data = input.data().as_slice().unwrap();
        let kernel_data = kernel.data().as_slice().unwrap();
        let len_out = input_data.len() - kernel_data.len() + 1;
        let mut output_data = vec![0.0f32; len_out];

        let d_input = DeviceBuffer::from_slice(input_data)?;
        let d_kernel = DeviceBuffer::from_slice(kernel_data)?;
        let mut d_output = DeviceBuffer::from_slice(&output_data)?;

        let func = self.module.get_function("conv1d")?;
        unsafe {
            launch!(
                func<<<(len_out as u32 + 255) / 256, 256, 0, self.stream>>>(
                    d_input.as_device_ptr(),
                    d_kernel.as_device_ptr(),
                    d_output.as_device_ptr(),
                    input_data.len() as u32,
                    kernel_data.len() as u32
                )
            )?;
        }

        self.stream.synchronize()?;
        d_output.copy_to(&mut output_data)?;
        Ok(Tensor::from_vec(output_data, vec![len_out]))
    }
}