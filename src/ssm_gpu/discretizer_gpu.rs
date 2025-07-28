use cust::prelude::*;
use std::error::Error;
use crate::tensor::Tensor;

pub struct DiscretizerGPU {
    context: Context,
    module: Module,
    stream: Stream,
}

impl DiscretizerGPU {
    pub fn new(ptx_path: &str) -> Result<Self, Box<dyn Error>> {
        let context = cust::quick_init()?;
        let ptx = std::fs::read_to_string(ptx_path)?;
        let module = Module::from_ptx(ptx.trim(), &[])?;
        let stream = Stream::new(StreamFlags::DEFAULT, None)?;
        Ok(Self { context, module, stream })
    }

    /// Discretiza A e B usando aproximação da exponencial (ordem 2)
    pub fn discretize(
        &self,
        a_matrix: &Tensor,  // [N, N]
        b_vector: &Tensor,  // [N]
        delta: f32
    ) -> Result<(Tensor, Tensor), Box<dyn Error>> {
        let n = b_vector.len();
        let mut a_out = vec![0.0f32; n * n];
        let mut b_out = vec![0.0f32; n];

        let d_a = DeviceBuffer::from_slice(a_matrix.data().as_slice().unwrap())?;
        let d_b = DeviceBuffer::from_slice(b_vector.data().as_slice().unwrap())?;
        let mut d_ad = DeviceBuffer::from_slice(&a_out)?;
        let mut d_bd = DeviceBuffer::from_slice(&b_out)?;

        let func = self.module.get_function("discretize_zoh")?;
        unsafe {
            launch!(
                func<<<(n as u32 + 255) / 256, 256, 0, self.stream>>>(
                    d_a.as_device_ptr(),
                    d_b.as_device_ptr(),
                    d_ad.as_device_ptr(),
                    d_bd.as_device_ptr(),
                    delta,
                    n as u32
                )
            )?;
        }

        self.stream.synchronize()?;
        d_ad.copy_to(&mut a_out)?;
        d_bd.copy_to(&mut b_out)?;

        Ok((
            Tensor::from_vec(a_out, vec![n, n]),
            Tensor::from_vec(b_out, vec![n])
        ))
    }
}