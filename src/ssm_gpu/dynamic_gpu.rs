use cust::prelude::*;
use std::error::Error;
use crate::tensor::Tensor;

pub struct DynamicGPU {
    context: Context,
    module: Module,
    stream: Stream,
}

impl DynamicGPU {
    pub fn new(ptx_path: &str) -> Result<Self, Box<dyn Error>> {
        let context = cust::quick_init()?;
        let ptx = std::fs::read_to_string(ptx_path)?;
        let module = Module::from_ptx(ptx.trim(), &[])?;
        let stream = Stream::new(StreamFlags::DEFAULT, None)?;
        Ok(Self { context, module, stream })
    }

    /// Aplica projeções lineares e softplus
    pub fn generate_parameters(
        &self,
        x: &Tensor,       // [D_in]
        w_b: &Tensor,     // [D_out, D_in]
        b_b: &Tensor,     // [D_out]
        w_c: &Tensor,
        b_c: &Tensor,
        w_delta: &Tensor,
        b_delta: &Tensor
    ) -> Result<(Tensor, Tensor, Tensor), Box<dyn Error>> {
        let d = b_b.len();
        let mut b_out = vec![0.0f32; d];
        let mut c_out = vec![0.0f32; d];
        let mut delta_out = vec![0.0f32; d];

        let d_x = DeviceBuffer::from_slice(x.data().as_slice().unwrap())?;
        let d_wb = DeviceBuffer::from_slice(w_b.data().as_slice().unwrap())?;
        let d_bb = DeviceBuffer::from_slice(b_b.data().as_slice().unwrap())?;
        let d_wc = DeviceBuffer::from_slice(w_c.data().as_slice().unwrap())?;
        let d_bc = DeviceBuffer::from_slice(b_c.data().as_slice().unwrap())?;
        let d_wd = DeviceBuffer::from_slice(w_delta.data().as_slice().unwrap())?;
        let d_bd = DeviceBuffer::from_slice(b_delta.data().as_slice().unwrap())?;

        let mut d_b = DeviceBuffer::from_slice(&b_out)?;
        let mut d_c = DeviceBuffer::from_slice(&c_out)?;
        let mut d_d = DeviceBuffer::from_slice(&delta_out)?;

        let func = self.module.get_function("generate_dynamic_params")?;
        unsafe {
            launch!(
                func<<<(d as u32 + 255) / 256, 256, 0, self.stream>>>(
                    d_x.as_device_ptr(),
                    d_wb.as_device_ptr(),
                    d_bb.as_device_ptr(),
                    d_wc.as_device_ptr(),
                    d_bc.as_device_ptr(),
                    d_wd.as_device_ptr(),
                    d_bd.as_device_ptr(),
                    d_b.as_device_ptr(),
                    d_c.as_device_ptr(),
                    d_d.as_device_ptr(),
                    x.len() as u32,
                    d as u32
                )
            )?;
        }

        self.stream.synchronize()?;
        d_b.copy_to(&mut b_out)?;
        d_c.copy_to(&mut c_out)?;
        d_d.copy_to(&mut delta_out)?;

        Ok((
            Tensor::from_vec(b_out, vec![d]),
            Tensor::from_vec(c_out, vec![d]),
            Tensor::from_vec(delta_out, vec![d])
        ))
    }
}