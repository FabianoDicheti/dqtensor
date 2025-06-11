use dqtensor::ssm::parametric::DynamicParamGenerator;
use dqtensor::ssm::kernel::{generate_dynamic_kernel};
use dqtensor::ssm::convolution::dynamic_causal_convolution;

use std::error::Error;

pub fn main() -> Result<(), Box<dyn Error>> {
    let C = vec![0.3, 0.7];
    let mut param_gen = DynamicParamGenerator::new(2, 2);

    param_gen.W_a = vec![vec![0.1, 0.1]; 2];
    param_gen.W_b = vec![vec![0.2, 0.2]; 2];
    param_gen.W_dt = vec![vec![1.0, 1.0]; 2];
    param_gen.bias_a = vec![0.0, 0.0];
    param_gen.bias_b = vec![0.0, 0.0];
    param_gen.bias_dt = vec![1.0, 1.0];

    let inputs: Vec<Vec<f64>> = vec![
        vec![1.0, 0.5],
        vec![0.8, 0.3],
        vec![0.2, 0.7],
        vec![0.0, 1.0],
    ];
    let u: Vec<f64> = vec![1.0, 1.0, 1.0, 1.0];

    let kernels = generate_dynamic_kernel(&param_gen, &inputs, &C);
    let outputs = dynamic_causal_convolution(&kernels, &u);

    for (t, y) in outputs.iter().enumerate() {
        println!("t = {} | y = {:.5}", t, y);
    }

    Ok(())
}
