#[allow(clippy::vec_init_then_push)]

//size dims deve conter as altura e largura de cada dimensao ex [3, 1], [2, 2], [2,1] que resulta
//na seguinte matriz:
//[  [[0.0], [0.0]],  [[0.0], [0.0]],
//   [[0.0], [0.0]],  [[0.0], [0.0]]],
//[  [[0.0], [0.0]],  [[0.0], [0.0]],
//   [[0.0], [0.0]],  [[0.0], [0.0]]],
//[  [[0.0], [0.0]],  [[0.0], [0.0]],
//   [[0.0], [0.0]],  [[0.0], [0.0]]]


pub fn aloca_tensor(dims: &[usize]) -> Vec<Box<dyn std::any::Any>> {
    if dims.is_empty() {
        return vec![Box::new(0.0) as Box<dyn std::any::Any>];
    }

    let mut tensor = Vec::with_capacity(dims[0]);
    for _ in 0..dims[0] {
        tensor.push(Box::new(aloca_tensor(&dims[1..])) as Box<dyn std::any::Any>);
    }
    tensor
}


