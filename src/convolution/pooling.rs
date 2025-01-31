// src/convolution/pooling.rs

/// Enum para representar os tipos de pooling.
pub enum PoolingType {
    Max,
    Average,
}

/// Função principal de pooling.
pub fn pool(input: &Vec<Vec<f32>>, pool_size: usize, stride: usize, pooling_type: PoolingType) -> Vec<Vec<f32>> {
    match pooling_type {
        PoolingType::Max => max_pooling(input, pool_size, stride),
        PoolingType::Average => average_pooling(input, pool_size, stride),
    }
}

/// Implementação do Max Pooling.
fn max_pooling(input: &Vec<Vec<f32>>, pool_size: usize, stride: usize) -> Vec<Vec<f32>> {
    let rows = input.len();
    let cols = input[0].len();

    let mut output = Vec::new();

    for i in (0..rows).step_by(stride) {
        let mut row = Vec::new();
        for j in (0..cols).step_by(stride) {
            let mut max_val = f32::MIN;
            for x in 0..pool_size {
                for y in 0..pool_size {
                    if i + x < rows && j + y < cols {
                        max_val = max_val.max(input[i + x][j + y]);
                    }
                }
            }
            row.push(max_val);
        }
        output.push(row);
    }

    output
}

/// Implementação do Average Pooling.
fn average_pooling(input: &Vec<Vec<f32>>, pool_size: usize, stride: usize) -> Vec<Vec<f32>> {
    let rows = input.len();
    let cols = input[0].len();

    let mut output = Vec::new();

    for i in (0..rows).step_by(stride) {
        let mut row = Vec::new();
        for j in (0..cols).step_by(stride) {
            let mut sum = 0.0;
            let mut count = 0;
            for x in 0..pool_size {
                for y in 0..pool_size {
                    if i + x < rows && j + y < cols {
                        sum += input[i + x][j + y];
                        count += 1;
                    }
                }
            }
            row.push(sum / count as f32);
        }
        output.push(row);
    }

    output
}