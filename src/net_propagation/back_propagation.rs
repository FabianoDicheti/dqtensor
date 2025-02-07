use rand::Rng;

// Estrutura da rede neural
struct NeuralNetwork {
    input_size: usize,      // Tamanho da camada de entrada
    hidden_size: usize,     // Tamanho da camada oculta
    output_size: usize,     // Tamanho da camada de saída
    weights_input_hidden: Vec<Vec<f64>>, // Pesos entre a camada de entrada e a oculta
    weights_hidden_output: Vec<Vec<f64>>, // Pesos entre a camada oculta e a de saída
    bias_hidden: Vec<f64>,  // Bias da camada oculta
    bias_output: Vec<f64>,  // Bias da camada de saída
}

impl NeuralNetwork {
    // Inicializa a rede neural com pesos e biases aleatórios
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Inicializa os pesos entre a camada de entrada e a oculta
        let weights_input_hidden = (0..input_size)
            .map(|_| (0..hidden_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();

        // Inicializa os pesos entre a camada oculta e a de saída
        let weights_hidden_output = (0..hidden_size)
            .map(|_| (0..output_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();

        // Inicializa os biases da camada oculta e de saída
        let bias_hidden = (0..hidden_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let bias_output = (0..output_size).map(|_| rng.gen_range(-1.0..1.0)).collect();

        NeuralNetwork {
            input_size,
            hidden_size,
            output_size,
            weights_input_hidden,
            weights_hidden_output,
            bias_hidden,
            bias_output,
        }
    }


    // B A C K    P R O P A G A T I O N
    // Backpropagation: ajusta os pesos e biases com base no erro
    fn train(&mut self, input: &Vec<f64>, target: &Vec<f64>, learning_rate: f64) {
        // Forward propagation
        let (hidden_output, output) = self.forward(input);

        // Calcula o erro na camada de saída
        let output_errors: Vec<f64> = (0..self.output_size)
            .map(|i| target[i] - output[i])
            .collect();

        // Calcula o gradiente para a camada de saída
        let output_gradients: Vec<f64> = (0..self.output_size)
            .map(|i| output_errors[i] * Self::sigmoid_derivative(output[i]))
            .collect();

        // Atualiza os pesos e biases da camada de saída
        for i in 0..self.hidden_size {
            for j in 0..self.output_size {
                self.weights_hidden_output[i][j] +=
                    learning_rate * output_gradients[j] * hidden_output[i];
            }
        }
        for j in 0..self.output_size {
            self.bias_output[j] += learning_rate * output_gradients[j];
        }

        // Calcula o erro na camada oculta
        let hidden_errors: Vec<f64> = (0..self.hidden_size)
            .map(|i| {
                (0..self.output_size)
                    .map(|j| output_gradients[j] * self.weights_hidden_output[i][j])
                    .sum()
            })
            .collect();

        // Calcula o gradiente para a camada oculta
        let hidden_gradients: Vec<f64> = (0..self.hidden_size)
            .map(|i| hidden_errors[i] * Self::sigmoid_derivative(hidden_output[i]))
            .collect();

        // Atualiza os pesos e biases da camada oculta
        for i in 0..self.input_size {
            for j in 0..self.hidden_size {
                self.weights_input_hidden[i][j] +=
                    learning_rate * hidden_gradients[j] * input[i];
            }
        }
        for j in 0..self.hidden_size {
            self.bias_hidden[j] += learning_rate * hidden_gradients[j];
        }
    }
}

fn main() {
    // Cria uma rede neural com 2 entradas, 2 neurônios na camada oculta e 1 saída
    let mut nn = NeuralNetwork::new(2, 2, 1);

    // Dados de treinamento (XOR problem)
    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    // Treina a rede neural
    let epochs = 10000;
    let learning_rate = 0.1;
    for epoch in 0..epochs {
        for (input, target) in inputs.iter().zip(targets.iter()) {
            nn.train(input, target, learning_rate);
        }
        if epoch % 1000 == 0 {
            println!("Epoch {}", epoch);
        }
    }

    // Testa a rede neural
    for input in &inputs {
        let (_, output) = nn.forward(input);
        println!("Input: {:?}, Output: {:?}", input, output);
    }
}