// parte do seguinte:         let (hidden_output, output) = self.forward(input);

    // Forward propagation: calcula a saída da rede para uma entrada
    fn forward(&self, input: &Vec<f64>) -> (Vec<f64>, Vec<f64>) {
        // Calcula a saída da camada oculta
        let hidden_output: Vec<f64> = (0..self.hidden_size)
            .map(|i| {
                let sum: f64 = input
                    .iter()
                    .enumerate()
                    .map(|(j, &x)| x * self.weights_input_hidden[j][i])
                    .sum();
                Self::sigmoid(sum + self.bias_hidden[i]) // Aplica a função de ativação
            })
            .collect();

        // Calcula a saída da camada de saída
        let output: Vec<f64> = (0..self.output_size)
            .map(|i| {
                let sum: f64 = hidden_output
                    .iter()
                    .enumerate()
                    .map(|(j, &x)| x * self.weights_hidden_output[j][i])
                    .sum();
                Self::sigmoid(sum + self.bias_output[i]) // Aplica a função de ativação
            })
            .collect();

        (hidden_output, output)
    }
