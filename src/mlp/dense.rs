use crate::f_not_linear::activation::ActivationFunction;
use crate::f_not_linear::derivatives::ActivationDerivatives;
use crate::optimizers::bp_optimizers::Optimizer;
use rand::random;

#[derive(Debug, Clone)]
pub struct DenseLayer {
    pub input_size: usize,
    pub output_size: usize,
    pub weights: Vec<Vec<f64>>,  // [output_size][input_size]
    pub biases: Vec<f64>,        // [output_size]
    pub activation: ActivationFunction,

    pub weight_optim: Box<dyn Optimizer>,
    pub bias_optim: Box<dyn Optimizer>,
}

impl DenseLayer {
    /// Cria uma camada densa
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: ActivationFunction,
        weight_optim: Box<dyn Optimizer>,
        bias_optim: Box<dyn Optimizer>,
    ) -> Self {
        let weights = (0..output_size)
            .map(|_| (0..input_size).map(|_| random::<f64>()).collect())
            .collect();

        let biases = (0..output_size).map(|_| random::<f64>()).collect();

        DenseLayer {
            input_size,
            output_size,
            weights,
            biases,
            activation,
            weight_optim,
            bias_optim,
        }
    }

    /// Forward
    pub fn forward(&self, input: &Vec<f64>) -> Vec<f64> {
        assert_eq!(input.len(), self.input_size);

        let mut output: Vec<f64> = (0..self.output_size)
            .map(|i| {
                let soma: f64 = self.weights[i]
                    .iter()
                    .zip(input.iter())
                    .map(|(w, x)| w * x)
                    .sum::<f64>() + self.biases[i];
                soma
            })
            .collect();

        match &self.activation {
            ActivationFunction::Softmax(_) => {
                output = ActivationFunction::apply_softmax(&output);
            }
            _ => {
                output = output
                    .iter()
                    .map(|x| self.activation.apply(*x))
                    .collect();
            }
        }

        output
    }

    /// Backward
    pub fn backward(
        &mut self,
        input: &Vec<f64>,
        output: &Vec<f64>,
        d_loss_d_output: &Vec<f64>,
    ) -> Vec<f64> {
        assert_eq!(output.len(), self.output_size);
        assert_eq!(d_loss_d_output.len(), self.output_size);

        let mut d_loss_d_input = vec![0.0; self.input_size];

        // Gradientes para bias e pesos
        let mut bias_grads = vec![0.0; self.output_size];
        let mut weight_grads = vec![vec![0.0; self.input_size]; self.output_size];

        for i in 0..self.output_size {
            //  Se for Softmax, não aplica derivada (CrossEntropy simplifica)
            let delta = match self.activation {
                ActivationFunction::Softmax(_) => d_loss_d_output[i],
                _ => d_loss_d_output[i] * self.activation.derivative(output[i]),
            };

            // Gradiente do bias
            bias_grads[i] = delta;

            for j in 0..self.input_size {
                // Gradiente do peso
                weight_grads[i][j] = delta * input[j];

                // Gradiente para a camada anterior (input)
                d_loss_d_input[j] += self.weights[i][j] * delta;
            }
        }

        // Flatten dos pesos para aplicar o otimizador
        let mut flat_weights: Vec<f64> = self.weights.iter().flatten().copied().collect();
        let flat_grads: Vec<f64> = weight_grads.iter().flatten().copied().collect();

        // Atualiza pesos
        self.weight_optim.update(&mut flat_weights, &flat_grads);

        // Reconstroi os pesos no formato [output][input]
        self.weights = flat_weights
            .chunks(self.input_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        // Atualiza bias
        self.bias_optim.update(&mut self.biases, &bias_grads);

        d_loss_d_input
    }
}
