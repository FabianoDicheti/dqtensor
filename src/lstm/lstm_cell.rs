use std::collections::HashMap;
use crate::f_not_linear::activation::ActivationFunction;
use crate::neuron::neuron::Layer;
use crate::lstm::state::EstadoLSTM;
use crate::optimizers::bp_optimizers::Optimizer;

#[derive(Debug)]
pub struct LSTMCell {
    pub porta_forget: Layer,
    pub porta_input: Layer,
    pub porta_output: Layer,
    pub porta_candidato: Layer,
    pub estado: EstadoLSTM,

    pub weight_optimizers: HashMap<String, Box<dyn Optimizer>>,
    pub bias_optimizers: HashMap<String, Box<dyn Optimizer>>,
}

impl LSTMCell {
    pub fn new(
        name: &str,
        num_neuronios: usize,
        input_size: usize,
        activation_funcs: (ActivationFunction, ActivationFunction),
        weight_optimizers: HashMap<String, Box<dyn Optimizer>>,
        bias_optimizers: HashMap<String, Box<dyn Optimizer>>,
    ) -> Self {
        let (sigmoid, tanh) = activation_funcs;

        Self {
            porta_forget: Layer::new(format!("{name}_forget"), num_neuronios, input_size + num_neuronios, sigmoid.clone()),
            porta_input: Layer::new(format!("{name}_input"), num_neuronios, input_size + num_neuronios, sigmoid.clone()),
            porta_output: Layer::new(format!("{name}_output"), num_neuronios, input_size + num_neuronios, sigmoid.clone()),
            porta_candidato: Layer::new(format!("{name}_candidato"), num_neuronios, input_size + num_neuronios, tanh.clone()),

            estado: EstadoLSTM::new(num_neuronios),

            weight_optimizers,
            bias_optimizers,
        }
    }

    pub fn forward(&mut self, entrada: &Vec<f64>) -> Vec<f64> {
        let entrada_completa = [&entrada[..], &self.estado.memoria_h[..]].concat();

        let forget = self.porta_forget.forward(&entrada_completa);
        let input = self.porta_input.forward(&entrada_completa);
        let output = self.porta_output.forward(&entrada_completa);
        let candidato = self.porta_candidato.forward(&entrada_completa);

        self.estado.memoria_c = forget
            .iter()
            .zip(&self.estado.memoria_c)
            .zip(&input)
            .zip(&candidato)
            .map(|(((f, c), i), g)| f * c + i * g)
            .collect();

        self.estado.memoria_h = output
            .iter()
            .zip(&self.estado.memoria_c)
            .map(|(o, c)| o * c.tanh())
            .collect();

        self.estado.memoria_h.clone()
    }

    pub fn backward(
        &mut self,
        entrada: &Vec<f64>,
        d_h: &Vec<f64>,
    ) -> Vec<f64> {
        let entrada_completa = [&entrada[..], &self.estado.memoria_h[..]].concat();

        let output = self.porta_output.last_output.clone();
        let input = self.porta_input.last_output.clone();
        let forget = self.porta_forget.last_output.clone();
        let candidato = self.porta_candidato.last_output.clone();
        let c = &self.estado.memoria_c;

        let tanh_c: Vec<f64> = c.iter().map(|v| v.tanh()).collect();

        let d_o: Vec<f64> = d_h
            .iter()
            .zip(tanh_c.iter())
            .map(|(dh, tc)| dh * tc)
            .collect();

        let d_c: Vec<f64> = d_h
            .iter()
            .zip(output.iter())
            .map(|(dh, o)| dh * o * (1.0 - tanh_c[0].powi(2)))
            .collect();

        let d_f: Vec<f64> = d_c
            .iter()
            .zip(self.estado.memoria_c.iter())
            .map(|(dc, prev_c)| dc * *prev_c)
            .collect();

        let d_i: Vec<f64> = d_c
            .iter()
            .zip(candidato.iter())
            .map(|(dc, g)| dc * *g)
            .collect();

        let d_g: Vec<f64> = d_c
            .iter()
            .zip(input.iter())
            .map(|(dc, i)| dc * *i)
            .collect();

        let grad_f = self.porta_forget.backward(&entrada_completa, &d_f);
        let grad_i = self.porta_input.backward(&entrada_completa, &d_i);
        let grad_o = self.porta_output.backward(&entrada_completa, &d_o);
        let grad_g = self.porta_candidato.backward(&entrada_completa, &d_g);

        let grad_total = grad_f
            .iter()
            .zip(grad_i.iter())
            .zip(grad_o.iter())
            .zip(grad_g.iter())
            .map(|(((f, i), o), g)| f + i + o + g)
            .collect();

        grad_total
    }
}
