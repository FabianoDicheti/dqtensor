/// Aplica uma função gate tipo sigmoide sobre a entrada `x`
/// Neste caso, retorna um vetor de mesma dimensão, com cada elemento ∈ (0, 1)
pub fn sigmoid_gate(x: &Vec<f64>) -> Vec<f64> {
    x.iter().map(|xi| 1.0 / (1.0 + (-xi).exp())).collect()
}


// impl DynamicSSMExecutor {
//     pub fn simulate(&self, inputs: &Vec<Vec<f64>>, h0: Vec<f64>) -> Vec<f64> {
//         let mut h = h0.clone();
//         let mut outputs = Vec::with_capacity(inputs.len());

//         for x_t in inputs {
//             let (A_t, B_t, Delta_t) = self.param_generator.generate(x_t);

//             let mut A_bar = vec![0.0; self.state_dim];
//             let mut B_bar = vec![0.0; self.state_dim];
//             for i in 0..self.state_dim {
//                 let a_dt = A_t[i] * Delta_t[i];
//                 A_bar[i] = a_dt.exp();

//                 if A_t[i].abs() > 1e-8 {
//                     B_bar[i] = ((A_bar[i] - 1.0) / A_t[i]) * B_t[i];
//                 } else {
//                     B_bar[i] = B_t[i] * Delta_t[i];
//                 }
//             }

//             // h_{t+1} = A_bar * h + B_bar * x_sum
//             let x_sum = x_t.iter().copied().sum::<f64>();
//             let mut h_next = vec![0.0; self.state_dim];
//             for i in 0..self.state_dim {
//                 h_next[i] = A_bar[i] * h[i] + B_bar[i] * x_sum;
//             }

//             // y_t = C * h
//             let y_raw = self.C.iter().zip(h.iter()).map(|(c, hi)| c * hi).sum::<f64>();

//             // Gate(x_t) ∈ (0, 1)
//             let gate = sigmoid_gate(x_t);
//             let gate_factor = gate.iter().copied().sum::<f64>() / gate.len() as f64; // média do gate

//             outputs.push(gate_factor * y_raw);
//             h = h_next;
//         }

//         outputs
//     }
// }
