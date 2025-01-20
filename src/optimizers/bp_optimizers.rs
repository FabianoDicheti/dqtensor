pub struct Adam {
    pub learning_rate: f64,
    pub beta1: f64, //decay for momentum (exponential)
    pub beta2: f64, //decay for RMSprop (exponential)
    pub epsilon: f64, // stability
    pub time_step: u64,
    pub momentum_first: Vec<f64>, // first order
    pub rmsprop_second: Vec<f64>, // second order
}

impl Adam {
    pub fn new(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64, param_size: usize) -> Self {
        Self {
            learning_rate, 
            beta1,
            beta2,
            epsilon,
            time_step: 0,
            momentum_first: vec![0.0; param_size], // zeros init
            rmsprop_second: vec![0.0; param_size], 
        }
    }

    pub fn update(&mut self, params: &mut [f64], grads: &[f64]) {
        self.time_step += 1;
        let beta1_time = self.beta1.powi(self.time_step as i32);
        let beta2_time = self.beta2.powi(self.time_step as i32);

        for i in 0..params.len(){
            self.momentum_first[i] = self.beta1 * self.momentum_first[i] + (1.0 - self.beta1) * grads[i]; // update
            self.rmsprop_second[i] = self.beta2 * self.rmsprop_second[i] + (1.0 - self.beta2) * grads[i] * grads[i]; //update

            let momentum_hat = self.momentum_first[i] / (1.0 - beta1_time); //bias correction
            let rmsprop_hat = self.rmsprop_second[i] / (1.0 - beta2_time);  //bias correction

            params[i] -= self.learning_rate * momentum_hat / (rmsprop_hat.sqrt() + self.epsilon); // final params update
        }

    }
}

