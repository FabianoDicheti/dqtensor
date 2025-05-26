use std::fmt;
use std::collections::HashMap;

/// Trait base para qualquer otimizador
pub trait Optimizer: CloneBoxOptimizer + fmt::Debug {
    fn update(&mut self, params: &mut [f64], grads: &[f64]);
}

/// Trait auxiliar para permitir `Clone` em `Box<dyn Optimizer>`
pub trait CloneBoxOptimizer {
    fn clone_box(&self) -> Box<dyn Optimizer>;
}

impl<T> CloneBoxOptimizer for T
where
    T: 'static + Optimizer + Clone,
{
    fn clone_box(&self) -> Box<dyn Optimizer> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn Optimizer> {
    fn clone(&self) -> Box<dyn Optimizer> {
        self.clone_box()
    }
}


//
// =============================== SGD ==================================
//
#[derive(Clone, Debug)]
pub struct SGD {
    pub learning_rate: f64,
}

impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for SGD {
    fn update(&mut self, params: &mut [f64], grads: &[f64]) {
        for (p, g) in params.iter_mut().zip(grads.iter()) {
            *p -= self.learning_rate * g;
        }
    }
}

//
// =============================== Adam ==================================
//
#[derive(Clone, Debug)]
pub struct Adam {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub time_step: u64,
    pub m: Vec<f64>,
    pub v: Vec<f64>,
}

impl Adam {
    pub fn new(
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        param_size: usize,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            time_step: 0,
            m: vec![0.0; param_size],
            v: vec![0.0; param_size],
        }
    }
}

impl Optimizer for Adam {
    fn update(&mut self, params: &mut [f64], grads: &[f64]) {
        self.time_step += 1;
        let t = self.time_step as f64;
        let beta1_t = self.beta1.powf(t);
        let beta2_t = self.beta2.powf(t);

        for i in 0..params.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grads[i].powi(2);

            let m_hat = self.m[i] / (1.0 - beta1_t);
            let v_hat = self.v[i] / (1.0 - beta2_t);

            params[i] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }
}

//
// =============================== RMSProp ==================================
//
#[derive(Clone, Debug)]
pub struct RMSProp {
    pub learning_rate: f64,
    pub gamma: f64,
    pub epsilon: f64,
    pub cache: Vec<f64>,
}

impl RMSProp {
    pub fn new(learning_rate: f64, gamma: f64, epsilon: f64, param_size: usize) -> Self {
        Self {
            learning_rate,
            gamma,
            epsilon,
            cache: vec![0.0; param_size],
        }
    }
}

impl Optimizer for RMSProp {
    fn update(&mut self, params: &mut [f64], grads: &[f64]) {
        for i in 0..params.len() {
            self.cache[i] = self.gamma * self.cache[i] + (1.0 - self.gamma) * grads[i].powi(2);

            let adapted_lr = self.learning_rate / (self.cache[i].sqrt() + self.epsilon);

            params[i] -= adapted_lr * grads[i];
        }
    }
}

//
// =============================== Momentum ==================================
//
#[derive(Clone, Debug)]
pub struct Momentum {
    pub learning_rate: f64,
    pub gamma: f64,
    pub velocity: Vec<f64>,
}

impl Momentum {
    pub fn new(learning_rate: f64, gamma: f64, param_size: usize) -> Self {
        Self {
            learning_rate,
            gamma,
            velocity: vec![0.0; param_size],
        }
    }
}

impl Optimizer for Momentum {
    fn update(&mut self, params: &mut [f64], grads: &[f64]) {
        for i in 0..params.len() {
            self.velocity[i] = self.gamma * self.velocity[i] + self.learning_rate * grads[i];
            params[i] -= self.velocity[i];
        }
    }
}

//
// =============================== Nadam ==================================
//
#[derive(Clone, Debug)]
pub struct Nadam {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub time_step: u64,
    pub m: Vec<f64>,
    pub v: Vec<f64>,
}

impl Nadam {
    pub fn new(
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        param_size: usize,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            time_step: 0,
            m: vec![0.0; param_size],
            v: vec![0.0; param_size],
        }
    }
}

impl Optimizer for Nadam {
    fn update(&mut self, params: &mut [f64], grads: &[f64]) {
        self.time_step += 1;
        let t = self.time_step as f64;

        let beta1_t = self.beta1.powf(t);
        let beta1_t_next = self.beta1 * beta1_t;
        let beta2_t = self.beta2.powf(t);

        for i in 0..params.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grads[i].powi(2);

            let m_hat = (self.beta1 * self.m[i]) / (1.0 - beta1_t_next)
                + ((1.0 - self.beta1) * grads[i]) / (1.0 - beta1_t);

            let v_hat = self.v[i] / (1.0 - beta2_t);

            params[i] -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
        }
    }
}

//
// =============================== AdaGrad ==================================
//
#[derive(Clone, Debug)]
pub struct AdaGrad {
    pub learning_rate: f64,
    pub epsilon: f64,
    pub cache: Vec<f64>,
}

impl AdaGrad {
    pub fn new(learning_rate: f64, epsilon: f64, param_size: usize) -> Self {
        Self {
            learning_rate,
            epsilon,
            cache: vec![0.0; param_size],
        }
    }
}

impl Optimizer for AdaGrad {
    fn update(&mut self, params: &mut [f64], grads: &[f64]) {
        for i in 0..params.len() {
            self.cache[i] += grads[i].powi(2);

            let adapted_lr = self.learning_rate / (self.cache[i].sqrt() + self.epsilon);

            params[i] -= adapted_lr * grads[i];
        }
    }
}
