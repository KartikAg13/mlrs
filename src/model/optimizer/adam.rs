use ndarray::{Array1, Array2, Zip};
use std::collections::HashMap;

use super::OptimizingStrategy;
use crate::model::error::ModelError;
use crate::{
    constants::{DEFAULT_BETA, DEFAULT_BETA2, DEFAULT_LEARNING_RATE, DEFAULT_TOLERANCE},
    model::layer::Layer,
};

type AdamMoments = (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>);

pub struct Adam {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,

    t: usize,
    moments: Vec<AdamMoments>,
}

impl Adam {
    pub fn new() -> Self {
        Self {
            learning_rate: DEFAULT_LEARNING_RATE,
            beta1: DEFAULT_BETA,
            beta2: DEFAULT_BETA2,
            epsilon: DEFAULT_TOLERANCE,
            t: 0,
            moments: Vec::new(),
        }
    }

    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        if self.learning_rate != learning_rate {
            ModelError::print_modifying(format!(
                "Learning rate from {} to {}",
                self.learning_rate, learning_rate
            ));
            self.learning_rate = learning_rate;
        }
        self
    }

    pub fn with_beta1(mut self, beta1: f64) -> Self {
        if self.beta1 != beta1 {
            ModelError::print_modifying(format!("Beta1 from {} to {}", self.beta1, beta1));
            self.beta1 = beta1;
        }
        self
    }

    pub fn with_beta2(mut self, beta2: f64) -> Self {
        if self.beta2 != beta2 {
            ModelError::print_modifying(format!("Beta2 from {} to {}", self.beta2, beta2));
            self.beta2 = beta2;
        }
        self
    }

    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        if self.epsilon != epsilon {
            ModelError::print_modifying(format!("Epsilon from {} to {}", self.epsilon, epsilon));
            self.epsilon = epsilon;
        }
        self
    }
}

impl Default for Adam {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizingStrategy for Adam {
    fn reset(&mut self) {
        self.t = 0;
        self.moments.clear();
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }

    fn hyperparameters(&self) -> HashMap<String, f64> {
        let mut map = HashMap::new();
        map.insert("learning_rate".to_string(), self.learning_rate);
        map.insert("beta1".to_string(), self.beta1);
        map.insert("beta2".to_string(), self.beta2);
        map.insert("epsilon".to_string(), self.epsilon);
        map
    }

    fn step(&mut self, layers: &mut Vec<Layer>, gradients: &[(Array2<f64>, Array1<f64>)]) {
        if self.moments.is_empty() {
            self.moments = layers
                .iter()
                .map(|layer| {
                    (
                        Array2::zeros(layer.weights.raw_dim()),
                        Array1::zeros(layer.bias.len()),
                        Array2::zeros(layer.weights.raw_dim()),
                        Array1::zeros(layer.bias.len()),
                    )
                })
                .collect();
        }

        self.t += 1;

        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let one_minus_beta1 = 1.0 - beta1;
        let one_minus_beta2 = 1.0 - beta2;
        let lr = self.learning_rate;
        let epsilon = self.epsilon;

        let bias_correction1 = 1.0 - beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - beta2.powi(self.t as i32);

        for ((layer, (dw, db)), (mw, mb, vw, vb)) in layers
            .iter_mut()
            .zip(gradients.iter())
            .zip(self.moments.iter_mut())
        {
            Zip::from(layer.weights.view_mut())
                .and(mw.view_mut())
                .and(vw.view_mut())
                .and(dw.view())
                .for_each(|w, m, v, &g| {
                    *m = beta1 * *m + one_minus_beta1 * g;
                    *v = beta2 * *v + one_minus_beta2 * g * g;

                    let m_hat = *m / bias_correction1;
                    let v_hat = *v / bias_correction2;

                    *w -= lr * m_hat / (v_hat.sqrt() + epsilon);
                });

            Zip::from(layer.bias.view_mut())
                .and(mb.view_mut())
                .and(vb.view_mut())
                .and(db.view())
                .for_each(|b, m, v, &g| {
                    *m = beta1 * *m + one_minus_beta1 * g;
                    *v = beta2 * *v + one_minus_beta2 * g * g;

                    let m_hat = *m / bias_correction1;
                    let v_hat = *v / bias_correction2;

                    *b -= lr * m_hat / (v_hat.sqrt() + epsilon);
                });
        }
    }
}
