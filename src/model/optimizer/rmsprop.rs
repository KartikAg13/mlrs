use ndarray::{Array1, Array2, Zip};
use std::collections::HashMap;

use super::OptimizingStrategy;
use crate::{
    constants::{DEFAULT_BETA, DEFAULT_LEARNING_RATE, DEFAULT_TOLERANCE},
    model::{error::ModelError, layer::Layer},
};

pub struct RMSProp {
    pub learning_rate: f64,
    pub beta: f64,
    pub epsilon: f64,

    squared_grad: Vec<(Array2<f64>, Array1<f64>)>,
}

impl RMSProp {
    pub fn new() -> Self {
        Self {
            learning_rate: DEFAULT_LEARNING_RATE,
            beta: DEFAULT_BETA,
            epsilon: DEFAULT_TOLERANCE,
            squared_grad: Vec::new(),
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

    pub fn with_beta(mut self, beta: f64) -> Self {
        if self.beta != beta {
            ModelError::print_modifying(format!("Beta from {} to {}", self.beta, beta));
            self.beta = beta;
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

impl Default for RMSProp {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizingStrategy for RMSProp {
    fn reset(&mut self) {
        self.squared_grad.clear();
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }

    fn hyperparameters(&self) -> HashMap<String, f64> {
        let mut map: HashMap<String, f64> = HashMap::new();
        map.insert("learning_rate".to_string(), self.learning_rate);
        map.insert("beta".to_string(), self.beta);
        map.insert("epsilon".to_string(), self.epsilon);
        map
    }

    fn step(&mut self, layers: &mut Vec<Layer>, gradients: &[(Array2<f64>, Array1<f64>)]) {
        if self.squared_grad.is_empty() {
            self.squared_grad = layers
                .iter()
                .map(|layer| {
                    (
                        Array2::zeros(layer.weights.raw_dim()),
                        Array1::zeros(layer.bias.len()),
                    )
                })
                .collect();
        }

        let beta = self.beta;
        let one_minus_beta = 1.0 - beta;
        let lr = self.learning_rate;
        let epsilon = self.epsilon;

        for ((layer, (dw, db)), (sw, sb)) in layers
            .iter_mut()
            .zip(gradients.iter())
            .zip(self.squared_grad.iter_mut())
        {
            Zip::from(layer.weights.view_mut())
                .and(sw.view_mut())
                .and(dw.view())
                .for_each(|w, s, &g| {
                    *s = beta * *s + one_minus_beta * g * g;
                    *w -= lr * g / (s.sqrt() + epsilon);
                });

            Zip::from(layer.bias.view_mut())
                .and(sb.view_mut())
                .and(db.view())
                .for_each(|b, s, &g| {
                    *s = beta * *s + one_minus_beta * g * g;
                    *b -= lr * g / (s.sqrt() + epsilon);
                });
        }
    }
}
