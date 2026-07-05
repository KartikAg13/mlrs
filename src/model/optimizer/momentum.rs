use ndarray::{Array1, Array2, Zip};
use std::collections::HashMap;

use super::OptimizingStrategy;
use crate::constants::{DEFAULT_LEARNING_RATE, DEFAULT_MOMENTUM};
use crate::model::error::ModelError;
use crate::model::layer::Layer;

pub struct Momentum {
    pub learning_rate: f64,
    pub momentum: f64,
    velocity: Vec<(Array2<f64>, Array1<f64>)>,
}

impl Momentum {
    pub fn new() -> Self {
        Self {
            learning_rate: DEFAULT_LEARNING_RATE,
            momentum: DEFAULT_MOMENTUM,
            velocity: Vec::new(),
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

    pub fn with_momentum(mut self, momentum: f64) -> Self {
        if self.momentum != momentum {
            ModelError::print_modifying(format!("Momentum from {} to {}", self.momentum, momentum));
            self.momentum = momentum;
        }
        self
    }
}

impl Default for Momentum {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizingStrategy for Momentum {
    fn reset(&mut self) {
        self.velocity.clear();
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
        map.insert("momentum".to_string(), self.momentum);
        map
    }

    fn step(&mut self, layers: &mut Vec<Layer>, gradients: &[(Array2<f64>, Array1<f64>)]) {
        let lr = self.learning_rate;
        let momentum = self.momentum;

        if self.velocity.is_empty() {
            self.velocity = layers
                .iter()
                .map(|layer| {
                    (
                        Array2::zeros(layer.weights.raw_dim()),
                        Array1::zeros(layer.bias.len()),
                    )
                })
                .collect()
        }

        for ((layer, (dw, db)), (vw, vb)) in layers
            .iter_mut()
            .zip(gradients.iter())
            .zip(self.velocity.iter_mut())
        {
            Zip::from(layer.weights.view_mut())
                .and(vw.view_mut())
                .and(dw.view())
                .for_each(|w, v, &g| {
                    *v = momentum * *v + lr * g;
                    *w -= *v;
                });

            Zip::from(layer.bias.view_mut())
                .and(vb.view_mut())
                .and(db.view())
                .for_each(|b, v, &g| {
                    *v = momentum * *v + lr * g;
                    *b -= *v;
                });
        }
    }
}
