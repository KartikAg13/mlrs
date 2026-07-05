use ndarray::{Array1, Array2, Zip};
use std::collections::HashMap;

use super::OptimizingStrategy;
use crate::{
    constants::{DEFAULT_ALPHA, DEFAULT_L1_RATIO, DEFAULT_L2_RATIO, DEFAULT_LEARNING_RATE},
    model::{error::ModelError, layer::Layer},
};

pub struct GradientDescent {
    pub learning_rate: f64,
    pub l1_ratio: f64,
    pub l2_ratio: f64,
    pub alpha: f64,
}

impl GradientDescent {
    pub fn new() -> Self {
        Self {
            learning_rate: DEFAULT_LEARNING_RATE,
            l1_ratio: DEFAULT_L1_RATIO,
            l2_ratio: DEFAULT_L2_RATIO,
            alpha: DEFAULT_ALPHA,
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

    pub fn with_l1_ratio(mut self, l1_ratio: f64) -> Self {
        if self.l1_ratio != l1_ratio {
            ModelError::print_modifying(format!("L1 ratio from {} to {}", self.l1_ratio, l1_ratio));
            self.l1_ratio = l1_ratio;
        }
        self
    }

    pub fn with_l2_ratio(mut self, l2_ratio: f64) -> Self {
        if self.l2_ratio != l2_ratio {
            ModelError::print_modifying(format!("L2 ratio from {} to {}", self.l2_ratio, l2_ratio));
            self.l2_ratio = l2_ratio;
        }
        self
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        if self.alpha != alpha {
            ModelError::print_modifying(format!("Alpha from {} to {}", self.alpha, alpha));
            self.alpha = alpha;
        }
        self
    }
}

impl Default for GradientDescent {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizingStrategy for GradientDescent {
    fn reset(&mut self) {}

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }

    fn hyperparameters(&self) -> HashMap<String, f64> {
        let mut map: HashMap<String, f64> = HashMap::new();
        map.insert("learning_rate".to_string(), self.learning_rate);
        map.insert("l1_ratio".to_string(), self.l1_ratio);
        map.insert("l2_ratio".to_string(), self.l2_ratio);
        map.insert("alpha".to_string(), self.alpha);
        map
    }

    fn step(&mut self, layers: &mut Vec<Layer>, gradients: &[(Array2<f64>, Array1<f64>)]) {
        let lr = self.learning_rate;
        let l1_threshold = lr * self.alpha * self.l1_ratio;
        let l2_scale = self.alpha * self.l2_ratio;

        for (layer, (dw, db)) in layers.iter_mut().zip(gradients.iter()) {
            Zip::from(&mut layer.weights).and(dw).for_each(|w, &d| {
                *w -= lr * (d + l2_scale * *w);
            });

            Zip::from(&mut layer.bias)
                .and(db)
                .for_each(|b, &d| *b -= lr * d);

            if l1_threshold > 0.0 {
                layer.weights.mapv_inplace(|w| {
                    if w > l1_threshold {
                        w - l1_threshold
                    } else if w < -l1_threshold {
                        w + l1_threshold
                    } else {
                        0.0
                    }
                });
            }
        }
    }
}
