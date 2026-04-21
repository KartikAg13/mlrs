use ndarray::{Array1, Array2};

use crate::constants::{
    DEFAULT_ALPHA, DEFAULT_L1_RATIO, DEFAULT_L2_RATIO, DEFAULT_LEARNING_RATE, DEFAULT_MAX_EPOCHS,
    DEFAULT_TOLERANCE,
};
use crate::model::ModelError;
use crate::model::activator::Activation;
use crate::model::optimizer::gradient_descent::GradientDescent;

pub struct LogisticRegressor {
    solver: GradientDescent,
    threshold: f64,
}

impl Default for LogisticRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl LogisticRegressor {
    fn new() -> Self {
        Self {
            solver: GradientDescent::new(
                DEFAULT_LEARNING_RATE,
                DEFAULT_MAX_EPOCHS,
                DEFAULT_TOLERANCE,
                DEFAULT_L1_RATIO,
                DEFAULT_L2_RATIO,
                DEFAULT_ALPHA,
                Activation::Sigmoid,
            ),
            threshold: 0.5,
        }
    }
}
