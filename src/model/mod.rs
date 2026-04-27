pub use classifier::LogisticRegressor;
pub use linear::LinearRegressor;

use ndarray::{Array1, Array2};

use crate::model::activator::Activation;
use crate::model::error::ModelError;
use crate::model::optimizer::gradient_descent::GradientDescent;

pub mod activator;
pub mod classifier;
pub mod error;
pub mod linear;
pub mod optimizer;

pub trait ModelingStrategy {
    fn activation(&self) -> &Activation;

    fn score(&self, y_true: &Array1<f64>, y_predited: &Array1<f64>) -> f64;

    fn threshold(&self) -> f64 {
        0.5
    }
}

pub struct ModelHandler<T: ModelingStrategy> {
    pub solver: GradientDescent,
    pub config: T,
}

impl<T: ModelingStrategy> ModelHandler<T> {
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        if self.solver.learning_rate != learning_rate {
            ModelError::print_modifying(format!(
                "Changing learning rate from {} to {}!",
                self.solver.learning_rate, learning_rate
            ));
        }
        self.solver.learning_rate = learning_rate;
        self
    }

    pub fn with_max_epochs(mut self, max_epochs: usize) -> Self {
        if self.solver.max_epochs != max_epochs {
            ModelError::print_modifying(format!(
                "Changing max epochs from {} to {}!",
                self.solver.max_epochs, max_epochs
            ));
        }
        self.solver.max_epochs = max_epochs;
        self
    }

    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        if self.solver.tolerance != tolerance {
            ModelError::print_modifying(format!(
                "Changing tolerance from {} to {}!",
                self.solver.tolerance, tolerance
            ));
        }
        self.solver.tolerance = tolerance;
        self
    }

    pub fn with_l1_ratio(mut self, l1_ratio: f64) -> Self {
        let l1 = l1_ratio.clamp(0.0, 1.0);
        if (l1 + self.solver.l2_ratio) > 1.0 {
            ModelError::print_warning(format!("Sum of l1 and l2 ratio is greater than {}.", 1.0));
        }
        if self.solver.l1_ratio != l1 {
            ModelError::print_modifying(format!(
                "Changing l1 ratio from {} to {}!",
                self.solver.l1_ratio, l1
            ));
        }
        self.solver.l1_ratio = l1;
        self
    }

    pub fn with_l2_ratio(mut self, l2_ratio: f64) -> Self {
        let l2 = l2_ratio.clamp(0.0, 1.0);
        if (l2 + self.solver.l1_ratio) > 1.0 {
            ModelError::print_warning(format!("Sum of l1 and l2 ratio is greater than {}.", 1.0));
        }
        if self.solver.l2_ratio != l2 {
            ModelError::print_modifying(format!(
                "Changing l2 ratio from {} to {}!",
                self.solver.l2_ratio, l2
            ));
        }
        self.solver.l2_ratio = l2;
        self
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        let al = alpha.clamp(0.0, 1.0);
        if self.solver.alpha != al {
            ModelError::print_modifying(format!(
                "Changing alpha from {} to {}!",
                self.solver.alpha, al
            ));
        }
        self.solver.alpha = al;
        self
    }

    pub fn fit(&mut self, x_train: &Array2<f64>, y_train: &Array1<f64>) -> Result<(), ModelError> {
        self.solver.fit(x_train, y_train)
    }

    pub fn predict(&mut self, x_test: &Array2<f64>) -> Result<Array1<f64>, ModelError> {
        let raw = self.solver.predict(x_test)?;
        let predictions = match self.config.activation() {
            Activation::Identity => raw,
            Activation::Sigmoid => {
                let threshold = self.config.threshold();
                raw.mapv(|p| if p >= threshold { 1.0 } else { 0.0 })
            }
        };
        Ok(predictions)
    }

    pub fn evaluate(&mut self, x_test: &Option<Array2<f64>>, y_test: &Array1<f64>) -> f64 {
        let predictions = match x_test {
            Some(x) => match self.predict(x) {
                Ok(v) => v,
                Err(error) => {
                    error.print_error();
                    return -1.0;
                }
            },
            None => self.solver.y_predicted.clone(),
        };
        self.config.score(y_test, &predictions)
    }

    pub fn weights(&self) -> Array1<f64> {
        self.solver.weights.clone()
    }

    pub fn bias(&self) -> f64 {
        self.solver.bias
    }
}
