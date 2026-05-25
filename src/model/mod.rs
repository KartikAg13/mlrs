pub use classifier::LogisticRegressor;
pub use linear::LinearRegressor;

use ndarray::{Array1, Array2};

use crate::model::activator::ActivationStrategy;
use crate::model::error::ModelError;
use crate::model::optimizer::OptimizingStrategy;

pub mod activator;
pub mod classifier;
pub mod error;
pub mod layer;
pub mod linear;
pub mod optimizer;

pub trait ModelingStrategy {
    fn activation(&self) -> &Activation;

    fn score(&self, y_true: &Array1<f64>, y_predited: &Array1<f64>) -> f64;

    fn threshold(&self) -> f64 {
        0.5
    }
}

pub struct ModelHandler<T: ModelingStrategy, O: OptimizingStrategy> {
    pub solver: O,
    pub config: T,
}

impl<T: ModelingStrategy> ModelHandler<T, Box<dyn OptimizingStrategy>> {
    pub fn with_optimizer<O2: OptimizingStrategy + 'static>(mut self, optimizer: O2) -> Self {
        self.solver = Box::new(optimizer);
        self
    }
}

impl<T: ModelingStrategy, O: OptimizingStrategy> ModelHandler<T, O> {
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        let base = self.solver.base_mut();

        if base.learning_rate != learning_rate {
            ModelError::print_modifying(format!(
                "Changing learning rate from {} to {}!",
                base.learning_rate, learning_rate
            ));
        }

        base.learning_rate = learning_rate;
        self
    }

    pub fn with_max_epochs(mut self, max_epochs: usize) -> Self {
        let base = self.solver.base_mut();

        if base.max_epochs != max_epochs {
            ModelError::print_modifying(format!(
                "Changing max epochs from {} to {}!",
                base.max_epochs, max_epochs
            ));
        }

        base.max_epochs = max_epochs;
        self
    }

    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        let base = self.solver.base_mut();

        if base.tolerance != tolerance {
            ModelError::print_modifying(format!(
                "Changing tolerance from {} to {}!",
                base.tolerance, tolerance
            ));
        }

        base.tolerance = tolerance;
        self
    }

    pub fn fit(&mut self, x_train: &Array2<f64>, y_train: &Array1<f64>) -> Result<(), ModelError> {
        self.solver.fit(x_train, y_train)
    }

    pub fn predict(&mut self, x_test: &Array2<f64>) -> Result<Array1<f64>, ModelError> {
        let mut raw = self.solver.predict(x_test)?;
        self.solver.base().activation.apply_inplace(&mut raw);
        Ok(raw)
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
            None => self.solver.base().y_predicted.clone(),
        };

        self.config.score(y_test, &predictions)
    }

    pub fn weights(&self) -> Array1<f64> {
        self.solver.base().weights.clone()
    }

    pub fn bias(&self) -> f64 {
        self.solver.base().bias
    }
}
