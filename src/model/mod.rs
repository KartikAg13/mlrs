pub use classifier::LogisticRegressor;
pub use linear::LinearRegressor;

use colored::Colorize;
use ndarray::{Array1, Array2};
use thiserror::Error;

use crate::model::optimizer::gradient_descent::GradientDescent;
use crate::score::r2_score;

pub mod activator;
pub mod classifier;
pub mod linear;
pub mod optimizer;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("Model is not fitted yet. Call .fit() first.")]
    NotFitted,

    #[error("Shape misatch. X has {0} rows, Y has {1} rows.")]
    ShapeMismatch(usize, usize),
}

impl ModelError {
    pub fn print_error(&self) {
        eprintln!("{}{}", "ERROR: ".red().bold(), self.to_string().red());
    }

    pub fn print_warning(message: String) {
        eprintln!("{}{}", "WARNING: ".yellow().bold(), message.yellow());
    }

    pub fn print_modifying(message: String) {
        println!("{}{}", "MODIFYING: ".green(), message.green());
    }
}

pub trait ModelingStrategy {}

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
            ModelError::print_warning(format!("Sum of l1 and l2 ratio is greater than 1.0."));
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
            ModelError::print_warning(format!("Sum of l1 and l2 ratio is greater than 1.0."));
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
        self.solver.predict(x_test)
    }

    pub fn evaluate(&mut self, x_test: &Option<Array2<f64>>, y_test: &Array1<f64>) -> f64 {
        // should get routed to the correct score automatically

        match x_test {
            Some(x) => match self.solver.predict(x) {
                Ok(_) => r2_score(y_test, &self.solver.y_predicted),
                Err(error) => {
                    error.print_error();
                    -1.0
                }
            },
            None => r2_score(y_test, &self.solver.y_predicted),
        }
    }

    pub fn weights(&self) -> Array1<f64> {
        self.solver.weights.clone()
    }

    pub fn bias(&self) -> f64 {
        self.solver.bias
    }
}
