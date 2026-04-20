use ndarray::{Array1, Array2};

use crate::model::ModelError;
use crate::model::activator::Activation;
use crate::model::optimizer::gradient_descent::GradientDescent;
use crate::score::r2_score;

const DEFAULT_LEARNING_RATE: f64 = 0.01;
const DEFAULT_MAX_EPOCHS: usize = 10000;
const DEFAULT_TOLERANCE: f64 = 1e-4;

pub struct LinearRegressor {
    solver: GradientDescent,
}

impl Default for LinearRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearRegressor {
    pub fn new() -> Self {
        Self {
            solver: GradientDescent::new(
                DEFAULT_LEARNING_RATE,
                DEFAULT_MAX_EPOCHS,
                DEFAULT_TOLERANCE,
                0.0,
                0.0,
                0.0,
                Activation::Identity,
            ),
        }
    }

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

    pub fn fit(&mut self, x_train: &Array2<f64>, y_train: &Array1<f64>) -> Result<(), ModelError> {
        self.solver.fit(x_train, y_train)
    }

    pub fn predict(&mut self, x_test: &Array2<f64>) -> Result<Array1<f64>, ModelError> {
        self.solver.predict(x_test)
    }

    pub fn evaluate(&mut self, x_test: &Option<Array2<f64>>, y_test: &Array1<f64>) -> f64 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    fn make_linear_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![10.0, 13.0, 16.0, 19.0, 22.0]; // 3x + 7
        (x, y)
    }

    #[test]
    fn test_linear_regression_fit_predict() {
        let (x, y) = make_linear_data();
        let mut model = LinearRegressor::new()
            .with_learning_rate(0.01)
            .with_max_epochs(5000)
            .with_tolerance(1e-6);
        assert!(model.fit(&x, &y).is_ok());

        let preds = model.predict(&x).unwrap();
        for (p, t) in preds.iter().zip(y.iter()) {
            assert!((p - t).abs() < 0.5, "pred={} target={}", p, t);
        }
    }

    #[test]
    fn test_linear_regression_score() {
        let (x, y) = make_linear_data();
        let mut model = LinearRegressor::new()
            .with_learning_rate(0.01)
            .with_max_epochs(5000);
        model.fit(&x, &y).unwrap();
        let r2 = model.evaluate(&None, &y);
        assert!(r2 > 0.99, "R2 should be close to 1.0, got {}", r2);
    }

    #[test]
    fn test_linear_regression_not_fitted_errors() {
        let mut model = LinearRegressor::new();
        let x = Array2::zeros((3, 1));
        assert!(matches!(model.predict(&x), Err(ModelError::NotFitted)));
    }

    #[test]
    fn test_linear_regression_shape_mismatch() {
        let (x, _) = make_linear_data();
        let mut model = LinearRegressor::new();
        // y has wrong length
        let y_bad = array![1.0, 2.0];
        assert!(matches!(
            model.fit(&x, &y_bad),
            Err(ModelError::ShapeMismatch(_, _))
        ));
    }

    #[test]
    fn test_linear_predict_shape_mismatch() {
        let (x, y) = make_linear_data();
        let mut model = LinearRegressor::new().with_max_epochs(100);
        model.fit(&x, &y).unwrap();
        // x_test has wrong number of features
        let x_bad = Array2::zeros((3, 5));
        assert!(matches!(
            model.predict(&x_bad),
            Err(ModelError::ShapeMismatch(_, _))
        ));
    }
}
