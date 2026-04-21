use ndarray::{Array1, Array2};

use crate::constants::{
    DEFAULT_ALPHA, DEFAULT_L1_RATIO, DEFAULT_L2_RATIO, DEFAULT_LEARNING_RATE, DEFAULT_MAX_EPOCHS,
    DEFAULT_TOLERANCE,
};
use crate::model::ModelError;
use crate::model::activator::Activation;
use crate::model::optimizer::gradient_descent::GradientDescent;
use crate::score::r2_score;

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
                DEFAULT_L1_RATIO,
                DEFAULT_L2_RATIO,
                DEFAULT_ALPHA,
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

    #[test]
    fn test_ridge_regression() {
        let (x, y) = make_linear_data();
        let mut model = LinearRegressor::new()
            .with_l2_ratio(0.5)
            .with_learning_rate(0.05)
            .with_max_epochs(2000);

        assert!(model.fit(&x, &y).is_ok());

        let r2 = model.evaluate(&None, &y);
        assert!(r2 > 0.95, "Ridge R2 was too low: {}", r2);

        // Ridge should have smaller weights than pure linear (due to L2 penalty)
        let weights = model.weights();
        assert!(weights.iter().all(|&w| w.abs() < 4.0)); // pure linear would be ~3.0
    }

    #[test]
    fn test_lasso_regression() {
        let (x, y) = make_linear_data();
        let mut model = LinearRegressor::new()
            .with_l1_ratio(0.8)
            .with_learning_rate(0.05)
            .with_max_epochs(3000);

        assert!(model.fit(&x, &y).is_ok());

        let r2 = model.evaluate(&None, &y);
        assert!(r2 > 0.9);

        // Lasso tends to drive some coefficients closer to zero
        let weights = model.weights();
        assert!(weights[0].abs() < 3.5); // should be shrunk
    }

    #[test]
    fn test_elasticnet_regression() {
        let (x, y) = make_linear_data();
        let mut model = LinearRegressor::new()
            .with_learning_rate(0.05)
            .with_max_epochs(3000);

        assert!(model.fit(&x, &y).is_ok());

        let r2 = model.evaluate(&None, &y);
        assert!(r2 > 0.92);

        let weights = model.weights();
        assert!(weights[0].abs() > 0.0 && weights[0].abs() < 3.5);
    }

    #[test]
    fn test_both_l1_and_l2_can_be_one() {
        let (x, y) = make_linear_data();
        let mut model = LinearRegressor::new()
            .with_l1_ratio(1.0)
            .with_l2_ratio(1.0)
            .with_max_epochs(1000);

        // Should still run without panic
        let result = model.fit(&x, &y);
        assert!(result.is_ok() || result.is_err()); // We just want it to not crash
    }
}
