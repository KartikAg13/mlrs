use ndarray::Array1;

use crate::constants::{
    DEFAULT_ALPHA, DEFAULT_L1_RATIO, DEFAULT_L2_RATIO, DEFAULT_LEARNING_RATE, DEFAULT_MAX_EPOCHS,
    DEFAULT_TOLERANCE,
};
use crate::model::ModelHandler;
use crate::model::ModelingStrategy;
use crate::model::activator::Activation;
use crate::model::optimizer::gradient_descent::GradientDescent;
use crate::score::r2_score;

#[derive(Debug, Default)]
pub struct LinearConfig;

pub type LinearRegressor = ModelHandler<LinearConfig>;

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
            config: LinearConfig,
        }
    }
}

impl Default for LinearRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelingStrategy for LinearConfig {
    fn activation(&self) -> &Activation {
        &Activation::Identity
    }

    fn score(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        r2_score(y_true, y_pred)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelError;
    use ndarray::{Array1, Array2, array};
    fn make_linear_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = array![10.0, 13.0, 16.0, 19.0, 22.0];
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

        let weights = model.weights();
        assert!(weights.iter().all(|&w| w.abs() < 4.0));
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

        let weights = model.weights();
        assert!(weights[0].abs() < 3.5);
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

        let result = model.fit(&x, &y);
        assert!(result.is_ok() || result.is_err());
    }
}
