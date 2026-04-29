use ndarray::Array1;

use crate::constants::{
    DEFAULT_ALPHA, DEFAULT_L1_RATIO, DEFAULT_L2_RATIO, DEFAULT_LEARNING_RATE, DEFAULT_MAX_EPOCHS,
    DEFAULT_TOLERANCE,
};
use crate::model::activator::Activation;
use crate::model::optimizer::Optimizer;
use crate::model::optimizer::gradient_descent::GradientDescent;
use crate::model::{ModelError, ModelHandler, ModelingStrategy};
use crate::score::accuracy;

#[derive(Debug)]
pub struct LogisticConfig {
    pub threshold: f64,
}

impl Default for LogisticConfig {
    fn default() -> Self {
        Self { threshold: 0.5 }
    }
}

pub type LogisticRegressor = ModelHandler<LogisticConfig>;

impl LogisticRegressor {
    pub fn new() -> Self {
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
            config: LogisticConfig::default(),
        }
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        let t = threshold.clamp(0.0, 1.0);
        if (self.config.threshold - t).abs() > f64::EPSILON {
            ModelError::print_modifying(format!(
                "Changing threshold from {} to {}!",
                self.config.threshold, t
            ));
        }
        self.config.threshold = t;
        self
    }

    pub fn predict_proba(
        &mut self,
        x_test: &ndarray::Array2<f64>,
    ) -> Result<Array1<f64>, ModelError> {
        self.solver.predict(x_test)
    }
}

impl Default for LogisticRegressor {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelingStrategy for LogisticConfig {
    fn activation(&self) -> &Activation {
        &Activation::Sigmoid
    }

    fn score(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        accuracy(y_true, y_pred)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelError;
    use ndarray::{Array2, array};

    fn make_classification_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec((6, 1), vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]).unwrap();
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        (x, y)
    }

    #[test]
    fn test_fit_predict_hard_labels() {
        let (x, y) = make_classification_data();
        let mut model = LogisticRegressor::new()
            .with_learning_rate(0.1)
            .with_max_epochs(2000);
        assert!(model.fit(&x, &y).is_ok());
        let preds = model.predict(&x).unwrap();
        // all predictions should be exactly 0.0 or 1.0
        for p in preds.iter() {
            assert!(*p == 0.0 || *p == 1.0);
        }
    }

    #[test]
    fn test_predict_proba_range() {
        let (x, y) = make_classification_data();
        let mut model = LogisticRegressor::new()
            .with_learning_rate(0.1)
            .with_max_epochs(2000);
        model.fit(&x, &y).unwrap();
        let probs = model.predict_proba(&x).unwrap();
        for p in probs.iter() {
            assert!(*p >= 0.0 && *p <= 1.0);
        }
    }

    #[test]
    fn test_evaluate_uses_accuracy() {
        let (x, y) = make_classification_data();
        let mut model = LogisticRegressor::new()
            .with_learning_rate(0.1)
            .with_max_epochs(2000);
        model.fit(&x, &y).unwrap();
        let acc = model.evaluate(&Some(x), &y);
        assert!(acc >= 0.0 && acc <= 1.0);
        assert!(acc > 0.8);
    }

    #[test]
    fn test_with_threshold() {
        let (x, y) = make_classification_data();
        let mut model = LogisticRegressor::new()
            .with_learning_rate(0.1)
            .with_max_epochs(2000)
            .with_threshold(0.3);
        assert!((model.config.threshold - 0.3).abs() < f64::EPSILON);
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        for p in preds.iter() {
            assert!(*p == 0.0 || *p == 1.0);
        }
    }

    #[test]
    fn test_not_fitted() {
        let mut model = LogisticRegressor::new();
        let x = Array2::zeros((3, 1));
        assert!(matches!(model.predict(&x), Err(ModelError::NotFitted)));
    }
}
