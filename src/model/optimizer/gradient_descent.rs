use ndarray::{Array1, Array2, Zip, azip};

use crate::model::activator::Activation;
use crate::model::optimizer::BaseOptimizer;
use crate::model::{ModelError, optimizer::OptimizingStrategy};

#[derive(Debug, Clone)]
pub struct GradientDescent {
    pub base: BaseOptimizer,

    pub l1_ratio: f64,
    pub l2_ratio: f64,
    pub alpha: f64,
}

impl GradientDescent {
    pub fn new(
        learning_rate: f64,
        max_epochs: usize,
        tolerance: f64,
        l1_ratio: f64,
        l2_ratio: f64,
        alpha: f64,
        activation: Activation,
    ) -> Self {
        Self {
            base: BaseOptimizer::new(learning_rate, max_epochs, tolerance, activation),
            l1_ratio: l1_ratio.clamp(0.0, 1.0),
            l2_ratio,
            alpha,
        }
    }

    pub fn with_l1_ratio(mut self, l1_ratio: f64) -> Self {
        let l1 = l1_ratio.clamp(0.0, 1.0);
        if (l1 + self.l2_ratio) > 1.0 {
            ModelError::print_warning(format!("Sum of l1 and l2 ratio is greater than {}.", 1.0));
        }
        if self.l1_ratio != l1 {
            ModelError::print_modifying(format!(
                "Changing l1 ratio from {} to {}!",
                self.l1_ratio, l1
            ));
        }
        self.l1_ratio = l1;
        self
    }

    pub fn with_l2_ratio(mut self, l2_ratio: f64) -> Self {
        let l2 = l2_ratio.clamp(0.0, 1.0);
        if (l2 + self.l1_ratio) > 1.0 {
            ModelError::print_warning(format!("Sum of l1 and l2 ratio is greater than {}.", 1.0));
        }
        if self.l2_ratio != l2 {
            ModelError::print_modifying(format!(
                "Changing l2 ratio from {} to {}!",
                self.l2_ratio, l2
            ));
        }
        self.l2_ratio = l2;
        self
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        let al = alpha.clamp(0.0, 1.0);
        if self.alpha != al {
            ModelError::print_modifying(format!("Changing alpha from {} to {}!", self.alpha, al));
        }
        self.alpha = al;
        self
    }
}

impl OptimizingStrategy for GradientDescent {
    fn base(&self) -> &BaseOptimizer {
        &self.base
    }

    fn base_mut(&mut self) -> &mut BaseOptimizer {
        &mut self.base
    }

    fn fit(&mut self, x_train: &Array2<f64>, y_train: &Array1<f64>) -> Result<(), ModelError> {
        let (n_samples, n_features) = x_train.dim();

        if n_samples != y_train.len() {
            let error = ModelError::ShapeMismatch(n_samples, y_train.len());
            error.print_error();
            return Err(error);
        }

        self.base.weights = Array1::<f64>::zeros(n_features);

        let n_samples_f = n_samples as f64;

        self.base.y_pred_buffer = Array1::zeros(n_samples);
        self.base.error_buffer = Array1::zeros(n_samples);
        self.base.dw_buffer = Array1::zeros(n_features);

        let l1_penalty: f64 = self.l1_ratio * self.alpha;
        let l2_penalty: f64 = (1.0 - self.l1_ratio) * self.alpha;
        let l1_threshold: f64 = self.base.learning_rate * l1_penalty;

        for epoch in 0..self.base.max_epochs {
            ndarray::linalg::general_mat_vec_mul(
                1.0,
                x_train,
                &self.base.weights,
                0.0,
                &mut self.base.y_pred_buffer,
            );

            self.base
                .y_pred_buffer
                .mapv_inplace(|prediction| self.base.activation.apply(prediction + self.base.bias));

            azip!((error in &mut self.base.error_buffer, &pred in &self.base.y_pred_buffer, &actual in y_train) {
                *error = pred - actual;
            });

            let mae = self.base.error_buffer.mapv(f64::abs).mean().unwrap_or(0.0);
            if mae < self.base.tolerance {
                break;
            }

            ndarray::linalg::general_mat_vec_mul(
                1.0 / n_samples_f,
                &x_train.t(),
                &self.base.error_buffer,
                0.0,
                &mut self.base.dw_buffer,
            );

            let db = self.base.error_buffer.sum() / n_samples_f;

            Zip::from(self.base.weights.view_mut())
                .and(&self.base.dw_buffer)
                .for_each(|w, &dw| {
                    *w -= self.base.learning_rate * (dw + l2_penalty * *w);
                });
            self.base.bias -= self.base.learning_rate * db;

            if l1_threshold > 0.0 {
                self.base.weights.mapv_inplace(|w| {
                    if w > l1_threshold {
                        w - l1_threshold
                    } else if w < -l1_threshold {
                        w + l1_threshold
                    } else {
                        0.0
                    }
                });
            }

            if epoch == self.base.max_epochs - 1 {
                ModelError::print_warning(format!(
                    "Model did not converge after {} iterations.",
                    self.base.max_epochs
                ));
            }
        }
        self.base.fitted = true;
        Ok(())
    }

    fn predict(&mut self, x_test: &Array2<f64>) -> Result<Array1<f64>, ModelError> {
        if !self.base.fitted {
            let error = ModelError::NotFitted;
            error.print_error();
            return Err(error);
        }

        let (n_features, n_columns) = x_test.dim();
        if n_columns != self.base.weights.len() {
            let error = ModelError::ShapeMismatch(n_columns, self.base.weights.len());
            error.print_error();
            return Err(error);
        }

        let mut y_predicted = Array1::<f64>::zeros(n_features);
        ndarray::linalg::general_mat_vec_mul(
            1.0,
            x_test,
            &self.base.weights,
            0.0,
            &mut y_predicted,
        );
        y_predicted
            .mapv_inplace(|prediction| self.base.activation.apply(prediction + self.base.bias));
        self.base.y_predicted = y_predicted.clone();
        Ok(y_predicted)
    }

    fn is_fitted(&self) -> bool {
        self.base.fitted
    }
}
