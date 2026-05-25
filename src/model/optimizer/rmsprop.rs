use ndarray::{Array1, Array2, azip};

use crate::model::OptimizingStrategy;
use crate::model::{ModelError, activator::Activation, optimizer::BaseOptimizer};

#[derive(Debug, Clone)]
pub struct RMSProp {
    pub base: BaseOptimizer,

    pub beta: f64,
    pub epsilon: f64,

    squared_grad: Array1<f64>,
    s_bias: f64,
}

impl RMSProp {
    pub fn new(
        learning_rate: f64,
        max_epochs: usize,
        tolerance: f64,
        beta: f64,
        epsilon: f64,
        activation: Activation,
    ) -> Self {
        Self {
            base: BaseOptimizer::new(learning_rate, max_epochs, tolerance, activation),
            beta: beta.clamp(0.0, 1.0),
            epsilon: epsilon.max(1e-8),

            squared_grad: Array1::zeros(0),
            s_bias: 0.0,
        }
    }
}

impl OptimizingStrategy for RMSProp {
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

        self.base.weights = Array1::zeros(n_features);
        self.squared_grad = Array1::zeros(n_features);
        self.base.y_pred_buffer = Array1::zeros(n_samples);
        self.base.error_buffer = Array1::zeros(n_samples);
        self.base.dw_buffer = Array1::zeros(n_features);

        let n_samples_f = n_samples as f64;

        for epoch in 0..self.base.max_epochs {
            ndarray::linalg::general_mat_vec_mul(
                1.0,
                x_train,
                &self.base.weights,
                0.0,
                &mut self.base.y_pred_buffer,
            );
            self.base
                .activation
                .apply_inplace(&mut self.base.y_pred_buffer);

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

            azip!((sg in &mut self.squared_grad, &dw in &self.base.dw_buffer) {
                *sg = self.beta * *sg + (1.0 - self.beta) * dw * dw;
            });
            self.s_bias = self.beta * self.s_bias + (1.0 - self.beta) * db * db;

            azip!((w in &mut self.base.weights, &sg in &self.squared_grad, &dw in &self.base.dw_buffer) {
                *w -= self.base.learning_rate * dw / (sg.sqrt() + self.epsilon);
            });
            self.base.bias -= self.base.learning_rate * db / (self.s_bias.sqrt() + self.epsilon);

            if epoch == self.base.max_epochs - 1 {
                ModelError::print_warning(format!(
                    "RMSProp did not converge after {} iterations.",
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
        let mut y_predicted = Array1::zeros(x_test.nrows());
        ndarray::linalg::general_mat_vec_mul(
            1.0,
            x_test,
            &self.base.weights,
            0.0,
            &mut y_predicted,
        );
        self.base.activation.apply_inplace(&mut y_predicted);
        Ok(y_predicted)
    }

    fn is_fitted(&self) -> bool {
        self.base.fitted
    }
}
