use ndarray::{Array1, Array2, azip};

use crate::model::OptimizingStrategy;
use crate::model::optimizer::BaseOptimizer;
use crate::model::{ModelError, activator::ActivationStrategy};

#[derive(Debug, Clone)]
pub struct Adam {
    pub base: BaseOptimizer,

    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,

    m: Array1<f64>,
    v: Array1<f64>,
    m_bias: f64,
    v_bias: f64,
    t: usize,
}

impl Adam {
    pub fn new(
        learning_rate: f64,
        max_epochs: usize,
        tolerance: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        activation: ActivationStrategy,
    ) -> Self {
        Self {
            base: BaseOptimizer::new(learning_rate, max_epochs, tolerance, activation),
            beta1: beta1.clamp(0.0, 1.0),
            beta2: beta2.clamp(0.0, 1.0),
            epsilon: epsilon.max(1e-8),

            m: Array1::zeros(0),
            v: Array1::zeros(0),
            m_bias: 0.0,
            v_bias: 0.0,
            t: 0,
        }
    }
}

impl OptimizingStrategy for Adam {
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
        self.m = Array1::zeros(n_features);
        self.v = Array1::zeros(n_features);
        self.base.y_pred_buffer = Array1::zeros(n_samples);
        self.base.error_buffer = Array1::zeros(n_samples);
        self.base.dw_buffer = Array1::zeros(n_features);
        self.t = 0;

        let n_samples_f = n_samples as f64;

        for epoch in 0..self.base.max_epochs {
            self.t += 1;

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

            azip!((m in &mut self.m, v in &mut self.v, &dw in &self.base.dw_buffer) {
                *m = self.beta1 * *m + (1.0 - self.beta1) * dw;
                *v = self.beta2 * *v + (1.0 - self.beta2) * dw * dw;
            });
            self.m_bias = self.beta1 * self.m_bias + (1.0 - self.beta1) * db;
            self.v_bias = self.beta2 * self.v_bias + (1.0 - self.beta2) * db * db;

            let m_hat = self.m.mapv(|m| m / (1.0 - self.beta1.powi(self.t as i32)));
            let v_hat = self.v.mapv(|v| v / (1.0 - self.beta2.powi(self.t as i32)));
            let m_bias_hat = self.m_bias / (1.0 - self.beta1.powi(self.t as i32));
            let v_bias_hat = self.v_bias / (1.0 - self.beta2.powi(self.t as i32));

            azip!((w in &mut self.base.weights, &mh in &m_hat, &vh in &v_hat) {
                *w -= self.base.learning_rate * mh / (vh.sqrt() + self.epsilon);
            });
            self.base.bias -=
                self.base.learning_rate * m_bias_hat / (v_bias_hat.sqrt() + self.epsilon);

            if epoch == self.base.max_epochs - 1 {
                ModelError::print_warning(format!(
                    "Adam did not converge after {} iterations.",
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
        let mut y_pred = Array1::zeros(x_test.nrows());
        ndarray::linalg::general_mat_vec_mul(1.0, x_test, &self.base.weights, 0.0, &mut y_pred);
        self.base.activation.apply_inplace(&mut y_pred);
        Ok(y_pred)
    }

    fn is_fitted(&self) -> bool {
        self.base.fitted
    }
}
