// src/model/optimizer/rmsprop.rs
use super::Optimizer;
use crate::model::{ModelError, activator::Activation};
use ndarray::{Array1, Array2, azip};

#[derive(Debug, Clone)]
pub struct RMSProp {
    pub weights: Array1<f64>,
    pub bias: f64,
    pub learning_rate: f64,
    pub max_epochs: usize,
    pub tolerance: f64,
    pub beta: f64,
    pub epsilon: f64,
    pub activation: Activation,
    pub fitted: bool,

    y_pred_buffer: Array1<f64>,
    error_buffer: Array1<f64>,
    dw_buffer: Array1<f64>,
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
            weights: Array1::zeros(0),
            bias: 0.0,
            learning_rate,
            max_epochs,
            tolerance,
            beta: beta.clamp(0.0, 1.0),
            epsilon: epsilon.max(1e-8),
            activation,
            fitted: false,
            y_pred_buffer: Array1::zeros(0),
            error_buffer: Array1::zeros(0),
            dw_buffer: Array1::zeros(0),
            squared_grad: Array1::zeros(0),
            s_bias: 0.0,
        }
    }
}

impl Optimizer for RMSProp {
    fn fit(&mut self, x_train: &Array2<f64>, y_train: &Array1<f64>) -> Result<(), ModelError> {
        let (n_samples, n_features) = x_train.dim();
        if n_samples != y_train.len() {
            let error = ModelError::ShapeMismatch(n_samples, y_train.len());
            error.print_error();
            return Err(error);
        }

        self.weights = Array1::zeros(n_features);
        self.squared_grad = Array1::zeros(n_features);
        self.y_pred_buffer = Array1::zeros(n_samples);
        self.error_buffer = Array1::zeros(n_samples);
        self.dw_buffer = Array1::zeros(n_features);

        let n_samples_f = n_samples as f64;

        for epoch in 0..self.max_epochs {
            ndarray::linalg::general_mat_vec_mul(
                1.0,
                x_train,
                &self.weights,
                0.0,
                &mut self.y_pred_buffer,
            );
            self.activation.apply_inplace(&mut self.y_pred_buffer);

            azip!((error in &mut self.error_buffer, &pred in &self.y_pred_buffer, &actual in y_train) {
                *error = pred - actual;
            });

            let mae = self.error_buffer.mapv(f64::abs).mean().unwrap_or(0.0);
            if mae < self.tolerance {
                break;
            }

            ndarray::linalg::general_mat_vec_mul(
                1.0 / n_samples_f,
                &x_train.t(),
                &self.error_buffer,
                0.0,
                &mut self.dw_buffer,
            );

            let db = self.error_buffer.sum() / n_samples_f;

            azip!((sg in &mut self.squared_grad, &dw in &self.dw_buffer) {
                *sg = self.beta * *sg + (1.0 - self.beta) * dw * dw;
            });
            self.s_bias = self.beta * self.s_bias + (1.0 - self.beta) * db * db;

            azip!((w in &mut self.weights, &sg in &self.squared_grad, &dw in &self.dw_buffer) {
                *w -= self.learning_rate * dw / (sg.sqrt() + self.epsilon);
            });
            self.bias -= self.learning_rate * db / (self.s_bias.sqrt() + self.epsilon);

            if epoch == self.max_epochs - 1 {
                ModelError::print_warning(format!(
                    "RMSProp did not converge after {} iterations.",
                    self.max_epochs
                ));
            }
        }

        self.fitted = true;
        Ok(())
    }

    fn predict(&mut self, x_test: &Array2<f64>) -> Result<Array1<f64>, ModelError> {
        if !self.fitted {
            let error = ModelError::NotFitted;
            error.print_error();
            return Err(error);
        }
        let mut y_predicted = Array1::zeros(x_test.nrows());
        ndarray::linalg::general_mat_vec_mul(1.0, x_test, &self.weights, 0.0, &mut y_predicted);
        self.activation.apply_inplace(&mut y_predicted);
        Ok(y_predicted)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}
