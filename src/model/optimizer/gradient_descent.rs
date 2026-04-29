use crate::model::activator::Activation;
use crate::model::{ModelError, optimizer::Optimizer};
use ndarray::{Array1, Array2, Zip, azip};

#[derive(Debug, Clone)]
pub struct GradientDescent {
    pub weights: Array1<f64>,
    pub bias: f64,
    pub learning_rate: f64,
    pub max_epochs: usize,
    pub tolerance: f64,
    pub l1_ratio: f64,
    pub l2_ratio: f64,
    pub alpha: f64,
    pub activation: Activation,
    pub fitted: bool,

    y_pred_buffer: Array1<f64>,
    error_buffer: Array1<f64>,
    dw_buffer: Array1<f64>,

    pub y_predicted: Array1<f64>,
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
            weights: Array1::zeros(0),
            bias: 0.0,
            learning_rate,
            max_epochs,
            tolerance,
            l1_ratio: l1_ratio.clamp(0.0, 1.0),
            l2_ratio,
            alpha,
            activation,
            fitted: false,
            y_pred_buffer: Array1::zeros(0),
            error_buffer: Array1::zeros(0),
            dw_buffer: Array1::zeros(0),
            y_predicted: Array1::zeros(0),
        }
    }
}

impl Optimizer for GradientDescent {
    fn fit(&mut self, x_train: &Array2<f64>, y_train: &Array1<f64>) -> Result<(), ModelError> {
        let (n_samples, n_features) = x_train.dim();

        if n_samples != y_train.len() {
            let error = ModelError::ShapeMismatch(n_samples, y_train.len());
            error.print_error();
            return Err(error);
        }

        self.weights = Array1::<f64>::zeros(n_features);

        let n_samples_f = n_samples as f64;

        self.y_pred_buffer = Array1::zeros(n_samples);
        self.error_buffer = Array1::zeros(n_samples);
        self.dw_buffer = Array1::zeros(n_features);

        let l1_penalty: f64 = self.l1_ratio * self.alpha;
        let l2_penalty: f64 = (1.0 - self.l1_ratio) * self.alpha;
        let l1_threshold: f64 = self.learning_rate * l1_penalty;

        for epoch in 0..self.max_epochs {
            ndarray::linalg::general_mat_vec_mul(
                1.0,
                x_train,
                &self.weights,
                0.0,
                &mut self.y_pred_buffer,
            );

            self.y_pred_buffer
                .mapv_inplace(|prediction| self.activation.apply(prediction + self.bias));

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

            Zip::from(self.weights.view_mut())
                .and(&self.dw_buffer)
                .for_each(|w, &dw| {
                    *w -= self.learning_rate * (dw + l2_penalty * *w);
                });
            self.bias -= self.learning_rate * db;

            if l1_threshold > 0.0 {
                self.weights.mapv_inplace(|w| {
                    if w > l1_threshold {
                        w - l1_threshold
                    } else if w < -l1_threshold {
                        w + l1_threshold
                    } else {
                        0.0
                    }
                });
            }

            if epoch == self.max_epochs - 1 {
                ModelError::print_warning(format!(
                    "Model did not converge after {} iterations.",
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

        let (n_features, n_columns) = x_test.dim();
        if n_columns != self.weights.len() {
            let error = ModelError::ShapeMismatch(n_columns, self.weights.len());
            error.print_error();
            return Err(error);
        }

        let mut y_predicted = Array1::<f64>::zeros(n_features);
        ndarray::linalg::general_mat_vec_mul(1.0, x_test, &self.weights, 0.0, &mut y_predicted);
        y_predicted.mapv_inplace(|prediction| self.activation.apply(prediction + self.bias));
        self.y_predicted = y_predicted.clone();
        Ok(y_predicted)
    }

    fn is_fitted(&self) -> bool {
        self.fitted
    }
}
