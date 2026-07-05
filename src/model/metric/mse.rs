use ndarray::Array1;

use super::MetricStrategy;

pub struct MSE;

impl MetricStrategy for MSE {
    fn compute(&self, y_true: &Array1<f64>, y_predicted: &Array1<f64>) -> f64 {
        let difference = y_true - y_predicted;
        (&difference * &difference).mean().unwrap_or(f64::NAN)
    }

    fn name(&self) -> &str {
        "mse"
    }
}
