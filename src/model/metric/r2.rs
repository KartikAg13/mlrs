use ndarray::Array1;

use super::MetricStrategy;

pub struct R2Score;

impl MetricStrategy for R2Score {
    fn compute(&self, y_true: &Array1<f64>, y_predicted: &Array1<f64>) -> f64 {
        let mean = y_true.mean().unwrap_or(0.0);
        let ss_tot = y_true.mapv(|y| (y - mean).powi(2)).sum();
        if ss_tot == 0.0 {
            return ss_tot;
        }
        let ss_res = (y_true - y_predicted).mapv(|y| (y - mean).powi(2)).sum();
        1.0 - (ss_res / ss_tot)
    }

    fn name(&self) -> &str {
        "r2"
    }
}
