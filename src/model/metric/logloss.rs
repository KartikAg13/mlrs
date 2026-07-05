use ndarray::Array1;

use super::MetricStrategy;

pub struct LogLoss;

impl MetricStrategy for LogLoss {
    fn compute(&self, y_true: &Array1<f64>, y_predicted: &Array1<f64>) -> f64 {
        let loss = y_true
            .iter()
            .zip(y_predicted.iter())
            .map(|(&t, &p)| {
                let p = p.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
                -(t * p.ln() + (1.0 - t) * (1.0 - p).ln())
            })
            .sum::<f64>();
        loss / y_true.len() as f64
    }

    fn name(&self) -> &str {
        "logloss"
    }
}
