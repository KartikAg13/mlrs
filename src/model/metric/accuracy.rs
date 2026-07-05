use ndarray::Array1;

use super::MetricStrategy;

pub struct Accuracy {
    pub threshold: f64,
}

impl Default for Accuracy {
    fn default() -> Self {
        Self { threshold: 0.5 }
    }
}

impl MetricStrategy for Accuracy {
    fn compute(&self, y_true: &Array1<f64>, y_predicted: &Array1<f64>) -> f64 {
        let correct = y_true
            .iter()
            .zip(y_predicted.iter())
            .filter(|&(t, p)| {
                let class = if p >= &self.threshold { 1.0 } else { 0.0 };
                (class - t).abs() < f64::EPSILON
            })
            .count();
        correct as f64 / y_true.len() as f64
    }

    fn name(&self) -> &str {
        "accuracy"
    }
}
