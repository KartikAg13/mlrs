use ndarray::Array1;

pub mod accuracy;
pub mod logloss;
pub mod mse;
pub mod r2;

pub trait MetricStrategy: Send + Sync {
    fn compute(&self, y_true: &Array1<f64>, y_predicted: &Array1<f64>) -> f64;
    fn name(&self) -> &str;
}
