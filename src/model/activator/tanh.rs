use super::ActivationStrategy;
use ndarray::Array1;

pub struct Tanh;

impl ActivationStrategy for Tanh {
    fn apply(&self, array: &Array1<f64>) -> Array1<f64> {
        array.mapv(|x| x.tanh())
    }

    fn derivative(&self, array: &Array1<f64>) -> Array1<f64> {
        array.mapv(|x| 1.0 - x.tanh().powi(2))
    }

    fn name(&self) -> &str {
        "tanh"
    }
}
