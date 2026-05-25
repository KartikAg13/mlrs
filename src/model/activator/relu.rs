use super::ActivationStrategy;
use ndarray::Array1;

pub struct ReLU;

impl ActivationStrategy for ReLU {
    fn apply(&self, array: &Array1<f64>) -> Array1<f64> {
        array.mapv(|x| x.max(0.0))
    }

    fn derivative(&self, array: &Array1<f64>) -> Array1<f64> {
        array.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }

    fn name(&self) -> &str {
        "relu"
    }
}
