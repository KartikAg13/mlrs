use super::ActivationStrategy;
use ndarray::Array1;

pub struct Sigmoid;

impl ActivationStrategy for Sigmoid {
    fn apply(&self, array: &Array1<f64>) -> Array1<f64> {
        array.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }

    fn derivative(&self, array: &Array1<f64>) -> Array1<f64> {
        let s = self.apply(array);
        &s * &(1.0 - &s)
    }

    fn name(&self) -> &str {
        "sigmoid"
    }
}
