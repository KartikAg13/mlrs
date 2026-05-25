use super::ActivationStrategy;
use ndarray::Array1;

pub struct Identity;

impl ActivationStrategy for Identity {
    fn apply(&self, array: &Array1<f64>) -> Array1<f64> {
        array.clone()
    }

    fn derivative(&self, array: &Array1<f64>) -> Array1<f64> {
        Array1::ones(array.len())
    }

    fn name(&self) -> &str {
        "identity"
    }
}
