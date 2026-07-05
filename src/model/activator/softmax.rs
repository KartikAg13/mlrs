use super::ActivationStrategy;
use ndarray::Array1;

pub struct Softmax;

impl ActivationStrategy for Softmax {
    fn apply(&self, z: &Array1<f64>) -> Array1<f64> {
        let max = z.fold(f64::NEG_INFINITY, |a, &b| a.max(b)); // numerical stability
        let exp = z.mapv(|x| (x - max).exp());
        let sum = exp.sum();
        exp / sum
    }

    fn derivative(&self, z: &Array1<f64>) -> Array1<f64> {
        Array1::ones(z.len())
    }

    fn name(&self) -> &str {
        "softmax"
    }
}
