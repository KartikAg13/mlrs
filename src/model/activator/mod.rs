use ndarray::Array1;

pub mod identity;
pub mod relu;
pub mod sigmoid;
pub mod softmax;
pub mod tanh;

pub use identity::Identity;
pub use relu::ReLU;
pub use sigmoid::Sigmoid;
pub use softmax::Softmax;
pub use tanh::Tanh;

pub trait ActivationStrategy: Send + Sync {
    fn apply(&self, z: &Array1<f64>) -> Array1<f64>;
    fn derivative(&self, z: &Array1<f64>) -> Array1<f64>;
    fn name(&self) -> &str;
}
