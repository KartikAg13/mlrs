use crate::model::ModelError;
use ndarray::{Array1, Array2};

pub mod gradient_descent;
pub use gradient_descent::GradientDescent;

pub mod momentum;
pub use momentum::Momentum;

pub mod rmsprop;
pub use rmsprop::RMSProp;

pub mod adam;
pub use adam::Adam;

pub trait Optimizer {
    fn fit(&mut self, x_train: &Array2<f64>, y_train: &Array1<f64>) -> Result<(), ModelError>;
    fn predict(&mut self, x_test: &Array2<f64>) -> Result<Array1<f64>, ModelError>;
    fn is_fitted(&self) -> bool;
}
