use ndarray::{Array1, Array2};
use std::collections::HashMap;

use crate::model::layer::Layer;

pub mod adam;
pub mod gradient_descent;
pub mod momentum;
pub mod rmsprop;

pub use adam::Adam;
pub use gradient_descent::GradientDescent;
pub use momentum::Momentum;
pub use rmsprop::RMSProp;

pub trait OptimizingStrategy {
    fn step(&mut self, layers: &mut Vec<Layer>, gradients: &[(Array2<f64>, Array1<f64>)]);

    fn reset(&mut self);

    fn learning_rate(&self) -> f64;

    fn set_learning_rate(&mut self, learning_rate: f64);

    fn hyperparameters(&self) -> HashMap<String, f64>;
}
