use ndarray::{Array1, Array2};

use crate::model::Activation;
use crate::model::ModelError;

pub mod gradient_descent;
pub use gradient_descent::GradientDescent;

pub mod momentum;
pub use momentum::Momentum;

pub mod rmsprop;
pub use rmsprop::RMSProp;

pub mod adam;
pub use adam::Adam;

#[derive(Debug, Clone)]
pub struct BaseOptimizer {
    pub weights: Array1<f64>,
    pub bias: f64,
    pub learning_rate: f64,
    pub max_epochs: usize,
    pub tolerance: f64,
    pub activation: Activation,
    pub fitted: bool,

    pub y_predicted: Array1<f64>,

    y_pred_buffer: Array1<f64>,
    error_buffer: Array1<f64>,
    dw_buffer: Array1<f64>,
}

impl BaseOptimizer {
    pub fn new(
        learning_rate: f64,
        max_epochs: usize,
        tolerance: f64,
        activation: Activation,
    ) -> Self {
        Self {
            weights: Array1::zeros(0),
            bias: 0.0,
            learning_rate,
            max_epochs,
            tolerance,
            activation,
            fitted: false,

            y_predicted: Array1::zeros(0),

            y_pred_buffer: Array1::zeros(0),
            error_buffer: Array1::zeros(0),
            dw_buffer: Array1::zeros(0),
        }
    }
}

pub trait OptimizingStrategy {
    fn base(&self) -> &BaseOptimizer;
    fn base_mut(&mut self) -> &mut BaseOptimizer;

    fn fit(&mut self, x_train: &Array2<f64>, y_train: &Array1<f64>) -> Result<(), ModelError>;
    fn predict(&mut self, x_test: &Array2<f64>) -> Result<Array1<f64>, ModelError>;
    fn is_fitted(&self) -> bool;
}

impl OptimizingStrategy for Box<dyn OptimizingStrategy> {
    fn base(&self) -> &BaseOptimizer {
        (**self).base()
    }
    fn base_mut(&mut self) -> &mut BaseOptimizer {
        (**self).base_mut()
    }
    fn fit(&mut self, x_train: &Array2<f64>, y_train: &Array1<f64>) -> Result<(), ModelError> {
        (**self).fit(x_train, y_train)
    }
    fn predict(&mut self, x_test: &Array2<f64>) -> Result<Array1<f64>, ModelError> {
        (**self).predict(x_test)
    }
    fn is_fitted(&self) -> bool {
        (**self).is_fitted()
    }
}
