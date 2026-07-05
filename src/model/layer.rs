use ndarray::{Array1, Array2};
use rand::rng;
use rand_distr::{Distribution, Normal};

use crate::model::{activator::ActivationStrategy, error::ModelError};

pub struct Layer {
    pub weights: Array2<f64>,
    pub bias: Array1<f64>,
    pub activation: Box<dyn ActivationStrategy>,
    initialized: bool,
}

impl Layer {
    pub fn new(activation: Box<dyn ActivationStrategy>) -> Self {
        Self {
            weights: Array2::zeros((0, 0)),
            bias: Array1::zeros(0),
            activation,
            initialized: false,
        }
    }

    pub fn initialize(&mut self, input_size: usize, output_size: usize) {
        if self.initialized {
            return;
        }
        let standard_deviation = match self.activation.name() {
            "relu" => (2.0 / input_size as f64).sqrt(),
            _ => (1.0 / input_size as f64).sqrt(),
        };
        let distribution = match Normal::new(0.0, standard_deviation) {
            Ok(d) => d,
            Err(_) => {
                let error = ModelError::InvalidNormalDistribution(input_size);
                error.print_error();
                return;
            }
        };

        let mut rng = rng();
        let data: Vec<f64> = (0..input_size * output_size)
            .map(|_| distribution.sample(&mut rng))
            .collect();

        self.weights = match Array2::from_shape_vec((input_size, output_size), data) {
            Ok(w) => w,
            Err(_) => {
                let error = ModelError::UnexpectedError;
                error.print_error();
                return;
            }
        };

        self.bias = Array1::zeros(output_size);
        self.initialized = true;
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    pub fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        input.dot(&self.weights) + &self.bias
    }
}
