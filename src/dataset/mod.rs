use ndarray::{Array1, Array2};

pub struct Dataset {
    pub features: Array2<f64>,
    pub target: Array1<f64>,
}

impl Dataset {
    pub fn new(features: Array2<f64>, target: Array1<f64>) -> Self {
        Self { features, target }
    }
}
