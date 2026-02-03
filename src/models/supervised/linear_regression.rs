use crate::dataset::Dataset;
use crate::training::Train;

pub struct LinearRegression {
    pub weight: f64,
    pub bias: f64,
}

impl LinearRegression {
    pub fn new() -> Self {
        Self {
            weight: 0.0,
            bias: 0.0,
        }
    }

    pub fn predict(&self, x: f64) -> f64 {
        self.weight * x + self.bias
    }
}

impl Train for LinearRegression {
    fn train(&mut self, data: &Dataset, epochs: u32, learning_rate: f64) {
        let size = data.features.len() as f64;

        for _ in 0..epochs {
            let mut weight_gradient: f64 = 0.0;
            let mut bias_gradient: f64 = 0.0;

            for i in 0..data.features.len() {
                let x = data.features[[i, 0]];
                let y = data.target[i];
                let prediction = self.predict(x);

                let error = prediction - y;
                weight_gradient += error * x;
                bias_gradient += error;
            }
            self.weight -= learning_rate * (weight_gradient / size);
            self.bias -= learning_rate * (bias_gradient / size);
        }
    }
}
