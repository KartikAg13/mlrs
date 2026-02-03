use crate::dataset::Dataset;

pub trait Train {
    fn train(&mut self, data: &Dataset, epochs: u32, learning_rate: f64);
}
