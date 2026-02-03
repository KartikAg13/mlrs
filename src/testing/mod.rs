use ndarray::Array1;

pub trait Test {
    fn predict(&mut self, data: &Array1<f64>);
}
