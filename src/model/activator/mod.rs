use ndarray::Array1;

#[derive(Debug, Clone)]
pub enum Activation {
    Identity,
    Sigmoid,
}

impl Activation {
    pub fn apply(&self, value: f64) -> f64 {
        match self {
            Activation::Identity => value,
            Activation::Sigmoid => {
                if value >= 0.0 {
                    1.0 / (1.0 + (-value).exp())
                } else {
                    let ex = value.exp();
                    ex / (1.0 + ex)
                }
            }
        }
    }

    pub fn apply_inplace(&self, arr: &mut Array1<f64>) {
        match self {
            Activation::Identity => {}
            Activation::Sigmoid => {
                arr.mapv_inplace(|x| {
                    if x >= 0.0 {
                        1.0 / (1.0 + (-x).exp())
                    } else {
                        let ex = x.exp();
                        ex / (1.0 + ex)
                    }
                });
            }
        }
    }
}
