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
}
