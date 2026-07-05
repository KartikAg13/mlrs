use crate::model::activator::Identity;
use crate::model::layer::Layer;
use crate::model::metric::mse::MSE;
use crate::model::optimizer::GradientDescent;
use crate::model::sequential::SequentialModel;

pub type LinearRegressor = SequentialModel<GradientDescent>;

pub fn new() -> LinearRegressor {
    SequentialModel::new(
        vec![Layer::new(Box::new(Identity))],
        GradientDescent::new(),
        Box::new(MSE),
    )
}

impl SequentialModel<GradientDescent> {
    pub fn with_l1_ratio(mut self, l1_ratio: f64) -> Self {
        self.optimizer.l1_ratio = l1_ratio;
        self
    }

    pub fn with_l2_ratio(mut self, l2_ratio: f64) -> Self {
        self.optimizer.l2_ratio = l2_ratio;
        self
    }

    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.optimizer.alpha = alpha;
        self
    }
}
