use crate::model::activator::{Sigmoid, Softmax};
use crate::model::layer::Layer;
use crate::model::metric::accuracy::Accuracy;
use crate::model::optimizer::GradientDescent;
use crate::model::sequential::SequentialModel;

pub type LogisticRegressor = SequentialModel<GradientDescent>;

/// Binary classification (sigmoid + 1 output)
pub fn new() -> LogisticRegressor {
    SequentialModel::new(
        vec![Layer::new(Box::new(Sigmoid))],
        GradientDescent::new(),
        Box::new(Accuracy::default()),
    )
}

/// Multiclass classification (softmax + n_classes outputs)
pub fn new_multiclass(n_classes: usize) -> LogisticRegressor {
    SequentialModel::new(
        vec![Layer::new(Box::new(Softmax))],
        GradientDescent::new(),
        Box::new(Accuracy::default()),
    )
    .with_n_classes(n_classes)
}
