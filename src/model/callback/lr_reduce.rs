use super::{Callback, CallbackContext, Monitor};
use crate::model::error::ModelError;

pub struct LRReduceOnPlateau {
    pub monitor: Monitor,
    pub factor: f64,
    pub patience: usize,
    pub min_lr: f64,
    pub min_delta: f64,

    best: f64,
    wait: usize,
}

impl LRReduceOnPlateau {
    pub fn new(
        monitor: Monitor,
        factor: f64,
        patience: usize,
        min_lr: f64,
        min_delta: f64,
    ) -> Self {
        Self {
            monitor,
            factor,
            patience,
            min_lr,
            min_delta,
            best: f64::INFINITY,
            wait: 0,
        }
    }
}

impl Callback for LRReduceOnPlateau {
    fn on_train_begin(&mut self, _context: &mut CallbackContext) {
        self.best = f64::INFINITY;
        self.wait = 0;
    }

    fn on_epoch_end(&mut self, context: &mut CallbackContext) {
        let value = match context.monitored_value(self.monitor) {
            Some(v) => v,
            None => {
                ModelError::print_warning(
                    "LRReduceOnPlateau monitored value unavailable. ValLoss needs validation data. Using monitor MetricScore.".to_string()
                );
                self.monitor = Monitor::MetricScore;
                context.metric_score
            }
        };

        if value < self.best - self.min_delta {
            self.best = value;
            self.wait = 0;
        } else if context.epoch >= context.max_epochs {
            ModelError::print_warning(
                "LRReduceOnPlateau could not trigger. Try lowering patience and/or min_delta."
                    .to_string(),
            );
        } else {
            self.wait += 1;
            if self.wait >= self.patience {
                let new_lr = (context.learning_rate * self.factor).max(self.min_lr);
                if (new_lr - context.learning_rate).abs() > f64::EPSILON {
                    ModelError::print_warning(format!(
                        "LRReduceOnPlateau reduced learning rate from {} to {} at epoch {}.",
                        context.learning_rate, new_lr, context.epoch
                    ));
                    context.learning_rate = new_lr;
                }
                self.wait = 0;
            }
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
