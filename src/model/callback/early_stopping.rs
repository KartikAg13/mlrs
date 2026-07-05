use crate::model::error::ModelError;

use super::{Callback, CallbackContext, Monitor};

pub struct EarlyStopping {
    pub monitor: Monitor,
    pub patience: usize,
    pub min_delta: f64,

    best: f64,
    wait: usize,
}

impl EarlyStopping {
    pub fn new(monitor: Monitor, patience: usize, min_delta: f64) -> Self {
        Self {
            monitor,
            patience,
            min_delta,
            best: f64::INFINITY,
            wait: 0,
        }
    }

    pub fn with_patience(mut self, patience: usize) -> Self {
        if self.patience != patience {
            ModelError::print_modifying(format!("Patience from {} to {}", self.patience, patience));
            self.patience = patience;
        }
        self
    }

    pub fn with_min_delta(mut self, min_delta: f64) -> Self {
        if self.min_delta != min_delta {
            ModelError::print_modifying(format!(
                "Min delta from {} to {}",
                self.min_delta, min_delta
            ));
            self.min_delta = min_delta;
        }
        self
    }
}

impl Callback for EarlyStopping {
    fn on_train_begin(&mut self, _context: &mut CallbackContext) {
        self.best = f64::INFINITY;
        self.wait = 0;
    }

    fn on_epoch_end(&mut self, context: &mut CallbackContext) {
        let value = match context.monitored_value(self.monitor) {
            Some(v) => v,
            None => {
                ModelError::print_warning(
                    "EarlyStopping monitored value unavailable. ValLoss needs validation data. Using monitor MetricScore.".to_string()
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
                "EarlyStopping could not trigger. Try lowering patience and/or min_delta."
                    .to_string(),
            );
        } else {
            self.wait += 1;
            if self.wait >= self.patience {
                ModelError::print_warning(format!(
                    "EarlyStopping triggered at epoch {}. Best value {}",
                    context.epoch, self.best
                ));
                context.stop_training = true;
            }
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
