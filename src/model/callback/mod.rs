use std::any::Any;

pub mod checkpoint;
pub mod early_stopping;
pub mod lr_reduce;

#[derive(Debug, Clone, Copy)]
pub enum Monitor {
    TrainLoss,
    ValLoss,
    MetricScore,
}

pub struct CallbackContext {
    pub epoch: usize,
    pub max_epochs: usize,
    pub train_loss: f64,
    pub val_loss: Option<f64>,
    pub metric_score: f64,
    pub learning_rate: f64,
    pub stop_training: bool,
    pub should_checkpoint: bool,
}

impl CallbackContext {
    pub fn monitored_value(&self, monitor: Monitor) -> Option<f64> {
        match monitor {
            Monitor::TrainLoss => Some(self.train_loss),
            Monitor::ValLoss => self.val_loss,
            Monitor::MetricScore => Some(self.metric_score),
        }
    }
}

pub trait Callback {
    fn on_train_begin(&mut self, _context: &mut CallbackContext) {}
    fn on_train_end(&mut self, _context: &mut CallbackContext) {}
    fn on_epoch_begin(&mut self, _context: &mut CallbackContext) {}
    fn on_epoch_end(&mut self, _context: &mut CallbackContext) {}
    fn on_predict_begin(&mut self, _context: &mut CallbackContext) {}
    fn on_predict_end(&mut self, _context: &mut CallbackContext) {}

    fn as_any(&self) -> &dyn Any;
}
