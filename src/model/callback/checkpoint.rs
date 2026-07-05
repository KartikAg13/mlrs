use ndarray::{Array1, Array2};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use super::{Callback, CallbackContext, Monitor};
use crate::model::error::ModelError;
use crate::model::layer::Layer;

pub struct ModelCheckpoint {
    pub path: PathBuf,
    pub monitor: Monitor,
    pub save_best_only: bool,

    best: f64,
}

impl ModelCheckpoint {
    pub fn new(path: PathBuf, monitor: Monitor, save_best_only: bool) -> Self {
        Self {
            path,
            monitor,
            save_best_only,
            best: f64::INFINITY,
        }
    }

    pub fn save(path: &PathBuf, layers: &[Layer]) -> std::io::Result<()> {
        fs::create_dir_all(path.parent().unwrap_or_else(|| Path::new(".")))?;
        let file = match File::create(path) {
            Ok(f) => f,
            Err(e) => {
                let error = ModelError::InvalidFilePath(path.to_str().unwrap().into());
                error.print_error();
                return Err(e);
            }
        };

        let mut writer = BufWriter::new(file);
        for layer in layers {
            let (rows, columns) = layer.weights.dim();
            writer.write_all(&(rows as u64).to_le_bytes())?;
            writer.write_all(&(columns as u64).to_le_bytes())?;
            for value in layer.weights.iter() {
                writer.write_all(&value.to_le_bytes())?;
            }

            let bias_len = layer.bias.len();
            writer.write_all(&(bias_len as u64).to_le_bytes())?;
            for value in layer.bias.iter() {
                writer.write_all(&value.to_le_bytes())?;
            }
        }

        Ok(())
    }

    pub fn load(path: &PathBuf, layers: &mut [Layer]) -> std::io::Result<()> {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) => {
                let error = ModelError::InvalidFilePath(path.to_str().unwrap().into());
                error.print_error();
                return Err(e);
            }
        };
        let mut reader = BufReader::new(file);

        let mut u64_buf = [0u8; 8];
        let mut f64_buf = [0u8; 8];

        for layer in layers.iter_mut() {
            reader.read_exact(&mut u64_buf)?;
            let rows = u64::from_le_bytes(u64_buf) as usize;
            reader.read_exact(&mut u64_buf)?;
            let cols = u64::from_le_bytes(u64_buf) as usize;

            let mut weights_data = vec![0.0f64; rows * cols];
            for value in weights_data.iter_mut() {
                reader.read_exact(&mut f64_buf)?;
                *value = f64::from_le_bytes(f64_buf);
            }
            layer.weights = Array2::from_shape_vec((rows, cols), weights_data)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

            reader.read_exact(&mut u64_buf)?;
            let bias_len = u64::from_le_bytes(u64_buf) as usize;

            let mut bias_data = vec![0.0f64; bias_len];
            for value in bias_data.iter_mut() {
                reader.read_exact(&mut f64_buf)?;
                *value = f64::from_le_bytes(f64_buf);
            }
            layer.bias = Array1::from_vec(bias_data);
        }

        Ok(())
    }
}

impl Callback for ModelCheckpoint {
    fn on_train_begin(&mut self, _context: &mut CallbackContext) {
        self.best = f64::INFINITY;
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

        if !self.save_best_only || value < self.best {
            self.best = value;
            context.should_checkpoint = true;
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
