use colored::Colorize;
use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::model::callback::checkpoint::ModelCheckpoint;
use crate::model::callback::{Callback, CallbackContext};
use crate::model::error::ModelError;
use crate::model::layer::Layer;
use crate::model::metric::MetricStrategy;
use crate::model::optimizer::OptimizingStrategy;

// ---------------------------------------------------------------------------
// Layer cache
// ---------------------------------------------------------------------------
struct LayerCache {
    input: Array2<f64>,
    z: Array2<f64>,
    _a: Array2<f64>,
}

// ---------------------------------------------------------------------------
// RunRecord
// ---------------------------------------------------------------------------
#[derive(serde::Serialize, serde::Deserialize)]
pub struct RunRecord {
    pub run_id: usize,
    pub hyperparameters: HashMap<String, f64>,
    pub train_losses: Vec<f64>,
    pub val_losses: Option<Vec<f64>>,
    pub final_score: f64,
    pub metric_name: String,
}

// ---------------------------------------------------------------------------
// SequentialModel
// ---------------------------------------------------------------------------
pub struct SequentialModel<O: OptimizingStrategy> {
    pub layers: Vec<Layer>,
    pub optimizer: O,
    pub metric: Box<dyn MetricStrategy>,
    pub max_epochs: usize,
    pub tolerance: f64,
    pub callbacks: Vec<Box<dyn Callback>>,
    pub plot_path: PathBuf,
    n_classes: Option<usize>,
    detected_classes: Option<usize>,
    fitted: bool,
    train_losses: Vec<f64>,
    val_losses: Vec<f64>,
}

impl<O: OptimizingStrategy> SequentialModel<O> {
    pub fn new(layers: Vec<Layer>, optimizer: O, metric: Box<dyn MetricStrategy>) -> Self {
        Self {
            layers,
            optimizer,
            metric,
            max_epochs: 1000,
            tolerance: 1e-4,
            callbacks: Vec::new(),
            plot_path: PathBuf::from("."),
            n_classes: None,
            detected_classes: None,
            fitted: false,
            train_losses: Vec::new(),
            val_losses: Vec::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Builder methods
    // -----------------------------------------------------------------------
    pub fn with_max_epochs(mut self, max_epochs: usize) -> Self {
        self.max_epochs = max_epochs;
        self
    }

    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    pub fn with_metric(mut self, metric: Box<dyn MetricStrategy>) -> Self {
        self.metric = metric;
        self
    }

    pub fn with_callbacks(mut self, callbacks: Vec<Box<dyn Callback>>) -> Self {
        self.callbacks = callbacks;
        self
    }

    pub fn with_plot_path(mut self, path: PathBuf) -> Self {
        self.plot_path = path;
        self
    }

    pub fn with_n_classes(mut self, n_classes: usize) -> Self {
        self.n_classes = Some(n_classes);
        self
    }

    pub fn reset_all_weights(&mut self) {
        for layer in &mut self.layers {
            let (rows, cols) = layer.weights.dim();
            layer.initialize(rows, cols);
        }
    }

    pub fn with_optimizer<O2: OptimizingStrategy>(self, optimizer: O2) -> SequentialModel<O2> {
        SequentialModel {
            layers: self.layers,
            optimizer,
            metric: self.metric,
            max_epochs: self.max_epochs,
            tolerance: self.tolerance,
            callbacks: self.callbacks,
            plot_path: self.plot_path,
            n_classes: self.n_classes,
            detected_classes: self.detected_classes,
            fitted: self.fitted,
            train_losses: self.train_losses,
            val_losses: self.val_losses,
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------
    fn is_multiclass(&self) -> bool {
        self.n_classes.or(self.detected_classes).unwrap_or(0) > 2
    }

    fn resolve_n_classes(&self) -> usize {
        self.n_classes.or(self.detected_classes).unwrap_or(1)
    }

    fn one_hot(y: &Array1<f64>, n_classes: usize) -> Array2<f64> {
        let n = y.len();
        let mut encoded = Array2::zeros((n, n_classes));
        for (i, &label) in y.iter().enumerate() {
            encoded[[i, label as usize]] = 1.0;
        }
        encoded
    }

    // -----------------------------------------------------------------------
    // Forward pass
    // -----------------------------------------------------------------------
    fn forward(&self, x: &Array2<f64>) -> (Array2<f64>, Vec<LayerCache>) {
        let mut cache = Vec::with_capacity(self.layers.len());
        let mut current = x.clone();

        for layer in &self.layers {
            let z = layer.forward(&current);
            let n = z.nrows();
            let out_size = z.ncols();
            let mut a = Array2::zeros((n, out_size));
            for (mut out_row, z_row) in a.axis_iter_mut(Axis(0)).zip(z.axis_iter(Axis(0))) {
                let activated = layer.activation.apply(&z_row.to_owned());
                out_row.assign(&activated);
            }
            cache.push(LayerCache {
                input: current.clone(),
                z: z.clone(),
                _a: a.clone(),
            });
            current = a;
        }

        (current, cache)
    }

    // -----------------------------------------------------------------------
    // Training loss
    // -----------------------------------------------------------------------
    fn compute_train_loss(&self, y_pred: &Array2<f64>, y_true: &Array1<f64>) -> f64 {
        let n = y_true.len() as f64;
        let eps = 1e-15;

        match self.layers.last().unwrap().activation.name() {
            "softmax" => {
                y_true
                    .iter()
                    .enumerate()
                    .map(|(i, &label)| -y_pred[[i, label as usize]].clamp(eps, 1.0).ln())
                    .sum::<f64>()
                    / n
            }
            "sigmoid" => {
                y_true
                    .iter()
                    .zip(y_pred.column(0).iter())
                    .map(|(&t, &p)| {
                        let p = p.clamp(eps, 1.0 - eps);
                        -(t * p.ln() + (1.0 - t) * (1.0 - p).ln())
                    })
                    .sum::<f64>()
                    / n
            }
            _ => {
                let pred = y_pred.column(0).to_owned();
                let diff = y_true - &pred;
                (&diff * &diff).mean().unwrap_or(f64::NAN)
            }
        }
    }

    // -----------------------------------------------------------------------
    // Backward pass
    // -----------------------------------------------------------------------
    fn backward(
        &self,
        y_true: &Array1<f64>,
        y_pred: &Array2<f64>,
        cache: &[LayerCache],
    ) -> Vec<(Array2<f64>, Array1<f64>)> {
        let n = y_true.len() as f64;
        let mut gradients = vec![(Array2::zeros((1, 1)), Array1::zeros(1)); self.layers.len()];

        let mut delta: Array2<f64> = match self.layers.last().unwrap().activation.name() {
            "softmax" => {
                let y_onehot = Self::one_hot(y_true, self.resolve_n_classes());
                (y_pred - &y_onehot) / n
            }
            "sigmoid" => {
                let y_col = y_true
                    .clone()
                    .into_shape_with_order((y_true.len(), 1))
                    .expect("Reshape failed");
                (y_pred - &y_col) / n
            }
            _ => {
                let y_col = y_true
                    .clone()
                    .into_shape_with_order((y_true.len(), 1))
                    .expect("Reshape failed");
                2.0 * (y_pred - &y_col) / n
            }
        };

        for i in (0..self.layers.len()).rev() {
            let dw = cache[i].input.t().dot(&delta);
            let db = delta.sum_axis(Axis(0));
            gradients[i] = (dw, db);

            if i > 0 {
                let delta_a = delta.dot(&self.layers[i].weights.t());
                let da_dz = Array2::from_shape_fn(cache[i - 1].z.dim(), |(r, c)| {
                    let z_val = Array1::from_elem(1, cache[i - 1].z[[r, c]]);
                    self.layers[i - 1].activation.derivative(&z_val)[0]
                });
                delta = delta_a * da_dz;
            }
        }

        gradients
    }

    // -----------------------------------------------------------------------
    // Fit
    // -----------------------------------------------------------------------
    pub fn fit(
        &mut self,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
        validation: Option<(&Array2<f64>, &Array1<f64>)>,
    ) -> Result<(), ModelError> {
        let n_features = x_train.ncols();

        // detect n_classes
        let detected = {
            let mut classes: Vec<usize> = y_train.iter().map(|&v| v as usize).collect();
            classes.sort_unstable();
            classes.dedup();
            classes.len()
        };
        let n_classes = self.n_classes.unwrap_or(detected);
        let output_size = if n_classes > 2 { n_classes } else { 1 };

        // warn and reinit output layer if n_classes changed
        if let Some(prev) = self.detected_classes
            && prev != n_classes
        {
            eprintln!(
                "{} n_classes changed from {} to {}. Reinitializing output layer.",
                "Warning:".yellow().bold(),
                prev,
                n_classes
            );
            if let Some(last) = self.layers.last_mut() {
                let input_size = last.weights.nrows();
                last.initialize(input_size, output_size);
            }
        }
        self.detected_classes = Some(n_classes);

        // lazy init layers
        let mut prev_size = n_features;
        let last_idx = self.layers.len() - 1;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if !layer.is_initialized() {
                let out = if i == last_idx {
                    output_size
                } else {
                    layer.weights.ncols().max(1)
                };
                layer.initialize(prev_size, out);
            }
            prev_size = layer.weights.ncols();
        }

        self.optimizer.reset();
        self.train_losses.clear();
        self.val_losses.clear();

        let mut ctx = CallbackContext {
            epoch: 0,
            max_epochs: self.max_epochs,
            train_loss: 0.0,
            val_loss: None,
            metric_score: 0.0,
            learning_rate: self.optimizer.learning_rate(),
            stop_training: false,
            should_checkpoint: false,
        };

        for cb in &mut self.callbacks {
            cb.on_train_begin(&mut ctx);
        }

        for epoch in 0..self.max_epochs {
            ctx.epoch = epoch;
            ctx.should_checkpoint = false;
            for cb in &mut self.callbacks {
                cb.on_epoch_begin(&mut ctx);
            }

            let (y_pred, cache) = self.forward(x_train);
            let train_loss = self.compute_train_loss(&y_pred, y_train);
            self.train_losses.push(train_loss);
            ctx.train_loss = train_loss;

            if let Some((x_val, y_val)) = validation {
                let (y_val_pred, _) = self.forward(x_val);
                ctx.val_loss = Some(self.compute_train_loss(&y_val_pred, y_val));
                self.val_losses.push(ctx.val_loss.unwrap());
                ctx.metric_score = self.metric.compute(y_val, &self.flatten_pred(&y_val_pred));
            } else {
                ctx.metric_score = self.metric.compute(y_train, &self.flatten_pred(&y_pred));
            }

            let gradients = self.backward(y_train, &y_pred, &cache);
            self.optimizer.step(&mut self.layers, &gradients);

            for cb in &mut self.callbacks {
                cb.on_epoch_end(&mut ctx);
            }
            self.optimizer.set_learning_rate(ctx.learning_rate);

            if ctx.should_checkpoint {
                let path = self.callbacks.iter().find_map(|cb| {
                    cb.as_any()
                        .downcast_ref::<ModelCheckpoint>()
                        .map(|c| c.path.clone())
                });
                if let Some(p) = path {
                    match ModelCheckpoint::save(&p, &self.layers) {
                        Ok(_) => eprintln!(
                            "{} Saved checkpoint at epoch {}.",
                            "ModelCheckpoint:".cyan().bold(),
                            epoch
                        ),
                        Err(e) => {
                            eprintln!("{} Failed to save: {}", "ModelCheckpoint:".red().bold(), e)
                        }
                    }
                }
            }

            if ctx.stop_training || train_loss < self.tolerance {
                break;
            }
        }

        for cb in &mut self.callbacks {
            cb.on_train_end(&mut ctx);
        }
        self.fitted = true;
        self.save_run_async();
        Ok(())
    }

    fn flatten_pred(&self, y_pred: &Array2<f64>) -> Array1<f64> {
        if self.is_multiclass() {
            y_pred.map_axis(Axis(1), |row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i as f64)
                    .unwrap_or(0.0)
            })
        } else {
            y_pred.column(0).to_owned()
        }
    }

    // -----------------------------------------------------------------------
    // Predict
    // -----------------------------------------------------------------------
    pub fn predict(&self, x_test: &Array2<f64>) -> Result<Array1<f64>, ModelError> {
        if !self.fitted {
            return Err(ModelError::NotFitted);
        }
        let (y_pred, _) = self.forward(x_test);
        Ok(self.flatten_pred(&y_pred))
    }

    pub fn predict_proba(&self, x_test: &Array2<f64>) -> Result<Array2<f64>, ModelError> {
        if !self.fitted {
            return Err(ModelError::NotFitted);
        }
        let (y_pred, _) = self.forward(x_test);
        Ok(y_pred)
    }

    // -----------------------------------------------------------------------
    // Evaluate
    // -----------------------------------------------------------------------
    pub fn evaluate(&self, x_test: &Array2<f64>, y_test: &Array1<f64>) -> f64 {
        match self.predict(x_test) {
            Ok(y_pred) => self.metric.compute(y_test, &y_pred),
            Err(e) => {
                e.print_error();
                f64::NAN
            }
        }
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------
    pub fn weights(&self) -> &Array2<f64> {
        &self.layers[0].weights
    }
    pub fn bias(&self) -> &Array1<f64> {
        &self.layers[0].bias
    }

    // -----------------------------------------------------------------------
    // Async run save + plot
    // -----------------------------------------------------------------------
    fn save_run_async(&self) {
        let train_losses = self.train_losses.clone();
        let val_losses = if self.val_losses.is_empty() {
            None
        } else {
            Some(self.val_losses.clone())
        };
        let hyperparameters = self.optimizer.hyperparameters();
        let metric_name = self.metric.name().to_string();
        let final_score = *train_losses.last().unwrap_or(&f64::NAN);
        let plot_path = self.plot_path.clone();

        std::thread::spawn(move || {
            let json_path = plot_path.join("mlrs_runs.json");
            let html_path = plot_path.join("mlrs_runs.html");

            let mut runs: Vec<RunRecord> = if json_path.exists() {
                std::fs::read_to_string(&json_path)
                    .ok()
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or_default()
            } else {
                Vec::new()
            };

            runs.push(RunRecord {
                run_id: runs.len() + 1,
                hyperparameters,
                train_losses,
                val_losses,
                final_score,
                metric_name,
            });

            if let Ok(json) = serde_json::to_string_pretty(&runs) {
                let _ = std::fs::write(&json_path, json);
            }

            generate_plot(&runs, &html_path);
        });
    }
}

// ---------------------------------------------------------------------------
// Plot generation
// ---------------------------------------------------------------------------
fn generate_plot(runs: &[RunRecord], html_path: &PathBuf) {
    use plotly::common::Mode;
    use plotly::layout::{Axis as PlotAxis, Layout};
    use plotly::{Plot, Scatter};

    let mut plot = Plot::new();

    for run in runs {
        let epochs: Vec<usize> = (1..=run.train_losses.len()).collect();

        let hover: String = {
            let mut params: Vec<_> = run.hyperparameters.iter().collect();
            params.sort_by_key(|(k, _)| k.as_str());
            let lines: Vec<String> = params
                .iter()
                .map(|(k, v)| format!("{}: {:.6}", k, v))
                .collect();
            format!("Run {}<br>{}", run.run_id, lines.join("<br>"))
        };

        plot.add_trace(
            Scatter::new(epochs.clone(), run.train_losses.clone())
                .name(format!("Run {} - Train Loss", run.run_id))
                .mode(Mode::Lines)
                .hover_text(hover.clone()),
        );

        if let Some(val_losses) = &run.val_losses {
            plot.add_trace(
                Scatter::new(epochs.clone(), val_losses.clone())
                    .name(format!(
                        "Run {} - Val Loss ({})",
                        run.run_id, run.metric_name
                    ))
                    .mode(Mode::Lines)
                    .hover_text(hover.clone()),
            );
        }
    }

    plot.set_layout(
        Layout::new()
            .title("mlrs Training Dashboard")
            .x_axis(PlotAxis::new().title("Epoch"))
            .y_axis(PlotAxis::new().title("Loss")),
    );

    plot.write_html(html_path);
}
