//! Project-wide constants for mlrs
//!
//! This is the **recommended** place for all default hyperparameters,
//! model configurations, and magic numbers.

// ==================== Training & Optimization ====================

/// Default learning rate for gradient descent based models
pub const DEFAULT_LEARNING_RATE: f64 = 0.01;

/// Default maximum number of training epochs
pub const DEFAULT_MAX_EPOCHS: usize = 10_000;

/// Default convergence tolerance
pub const DEFAULT_TOLERANCE: f64 = 1e-4;

/// Default momentum
pub const DEFAULT_MOMENTUM: f64 = 0.9;

/// Default beta for rms prop
pub const DEFAULT_BETA: f64 = 0.9;

/// Default beta2 for adam
pub const DEFAULT_BETA2: f64 = 0.999;

// ==================== Regularization ====================

/// Default L1 ratio (Lasso strength). 0.0 = no L1 regularization
pub const DEFAULT_L1_RATIO: f64 = 0.0;

/// Default L2 ratio (Ridge strength). 0.0 = no L2 regularization
pub const DEFAULT_L2_RATIO: f64 = 0.0;

/// Default overall regularization strength (used in ElasticNet)
pub const DEFAULT_ALPHA: f64 = 0.0;
