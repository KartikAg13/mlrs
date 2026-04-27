//! Preprocessing utilities for machine learning datasets.
//!
//! This module contains tools for handling missing values, scaling features,
//! and encoding categorical variables — all designed to work seamlessly with
//! Polars `DataFrame`s.

pub mod encoder;
pub mod error;
pub mod handler;
pub mod scaler;
pub mod utils;
