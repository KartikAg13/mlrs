//! Missing value handlers for machine learning preprocessing.
//!
//! This module provides utilities to handle missing (null) values in
//! [`polars::prelude::DataFrame`]s, including dropping rows/columns and
//! imputing values using various strategies.
//!
//! The recommended approach for most pipelines is to use [`SimpleImputer`]
//! for imputation followed by scaling/encoding.

pub mod dropper;
pub mod imputer;

pub use dropper::{Dropper, DropperStrategy};
pub use imputer::{ImputerStrategy, SimpleImputer};
