//! Feature scaling utilities for machine learning pipelines.
//!
//! This module provides two common scalers:
//! - [`MinMaxScaler`] — scales features to a given range (default `[0, 1]`)
//! - [`StandardScaler`] — standardizes features to zero mean and unit variance
//!
//! All scalers support parallel computation via Rayon and work directly on
//! `polars::DataFrame`. They follow the familiar scikit-learn style API:
//! `.fit()`, `.transform()`, and `.fit_transform()`.

pub use min_max::MinMaxScaler;
pub use standard::StandardScaler;

use polars::prelude::*;
use rayon::prelude::*;
use std::{collections::HashMap, sync::Mutex};

pub mod min_max;
pub mod standard;
use crate::dataset::preprocesser::{PreprocessingError, is_numeric};

/// Common trait for all scaling strategies.
///
/// Implementors define how statistics are computed and how values are scaled.
pub trait Scaling: Sync {
    fn compute_stats(&self, column: Column) -> (f64, f64);
    fn scale_value(&self, column: Column, stats: (f64, f64)) -> Column;
}

/// Generic scaler that holds fitted statistics for selected columns.
///
/// Use [`MinMaxScaler`] or [`StandardScaler`] for concrete implementations.
pub struct Scaler<T: Scaling> {
    pub stats: HashMap<String, (f64, f64)>,
    pub fitted: bool,
    pub config: T,
}

impl<T: Scaling + Sync> Scaler<T> {
    /// Fits the scaler on the specified columns using parallel computation.
    ///
    /// Only numeric columns are processed. Non-numeric columns and missing
    /// columns emit a warning and are skipped.
    ///
    /// # Examples
    ///
    /// **Basic usage with MinMaxScaler:**
    /// ```
    /// use mlrs::dataset::preprocesser::scaler::min_max::MinMaxScaler;
    /// use polars::prelude::*;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut df = df![
    ///     "age" => [25, 30, 35, 40],
    ///     "income" => [50000, 60000, 70000, 80000]
    /// ]?;
    ///
    /// let mut scaler = MinMaxScaler::new((0.0, 1.0));
    /// scaler.fit(&df, &["age", "income"])?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// **Using with StandardScaler:**
    /// ```
    /// # use mlrs::dataset::preprocesser::scaler::standard::StandardScaler;
    /// # use polars::prelude::*;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut df = df!["feature" => [1.0, 2.0, 3.0, 4.0]]?;
    /// let mut scaler = StandardScaler::new();
    /// scaler.fit(&df, &["feature"])?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// **Edge case — mixed numeric/non-numeric columns:**
    /// ```
    /// # use mlrs::dataset::preprocesser::scaler::MinMaxScaler;
    /// # use polars::prelude::*;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let df = df![
    ///     "numeric" => [1.0, 2.0, 3.0],
    ///     "categorical" => ["a", "b", "c"]
    /// ]?;
    ///
    /// let mut scaler = MinMaxScaler::new((0.0, 1.0));
    /// // Only "numeric" will be fitted; warning printed for "categorical"
    /// let _ = scaler.fit(&df, &["numeric", "categorical"]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn fit(
        &mut self,
        dataframe: &DataFrame,
        columns: &[&str],
    ) -> Result<(), PreprocessingError> {
        let stats: Mutex<HashMap<String, (f64, f64)>> = Mutex::new(HashMap::new());

        columns.par_iter().for_each(|&name| {
            let column = match dataframe.column(name) {
                Ok(c) => c,
                Err(_) => {
                    PreprocessingError::print_warning(format!(
                        "Column '{}' not found.",
                        name
                    ));
                    return;
                }
            };
            if !is_numeric(column.dtype()) {
                PreprocessingError::print_warning(format!(
                    "Column '{}' is not the expected type. Expected datatype: '{}', Found datatype: '{}'.",
                    name,
                    "Float / Int",
                    column.dtype()
                ));
                return ;
            }
            let column_f64 = column.cast(&DataType::Float64).unwrap();
            let (a, b) = self.config.compute_stats(column_f64);
            stats.lock().unwrap().insert(name.to_string(), (a, b));
        });
        self.stats = stats.into_inner().unwrap();
        self.fitted = true;
        Ok(())
    }

    /// Transforms the DataFrame in-place using previously fitted statistics.
    ///
    /// Requires a prior call to `.fit()` or `.fit_transform()`.
    pub fn transform(&self, dataframe: &mut DataFrame) -> Result<(), PreprocessingError> {
        if !self.fitted {
            let error = PreprocessingError::NotFitted;
            error.print_error();
            return Err(error);
        }
        for (name, &(a, b)) in &self.stats {
            let index = match dataframe.get_column_index(name.as_str()) {
                Some(i) => i,
                _ => {
                    let error = PreprocessingError::ColumnNotFound(name.to_string());
                    error.print_error();
                    return Err(error);
                }
            };
            let column = match dataframe
                .column(name.as_str())
                .unwrap()
                .cast(&DataType::Float64)
            {
                Ok(c) => c,
                Err(_) => {
                    let error = PreprocessingError::InvalidColumnType(
                        name.to_string(),
                        "Float / Int".to_string(),
                        dataframe.column(name)?.dtype().to_string(),
                    );
                    error.print_error();
                    return Err(error);
                }
            };
            let scaled_column = self.config.scale_value(column, (a, b));
            match dataframe.replace_column(index, scaled_column) {
                Ok(_) => continue,
                Err(_) => {
                    let error = PreprocessingError::ColumnNotFound(name.to_string());
                    error.print_error();
                    return Err(error);
                }
            }
        }
        Ok(())
    }

    /// Convenience method that fits and then transforms the data in one step.
    pub fn fit_transform(
        &mut self,
        dataframe: &mut DataFrame,
        columns: &[&str],
    ) -> Result<(), PreprocessingError> {
        self.fit(dataframe, columns)?;
        self.transform(dataframe)
    }
}
