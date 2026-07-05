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

use colored::Colorize;
use polars::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

pub mod min_max;
pub mod standard;
use crate::dataset::preprocessor::error::{PreprocessingError, PreprocessingErrorInner};
use crate::dataset::preprocessor::utils::is_numeric;

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
    stats: HashMap<String, (f64, f64)>,
    fitted: bool,
    config: T,
}

impl<T: Scaling> Scaler<T> {
    /// Creates a new `Scaler` with the given configuration.
    pub(crate) fn with_config(config: T) -> Self {
        Self {
            stats: HashMap::new(),
            fitted: false,
            config,
        }
    }

    /// Returns true if the scaler has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Returns the fitted statistics (column_name -> (stat1, stat2)).
    pub fn get_stats(&self) -> &HashMap<String, (f64, f64)> {
        &self.stats
    }
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
    #[must_use = "this returns a Result that should be handled"]
    pub fn fit(
        &mut self,
        dataframe: &DataFrame,
        columns: &[&str],
    ) -> Result<(), PreprocessingError> {
        let results: Vec<(String, (f64, f64))> = columns
            .par_iter()
            .filter_map(|&name| {
                let column = match dataframe.column(name) {
                    Ok(c) => c,
                    Err(_) => {
                        let msg = format!("Column '{}' not found. Skipping.", name);
                        eprintln!(
                            "{} {}",
                            "WARNING from Scaler<T>: ".yellow().bold(),
                            msg.yellow()
                        );
                        return None;
                    }
                };

                if !is_numeric(column.dtype()) {
                    let msg = format!(
                        "Column '{}' is not numeric (found '{}'). Skipping.",
                        name,
                        column.dtype()
                    );
                    eprintln!(
                        "{} {}",
                        "WARNING from Scaler<T>: ".yellow().bold(),
                        msg.yellow()
                    );
                    return None;
                }

                let column_f64 = match column.cast(&DataType::Float64) {
                    Ok(c) => c,
                    Err(_) => {
                        let msg =
                            format!("Column '{}' could not be cast to Float64. Skipping.", name);
                        eprintln!(
                            "{} {}",
                            "WARNING from Scaler<T>: ".yellow().bold(),
                            msg.yellow()
                        );
                        return None;
                    }
                };

                let stats = self.config.compute_stats(column_f64);
                Some((name.to_string(), stats))
            })
            .collect();

        if results.is_empty() && !columns.is_empty() {
            let msg = "No columns were fitted. All were missing or non-numeric.";
            eprintln!(
                "{} {}",
                "WARNING from Scaler<T>: ".yellow().bold(),
                msg.yellow()
            );
        }

        self.stats = results.into_iter().collect();
        self.fitted = true;
        Ok(())
    }

    /// Transforms the DataFrame in-place using previously fitted statistics.
    ///
    /// Requires a prior call to `.fit()` or `.fit_transform()`.
    #[must_use = "this returns a Result that should be handled"]
    pub fn transform(&self, dataframe: &mut DataFrame) -> Result<(), PreprocessingError> {
        if !self.fitted {
            return Err((PreprocessingErrorInner::NotFitted).into());
        }

        for (name, &(a, b)) in &self.stats {
            let index = dataframe
                .get_column_index(name)
                .ok_or_else(|| PreprocessingErrorInner::ColumnNotFound(name.clone()))?;

            let raw_column = dataframe.column(name)?;

            let column = if raw_column.dtype() == &DataType::Float64 {
                raw_column.clone()
            } else {
                raw_column.cast(&DataType::Float64).map_err(|_| {
                    PreprocessingErrorInner::InvalidColumnType {
                        column: name.clone(),
                        expected: "Float64".to_string(),
                        actual: raw_column.dtype().to_string(),
                    }
                })?
            };

            let scaled_column = self.config.scale_value(column, (a, b));
            dataframe.replace_column(index, scaled_column)?;
        }

        Ok(())
    }

    /// Convenience method that fits and then transforms the data in one step.
    #[must_use = "this returns a Result that should be handled"]
    pub fn fit_transform(
        &mut self,
        dataframe: &mut DataFrame,
        columns: &[&str],
    ) -> Result<(), PreprocessingError> {
        self.fit(dataframe, columns)?;
        self.transform(dataframe)
    }
}
