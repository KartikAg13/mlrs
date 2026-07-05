//! Encoding strategies for categorical features.
//!
//! This module provides two common encoders:
//! - [`LabelEncoder`] — maps categories to integers
//! - [`OneHotEncoder`] — creates binary columns for each category

pub use label::LabelEncoder;
pub use one_hot::OneHotEncoder;

use polars::prelude::*;

pub mod label;
pub mod one_hot;
use crate::dataset::preprocessor::error::{PreprocessingError, PreprocessingErrorInner};

/// Common interface for all encoding strategies.
pub trait EncodingStrategy {
    fn fit_column(&mut self, column: &Column) -> Result<(), PreprocessingError>;
    fn transform_column(
        &self,
        dataframe: &mut DataFrame,
        name: &str,
    ) -> Result<(), PreprocessingError>;
}

/// Generic encoder that wraps any [`EncodingStrategy`].
///
/// Use [`LabelEncoder`] or [`OneHotEncoder`] for concrete implementations.
pub struct Encoder<T: EncodingStrategy> {
    fitted: bool,
    fitted_columns: Vec<PlSmallStr>,
    pub(crate) config: T,
}

impl<T: EncodingStrategy> Encoder<T> {
    /// Returns true if the encoder has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Returns the columns that were used during fitting.
    pub fn fitted_columns(&self) -> &[PlSmallStr] {
        &self.fitted_columns
    }

    /// Returns a reference to the underlying config.
    pub fn config(&self) -> &T {
        &self.config
    }

    pub(crate) fn with_config(config: T) -> Self {
        Self {
            fitted: false,
            fitted_columns: Vec::new(),
            config,
        }
    }

    /// Fits the encoder on the specified columns.
    ///
    /// # Examples
    ///
    /// **Normal usage:**
    /// ```
    /// # use mlrs::dataset::preprocesser::encoder::{LabelEncoder, Encoder};
    /// # use polars::prelude::*;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut df = df!["category" => ["cat", "dog", "cat", "bird"]]?;
    /// let mut encoder = LabelEncoder::new();
    /// encoder.fit(&df, &["category"])?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// **Chained with transform:**
    /// ```
    /// # use mlrs::dataset::preprocesser::encoder::LabelEncoder;
    /// # use polars::prelude::*;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut df = df!["color" => ["red", "blue", "red"]]?;
    /// let mut encoder = LabelEncoder::new();
    /// encoder.fit_transform(&mut df, &["color"])?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// **Edge case — missing column:**
    /// ```
    /// # use mlrs::dataset::preprocesser::encoder::LabelEncoder;
    /// # use polars::prelude::*;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let df = df!["existing" => [1, 2, 3]]?;
    /// let mut encoder = LabelEncoder::new();
    /// let result = encoder.fit(&df, &["missing_column"]);
    /// assert!(result.is_err());
    /// # Ok(())
    /// # }
    /// ```
    #[must_use = "this returns a Result that should be handled"]
    pub fn fit(
        &mut self,
        dataframe: &DataFrame,
        columns: &[&str],
    ) -> Result<(), PreprocessingError> {
        self.fitted_columns.reserve(columns.len());

        for &name in columns {
            let column = dataframe
                .column(name)
                .map_err(|_| PreprocessingErrorInner::ColumnNotFound(name.to_string()))?;

            if !matches!(column.dtype(), DataType::String) {
                return Err((PreprocessingErrorInner::InvalidColumnType {
                    column: name.to_string(),
                    expected: "String".to_string(),
                    actual: column.dtype().to_string(),
                })
                .into());
            }

            self.config.fit_column(column)?;
            self.fitted_columns.push(name.into());
        }

        self.fitted = true;
        Ok(())
    }

    /// Transforms the DataFrame using the fitted encoding.
    ///
    /// Requires a prior call to `.fit()` or `.fit_transform()`.
    /// Applies encoding to all columns that were specified during fitting.
    #[must_use = "this returns a Result that should be handled"]
    pub fn transform(&self, dataframe: &mut DataFrame) -> Result<(), PreprocessingError> {
        if !self.fitted {
            return Err((PreprocessingErrorInner::NotFitted).into());
        }

        for name in &self.fitted_columns {
            self.config.transform_column(dataframe, name.as_str())?;
        }
        Ok(())
    }

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
