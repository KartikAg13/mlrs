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
use crate::dataset::preprocesser::PreprocessingError;

/// Common interface for all encoding strategies.
pub trait EncodingStrategy {
    fn compute_encoding(&mut self, column: &Column) -> Result<(), PreprocessingError>;
    fn apply_encoding(
        &self,
        dataframe: &mut DataFrame,
        name: &str,
    ) -> Result<(), PreprocessingError>;
}

/// Generic encoder that wraps any [`EncodingStrategy`].
///
/// Use [`LabelEncoder`] or [`OneHotEncoder`] for concrete implementations.
pub struct Encoder<T: EncodingStrategy> {
    pub fitted: bool,
    pub config: T,
}

impl<T: EncodingStrategy> Encoder<T> {
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
    pub fn fit(
        &mut self,
        dataframe: &DataFrame,
        columns: &[&str],
    ) -> Result<(), PreprocessingError> {
        for &name in columns {
            let column = match dataframe.column(name) {
                Ok(c) => c.clone(),
                Err(_) => {
                    let error = PreprocessingError::ColumnNotFound(name.to_string());
                    error.print_error();
                    return Err(error);
                }
            };
            if !matches!(column.dtype(), DataType::String) {
                let error = PreprocessingError::InvalidColumnType(
                    name.to_string(),
                    "String".to_string(),
                    column.dtype().to_string(),
                );
                error.print_error();
                return Err(error);
            }
            match self.config.compute_encoding(&column) {
                Ok(_) => continue,
                Err(error) => {
                    error.print_error();
                    return Err(error);
                }
            }
        }
        self.fitted = true;
        Ok(())
    }

    pub fn transform(
        &self,
        dataframe: &mut DataFrame,
        columns: &[&str],
    ) -> Result<(), PreprocessingError> {
        if !self.fitted {
            let error = PreprocessingError::NotFitted;
            error.print_error();
            return Err(error);
        }
        for &name in columns {
            match self.config.apply_encoding(dataframe, name) {
                Ok(_) => continue,
                Err(error) => {
                    error.print_error();
                    return Err(error);
                }
            }
        }
        Ok(())
    }

    pub fn fit_transform(
        &mut self,
        dataframe: &mut DataFrame,
        columns: &[&str],
    ) -> Result<(), PreprocessingError> {
        self.fit(dataframe, columns)?;
        self.transform(dataframe, columns)
    }
}
