//! One-hot encoding for categorical features.
//!
//! [`OneHotEncoder`] creates a new binary column for each unique category
//! found during fitting. This is the standard approach for linear models,
//! neural networks, and any algorithm that cannot handle integer categorical
//! data directly.
//!
//! **Note:** One-hot encoding increases the number of columns. Use it
//! judiciously with high-cardinality features.

use polars::prelude::*;
use std::collections::{HashMap, HashSet};

use crate::dataset::preprocesser::encoder::{Encoder, EncodingStrategy};
use crate::dataset::preprocesser::error::PreprocessingError;

/// Configuration for one-hot encoding.
///
/// Stores the sorted list of categories seen for each column.
#[derive(Debug, Default)]
pub struct OneHotConfig {
    pub categories: HashMap<String, Vec<String>>,
}

/// Convenient type alias for a fully configured one-hot encoder.
///
/// # Examples
///
/// **Basic usage:**
/// ```
/// use mlrs::dataset::preprocesser::encoder::one_hot::OneHotEncoder;
/// use polars::prelude::*;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut df = df![
///     "fruit" => ["apple", "banana", "apple", "cherry"]
/// ]?;
///
/// let mut encoder = OneHotEncoder::new();
/// encoder.fit_transform(&mut df, &["fruit"])?;
///
/// // New columns: fruit_apple, fruit_banana, fruit_cherry
/// assert_eq!(df.width(), 4); // original + 3 one-hot columns
/// # Ok(())
/// # }
/// ```
///
/// **Train/test split usage (recommended):**
/// ```
/// # use mlrs::dataset::preprocesser::encoder::one_hot::OneHotEncoder;
/// # use polars::prelude::*;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut train = df!["city" => ["NY", "LA", "NY", "SF"]]?;
/// let mut test  = df!["city" => ["LA", "NY", "Chicago"]]?; // Chicago is unseen
///
/// let mut encoder = OneHotEncoder::new();
/// encoder.fit(&train, &["city"])?;
/// encoder.transform(&mut test, &["city"])?;
///
/// // Unseen categories produce all-zero rows for that column
/// # Ok(())
/// # }
/// ```
///
/// **Edge case — single unique value (constant column):**
/// ```
/// # use mlrs::dataset::preprocesser::encoder::one_hot::OneHotEncoder;
/// # use polars::prelude::*;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut df = df!["constant" => ["yes", "yes", "yes"]]?; // only one category
///
/// let mut encoder = OneHotEncoder::new();
/// encoder.fit_transform(&mut df, &["constant"])?;
///
/// // Only one new column is created: constant_yes
/// assert_eq!(df.width(), 2);
/// # Ok(())
/// # }
/// ```
pub type OneHotEncoder = Encoder<OneHotConfig>;

impl OneHotEncoder {
    /// Creates a new unfitted [`OneHotEncoder`].
    pub fn new() -> Self {
        Self {
            fitted: false,
            config: OneHotConfig {
                categories: HashMap::new(),
            },
        }
    }
}

impl Default for OneHotEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EncodingStrategy for OneHotConfig {
    fn compute_encoding(&mut self, column: &Column) -> Result<(), PreprocessingError> {
        let name = column.name().to_string();
        let column_chuncked = match column.str() {
            Ok(c) => c,
            Err(_) => {
                let error = PreprocessingError::InvalidColumnType(
                    name,
                    "String".to_string(),
                    column.dtype().to_string(),
                );
                error.print_error();
                return Err(error);
            }
        };
        let mut seen: HashSet<String> = HashSet::new();
        let mut cats: Vec<String> = Vec::new();
        for value in column_chuncked.into_iter().flatten() {
            if seen.insert(value.to_string()) {
                cats.push(value.to_string());
            }
        }
        cats.sort();
        self.categories.insert(name, cats);
        Ok(())
    }

    fn apply_encoding(
        &self,
        dataframe: &mut DataFrame,
        name: &str,
    ) -> Result<(), PreprocessingError> {
        let cats = match self.categories.get(name) {
            Some(c) => c,
            _ => {
                let error = PreprocessingError::ColumnNotFound(name.to_string());
                error.print_error();
                return Err(error);
            }
        };
        let column = dataframe.column(name)?.clone();
        let column_chuncked = match column.str() {
            Ok(c) => c,
            Err(_) => {
                let error = PreprocessingError::InvalidColumnType(
                    name.to_string(),
                    "String".to_string(),
                    column.dtype().to_string(),
                );
                error.print_error();
                return Err(error);
            }
        };
        let new_columns: Vec<Column> = cats
            .iter()
            .map(|cat| {
                let column_name = format!("{}_{}", name, cat);
                let binary: BooleanChunked = column_chuncked
                    .apply_nonnull_values_generic(DataType::Boolean, |value| value == cat.as_str());
                Column::from(binary.into_series().with_name(column_name.as_str().into()))
            })
            .collect();
        dataframe.hstack_mut(&new_columns)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::preprocesser::encoder::PreprocessingError;

    fn make_df() -> DataFrame {
        df![
            "color"  => ["red", "blue", "red", "green"],
            "size"   => ["S", "M", "L", "M"],
            "weight" => [1.0_f64, 2.0, 3.0, 4.0]
        ]
        .unwrap()
    }

    #[test]
    fn test_onehot_fit_basic() {
        let df = make_df();
        let mut encoder = OneHotEncoder::new();
        assert!(encoder.fit(&df, &["color"]).is_ok());
        assert!(encoder.fitted);
        assert!(encoder.config.categories.contains_key("color"));
        // sorted: blue, green, red
        assert_eq!(
            encoder.config.categories["color"],
            vec!["blue", "green", "red"]
        );
    }

    #[test]
    fn test_onehot_fit_column_not_found() {
        let df = make_df();
        let mut encoder = OneHotEncoder::new();
        assert!(matches!(
            encoder.fit(&df, &["nonexistent"]),
            Err(PreprocessingError::ColumnNotFound(_))
        ));
    }

    #[test]
    fn test_onehot_fit_non_string_column() {
        let df = make_df();
        let mut encoder = OneHotEncoder::new();
        assert!(matches!(
            encoder.fit(&df, &["weight"]),
            Err(PreprocessingError::InvalidColumnType(_, _, _))
        ));
    }

    #[test]
    fn test_onehot_transform_not_fitted() {
        let mut df = make_df();
        let encoder = OneHotEncoder::new();
        assert!(matches!(
            encoder.transform(&mut df, &["color"]),
            Err(PreprocessingError::NotFitted)
        ));
    }

    #[test]
    fn test_onehot_transform_adds_columns() {
        let mut df = make_df();
        let original_width = df.width();
        let mut encoder = OneHotEncoder::new();
        encoder.fit(&df, &["color"]).unwrap();
        encoder.transform(&mut df, &["color"]).unwrap();

        // 3 new columns: color_blue, color_green, color_red
        assert_eq!(df.width(), original_width + 3);
        assert!(df.column("color_red").is_ok());
        assert!(df.column("color_blue").is_ok());
        assert!(df.column("color_green").is_ok());
    }

    #[test]
    fn test_onehot_transform_correct_values() {
        let mut df = make_df();
        let mut encoder = OneHotEncoder::new();
        encoder.fit_transform(&mut df, &["color"]).unwrap();

        let red_col = df.column("color_red").unwrap().bool().unwrap();
        // rows 0 and 2 are "red"
        assert_eq!(red_col.get(0), Some(true));
        assert_eq!(red_col.get(1), Some(false));
        assert_eq!(red_col.get(2), Some(true));
        assert_eq!(red_col.get(3), Some(false));
    }

    #[test]
    fn test_onehot_fit_transform_end_to_end() {
        let mut df = make_df();
        let mut encoder = OneHotEncoder::new();
        assert!(encoder.fit_transform(&mut df, &["color", "size"]).is_ok());
        // color: 3 cats, size: 3 cats (L, M, S) → 6 new columns
        assert!(df.column("color_red").is_ok());
        assert!(df.column("size_L").is_ok());
    }
}
