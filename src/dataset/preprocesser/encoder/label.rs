//! Label encoding for categorical features.
//!
//! [`LabelEncoder`] maps each unique string category to a unique integer.
//! This is useful for ordinal or tree-based models that can handle integer
//! categorical data efficiently.
//!
//! The encoder learns the mapping during `.fit()` / `.fit_transform()` and
//! reuses it during `.transform()`. Unseen labels during transform are mapped
//! to `u32::MAX` (a clear sentinel value).

use polars::prelude::*;
use std::collections::HashMap;

use crate::dataset::preprocesser::encoder::{Encoder, EncodingStrategy, PreprocessingError};

/// Configuration for label encoding.
///
/// Stores a per-column mapping from category string to integer label.
#[derive(Debug, Default)]
pub struct LabelConfig {
    pub mapping: HashMap<String, HashMap<String, u32>>,
}

/// Convenient type alias for a fully configured label encoder.
///
/// # Examples
///
/// **Basic usage:**
/// ```
/// use mlrs::dataset::preprocesser::encoder::label::LabelEncoder;
/// use polars::prelude::*;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut df = df![
///     "animal" => ["cat", "dog", "cat", "bird"]
/// ]?;
///
/// let mut encoder = LabelEncoder::new();
/// encoder.fit_transform(&mut df, &["animal"])?;
///
/// // "cat" -> 0, "dog" -> 1, "bird" -> 2 (order depends on first appearance)
/// assert!(df.column("animal")?.u32()?.get(0) == Some(0));
/// # Ok(())
/// # }
/// ```
///
/// **Separate fit and transform (recommended for train/test split):**
/// ```
/// # use mlrs::dataset::preprocesser::encoder::label::LabelEncoder;
/// # use polars::prelude::*;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut train = df!["color" => ["red", "blue", "red", "green"]]?;
/// let mut test  = df!["color" => ["blue", "red", "yellow"]]?;  // "yellow" is unseen
///
/// let mut encoder = LabelEncoder::new();
/// encoder.fit(&train, &["color"])?;
/// encoder.transform(&mut test, &["color"])?;
///
/// // "yellow" becomes u32::MAX
/// # Ok(())
/// # }
/// ```
///
/// **Edge case — empty column (all nulls):**
/// ```
/// # use mlrs::dataset::preprocesser::encoder::label::LabelEncoder;
/// # use polars::prelude::*;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut df = df!["empty" => [Option::<&str>::None, None, None]]?;
///
/// let mut encoder = LabelEncoder::new();
/// // No categories learned → mapping remains empty
/// encoder.fit(&df, &["empty"])?;
/// # Ok(())
/// # }
/// ```
pub type LabelEncoder = Encoder<LabelConfig>;

impl LabelEncoder {
    /// Creates a new unfitted [`LabelEncoder`].
    pub fn new() -> Self {
        Self {
            fitted: false,
            config: LabelConfig {
                mapping: HashMap::new(),
            },
        }
    }
}

impl Default for LabelEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EncodingStrategy for LabelConfig {
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
        let mut label_map: HashMap<String, u32> = HashMap::new();
        let mut counter: u32 = 0;
        for value in column_chuncked.into_iter().flatten() {
            label_map.entry(value.to_string()).or_insert_with(|| {
                let label = counter;
                counter += 1;
                label
            });
        }
        self.mapping.insert(name, label_map);
        Ok(())
    }

    fn apply_encoding(
        &self,
        dataframe: &mut DataFrame,
        name: &str,
    ) -> Result<(), PreprocessingError> {
        let mapping = match self.mapping.get(name) {
            Some(m) => m,
            _ => {
                let error = PreprocessingError::ColumnNotFound(name.to_string());
                error.print_error();
                return Err(error);
            }
        };
        let index = match dataframe.get_column_index(name) {
            Some(i) => i,
            _ => {
                let error = PreprocessingError::ColumnNotFound(name.to_string());
                error.print_error();
                return Err(error);
            }
        };
        let column = dataframe.column(name)?;
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
        let encoded: UInt32Chunked = column_chuncked
            .apply_nonnull_values_generic(DataType::UInt32, |val| {
                *mapping.get(val).unwrap_or(&u32::MAX)
            });
        let series = encoded.into_series().with_name(name.into());
        dataframe.replace_column(index, Column::from(series))?;
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
    fn test_label_fit_basic() {
        let df = make_df();
        let mut encoder = LabelEncoder::new();
        assert!(encoder.fit(&df, &["color", "size"]).is_ok());
        assert!(encoder.fitted);
        assert!(encoder.config.mapping.contains_key("color"));
        assert!(encoder.config.mapping.contains_key("size"));
    }

    #[test]
    fn test_label_fit_column_not_found() {
        let df = make_df();
        let mut encoder = LabelEncoder::new();
        assert!(matches!(
            encoder.fit(&df, &["nonexistent"]),
            Err(PreprocessingError::ColumnNotFound(_))
        ));
    }

    #[test]
    fn test_label_fit_non_string_column() {
        let df = make_df();
        let mut encoder = LabelEncoder::new();
        assert!(matches!(
            encoder.fit(&df, &["weight"]),
            Err(PreprocessingError::InvalidColumnType(_, _, _))
        ));
    }

    #[test]
    fn test_label_transform_not_fitted() {
        let mut df = make_df();
        let encoder = LabelEncoder::new();
        assert!(matches!(
            encoder.transform(&mut df, &["color"]),
            Err(PreprocessingError::NotFitted)
        ));
    }

    #[test]
    fn test_label_transform_correct_values() {
        let mut df = make_df();
        let mut encoder = LabelEncoder::new();
        encoder.fit(&df, &["color"]).unwrap();
        encoder.transform(&mut df, &["color"]).unwrap();

        // dtype should now be UInt32
        assert_eq!(*df.column("color").unwrap().dtype(), DataType::UInt32);

        // same input value should map to same label
        let col = df.column("color").unwrap().u32().unwrap();
        assert_eq!(col.get(0), col.get(2)); // both "red"
    }

    #[test]
    fn test_label_unfitted_columns_unchanged() {
        let mut df = make_df();
        let original = df.column("size").unwrap().clone();
        let mut encoder = LabelEncoder::new();
        encoder.fit(&df, &["color"]).unwrap();
        encoder.transform(&mut df, &["color"]).unwrap();
        assert_eq!(df.column("size").unwrap(), &original);
    }

    #[test]
    fn test_label_fit_transform_end_to_end() {
        let mut df = make_df();
        let mut encoder = LabelEncoder::new();
        assert!(encoder.fit_transform(&mut df, &["color", "size"]).is_ok());
        assert!(encoder.fitted);
        assert_eq!(*df.column("color").unwrap().dtype(), DataType::UInt32);
        assert_eq!(*df.column("size").unwrap().dtype(), DataType::UInt32);
    }
}
