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

use crate::dataset::preprocessor::encoder::{Encoder, EncodingStrategy};
use crate::dataset::preprocessor::error::{PreprocessingError, PreprocessingErrorInner};

/// Configuration for label encoding.
///
/// Stores a per-column mapping from category string to integer label.
#[derive(Debug, Default)]
pub struct LabelConfig {
    pub(crate) mapping: HashMap<PlSmallStr, HashMap<PlSmallStr, u32>>,
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
/// encoder.transform(&mut test)?;
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
        Self::with_config(LabelConfig {
            mapping: HashMap::new(),
        })
    }

    /// Returns the learned label mappings (column_name -> {label -> integer}).
    pub fn get_mapping(&self) -> &HashMap<PlSmallStr, HashMap<PlSmallStr, u32>> {
        &self.config().mapping
    }
}

impl Default for LabelEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EncodingStrategy for LabelConfig {
    fn fit_column(&mut self, column: &Column) -> Result<(), PreprocessingError> {
        let name: PlSmallStr = column.name().clone().into();

        let column_chunked =
            column
                .str()
                .map_err(|_| PreprocessingErrorInner::InvalidColumnType {
                    column: name.to_string(),
                    expected: "String".to_string(),
                    actual: column.dtype().to_string(),
                })?;

        let label_map = self
            .mapping
            .entry(name.clone())
            .or_insert_with(|| HashMap::with_capacity(16));

        let mut counter: u32 = label_map.len() as u32;

        for value in column_chunked.iter().flatten() {
            let key: PlSmallStr = value.into();
            if let std::collections::hash_map::Entry::Vacant(e) = label_map.entry(key) {
                e.insert(counter);
                counter += 1;
            }
        }
        Ok(())
    }

    fn transform_column(
        &self,
        dataframe: &mut DataFrame,
        name: &str,
    ) -> Result<(), PreprocessingError> {
        let name_small: PlSmallStr = name.into();

        let mapping = self
            .mapping
            .get(&name_small)
            .ok_or_else(|| PreprocessingErrorInner::ColumnNotFound(name.to_string()))?;

        let index = dataframe
            .get_column_index(name)
            .ok_or_else(|| PreprocessingErrorInner::ColumnNotFound(name.to_string()))?;

        let column = dataframe.column(name)?;

        let dtype = column.dtype();
        if !matches!(dtype, DataType::String) {
            return Err((PreprocessingErrorInner::InvalidColumnType {
                column: name.to_string(),
                expected: "String".to_string(),
                actual: dtype.to_string(),
            })
            .into());
        }

        let column_chunked = column.str().unwrap();

        let encoded: UInt32Chunked =
            column_chunked.apply_nonnull_values_generic(DataType::UInt32, |val| {
                let key: PlSmallStr = val.into();
                *mapping.get(&key).unwrap_or(&u32::MAX)
            });

        let series = encoded.into_series().with_name(name_small);
        dataframe.replace_column(index, Column::from(series))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::preprocessor::error::PreprocessingErrorInner;

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
        assert!(encoder.is_fitted());
        assert!(encoder.config().mapping.contains_key("color"));
        assert!(encoder.config().mapping.contains_key("size"));
    }

    #[test]
    fn test_label_fit_column_not_found() {
        let df = make_df();
        let mut encoder = LabelEncoder::new();
        assert!(matches!(
            encoder
                .fit(&df, &["nonexistent"])
                .map_err(|e| e.into_inner()),
            Err(PreprocessingErrorInner::ColumnNotFound(_))
        ));
    }

    #[test]
    fn test_label_fit_non_string_column() {
        let df = make_df();
        let mut encoder = LabelEncoder::new();
        assert!(matches!(
            encoder.fit(&df, &["weight"]).map_err(|e| e.into_inner()),
            Err(PreprocessingErrorInner::InvalidColumnType {
                column: _,
                expected: _,
                actual: _
            })
        ));
    }

    #[test]
    fn test_label_transform_not_fitted() {
        let mut df = make_df();
        let encoder = LabelEncoder::new();
        assert!(matches!(
            encoder.transform(&mut df).map_err(|e| e.into_inner()),
            Err(PreprocessingErrorInner::NotFitted)
        ));
    }

    #[test]
    fn test_label_transform_correct_values() {
        let mut df = make_df();
        let mut encoder = LabelEncoder::new();
        encoder.fit(&df, &["color"]).unwrap();
        encoder.transform(&mut df).unwrap();

        assert_eq!(*df.column("color").unwrap().dtype(), DataType::UInt32);

        let col = df.column("color").unwrap().u32().unwrap();
        assert_eq!(col.get(0), col.get(2));
    }

    #[test]
    fn test_label_unfitted_columns_unchanged() {
        let mut df = make_df();
        let original = df.column("size").unwrap().clone();
        let mut encoder = LabelEncoder::new();
        encoder.fit(&df, &["color"]).unwrap();
        encoder.transform(&mut df).unwrap();
        assert_eq!(df.column("size").unwrap(), &original);
    }

    #[test]
    fn test_label_fit_transform_end_to_end() {
        let mut df = make_df();
        let mut encoder = LabelEncoder::new();
        assert!(encoder.fit_transform(&mut df, &["color", "size"]).is_ok());
        assert!(encoder.is_fitted());
        assert_eq!(*df.column("color").unwrap().dtype(), DataType::UInt32);
        assert_eq!(*df.column("size").unwrap().dtype(), DataType::UInt32);
    }
}
