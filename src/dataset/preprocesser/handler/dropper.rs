//! Row and column dropping utilities for DataFrame preprocessing.
//!
//! [`Dropper`] provides two main operations:
//! - `drop_rows`: Remove rows that contain missing values (nulls) based on
//!   selected columns using `Any` (drop if *any* column is null) or `All`
//!   (drop only if *all* columns are null) semantics.
//! - `drop_columns`: Remove entire columns by name.
//!
//! Both methods operate **in-place** on a `polars::DataFrame` and integrate
//! seamlessly into the ML preprocessing pipeline (typically after loading
//! and before imputation/encoding/scaling).

use crate::dataset::preprocesser::error::PreprocessingError;
use polars::prelude::*;

/// Strategy for determining which rows to drop based on null values.
///
/// This enum controls the logical combination when multiple columns are provided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropperStrategy {
    /// Drop a row if **any** of the specified columns contains a null value.
    ///
    /// This is the most common strategy (logical OR on null masks).
    Any,
    /// Drop a row only if **all** of the specified columns contain null values.
    ///
    /// Useful for keeping rows that have at least one valid value among the selected columns.
    All,
}

/// Stateless dropper for removing rows or columns from a DataFrame.
///
/// This struct is intentionally stateless (no `.fit()` or internal state)
/// because dropping operations are purely declarative and compute masks on-the-fly.
///
/// # Examples
///
/// **Basic row dropping with `Any` strategy (most common use case):**
/// ```
/// use mlrs::dataset::preprocesser::handler::{Dropper, DropperStrategy};
/// use polars::prelude::*;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut df = df![
///     "age" => [Some(25.0), None, Some(35.0), Some(40.0)],
///     "income" => [Some(50000.0), Some(60000.0), None, Some(80000.0)],
///     "score" => [Some(85.0), Some(90.0), Some(92.0), None]
/// ]?;
///
/// // Drop rows where *any* of age or income is null
/// Dropper::drop_rows(&mut df, &[
///     ("age", DropperStrategy::Any),
///     ("income", DropperStrategy::Any),
/// ])?;
///
/// assert_eq!(df.height(), 2);
/// # Ok(())
/// # }
/// ```
///
/// **Column dropping (cleaning unnecessary features):**
/// ```
/// # use mlrs::dataset::preprocesser::handler::Dropper;
/// # use polars::prelude::*;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut df = df![
///     "id" => [1, 2, 3],
///     "loan_amnt" => [5000.0, 2500.0, 10000.0],
///     "drop_me" => ["A", "B", "C"]
/// ]?;
///
/// Dropper::drop_columns(&mut df, &["id", "drop_me"])?;
///
/// assert!(df.get_column_index("id").is_none());
/// assert!(df.get_column_index("drop_me").is_none());
/// assert!(df.get_column_index("loan_amnt").is_some());
/// # Ok(())
/// # }
/// ```
///
/// **Edge case — `All` strategy with mixed null patterns:**
/// ```
/// # use mlrs::dataset::preprocesser::handler::{Dropper, DropperStrategy};
/// # use polars::prelude::*;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut df = df![
///     "feature_a" => [Some(1.0), None, None, Some(4.0)],
///     "feature_b" => [None, Some(2.0), None, Some(5.0)]
/// ]?;
///
/// // Drop only rows where *both* feature_a and feature_b are null
/// Dropper::drop_rows(&mut df, &[
///     ("feature_a", DropperStrategy::All),
///     ("feature_b", DropperStrategy::All),
/// ])?;
///
/// assert_eq!(df.height(), 3); // Keeps row with (None, Some(2.0))
/// # Ok(())
/// # }
/// ```
pub struct Dropper;

impl Dropper {
    /// Drops rows based on null values in the specified columns using the given strategy.
    ///
    /// - `DropperStrategy::Any`: Drops a row if **any** selected column is null (most common).
    /// - `DropperStrategy::All`: Drops a row only if **all** selected columns are null.
    ///
    /// Columns that do not exist emit a warning and are skipped.
    /// If no valid columns remain, the operation is a no-op.
    ///
    /// # Arguments
    ///
    /// * `dataframe` - Mutable reference to the DataFrame to modify **in-place**.
    /// * `columns` - Slice of `(column_name, strategy)` pairs that define the dropping logic.
    pub fn drop_rows(
        dataframe: &mut DataFrame,
        columns: &[(&str, DropperStrategy)],
    ) -> Result<(), PreprocessingError> {
        if columns.is_empty() {
            return Ok(());
        }
        let valid: Vec<(&Series, DropperStrategy)> = columns
            .iter()
            .filter_map(|&(name, strategy)| match dataframe.column(name) {
                Ok(c) => Some((c.as_materialized_series(), strategy)),
                Err(_) => {
                    PreprocessingError::print_warning(format!("Column '{}' not found.", name));
                    None
                }
            })
            .collect();
        if valid.is_empty() {
            return Ok(());
        }
        let mut mask: BooleanChunked = valid[0].0.is_not_null();
        for (col, strategy) in &valid[1..] {
            let col_mask = col.is_not_null();
            mask = match strategy {
                DropperStrategy::Any => mask & col_mask,
                DropperStrategy::All => mask | col_mask,
            };
        }
        *dataframe = dataframe.filter(&mask)?;
        Ok(())
    }

    /// Drops the specified columns from the DataFrame.
    ///
    /// Columns that do not exist emit a warning and are skipped.
    /// Uses `drop_in_place` for efficiency (avoids unnecessary allocations).
    ///
    /// # Arguments
    ///
    /// * `dataframe` - Mutable reference to the DataFrame to modify **in-place**.
    /// * `columns` - Slice of column names to remove.
    pub fn drop_columns(
        dataframe: &mut DataFrame,
        columns: &[&str],
    ) -> Result<(), PreprocessingError> {
        if columns.is_empty() {
            return Ok(());
        }
        let to_drop: Vec<&str> = columns
            .iter()
            .filter_map(|&name| match dataframe.column(name) {
                Ok(_) => Some(name),
                Err(_) => {
                    PreprocessingError::print_warning(format!("Column '{}' not found.", name));
                    None
                }
            })
            .collect();
        for name in to_drop {
            let _ = dataframe.drop_in_place(name); // ignore error if column already gone
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_df() -> DataFrame {
        df![
            "a" => [Some(1.0_f64), None, Some(3.0), None],
            "b" => [Some(4.0_f64), Some(5.0), None, None],
            "c" => [Some(7.0_f64), Some(8.0), Some(9.0), Some(10.0)]
        ]
        .unwrap()
    }

    #[test]
    fn test_drop_rows_any_removes_rows_with_any_null() {
        let mut df = make_df();
        Dropper::drop_rows(
            &mut df,
            &[("a", DropperStrategy::Any), ("b", DropperStrategy::Any)],
        )
        .unwrap();
        assert_eq!(df.height(), 1);
    }

    #[test]
    fn test_drop_rows_any_single_column() {
        let mut df = make_df();
        Dropper::drop_rows(&mut df, &[("c", DropperStrategy::Any)]).unwrap();
        assert_eq!(df.height(), 4);
    }

    #[test]
    fn test_drop_rows_all_removes_only_fully_null_rows() {
        let mut df = make_df();
        Dropper::drop_rows(
            &mut df,
            &[("a", DropperStrategy::All), ("b", DropperStrategy::All)],
        )
        .unwrap();
        assert_eq!(df.height(), 3);
    }

    #[test]
    fn test_drop_rows_all_no_fully_null_rows() {
        let mut df = make_df();
        Dropper::drop_rows(
            &mut df,
            &[("a", DropperStrategy::All), ("c", DropperStrategy::All)],
        )
        .unwrap();
        assert_eq!(df.height(), 4);
    }

    #[test]
    fn test_drop_columns_removes_specified_columns() {
        let mut df = make_df();
        Dropper::drop_columns(&mut df, &["a", "b"]).unwrap();
        assert_eq!(df.width(), 1);
        assert!(df.column("c").is_ok());
    }

    #[test]
    fn test_drop_columns_no_nulls_unchanged() {
        let mut df = make_df();
        let original_width = df.width();
        Dropper::drop_columns(&mut df, &["c"]).unwrap();
        assert_eq!(df.width(), original_width - 1);
    }

    #[test]
    fn test_drop_columns_subset_only() {
        let mut df = make_df();
        Dropper::drop_columns(&mut df, &["a"]).unwrap();
        assert!(df.column("a").is_err());
        assert!(df.column("b").is_ok());
    }

    #[test]
    fn test_drop_rows_missing_column_warns_and_continues() {
        let mut df = make_df();
        let result = Dropper::drop_rows(
            &mut df,
            &[
                ("a", DropperStrategy::Any),
                ("nonexistent", DropperStrategy::Any),
            ],
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_drop_columns_missing_column_warns_and_continues() {
        let mut df = make_df();
        let result = Dropper::drop_columns(&mut df, &["nonexistent", "c"]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_drop_rows_empty_columns_slice_is_noop() {
        let mut df = make_df();
        let original_height = df.height();
        Dropper::drop_rows(&mut df, &[]).unwrap();
        assert_eq!(df.height(), original_height);
    }

    #[test]
    fn test_drop_columns_empty_slice_is_noop() {
        let mut df = make_df();
        let original_width = df.width();
        Dropper::drop_columns(&mut df, &[]).unwrap();
        assert_eq!(df.width(), original_width);
    }
}
