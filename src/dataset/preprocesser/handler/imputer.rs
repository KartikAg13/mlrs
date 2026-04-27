//! Simple imputation strategies for missing values.
//!
//! [`SimpleImputer`] provides a convenient way to fill null values in
//! selected columns using common statistical or constant strategies.
//! It works directly on `polars::DataFrame` and integrates well with
//! the rest of the preprocessing pipeline.

use polars::prelude::*;

use crate::dataset::preprocesser::error::PreprocessingError;

/// Imputation strategy for missing values.
///
/// These strategies map directly to Polars' [`FillNullStrategy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Replace nulls with the mean of the column (numeric only).
    Mean,
    /// Replace nulls with 0.
    Zero,
    /// Replace nulls with 1.
    One,
    /// Replace nulls with the minimum value of the column.
    MinimumValue,
    /// Replace nulls with the maximum value of the column.
    MaximumValue,
    /// Forward fill (last observation carried forward).
    ForwardFill,
    /// Backward fill (next observation carried backward).
    BackwardFill,
}

/// Simple imputer for filling missing values in a DataFrame.
///
/// This struct is intentionally stateless (no `.fit()` needed) because
/// most strategies compute statistics on-the-fly during filling.
///
/// # Examples
///
/// **Basic usage with multiple strategies:**
/// ```
/// use mlrs::dataset::preprocesser::handler::{SimpleImputer, Strategy};
/// use polars::prelude::*;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut df = df![
///     "age"    => [Some(25.0), None, Some(35.0), Some(40.0)],
///     "income" => [Some(50000.0), Some(60000.0), None, Some(80000.0)]
/// ]?;
///
/// SimpleImputer::fill_null(&mut df, &[
///     ("age", Strategy::Mean),
///     ("income", Strategy::Zero),
/// ])?;
///
/// // Nulls have been replaced according to the chosen strategies
/// # Ok(())
/// # }
/// ```
///
/// **Using constant and fill strategies:**
/// ```
/// # use mlrs::dataset::preprocesser::handler::{SimpleImputer, Strategy};
/// # use polars::prelude::*;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut df = df![
///     "score" => [Some(85.0), None, Some(92.0), None],
///     "category" => [Some("A"), None, Some("B"), Some("A")]
/// ]?;
///
/// SimpleImputer::fill_null(&mut df, &[
///     ("score", Strategy::Mean),
///     ("category", Strategy::ForwardFill),
/// ])?;
/// # Ok(())
/// # }
/// ```
///
/// **Edge case — column with all nulls:**
/// ```
/// # use mlrs::dataset::preprocesser::handler::{SimpleImputer, Strategy};
/// # use polars::prelude::*;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut df = df![
///     "all_null" => [Option::<f64>::None, None, None]
/// ]?;
///
/// // Mean of empty column falls back gracefully
/// SimpleImputer::fill_null(&mut df, &[("all_null", Strategy::Mean)])?;
/// # Ok(())
/// # }
/// ```
pub struct SimpleImputer;

impl SimpleImputer {
    /// Fills null values in the specified columns using the given strategies.
    ///
    /// Columns that do not exist will emit a warning and be skipped.
    /// If a Polars error occurs during filling, it is converted to
    /// [`PreprocessingError`] and the operation aborts.
    ///
    /// # Arguments
    ///
    /// * `dataframe` - Mutable reference to the DataFrame to modify in-place.
    /// * `columns` - Slice of `(column_name, strategy)` pairs.
    pub fn fill_null(
        dataframe: &mut DataFrame,
        columns: &[(&str, Strategy)],
    ) -> Result<(), PreprocessingError> {
        for (name, strategy) in columns {
            let index = match dataframe.get_column_index(name) {
                Some(i) => i,
                None => {
                    PreprocessingError::print_warning(format!("Column '{}' not found.", name));
                    continue;
                }
            };

            let column = dataframe.column(name)?;

            let strategy = match strategy {
                Strategy::Mean => FillNullStrategy::Mean,
                Strategy::MinimumValue => FillNullStrategy::Min,
                Strategy::MaximumValue => FillNullStrategy::Max,
                Strategy::Zero => FillNullStrategy::Zero,
                Strategy::One => FillNullStrategy::One,
                Strategy::ForwardFill => FillNullStrategy::Forward(None),
                Strategy::BackwardFill => FillNullStrategy::Backward(None),
            };

            let filled = match column.fill_null(strategy) {
                Ok(c) => c,
                Err(e) => {
                    let error = PreprocessingError::PolarsError(e);
                    error.print_error();
                    return Err(error);
                }
            };

            match dataframe.replace_column(index, filled) {
                Ok(d) => d,
                Err(e) => {
                    let error = PreprocessingError::PolarsError(e);
                    error.print_error();
                    return Err(error);
                }
            };
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_df() -> DataFrame {
        df![
            "age"    => [Some(10.0_f64), None, Some(30.0), None, Some(50.0)],
            "salary" => [Some(100.0_f64), Some(200.0), None, Some(400.0), None],
            "score"  => [Some(1.0_f64), Some(1.0), None, Some(3.0), None]
        ]
        .unwrap()
    }

    #[test]
    fn test_mean_fills_nulls() {
        let mut df = make_df();
        SimpleImputer::fill_null(&mut df, &[("age", Strategy::Mean)]).unwrap();
        let col = df.column("age").unwrap().f64().unwrap();
        assert_eq!(col.null_count(), 0);
        // mean of [10, 30, 50] = 30.0
        assert!((col.get(1).unwrap() - 30.0).abs() < 1e-6);
        assert!((col.get(3).unwrap() - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_min_fills_nulls() {
        let mut df = make_df();
        SimpleImputer::fill_null(&mut df, &[("salary", Strategy::MinimumValue)]).unwrap();
        let col = df.column("salary").unwrap().f64().unwrap();
        assert_eq!(col.null_count(), 0);
        assert!((col.get(2).unwrap() - 100.0).abs() < 1e-6);
        assert!((col.get(4).unwrap() - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_fills_nulls() {
        let mut df = make_df();
        SimpleImputer::fill_null(&mut df, &[("salary", Strategy::MaximumValue)]).unwrap();
        let col = df.column("salary").unwrap().f64().unwrap();
        assert_eq!(col.null_count(), 0);
        assert!((col.get(2).unwrap() - 400.0).abs() < 1e-6);
        assert!((col.get(4).unwrap() - 400.0).abs() < 1e-6);
    }

    #[test]
    fn test_zero_fills_nulls() {
        let mut df = make_df();
        SimpleImputer::fill_null(&mut df, &[("age", Strategy::Zero)]).unwrap();
        let col = df.column("age").unwrap().f64().unwrap();
        assert_eq!(col.null_count(), 0);
        assert!((col.get(1).unwrap() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_one_fills_nulls() {
        let mut df = make_df();
        SimpleImputer::fill_null(&mut df, &[("age", Strategy::One)]).unwrap();
        let col = df.column("age").unwrap().f64().unwrap();
        assert_eq!(col.null_count(), 0);
        assert!((col.get(1).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_forward_fill() {
        let mut df = make_df();
        SimpleImputer::fill_null(&mut df, &[("age", Strategy::ForwardFill)]).unwrap();
        let col = df.column("age").unwrap().f64().unwrap();
        assert!((col.get(1).unwrap() - 10.0).abs() < 1e-6);
        assert!((col.get(3).unwrap() - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_backward_fill() {
        let mut df = make_df();
        SimpleImputer::fill_null(&mut df, &[("age", Strategy::BackwardFill)]).unwrap();
        let col = df.column("age").unwrap().f64().unwrap();
        assert!((col.get(1).unwrap() - 30.0).abs() < 1e-6);
        assert!((col.get(3).unwrap() - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_per_column_different_strategies() {
        let mut df = make_df();
        SimpleImputer::fill_null(
            &mut df,
            &[
                ("age", Strategy::Mean),
                ("salary", Strategy::Zero),
                ("score", Strategy::ForwardFill),
            ],
        )
        .unwrap();
        assert_eq!(df.column("age").unwrap().null_count(), 0);
        assert_eq!(df.column("salary").unwrap().null_count(), 0);
        assert_eq!(df.column("score").unwrap().null_count(), 0);
    }

    #[test]
    fn test_missing_column_warns_and_continues() {
        let mut df = make_df();
        let result = SimpleImputer::fill_null(
            &mut df,
            &[("age", Strategy::Mean), ("nonexistent", Strategy::Zero)],
        );
        assert!(result.is_ok());
        assert_eq!(df.column("age").unwrap().null_count(), 0);
    }

    #[test]
    fn test_no_nulls_is_noop() {
        let mut df = df!["x" => [1.0_f64, 2.0, 3.0]].unwrap();
        let original = df.column("x").unwrap().clone();
        SimpleImputer::fill_null(&mut df, &[("x", Strategy::Mean)]).unwrap();
        assert_eq!(df.column("x").unwrap(), &original);
    }
}
