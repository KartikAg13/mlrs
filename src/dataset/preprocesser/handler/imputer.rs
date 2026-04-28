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
pub enum ImputerStrategy {
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
/// use mlrs::dataset::preprocesser::handler::{SimpleImputer, ImputerStrategy};
/// use polars::prelude::*;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut df = df![
///     "age"    => [Some(25.0), None, Some(35.0), Some(40.0)],
///     "income" => [Some(50000.0), Some(60000.0), None, Some(80000.0)]
/// ]?;
///
/// SimpleImputer::fill_null(&mut df, &[
///     ("age", ImputerStrategy::Mean),
///     ("income", ImputerStrategy::Zero),
/// ])?;
///
/// // Nulls have been replaced according to the chosen strategies
/// # Ok(())
/// # }
/// ```
///
/// **Using constant and fill strategies:**
/// ```
/// # use mlrs::dataset::preprocesser::handler::{SimpleImputer, ImputerStrategy};
/// # use polars::prelude::*;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut df = df![
///     "score" => [Some(85.0), None, Some(92.0), None],
///     "category" => [Some("A"), None, Some("B"), Some("A")]
/// ]?;
///
/// SimpleImputer::fill_null(&mut df, &[
///     ("score", ImputerStrategy::Mean),
///     ("category", ImputerStrategy::ForwardFill),
/// ])?;
/// # Ok(())
/// # }
/// ```
///
/// **Edge case — column with all nulls:**
/// ```
/// # use mlrs::dataset::preprocesser::handler::{SimpleImputer, ImputerStrategy};
/// # use polars::prelude::*;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut df = df![
///     "all_null" => [Option::<f64>::None, None, None]
/// ]?;
///
/// // Mean of empty column falls back gracefully
/// SimpleImputer::fill_null(&mut df, &[("all_null", ImputerStrategy::Mean)])?;
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
        columns: &[(&str, ImputerStrategy)],
    ) -> Result<(), PreprocessingError> {
        if columns.is_empty() {
            return Ok(());
        }
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
                ImputerStrategy::Mean => FillNullStrategy::Mean,
                ImputerStrategy::MinimumValue => FillNullStrategy::Min,
                ImputerStrategy::MaximumValue => FillNullStrategy::Max,
                ImputerStrategy::Zero => FillNullStrategy::Zero,
                ImputerStrategy::One => FillNullStrategy::One,
                ImputerStrategy::ForwardFill => FillNullStrategy::Forward(None),
                ImputerStrategy::BackwardFill => FillNullStrategy::Backward(None),
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
        SimpleImputer::fill_null(&mut df, &[("age", ImputerStrategy::Mean)]).unwrap();
        let col = df.column("age").unwrap().f64().unwrap();
        assert_eq!(col.null_count(), 0);
        // mean of [10, 30, 50] = 30.0
        assert!((col.get(1).unwrap() - 30.0).abs() < 1e-6);
        assert!((col.get(3).unwrap() - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_min_fills_nulls() {
        let mut df = make_df();
        SimpleImputer::fill_null(&mut df, &[("salary", ImputerStrategy::MinimumValue)]).unwrap();
        let col = df.column("salary").unwrap().f64().unwrap();
        assert_eq!(col.null_count(), 0);
        assert!((col.get(2).unwrap() - 100.0).abs() < 1e-6);
        assert!((col.get(4).unwrap() - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_fills_nulls() {
        let mut df = make_df();
        SimpleImputer::fill_null(&mut df, &[("salary", ImputerStrategy::MaximumValue)]).unwrap();
        let col = df.column("salary").unwrap().f64().unwrap();
        assert_eq!(col.null_count(), 0);
        assert!((col.get(2).unwrap() - 400.0).abs() < 1e-6);
        assert!((col.get(4).unwrap() - 400.0).abs() < 1e-6);
    }

    #[test]
    fn test_zero_fills_nulls() {
        let mut df = make_df();
        SimpleImputer::fill_null(&mut df, &[("age", ImputerStrategy::Zero)]).unwrap();
        let col = df.column("age").unwrap().f64().unwrap();
        assert_eq!(col.null_count(), 0);
        assert!((col.get(1).unwrap() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_one_fills_nulls() {
        let mut df = make_df();
        SimpleImputer::fill_null(&mut df, &[("age", ImputerStrategy::One)]).unwrap();
        let col = df.column("age").unwrap().f64().unwrap();
        assert_eq!(col.null_count(), 0);
        assert!((col.get(1).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_forward_fill() {
        let mut df = make_df();
        SimpleImputer::fill_null(&mut df, &[("age", ImputerStrategy::ForwardFill)]).unwrap();
        let col = df.column("age").unwrap().f64().unwrap();
        assert!((col.get(1).unwrap() - 10.0).abs() < 1e-6);
        assert!((col.get(3).unwrap() - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_backward_fill() {
        let mut df = make_df();
        SimpleImputer::fill_null(&mut df, &[("age", ImputerStrategy::BackwardFill)]).unwrap();
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
                ("age", ImputerStrategy::Mean),
                ("salary", ImputerStrategy::Zero),
                ("score", ImputerStrategy::ForwardFill),
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
            &[
                ("age", ImputerStrategy::Mean),
                ("nonexistent", ImputerStrategy::Zero),
            ],
        );
        assert!(result.is_ok());
        assert_eq!(df.column("age").unwrap().null_count(), 0);
    }

    #[test]
    fn test_no_nulls_is_noop() {
        let mut df = df!["x" => [1.0_f64, 2.0, 3.0]].unwrap();
        let original = df.column("x").unwrap().clone();
        SimpleImputer::fill_null(&mut df, &[("x", ImputerStrategy::Mean)]).unwrap();
        assert_eq!(df.column("x").unwrap(), &original);
    }
}
