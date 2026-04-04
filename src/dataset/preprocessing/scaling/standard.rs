use polars::prelude::*;
use std::collections::HashMap;

use crate::dataset::preprocessing::{
    PreprocessingError,
    scaling::{Scaler, Scaling},
};

pub struct StandardConfig;

pub type StandardScaler = Scaler<StandardConfig>;

impl StandardScaler {
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
            fitted: false,
            config: StandardConfig,
        }
    }
}

impl Scaling for StandardConfig {
    fn compute_stats(&self, column: Column) -> (f64, f64) {
        let column_chunked = match column.f64() {
            Ok(c) => c,
            Err(_) => {
                PreprocessingError::print_warning(format!(
                    "Column {} is not Float type.",
                    column.name()
                ));
                return (0.0, 0.0);
            }
        };
        let mean = column_chunked.mean().unwrap_or(0.0);
        let std = column_chunked.std(1).unwrap_or(1.0);
        (mean, std)
    }

    fn scale_value(&self, column: Column, stats: (f64, f64)) -> Column {
        (column - stats.0) / stats.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::preprocessing::scaling::PreprocessingError;

    fn make_df() -> DataFrame {
        df![
            "age"    => [10.0_f64, 20.0, 30.0],
            "salary" => [100.0_f64, 200.0, 300.0],
            "name"   => ["alice", "bob", "charlie"]
        ]
        .unwrap()
    }

    const EPSILON: f64 = 1e-4;

    #[test]
    fn test_fit_basic() {
        let df = make_df();
        let mut scaler = StandardScaler::new();
        assert!(scaler.fit(&df, &["age", "salary"]).is_ok());
        assert!(scaler.fitted);
        assert!(scaler.stats.contains_key("age"));
        assert!(scaler.stats.contains_key("salary"));
    }

    #[test]
    fn test_fit_column_not_found() {
        let df = make_df();
        let mut scaler = StandardScaler::new();

        let result = scaler.fit(&df, &["nonexistent"]);
        assert!(
            result.is_ok(),
            "fit should succeed with warning when column not found"
        );
        assert!(scaler.fitted);
        assert!(!scaler.stats.contains_key("nonexistent"));
    }

    #[test]
    fn test_fit_non_numeric_column() {
        let df = make_df();
        let mut scaler = StandardScaler::new();

        let result = scaler.fit(&df, &["name"]);
        assert!(
            result.is_ok(),
            "fit should succeed with warning for non-numeric column"
        );
        assert!(scaler.fitted);
        assert!(scaler.stats.is_empty());
    }

    #[test]
    fn test_transform_not_fitted() {
        let mut df = make_df();
        let scaler = StandardScaler::new();
        assert!(matches!(
            scaler.transform(&mut df),
            Err(PreprocessingError::NotFitted)
        ));
    }

    #[test]
    fn test_transform_correct_values() {
        let mut df = make_df();
        let mut scaler = StandardScaler::new();
        scaler.fit(&df, &["age", "salary"]).unwrap();
        scaler.transform(&mut df).unwrap();

        let age = df.column("age").unwrap().cast(&DataType::Float64).unwrap();
        let ca = age.f64().unwrap();
        let mean = ca.mean().unwrap();
        let std = ca.std(1).unwrap();

        assert!((mean).abs() < EPSILON, "mean should be ~0, got {}", mean);
        assert!((std - 1.0).abs() < EPSILON, "std should be ~1, got {}", std);
    }

    #[test]
    fn test_transform_unfitted_columns_unchanged() {
        let mut df = make_df();
        let original_name_col = df.column("name").unwrap().clone();
        let mut scaler = StandardScaler::new();
        scaler.fit(&df, &["age"]).unwrap();
        scaler.transform(&mut df).unwrap();
        assert_eq!(df.column("name").unwrap(), &original_name_col);
    }

    #[test]
    fn test_fit_transform_end_to_end() {
        let mut df = make_df();
        let mut scaler = StandardScaler::new();
        assert!(scaler.fit_transform(&mut df, &["age", "salary"]).is_ok());
        assert!(scaler.fitted);

        for col_name in ["age", "salary"] {
            let col = df
                .column(col_name)
                .unwrap()
                .cast(&DataType::Float64)
                .unwrap();
            let ca = col.f64().unwrap();
            assert!((ca.mean().unwrap()).abs() < EPSILON);
            assert!((ca.std(1).unwrap() - 1.0).abs() < EPSILON);
        }
    }
}
