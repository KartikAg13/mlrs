use polars::prelude::*;
use std::collections::HashMap;

use crate::dataset::preprocessing::{
    PreprocessingError,
    scaling::{Scaler, Scaling},
};

pub struct MinMaxConfig {
    pub feature_range: (f64, f64),
}

impl Default for MinMaxConfig {
    fn default() -> Self {
        Self {
            feature_range: (0.0, 1.0),
        }
    }
}

pub type MinMaxScaler = Scaler<MinMaxConfig>;

impl MinMaxScaler {
    pub fn new(feature_range: (f64, f64)) -> Self {
        Self {
            stats: HashMap::new(),
            fitted: false,
            config: MinMaxConfig { feature_range },
        }
    }
}

impl Scaling for MinMaxConfig {
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
        let min = column_chunked.min().unwrap_or(0.0);
        let max = column_chunked.max().unwrap_or(1.0);

        if (max - min).abs() < f64::EPSILON {
            PreprocessingError::print_warning(format!(
                "Column {} has min == max ({}). Column will be scaled to 0.0.",
                column.name(),
                min
            ));
        }

        (min, max)
    }

    fn scale_value(&self, column: Column, stats: (f64, f64)) -> Column {
        if (stats.1 - stats.0).abs() < f64::EPSILON {
            return column * 0.0;
        }

        (column - stats.0) / (stats.1 - stats.0) * (self.feature_range.1 - self.feature_range.0)
            + self.feature_range.0
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
    fn test_minmax_fit_basic() {
        let df = make_df();
        let mut scaler = MinMaxScaler::new((0.0, 1.0));
        assert!(scaler.fit(&df, &["age", "salary"]).is_ok());
        assert!(scaler.fitted);
        assert!(scaler.stats.contains_key("age"));
        assert!(scaler.stats.contains_key("salary"));
    }

    #[test]
    fn test_minmax_fit_column_not_found() {
        let df = make_df();
        let mut scaler = MinMaxScaler::new((0.0, 1.0));

        let result = scaler.fit(&df, &["nonexistent"]);
        assert!(
            result.is_ok(),
            "fit should succeed even if column is missing"
        );
        assert!(scaler.fitted);
        assert!(!scaler.stats.contains_key("nonexistent"));
    }

    #[test]
    fn test_minmax_fit_non_numeric_column() {
        let df = make_df();
        let mut scaler = MinMaxScaler::new((0.0, 1.0));

        let result = scaler.fit(&df, &["name"]);
        assert!(
            result.is_ok(),
            "fit should succeed with warning for non-numeric column"
        );
        assert!(scaler.fitted);
        assert!(scaler.stats.is_empty());
    }

    #[test]
    fn test_minmax_transform_not_fitted() {
        let mut df = make_df();
        let scaler = MinMaxScaler::new((0.0, 1.0));
        assert!(matches!(
            scaler.transform(&mut df),
            Err(PreprocessingError::NotFitted)
        ));
    }

    #[test]
    fn test_minmax_transform_range_0_1() {
        let mut df = make_df();
        let mut scaler = MinMaxScaler::new((0.0, 1.0));
        scaler.fit(&df, &["age", "salary"]).unwrap();
        scaler.transform(&mut df).unwrap();

        for col_name in ["age", "salary"] {
            let col = df
                .column(col_name)
                .unwrap()
                .cast(&DataType::Float64)
                .unwrap();
            let ca = col.f64().unwrap();
            assert!(
                (ca.min().unwrap() - 0.0).abs() < EPSILON,
                "{} min should be 0",
                col_name
            );
            assert!(
                (ca.max().unwrap() - 1.0).abs() < EPSILON,
                "{} max should be 1",
                col_name
            );
        }
    }

    #[test]
    fn test_minmax_transform_range_neg1_1() {
        let mut df = make_df();
        let mut scaler = MinMaxScaler::new((-1.0, 1.0));
        scaler.fit(&df, &["age"]).unwrap();
        scaler.transform(&mut df).unwrap();

        let col = df.column("age").unwrap().cast(&DataType::Float64).unwrap();
        let ca = col.f64().unwrap();
        assert!((ca.min().unwrap() - (-1.0)).abs() < EPSILON);
        assert!((ca.max().unwrap() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_minmax_unfitted_columns_unchanged() {
        let mut df = make_df();
        let original = df.column("name").unwrap().clone();
        let mut scaler = MinMaxScaler::new((0.0, 1.0));
        scaler.fit(&df, &["age"]).unwrap();
        scaler.transform(&mut df).unwrap();
        assert_eq!(df.column("name").unwrap(), &original);
    }

    #[test]
    fn test_minmax_fit_transform_end_to_end() {
        let mut df = make_df();
        let mut scaler = MinMaxScaler::new((0.0, 1.0));
        assert!(scaler.fit_transform(&mut df, &["age", "salary"]).is_ok());
        assert!(scaler.fitted);

        for col_name in ["age", "salary"] {
            let col = df
                .column(col_name)
                .unwrap()
                .cast(&DataType::Float64)
                .unwrap();
            let ca = col.f64().unwrap();
            assert!((ca.min().unwrap() - 0.0).abs() < EPSILON);
            assert!((ca.max().unwrap() - 1.0).abs() < EPSILON);
        }
    }
}
