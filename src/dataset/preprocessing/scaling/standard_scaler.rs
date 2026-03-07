use log::error;
use polars::prelude::*;
use rayon::prelude::*;
use std::{collections::HashMap, sync::Mutex};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ScalerError {
    #[error("Scaler not fitted yet")]
    NotFitted,
    #[error("Column not found: {0}")]
    ColumnNotFound(#[from] PolarsError),
    #[error("Column is not numeric: {0}")]
    InvalidColumnType(String),
}

pub struct StandardScaler {
    stats: HashMap<String, (f64, f64)>,
    fitted: bool,
}

impl StandardScaler {
    pub fn new() -> Self {
        Self {
            stats: HashMap::new(),
            fitted: false,
        }
    }

    pub fn fit(&mut self, dataframe: &DataFrame, columns: &[&str]) -> Result<(), ScalerError> {
        let stats: Mutex<HashMap<String, (f64, f64)>> = Mutex::new(HashMap::new());

        match columns.par_iter().try_for_each(|&name| {
            let column = match dataframe.column(name) {
                Ok(c) => c,
                Err(e) => {
                    error!("Error searching column {} in dataframe: {}", name, e);
                    return Err(ScalerError::ColumnNotFound(e));
                }
            };
            if !matches!(
                column.dtype(),
                DataType::Float32
                    | DataType::Float64
                    | DataType::Int32
                    | DataType::Int64
                    | DataType::Int16
                    | DataType::UInt32
            ) {
                error!(
                    "Error converting column {} from {} to Float64",
                    name,
                    column.dtype()
                );
                return Err(ScalerError::InvalidColumnType(name.to_string()));
            }
            let column_f64 = column.cast(&DataType::Float64).unwrap();
            let mean = column_f64.f64().unwrap().mean().unwrap_or(0.0);
            let std = column_f64.f64().unwrap().std(1).unwrap_or(1.0);
            stats.lock().unwrap().insert(name.to_string(), (mean, std));
            Ok(())
        }) {
            Ok(()) => {
                self.stats = stats.into_inner().unwrap();
                self.fitted = true;
                return Ok(());
            }
            Err(e) => {
                error!("Unexpected error occured: {}", e);
                return Err(e);
            }
        };
    }

    pub fn transform(&self, dataframe: &mut DataFrame) -> Result<(), ScalerError> {
        if self.fitted == false {
            error!("Dataframe has not been fitted yet");
            return Err(ScalerError::NotFitted);
        }
        for (name, &(mean, std)) in &self.stats {
            let index = match dataframe.get_column_index(name.as_str()) {
                Some(i) => i,
                _ => {
                    let e = PolarsError::ColumnNotFound(name.clone().into());
                    error!("Error searching column {} in dataframe: {}", name, e);
                    return Err(ScalerError::ColumnNotFound(e));
                }
            };
            let column = match dataframe
                .column(name.as_str())
                .unwrap()
                .cast(&DataType::Float64)
            {
                Ok(c) => c,
                Err(e) => {
                    error!("Error converting column {} to Float64: {}", name, e);
                    return Err(ScalerError::InvalidColumnType(name.to_string()));
                }
            };
            let scaled_column = (column - mean) / std;
            match dataframe.replace_column(index, scaled_column) {
                Ok(_) => continue,
                Err(e) => {
                    error!("Unexpected error occured: {}", e);
                    return Err(ScalerError::ColumnNotFound(e));
                }
            }
        }
        Ok(())
    }

    pub fn fit_transform(
        &mut self,
        dataframe: &mut DataFrame,
        columns: &[&str],
    ) -> Result<(), ScalerError> {
        self.fit(dataframe, columns)?;
        self.transform(dataframe)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(matches!(result, Err(ScalerError::ColumnNotFound(_))));
    }

    #[test]
    fn test_fit_non_numeric_column() {
        let df = make_df();
        let mut scaler = StandardScaler::new();
        let result = scaler.fit(&df, &["name"]);
        assert!(matches!(result, Err(ScalerError::InvalidColumnType(_))));
    }

    #[test]
    fn test_transform_not_fitted() {
        let mut df = make_df();
        let scaler = StandardScaler::new();
        assert!(matches!(
            scaler.transform(&mut df),
            Err(ScalerError::NotFitted)
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
