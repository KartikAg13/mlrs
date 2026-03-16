pub use min_max::MinMaxScaler;
pub use standard::StandardScaler;

use log::error;
use polars::prelude::*;
use rayon::prelude::*;
use std::{collections::HashMap, sync::Mutex};
use thiserror::Error;

pub mod min_max;
pub mod standard;

#[derive(Debug, Error)]
pub enum ScalerError {
    #[error("Scaler not fitted yet")]
    NotFitted,
    #[error("Column not found: {0}")]
    ColumnNotFound(#[from] PolarsError),
    #[error("Column is not numeric: {0}")]
    InvalidColumnType(String),
    #[error("Invalid feature range: min must be less than max")]
    InvalidFeatureRange,
}

pub trait Scaling: Sync {
    fn compute_stats(&self, column: Column) -> (f64, f64);
    fn scale_value(&self, column: Column, stats: (f64, f64)) -> Column;
}

pub struct Scaler<T: Scaling> {
    pub stats: HashMap<String, (f64, f64)>,
    pub fitted: bool,
    pub config: T,
}

impl<T: Scaling + Sync> Scaler<T> {
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
            let (a, b) = self.config.compute_stats(column_f64);
            stats.lock().unwrap().insert(name.to_string(), (a, b));
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
        if !self.fitted {
            error!("Dataframe has not been fitted yet");
            return Err(ScalerError::NotFitted);
        }
        for (name, &(a, b)) in &self.stats {
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
            let scaled_column = self.config.scale_value(column, (a, b));
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
