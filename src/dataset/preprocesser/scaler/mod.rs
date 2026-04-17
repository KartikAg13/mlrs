pub use min_max::MinMaxScaler;
pub use standard::StandardScaler;

use polars::prelude::*;
use rayon::prelude::*;
use std::{collections::HashMap, sync::Mutex};

pub mod min_max;
pub mod standard;
use crate::dataset::preprocesser::{PreprocessingError, is_numeric};

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
    pub fn fit(
        &mut self,
        dataframe: &DataFrame,
        columns: &[&str],
    ) -> Result<(), PreprocessingError> {
        let stats: Mutex<HashMap<String, (f64, f64)>> = Mutex::new(HashMap::new());

        columns.par_iter().for_each(|&name| {
            let column = match dataframe.column(name) {
                Ok(c) => c,
                Err(_) => {
                    PreprocessingError::print_warning(format!(
                        "Column '{}' not found.",
                        name
                    ));
                    return;
                }
            };
            if !is_numeric(column.dtype()) {
                PreprocessingError::print_warning(format!(
                    "Column '{}' is not the expected type. Expected datatype: '{}', Found datatype: '{}'.",
                    name,
                    "Float / Int",
                    column.dtype()
                ));
                return ;
            }
            let column_f64 = column.cast(&DataType::Float64).unwrap();
            let (a, b) = self.config.compute_stats(column_f64);
            stats.lock().unwrap().insert(name.to_string(), (a, b));
        });
        self.stats = stats.into_inner().unwrap();
        self.fitted = true;
        Ok(())
    }

    pub fn transform(&self, dataframe: &mut DataFrame) -> Result<(), PreprocessingError> {
        if !self.fitted {
            let error = PreprocessingError::NotFitted;
            error.print_error();
            return Err(error);
        }
        for (name, &(a, b)) in &self.stats {
            let index = match dataframe.get_column_index(name.as_str()) {
                Some(i) => i,
                _ => {
                    let error = PreprocessingError::ColumnNotFound(name.to_string());
                    error.print_error();
                    return Err(error);
                }
            };
            let column = match dataframe
                .column(name.as_str())
                .unwrap()
                .cast(&DataType::Float64)
            {
                Ok(c) => c,
                Err(_) => {
                    let error = PreprocessingError::InvalidColumnType(
                        name.to_string(),
                        "Float / Int".to_string(),
                        dataframe.column(name)?.dtype().to_string(),
                    );
                    error.print_error();
                    return Err(error);
                }
            };
            let scaled_column = self.config.scale_value(column, (a, b));
            match dataframe.replace_column(index, scaled_column) {
                Ok(_) => continue,
                Err(_) => {
                    let error = PreprocessingError::ColumnNotFound(name.to_string());
                    error.print_error();
                    return Err(error);
                }
            }
        }
        Ok(())
    }

    pub fn fit_transform(
        &mut self,
        dataframe: &mut DataFrame,
        columns: &[&str],
    ) -> Result<(), PreprocessingError> {
        self.fit(dataframe, columns)?;
        self.transform(dataframe)
    }
}
