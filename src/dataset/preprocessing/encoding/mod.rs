pub use label::LabelEncoder;
pub use one_hot::OneHotEncoder;

use log::error;
use polars::prelude::*;
use thiserror::Error;

pub mod label;
pub mod one_hot;

#[derive(Debug, Error)]
pub enum EncoderError {
    #[error("Encoder not fitted yet")]
    NotFitted,
    #[error("Column not found: {0}")]
    ColumnNotFound(#[from] PolarsError),
    #[error("Column is not string type: {0}")]
    InvalidColumnType(String),
}

pub trait EncodingStrategy {
    fn compute_encoding(&mut self, column: &Column) -> Result<(), EncoderError>;
    fn apply_encoding(&self, dataframe: &mut DataFrame, name: &str) -> Result<(), EncoderError>;
}

pub struct Encoder<T: EncodingStrategy> {
    pub fitted: bool,
    pub config: T,
}

impl<T: EncodingStrategy> Encoder<T> {
    pub fn fit(&mut self, dataframe: &DataFrame, columns: &[&str]) -> Result<(), EncoderError> {
        for &name in columns {
            let column = match dataframe.column(name) {
                Ok(c) => c.clone(),
                Err(e) => {
                    error!("Error searching column {} in dataframe: {}", name, e);
                    return Err(EncoderError::ColumnNotFound(e));
                }
            };
            if !matches!(column.dtype(), DataType::String) {
                return Err(EncoderError::InvalidColumnType(name.to_string()));
            }
            self.config.compute_encoding(&column)?;
        }
        self.fitted = true;
        Ok(())
    }

    pub fn transform(
        &self,
        dataframe: &mut DataFrame,
        columns: &[&str],
    ) -> Result<(), EncoderError> {
        if !self.fitted {
            return Err(EncoderError::NotFitted);
        }
        for &name in columns {
            self.config.apply_encoding(dataframe, name)?;
        }
        Ok(())
    }

    pub fn fit_transform(
        &mut self,
        dataframe: &mut DataFrame,
        columns: &[&str],
    ) -> Result<(), EncoderError> {
        self.fit(dataframe, columns)?;
        self.transform(dataframe, columns)
    }
}
