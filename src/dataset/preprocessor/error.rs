use polars::prelude::*;
use thiserror::Error;
use thiserror_context::{Context, impl_context};

#[derive(Debug, Error)]
pub enum PreprocessingErrorInner {
    #[error("column not found: {0}")]
    ColumnNotFound(String),

    #[error("invalid column type for '{column}': expected {expected}, got {actual}")]
    InvalidColumnType {
        column: String,
        expected: String,
        actual: String,
    },

    #[error("encoder has not been fitted")]
    NotFitted,

    #[error("{0}")]
    Other(String),
}

impl_context!(PreprocessingError(PreprocessingErrorInner));

impl From<PolarsError> for PreprocessingError {
    fn from(err: PolarsError) -> Self {
        PreprocessingErrorInner::Other(err.to_string()).into()
    }
}
