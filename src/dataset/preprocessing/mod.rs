pub mod encoding;
pub mod handling;
pub mod scaling;

use colored::Colorize;
use polars::prelude::*;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PreprocessingError {
    #[error("Column not fitted yet. Call .fit() or .fit_transform() before .transform()")]
    NotFitted,

    #[error("Column '{0}' not found")]
    ColumnNotFound(String),

    #[error(
        "Column '{0}' is not the expected type. Expected datatype: '{1}', Found datatype: '{2}'"
    )]
    InvalidColumnType(String, String, String),

    #[error(
        "Column '{0}' has unseen label '{1}'. Only labels found during .fit() or .fit_transform() are valid"
    )]
    UnseenLabel(String, String),

    #[error("Unexpected PolarsError: {0}")]
    PolarsError(#[from] PolarsError),
}

impl PreprocessingError {
    pub fn print_error(&self) {
        eprintln!("{}{}", "ERROR: ".red().bold(), self.to_string().red());
    }

    pub fn print_warning(message: String) {
        eprintln!(
            "{}{}{}",
            "WARNING: ".yellow().bold(),
            message.yellow(),
            " Skipping!".yellow()
        );
    }
}

pub fn is_numeric(datatype: &DataType) -> bool {
    matches!(
        datatype,
        DataType::Float32
            | DataType::Float64
            | DataType::Int32
            | DataType::Int64
            | DataType::Int16
            | DataType::UInt32
    )
}

pub fn is_categorical(datatype: &DataType) -> bool {
    matches!(datatype, DataType::String)
}
