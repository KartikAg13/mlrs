use colored::Colorize;
use polars::prelude::*;
use thiserror::Error;

/// Errors that can occur during preprocessing operations.
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
    /// Prints the error in red with a bold "ERROR:" prefix.
    pub fn print_error(&self) {
        eprintln!("{}{}", "ERROR: ".red().bold(), self.to_string().red());
    }

    /// Prints a warning in yellow.
    pub fn print_warning(message: String) {
        eprintln!(
            "{}{}{}",
            "WARNING: ".yellow().bold(),
            message.yellow(),
            " Skipping!".yellow()
        );
    }
}
