use colored::Colorize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("Model is not fitted yet. Call .fit() first.")]
    NotFitted,

    #[error("Shape misatch. X has {0} rows, Y has {1} rows.")]
    ShapeMismatch(usize, usize),

    #[error("Invalid normal distribution parameter using input_size {0}.")]
    InvalidNormalDistribution(usize),

    #[error("File could not be created {0}.")]
    InvalidFilePath(String),

    #[error("Unexpected error occured. Please retry.")]
    UnexpectedError,
}

impl ModelError {
    pub fn print_error(&self) {
        eprintln!("{}{}", "ERROR: ".red().bold(), self.to_string().red());
    }

    pub fn print_warning(message: String) {
        eprintln!("{}{}", "WARNING: ".yellow().bold(), message.yellow());
    }

    pub fn print_modifying(message: String) {
        println!("{}{}", "MODIFICATION: ".green(), message.green());
    }
}
