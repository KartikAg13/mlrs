use colored::Colorize;
use thiserror::Error;

pub mod activator;
pub mod classifier;
pub mod linear;
pub mod optimizer;

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("Model is not fitted yet. Call .fit() first.")]
    NotFitted,

    #[error("Shape misatch. X has {0} rows, Y has {1} rows.")]
    ShapeMismatch(usize, usize),
}

impl ModelError {
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

    pub fn print_modifying(message: String) {
        println!("{}{}", "MODIFYING: ".green(), message.green());
    }
}
