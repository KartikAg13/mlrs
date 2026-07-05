use polars::prelude::*;
use std::path::PathBuf;
use thiserror::Error;
use thiserror_context::{Context, impl_context};

#[derive(Debug, Error)]
pub enum DatasetErrorInner {
    #[error("CSV file '{path}' does not exist")]
    FileNotFound { path: PathBuf },

    #[error("CSV file '{path}' is empty")]
    EmptyFile { path: PathBuf },

    #[error("Failed to parse CSV '{path}'")]
    Parse {
        path: PathBuf,

        #[source]
        source: PolarsError,
    },
}

impl_context!(DatasetError(DatasetErrorInner));
