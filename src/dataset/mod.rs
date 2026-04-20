//! Dataset loading utilities for ML pipelines.
//!
//! This module provides a convenient and configurable way to load CSV files into
//! [`polars::prelude::DataFrame`] with sensible defaults optimized for machine learning workloads.
//!
//! # Quick Start
//!
//! ```
//! use mlrs::dataset::read_csv;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let filepath = "tests/fixtures/sample.csv";
//! let df = read_csv(filepath);
//! println!("Loaded dataset with {} rows and {} columns", df.height(), df.width());
//! # Ok(())
//! # }
//! ```

use std::path::PathBuf;

use polars::prelude::*;

pub mod preprocesser;

/// Configuration builder for reading CSV files.
///
/// This struct allows fine-grained control over how Polars reads CSV files.
/// It uses the builder pattern for ergonomic configuration.
pub struct CSVConfig {
    filepath: PathBuf,
    has_header: Option<bool>,
    rechunk: Option<bool>,
    low_memory: Option<bool>,
    n_threads: Option<usize>,
    chunk_size: Option<usize>,
    ignore_errors: Option<bool>,
}

impl CSVConfig {
    /// Creates a new `CSVConfig` with sensible defaults for ML datasets.
    ///
    /// # Defaults
    ///
    /// - `has_header`: `true`
    /// - `rechunk`: `false`
    /// - `low_memory`: `false`
    /// - `chunk_size`: 524288 (512 KiB)
    /// - `ignore_errors`: `false`
    ///
    /// # Examples
    ///
    /// **Basic usage:**
    /// ```
    /// use mlrs::dataset::CSVConfig;
    ///
    /// let config = CSVConfig::new("data/train.csv");
    /// ```
    ///
    /// **With custom options:**
    /// ```
    /// # use mlrs::dataset::CSVConfig;
    /// let config = CSVConfig::new("data/large.csv")
    ///     .with_has_header(true)
    ///     .with_n_threads(8)
    ///     .with_chunk_size(1 << 20)
    ///     .with_ignore_errors(false);
    /// ```
    ///
    /// **Edge case — minimal configuration (no header):**
    /// ```
    /// # use mlrs::dataset::CSVConfig;
    /// let config = CSVConfig::new("data/no_header.csv")
    ///     .with_has_header(false)
    ///     .with_ignore_errors(true);
    /// ```
    pub fn new(filepath: impl Into<PathBuf>) -> Self {
        Self {
            filepath: filepath.into(),
            has_header: Some(true),
            rechunk: Some(false),
            low_memory: Some(false),
            n_threads: None,
            chunk_size: Some(1 << 19),
            ignore_errors: Some(false),
        }
    }

    pub fn with_has_header(mut self, has_header: bool) -> Self {
        self.has_header = Some(has_header);
        self
    }

    pub fn with_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = Some(rechunk);
        self
    }

    pub fn with_low_memory(mut self, low_memory: bool) -> Self {
        self.low_memory = Some(low_memory);
        self
    }

    pub fn with_n_threads(mut self, n_threads: usize) -> Self {
        self.n_threads = Some(n_threads);
        self
    }

    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = Some(chunk_size);
        self
    }

    pub fn with_ignore_errors(mut self, ignore_errors: bool) -> Self {
        self.ignore_errors = Some(ignore_errors);
        self
    }

    pub fn get_has_header(&self) -> bool {
        self.has_header.unwrap_or(true)
    }

    pub fn get_rechunk(&self) -> bool {
        self.rechunk.unwrap_or(false)
    }

    pub fn get_low_memory(&self) -> bool {
        self.low_memory.unwrap_or(false)
    }

    pub fn get_n_threads(&self) -> Option<usize> {
        self.n_threads
    }

    pub fn get_chunk_size(&self) -> usize {
        self.chunk_size.unwrap_or(1 << 19)
    }

    pub fn get_ignore_errors(&self) -> bool {
        self.ignore_errors.unwrap_or(false)
    }
}

impl Default for CSVConfig {
    fn default() -> Self {
        Self::new("")
    }
}

/// Unified way to specify CSV input — either a simple path or a full [`CSVConfig`].
pub enum CSVRead {
    Filepath(PathBuf),
    Config(CSVConfig),
}

impl From<String> for CSVRead {
    fn from(value: String) -> Self {
        CSVRead::Filepath(value.into())
    }
}

impl From<&str> for CSVRead {
    fn from(value: &str) -> Self {
        CSVRead::Filepath(value.into())
    }
}

impl From<CSVConfig> for CSVRead {
    fn from(value: CSVConfig) -> Self {
        CSVRead::Config(value)
    }
}

fn load_csv(read: impl Into<CSVRead>) -> PolarsResult<DataFrame> {
    let read = read.into();
    let config = match read {
        CSVRead::Filepath(path) => CSVConfig::new(path),
        CSVRead::Config(cfg) => cfg,
    };
    load(config)
}

/// Reads a CSV file into a [`DataFrame`] using the provided configuration.
///
/// This is the main entry point for loading datasets in `mlrs`.
///
/// # Examples
///
/// **Simple usage with path:**
/// ```
/// use mlrs::dataset::read_csv;
/// # let filepath = "tests/fixtures/sample.csv";
/// let df = read_csv(filepath);
/// assert!(df.height() > 0);
/// ```
///
/// **Using custom configuration:**
/// ```
/// # use mlrs::dataset::{CSVConfig, read_csv};
/// # let filepath = "tests/fixtures/sample.csv";
/// let df = read_csv(
///     CSVConfig::new(filepath)
///         .with_has_header(true)
///         .with_n_threads(4)
/// );
/// ```
///
/// **Edge case — file with no header and error tolerance:**
/// ```
/// # use mlrs::dataset::{CSVConfig, read_csv};
/// # let filepath = "tests/fixtures/sample.csv";
/// let df = read_csv(
///     CSVConfig::new(filepath)
///         .with_has_header(false)
///         .with_ignore_errors(true)
/// );
/// ```
pub fn read_csv(read: impl Into<CSVRead>) -> DataFrame {
    load_csv(read).expect("Failed to load csv")
}

fn load(config: CSVConfig) -> PolarsResult<DataFrame> {
    let dataframe = CsvReadOptions::default()
        .with_has_header(config.get_has_header())
        .with_rechunk(config.get_rechunk())
        .with_low_memory(config.get_low_memory())
        .with_n_threads(config.get_n_threads())
        .with_chunk_size(config.get_chunk_size())
        .with_ignore_errors(config.get_ignore_errors())
        .try_into_reader_with_file_path(Some(config.filepath.clone()))?
        .finish()?;

    // println!(
    //     "Dataset has {} rows and {} columns",
    //     dataframe.height(),
    //     dataframe.width()
    // );
    // let schema = dataframe.schema();
    // for (name, datatype) in schema.iter() {
    //     println!("Column: {}, Datatype: {:?}", name, datatype);
    // }

    Ok(dataframe)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "tests/fixtures/sample.csv";
    const EMPTY: &str = "tests/fixtures/empty.csv";
    const INCONSISTENT: &str = "tests/fixtures/inconsistent.csv";

    const NUMERIC_COL: &str = "loan_amnt";
    const STRING_COL: &str = "grade";

    #[test]
    fn test_load_csv_filepath_string() {
        assert!(load_csv(SAMPLE.to_string()).is_ok());
    }

    #[test]
    fn test_load_csv_filepath_str() {
        assert!(load_csv(SAMPLE).is_ok());
    }

    #[test]
    fn test_load_csv_file_not_found() {
        assert!(load_csv("tests/fixtures/nonexistent.csv").is_err());
    }

    #[test]
    fn test_load_csv_empty_file() {
        match load_csv(EMPTY) {
            Ok(df) => assert_eq!(df.height(), 0),
            Err(_) => {}
        }
    }

    #[test]
    fn test_sample_csv_loads_and_has_expected_columns() {
        let df = load_csv(SAMPLE).unwrap();
        assert!(df.height() > 1000);
        assert!(df.column(NUMERIC_COL).is_ok());
        assert!(df.column(STRING_COL).is_ok());
    }

    #[test]
    fn test_with_header_true() {
        let df = load_csv(SAMPLE).unwrap();
        assert!(df.column("loan_amnt").is_ok());
        assert!(df.column("grade").is_ok());
    }

    #[test]
    fn test_with_header_false() {
        let df = load_csv(CSVConfig::new(SAMPLE).with_has_header(false)).unwrap();
        assert!(df.column("column_1").is_ok());
        assert!(df.column("loan_amnt").is_err());
    }

    #[test]
    fn test_config_defaults() {
        let cfg = CSVConfig::new(SAMPLE);
        assert_eq!(cfg.get_has_header(), true);
        assert_eq!(cfg.get_rechunk(), false);
        assert_eq!(cfg.get_low_memory(), false);
        assert_eq!(cfg.get_n_threads(), None);
        assert_eq!(cfg.get_chunk_size(), 1 << 19);
    }

    #[test]
    fn test_config_builder_chain() {
        let cfg = CSVConfig::new(SAMPLE)
            .with_has_header(false)
            .with_rechunk(true)
            .with_low_memory(true)
            .with_n_threads(4)
            .with_chunk_size(1024);
        assert_eq!(cfg.get_has_header(), false);
        assert_eq!(cfg.get_rechunk(), true);
        assert_eq!(cfg.get_low_memory(), true);
        assert_eq!(cfg.get_n_threads(), Some(4));
        assert_eq!(cfg.get_chunk_size(), 1024);
    }

    #[test]
    fn test_load_with_config() {
        let cfg = CSVConfig::new(SAMPLE)
            .with_has_header(true)
            .with_rechunk(true);
        assert!(load_csv(cfg).is_ok());
    }

    #[test]
    fn test_low_memory_mode() {
        let cfg = CSVConfig::new(SAMPLE).with_low_memory(true);
        let df = load_csv(cfg).unwrap();
        assert!(df.height() > 0);
    }

    #[test]
    fn test_n_threads() {
        let cfg = CSVConfig::new(SAMPLE).with_n_threads(2);
        assert!(load_csv(cfg).is_ok());
    }

    #[test]
    fn test_ignore_errors() {
        let cfg = CSVConfig::new(INCONSISTENT).with_ignore_errors(true);
        assert!(load_csv(cfg).is_ok());
    }

    #[test]
    fn test_csvread_from_string() {
        let r: CSVRead = SAMPLE.to_string().into();
        assert!(matches!(r, CSVRead::Filepath(_)));
    }

    #[test]
    fn test_csvread_from_str() {
        let r: CSVRead = SAMPLE.into();
        assert!(matches!(r, CSVRead::Filepath(_)));
    }

    #[test]
    fn test_csvread_from_config() {
        let r: CSVRead = CSVConfig::new(SAMPLE).into();
        assert!(matches!(r, CSVRead::Config(_)));
    }

    #[test]
    fn test_inconsistent_csv() {
        match load_csv(INCONSISTENT) {
            Ok(df) => assert!(df.height() > 0),
            Err(_) => {}
        }
    }

    #[test]
    fn test_csv_config_default_filepath_is_empty() {
        let cfg = CSVConfig::default();
        assert_eq!(cfg.filepath, PathBuf::from(""));
    }
}
