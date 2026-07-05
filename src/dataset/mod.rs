//! CSV dataset loader for `mlrs`.
//!
//! Zero-allocation hot paths where possible, cache-friendly chunked reading,
//! and a clean builder API that matches Polars' performance characteristics
//! without Pythonic sugar.
//! # Quick Start
//!
//! ```
//! use mlrs::dataset::read_csv;
//!
//! let df = read_csv("tests/fixtures/sample.csv");
//!
//! println!("Rows: {}", df.height());
//! println!("Columns: {}", df.width());
//! ```
//!
//! For more control over parsing options, use [`CSVReader`].
// //! For lazy evaluation of large datasets, use [`scan_csv`].

use colored::Colorize;
use polars::prelude::*;
use std::path::PathBuf;

pub mod error;
pub mod preprocessor;

use error::{DatasetError, DatasetErrorInner};

/// Builder for reading CSV files into a `DataFrame`.
///
/// Designed for ML workloads: sensible defaults, explicit control over
/// threading, memory, and chunking. Consumes itself on `finish()` to avoid
/// accidental reuse.
#[derive(Debug, Clone)]
pub struct CSVReader {
    filepath: PathBuf,
    has_header: bool,
    rechunk: bool,
    low_memory: bool,
    n_threads: Option<usize>,
    chunk_size: usize,
    ignore_errors: bool,
}

impl CSVReader {
    /// Creates a new CSV reader with sensible defaults for machine learning datasets.
    ///
    /// By default:
    ///
    /// - Assumes the first row contains column names.
    /// - Uses Polars' automatic thread selection.
    /// - Reads in 512 KiB chunks.
    /// - Does not ignore malformed rows.
    ///
    /// # Examples
    ///
    /// **Basic usage:**
    ///
    /// ```
    /// use mlrs::dataset::CSVReader;
    ///
    /// let df = CSVReader::new("tests/fixtures/sample.csv")
    ///     .finish();
    ///
    /// assert!(df.height() > 0);
    /// ```
    ///
    /// **Advanced configuration:**
    ///
    /// ```
    /// use mlrs::dataset::CSVReader;
    ///
    /// let df = CSVReader::new("tests/fixtures/sample.csv")
    ///     .threads(4)
    ///     .rechunk(true)
    ///     .low_memory(false)
    ///     .chunk_size(1 << 20)
    ///     .finish();
    ///
    /// assert!(df.width() > 0);
    /// ```
    ///
    /// **Edge case (CSV without a header):**
    ///
    /// ```
    /// use mlrs::dataset::CSVReader;
    ///
    /// let df = CSVReader::new("tests/fixtures/sample.csv")
    ///     .header(false)
    ///     .finish();
    ///
    /// assert_eq!(df.get_column_names()[0], "column_1");
    /// ```
    #[inline]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            filepath: path.into(),
            has_header: true,
            rechunk: false,
            low_memory: false,
            n_threads: None,
            chunk_size: 1 << 19,
            ignore_errors: false,
        }
    }

    /// Whether the first row is a header.
    ///
    /// # Example
    /// ```
    /// use mlrs::dataset::CSVReader;
    /// let df = CSVReader::new("tests/fixtures/sample.csv").header(true).finish();
    /// ```
    #[inline]
    pub fn header(mut self, has_header: bool) -> Self {
        self.has_header = has_header;
        self
    }

    /// Number of threads for parallel CSV parsing.
    ///
    /// # Example
    /// ```
    /// use mlrs::dataset::CSVReader;
    /// let df = CSVReader::new("tests/fixtures/sample.csv").threads(4).finish();
    /// ```
    #[inline]
    pub fn threads(mut self, n: usize) -> Self {
        self.n_threads = Some(n);
        self
    }

    /// Force rechunk after reading.
    ///
    /// # Example
    /// ```
    /// use mlrs::dataset::CSVReader;
    /// let df = CSVReader::new("tests/fixtures/sample.csv").rechunk(true).finish();
    /// ```
    #[inline]
    pub fn rechunk(mut self, v: bool) -> Self {
        self.rechunk = v;
        self
    }

    /// Low-memory mode (slower, lower peak RAM).
    ///
    /// # Example
    /// ```
    /// use mlrs::dataset::CSVReader;
    /// let df = CSVReader::new("tests/fixtures/sample.csv").low_memory(true).finish();
    /// ```
    #[inline]
    pub fn low_memory(mut self, v: bool) -> Self {
        self.low_memory = v;
        self
    }

    /// Read chunk size in bytes.
    ///
    /// # Example
    /// ```
    /// use mlrs::dataset::CSVReader;
    /// let df = CSVReader::new("tests/fixtures/sample.csv").chunk_size(1 << 20).finish();
    /// ```
    #[inline]
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Ignore malformed rows instead of failing.
    ///
    /// # Example
    /// ```
    /// use mlrs::dataset::CSVReader;
    /// let df = CSVReader::new("tests/fixtures/inconsistent.csv").ignore_errors(true).finish();
    /// ```
    #[inline]
    pub fn ignore_errors(mut self, v: bool) -> Self {
        self.ignore_errors = v;
        self
    }

    /// Materialize the `DataFrame`.
    ///
    /// # Example
    /// ```
    /// use mlrs::dataset::CSVReader;
    /// let df = CSVReader::new("tests/fixtures/sample.csv").try_finish().unwrap();
    /// ```
    pub fn try_finish(self) -> Result<DataFrame, DatasetError> {
        let path = self.filepath.clone();

        CsvReadOptions::default()
            .with_has_header(self.has_header)
            .with_rechunk(self.rechunk)
            .with_low_memory(self.low_memory)
            .with_n_threads(self.n_threads)
            .with_chunk_size(self.chunk_size)
            .with_ignore_errors(self.ignore_errors)
            .try_into_reader_with_file_path(Some(path.clone()))
            .map_err(|e| DatasetErrorInner::Parse {
                path: path.clone(),
                source: e,
            })?
            .finish()
            .map_err(|e| {
                use polars::error::PolarsError::*;

                match &e {
                    NoData(_) => DatasetErrorInner::EmptyFile { path: path.clone() }.into(),

                    ComputeError(_) | SchemaMismatch(_) | ShapeMismatch(_) => {
                        DatasetErrorInner::Parse {
                            path: path.clone(),
                            source: e,
                        }
                        .into()
                    }

                    _ => DatasetErrorInner::Parse { path, source: e }.into(),
                }
            })
    }

    /// Materialize the `DataFrame`, panicking with a clear message on failure.
    ///
    /// # Example
    /// ```
    /// use mlrs::dataset::CSVReader;
    /// let df = CSVReader::new("tests/fixtures/sample.csv").finish();
    /// ```
    pub fn finish(self) -> DataFrame {
        match self.try_finish() {
            Ok(df) => df,

            Err(e) => {
                eprintln!(
                    "\n{} {}\n",
                    "ERROR".red().bold(),
                    e.to_string().bright_red()
                );

                panic!("{e}");
            }
        }
    }
}

/// Reads a CSV file into a [`DataFrame`] using the default configuration.
///
/// This is the simplest way to load a dataset.
///
/// Internally this is equivalent to:
///
/// ```ignore
/// CSVReader::new(path).finish()
/// ```
///
/// # Examples
///
/// **Basic usage:**
///
/// ```
/// use mlrs::dataset::read_csv;
///
/// let df = read_csv("tests/fixtures/sample.csv");
///
/// assert!(df.height() > 0);
/// ```
///
/// **Edge case (missing file):**
///
/// ```should_panic
/// use mlrs::dataset::read_csv;
///
/// read_csv("tests/fixtures/does_not_exist.csv");
/// ```
#[inline]
pub fn read_csv(path: impl Into<PathBuf>) -> DataFrame {
    CSVReader::new(path).finish()
}

// /// Lazily scans a CSV file without immediately loading it into memory.
// ///
// /// Unlike [`read_csv`], this function returns a [`LazyFrame`], allowing
// /// predicate pushdown, projection pushdown, and query optimization before
// /// materialization.
// ///
// /// Call `.collect()` to execute the query.
// ///
// /// # Examples
// ///
// /// Basic usage:
// ///
// /// ```
// /// use mlrs::dataset::scan_csv;
// ///
// /// let df = scan_csv("tests/fixtures/sample.csv")
// ///     .collect()
// ///     .unwrap();
// ///
// /// assert!(df.height() > 0);
// /// ```
// ///
// /// Filtering before collecting:
// ///
// /// ```
// /// use mlrs::dataset::scan_csv;
// /// use polars::prelude::*;
// ///
// /// let df = scan_csv("tests/fixtures/sample.csv")
// ///     .filter(col("loan_amnt").gt(lit(10000)))
// ///     .collect()
// ///     .unwrap();
// ///
// /// assert!(df.height() > 0);
// /// ```
// ///
// /// Edge case (non-existent file):
// ///
// /// ```should_panic
// /// use mlrs::dataset::scan_csv;
// ///
// /// scan_csv("tests/fixtures/does_not_exist.csv");
// /// ```
// #[inline]
// pub fn scan_csv(path: impl AsRef<Path>) -> LazyFrame {
//     LazyCsvReader::new(path.as_ref())
//         .with_has_header(true)
//         .finish()
//         .expect("failed to create lazy CSV reader")
// }

#[cfg(test)]
mod tests {
    use super::*;

    // Use a real fixture in CI. These tests assume it exists.
    const SAMPLE: &str = "tests/fixtures/sample.csv";

    #[test]
    fn basic_load_has_rows_and_columns() {
        let df = CSVReader::new(SAMPLE).finish();
        assert!(df.height() > 0);
        assert!(df.width() > 0);
    }

    #[test]
    fn header_true_exposes_named_columns() {
        let df = CSVReader::new(SAMPLE).header(true).finish();
        assert!(df.column("loan_amnt").is_ok());
        assert!(df.column("grade").is_ok());
    }

    #[test]
    fn header_false_uses_column_n() {
        let df = CSVReader::new(SAMPLE).header(false).finish();
        assert!(df.column("column_1").is_ok());
        assert!(df.column("loan_amnt").is_err());
    }

    #[test]
    fn try_finish_returns_error_on_missing_file() {
        let err = CSVReader::new("nonexistent.csv").try_finish();
        assert!(err.is_err());
    }

    // #[test]
    // fn scan_csv_produces_lazy_frame() {
    //     let lf = scan_csv(SAMPLE);
    //     let df = lf.collect().unwrap();
    //     assert!(df.height() > 0);
    // }

    #[test]
    fn default_configuration() {
        let reader = CSVReader::new(SAMPLE);

        assert!(reader.has_header);
        assert!(!reader.rechunk);
        assert!(!reader.low_memory);
        assert_eq!(reader.n_threads, None);
        assert_eq!(reader.chunk_size, 1 << 19);
        assert!(!reader.ignore_errors);
    }

    #[test]
    fn builder_chain_updates_every_field() {
        let reader = CSVReader::new(SAMPLE)
            .header(false)
            .threads(4)
            .rechunk(true)
            .low_memory(true)
            .chunk_size(1024)
            .ignore_errors(true);

        assert!(!reader.has_header);
        assert_eq!(reader.n_threads, Some(4));
        assert!(reader.rechunk);
        assert!(reader.low_memory);
        assert_eq!(reader.chunk_size, 1024);
        assert!(reader.ignore_errors);
    }

    #[test]
    fn sample_dimensions_match_fixture() {
        let df = read_csv(SAMPLE);

        assert_eq!(df.height(), 1001);
        assert_eq!(df.width(), 142);
    }

    #[test]
    fn expected_columns_exist() {
        let df = read_csv(SAMPLE);

        let names = df.get_column_names();

        assert!(names.iter().any(|c| *c == "loan_amnt"));
        assert!(names.iter().any(|c| *c == "grade"));
        assert!(names.iter().any(|c| *c == "term"));
    }

    #[test]
    fn column_types_are_correct() {
        let df = read_csv(SAMPLE);

        assert_eq!(df.column("loan_amnt").unwrap().dtype(), &DataType::Int64);
        assert_eq!(df.column("grade").unwrap().dtype(), &DataType::String);
    }

    #[test]
    fn loads_with_multiple_threads() {
        for threads in [1, 2, 4, 8] {
            let df = CSVReader::new(SAMPLE).threads(threads).finish();

            assert!(df.height() > 0);
        }
    }

    #[test]
    fn rechunk_loads_successfully() {
        let df = CSVReader::new(SAMPLE).rechunk(true).finish();

        assert!(df.height() > 0);
    }

    #[test]
    fn low_memory_mode_loads() {
        let df = CSVReader::new(SAMPLE).low_memory(true).finish();

        assert!(df.height() > 0);
    }

    #[test]
    fn multiple_chunk_sizes_work() {
        for size in [1024, 8192, 65536, 1 << 20] {
            let df = CSVReader::new(SAMPLE).chunk_size(size).finish();

            assert!(df.height() > 0);
        }
    }

    const INCONSISTENT: &str = "tests/fixtures/inconsistent.csv";

    #[test]
    fn malformed_csv_can_be_ignored() {
        let df = CSVReader::new(INCONSISTENT).ignore_errors(true).finish();

        assert!(df.height() > 0);
    }

    const EMPTY: &str = "tests/fixtures/empty.csv";

    #[test]
    fn empty_csv_returns_error() {
        assert!(CSVReader::new(EMPTY).try_finish().is_err());
    }

    #[test]
    fn read_csv_panics_on_missing_file() {
        let result = std::panic::catch_unwind(|| {
            read_csv("missing.csv");
        });

        assert!(result.is_err());
    }

    // #[test]
    // fn lazy_frame_collect_matches_eager() {
    //     let eager = read_csv(SAMPLE);
    //     let lazy = scan_csv(SAMPLE).collect().unwrap();

    //     assert_eq!(eager.height(), lazy.height());
    //     assert_eq!(eager.width(), lazy.width());
    // }

    #[test]
    fn cloned_reader_produces_identical_dataframe() {
        let reader = CSVReader::new(SAMPLE);

        let df1 = reader.clone().finish();
        let df2 = reader.finish();

        assert_eq!(df1.shape(), df2.shape());
    }

    const UTF8: &str = "tests/fixtures/utf8.csv";

    #[test]
    fn utf8_is_preserved() {
        let df = read_csv(UTF8);

        assert_eq!(
            df.column("city").unwrap().str().unwrap().get(0),
            Some("東京")
        );
    }

    #[test]
    fn quoted_commas_are_parsed_correctly() {
        let df = read_csv("tests/fixtures/quoted.csv");

        assert_eq!(
            df.column("address").unwrap().str().unwrap().get(0),
            Some("New York, USA")
        );
    }

    #[test]
    fn escaped_quotes_are_preserved() {
        let df = read_csv("tests/fixtures/escaped_quotes.csv");

        assert_eq!(
            df.column("text").unwrap().str().unwrap().get(0),
            Some("He said \"hello\"")
        );
    }

    #[test]
    fn null_values_are_detected() {
        let df = read_csv("tests/fixtures/nulls.csv");

        assert_eq!(df.column("b").unwrap().null_count(), 1);
    }

    #[test]
    fn no_header_generates_default_names() {
        let df = CSVReader::new(SAMPLE).header(false).finish();

        let names = df.get_column_names();

        assert_eq!(names[0], "column_1");
        assert_eq!(names[1], "column_2");
    }
}
