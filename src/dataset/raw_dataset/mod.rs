use csv::{Reader, ReaderBuilder, StringRecord};
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DatasetError {
    #[error("I/O error while reading file: {0}")]
    IOReadError(#[from] std::io::Error),

    #[error("Error parsing CSV: {0}")]
    CSVParseError(#[from] csv::Error),

    #[error("Dataset is empty")]
    EmptyDatasetError,

    #[error("Failed to assign column type in column {c} at index {i}")]
    ColumnTypeAssignError { c: String, i: usize },
}

#[derive(Debug, PartialEq)]
pub enum ColumnTag {
    Float,
    String,
    Boolean,
}

fn load_csv(filepath: String, delimiter: char) -> Result<Reader<File>, DatasetError> {
    Ok(ReaderBuilder::new()
        .delimiter(delimiter as u8)
        .from_path(&filepath)?)
}

fn infer_type(value: String) -> ColumnTag {
    let v = value.trim().to_lowercase();
    match v.as_str() {
        "true" | "1" | "yes" => ColumnTag::Boolean,
        "false" | "0" | "no" => ColumnTag::Boolean,
        _ => match value.trim().parse::<f32>() {
            Ok(_) => ColumnTag::Float,
            Err(_) => ColumnTag::String,
        },
    }
}

pub fn assign_type(
    filepath: String,
    delimiter: Option<char>,
) -> Result<HashMap<String, ColumnTag>, DatasetError> {
    let deli: char = delimiter.unwrap_or(',');

    let mut csv_reader: Reader<File> = load_csv(filepath.clone(), deli)?;

    let headers: Vec<String> = csv_reader
        .headers()?
        .iter()
        .map(|s| s.to_string())
        .collect();

    const SAMPLE_SIZE: usize = 100;

    let mut current_row: usize = 0;
    let mut reservoir: Vec<Vec<String>> = Vec::with_capacity(SAMPLE_SIZE);
    let mut string_record: StringRecord = StringRecord::new();
    let mut map: HashMap<String, ColumnTag> = HashMap::new();

    while csv_reader.read_record(&mut string_record)? {
        if current_row < SAMPLE_SIZE {
            reservoir.push(string_record.iter().map(|s| s.to_string()).collect());
        } else {
            let j = fastrand::usize(0..=current_row);
            if j < SAMPLE_SIZE {
                reservoir[j] = string_record.iter().map(|s| s.to_string()).collect();
            }
        }
        current_row += 1;
    }

    if current_row == 0 {
        return Err(DatasetError::EmptyDatasetError);
    }

    for row in &reservoir {
        for (index, name) in headers.iter().enumerate() {
            if map.get(name) == Some(&ColumnTag::String) {
                continue;
            }
            let inferred_tag = infer_type(row[index].clone());
            map.entry(name.clone())
                .and_modify(|e| {
                    if *e != inferred_tag {
                        if *e == ColumnTag::Boolean && inferred_tag == ColumnTag::Float {
                            *e = ColumnTag::Float;
                        } else if *e == ColumnTag::Float && inferred_tag == ColumnTag::Boolean {
                            *e = ColumnTag::Float;
                        } else {
                            *e = ColumnTag::String;
                        }
                    }
                })
                .or_insert_with(|| inferred_tag);
        }
    }
    Ok(map)
}

pub fn assign_type_parallel(
    filepath: String,
    delimiter: Option<char>,
) -> Result<HashMap<String, ColumnTag>, DatasetError> {
    let deli: char = delimiter.unwrap_or(',');

    let mut csv_reader: Reader<File> = load_csv(filepath.clone(), deli)?;

    let headers: Vec<String> = csv_reader
        .headers()?
        .iter()
        .map(|s| s.to_string())
        .collect();

    const SAMPLE_SIZE: usize = 100;

    let mut current_row: usize = 0;
    let mut reservoir: Vec<Vec<String>> = Vec::with_capacity(SAMPLE_SIZE);
    let mut string_record: StringRecord = StringRecord::new();

    while csv_reader.read_record(&mut string_record)? {
        if current_row < SAMPLE_SIZE {
            reservoir.push(string_record.iter().map(|s| s.to_string()).collect());
        } else {
            let j = fastrand::usize(0..=current_row);
            if j < SAMPLE_SIZE {
                reservoir[j] = string_record.iter().map(|s| s.to_string()).collect();
            }
        }
        current_row += 1;
    }

    if current_row == 0 {
        return Err(DatasetError::EmptyDatasetError);
    }

    let map: HashMap<String, ColumnTag> = headers
        .par_iter()
        .enumerate()
        .map(|(index, name)| {
            let tag = reservoir
                .iter()
                .fold(None::<ColumnTag>, |acc, row| {
                    let inferred_tag = infer_type(row[index].clone());
                    Some(match acc {
                        None => inferred_tag,
                        Some(e) => match (&e, &inferred_tag) {
                            (ColumnTag::String, _) | (_, ColumnTag::String) => ColumnTag::String,
                            (ColumnTag::Float, _) | (_, ColumnTag::Float) => ColumnTag::Float,
                            _ => e,
                        },
                    })
                })
                .unwrap_or(ColumnTag::String);
            (name.clone(), tag)
        })
        .collect();

    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assign_type_basic() {
        let map = assign_type("tests/fixtures/sample.csv".to_string(), None).unwrap();
        let map_parallel =
            assign_type_parallel("tests/fixtures/sample.csv".to_string(), None).unwrap();

        assert_eq!(map, map_parallel);
    }

    #[test]
    fn test_empty_csv_error() {
        let result = assign_type("tests/fixtures/empty.csv".to_string(), Some(','));
        let result_parallel =
            assign_type_parallel("tests/fixtures/empty.csv".to_string(), Some(','));

        assert!(matches!(result, Err(DatasetError::EmptyDatasetError)));
        assert!(matches!(
            result_parallel,
            Err(DatasetError::EmptyDatasetError)
        ));
    }

    #[test]
    fn test_inconsistent_row_length_error() {
        let result = assign_type("tests/fixtures/inconsistent.csv".to_string(), None);
        let result_parallel =
            assign_type_parallel("tests/fixtures/inconsistent.csv".to_string(), None);

        assert!(matches!(result, Err(DatasetError::CSVParseError(..))));
        assert!(matches!(
            result_parallel,
            Err(DatasetError::CSVParseError(..))
        ));
    }
}
