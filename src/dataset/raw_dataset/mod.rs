use csv::{Reader, ReaderBuilder, StringRecord};
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assign_type_basic() {
        let result = assign_type("tests/fixtures/sample.csv".to_string(), None);
        assert!(result.is_ok());
        let map = result.unwrap();
        assert_eq!(map["id"], ColumnTag::Float);
        assert_eq!(map["name"], ColumnTag::String);
        assert_eq!(map["age"], ColumnTag::Float);
        assert_eq!(map["salary"], ColumnTag::Float);
        assert_eq!(map["active"], ColumnTag::Boolean);
        assert_eq!(map["score"], ColumnTag::String);
        assert_eq!(map["department"], ColumnTag::String);
        assert_eq!(map["joined"], ColumnTag::String);
        assert_eq!(map["rating"], ColumnTag::String);
        assert_eq!(map["verified"], ColumnTag::String);
        assert_eq!(map["code"], ColumnTag::String);
        assert_eq!(map["region"], ColumnTag::String);
        assert_eq!(map["level"], ColumnTag::Float);
        assert_eq!(map["flag"], ColumnTag::String);
        assert_eq!(map["count"], ColumnTag::String);
        assert_eq!(map["value"], ColumnTag::String);
        assert_eq!(map["status"], ColumnTag::String);
        assert_eq!(map["tag"], ColumnTag::String);
        assert_eq!(map["priority"], ColumnTag::String);
        assert_eq!(map["notes"], ColumnTag::String);
    }

    #[test]
    fn test_empty_csv_error() {
        let result = assign_type("tests/fixtures/empty.csv".to_string(), Some(','));
        assert!(matches!(result, Err(DatasetError::EmptyDatasetError)));
    }

    #[test]
    fn test_inconsistent_row_length_error() {
        let result = assign_type("tests/fixtures/inconsistent.csv".to_string(), None);
        dbg!(&result);
        assert!(matches!(result, Err(DatasetError::CSVParseError(..))));
    }
}
