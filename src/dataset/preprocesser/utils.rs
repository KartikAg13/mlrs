use polars::prelude::*;

/// Returns `true` if the given [`DataType`] is considered numeric for ML preprocessing.
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

/// Returns `true` if the given [`DataType`] is a string (categorical).
pub fn is_categorical(datatype: &DataType) -> bool {
    matches!(datatype, DataType::String)
}
