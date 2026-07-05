use polars::prelude::*;

/// Returns `true` if the given [`DataType`] is considered numeric for ML preprocessing.
pub fn is_numeric(datatype: &DataType) -> bool {
    matches!(
        datatype,
        DataType::Float32
            | DataType::Float64
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
    )
}

/// Returns `true` if the given [`DataType`] is a string (categorical).
pub fn is_categorical(datatype: &DataType) -> bool {
    matches!(
        datatype,
        DataType::String | DataType::Categorical(_, _) | DataType::Enum(_, _)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_numeric_float_types() {
        assert!(is_numeric(&DataType::Float32));
        assert!(is_numeric(&DataType::Float64));
    }

    #[test]
    fn test_is_numeric_signed_int_types() {
        assert!(is_numeric(&DataType::Int8));
        assert!(is_numeric(&DataType::Int16));
        assert!(is_numeric(&DataType::Int32));
        assert!(is_numeric(&DataType::Int64));
    }

    #[test]
    fn test_is_numeric_unsigned_int_types() {
        assert!(is_numeric(&DataType::UInt8));
        assert!(is_numeric(&DataType::UInt16));
        assert!(is_numeric(&DataType::UInt32));
        assert!(is_numeric(&DataType::UInt64));
    }

    #[test]
    fn test_is_numeric_rejects_non_numeric() {
        assert!(!is_numeric(&DataType::String));
        assert!(!is_numeric(&DataType::Boolean));
        assert!(!is_numeric(&DataType::Date));
    }

    #[test]
    fn test_is_categorical_string() {
        assert!(is_categorical(&DataType::String));
    }

    #[test]
    fn test_is_categorical_rejects_non_categorical() {
        assert!(!is_categorical(&DataType::Float64));
        assert!(!is_categorical(&DataType::Int32));
        assert!(!is_categorical(&DataType::Boolean));
    }
}
