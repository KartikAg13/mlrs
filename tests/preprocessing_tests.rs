//! Comprehensive tests for the preprocessing pipeline.
//!
//! This module validates the behavior of:
//! - MinMaxScaler and StandardScaler
//! - LabelEncoder and OneHotEncoder
//! - SimpleImputer
//! - Dropper (new)

use mlrs::dataset::preprocesser::encoder::{LabelEncoder, OneHotEncoder};
use mlrs::dataset::preprocesser::handler::{
    Dropper, DropperStrategy, ImputerStrategy, SimpleImputer,
};
use mlrs::dataset::preprocesser::scaler::{MinMaxScaler, StandardScaler};
use polars::prelude::*;

/// Creates a small, reproducible test DataFrame with numeric and categorical columns.
fn make_test_dataframe() -> DataFrame {
    df![
        "loan_amnt" => [5000.0, 2500.0, 2400.0, 10000.0, 3000.0],
        "funded_amnt" => [5000.0, 2500.0, 2400.0, 10000.0, 3000.0],
        "installment" => [162.87, 59.83, 84.33, 339.31, 67.79],
        "annual_inc" => [24000.0, 30000.0, 12252.0, 49200.0, 80000.0],
        "grade" => ["B", "C", "C", "C", "B"],
        "home_ownership" => ["RENT", "RENT", "RENT", "RENT", "RENT"],
        "verification_status"=> ["Verified", "Source Verified", "Not Verified", "Source Verified", "Source Verified"],
    ]
    .unwrap()
}

fn make_df_with_nulls() -> DataFrame {
    df![
        "a" => [Some(1.0_f64), None, Some(3.0), None],
        "b" => [Some(4.0_f64), Some(5.0), None, None],
        "c" => [Some(7.0_f64), Some(8.0), Some(9.0), Some(10.0)]
    ]
    .unwrap()
}

// -----------------------------------------------------------------------------
// SimpleImputer Tests
// -----------------------------------------------------------------------------
#[test]
fn test_simple_imputer_mean_strategy() {
    let mut df = df![
        "age" => [Some(25.0), None, Some(35.0), Some(40.0)],
        "income" => [Some(50000.0), Some(60000.0), None, Some(80000.0)]
    ]
    .unwrap();

    SimpleImputer::fill_null(
        &mut df,
        &[
            ("age", ImputerStrategy::Mean),
            ("income", ImputerStrategy::Mean),
        ],
    )
    .unwrap();

    let age = df.column("age").unwrap().f64().unwrap();
    let income = df.column("income").unwrap().f64().unwrap();

    assert!((age.get(1).unwrap() - 33.333).abs() < 1e-3);
    assert!((income.get(2).unwrap() - 63333.333).abs() < 1e-3);
}

#[test]
fn test_simple_imputer_constant_strategies() {
    let mut df = df![
        "score" => [Some(85.0), None, Some(92.0)],
        "flag" => [Some("A"), None, Some("B")]
    ]
    .unwrap();

    SimpleImputer::fill_null(
        &mut df,
        &[
            ("score", ImputerStrategy::Zero),
            ("flag", ImputerStrategy::ForwardFill),
        ],
    )
    .unwrap();

    let score = df.column("score").unwrap().f64().unwrap();
    let flag = df.column("flag").unwrap().str().unwrap();

    assert_eq!(score.get(1), Some(0.0));
    assert_eq!(flag.get(1), Some("A"));
}

#[test]
fn test_simple_imputer_min_max_strategies() {
    let mut df = df![
        "value" => [Some(10.0), None, Some(30.0), Some(50.0)]
    ]
    .unwrap();

    SimpleImputer::fill_null(&mut df, &[("value", ImputerStrategy::MinimumValue)]).unwrap();

    let value = df.column("value").unwrap().f64().unwrap();
    assert_eq!(value.get(1), Some(10.0));
}

#[test]
fn test_simple_imputer_missing_column_warns_and_continues() {
    let mut df = make_test_dataframe();
    let result = SimpleImputer::fill_null(
        &mut df,
        &[
            ("loan_amnt", ImputerStrategy::Mean),
            ("nonexistent", ImputerStrategy::Zero),
        ],
    );
    assert!(result.is_ok());
}

#[test]
fn test_simple_imputer_all_null_column() {
    let mut df = df![
        "all_null" => [Option::<f64>::None, None, None]
    ]
    .unwrap();

    let result = SimpleImputer::fill_null(&mut df, &[("all_null", ImputerStrategy::Mean)]);
    assert!(result.is_ok());
}

// -----------------------------------------------------------------------------
// MinMaxScaler Tests
// -----------------------------------------------------------------------------
#[test]
fn test_minmax_scaler_basic_scaling() {
    let mut df = make_test_dataframe();
    let mut scaler = MinMaxScaler::new((0.0, 1.0));
    scaler
        .fit_transform(&mut df, &["loan_amnt", "annual_inc"])
        .unwrap();

    for col_name in ["loan_amnt", "annual_inc"] {
        let ca = df.column(col_name).unwrap().f64().unwrap();
        let min = ca.min().unwrap();
        let max = ca.max().unwrap();
        assert!(min >= -1e-9);
        assert!(max <= 1.0 + 1e-9);
    }
}

#[test]
fn test_minmax_scaler_column_not_found_continues_with_warning() {
    let mut df = make_test_dataframe();
    let mut scaler = MinMaxScaler::new((0.0, 1.0));
    let result = scaler.fit_transform(&mut df, &["loan_amnt", "nonexistent_column"]);
    assert!(result.is_ok());
}

#[test]
fn test_minmax_scaler_non_numeric_column_skipped_with_warning() {
    let mut df = make_test_dataframe();
    let mut scaler = MinMaxScaler::new((0.0, 1.0));
    let result = scaler.fit_transform(&mut df, &["loan_amnt", "grade"]);
    assert!(result.is_ok());
}

// -----------------------------------------------------------------------------
// StandardScaler Tests
// -----------------------------------------------------------------------------
#[test]
fn test_standard_scaler_zero_mean_unit_variance() {
    let mut df = make_test_dataframe();
    let mut scaler = StandardScaler::new();
    scaler
        .fit_transform(&mut df, &["loan_amnt", "annual_inc"])
        .unwrap();

    const EPSILON: f64 = 1e-8;
    for col_name in ["loan_amnt", "annual_inc"] {
        let ca = df.column(col_name).unwrap().f64().unwrap();
        let mean = ca.mean().unwrap();
        let std = ca.std(1).unwrap();
        assert!(mean.abs() < EPSILON);
        assert!((std - 1.0).abs() < EPSILON);
    }
}

// -----------------------------------------------------------------------------
// LabelEncoder Tests
// -----------------------------------------------------------------------------
#[test]
fn test_label_encoder_replaces_strings_with_uint32() {
    let mut df = make_test_dataframe();
    let original_width = df.width();
    let mut encoder = LabelEncoder::new();
    encoder
        .fit_transform(&mut df, &["grade", "home_ownership"])
        .unwrap();

    assert_eq!(df.width(), original_width);
    assert_eq!(*df.column("grade").unwrap().dtype(), DataType::UInt32);
}

#[test]
fn test_label_encoder_unseen_labels_map_to_max() {
    let train = df!["category" => ["A", "B", "A", "C"]].unwrap();
    let mut test = df!["category" => ["A", "B", "D", "E"]].unwrap();

    let mut encoder = LabelEncoder::new();
    encoder.fit(&train, &["category"]).unwrap();
    encoder.transform(&mut test, &["category"]).unwrap();

    let encoded = test.column("category").unwrap().u32().unwrap();
    assert_eq!(encoded.get(2), Some(u32::MAX));
    assert_eq!(encoded.get(3), Some(u32::MAX));
}

// -----------------------------------------------------------------------------
// OneHotEncoder Tests
// -----------------------------------------------------------------------------
#[test]
fn test_onehot_encoder_creates_new_binary_columns() {
    let mut df = make_test_dataframe();
    let original_width = df.width();
    let mut encoder = OneHotEncoder::new();
    encoder.fit_transform(&mut df, &["home_ownership"]).unwrap();

    assert!(df.width() > original_width);
    assert!(df.column("home_ownership_RENT").is_ok());
}

// -----------------------------------------------------------------------------
// Dropper Integration Tests
// -----------------------------------------------------------------------------
#[test]
fn test_dropper_drop_rows_any() {
    let mut df = make_df_with_nulls();
    Dropper::drop_rows(
        &mut df,
        &[("a", DropperStrategy::Any), ("b", DropperStrategy::Any)],
    )
    .unwrap();
    assert_eq!(df.height(), 1);
}

#[test]
fn test_dropper_drop_rows_all() {
    let mut df = make_df_with_nulls();
    Dropper::drop_rows(
        &mut df,
        &[("a", DropperStrategy::All), ("b", DropperStrategy::All)],
    )
    .unwrap();
    assert_eq!(df.height(), 3);
}

#[test]
fn test_dropper_drop_rows_single_column_any() {
    let mut df = make_df_with_nulls();
    Dropper::drop_rows(&mut df, &[("c", DropperStrategy::Any)]).unwrap();
    assert_eq!(df.height(), 4);
}

#[test]
fn test_dropper_drop_columns() {
    let mut df = make_test_dataframe();
    let original_width = df.width();
    Dropper::drop_columns(&mut df, &["grade", "home_ownership"]).unwrap();
    assert_eq!(df.width(), original_width - 2);
}

#[test]
fn test_dropper_missing_column_continues() {
    let mut df = make_test_dataframe();
    let result = Dropper::drop_rows(
        &mut df,
        &[
            ("loan_amnt", DropperStrategy::Any),
            ("nonexistent", DropperStrategy::Any),
        ],
    );
    assert!(result.is_ok());
}

#[test]
fn test_dropper_empty_input_is_noop() {
    let mut df = make_test_dataframe();
    let original_height = df.height();
    let original_width = df.width();

    Dropper::drop_rows(&mut df, &[]).unwrap();
    Dropper::drop_columns(&mut df, &[]).unwrap();

    assert_eq!(df.height(), original_height);
    assert_eq!(df.width(), original_width);
}

// -----------------------------------------------------------------------------
// Full Pipeline Integration Test
// -----------------------------------------------------------------------------
#[test]
fn test_full_preprocessing_pipeline() {
    let mut df = make_test_dataframe();

    // Step 1: Drop unnecessary or problematic columns
    Dropper::drop_columns(
        &mut df,
        &["funded_amnt", "verification_status"], // example of columns we don't need
    )
    .unwrap();

    // Step 2: Impute any nulls
    SimpleImputer::fill_null(
        &mut df,
        &[
            ("loan_amnt", ImputerStrategy::Mean),
            ("annual_inc", ImputerStrategy::Mean),
        ],
    )
    .unwrap();

    // Step 3: Encode categorical columns
    let mut label_encoder = LabelEncoder::new();
    label_encoder
        .fit_transform(&mut df, &["grade", "home_ownership"])
        .unwrap();

    // Step 4: Scale numeric features
    let mut scaler = MinMaxScaler::new((0.0, 1.0));
    scaler
        .fit_transform(&mut df, &["loan_amnt", "annual_inc", "installment"])
        .unwrap();

    // Final assertions
    assert_eq!(*df.column("grade").unwrap().dtype(), DataType::UInt32);

    // Check that dropped columns are really gone
    assert!(df.column("funded_amnt").is_err());
    assert!(df.column("verification_status").is_err());

    // Check scaling worked correctly
    let scaled = df.column("loan_amnt").unwrap().f64().unwrap();
    assert!(scaled.min().unwrap() >= -1e-9);
    assert!(scaled.max().unwrap() <= 1.0 + 1e-9);
}
