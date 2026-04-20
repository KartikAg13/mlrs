//! Comprehensive tests for the preprocessing pipeline.
//!
//! This module validates the behavior of:
//! - [`MinMaxScaler`] and [`StandardScaler`]
//! - [`LabelEncoder`] and [`OneHotEncoder`]
//! - [`SimpleImputer`]
//!
//! All tests use a consistent synthetic dataset via [`make_test_dataframe()`].

use mlrs::dataset::preprocesser::encoder::{LabelEncoder, OneHotEncoder};
use mlrs::dataset::preprocesser::handler::{SimpleImputer, Strategy};
use mlrs::dataset::preprocesser::scaler::{MinMaxScaler, StandardScaler};
use polars::prelude::*;

/// Creates a small, reproducible test DataFrame with numeric and categorical columns.
fn make_test_dataframe() -> DataFrame {
    df![
        "loan_amnt"          => [5000.0, 2500.0, 2400.0, 10000.0, 3000.0],
        "funded_amnt"        => [5000.0, 2500.0, 2400.0, 10000.0, 3000.0],
        "installment"        => [162.87, 59.83, 84.33, 339.31, 67.79],
        "annual_inc"         => [24000.0, 30000.0, 12252.0, 49200.0, 80000.0],
        "grade"              => ["B", "C", "C", "C", "B"],
        "home_ownership"     => ["RENT", "RENT", "RENT", "RENT", "RENT"],
        "verification_status"=> ["Verified", "Source Verified", "Not Verified", "Source Verified", "Source Verified"],
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
        &[("age", Strategy::Mean), ("income", Strategy::Mean)],
    )
    .unwrap();

    let age = df.column("age").unwrap().f64().unwrap();
    let income = df.column("income").unwrap().f64().unwrap();

    // Mean of age: (25+35+40)/3 = 33.333...
    assert!((age.get(1).unwrap() - 33.333).abs() < 1e-3);
    // Mean of income: (50000+60000+80000)/3 = 63333.333...
    assert!((income.get(2).unwrap() - 63333.333).abs() < 1e-3);
}

#[test]
fn test_simple_imputer_constant_strategies() {
    let mut df = df![
        "score" => [Some(85.0), None, Some(92.0)],
        "flag"  => [Some("A"), None, Some("B")]
    ]
    .unwrap();

    SimpleImputer::fill_null(
        &mut df,
        &[("score", Strategy::Zero), ("flag", Strategy::ForwardFill)],
    )
    .unwrap();

    let score = df.column("score").unwrap().f64().unwrap();
    let flag = df.column("flag").unwrap().str().unwrap();

    assert_eq!(score.get(1), Some(0.0));
    assert_eq!(flag.get(1), Some("A")); // forward filled from previous value
}

#[test]
fn test_simple_imputer_min_max_strategies() {
    let mut df = df![
        "value" => [Some(10.0), None, Some(30.0), Some(50.0)]
    ]
    .unwrap();

    SimpleImputer::fill_null(&mut df, &[("value", Strategy::MinimumValue)]).unwrap();

    let value = df.column("value").unwrap().f64().unwrap();
    assert_eq!(value.get(1), Some(10.0)); // filled with min value
}

#[test]
fn test_simple_imputer_missing_column_warns_and_continues() {
    let mut df = make_test_dataframe();

    let result = SimpleImputer::fill_null(
        &mut df,
        &[
            ("loan_amnt", Strategy::Mean),
            ("nonexistent", Strategy::Zero),
        ],
    );

    assert!(result.is_ok(), "Should skip missing columns with a warning");
}

#[test]
fn test_simple_imputer_all_null_column() {
    let mut df = df![
        "all_null" => [Option::<f64>::None, None, None]
    ]
    .unwrap();

    // Mean of all-null column should not panic
    let result = SimpleImputer::fill_null(&mut df, &[("all_null", Strategy::Mean)]);
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
    let mut test = df!["category" => ["A", "B", "D", "E"]].unwrap(); // D and E unseen

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
// Full Pipeline Integration Test
// -----------------------------------------------------------------------------

#[test]
fn test_full_preprocessing_pipeline() {
    let mut df = make_test_dataframe();

    // Step 1: Impute any nulls (none in this dataset, but good to test)
    SimpleImputer::fill_null(
        &mut df,
        &[
            ("loan_amnt", Strategy::Mean),
            ("annual_inc", Strategy::Mean),
        ],
    )
    .unwrap();

    // Step 2: Encode categoricals
    let mut label_encoder = LabelEncoder::new();
    label_encoder
        .fit_transform(&mut df, &["grade", "home_ownership"])
        .unwrap();

    // Step 3: Scale numeric features
    let mut scaler = MinMaxScaler::new((0.0, 1.0));
    scaler
        .fit_transform(&mut df, &["loan_amnt", "annual_inc", "installment"])
        .unwrap();

    // Final assertions
    assert_eq!(*df.column("grade").unwrap().dtype(), DataType::UInt32);
    let scaled = df.column("loan_amnt").unwrap().f64().unwrap();
    assert!(scaled.min().unwrap() >= -1e-9);
    assert!(scaled.max().unwrap() <= 1.0 + 1e-9);
}
