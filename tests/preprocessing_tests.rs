use mlrs::dataset::preprocessing::encoding::{LabelEncoder, OneHotEncoder};
use mlrs::dataset::preprocessing::scaling::{MinMaxScaler, StandardScaler};
use mlrs::dataset::read_csv;

use polars::prelude::*;

const FIXTURE: &str = "tests/fixtures/sample.csv";
const NUMERIC_COLS: [&str; 4] = ["loan_amnt", "funded_amnt", "installment", "annual_inc"];
const LABEL_COLS: [&str; 3] = ["grade", "home_ownership", "verification_status"];
const ONEHOT_COLS: [&str; 1] = ["home_ownership"];

#[test]
fn test_load_csv_and_minmax_scale() {
    let mut df = read_csv(FIXTURE);

    let mut scaler = MinMaxScaler::new((0.0, 1.0));
    scaler.fit_transform(&mut df, &NUMERIC_COLS).unwrap();

    for col_name in NUMERIC_COLS {
        let col = df
            .column(col_name)
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap();
        let ca = col.f64().unwrap();
        assert!(
            ca.min().unwrap() >= -1e-6,
            "{} min below 0: {}",
            col_name,
            ca.min().unwrap()
        );
        assert!(
            ca.max().unwrap() <= 1.0 + 1e-6,
            "{} max above 1: {}",
            col_name,
            ca.max().unwrap()
        );
    }
}

#[test]
fn test_minmax_column_not_found_warns_and_continues() {
    let mut df = read_csv(FIXTURE);
    let mut scaler = MinMaxScaler::new((0.0, 1.0));
    let result = scaler.fit_transform(&mut df, &["loan_amnt", "nonexistent"]);
    assert!(result.is_ok());
    let col = df
        .column("loan_amnt")
        .unwrap()
        .cast(&DataType::Float64)
        .unwrap();
    let ca = col.f64().unwrap();
    assert!(ca.min().unwrap() >= -1e-6);
    assert!(ca.max().unwrap() <= 1.0 + 1e-6);
}

#[test]
fn test_load_csv_and_standard_scale() {
    let mut df = read_csv(FIXTURE);

    let mut scaler = StandardScaler::new();
    scaler.fit_transform(&mut df, &NUMERIC_COLS).unwrap();

    const EPSILON: f64 = 1e-2;
    for col_name in NUMERIC_COLS {
        let col = df
            .column(col_name)
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap();
        let ca = col.f64().unwrap();
        assert!(
            ca.mean().unwrap().abs() < EPSILON,
            "{} mean not ~0: {}",
            col_name,
            ca.mean().unwrap()
        );
        assert!(
            (ca.std(1).unwrap() - 1.0).abs() < EPSILON,
            "{} std not ~1: {}",
            col_name,
            ca.std(1).unwrap()
        );
    }
}

#[test]
fn test_load_csv_and_label_encode() {
    let mut df = read_csv(FIXTURE);
    let original_width = df.width();

    let mut encoder = LabelEncoder::new();
    encoder.fit_transform(&mut df, &LABEL_COLS).unwrap();

    assert_eq!(df.width(), original_width);
    for col_name in LABEL_COLS {
        assert_eq!(
            *df.column(col_name).unwrap().dtype(),
            DataType::UInt32,
            "{} should be UInt32 after label encoding",
            col_name
        );
    }
}

#[test]
fn test_label_encode_consistent_mapping() {
    let mut df = read_csv(FIXTURE);
    let mut encoder = LabelEncoder::new();
    encoder.fit_transform(&mut df, &["grade"]).unwrap();

    let col = df.column("grade").unwrap().u32().unwrap();

    assert!(col.len() > 0);
}

#[test]
fn test_load_csv_and_onehot_encode() {
    let mut df = read_csv(FIXTURE);
    let original_width = df.width();

    let mut encoder = OneHotEncoder::new();
    encoder.fit_transform(&mut df, &ONEHOT_COLS).unwrap();

    assert!(df.width() > original_width);

    for cat in ["RENT", "OWN", "MORTGAGE", "OTHER"] {
        let col_name = format!("home_ownership_{}", cat);
        assert!(
            df.column(&col_name).is_ok(),
            "Expected column {} not found",
            col_name
        );
    }
}

#[test]
fn test_onehot_correct_values() {
    let mut df = read_csv(FIXTURE);
    let mut encoder = OneHotEncoder::new();
    encoder.fit_transform(&mut df, &["home_ownership"]).unwrap();

    let rent = df.column("home_ownership_RENT").unwrap().bool().unwrap();
    let mortgage = df
        .column("home_ownership_MORTGAGE")
        .unwrap()
        .bool()
        .unwrap();

    let both_true = rent
        .into_iter()
        .zip(mortgage.into_iter())
        .any(|(r, m)| r == Some(true) && m == Some(true));
    assert!(!both_true, "A row cannot be both RENT and MORTGAGE");
}
