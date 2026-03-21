use mlrs::dataset::preprocessing::encoding::{LabelEncoder, OneHotEncoder};
use mlrs::dataset::preprocessing::scaling::MinMaxScaler;
use mlrs::dataset::read_csv;

use polars::prelude::*;

#[test]
fn test_load_csv_and_minmax_scale() {
    let mut df = read_csv("tests/fixtures/sample.csv");

    let col = df.column("age").unwrap();
    dbg!(col.dtype());
    dbg!(col.null_count());

    let mut scaler = MinMaxScaler::new((0.0, 1.0));
    scaler
        .fit_transform(&mut df, &["age", "salary", "value", "id"])
        .unwrap();

    let col = df.column("age").unwrap().cast(&DataType::Float64).unwrap();
    let ca = col.f64().unwrap();
    dbg!(ca.min());
    dbg!(ca.max());

    assert!(ca.min().unwrap() >= 0.0 - 1e-6);
    assert!(ca.max().unwrap() <= 1.0 + 1e-6);
}

#[test]
fn test_load_csv_and_label_encode() {
    let mut df = read_csv("tests/fixtures/sample.csv");
    let original_width = df.width();

    let mut encoder = LabelEncoder::new();
    encoder
        .fit_transform(&mut df, &["department", "status"])
        .unwrap();

    assert_eq!(df.width(), original_width);
    assert_eq!(*df.column("department").unwrap().dtype(), DataType::UInt32);
    assert_eq!(*df.column("status").unwrap().dtype(), DataType::UInt32);
}

#[test]
fn test_load_csv_and_onehot_encode() {
    let mut df = read_csv("tests/fixtures/sample.csv");
    let original_width = df.width();

    let mut encoder = OneHotEncoder::new();
    encoder.fit_transform(&mut df, &["status"]).unwrap(); // status has 3 categories

    assert!(df.width() > original_width);
    assert!(df.column("status_active").is_ok());
}
