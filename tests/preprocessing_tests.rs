use mlrs::dataset::preprocesser::encoder::{LabelEncoder, OneHotEncoder};
use mlrs::dataset::preprocesser::scaler::{MinMaxScaler, StandardScaler};
use polars::prelude::*;

fn make_test_dataframe() -> DataFrame {
    df![
        "loan_amnt"     => [5000.0, 2500.0, 2400.0, 10000.0, 3000.0],
        "funded_amnt"   => [5000.0, 2500.0, 2400.0, 10000.0, 3000.0],
        "installment"   => [162.87, 59.83, 84.33, 339.31, 67.79],
        "annual_inc"    => [24000.0, 30000.0, 12252.0, 49200.0, 80000.0],
        "grade"         => ["B", "C", "C", "C", "B"],
        "home_ownership"=> ["RENT", "RENT", "RENT", "RENT", "RENT"],
        "verification_status" => ["Verified", "Source Verified", "Not Verified", "Source Verified", "Source Verified"],
    ].unwrap()
}

#[test]
fn test_load_csv_and_minmax_scale() {
    let mut df = make_test_dataframe();
    let mut scaler = MinMaxScaler::new((0.0, 1.0));
    scaler
        .fit_transform(&mut df, &["loan_amnt", "annual_inc"])
        .unwrap();

    for col_name in ["loan_amnt", "annual_inc"] {
        let ca = df.column(col_name).unwrap().f64().unwrap();
        assert!(ca.min().unwrap() >= -1e-6);
        assert!(ca.max().unwrap() <= 1.0 + 1e-6);
    }
}

#[test]
fn test_minmax_column_not_found_warns_and_continues() {
    let mut df = make_test_dataframe();
    let mut scaler = MinMaxScaler::new((0.0, 1.0));
    let result = scaler.fit_transform(&mut df, &["loan_amnt", "nonexistent"]);
    assert!(result.is_ok());
}

#[test]
fn test_minmax_wrong_type_warns_and_continues() {
    let mut df = make_test_dataframe();
    let mut scaler = MinMaxScaler::new((0.0, 1.0));
    let result = scaler.fit_transform(&mut df, &["loan_amnt", "grade"]);
    assert!(result.is_ok());
}

#[test]
fn test_load_csv_and_standard_scale() {
    let mut df = make_test_dataframe();
    let mut scaler = StandardScaler::new();
    scaler
        .fit_transform(&mut df, &["loan_amnt", "annual_inc"])
        .unwrap();

    const EPSILON: f64 = 1e-2;
    for col_name in ["loan_amnt", "annual_inc"] {
        let ca = df.column(col_name).unwrap().f64().unwrap();
        assert!(ca.mean().unwrap().abs() < EPSILON);
        assert!((ca.std(1).unwrap() - 1.0).abs() < EPSILON);
    }
}

#[test]
fn test_load_csv_and_label_encode() {
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
fn test_load_csv_and_onehot_encode() {
    let mut df = make_test_dataframe();
    let original_width = df.width();

    let mut encoder = OneHotEncoder::new();
    encoder.fit_transform(&mut df, &["home_ownership"]).unwrap();

    assert!(df.width() > original_width);
    assert!(df.column("home_ownership_RENT").is_ok());
}
