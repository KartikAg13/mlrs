// tests/preprocessing_tests.rs
use mlrs::dataset::load_csv;
use mlrs::dataset::preprocessing::scaling::standard_scaler::StandardScaler;

use polars::prelude::*;

#[test]
fn test_load_csv_and_scale() {
    let mut df = load_csv("tests/fixtures/sample.csv").unwrap();
    let original_shape = (df.height(), df.width());

    let mut scaler = StandardScaler::new();
    scaler.fit_transform(&mut df, &["age", "salary"]).unwrap();

    // shape preserved
    assert_eq!(df.height(), original_shape.0);
    assert_eq!(df.width(), original_shape.1);

    // dtypes of scaled columns are Float64
    assert_eq!(*df.column("age").unwrap().dtype(), DataType::Float64);
    assert_eq!(*df.column("salary").unwrap().dtype(), DataType::Float64);

    // mean~0, std~1 for scaled columns
    for col_name in ["age", "salary"] {
        let col = df
            .column(col_name)
            .unwrap()
            .cast(&DataType::Float64)
            .unwrap();
        let ca = col.f64().unwrap();
        assert!(ca.mean().unwrap().abs() < 1e-2);
        assert!((ca.std(1).unwrap() - 1.0).abs() < 1e-2);
    }
}
