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
