//! Benchmark suite for mlrs preprocessing pipeline.
//!
//! Run individual groups to avoid crashes:
//!   cargo bench --bench preprocessing_benches csv_loading
//!   cargo bench --bench preprocessing_benches scaling
//!   cargo bench --bench preprocessing_benches encoding
//!   cargo bench --bench preprocessing_benches imputation
//!   cargo bench --bench preprocessing_benches full_pipeline

use criterion::{Criterion, criterion_group, criterion_main};
use polars::prelude::*;
use std::hint::black_box;

use mlrs::dataset::preprocessor::encoder::{LabelEncoder, OneHotEncoder};
use mlrs::dataset::preprocessor::handler::{ImputerStrategy, SimpleImputer};
use mlrs::dataset::preprocessor::scaler::{MinMaxScaler, StandardScaler};
// use mlrs::dataset::{CSVReader, read_csv};

// -----------------------------------------------------------------------------
// Small in-memory test DataFrame (fast, no disk I/O)
// -----------------------------------------------------------------------------

fn make_bench_dataframe() -> DataFrame {
    let n = 50_000;
    df![
        "loan_amnt" => (0..n).map(|i| (i % 10000) as f64).collect::<Vec<_>>(),
        "annual_inc" => (0..n).map(|i| 30000.0 + (i % 70000) as f64).collect::<Vec<_>>(),
        "grade" => (0..n).map(|i| match i%5 { 0=>"A", 1=>"B", 2=>"C", 3=>"D", _=>"E" }).collect::<Vec<_>>(),
        "home_ownership" => vec!["RENT"; n],
    ].unwrap()
}

fn make_small_test_df() -> DataFrame {
    df![
        "loan_amnt"   => [5000.0, 2500.0, 2400.0, 10000.0, 3000.0],
        "annual_inc"  => [24000.0, 30000.0, 12252.0, 49200.0, 80000.0],
        "grade"       => ["B", "C", "C", "C", "B"],
        "home_ownership" => ["RENT", "RENT", "RENT", "RENT", "RENT"],
    ]
    .unwrap()
}

// const LARGE_CSV: &str = "tests/fixtures/sample.csv";
const NUMERIC_COLS: [&str; 2] = ["loan_amnt", "annual_inc"];
const CATEGORICAL_COLS: [&str; 2] = ["grade", "home_ownership"];

// -----------------------------------------------------------------------------
// Scaling
// -----------------------------------------------------------------------------

fn bench_scaling(c: &mut Criterion) {
    let df = make_bench_dataframe();

    let mut group = c.benchmark_group("scaling");

    group.bench_function("minmax_scaler_fit_transform", |b| {
        b.iter_batched(
            || df.clone(),
            |mut df_clone| {
                let mut scaler = MinMaxScaler::new((0.0, 1.0));
                scaler.fit_transform(&mut df_clone, &NUMERIC_COLS).unwrap();
                black_box(());
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("standard_scaler_fit_transform", |b| {
        b.iter_batched(
            || df.clone(),
            |mut df_clone| {
                let mut scaler = StandardScaler::new();
                scaler.fit_transform(&mut df_clone, &NUMERIC_COLS).unwrap();
                black_box(());
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

// -----------------------------------------------------------------------------
// Encoding
// -----------------------------------------------------------------------------

fn bench_encoding(c: &mut Criterion) {
    let df = make_bench_dataframe();

    let mut group = c.benchmark_group("encoding");

    group.bench_function("label_encoder_fit_transform", |b| {
        b.iter_batched(
            || df.clone(),
            |mut df_clone| {
                let mut encoder = LabelEncoder::new();
                encoder
                    .fit_transform(&mut df_clone, &CATEGORICAL_COLS)
                    .unwrap();
                black_box(());
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("onehot_encoder_fit_transform", |b| {
        b.iter_batched(
            || df.clone(),
            |mut df_clone| {
                let mut encoder = OneHotEncoder::new();
                encoder
                    .fit_transform(&mut df_clone, &["home_ownership"])
                    .unwrap();
                black_box(());
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

// -----------------------------------------------------------------------------
// Imputation
// -----------------------------------------------------------------------------

fn bench_imputation(c: &mut Criterion) {
    let mut group = c.benchmark_group("imputation");

    let df_with_nulls = {
        let mut df = make_small_test_df();
        // Introduce a few nulls
        let _ = df.replace_column(
            df.get_column_index("annual_inc").unwrap(),
            df.column("annual_inc")
                .unwrap()
                .f64()
                .unwrap()
                .iter()
                .map(|v| if v.unwrap_or(0.0) > 40000.0 { None } else { v })
                .collect::<Float64Chunked>()
                .into_series()
                .with_name("annual_inc".into())
                .into(),
        );
        df
    };

    group.bench_function("simple_imputer_mean", |b| {
        b.iter_batched(
            || df_with_nulls.clone(),
            |mut df_clone| {
                SimpleImputer::fill_null(&mut df_clone, &[("annual_inc", ImputerStrategy::Mean)])
                    .unwrap();
                black_box(());
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

// -----------------------------------------------------------------------------
// Full Pipeline (lightweight version)
// -----------------------------------------------------------------------------

fn bench_full_pipeline(c: &mut Criterion) {
    let df = make_bench_dataframe();

    c.bench_function("full_preprocessing_pipeline", |b| {
        b.iter_batched(
            || df.clone(),
            |mut df_clone| {
                // Impute
                SimpleImputer::fill_null(&mut df_clone, &[("annual_inc", ImputerStrategy::Mean)])
                    .unwrap();

                // Encode
                let mut label_encoder = LabelEncoder::new();
                label_encoder
                    .fit_transform(&mut df_clone, &CATEGORICAL_COLS)
                    .unwrap();

                // Scale
                let mut scaler = StandardScaler::new();
                scaler.fit_transform(&mut df_clone, &NUMERIC_COLS).unwrap();

                black_box(df_clone);
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

// -----------------------------------------------------------------------------
// Dropper
// -----------------------------------------------------------------------------

fn bench_dropper(c: &mut Criterion) {
    let mut group = c.benchmark_group("dropper");

    let df_with_nulls = {
        let n = 50_000;
        let data: Vec<Option<f64>> = (0..n)
            .map(|i| if i % 10 == 0 { None } else { Some(i as f64) })
            .collect();
        df!["value" => data, "name" => vec!["test"; n]].unwrap()
    };

    group.bench_function("drop_rows_any", |b| {
        b.iter_batched(
            || df_with_nulls.clone(),
            |mut df_clone| {
                use mlrs::dataset::preprocessor::handler::Dropper;
                Dropper::drop_rows_any(&mut df_clone, &["value"]).unwrap();
                black_box(());
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("drop_columns", |b| {
        b.iter_batched(
            || df_with_nulls.clone(),
            |mut df_clone| {
                use mlrs::dataset::preprocessor::handler::Dropper;
                Dropper::drop_columns(&mut df_clone, &["name"]).unwrap();
                black_box(());
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

// -----------------------------------------------------------------------------
// Groups — run one at a time
// -----------------------------------------------------------------------------

criterion_group!(scaling, bench_scaling);
criterion_group!(encoding, bench_encoding);
criterion_group!(imputation, bench_imputation);
criterion_group!(full_pipeline, bench_full_pipeline);
criterion_group!(dropper, bench_dropper);

criterion_main!(scaling, encoding, imputation, full_pipeline, dropper);
