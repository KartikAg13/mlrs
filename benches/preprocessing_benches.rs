//! Benchmark suite for mlrs preprocessing pipeline.
//!
//! Run individual groups to avoid crashes:
//!   cargo bench --bench preprocessing_benches csv_loading
//!   cargo bench --bench preprocessing_benches scaling
//!   cargo bench --bench preprocessing_benches encoding
//!   cargo bench --bench preprocessing_benches imputation
//!   cargo bench --bench preprocessing_benches full_pipeline

use criterion::{Criterion, criterion_group, criterion_main};
use mlrs::dataset::preprocesser::encoder::{LabelEncoder, OneHotEncoder};
use mlrs::dataset::preprocesser::handler::{SimpleImputer, Strategy};
use mlrs::dataset::preprocesser::scaler::{MinMaxScaler, StandardScaler};
use mlrs::dataset::{CSVConfig, read_csv};
use polars::prelude::*;
use std::hint::black_box;

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

const LARGE_CSV: &str = "tests/fixtures/sample.csv";
const NUMERIC_COLS: [&str; 2] = ["loan_amnt", "annual_inc"];
const CATEGORICAL_COLS: [&str; 2] = ["grade", "home_ownership"];

// -----------------------------------------------------------------------------
// CSV Loading
// -----------------------------------------------------------------------------

fn bench_csv_loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("csv_loading");

    group.bench_function("read_csv_rows", |b| {
        b.iter(|| {
            let cfg = CSVConfig::new(black_box(LARGE_CSV))
                .with_ignore_errors(true)
                .with_n_threads(num_cpus::get());
            black_box(read_csv(cfg));
        })
    });

    group.finish();
}

// -----------------------------------------------------------------------------
// Scaling
// -----------------------------------------------------------------------------

fn bench_scaling(c: &mut Criterion) {
    let df = make_small_test_df(); // Use small DF to avoid crashes

    let mut group = c.benchmark_group("scaling");

    group.bench_function("minmax_scaler_fit_transform", |b| {
        b.iter_batched(
            || df.clone(),
            |mut df_clone| {
                let mut scaler = MinMaxScaler::new((0.0, 1.0));
                black_box(scaler.fit_transform(&mut df_clone, &NUMERIC_COLS).unwrap());
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("standard_scaler_fit_transform", |b| {
        b.iter_batched(
            || df.clone(),
            |mut df_clone| {
                let mut scaler = StandardScaler::new();
                black_box(scaler.fit_transform(&mut df_clone, &NUMERIC_COLS).unwrap());
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
    let df = make_small_test_df();

    let mut group = c.benchmark_group("encoding");

    group.bench_function("label_encoder_fit_transform", |b| {
        b.iter_batched(
            || df.clone(),
            |mut df_clone| {
                let mut encoder = LabelEncoder::new();
                black_box(
                    encoder
                        .fit_transform(&mut df_clone, &CATEGORICAL_COLS)
                        .unwrap(),
                );
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.bench_function("onehot_encoder_fit_transform", |b| {
        b.iter_batched(
            || df.clone(),
            |mut df_clone| {
                let mut encoder = OneHotEncoder::new();
                black_box(
                    encoder
                        .fit_transform(&mut df_clone, &["home_ownership"])
                        .unwrap(),
                );
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
                .into_iter()
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
                black_box(
                    SimpleImputer::fill_null(&mut df_clone, &[("annual_inc", Strategy::Mean)])
                        .unwrap(),
                );
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
                let _ = SimpleImputer::fill_null(&mut df_clone, &[("annual_inc", Strategy::Mean)]);

                // Encode
                let mut label_encoder = LabelEncoder::new();
                let _ = label_encoder.fit_transform(&mut df_clone, &CATEGORICAL_COLS);

                // Scale
                let mut scaler = StandardScaler::new();
                let _ = scaler.fit_transform(&mut df_clone, &NUMERIC_COLS);

                black_box(df_clone);
            },
            criterion::BatchSize::SmallInput,
        )
    });
}

// -----------------------------------------------------------------------------
// Groups — run one at a time
// -----------------------------------------------------------------------------

criterion_group!(csv_loading, bench_csv_loading);
criterion_group!(scaling, bench_scaling);
criterion_group!(encoding, bench_encoding);
criterion_group!(imputation, bench_imputation);
criterion_group!(full_pipeline, bench_full_pipeline);

criterion_main!(csv_loading, scaling, encoding, imputation, full_pipeline);
