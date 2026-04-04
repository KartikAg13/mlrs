use criterion::{Criterion, criterion_group, criterion_main};
use mlrs::dataset::preprocessing::encoding::label::LabelEncoder;
use mlrs::dataset::preprocessing::encoding::one_hot::OneHotEncoder;
use mlrs::dataset::preprocessing::scaling::min_max::MinMaxScaler;
use mlrs::dataset::preprocessing::scaling::standard::StandardScaler;
use mlrs::dataset::{CSVConfig, read_csv};

const SAMPLE_FILEPATH: &str = "tests/fixtures/sample_1M.csv";
const NUMERIC_COLS: [&str; 4] = ["loan_amnt", "funded_amnt", "installment", "annual_inc"];
const LABEL_COLS: [&str; 3] = ["grade", "home_ownership", "verification_status"];
const ONEHOT_COLS: [&str; 1] = ["home_ownership"];

fn load_bench_df() -> polars::frame::DataFrame {
    let cfg = CSVConfig::new(SAMPLE_FILEPATH).with_ignore_errors(true);
    read_csv(cfg)
}

fn bench_load_csv(c: &mut Criterion) {
    c.bench_function("bench_load_csv_1M", |b| b.iter(|| load_bench_df()));
}

fn bench_standard_scaler(c: &mut Criterion) {
    let df = load_bench_df();
    c.bench_function("standard_scaler_fit_transform_1M", |b| {
        b.iter(|| {
            let mut df_clone = df.clone();
            let mut scaler = StandardScaler::new();
            scaler.fit_transform(&mut df_clone, &NUMERIC_COLS).unwrap();
        })
    });
}

fn bench_minmax_scaler(c: &mut Criterion) {
    let df = load_bench_df();
    c.bench_function("minmax_scaler_fit_transform_1M", |b| {
        b.iter(|| {
            let mut df_clone = df.clone();
            let mut scaler = MinMaxScaler::new((0.0, 1.0));
            scaler.fit_transform(&mut df_clone, &NUMERIC_COLS).unwrap();
        })
    });
}

fn bench_label_encoder(c: &mut Criterion) {
    let df = load_bench_df();
    c.bench_function("label_encoder_fit_transform_1M", |b| {
        b.iter(|| {
            let mut df_clone = df.clone();
            let mut encoder = LabelEncoder::new();
            encoder.fit_transform(&mut df_clone, &LABEL_COLS).unwrap();
        })
    });
}

fn bench_onehot_encoder(c: &mut Criterion) {
    let df = load_bench_df();
    c.bench_function("onehot_encoder_fit_transform_1M", |b| {
        b.iter(|| {
            let mut df_clone = df.clone();
            let mut encoder = OneHotEncoder::new();
            encoder.fit_transform(&mut df_clone, &ONEHOT_COLS).unwrap();
        })
    });
}

criterion_group!(
    benches,
    bench_load_csv,
    bench_standard_scaler,
    bench_minmax_scaler,
    bench_label_encoder,
    bench_onehot_encoder
);
criterion_main!(benches);
