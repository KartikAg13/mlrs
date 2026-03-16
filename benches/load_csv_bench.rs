use criterion::{Criterion, criterion_group, criterion_main};
use mlrs::dataset::preprocessing::scaling::standard::StandardScaler;
use mlrs::dataset::read_csv;

const SAMPLE_FILEPATH: &str = "tests/fixtures/sample.csv";

fn bench_load_csv(c: &mut Criterion) {
    c.bench_function("bench_load_csv_1M", |b| {
        b.iter(|| read_csv(SAMPLE_FILEPATH))
    });
}

fn bench_standard_scaler(c: &mut Criterion) {
    let df = read_csv(SAMPLE_FILEPATH);
    let columns = ["age", "salary", "value", "id"];

    c.bench_function("standard_scaler_fit_transform_1M", |b| {
        b.iter(|| {
            let mut df_clone = df.clone();
            let mut scaler = StandardScaler::new();
            scaler.fit_transform(&mut df_clone, &columns).unwrap();
        })
    });
}

criterion_group!(benches, bench_load_csv, bench_standard_scaler);
criterion_main!(benches);
