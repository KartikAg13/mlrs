use criterion::{Criterion, criterion_group, criterion_main};
use mlrs::dataset::raw_dataset::load_csv;

const SAMPLE_FILEPATH: &str = "tests/fixtures/sample.csv";

fn bench_load_csv(c: &mut Criterion) {
    c.bench_function("bench_load_csv_1M", |b| {
        b.iter(|| load_csv(SAMPLE_FILEPATH))
    });
}

criterion_group!(benches, bench_load_csv);
criterion_main!(benches);
