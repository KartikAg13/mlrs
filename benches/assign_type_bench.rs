use criterion::{Criterion, criterion_group, criterion_main};
use mlrs::dataset::raw_dataset::{assign_type, assign_type_parallel};

fn bench_sequential(c: &mut Criterion) {
    c.bench_function("bench_assign_type_1M", |b| {
        b.iter(|| assign_type("tests/fixtures/sample.csv".to_string(), Some(',')));
    });
}

fn bench_parallel(c: &mut Criterion) {
    c.bench_function("bench_parallel_assign_type_1M", |b| {
        b.iter(|| assign_type_parallel("tests/fixtures/sample.csv".to_string(), Some(',')));
    });
}

criterion_group!(benches, bench_sequential, bench_parallel);
criterion_main!(benches);
