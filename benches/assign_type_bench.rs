use criterion::{Criterion, criterion_group, criterion_main};
use mlrs::dataset::raw_dataset::assign_type;

fn bench_assign_type_basic(c: &mut Criterion) {
    c.bench_function("assign_type_1M_rows", |b| {
        b.iter(|| assign_type("tests/fixtures/sample.csv".to_string(), Some(',')));
    });
}

criterion_group!(benches, bench_assign_type_basic);
criterion_main!(benches);
