use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use mlrs::dataset::{CSVReader, read_csv};

const SAMPLE: &str = "tests/fixtures/sample_1M.csv";

fn bench_default(c: &mut Criterion) {
    c.bench_function("read_csv/default", |b| {
        b.iter(|| {
            let df = read_csv(black_box(SAMPLE));
            black_box(df);
        })
    });
}

fn bench_threads(c: &mut Criterion) {
    let mut group = c.benchmark_group("threads");

    for threads in [1usize, 2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::from_parameter(threads),
            &threads,
            |b, &threads| {
                b.iter(|| {
                    let df = CSVReader::new(black_box(SAMPLE)).threads(threads).finish();

                    black_box(df);
                });
            },
        );
    }

    group.finish();
}

fn bench_chunk_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_size");

    for size in [4096usize, 65536, 1 << 19, 1 << 20] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| {
                let df = CSVReader::new(black_box(SAMPLE)).chunk_size(size).finish();

                black_box(df);
            });
        });
    }

    group.finish();
}

fn bench_rechunk(c: &mut Criterion) {
    let mut group = c.benchmark_group("rechunk");

    group.bench_function("false", |b| {
        b.iter(|| {
            black_box(CSVReader::new(SAMPLE).rechunk(false).finish());
        });
    });

    group.bench_function("true", |b| {
        b.iter(|| {
            black_box(CSVReader::new(SAMPLE).rechunk(true).finish());
        });
    });

    group.finish();
}

fn bench_low_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("low_memory");

    group.bench_function("false", |b| {
        b.iter(|| {
            black_box(CSVReader::new(SAMPLE).low_memory(false).finish());
        });
    });

    group.bench_function("true", |b| {
        b.iter(|| {
            black_box(CSVReader::new(SAMPLE).low_memory(true).finish());
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_default,
    bench_threads,
    bench_chunk_size,
    bench_rechunk,
    bench_low_memory
);

criterion_main!(benches);
