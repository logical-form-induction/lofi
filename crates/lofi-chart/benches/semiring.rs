//! Instruction counts for each native semiring carrier.

use gungraun::main;

mod groups {
    use std::convert::Infallible;
    use std::hint::black_box;
    use std::num::NonZeroUsize;

    use gungraun::{library_benchmark, library_benchmark_group};
    use lofi_chart::semiring::{
        Boolean, Count, Counting, Expectation, KBest, LogInside, Semiring, Viterbi,
    };

    fn infallible<T>(result: Result<T, Infallible>) -> T {
        match result {
            Ok(value) => value,
            Err(error) => match error {},
        }
    }

    #[library_benchmark]
    fn counting() -> Count {
        infallible(Counting.times(&black_box(Count::Small(u64::MAX)), &Count::Small(2)))
    }

    #[library_benchmark]
    fn viterbi() -> f64 {
        let mut accumulator = black_box(-2.0);
        infallible(Viterbi.plus_assign(&mut accumulator, &-3.0));
        accumulator
    }

    #[library_benchmark]
    fn log_inside() -> f64 {
        let mut accumulator = black_box(-2.0);
        infallible(LogInside.plus_assign(&mut accumulator, &-3.0));
        accumulator
    }

    #[library_benchmark]
    fn boolean() -> bool {
        infallible(Boolean.times(&black_box(true), &true))
    }

    #[library_benchmark]
    fn k_best() -> Box<[f64]> {
        let Some(width) = NonZeroUsize::new(4) else {
            unreachable!("four is positive")
        };
        let left: Box<[f64]> = Box::new([-1.0, -2.0]);
        let right: Box<[f64]> = Box::new([-3.0, -4.0]);
        infallible(KBest::new(width).times(&black_box(left), &right))
    }

    #[library_benchmark]
    fn expectation() -> (f64, Box<[f64]>) {
        let Some(dim) = NonZeroUsize::new(2) else {
            unreachable!("two is positive")
        };
        infallible(Expectation::new(dim).times(
            &black_box((0.5, Box::new([1.0, 0.0]))),
            &(0.4, Box::new([0.0, 2.0])),
        ))
    }

    library_benchmark_group!(name = semiring; benchmarks = counting, viterbi, log_inside, boolean, k_best, expectation);
}

use groups::semiring;

main!(library_benchmark_groups = semiring);
