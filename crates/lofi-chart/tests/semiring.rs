//! Native semirings through the core crate's public interface.

use std::convert::Infallible;
use std::num::NonZeroUsize;

use lofi_chart::semiring::{Count, Counting, KBest, Semiring};
use num_bigint::BigUint;

fn infallible<T>(result: Result<T, Infallible>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => match error {},
    }
}

#[test]
fn counting_is_exact_beyond_a_machine_word() {
    let product = infallible(Counting.times(&Count::Small(1_u64 << 63), &Count::Small(4)));
    assert_eq!(
        product.into_big_uint(),
        BigUint::from(1_u8) << 65_usize,
        "the public carrier promotes instead of overflowing"
    );
}

#[test]
fn k_best_retains_duplicate_scores() {
    let Some(width) = NonZeroUsize::new(3) else {
        unreachable!("three is positive")
    };
    let mut alternatives: Box<[f64]> = Box::new([-1.0]);
    let other: Box<[f64]> = Box::new([-1.0, -2.0]);
    infallible(KBest::new(width).plus_assign(&mut alternatives, &other));
    assert_eq!(
        &*alternatives,
        &[-1.0, -1.0, -2.0],
        "equal-scoring derivations remain distinct"
    );
}
