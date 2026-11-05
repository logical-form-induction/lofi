//! Native chart-weight semirings.

use std::cmp::Ordering;
use std::convert::Infallible;
use std::num::NonZeroUsize;

use num_bigint::BigUint;

/// One algebra's native carrier.
///
/// Native carriers use [`Infallible`] as their error. The chart kernel also
/// implements this interface for Python-carried weights, whose operations can
/// raise Python exceptions.
pub trait Semiring {
    /// A weight carried by the algebra.
    type Weight: Clone;
    /// An error raised by an arithmetic operation.
    type Error;

    /// Return the additive identity.
    fn zero(&self) -> Self::Weight;

    /// Return the multiplicative identity.
    fn one(&self) -> Self::Weight;

    /// Add one alternative into an accumulated weight.
    ///
    /// # Errors
    ///
    /// Returns the carrier's error if it cannot add the weight.
    fn plus_assign(
        &self,
        accumulator: &mut Self::Weight,
        weight: &Self::Weight,
    ) -> Result<(), Self::Error>;

    /// Multiply two antecedent weights.
    ///
    /// # Errors
    ///
    /// Returns the carrier's error if it cannot multiply the weights.
    fn times(&self, left: &Self::Weight, right: &Self::Weight)
    -> Result<Self::Weight, Self::Error>;
}

/// An exact derivation count with an allocation-free small representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Count {
    /// A count representable by an unsigned machine-independent 64-bit value.
    Small(u64),
    /// A count promoted after addition or multiplication overflows 64 bits.
    Big(BigUint),
}

impl Count {
    /// Convert the count to its arbitrary-precision representation.
    #[must_use]
    pub fn into_big_uint(self) -> BigUint {
        match self {
            Self::Small(value) => BigUint::from(value),
            Self::Big(value) => value,
        }
    }

    /// Return whether this is the additive identity.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        match self {
            Self::Small(value) => *value == 0,
            Self::Big(value) => *value == BigUint::from(0_u8),
        }
    }
}

impl From<u64> for Count {
    fn from(value: u64) -> Self {
        Self::Small(value)
    }
}

impl From<BigUint> for Count {
    fn from(value: BigUint) -> Self {
        u64::try_from(&value).map_or(Self::Big(value), Self::Small)
    }
}

/// Count derivations exactly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Counting;

impl Counting {
    /// Return the algebra's selection name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        "counting"
    }

    /// Report whether weights are log probabilities that `plus` sums by log-sum-exp.
    #[must_use]
    pub const fn log_probability(self) -> bool {
        false
    }

    /// Return whether a count is the additive identity.
    #[must_use]
    pub fn is_zero(self, weight: &Count) -> bool {
        weight.is_zero()
    }
}

impl Semiring for Counting {
    type Weight = Count;
    type Error = Infallible;

    fn zero(&self) -> Self::Weight {
        Count::Small(0)
    }

    fn one(&self) -> Self::Weight {
        Count::Small(1)
    }

    fn plus_assign(
        &self,
        accumulator: &mut Self::Weight,
        weight: &Self::Weight,
    ) -> Result<(), Self::Error> {
        match accumulator {
            Count::Small(left) => match weight {
                Count::Small(right) => {
                    if let Some(sum) = left.checked_add(*right) {
                        *left = sum;
                    } else {
                        *accumulator = Count::Big(BigUint::from(*left) + BigUint::from(*right));
                    }
                }
                Count::Big(right) => {
                    *accumulator = Count::Big(BigUint::from(*left) + right);
                }
            },
            Count::Big(left) => match weight {
                Count::Small(right) => *left += *right,
                Count::Big(right) => *left += right,
            },
        }
        Ok(())
    }

    fn times(
        &self,
        left: &Self::Weight,
        right: &Self::Weight,
    ) -> Result<Self::Weight, Self::Error> {
        let product = match (left, right) {
            (Count::Small(a), Count::Small(b)) => a.checked_mul(*b).map_or_else(
                || Count::Big(BigUint::from(*a) * BigUint::from(*b)),
                Count::Small,
            ),
            (Count::Small(a), Count::Big(b)) => Count::Big(BigUint::from(*a) * b),
            (Count::Big(a), Count::Small(b)) => Count::Big(a * BigUint::from(*b)),
            (Count::Big(a), Count::Big(b)) => Count::Big(a * b),
        };
        Ok(product)
    }
}

/// Keep the best derivation's log probability.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Viterbi;

impl Viterbi {
    /// Return the algebra's selection name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        "viterbi"
    }

    /// Report whether weights are log probabilities that `plus` sums by log-sum-exp.
    #[must_use]
    pub const fn log_probability(self) -> bool {
        false
    }

    /// Return whether a log probability is the additive identity.
    #[must_use]
    pub fn is_zero(self, weight: f64) -> bool {
        weight == f64::NEG_INFINITY
    }
}

impl Semiring for Viterbi {
    type Weight = f64;
    type Error = Infallible;

    fn zero(&self) -> Self::Weight {
        f64::NEG_INFINITY
    }

    fn one(&self) -> Self::Weight {
        0.0
    }

    fn plus_assign(
        &self,
        accumulator: &mut Self::Weight,
        weight: &Self::Weight,
    ) -> Result<(), Self::Error> {
        if weight > accumulator {
            *accumulator = *weight;
        }
        Ok(())
    }

    fn times(
        &self,
        left: &Self::Weight,
        right: &Self::Weight,
    ) -> Result<Self::Weight, Self::Error> {
        Ok(left + right)
    }
}

/// Accumulate inside probabilities in log space.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct LogInside;

impl LogInside {
    /// Return the algebra's selection name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        "log-inside"
    }

    /// Report whether weights are log probabilities that `plus` sums by log-sum-exp.
    #[must_use]
    pub const fn log_probability(self) -> bool {
        true
    }

    /// Return whether a log probability is the additive identity.
    #[must_use]
    pub fn is_zero(self, weight: f64) -> bool {
        weight == f64::NEG_INFINITY
    }
}

impl Semiring for LogInside {
    type Weight = f64;
    type Error = Infallible;

    fn zero(&self) -> Self::Weight {
        f64::NEG_INFINITY
    }

    fn one(&self) -> Self::Weight {
        0.0
    }

    fn plus_assign(
        &self,
        accumulator: &mut Self::Weight,
        weight: &Self::Weight,
    ) -> Result<(), Self::Error> {
        if *accumulator == f64::NEG_INFINITY {
            *accumulator = *weight;
        } else if *weight != f64::NEG_INFINITY {
            *accumulator = if *accumulator > *weight {
                *accumulator + (weight - *accumulator).exp().ln_1p()
            } else {
                *weight + (*accumulator - weight).exp().ln_1p()
            };
        }
        Ok(())
    }

    fn times(
        &self,
        left: &Self::Weight,
        right: &Self::Weight,
    ) -> Result<Self::Weight, Self::Error> {
        Ok(left + right)
    }
}

/// Track whether an item is derivable.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Boolean;

impl Boolean {
    /// Return the algebra's selection name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        "boolean"
    }

    /// Report whether weights are log probabilities that `plus` sums by log-sum-exp.
    #[must_use]
    pub const fn log_probability(self) -> bool {
        false
    }

    /// Return whether a derivability weight is the additive identity.
    #[must_use]
    pub const fn is_zero(self, weight: bool) -> bool {
        !weight
    }
}

impl Semiring for Boolean {
    type Weight = bool;
    type Error = Infallible;

    fn zero(&self) -> Self::Weight {
        false
    }

    fn one(&self) -> Self::Weight {
        true
    }

    fn plus_assign(
        &self,
        accumulator: &mut Self::Weight,
        weight: &Self::Weight,
    ) -> Result<(), Self::Error> {
        *accumulator |= *weight;
        Ok(())
    }

    fn times(
        &self,
        left: &Self::Weight,
        right: &Self::Weight,
    ) -> Result<Self::Weight, Self::Error> {
        Ok(*left && *right)
    }
}

/// Keep the highest `k` derivation log probabilities.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KBest {
    k: NonZeroUsize,
}

impl KBest {
    /// Construct an algebra retaining `k` derivations.
    #[must_use]
    pub const fn new(k: NonZeroUsize) -> Self {
        Self { k }
    }

    /// Return the number of derivations retained by each weight.
    #[must_use]
    pub const fn k(self) -> NonZeroUsize {
        self.k
    }

    /// Return the algebra's selection name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        "k-best"
    }

    /// Report whether weights are log probabilities that `plus` sums by log-sum-exp.
    #[must_use]
    pub const fn log_probability(self) -> bool {
        false
    }

    /// Return whether a ranking is the additive identity.
    #[must_use]
    pub const fn is_zero(self, weight: &[f64]) -> bool {
        weight.is_empty()
    }

    fn ranked(self, mut scores: Vec<f64>) -> Box<[f64]> {
        scores.sort_by(|left, right| right.partial_cmp(left).unwrap_or(Ordering::Equal));
        scores.truncate(self.k.get());
        scores.into_boxed_slice()
    }
}

impl Semiring for KBest {
    type Weight = Box<[f64]>;
    type Error = Infallible;

    fn zero(&self) -> Self::Weight {
        Box::new([])
    }

    fn one(&self) -> Self::Weight {
        Box::new([0.0])
    }

    fn plus_assign(
        &self,
        accumulator: &mut Self::Weight,
        weight: &Self::Weight,
    ) -> Result<(), Self::Error> {
        let mut alternatives = Vec::with_capacity(accumulator.len() + weight.len());
        alternatives.extend_from_slice(accumulator);
        alternatives.extend_from_slice(weight);
        *accumulator = self.ranked(alternatives);
        Ok(())
    }

    fn times(
        &self,
        left: &Self::Weight,
        right: &Self::Weight,
    ) -> Result<Self::Weight, Self::Error> {
        let mut products = Vec::with_capacity(left.len().saturating_mul(right.len()));
        for a in left {
            for b in right {
                products.push(a + b);
            }
        }
        Ok(self.ranked(products))
    }
}

/// A probability and its accumulated expectation coordinates.
pub type ExpectedWeight = (f64, Box<[f64]>);

/// Accumulate expectations for a fixed number of coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Expectation {
    dim: NonZeroUsize,
}

impl Expectation {
    /// Construct an expectation algebra with `dim` coordinates.
    #[must_use]
    pub const fn new(dim: NonZeroUsize) -> Self {
        Self { dim }
    }

    /// Return the number of coordinates in each weight.
    #[must_use]
    pub const fn dim(self) -> NonZeroUsize {
        self.dim
    }

    /// Return the algebra's selection name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        "expectation"
    }

    /// Report whether weights are log probabilities that `plus` sums by log-sum-exp.
    #[must_use]
    pub const fn log_probability(self) -> bool {
        false
    }

    /// Return whether a weight is the additive identity.
    #[must_use]
    pub fn is_zero(self, weight: &ExpectedWeight) -> bool {
        weight.1.len() == self.dim.get()
            && weight.0 == 0.0
            && weight.1.iter().all(|coordinate| *coordinate == 0.0)
    }
}

impl Semiring for Expectation {
    type Weight = ExpectedWeight;
    type Error = Infallible;

    fn zero(&self) -> Self::Weight {
        (0.0, vec![0.0; self.dim.get()].into_boxed_slice())
    }

    fn one(&self) -> Self::Weight {
        (1.0, vec![0.0; self.dim.get()].into_boxed_slice())
    }

    fn plus_assign(
        &self,
        accumulator: &mut Self::Weight,
        weight: &Self::Weight,
    ) -> Result<(), Self::Error> {
        accumulator.0 += weight.0;
        for (left, right) in accumulator.1.iter_mut().zip(&weight.1) {
            *left += right;
        }
        Ok(())
    }

    fn times(
        &self,
        left: &Self::Weight,
        right: &Self::Weight,
    ) -> Result<Self::Weight, Self::Error> {
        let expectations = left
            .1
            .iter()
            .zip(&right.1)
            .map(|(a, b)| left.0 * b + right.0 * a)
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Ok((left.0 * right.0, expectations))
    }
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use std::convert::Infallible;
    use std::num::NonZeroUsize;

    use num_bigint::BigUint;
    use proptest::prelude::*;

    use super::{Boolean, Count, Counting, Expectation, KBest, LogInside, Semiring, Viterbi};

    fn infallible<T>(result: Result<T, Infallible>) -> T {
        match result {
            Ok(value) => value,
            Err(error) => match error {},
        }
    }

    fn plus<S>(semiring: &S, mut left: S::Weight, right: &S::Weight) -> S::Weight
    where
        S: Semiring<Error = Infallible>,
    {
        infallible(semiring.plus_assign(&mut left, right));
        left
    }

    fn kbest(k: usize) -> KBest {
        let Some(width) = NonZeroUsize::new(k) else {
            unreachable!("the test widths are positive")
        };
        KBest::new(width)
    }

    fn expectation(dim: usize) -> Expectation {
        let Some(width) = NonZeroUsize::new(dim) else {
            unreachable!("the test dimensions are positive")
        };
        Expectation::new(width)
    }

    #[test]
    fn only_log_inside_declares_log_probability_weights() {
        assert!(LogInside.log_probability());
        assert!(!Viterbi.log_probability());
        assert!(!Boolean.log_probability());
        assert!(!Counting.log_probability());
        assert!(!kbest(2).log_probability());
        assert!(!expectation(1).log_probability());
    }

    proptest! {
        #[test]
        fn counting_obeys_semiring_laws(a in any::<u64>(), b in any::<u64>(), c in any::<u64>()) {
            let semiring = Counting;
            let a = Count::from(a);
            let b = Count::from(b);
            let c = Count::from(c);
            prop_assert_eq!(plus(&semiring, a.clone(), &b), plus(&semiring, b.clone(), &a));
            prop_assert_eq!(
                plus(&semiring, plus(&semiring, a.clone(), &b), &c),
                plus(&semiring, a.clone(), &plus(&semiring, b.clone(), &c)),
            );
            prop_assert_eq!(&plus(&semiring, a.clone(), &semiring.zero()), &a);
            prop_assert_eq!(&infallible(semiring.times(&a, &semiring.one())), &a);
            prop_assert_eq!(infallible(semiring.times(&a, &semiring.zero())), semiring.zero());
            let distributed = infallible(semiring.times(&a, &plus(&semiring, b.clone(), &c)));
            let products = plus(
                &semiring,
                infallible(semiring.times(&a, &b)),
                &infallible(semiring.times(&a, &c)),
            );
            prop_assert_eq!(distributed, products);
        }

        #[test]
        fn boolean_obeys_semiring_laws(a in any::<bool>(), b in any::<bool>(), c in any::<bool>()) {
            let semiring = Boolean;
            prop_assert_eq!(plus(&semiring, a, &b), plus(&semiring, b, &a));
            prop_assert_eq!(plus(&semiring, plus(&semiring, a, &b), &c), plus(&semiring, a, &plus(&semiring, b, &c)));
            prop_assert_eq!(infallible(semiring.times(&a, &plus(&semiring, b, &c))), plus(&semiring, infallible(semiring.times(&a, &b)), &infallible(semiring.times(&a, &c))));
        }

        #[test]
        fn viterbi_obeys_semiring_laws(a in -1000_i16..=1000, b in -1000_i16..=1000, c in -1000_i16..=1000) {
            let semiring = Viterbi;
            let (a, b, c) = (f64::from(a), f64::from(b), f64::from(c));
            prop_assert_eq!(plus(&semiring, a, &b).to_bits(), plus(&semiring, b, &a).to_bits());
            prop_assert_eq!(plus(&semiring, plus(&semiring, a, &b), &c).to_bits(), plus(&semiring, a, &plus(&semiring, b, &c)).to_bits());
            prop_assert_eq!(infallible(semiring.times(&a, &semiring.one())).to_bits(), a.to_bits());
            prop_assert_eq!(infallible(semiring.times(&semiring.zero(), &a)).to_bits(), semiring.zero().to_bits());
            prop_assert_eq!(infallible(semiring.times(&a, &plus(&semiring, b, &c))).to_bits(), plus(&semiring, infallible(semiring.times(&a, &b)), &infallible(semiring.times(&a, &c))).to_bits());
        }

        #[test]
        fn log_inside_obeys_semiring_laws(a in -50_i16..=50, b in -50_i16..=50, c in -50_i16..=50) {
            let semiring = LogInside;
            let (a, b, c) = (f64::from(a), f64::from(b), f64::from(c));
            let close = |left: f64, right: f64| (left - right).abs() <= 1e-12;
            prop_assert!(close(plus(&semiring, a, &b), plus(&semiring, b, &a)));
            prop_assert!(close(plus(&semiring, plus(&semiring, a, &b), &c), plus(&semiring, a, &plus(&semiring, b, &c))));
            prop_assert!(close(infallible(semiring.times(&a, &plus(&semiring, b, &c))), plus(&semiring, infallible(semiring.times(&a, &b)), &infallible(semiring.times(&a, &c)))));
        }

        #[test]
        fn k_best_obeys_semiring_laws(
            a in prop::collection::vec(-20_i8..=20, 0..=3),
            b in prop::collection::vec(-20_i8..=20, 0..=3),
            c in prop::collection::vec(-20_i8..=20, 0..=3),
        ) {
            let semiring = kbest(3);
            let ranked = |values: Vec<i8>| {
                let mut values: Vec<f64> = values.into_iter().map(f64::from).collect();
                values.sort_by(|left, right| right.partial_cmp(left).unwrap_or(Ordering::Equal));
                values.into_boxed_slice()
            };
            let (a, b, c) = (ranked(a), ranked(b), ranked(c));
            prop_assert_eq!(plus(&semiring, a.clone(), &b), plus(&semiring, b.clone(), &a));
            prop_assert_eq!(plus(&semiring, plus(&semiring, a.clone(), &b), &c), plus(&semiring, a.clone(), &plus(&semiring, b.clone(), &c)));
            prop_assert_eq!(&infallible(semiring.times(&a, &semiring.one())), &a);
            prop_assert_eq!(infallible(semiring.times(&a, &semiring.zero())), semiring.zero());
            prop_assert_eq!(infallible(semiring.times(&a, &plus(&semiring, b.clone(), &c))), plus(&semiring, infallible(semiring.times(&a, &b)), &infallible(semiring.times(&a, &c))));
        }

        #[test]
        fn expectation_obeys_semiring_laws(
            ap in 0_u8..=4, ax in -4_i8..=4, ay in -4_i8..=4,
            bp in 0_u8..=4, bx in -4_i8..=4, by in -4_i8..=4,
            cp in 0_u8..=4, cx in -4_i8..=4, cy in -4_i8..=4,
        ) {
            let semiring = expectation(2);
            let weight = |p: u8, x: i8, y: i8| (f64::from(p), Vec::from([f64::from(x), f64::from(y)]).into_boxed_slice());
            let (a, b, c) = (weight(ap, ax, ay), weight(bp, bx, by), weight(cp, cx, cy));
            prop_assert_eq!(plus(&semiring, a.clone(), &b), plus(&semiring, b.clone(), &a));
            prop_assert_eq!(plus(&semiring, plus(&semiring, a.clone(), &b), &c), plus(&semiring, a.clone(), &plus(&semiring, b.clone(), &c)));
            prop_assert_eq!(&infallible(semiring.times(&a, &semiring.one())), &a);
            prop_assert_eq!(infallible(semiring.times(&a, &semiring.zero())), semiring.zero());
            prop_assert_eq!(infallible(semiring.times(&a, &plus(&semiring, b.clone(), &c))), plus(&semiring, infallible(semiring.times(&a, &b)), &infallible(semiring.times(&a, &c))));
        }
    }

    #[test]
    fn counting_promotes_on_addition_and_multiplication_overflow() {
        let semiring = Counting;
        let mut sum = Count::Small(u64::MAX);
        infallible(semiring.plus_assign(&mut sum, &Count::Small(1)));
        let product = infallible(semiring.times(&Count::Small(1_u64 << 63), &Count::Small(4)));
        assert_eq!(
            sum.into_big_uint(),
            BigUint::from(u64::MAX) + BigUint::from(1_u8),
            "addition promotes exactly"
        );
        assert_eq!(
            product.into_big_uint(),
            BigUint::from(1_u8) << 65_usize,
            "multiplication promotes exactly"
        );
    }
}
