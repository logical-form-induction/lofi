//! Python bindings for native chart-weight semirings.

use std::convert::Infallible;
use std::num::NonZeroUsize;

use lofi_chart::semiring::{
    Boolean as CoreBoolean, Count, Counting as CoreCounting, Expectation as CoreExpectation,
    ExpectedWeight, KBest as CoreKBest, LogInside as CoreLogInside, Semiring,
    Viterbi as CoreViterbi,
};
use num_bigint::BigUint;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

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

fn float_tuple(py: Python<'_>, values: &[f64]) -> PyResult<Py<PyTuple>> {
    Ok(PyTuple::new(py, values.iter().copied())?.unbind())
}

/// Select Python-object weights for a third-party semiring.
#[pyclass(frozen, eq, hash, module = "lofi._native")]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct PythonCarrier;

#[pymethods]
impl PythonCarrier {
    #[new]
    const fn new() -> Self {
        Self
    }

    fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

/// Count derivations exactly.
#[pyclass(frozen, eq, hash, module = "lofi._native", name = "Counting")]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct PyCounting {
    inner: CoreCounting,
}

#[pymethods]
impl PyCounting {
    #[new]
    const fn new() -> Self {
        Self {
            inner: CoreCounting,
        }
    }

    #[getter]
    const fn name(&self) -> &'static str {
        self.inner.name()
    }

    #[getter]
    const fn log_probability(&self) -> bool {
        self.inner.log_probability()
    }

    #[getter]
    fn zero(&self) -> BigUint {
        self.inner.zero().into_big_uint()
    }

    #[getter]
    fn one(&self) -> BigUint {
        self.inner.one().into_big_uint()
    }

    #[getter]
    fn lowering(&self) -> Self {
        self.clone()
    }

    fn plus(&self, left: BigUint, right: BigUint) -> BigUint {
        plus(&self.inner, Count::from(left), &Count::from(right)).into_big_uint()
    }

    fn times(&self, left: BigUint, right: BigUint) -> BigUint {
        infallible(self.inner.times(&Count::from(left), &Count::from(right))).into_big_uint()
    }

    fn is_zero(&self, weight: BigUint) -> bool {
        self.inner.is_zero(&Count::from(weight))
    }

    const fn __repr__(&self) -> &'static str {
        match self.inner {
            CoreCounting => "Counting()",
        }
    }
}

/// Keep the best derivation's log probability.
#[pyclass(frozen, eq, hash, module = "lofi._native", name = "Viterbi")]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct PyViterbi {
    inner: CoreViterbi,
}

#[pymethods]
impl PyViterbi {
    #[new]
    const fn new() -> Self {
        Self { inner: CoreViterbi }
    }

    #[getter]
    const fn name(&self) -> &'static str {
        self.inner.name()
    }

    #[getter]
    const fn log_probability(&self) -> bool {
        self.inner.log_probability()
    }

    #[getter]
    fn zero(&self) -> f64 {
        self.inner.zero()
    }

    #[getter]
    fn one(&self) -> f64 {
        self.inner.one()
    }

    #[getter]
    fn lowering(&self) -> Self {
        self.clone()
    }

    fn plus(&self, left: f64, right: f64) -> f64 {
        plus(&self.inner, left, &right)
    }

    fn times(&self, left: f64, right: f64) -> f64 {
        infallible(self.inner.times(&left, &right))
    }

    fn is_zero(&self, weight: f64) -> bool {
        self.inner.is_zero(weight)
    }

    const fn __repr__(&self) -> &'static str {
        match self.inner {
            CoreViterbi => "Viterbi()",
        }
    }
}

/// Accumulate inside probabilities in log space.
#[pyclass(frozen, eq, hash, module = "lofi._native", name = "LogInside")]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct PyLogInside {
    inner: CoreLogInside,
}

#[pymethods]
impl PyLogInside {
    #[new]
    const fn new() -> Self {
        Self {
            inner: CoreLogInside,
        }
    }

    #[getter]
    const fn name(&self) -> &'static str {
        self.inner.name()
    }

    #[getter]
    const fn log_probability(&self) -> bool {
        self.inner.log_probability()
    }

    #[getter]
    fn zero(&self) -> f64 {
        self.inner.zero()
    }

    #[getter]
    fn one(&self) -> f64 {
        self.inner.one()
    }

    #[getter]
    fn lowering(&self) -> Self {
        self.clone()
    }

    fn plus(&self, left: f64, right: f64) -> f64 {
        plus(&self.inner, left, &right)
    }

    fn times(&self, left: f64, right: f64) -> f64 {
        infallible(self.inner.times(&left, &right))
    }

    fn is_zero(&self, weight: f64) -> bool {
        self.inner.is_zero(weight)
    }

    const fn __repr__(&self) -> &'static str {
        match self.inner {
            CoreLogInside => "LogInside()",
        }
    }
}

/// Track whether an item is derivable.
#[pyclass(frozen, eq, hash, module = "lofi._native", name = "Boolean")]
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct PyBoolean {
    inner: CoreBoolean,
}

#[pymethods]
impl PyBoolean {
    #[new]
    const fn new() -> Self {
        Self { inner: CoreBoolean }
    }

    #[getter]
    const fn name(&self) -> &'static str {
        self.inner.name()
    }

    #[getter]
    const fn log_probability(&self) -> bool {
        self.inner.log_probability()
    }

    #[getter]
    fn zero(&self) -> bool {
        self.inner.zero()
    }

    #[getter]
    fn one(&self) -> bool {
        self.inner.one()
    }

    #[getter]
    fn lowering(&self) -> Self {
        self.clone()
    }

    fn plus(&self, left: bool, right: bool) -> bool {
        plus(&self.inner, left, &right)
    }

    fn times(&self, left: bool, right: bool) -> bool {
        infallible(self.inner.times(&left, &right))
    }

    const fn is_zero(&self, weight: bool) -> bool {
        self.inner.is_zero(weight)
    }

    const fn __repr__(&self) -> &'static str {
        match self.inner {
            CoreBoolean => "Boolean()",
        }
    }
}

/// Keep the highest `k` derivation log probabilities.
#[pyclass(frozen, eq, hash, module = "lofi._native", name = "KBest")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PyKBest {
    inner: CoreKBest,
}

#[pymethods]
impl PyKBest {
    #[new]
    fn new(k: usize) -> PyResult<Self> {
        NonZeroUsize::new(k).map_or_else(
            || {
                Err(PyValueError::new_err(
                    "a k-best width is at least one, not 0",
                ))
            },
            |width| {
                Ok(Self {
                    inner: CoreKBest::new(width),
                })
            },
        )
    }

    #[getter]
    const fn k(&self) -> usize {
        self.inner.k().get()
    }

    #[getter]
    const fn name(&self) -> &'static str {
        self.inner.name()
    }

    #[getter]
    const fn log_probability(&self) -> bool {
        self.inner.log_probability()
    }

    #[getter]
    fn zero(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        float_tuple(py, &self.inner.zero())
    }

    #[getter]
    fn one(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        float_tuple(py, &self.inner.one())
    }

    #[getter]
    fn lowering(&self) -> Self {
        self.clone()
    }

    fn plus(&self, py: Python<'_>, left: Vec<f64>, right: Vec<f64>) -> PyResult<Py<PyTuple>> {
        let ranked = plus(
            &self.inner,
            left.into_boxed_slice(),
            &right.into_boxed_slice(),
        );
        float_tuple(py, &ranked)
    }

    fn times(&self, py: Python<'_>, left: Vec<f64>, right: Vec<f64>) -> PyResult<Py<PyTuple>> {
        let ranked = infallible(
            self.inner
                .times(&left.into_boxed_slice(), &right.into_boxed_slice()),
        );
        float_tuple(py, &ranked)
    }

    fn is_zero(&self, weight: Vec<f64>) -> bool {
        self.inner.is_zero(&weight.into_boxed_slice())
    }

    fn __repr__(&self) -> String {
        format!("KBest({})", self.k())
    }
}

/// Accumulate expectations for a fixed coordinate count.
#[pyclass(frozen, eq, hash, module = "lofi._native", name = "Expectation")]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PyExpectation {
    inner: CoreExpectation,
}

impl PyExpectation {
    fn weight(&self, weight: (f64, Vec<f64>)) -> PyResult<ExpectedWeight> {
        if weight.1.len() != self.dim() {
            return Err(PyValueError::new_err(format!(
                "an expectation weight has {} coordinates, not {}",
                self.dim(),
                weight.1.len()
            )));
        }
        Ok((weight.0, weight.1.into_boxed_slice()))
    }

    fn into_python(py: Python<'_>, weight: ExpectedWeight) -> PyResult<(f64, Py<PyTuple>)> {
        let (probability, expectations) = weight;
        Ok((probability, float_tuple(py, &expectations)?))
    }
}

#[pymethods]
impl PyExpectation {
    #[new]
    fn new(dim: usize) -> PyResult<Self> {
        NonZeroUsize::new(dim).map_or_else(
            || {
                Err(PyValueError::new_err(
                    "an expectation has at least one coordinate, not 0",
                ))
            },
            |width| {
                Ok(Self {
                    inner: CoreExpectation::new(width),
                })
            },
        )
    }

    #[getter]
    const fn dim(&self) -> usize {
        self.inner.dim().get()
    }

    #[getter]
    const fn name(&self) -> &'static str {
        self.inner.name()
    }

    #[getter]
    const fn log_probability(&self) -> bool {
        self.inner.log_probability()
    }

    #[getter]
    fn zero(&self, py: Python<'_>) -> PyResult<(f64, Py<PyTuple>)> {
        Self::into_python(py, self.inner.zero())
    }

    #[getter]
    fn one(&self, py: Python<'_>) -> PyResult<(f64, Py<PyTuple>)> {
        Self::into_python(py, self.inner.one())
    }

    #[getter]
    fn lowering(&self) -> Self {
        self.clone()
    }

    fn plus(
        &self,
        py: Python<'_>,
        left: (f64, Vec<f64>),
        right: (f64, Vec<f64>),
    ) -> PyResult<(f64, Py<PyTuple>)> {
        let right = self.weight(right)?;
        Self::into_python(py, plus(&self.inner, self.weight(left)?, &right))
    }

    fn times(
        &self,
        py: Python<'_>,
        left: (f64, Vec<f64>),
        right: (f64, Vec<f64>),
    ) -> PyResult<(f64, Py<PyTuple>)> {
        Self::into_python(
            py,
            infallible(self.inner.times(&self.weight(left)?, &self.weight(right)?)),
        )
    }

    fn is_zero(&self, weight: (f64, Vec<f64>)) -> PyResult<bool> {
        Ok(self.inner.is_zero(&self.weight(weight)?))
    }

    fn __repr__(&self) -> String {
        format!("Expectation({})", self.dim())
    }
}
