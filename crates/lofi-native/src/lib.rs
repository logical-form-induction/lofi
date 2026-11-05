//! The Python bindings of the chart kernel, built as `lofi._native`.
//!
//! The crate converts between Python values and the values of `lofi-chart`
//! and contains no algorithm of its own.

use lofi_chart::build;
use pyo3::prelude::*;

mod chart;
pub mod grammar;
pub mod rule;
mod semiring;
pub mod term;

/// The built-in Cargo profile a build inherits from.
#[pyclass(frozen, eq, hash, module = "lofi._native", name = "Profile")]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum PyProfile {
    /// The `dev` profile, or a profile inheriting from it.
    #[pyo3(name = "DEBUG")]
    Debug,
    /// The `release` profile, or a profile inheriting from it.
    #[pyo3(name = "RELEASE")]
    Release,
}

impl From<build::Profile> for PyProfile {
    fn from(profile: build::Profile) -> Self {
        match profile {
            build::Profile::Debug => Self::Debug,
            build::Profile::Release => Self::Release,
        }
    }
}

/// The version and Cargo profile of the compiled extension.
#[pyclass(frozen, eq, hash, module = "lofi._native", name = "BuildInfo")]
#[derive(PartialEq, Eq, Hash)]
struct PyBuildInfo(build::BuildInfo);

#[pymethods]
impl PyBuildInfo {
    /// The crate version, as Cargo.toml declares it.
    #[getter]
    const fn version(&self) -> &'static str {
        self.0.version()
    }

    /// The profile the extension was compiled under.
    #[getter]
    fn profile(&self) -> PyProfile {
        self.0.profile().into()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let profile = Bound::new(py, self.profile())?.as_any().repr()?;
        Ok(format!(
            "BuildInfo(version='{}', profile={profile})",
            self.0.version()
        ))
    }
}

/// Return the version and Cargo profile the extension was compiled with.
#[pyfunction]
const fn build_info() -> PyBuildInfo {
    PyBuildInfo(build::BuildInfo::current())
}

/// The compiled half of LoFI.
#[pymodule(name = "_native")]
mod native {
    #[pymodule_export]
    use super::{PyBuildInfo, PyProfile, build_info};
    #[pymodule_export]
    use crate::chart::{
        PyChartProgram, PyForestTable, PyInsideTable, PyMissBatch, SpanDependent, SpanFree,
        UnitEdges,
    };
    #[pymodule_export]
    use crate::grammar::{
        PyCompiledGrammar, PyCompiledNormalFormRole, PyCompiledPattern, PyCompiledRule,
        compile_grammar,
    };
    #[pymodule_export]
    use crate::rule::{PyRepresentativeKernel, PyRuleKernel, PyRuleProgram};
    #[pymodule_export]
    use crate::semiring::{
        PyBoolean, PyCounting, PyExpectation, PyKBest, PyLogInside, PyViterbi, PythonCarrier,
    };
    #[pymodule_export]
    use crate::term::{
        PyArityError, PyBindingAtomCountError, PyBindingShapeError, PyDuplicateNameError,
        PyEncodingOverflowError, PyEncodingShapeError, PyErasureError, PyForeignTermError,
        PyNonTerminatingError, PyNotLocallyClosedError, PyPayloadError, PyRegion,
        PyRegionStateError, PyReservedDeclarationError, PySortError, PySortView, PyStoreFullError,
        PyTerm, PyTermError, PyTermStore, PyUndeclaredSortError, PyUnknownConstantError,
        PyUnknownOperatorError, PyVacuousBindingError, abstract_, close_binder, instantiate,
        open_binder, substitute,
    };
}
