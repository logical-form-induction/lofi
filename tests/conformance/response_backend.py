from typing import TYPE_CHECKING, Final, NamedTuple, NoReturn, cast

import pytest
import torch

from lofi._checks import LawViolationError

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from lofi.judgment import JudgmentDesign, ResponseBackend, ResponseSpec

__all__ = [
    "RESPONSE_BACKEND_LAWS",
    "BackendEvaluator",
    "ResponseBackendCase",
    "ResponseBackendConformance",
    "ResponseBackendSuite",
    "bind_response_backend",
    "response_backends_agree",
]

_AGREEMENT_ATOL: Final = 1e-6
_MINIMUM_BACKENDS: Final = 2
_MINIMUM_CASES: Final = 3


class BackendEvaluator(NamedTuple):
    """A named, type-erased backend adapter."""

    name: str
    compile: Callable[[ResponseSpec], object]
    log_likelihood: Callable[[object, Tensor, JudgmentDesign], Tensor]


class ResponseBackendCase(NamedTuple):
    """One response specification, parameter vector, and design."""

    spec: ResponseSpec
    params: Tensor
    design: JudgmentDesign


class ResponseBackendConformance(NamedTuple):
    """Explicit witnesses for the cross-backend agreement law."""

    backends: tuple[BackendEvaluator, ...]
    cases: tuple[ResponseBackendCase, ...]


def bind_response_backend[Compiled](
    name: str,
    backend: ResponseBackend[Compiled],
    /,
) -> BackendEvaluator:
    """Bind a backend without exposing its compiled representation."""

    def log_likelihood(
        compiled: object,
        params: Tensor,
        design: JudgmentDesign,
    ) -> Tensor:
        return backend.log_likelihood(cast("Compiled", compiled), params, design)

    return BackendEvaluator(name, backend.compile, log_likelihood)


def _fail(detail: str) -> NoReturn:
    """Raise the backend-agreement law violation."""
    message = f"response_backends_agree: {detail}"

    raise LawViolationError(message)


def _validate_witnesses(conformance: ResponseBackendConformance) -> None:
    """Validate the explicit backend and case witnesses."""
    if len(conformance.backends) < _MINIMUM_BACKENDS:
        _fail(f"needs at least {_MINIMUM_BACKENDS} named backends")

    if len(conformance.cases) < _MINIMUM_CASES:
        _fail(f"needs at least {_MINIMUM_CASES} response cases")

    names = tuple(backend.name for backend in conformance.backends)

    if any(not name for name in names):
        _fail("every backend needs a nonempty name")

    if len(frozenset(names)) != len(names):
        _fail(f"backend names are not unique: {names!r}")


def _compiled_for(
    case_index: int,
    spec: ResponseSpec,
    backends: tuple[BackendEvaluator, ...],
) -> tuple[object, ...]:
    """Compile one case under every named backend."""
    compiled: list[object] = []
    failures: list[tuple[str, Exception]] = []

    for backend in backends:
        try:
            compiled.append(backend.compile(spec))
        except Exception as error:
            failures.append((backend.name, error))

    if failures:
        failed = ", ".join(
            f"{name} ({type(error).__name__}: {error})" for name, error in failures
        )
        succeeded = tuple(
            backend.name
            for backend in backends
            if backend.name not in {name for name, _ in failures}
        )
        _fail(
            f"case {case_index} has asymmetric compilation; "
            f"compiled={succeeded!r}, failed={failed}"
        )

    return tuple(compiled)


def response_backends_agree(conformance: ResponseBackendConformance, /) -> None:
    """Check compilation, shape, and numerical agreement across backends.

    Parameters
    ----------
    conformance
        Named backends and the cases they must evaluate.

    Raises
    ------
    LawViolationError
        If witnesses are insufficient, compilation is asymmetric, or results disagree.
    """
    _validate_witnesses(conformance)

    for case_index, case in enumerate(conformance.cases):
        compiled = _compiled_for(case_index, case.spec, conformance.backends)
        results = tuple(
            backend.log_likelihood(program, case.params, case.design)
            for backend, program in zip(conformance.backends, compiled, strict=True)
        )

        for left_index, (left_backend, left) in enumerate(
            zip(conformance.backends, results, strict=True)
        ):
            for right_backend, right in zip(
                conformance.backends[left_index + 1 :],
                results[left_index + 1 :],
                strict=True,
            ):
                if left.shape != right.shape:
                    _fail(
                        f"case {case_index} shape disagreement: "
                        f"{left_backend.name} returned {tuple(left.shape)}, "
                        f"{right_backend.name} returned {tuple(right.shape)}"
                    )

                if not torch.allclose(
                    left,
                    right,
                    atol=_AGREEMENT_ATOL,
                    rtol=0.0,
                ):
                    distance = float(torch.max(torch.abs(left - right)))
                    _fail(
                        f"case {case_index} numerical disagreement: "
                        f"{left_backend.name} and {right_backend.name} differ by "
                        f"{distance} (tolerance {_AGREEMENT_ATOL})"
                    )


class ResponseBackendSuite:
    """Reusable response-backend conformance suite."""

    pytestmark = pytest.mark.conformance

    def test_backends_agree(
        self,
        conformance: ResponseBackendConformance,
    ) -> None:
        """Check the cross-backend agreement law."""
        response_backends_agree(conformance)


RESPONSE_BACKEND_LAWS: Final = (ResponseBackendSuite.test_backends_agree,)
