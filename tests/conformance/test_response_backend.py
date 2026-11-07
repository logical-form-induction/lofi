from typing import Final, Literal, NamedTuple, override

import pytest
import torch
from tests.conformance.response_backend import (
    BackendEvaluator,
    ResponseBackendCase,
    ResponseBackendConformance,
    ResponseBackendSuite,
    bind_response_backend,
    response_backends_agree,
)

from lofi._checks import LawViolationError
from lofi.judgment import (
    ConditionalLikelihood,
    DesignCodec,
    FixedDiscrimination,
    JudgmentDesign,
    JudgmentRow,
    Metric,
    Predictor,
    ResponseDeclaration,
    ResponseDesign,
    ResponseSpec,
    RowColumns,
    ScaleNormalization,
)


class Readout(Predictor, extra="forbid"):
    kind: Literal["backend_readout"] = "backend_readout"
    name: str


class ObjectProgram(NamedTuple):
    levels: int
    normalization: ScaleNormalization


type TupleProgram = tuple[int, ScaleNormalization]


def _values(
    levels: int,
    normalization: ScaleNormalization,
    params: torch.Tensor,
    design: JudgmentDesign,
) -> torch.Tensor:
    """Return the fixture likelihoods for one compiled declaration."""
    offset = {
        ScaleNormalization.IDENTITY: 0.0,
        ScaleNormalization.LOG: 0.25,
        ScaleNormalization.UNIT_INTERVAL: 0.5,
        ScaleNormalization.RIDIT: 0.75,
    }[normalization]

    return params[design.responses] - float(levels) / 10.0 - offset


class ObjectBackend:
    def compile(self, spec: ResponseDeclaration, /) -> ObjectProgram:
        if not isinstance(spec, ResponseSpec):
            msg = "the object fixture backend serves ordinal declarations alone"

            raise TypeError(msg)

        return ObjectProgram(spec.levels, spec.normalization)

    def log_likelihood(
        self,
        compiled: ObjectProgram,
        params: torch.Tensor,
        design: ResponseDesign,
        /,
    ) -> torch.Tensor:
        if not isinstance(design, JudgmentDesign):
            msg = "the object fixture backend serves ordinal designs alone"

            raise TypeError(msg)

        return _values(compiled.levels, compiled.normalization, params, design)


class TupleBackend:
    def compile(self, spec: ResponseDeclaration, /) -> TupleProgram:
        if not isinstance(spec, ResponseSpec):
            msg = "the tuple fixture backend serves ordinal declarations alone"

            raise TypeError(msg)

        return spec.levels, spec.normalization

    def log_likelihood(
        self,
        compiled: TupleProgram,
        params: torch.Tensor,
        design: ResponseDesign,
        /,
    ) -> torch.Tensor:
        if not isinstance(design, JudgmentDesign):
            msg = "the tuple fixture backend serves ordinal designs alone"

            raise TypeError(msg)

        levels, normalization = compiled

        return _values(levels, normalization, params, design)


class RefusingBackend(TupleBackend):
    @override
    def compile(self, spec: ResponseDeclaration, /) -> TupleProgram:
        if (
            isinstance(spec, ResponseSpec)
            and spec.normalization is ScaleNormalization.LOG
        ):
            message = "log normalization is unsupported"

            raise ValueError(message)

        return super().compile(spec)


class WrongShapeBackend(TupleBackend):
    @override
    def log_likelihood(
        self,
        compiled: TupleProgram,
        params: torch.Tensor,
        design: ResponseDesign,
        /,
    ) -> torch.Tensor:
        values = super().log_likelihood(compiled, params, design)

        return torch.cat((values, values.new_zeros(1)))


class ShiftedBackend(TupleBackend):
    @override
    def log_likelihood(
        self,
        compiled: TupleProgram,
        params: torch.Tensor,
        design: ResponseDesign,
        /,
    ) -> torch.Tensor:
        values = super().log_likelihood(compiled, params, design).clone()
        values[0] += 2e-6

        return values


def _case(
    levels: int,
    normalization: ScaleNormalization,
    responses: tuple[int, ...],
) -> ResponseBackendCase:
    """Build one explicit agreement case."""
    spec = ResponseSpec(
        family=Metric(),
        levels=levels,
        objective=ConditionalLikelihood(),
        predictor=(Readout(name="p_gram"),),
        identification=FixedDiscrimination(value=1.0),
        normalization=normalization,
    )
    rows = tuple(
        JudgmentRow(
            f"participant-{index}",
            f"item-{index}",
            response,
            RowColumns({}, {"p_gram": float(index) / 10.0}),
        )
        for index, response in enumerate(responses)
    )
    design = DesignCodec(levels, readouts=("p_gram",)).encode(rows)
    params = torch.linspace(-0.4, 0.4, levels, dtype=torch.float64)

    return ResponseBackendCase(spec, params, design)


CASES: Final = (
    _case(3, ScaleNormalization.IDENTITY, (0, 1, 2)),
    _case(4, ScaleNormalization.LOG, (3, 1, 0, 2)),
    _case(5, ScaleNormalization.UNIT_INTERVAL, (4, 2, 0)),
)


def _backends(second: TupleBackend) -> tuple[BackendEvaluator, ...]:
    """Bind the reference backend and one comparison backend."""
    return (
        bind_response_backend("object", ObjectBackend()),
        bind_response_backend("comparison", second),
    )


def _conformance(second: TupleBackend | None = None) -> ResponseBackendConformance:
    """Build the agreement fixture around a comparison backend."""
    comparison = TupleBackend() if second is None else second

    return ResponseBackendConformance(_backends(comparison), CASES)


class TestConformingBackends(ResponseBackendSuite):
    @pytest.fixture
    def conformance(self) -> ResponseBackendConformance:
        return _conformance()


def test_conforming_backends_with_distinct_compiled_types_agree() -> None:
    response_backends_agree(_conformance())


def test_one_sided_compilation_names_the_asymmetric_backend() -> None:
    with pytest.raises(
        LawViolationError,
        match=r"case 1 has asymmetric compilation.*comparison.*unsupported",
    ):
        response_backends_agree(_conformance(RefusingBackend()))


def test_a_shape_disagreement_names_both_backends() -> None:
    with pytest.raises(
        LawViolationError,
        match=r"case 0 shape disagreement.*object.*comparison",
    ):
        response_backends_agree(_conformance(WrongShapeBackend()))


def test_a_numerical_disagreement_beyond_tolerance_names_both_backends() -> None:
    with pytest.raises(
        LawViolationError,
        match=r"case 0 numerical disagreement.*object.*comparison",
    ):
        response_backends_agree(_conformance(ShiftedBackend()))


@pytest.mark.parametrize(
    "conformance",
    [
        ResponseBackendConformance(
            (bind_response_backend("only", ObjectBackend()),), CASES
        ),
        ResponseBackendConformance(_backends(TupleBackend()), CASES[:2]),
    ],
    ids=("one-backend", "two-cases"),
)
def test_the_law_requires_explicit_discriminating_witnesses(
    conformance: ResponseBackendConformance,
) -> None:
    with pytest.raises(LawViolationError, match="needs at least"):
        response_backends_agree(conformance)
