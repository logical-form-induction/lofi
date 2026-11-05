"""Reusable laws for linkings from readout columns to the latent scale."""

from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Final, NamedTuple, NoReturn, final

import pytest
import torch

from lofi._checks import LawViolationError
from lofi.judgment import RecordedColumns, UndeclaredColumnError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from torch import Tensor

    from lofi.judgment import JudgmentDesign, Linking, LinkingDomain
    from lofi.stats import Interval

__all__ = [
    "LinkingSubject",
    "LinkingSuite",
    "SampledColumns",
    "contribution_is_non_decreasing_in_the_argument",
    "special_cases_recover_their_closed_forms",
    "the_response_is_never_read",
]

#: Points of the declared domain the monotonicity law samples.
MONOTONE_POINTS: Final = 10_000

#: Parameter vectors drawn per run of the monotonicity law.
PARAMETER_DRAWS: Final = 10

#: Rows the closed-form law samples beyond the subject's own design.
CLOSED_FORM_ROWS: Final = 1_000

#: Largest departure from a closed form the special-case law accepts.
CLOSED_FORM_ATOL: Final = 1e-9

#: Seed of the generator every sampling law draws from.
SEED: Final = 20_260_915


class LinkingSubject(NamedTuple):
    """Inputs for the linking-function conformance laws.

    Attributes
    ----------
    linking : Linking
        The linking under test.
    design : JudgmentDesign
        A design holding every readout column the linking declares, whose
        recorded responses are what the no-peeking law watches for.
    parameters : Tensor
        One parameter vector inside the declared parameter intervals.
    references : Mapping[str, Callable[[Mapping[str, Tensor]], Tensor]]
        The closed form of each declared special case, keyed by the case's
        name, computed from the readout columns alone.
    """

    linking: Linking
    design: JudgmentDesign
    parameters: Tensor
    references: Mapping[str, Callable[[Mapping[str, Tensor]], Tensor]]


@final
class SampledColumns:
    """Readout columns drawn from a linking's declared intervals.

    Parameters
    ----------
    readouts : Mapping[str, Tensor]
        One column per declared readout, all of one length.
    """

    __slots__ = ("_readouts", "_rows")

    def __init__(self, readouts: Mapping[str, Tensor]) -> None:
        self._readouts: Mapping[str, Tensor] = MappingProxyType(dict(readouts))
        self._rows = next(iter(self._readouts.values())).numel()

    @property
    def n_rows(self) -> int:
        """Return the number of rows each column has."""
        return self._rows

    @property
    def readouts(self) -> Mapping[str, Tensor]:
        """Return the readout columns by name."""
        return self._readouts


def _fail(detail: str, *, law: str) -> NoReturn:
    """Raise a linking law violation.

    Parameters
    ----------
    detail : str
        Description of the failed condition.
    law : str
        Name of the failed law.

    Raises
    ------
    LawViolationError
        Always raised with the law name and detail.
    """
    message = f"{law}: {detail}"

    raise LawViolationError(message)


def _uniform(interval: Interval, rows: int, generator: torch.Generator) -> Tensor:
    """Return uniform draws from an interval.

    Parameters
    ----------
    interval : Interval
        Interval to draw from.
    rows : int
        Number of draws.
    generator : torch.Generator
        Source of the draws.
    """
    unit = torch.rand(rows, generator=generator, dtype=torch.float64)

    return interval.low + unit * interval.width


def _sample_columns(
    domain: LinkingDomain, rows: int, generator: torch.Generator
) -> dict[str, Tensor]:
    """Return one uniform column per declared readout.

    Parameters
    ----------
    domain : LinkingDomain
        Domain naming the readouts and their intervals.
    rows : int
        Number of rows per column.
    generator : torch.Generator
        Source of the draws.
    """
    return {
        name: _uniform(interval, rows, generator)
        for name, interval in domain.columns.items()
    }


def _sample_parameters(domain: LinkingDomain, generator: torch.Generator) -> Tensor:
    """Return one parameter vector drawn from the declared intervals.

    Parameters
    ----------
    domain : LinkingDomain
        Domain naming the parameters and their intervals.
    generator : torch.Generator
        Source of the draws.
    """
    if not domain.parameters:
        return torch.empty(0, dtype=torch.float64)

    return torch.stack(
        [_uniform(interval, 1, generator)[0] for interval in domain.parameters.values()]
    )


def _check_subject(subject: LinkingSubject, *, law: str) -> None:
    """Check that a subject's parameters and design fit the declared domain.

    Parameters
    ----------
    subject : LinkingSubject
        Subject to check.
    law : str
        Name of the law reporting a defect.

    Raises
    ------
    LawViolationError
        If the parameter vector has another length than the domain declares,
        a parameter lies outside its interval, or the design lacks a declared
        readout.
    """
    domain = subject.linking.domain
    names = domain.parameter_names

    if subject.parameters.ndim != 1 or subject.parameters.numel() != len(names):
        _fail(
            f"the subject's parameters have shape {tuple(subject.parameters.shape)} "
            f"against the declared parameters {names!r}",
            law=law,
        )

    for name, value in zip(names, subject.parameters.tolist(), strict=True):
        if not domain.parameters[name].covers(value):
            _fail(
                f"the subject's parameter {name!r} is {value}, outside "
                f"{domain.parameters[name]!r}",
                law=law,
            )

    missing = [name for name in domain.readouts if name not in subject.design.readouts]

    if missing:
        _fail(f"the subject's design lacks the declared readouts {missing!r}", law=law)


def contribution_is_non_decreasing_in_the_argument(subject: LinkingSubject, /) -> None:
    """Check that raising the argument never lowers the contribution.

    Ten thousand points are drawn from the declared domain, each pairing a
    parameter vector inside the declared parameter intervals with a row of
    readouts inside the declared column intervals. Every point is compared
    with the same point after its argument is raised to a value still inside
    the argument's interval.

    Parameters
    ----------
    subject : LinkingSubject
        Linking and the design it is declared over.

    Raises
    ------
    LawViolationError
        If a raised argument lowers the contribution, a contribution is not
        finite, or the contribution is not one value per row.
    """
    law = "contribution_is_non_decreasing_in_the_argument"
    _check_subject(subject, law=law)
    domain = subject.linking.domain
    generator = torch.Generator().manual_seed(SEED)
    rows = MONOTONE_POINTS // PARAMETER_DRAWS

    for draw in range(PARAMETER_DRAWS):
        parameters = _sample_parameters(domain, generator)
        base = _sample_columns(domain, rows, generator)
        argument = domain.columns[domain.argument]
        room = argument.high - base[domain.argument]
        raised = dict(base)
        raised[domain.argument] = base[domain.argument] + room * torch.rand(
            rows, generator=generator, dtype=torch.float64
        )
        before = subject.linking.contribution(SampledColumns(base), parameters)
        after = subject.linking.contribution(SampledColumns(raised), parameters)

        for name, values in (("base", before), ("raised", after)):
            if values.shape != (rows,):
                _fail(
                    f"the {name} contribution has shape {tuple(values.shape)} over "
                    f"{rows} rows",
                    law=law,
                )

            if not bool(torch.isfinite(values).all()):
                _fail(
                    f"the {name} contribution is not finite at parameters "
                    f"{parameters.tolist()!r}",
                    law=law,
                )

        drop = before - after

        if bool((drop > 0.0).any()):
            worst = int(torch.argmax(drop))
            _fail(
                f"raising {domain.argument!r} from "
                f"{float(base[domain.argument][worst])} to "
                f"{float(raised[domain.argument][worst])} lowered the contribution "
                f"by {float(drop[worst])} at parameters {parameters.tolist()!r} "
                f"(draw {draw})",
                law=law,
            )


def special_cases_recover_their_closed_forms(subject: LinkingSubject, /) -> None:
    """Check that each declared special case matches its closed form.

    Every declared case is evaluated at its declared parameters over the
    subject's design and over a thousand rows drawn from the declared
    domain, and compared with the reference the subject supplies for it.

    Parameters
    ----------
    subject : LinkingSubject
        Linking, the design it is declared over, and the closed forms.

    Raises
    ------
    LawViolationError
        If a declared case has no reference or a reference no declared case,
        a case's parameters do not match the declared parameters, or the
        linking departs from a closed form by more than the tolerance.
    """
    law = "special_cases_recover_their_closed_forms"
    _check_subject(subject, law=law)
    linking = subject.linking
    domain = linking.domain
    declared = tuple(case.name for case in linking.special_cases)

    if len(frozenset(declared)) != len(declared):
        _fail(f"the declared special cases repeat a name: {declared!r}", law=law)

    if frozenset(declared) != frozenset(subject.references):
        _fail(
            f"the declared special cases are {sorted(declared)!r} and the "
            f"references cover {sorted(subject.references)!r}",
            law=law,
        )

    generator = torch.Generator().manual_seed(SEED)
    sampled = SampledColumns(_sample_columns(domain, CLOSED_FORM_ROWS, generator))

    for case in linking.special_cases:
        if len(case.parameters) != len(domain.parameter_names):
            _fail(
                f"the special case {case.name!r} declares {len(case.parameters)} "
                f"parameters against the declared "
                f"{domain.parameter_names!r}",
                law=law,
            )

        parameters = torch.tensor(case.parameters, dtype=torch.float64)
        reference = subject.references[case.name]

        for source, columns in (("the design", subject.design), ("sampled", sampled)):
            actual = linking.contribution(columns, parameters)
            expected = reference(columns.readouts)

            if actual.shape != expected.shape:
                _fail(
                    f"the special case {case.name!r} has shape "
                    f"{tuple(actual.shape)} against its closed form's "
                    f"{tuple(expected.shape)} over {source} rows",
                    law=law,
                )

            if not torch.allclose(actual, expected, atol=CLOSED_FORM_ATOL, rtol=0.0):
                departure = float(torch.max(torch.abs(actual - expected)))
                _fail(
                    f"the special case {case.name!r} departs from its closed form "
                    f"by {departure} over {source} rows (tolerance "
                    f"{CLOSED_FORM_ATOL})",
                    law=law,
                )


def the_response_is_never_read(subject: LinkingSubject, /) -> None:
    """Check that the linking reads only the readouts it declares.

    The design is handed to the linking through a recording view that
    admits the declared readouts and refuses every other column, the
    recorded responses first among them.

    Parameters
    ----------
    subject : LinkingSubject
        Linking and the design it is declared over.

    Raises
    ------
    LawViolationError
        If the linking reads a column outside its declared readouts, or does
        not return one finite value per row of the design.
    """
    law = "the_response_is_never_read"
    _check_subject(subject, law=law)
    view = RecordedColumns(subject.design, subject.linking.domain.readouts)

    try:
        result = subject.linking.contribution(view, subject.parameters)
    except UndeclaredColumnError as error:
        _fail(
            f"the linking read {error.column!r}, having declared "
            f"{error.admitted!r}; it had read {sorted(view.touched)!r}",
            law=law,
        )

    if result.shape != (subject.design.n_rows,):
        _fail(
            f"the contribution has shape {tuple(result.shape)} over "
            f"{subject.design.n_rows} rows",
            law=law,
        )

    if not bool(torch.isfinite(result).all()):
        _fail("the contribution over the design is not finite", law=law)


class LinkingSuite:
    """Reusable linking conformance laws."""

    pytestmark: ClassVar[list[pytest.MarkDecorator]] = [pytest.mark.conformance]

    def test_contribution_is_non_decreasing_in_the_argument(
        self, linking_subject: LinkingSubject
    ) -> None:
        """Run the monotonicity law.

        Parameters
        ----------
        linking_subject : LinkingSubject
            Linking and the design it is declared over.
        """
        contribution_is_non_decreasing_in_the_argument(linking_subject)

    def test_special_cases_recover_their_closed_forms(
        self, linking_subject: LinkingSubject
    ) -> None:
        """Run the closed-form law.

        Parameters
        ----------
        linking_subject : LinkingSubject
            Linking, the design it is declared over, and the closed forms.
        """
        special_cases_recover_their_closed_forms(linking_subject)

    def test_the_response_is_never_read(self, linking_subject: LinkingSubject) -> None:
        """Run the no-peeking law.

        Parameters
        ----------
        linking_subject : LinkingSubject
            Linking and the design it is declared over.
        """
        the_response_is_never_read(linking_subject)
