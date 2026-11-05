from typing import TYPE_CHECKING, Final, NamedTuple, NoReturn

import pytest
import torch

from lofi._checks import LawViolationError

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from lofi._types import JsonValue
    from lofi.nn.ports import InstrumentScope, Observable, Reading
    from lofi.nn.sites import Site

__all__ = [
    "OBSERVABLE_LAWS",
    "ObservableConformance",
    "ObservableSuite",
]

_FIDELITY: Final = "the_site_map_is_what_a_scope_yields"

_PURITY: Final = "observing_leaves_the_observed_alone"


def _fail(detail: str, *, law: str) -> NoReturn:
    """Raise a law violation, naming the law and what went wrong.

    Parameters
    ----------
    detail : str
        The counterexample, in that law's own terms.
    law : str
        The law's short name.

    Raises
    ------
    LawViolationError
        Always.
    """
    message = f"{law}: {detail}"

    raise LawViolationError(message)


def _same(before: tuple[Tensor, ...], after: tuple[Tensor, ...]) -> bool:
    """Return whether two runs computed the same numbers, tensor for tensor.

    Parameters
    ----------
    before : tuple[Tensor, ...]
        What one call returned.
    after : tuple[Tensor, ...]
        What the other returned.

    Returns
    -------
    bool
        Whether the two agree in length and in every entry, bit for bit.
    """
    if len(before) != len(after):
        return False

    return all(
        one.shape == other.shape and torch.equal(one, other)
        for one, other in zip(before, after, strict=True)
    )


class ObservableConformance[SubjectT: Observable, ReadingT: Reading](NamedTuple):
    """Inputs for the observable conformance laws.

    Parameters
    ----------
    build : Callable[[], SubjectT]
        Return the component under test, ready to be run.
    scope : Callable[[SubjectT, frozenset[Site]], InstrumentScope[ReadingT]]
        Open a scope over a set of published points, not yet entered.
    run : Callable[[SubjectT], tuple[Tensor, ...]]
        Run the component and return the tensors it computed, in a fixed order.
    configuration : Callable[[SubjectT], JsonValue]
        Return the component's configuration as a value that can be compared, in the
        component's own terms.
    submodules : Callable[[SubjectT], frozenset[str]]
        Return the names of the submodules the component holds.
    placements : Callable[[SubjectT], tuple[str, ...]]
        Return where each of the component's tensors lives, in a fixed order.
    """

    build: Callable[[], SubjectT]
    scope: Callable[[SubjectT, frozenset[Site]], InstrumentScope[ReadingT]]
    run: Callable[[SubjectT], tuple[Tensor, ...]]
    configuration: Callable[[SubjectT], JsonValue]
    submodules: Callable[[SubjectT], frozenset[str]]
    placements: Callable[[SubjectT], tuple[str, ...]]


class ObservableSuite:
    pytestmark = pytest.mark.conformance

    def test_the_site_map_is_what_a_scope_yields[
        SubjectT: Observable,
        ReadingT: Reading,
    ](
        self,
        conformance: ObservableConformance[SubjectT, ReadingT],
    ) -> None:
        """Check every published point against the reading a scope takes there.

        Parameters
        ----------
        conformance : ObservableConformance[SubjectT, ReadingT]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the component publishes nothing, if a scope over one point yields a
            reading at another or none at all, if a reading is filed under a point other
            than its own, if its axes are not the ones the specification names, or if
            its rank is not the number of those axes.
        """
        subject = conformance.build()
        published = subject.site_map()

        if not published:
            _fail("the component publishes no point to read", law=_FIDELITY)

        for site, spec in published.items():
            with conformance.scope(subject, frozenset({site})) as scope:
                conformance.run(subject)

            taken = scope.readings

            if frozenset(taken) != frozenset({site}):
                _fail(
                    f"a scope over {site!r} yielded readings at "
                    f"{sorted(map(repr, taken))}",
                    law=_FIDELITY,
                )

            reading = taken[site]

            if reading.site != site:
                _fail(
                    f"the reading filed under {site!r} was taken at {reading.site!r}",
                    law=_FIDELITY,
                )

            if reading.axes != spec.axes:
                _fail(
                    f"{site!r} declares the axes {spec.axes} and was read over "
                    f"{reading.axes}",
                    law=_FIDELITY,
                )

            if reading.tensor.ndim != len(spec.axes):
                _fail(
                    f"{site!r} declares {len(spec.axes)} axes and was read at "
                    f"rank {reading.tensor.ndim}",
                    law=_FIDELITY,
                )

    def test_observing_leaves_the_observed_alone[
        SubjectT: Observable,
        ReadingT: Reading,
    ](
        self,
        conformance: ObservableConformance[SubjectT, ReadingT],
    ) -> None:
        """Check that running inside a scope changes nothing about the component.

        Parameters
        ----------
        conformance : ObservableConformance[SubjectT, ReadingT]
            What the laws run against.

        Raises
        ------
        LawViolationError
            If the configuration, the submodule names or the placements differ after a
            scope has closed, or if the numbers computed inside the scope or after it
            differ from the ones computed before it.
        """
        subject = conformance.build()
        watched = frozenset(subject.site_map())

        outside = tuple(tensor.clone() for tensor in conformance.run(subject))
        configured = conformance.configuration(subject)
        held = conformance.submodules(subject)
        placed = conformance.placements(subject)

        with conformance.scope(subject, watched):
            inside = tuple(tensor.clone() for tensor in conformance.run(subject))

        if conformance.configuration(subject) != configured:
            _fail(
                f"the configuration was {configured!r} and is now "
                f"{conformance.configuration(subject)!r}",
                law=_PURITY,
            )

        if conformance.submodules(subject) != held:
            _fail(
                f"the submodules were {sorted(held)} and are now "
                f"{sorted(conformance.submodules(subject))}",
                law=_PURITY,
            )

        if conformance.placements(subject) != placed:
            _fail(
                f"the tensors sat on {placed} and now sit on "
                f"{conformance.placements(subject)}",
                law=_PURITY,
            )

        if not _same(outside, inside):
            _fail(
                "the numbers computed inside the scope are not the ones "
                "computed outside it",
                law=_PURITY,
            )

        if not _same(outside, conformance.run(subject)):
            _fail(
                "the numbers computed after the scope closed are not the ones "
                "computed before it opened",
                law=_PURITY,
            )


OBSERVABLE_LAWS: Final = (
    ObservableSuite.test_the_site_map_is_what_a_scope_yields,
    ObservableSuite.test_observing_leaves_the_observed_alone,
)
