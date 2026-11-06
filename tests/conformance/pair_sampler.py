"""Reusable laws for labeled rule-pair samplers."""

from typing import TYPE_CHECKING, ClassVar, Final, NamedTuple, NoReturn, Protocol

import pytest

from lofi._checks import LawViolationError
from lofi.data import PairDistribution, PairSampler, term_depth

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.judge import Action
    from lofi.terms import Term

__all__ = [
    "PAIR_SAMPLER_CONFORMANCE_LAWS",
    "PairSamplerConformance",
    "PairSamplerFixture",
    "labels_are_exact",
    "pair_reports_match_samples",
    "pair_samples_are_deterministic",
]


class PairRule[T](Protocol):
    """Rule fields inspected by the pair-sampler laws."""

    @property
    def name(self) -> str: ...

    @property
    def arity(self) -> int: ...

    def admits(self, inputs: tuple[T, ...], output: T, /) -> bool: ...


class PairSamplerFixture(NamedTuple):
    """A pair sampler, distribution, and rule inventory."""

    sampler: PairSampler[Term, Action]
    distribution: PairDistribution
    rules: Sequence[PairRule[Term]]


def _fail(detail: str, *, law: str) -> NoReturn:
    """Raise a named law violation."""
    message = f"{law}: {detail}"
    raise LawViolationError(message)


def pair_samples_are_deterministic(fixture: PairSamplerFixture) -> None:
    """Check that a seed determines pairs and labels."""
    first, _ = fixture.sampler.sample(30, fixture.distribution, seed=421)
    second, _ = fixture.sampler.sample(30, fixture.distribution, seed=421)
    first_keys = tuple(
        (
            tuple(term.key() for term in pair.inputs),
            pair.output.key(),
            tuple(action.name for action in pair.actions),
        )
        for pair in first
    )
    second_keys = tuple(
        (
            tuple(term.key() for term in pair.inputs),
            pair.output.key(),
            tuple(action.name for action in pair.actions),
        )
        for pair in second
    )

    if first_keys != second_keys:
        _fail("equal seeds produced different labeled pairs", law="seed determinism")


def labels_are_exact(fixture: PairSamplerFixture) -> None:
    """Check exact rule labels and all sampled-term depth bounds."""
    pairs, _ = fixture.sampler.sample(30, fixture.distribution, seed=119)

    for pair in pairs:
        admitted = {
            rule.name
            for rule in fixture.rules
            if rule.arity == len(pair.inputs) and rule.admits(pair.inputs, pair.output)
        }
        labeled = {action.rule for action in pair.actions if action.rule is not None}

        if labeled != admitted:
            _fail(
                f"labels {sorted(labeled)!r} differ from rules {sorted(admitted)!r}",
                law="exact labels",
            )

        unbounded = tuple(
            term
            for term in (*pair.inputs, pair.output)
            if term_depth(term) > fixture.distribution.max_depth
        )

        if unbounded:
            _fail(f"{unbounded!r} exceeds the bound", law="depth bound")


def pair_reports_match_samples(fixture: PairSamplerFixture) -> None:
    """Check rule-instance counts and the achieved arity distribution."""
    pairs, report = fixture.sampler.sample(30, fixture.distribution, seed=319)
    arities: dict[int, int] = {}

    for pair in pairs:
        arity = len(pair.inputs)
        arities[arity] = arities.get(arity, 0) + 1

    if report.emitted != len(pairs) or dict(report.arities) != arities:
        _fail("arity counts do not match the emitted instances", law="report")

    expected_distribution = {
        arity: count / len(pairs) for arity, count in arities.items()
    }

    if dict(report.arity_distribution) != expected_distribution:
        _fail(
            f"reported arity distribution {report.arity_distribution!r}, expected "
            f"{expected_distribution!r}",
            law="report",
        )


class PairSamplerConformance:
    """Reusable pair-sampler conformance laws."""

    pytestmark: ClassVar[list[pytest.MarkDecorator]] = [pytest.mark.conformance]

    def test_pair_samples_are_deterministic(
        self, pair_sampler_fixture: PairSamplerFixture
    ) -> None:
        """Run the seed-determinism law."""
        pair_samples_are_deterministic(pair_sampler_fixture)

    def test_labels_are_exact(self, pair_sampler_fixture: PairSamplerFixture) -> None:
        """Run the exact-label law."""
        labels_are_exact(pair_sampler_fixture)

    def test_pair_reports_match_samples(
        self, pair_sampler_fixture: PairSamplerFixture
    ) -> None:
        """Run the report law."""
        pair_reports_match_samples(pair_sampler_fixture)


PAIR_SAMPLER_CONFORMANCE_LAWS: Final = (
    PairSamplerConformance.test_pair_samples_are_deterministic,
    PairSamplerConformance.test_labels_are_exact,
    PairSamplerConformance.test_pair_reports_match_samples,
)
