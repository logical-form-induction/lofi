from typing import TYPE_CHECKING, ClassVar, Final, NamedTuple, NoReturn

import pytest

from lofi._checks import LawViolationError

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    from lofi.data._selection import CorpusProfile, SelectionPolicy, SelectionReport

__all__ = [
    "SELECTION_POLICY_LAWS",
    "SelectionPolicyConformance",
    "SelectionPolicyFixture",
]


class SelectionPolicyFixture[C: Hashable](NamedTuple):
    """Inputs for selection-policy conformance laws.

    Parameters
    ----------
    policy : SelectionPolicy[C, C]
        The policy under test.
    small : CorpusProfile[C]
        The smaller of two corpora drawn from one declaration.
    large : CorpusProfile[C]
        The larger of the two, drawn from the same declaration.
    tolerance : float
        How far the two covered mass shares may differ.
    """

    policy: SelectionPolicy[C, C]
    small: CorpusProfile[C]
    large: CorpusProfile[C]
    tolerance: float


def _fail(detail: str, *, law: str) -> NoReturn:
    message = f"{law}: {detail}"

    raise LawViolationError(message)


def _budget[C: Hashable](
    policy: SelectionPolicy[C, C], profile: CorpusProfile[C]
) -> SelectionReport[C]:
    selected = policy.selected(profile)

    return policy.report(profile, {form: profile.counts[form] for form in selected})


def the_selector_declares_no_absolute_count[C: Hashable](
    fixture: SelectionPolicyFixture[C],
) -> None:
    law = "the_selector_declares_no_absolute_count"

    if not fixture.policy.selector.size_invariant:
        _fail(
            f"the policy selects by {fixture.policy.selector!r}, which counts",
            law=law,
        )


def every_selection_names_its_selector[C: Hashable](
    fixture: SelectionPolicyFixture[C],
) -> None:
    law = "every_selection_names_its_selector"

    for profile in (fixture.small, fixture.large):
        report = _budget(fixture.policy, profile)

        if report.selector != fixture.policy.selector or not report.size_invariant:
            _fail(
                f"a selection under {fixture.policy.selector!r} reported "
                f"{report.selector!r}",
                law=law,
            )


def applying_a_policy_does_not_change_it[C: Hashable](
    fixture: SelectionPolicyFixture[C],
) -> None:
    law = "applying_a_policy_does_not_change_it"
    before = repr(fixture.policy).encode()
    fixture.policy.selected(fixture.small)
    fixture.policy.selected(fixture.large)
    after = repr(fixture.policy).encode()

    if before != after:
        _fail(f"the declaration read {before!r} and now reads {after!r}", law=law)

    if fixture.policy != fixture.policy.with_seed(fixture.policy.seed):
        _fail("the policy no longer equals its own reseeding", law=law)


def the_same_mass_share_is_covered_at_two_sizes[C: Hashable](
    fixture: SelectionPolicyFixture[C],
) -> None:
    law = "the_same_mass_share_is_covered_at_two_sizes"

    if fixture.small.tokens >= fixture.large.tokens:
        _fail(
            f"the two corpora hold {fixture.small.tokens} and "
            f"{fixture.large.tokens} tokens, so neither is the larger",
            law=law,
        )

    small = _budget(fixture.policy, fixture.small).mass_share
    large = _budget(fixture.policy, fixture.large).mass_share

    if abs(small - large) > fixture.tolerance:
        _fail(
            f"the policy covered {small} of the smaller corpus and {large} of "
            f"the larger, further apart than {fixture.tolerance}",
            law=law,
        )


def every_selection_names_the_concordance_of_its_corpus[C: Hashable](
    fixture: SelectionPolicyFixture[C],
) -> None:
    law = "every_selection_names_the_concordance_of_its_corpus"

    for profile in (fixture.small, fixture.large):
        reported = _budget(fixture.policy, profile).concordance

        if reported != profile.concordance:
            _fail(
                f"a selection reported the concordance {reported} on a corpus "
                f"whose concordance is {profile.concordance}",
                law=law,
            )


SELECTION_POLICY_LAWS: Final = (
    the_selector_declares_no_absolute_count,
    every_selection_names_its_selector,
    applying_a_policy_does_not_change_it,
    the_same_mass_share_is_covered_at_two_sizes,
    every_selection_names_the_concordance_of_its_corpus,
)

_LAW_NAMES: Final = tuple(law.__name__ for law in SELECTION_POLICY_LAWS)


class SelectionPolicyConformance:
    """Reusable selection-policy conformance laws.

    A subclass supplies one ``selection_policy_fixture`` fixture: the policy,
    two corpora of different sizes drawn from one declaration, and how far the
    covered mass shares may differ.
    """

    pytestmark: ClassVar[list[pytest.MarkDecorator]] = [pytest.mark.conformance]

    @pytest.mark.parametrize("law", SELECTION_POLICY_LAWS, ids=_LAW_NAMES)
    def test_the_policy_holds_to_each_law[C: Hashable](
        self,
        law: Callable[[SelectionPolicyFixture[C]], None],
        selection_policy_fixture: SelectionPolicyFixture[C],
    ) -> None:
        law(selection_policy_fixture)
