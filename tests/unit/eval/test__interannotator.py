import json
from itertools import combinations
from math import cos, log, pi, sin, sqrt, tau
from pathlib import Path
from statistics import correlation, fmean
from typing import TYPE_CHECKING, Final, cast

import pytest

from lofi.eval._interannotator import (
    INTERANNOTATOR_COEFFICIENTS,
    InterannotatorCeilingRow,
    ParticipantResponse,
    interannotator_ceiling,
)
from lofi.stats import Bootstrap

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from lofi.eval._interannotator import InterannotatorCoefficient

UNVERIFIED_QUOTED_CEILING: Final = (0.70, 0.62, 0.78)
"""The unverified published point, lower bound, and upper bound."""


def _bootstrap(seed: int = 7, replicates: int = 999) -> Bootstrap:
    return Bootstrap(
        unit="participant-pair", replicates=replicates, seed=seed, level=0.95
    )


def _examples(
    participants: Mapping[str, tuple[tuple[str, float], ...]],
) -> tuple[ParticipantResponse, ...]:
    return tuple(
        ParticipantResponse(participant, item, value)
        for participant, responses in participants.items()
        for item, value in responses
    )


def ceiling(
    responses: Iterable[ParticipantResponse],
    *,
    bootstrap: Bootstrap,
    coefficients: tuple[InterannotatorCoefficient, ...] = INTERANNOTATOR_COEFFICIENTS,
) -> tuple[InterannotatorCeilingRow, ...]:
    return interannotator_ceiling(
        responses, coefficients=coefficients, bootstrap=bootstrap
    )


def test_ceiling_reports_both_coefficients_and_hand_checked_counts() -> None:
    values = (
        (1.0, 2.0, 3.0, 4.0, 5.0),
        (1.0, 4.0, 9.0, 16.0, 25.0),
        (1.0, 8.0, 27.0, 64.0, 125.0),
    )
    examples = _examples(
        {
            participant: tuple(
                (f"item-{position}", value) for position, value in enumerate(responses)
            )
            for participant, responses in zip(("a", "b", "c"), values, strict=True)
        }
    )

    spearman, pearson = ceiling(
        iter(examples),
        bootstrap=_bootstrap(),
    )
    expected_pearson = fmean(
        correlation(left, right) for left, right in combinations(values, 2)
    )

    assert spearman.coefficient == "spearman"
    assert spearman.result.value == 1.0
    assert pearson.coefficient == "pearson"
    pearson_value = pearson.result.value
    spearman_value = spearman.result.value

    assert pearson_value == pytest.approx(expected_pearson)
    assert pearson_value is not None
    assert spearman_value is not None
    assert pearson_value < spearman_value

    for row in (spearman, pearson):
        assert row.participant_count == 3
        assert row.pair_count == 3
        assert row.result.n == 3
        assert row.result.components == {
            "participants": 3.0,
            "pairs": 3.0,
            "bootstrap_replicates": 999.0,
        }
        assert row.result.resampling.unit == "participant-pair"
        assert row.result.resampling.seed == 7
        assert row.result.ci is not None
        assert row.result.value is not None
        assert row.result.ci.covers(row.result.value)


def test_tied_rank_spearman_matches_a_hand_derived_value() -> None:
    rows = {
        "a": tuple(zip(("w", "x", "y", "z"), (1.0, 2.0, 2.0, 4.0), strict=True)),
        "b": tuple(zip(("w", "x", "y", "z"), (4.0, 1.0, 2.0, 3.0), strict=True)),
        "c": tuple(zip(("w", "x", "y", "z"), (1.0, 2.0, 2.0, 4.0), strict=True)),
    }
    spearman, _ = ceiling(
        _examples(rows),
        bootstrap=_bootstrap(),
    )

    assert spearman.result.value == pytest.approx((1.0 - 2.0 / sqrt(10.0)) / 3.0)


def test_pearson_remains_defined_when_finite_products_would_overflow() -> None:
    rows = {
        "a": (1e200, 2e200, 3e200),
        "b": (1e200, 2e200, 3e200),
        "c": (3e200, 2e200, 1e200),
    }
    examples = tuple(
        ParticipantResponse(participant, f"item-{position}", value)
        for participant, responses in rows.items()
        for position, value in enumerate(responses)
    )

    spearman, pearson = ceiling(
        examples,
        bootstrap=_bootstrap(),
    )

    assert spearman.result.value == pytest.approx(-1.0 / 3.0)
    assert pearson.result.value == pytest.approx(-1.0 / 3.0)


def test_only_participants_with_the_same_item_list_form_pairs() -> None:
    rows = {
        "a": (("x", 1.0), ("y", 2.0), ("z", 3.0)),
        "b": (("z", 4.0), ("x", 2.0), ("y", 3.0)),
        "c": (("x", 3.0), ("y", 2.0), ("z", 1.0)),
        "d": (("q", 1.0), ("r", 2.0), ("s", 3.0)),
        "e": (("q", 3.0), ("r", 2.0), ("s", 1.0)),
        "unpaired": (("other", 1.0), ("items", 2.0)),
    }

    reported = ceiling(
        _examples(rows),
        bootstrap=_bootstrap(),
    )

    assert all(row.participant_count == 5 for row in reported)
    assert all(row.pair_count == 4 for row in reported)


def test_row_values_are_frozen_hashable_and_serializable() -> None:
    rows = {
        "a": (("x", 1.0), ("y", 2.0), ("z", 3.0)),
        "b": (("x", 1.0), ("y", 3.0), ("z", 2.0)),
        "c": (("x", 3.0), ("y", 2.0), ("z", 1.0)),
    }
    row = ceiling(
        _examples(rows),
        bootstrap=_bootstrap(),
    )[0]

    assert isinstance(row, InterannotatorCeilingRow)
    assert hash(row) == hash(tuple(row))
    assert row.to_spec()["coefficient"] == "spearman"
    assert row.to_spec()["participant_count"] == 3
    assert row.to_spec()["pair_count"] == 3
    assert row.to_spec()["bootstrap_replicates"] == 999
    assert json.loads(json.dumps(row.to_spec())) == row.to_spec()

    with pytest.raises(AttributeError):
        row.__setattr__("participant_count", 9)


class _NormalStream:
    """A deterministic Gaussian stream for the coverage fixture."""

    __slots__ = ("_state",)

    def __init__(self, seed: int) -> None:
        self._state = seed

    def _uniform(self) -> float:
        self._state = (1_664_525 * self._state + 1_013_904_223) % (1 << 32)

        return (self._state + 0.5) / (1 << 32)

    def draw(self) -> float:
        """Draw one standard-normal fixture value."""
        radius = sqrt(-2.0 * log(self._uniform()))

        return radius * cos(tau * self._uniform())


def test_the_pair_bootstrap_covers_a_known_rank_correlation() -> None:
    rank_correlation = 0.7
    gaussian_correlation = 2.0 * sin(pi * rank_correlation / 6.0)
    noise = sqrt(1.0 - gaussian_correlation**2)
    covered = 0

    for trial in range(100):
        stream = _NormalStream(6_100 + trial)
        examples: list[ParticipantResponse] = []

        for pair in range(30):
            left: list[float] = []
            right: list[float] = []

            for _ in range(200):
                value = stream.draw()
                left.append(value)
                right.append(gaussian_correlation * value + noise * stream.draw())

            for side, values in (("a", left), ("b", right)):
                examples.extend(
                    ParticipantResponse(f"{pair}-{side}", f"{pair}-{item}", value)
                    for item, value in enumerate(values)
                )

        interval = ceiling(
            examples,
            bootstrap=_bootstrap(seed=trial),
        )[0].result.ci
        assert interval is not None
        covered += interval.covers(rank_correlation)

    assert covered >= 90


@pytest.mark.parametrize(
    ("examples", "message"),
    [
        ((), "at least 2 same-list participant pairs"),
        (
            _examples(
                {
                    "a": (("x", 1.0), ("y", 2.0)),
                    "b": (("q", 1.0), ("r", 2.0)),
                }
            ),
            "at least 2 same-list participant pairs",
        ),
        (
            _examples(
                {
                    "a": (("x", 1.0), ("y", 1.0)),
                    "b": (("x", 2.0), ("y", 3.0)),
                    "c": (("x", 3.0), ("y", 2.0)),
                }
            ),
            "undefined correlation",
        ),
    ],
)
def test_undefined_pair_collections_are_refused(
    examples: tuple[ParticipantResponse, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ceiling(examples, bootstrap=_bootstrap())


def test_missing_anonymous_and_duplicate_responses_are_refused() -> None:
    missing = ParticipantResponse("a", "x", float("nan"))
    anonymous = ParticipantResponse("", "x", 1.0)
    duplicate = ParticipantResponse("a", "x", 1.0)

    with pytest.raises(ValueError, match="no finite response"):
        ceiling((missing,), bootstrap=_bootstrap())

    with pytest.raises(ValueError, match="has no participant"):
        ceiling((anonymous,), bootstrap=_bootstrap())

    with pytest.raises(ValueError, match="responds twice"):
        ceiling((duplicate, duplicate), bootstrap=_bootstrap())


def test_only_the_declared_coefficients_are_reported_at_any_replicate_count() -> None:
    rows = {
        "a": (("x", 1.0), ("y", 2.0), ("z", 3.0)),
        "b": (("x", 1.0), ("y", 3.0), ("z", 2.0)),
        "c": (("x", 3.0), ("y", 2.0), ("z", 1.0)),
    }

    (spearman,) = ceiling(
        _examples(rows),
        bootstrap=_bootstrap(replicates=2000),
        coefficients=("spearman",),
    )

    assert spearman.coefficient == "spearman"
    assert spearman.result.components["bootstrap_replicates"] == 2000.0


@pytest.mark.parametrize(
    "coefficients",
    [
        (),
        ("spearman", "spearman"),
        cast("tuple[InterannotatorCoefficient]", ("kendall",)),
    ],
)
def test_empty_repeated_or_unknown_coefficients_are_refused(
    coefficients: tuple[InterannotatorCoefficient, ...],
) -> None:
    with pytest.raises(ValueError, match="coefficients"):
        ceiling(
            _examples({"a": (("x", 1.0),)}),
            bootstrap=_bootstrap(),
            coefficients=coefficients,
        )


@pytest.mark.parametrize(
    ("bootstrap", "message"),
    [
        (cast("Bootstrap", 999), "is not one"),
        (
            Bootstrap(unit="participant", replicates=999, seed=1, level=0.95),
            "not 'participant-pair'",
        ),
        (
            Bootstrap(
                unit="participant-pair",
                replicates=999,
                seed=1,
                groups=("a", "b"),
                level=0.95,
            ),
            "already the resampling units",
        ),
    ],
)
def test_invalid_bootstrap_declarations_are_refused(
    bootstrap: Bootstrap,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        ceiling((), bootstrap=bootstrap)


def test_a_response_that_is_not_a_participant_response_is_refused() -> None:
    with pytest.raises(TypeError, match="not a ParticipantResponse"):
        ceiling(
            cast("tuple[ParticipantResponse, ...]", (("a", "x", 1.0),)),
            bootstrap=_bootstrap(),
        )


def test_quoted_ceiling_stays_an_unverified_test_fixture() -> None:
    source = Path(__file__).parents[3] / "lofi" / "eval" / "_interannotator.py"

    assert UNVERIFIED_QUOTED_CEILING == (0.70, 0.62, 0.78)
    assert "0.70" not in source.read_text(encoding="utf-8")
