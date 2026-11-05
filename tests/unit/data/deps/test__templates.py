from copy import copy, deepcopy
from pickle import dumps, loads
from typing import cast

import pytest

from lofi.data.deps import (
    CategoryDirection,
    CCGDependencyConvention,
    CCGDependencyTemplate,
    CCGProjectionReport,
    CoindexedPath,
)


def _template() -> CCGDependencyTemplate[str]:
    return CCGDependencyTemplate(
        "subject-sharing",
        "synthetic-category",
        (
            CoindexedPath(("result", "argument"), 1),
            CoindexedPath(("argument", "argument"), 1),
        ),
    )


def test_template_declarations_are_frozen_hashable_and_pickleable() -> None:
    template = _template()
    convention = CCGDependencyConvention("synthetic", (template,))

    assert copy(template) is template
    assert deepcopy(convention) is convention
    assert loads(dumps(template)) == template
    assert loads(dumps(convention)) == convention
    assert len({convention, loads(dumps(convention))}) == 1

    with pytest.raises(AttributeError, match="immutable"):
        template.__setattr__("_name", "changed")


@pytest.mark.parametrize(
    ("path", "index", "error"),
    [
        ((), 1, ValueError),
        (cast("tuple[CategoryDirection, ...]", ("invalid",)), 1, ValueError),
        (("argument",), True, TypeError),
        (("argument",), 0, ValueError),
    ],
)
def test_a_coindexed_path_requires_structural_steps_and_a_positive_index(
    path: tuple[CategoryDirection, ...],
    index: int,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        CoindexedPath(path, index)


def test_a_template_requires_repeated_coindices_at_distinct_paths() -> None:
    with pytest.raises(ValueError, match="has no coindices"):
        CCGDependencyTemplate("empty", "category", ())

    with pytest.raises(ValueError, match="unshared coindices"):
        CCGDependencyTemplate(
            "unshared",
            "category",
            (CoindexedPath(("argument",), 1),),
        )

    with pytest.raises(ValueError, match="repeats a category path"):
        CCGDependencyTemplate(
            "duplicate",
            "category",
            (
                CoindexedPath(("argument",), 1),
                CoindexedPath(("argument",), 1),
            ),
        )


def test_reports_distinguish_absent_empty_and_unmatched_conventions() -> None:
    absent = CCGProjectionReport(None, 0, 0)
    empty = CCGProjectionReport("empty", 0, 0)
    unmatched = CCGProjectionReport("declared", 2, 0)

    assert not absent.convention_present
    assert absent.convention_name is None
    assert empty.convention_present
    assert empty.convention_name == "empty"
    assert unmatched.declared_template_count == 2
    assert unmatched.matched_template_count == 0
    assert len({absent, empty, unmatched}) == 3


@pytest.mark.parametrize(
    ("name", "declared", "matched", "error"),
    [
        (None, 1, 0, ValueError),
        ("declared", -1, 0, ValueError),
        ("declared", 0, True, TypeError),
    ],
)
def test_projection_reports_reject_inconsistent_counts(
    name: str | None,
    declared: int,
    matched: int,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        CCGProjectionReport(name, declared, matched)
