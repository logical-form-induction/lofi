"""Tests for baseline-system component declarations."""

from typing import Literal

import didactic.api as dx
import pytest

from lofi.spec.components.baseline import BaselineSystemSpec


class EmptyBaseline(BaselineSystemSpec):
    """A baseline variant that declares no system-specific behavior."""

    kind: Literal["empty-component-test"] = "empty-component-test"


def test_the_family_names_its_own_plugin_group() -> None:
    assert BaselineSystemSpec.family() == "baseline_system"


def test_the_family_root_promises_no_rows_or_conventions() -> None:
    root = EmptyBaseline()

    assert root.metrics() == ()
    assert root.conventions() == {}


def test_an_unknown_baseline_system_is_refused() -> None:
    with pytest.raises(dx.ValidationError):
        BaselineSystemSpec.model_validate({"kind": "unknown"})
