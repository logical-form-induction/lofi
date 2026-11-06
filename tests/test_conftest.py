"""The test harness preserves modules imported by its caller."""

import runpy
from pathlib import Path

from lofi import _checks


def test_loading_the_harness_preserves_public_exception_identity() -> None:
    law_error = _checks.LawViolationError
    numerical_error = _checks.NumericalFailureError

    runpy.run_path(str(Path(__file__).with_name("conftest.py")))

    assert _checks.LawViolationError is law_error
    assert _checks.NumericalFailureError is numerical_error
