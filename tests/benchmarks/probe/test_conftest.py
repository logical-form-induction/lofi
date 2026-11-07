"""Check that timing controls leave the surrounding test process unchanged."""

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest


def test_the_thread_pool_is_restored_after_success_and_failure(
    pytester: pytest.Pytester,
) -> None:
    pytester.makeconftest(Path(__file__).with_name("conftest.py").read_text())
    pytester.makepyfile(
        """
        import pytest
        import torch

        @pytest.fixture(scope="session", autouse=True)
        def starting_pool():
            previous = torch.get_num_threads()
            torch.set_num_threads(2)
            yield
            torch.set_num_threads(previous)

        def test_limited(single_threaded_cpu):
            assert torch.get_num_threads() == 1

        def test_after_success():
            assert torch.get_num_threads() == 2

        def test_failing(single_threaded_cpu):
            assert torch.get_num_threads() == 1
            pytest.fail("exercise fixture cleanup after a benchmark failure")

        def test_after_failure():
            assert torch.get_num_threads() == 2
        """
    )

    result = pytester.runpytest_subprocess("-q")

    result.assert_outcomes(passed=3, failed=1)
