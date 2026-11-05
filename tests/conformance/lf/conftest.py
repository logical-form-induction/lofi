import pytest
from tests.conformance.lf.adapter import LogicalFormAdapter


@pytest.fixture(scope="session")
def lf_adapter() -> LogicalFormAdapter:
    return LogicalFormAdapter()
