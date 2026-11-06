import lofi.report
from lofi.report import _table


def test_the_package_publishes_what_its_table_module_publishes() -> None:
    assert sorted(lofi.report.__all__) == sorted(_table.__all__)

    for name in _table.__all__:
        assert getattr(lofi.report, name) is getattr(_table, name)
