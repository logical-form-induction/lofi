from typing import TYPE_CHECKING

from lofi.data.corpora import ccgbank
from lofi.data.corpora.ccgbank import (
    _auto_parse,
    _parg_parse,
    roundtrip,
)

if TYPE_CHECKING:
    from types import ModuleType


def test_the_package_re_exports_the_lairs_free_ccgbank_interface() -> None:
    modules: tuple[ModuleType, ...] = (
        _auto_parse,
        _parg_parse,
        roundtrip,
    )
    names = {name for module in modules for name in module.__all__}

    assert set(ccgbank.__all__) == names

    for module in modules:
        for name in module.__all__:
            assert vars(ccgbank)[name] is vars(module)[name]
