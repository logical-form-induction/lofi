from typing import TYPE_CHECKING

from lofi.data.corpora import nli
from lofi.data.corpora.nli import _codec

if TYPE_CHECKING:
    from types import ModuleType


def test_package_re_exports_the_nli_surface() -> None:
    modules: tuple[ModuleType, ...] = (_codec,)
    names = {name for module in modules for name in module.__all__}

    assert set(nli.__all__) == names

    for module in modules:
        for name in module.__all__:
            assert vars(nli)[name] is vars(module)[name]
