from typing import TYPE_CHECKING

from lofi.data.corpora import fracas
from lofi.data.corpora.fracas import _codec

if TYPE_CHECKING:
    from types import ModuleType


def test_package_re_exports_the_fracas_surface() -> None:
    modules: tuple[ModuleType, ...] = (_codec,)
    names = {name for module in modules for name in module.__all__}

    assert set(fracas.__all__) == names

    for module in modules:
        for name in module.__all__:
            assert vars(fracas)[name] is vars(module)[name]
