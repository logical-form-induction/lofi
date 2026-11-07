from typing import TYPE_CHECKING

from lofi.data.corpora import megaacceptability
from lofi.data.corpora.megaacceptability import _codec, spec

if TYPE_CHECKING:
    from types import ModuleType


def test_package_re_exports_the_codec_surface() -> None:
    modules: tuple[ModuleType, ...] = (_codec, spec)
    names = {name for module in modules for name in module.__all__}

    assert set(megaacceptability.__all__) == names

    for module in modules:
        for name in module.__all__:
            assert vars(megaacceptability)[name] is vars(module)[name]
