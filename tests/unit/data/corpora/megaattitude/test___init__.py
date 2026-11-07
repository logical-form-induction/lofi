from typing import TYPE_CHECKING

from lofi.data.corpora import megaattitude
from lofi.data.corpora.megaattitude import _codec, spec

if TYPE_CHECKING:
    from types import ModuleType


def test_package_re_exports_the_codec_surface() -> None:
    modules: tuple[ModuleType, ...] = (_codec, spec)
    names = {name for module in modules for name in module.__all__}

    assert set(megaattitude.__all__) == names

    for module in modules:
        for name in module.__all__:
            assert vars(megaattitude)[name] is vars(module)[name]
