from typing import TYPE_CHECKING

from lofi.data import deps
from lofi.data.deps import _ccg, _induced, _projection, _templates

if TYPE_CHECKING:
    from types import ModuleType


def test_the_package_publishes_what_its_modules_publish() -> None:
    source: dict[str, ModuleType] = {}

    for module in (_ccg, _induced, _projection, _templates):
        source |= dict.fromkeys(module.__all__, module)

    assert sorted(deps.__all__) == sorted(source)

    for name, module in source.items():
        assert vars(deps)[name] is vars(module)[name]
