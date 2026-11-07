from typing import TYPE_CHECKING

from lofi import registry
from lofi.registry import _errors, _plugin, _registry

if TYPE_CHECKING:
    from types import ModuleType


def test_the_package_re_exports_its_public_modules() -> None:
    sources: dict[str, ModuleType] = {}

    for module in (_errors, _plugin, _registry):
        sources |= dict.fromkeys(module.__all__, module)

    assert sorted(registry.__all__) == sorted(sources)

    for name, module in sources.items():
        assert vars(registry)[name] is vars(module)[name]
