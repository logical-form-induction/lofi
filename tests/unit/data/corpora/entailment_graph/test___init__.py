from typing import TYPE_CHECKING

from lofi.data.corpora import entailment_graph
from lofi.data.corpora.entailment_graph import _generator, _reader, spec

if TYPE_CHECKING:
    from types import ModuleType


def test_package_re_exports_the_entailment_graph_surface() -> None:
    modules: tuple[ModuleType, ...] = (_generator, _reader, spec)
    names = {name for module in modules for name in module.__all__}

    assert set(entailment_graph.__all__) == names

    for module in modules:
        for name in module.__all__:
            assert vars(entailment_graph)[name] is vars(module)[name]
