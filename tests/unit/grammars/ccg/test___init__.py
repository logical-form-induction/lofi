from lofi.grammars import ccg
from lofi.grammars.ccg._assembly import CCGAssembly
from lofi.grammars.ccg._functors import FunctorView


def test_the_package_exports_the_ccg_runtime_capabilities() -> None:
    assert ccg.__all__ == ["CCGAssembly", "FunctorView"]
    assert ccg.CCGAssembly is CCGAssembly
    assert ccg.FunctorView is FunctorView
