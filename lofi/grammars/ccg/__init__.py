"""Runtime assembly for declared directed and undirected CCG grammars."""

from lofi.grammars.ccg._assembly import CCGAssembly
from lofi.grammars.ccg._functors import FunctorView

__all__ = ["CCGAssembly", "FunctorView"]
