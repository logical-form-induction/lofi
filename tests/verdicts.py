"""Typed entailment evidence used by tests outside the logical-form unit."""

from lofi.lf import CounterSatisfiable, Proof, Proved, Refuted

__all__ = ["proved", "refuted"]


def proved(derivation: str, /, *, system: str = "fixture") -> Proved:
    """Return a typed fixture proof verdict."""
    return Proved(Proof(system, derivation))


def refuted(detail: str, /, *, system: str = "fixture") -> Refuted:
    """Return typed fixture counter-satisfiable evidence."""
    return Refuted(CounterSatisfiable(system, "CounterSatisfiable", detail))
