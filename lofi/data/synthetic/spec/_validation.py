"""Shared declaration axioms for synthetic data specifications."""

import didactic.api as dx

__all__ = ["all_named", "nonempty"]


def nonempty(field: str, message: str) -> dx.Axiom:
    """Return an axiom requiring a text field to hold a character."""
    return dx.axiom(f"len({field}) > 0", message=message)


def all_named(field: str, message: str) -> dx.Axiom:
    """Return an axiom requiring every member of a text tuple to hold a character."""
    return dx.axiom(f"all({field})", message=message)
