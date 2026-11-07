"""Component output metadata."""

import enum

import didactic.api as dx

__all__ = ["DeclaredOutput", "DtypeClass"]


class DtypeClass(enum.StrEnum):
    """Classifies a produced value's element type."""

    #: Real-valued at the precision selected by policy.
    REAL = "real"

    #: Integral: an index, a code or a count.
    INTEGER = "integer"

    #: Two valued: a mask or a flag.
    BOOLEAN = "boolean"


class DeclaredOutput(dx.Model, extra="forbid"):
    """Declares output shape and element type.

    Parameters
    ----------
    width : int
        Size of the final axis.
    rank : int
        Number of tensor axes.
    dtype_class : DtypeClass
        Element-type category.
    """

    width: int
    rank: int
    dtype_class: DtypeClass

    __axioms__ = (
        dx.axiom("width > 0", message="width must be positive"),
        dx.axiom("rank > 0", message="rank must be positive"),
    )
