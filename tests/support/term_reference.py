"""Test reference category and formula signatures."""

from typing import Final

from lofi.terms._ports import Calculus
from lofi.terms._text import ConstructorNotation, ConstructorSyntax
from lofi.terms._values import BindingSite, Operator, Signature, Vocabulary

__all__ = ["CALCULUS", "categories", "formulas", "infix_syntax"]

CALCULUS: Final = Calculus(lam="lam", app="app")
"""The formula abstraction and application operators."""


def infix_syntax(
    signature: Signature,
    *constructors: tuple[str, str],
    max_depth: int | None = None,
) -> ConstructorSyntax:
    """Build declared left-associative infix syntax for a test signature."""
    return ConstructorSyntax(
        signature,
        tuple(
            ConstructorNotation.infix(
                name,
                token,
                precedence=10,
                associativity="left",
            )
            for token, name in constructors
        ),
        max_depth=max_depth,
    )


def categories() -> Signature:
    """Return the categorial reference signature.

    Returns
    -------
    Signature
        Reference category signature.
    """
    return Signature(
        "categories",
        ("Cat",),
        (
            Operator("np", "Cat"),
            Operator("n", "Cat"),
            Operator("s", "Cat"),
            Operator("fwd", "Cat", arg_sorts=("Cat", "Cat"), family="slash"),
            Operator("bwd", "Cat", arg_sorts=("Cat", "Cat"), family="slash"),
        ),
        features={"s": ("dcl", "b", "ng")},
    )


def formulas() -> Signature:
    """Return the reference formula signature with binding.

    Returns
    -------
    Signature
        Reference logical-form signature over one sort ``Lf``, whose
        vocabulary is the four constants ``p``, ``q``, ``r`` and ``s`` and
        whose abstraction binds a variable of ``Lf`` in its one argument.
    """
    return Signature(
        "formulas",
        ("Lf",),
        (
            Operator(
                CALCULUS.app,
                "Lf",
                arg_sorts=("Lf", "Lf"),
                family="application",
            ),
            Operator(
                CALCULUS.lam,
                "Lf",
                arg_sorts=("Lf",),
                binding=(BindingSite((0,), "Lf"),),
                family="binder",
            ),
            Operator(
                "ite",
                "Lf",
                arg_sorts=("Lf", "Lf", "Lf"),
                family="connective",
            ),
        ),
        vocabularies=(Vocabulary("Lf", ("p", "q", "r", "s")),),
    )
