"""Logical fragments served by prover translators and engines."""

from typing import NamedTuple

__all__ = [
    "FIRST_ORDER_CAPABILITY",
    "HIGHER_ORDER_CAPABILITY",
    "QUANTIFIER_FREE_CAPABILITY",
    "TranslatorCapability",
    "joint_capability",
]


class TranslatorCapability(NamedTuple):
    """The logical resources one translator and engine pairing serves."""

    order: str
    quantifiers: bool
    functions: bool
    equality: bool
    world_types: tuple[str, ...]


QUANTIFIER_FREE_CAPABILITY = TranslatorCapability(
    order="first",
    quantifiers=False,
    functions=True,
    equality=False,
    world_types=(),
)
"""The fragment served by exhaustive propositional translation."""

FIRST_ORDER_CAPABILITY = TranslatorCapability(
    order="first",
    quantifiers=True,
    functions=True,
    equality=True,
    world_types=(),
)
"""The fragment served by first-order TPTP and SMT translation."""

HIGHER_ORDER_CAPABILITY = TranslatorCapability(
    order="higher",
    quantifiers=True,
    functions=True,
    equality=True,
    world_types=("*",),
)
"""The fragment served by THF translation."""


_ORDER_RANK = {"first": 0, "second": 1, "higher": 2}


def _narrower_order(translator: str, engine: str, /) -> str:
    """Return the narrower comparable order, preserving open engine orders."""
    if translator == engine:
        return translator

    translator_rank = _ORDER_RANK.get(translator)
    engine_rank = _ORDER_RANK.get(engine)

    if translator_rank is None or engine_rank is None:
        return engine

    return translator if translator_rank < engine_rank else engine


def _shared_world_types(
    translator: tuple[str, ...], engine: tuple[str, ...], /
) -> tuple[str, ...]:
    """Return the world types both the translator and engine serve."""
    if "*" in translator:
        return engine

    if "*" in engine:
        return translator

    return tuple(sorted(set(translator) & set(engine)))


def joint_capability(
    translator: TranslatorCapability,
    engine: TranslatorCapability,
    /,
) -> TranslatorCapability:
    """Return the fragment jointly served by a translator and an engine."""
    return TranslatorCapability(
        order=_narrower_order(translator.order, engine.order),
        quantifiers=translator.quantifiers and engine.quantifiers,
        functions=translator.functions and engine.functions,
        equality=translator.equality and engine.equality,
        world_types=_shared_world_types(translator.world_types, engine.world_types),
    )
