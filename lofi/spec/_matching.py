"""Term congruence declarations.

A grammar declares the congruence its categories are compared under, and each
rule may declare its own congruence for matching its premises. Every
declaration root here registers only variants that supply a typed inverse, so
that a built congruence reifies to the declaration it came from.
"""

from inspect import (
    CO_ASYNC_GENERATOR,
    CO_COROUTINE,
    CO_GENERATOR,
    CO_VARARGS,
    CO_VARKEYWORDS,
)
from types import FunctionType
from typing import TYPE_CHECKING, Literal, Self, cast

import didactic.api as dx

from lofi._frozen import FrozenValue
from lofi.registry import resolve_plugin
from lofi.terms import Alpha, FeatureErasure, Identity, Product, Subsumption, Term

if TYPE_CHECKING:
    from collections import abc

    from lofi.terms import Matching

__all__ = [
    "AlphaMatchingSpec",
    "FeatureErasureMatchingSpec",
    "IdentityMatchingSpec",
    "MatchingPlugin",
    "MatchingSpec",
    "ProductMatchingSpec",
    "SubsumptionMatchingSpec",
    "matching_plugin",
    "require_reifier",
]


def require_reifier(cls: type[dx.TaggedUnion]) -> None:
    """Refuse a registered declaration variant without its typed inverse.

    Parameters
    ----------
    cls : type[dx.TaggedUnion]
        The variant being created.

    Raises
    ------
    TypeError
        If the variant does not define ``reify`` as its own class-level method.
    """
    descriptor = cls.__dict__.get("reify")

    if type(descriptor) is not classmethod:
        msg = (
            f"{cls.__name__}.reify must be its own class-level method accepting "
            "exactly the class and one runtime positional argument before the "
            "variant can be registered"
        )
        raise TypeError(msg)

    implementation = descriptor.__func__
    bound = descriptor.__get__(None, cls)
    disallowed_flags = (
        CO_VARARGS | CO_VARKEYWORDS | CO_COROUTINE | CO_GENERATOR | CO_ASYNC_GENERATOR
    )
    exact_signature = (
        isinstance(implementation, FunctionType)
        and implementation.__code__.co_argcount == 2
        and implementation.__code__.co_kwonlyargcount == 0
        and implementation.__code__.co_flags & disallowed_flags == 0
        and implementation.__defaults__ is None
        and implementation.__kwdefaults__ is None
    )

    if not callable(implementation) or not callable(bound) or not exact_signature:
        msg = (
            f"{cls.__name__}.reify must be a callable class-level method accepting "
            "exactly the class and one runtime positional argument before the "
            "variant can be registered"
        )
        raise TypeError(msg)


class MatchingSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """A term congruence declaration."""

    def __init_subclass__[K](
        cls, *, discriminator: str | None = None, **kwargs: K
    ) -> None:
        """Register only variants that supply their typed inverse."""
        require_reifier(cls)

        if "build" in cls.__dict__:
            msg = (
                f"{cls.__name__} declares a closed builder; matching variants are "
                "constructed only through the lofi.matching plugin group"
            )
            raise TypeError(msg)

        super().__init_subclass__(discriminator=discriminator, **kwargs)

    def build(self) -> Matching[Term]:
        """Build the selected installed matching relation."""
        return matching_plugin(self).construct(self)


class MatchingPlugin[Declaration](FrozenValue):
    """Pair one open matching declaration with its runtime builder."""

    _noun = "matching plugin"
    __slots__ = ("_build", "_spec")

    def __init__(
        self,
        *,
        spec: type[Declaration],
        build: abc.Callable[[Declaration], Matching[Term]],
    ) -> None:
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_build", build)

    @property
    def spec(self) -> type[Declaration]:
        """Return the declaration class this plugin builds."""
        return self._spec

    def construct(self, declaration: Declaration, /) -> Matching[Term]:
        """Build one relation from a validated declaration."""
        return self._build(declaration)


def matching_plugin(declaration: MatchingSpec, /) -> MatchingPlugin[MatchingSpec]:
    """Return the unique installed plugin selected by a declaration."""
    plugin = resolve_plugin(
        "matching", cast("str", declaration.kind), MatchingPlugin, type(declaration)
    )

    return cast("MatchingPlugin[MatchingSpec]", plugin)


class IdentityMatchingSpec(MatchingSpec, extra="forbid"):
    """Structural identity, under which a pattern matches exactly what it names."""

    kind: Literal["identity"] = "identity"

    @classmethod
    def reify(cls, matching: Identity[Term], /) -> Self:
        """Recover the identity declaration from its native relation.

        Parameters
        ----------
        matching : Identity[Term]
            The native identity relation.
        """
        del matching

        return cls()


class AlphaMatchingSpec(MatchingSpec, extra="forbid"):
    """Identity up to bound-variable renaming."""

    kind: Literal["alpha"] = "alpha"

    @classmethod
    def reify(cls, matching: Alpha[Term], /) -> Self:
        """Recover the alpha declaration from its native relation.

        Parameters
        ----------
        matching : Alpha[Term]
            The native alpha relation.
        """
        del matching

        return cls()


class FeatureErasureMatchingSpec(MatchingSpec, extra="forbid"):
    """Identity of feature-erased cores."""

    kind: Literal["feature-erasure"] = "feature-erasure"

    @classmethod
    def reify(cls, matching: FeatureErasure, /) -> Self:
        """Recover the feature erasure declaration from its native relation.

        Parameters
        ----------
        matching : FeatureErasure
            The native feature erasure relation.
        """
        del matching

        return cls()


class ProductMatchingSpec(MatchingSpec, extra="forbid"):
    """Intersection of two declared relations over the same terms."""

    kind: Literal["product"] = "product"
    left: MatchingSpec
    right: MatchingSpec

    @classmethod
    def reify(cls, matching: Product[Term], /) -> Self:
        """Recover a product declaration from its component relations."""
        return cls(
            left=_reify(matching.left),
            right=_reify(matching.right),
        )


class SubsumptionMatchingSpec(MatchingSpec, extra="forbid"):
    """Mutual structural subsumption with free variables as placeholders."""

    kind: Literal["subsumption"] = "subsumption"

    @classmethod
    def reify(cls, matching: Subsumption, /) -> Self:
        """Recover the subsumption declaration."""
        del matching
        return cls()


def _reify(matching: Matching[Term], /) -> MatchingSpec:
    """Recover a built-in matching declaration."""
    if isinstance(matching, Alpha):
        return AlphaMatchingSpec.reify(matching)
    if isinstance(matching, Identity):
        return IdentityMatchingSpec.reify(matching)
    if isinstance(matching, FeatureErasure):
        return FeatureErasureMatchingSpec.reify(matching)
    if isinstance(matching, Subsumption):
        return SubsumptionMatchingSpec.reify(matching)
    if isinstance(matching, Product):
        return ProductMatchingSpec.reify(matching)

    msg = f"no matching declaration reifies {matching!r}"
    raise TypeError(msg)


def _identity(declaration: IdentityMatchingSpec, /) -> Matching[Term]:
    del declaration
    return Identity[Term]()


def _alpha(declaration: AlphaMatchingSpec, /) -> Matching[Term]:
    del declaration
    return Alpha[Term]()


def _feature_erasure(declaration: FeatureErasureMatchingSpec, /) -> Matching[Term]:
    del declaration
    return FeatureErasure()


def _product(declaration: ProductMatchingSpec, /) -> Matching[Term]:
    return Product(declaration.left.build(), declaration.right.build())


def _subsumption(declaration: SubsumptionMatchingSpec, /) -> Matching[Term]:
    del declaration
    return Subsumption()


IDENTITY_PLUGIN = MatchingPlugin(spec=IdentityMatchingSpec, build=_identity)
ALPHA_PLUGIN = MatchingPlugin(spec=AlphaMatchingSpec, build=_alpha)
FEATURE_ERASURE_PLUGIN = MatchingPlugin(
    spec=FeatureErasureMatchingSpec,
    build=_feature_erasure,
)
PRODUCT_PLUGIN = MatchingPlugin(spec=ProductMatchingSpec, build=_product)
SUBSUMPTION_PLUGIN = MatchingPlugin(
    spec=SubsumptionMatchingSpec,
    build=_subsumption,
)
