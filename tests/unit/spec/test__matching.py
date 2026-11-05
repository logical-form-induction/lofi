from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal, Self, override

import didactic.api as dx
import pytest

from lofi.spec._matching import (
    AlphaMatchingSpec,
    FeatureErasureMatchingSpec,
    IdentityMatchingSpec,
    MatchingSpec,
    ProductMatchingSpec,
    SubsumptionMatchingSpec,
    require_reifier,
)
from lofi.terms import (
    Alpha,
    FeatureErasure,
    Identity,
    Product,
    Subsumption,
    Term,
    TermMatching,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


def _check_r7[S, R](spec: S, runtime: R, reify: Callable[[R], S]) -> None:
    """Require a boundary adapter to recover the declaration it built.

    Parameters
    ----------
    spec
        The source declaration.
    runtime
        The native value built from it.
    reify
        The typed inverse for the declaration variant.
    """
    rebuilt = reify(runtime)

    assert rebuilt == spec, f"{spec!r} reified as {rebuilt!r}"


@contextmanager
def _isolated_variants(
    *roots: type[dx.TaggedUnion],
) -> Iterator[None]:
    """Restore tagged-union registries after temporary variants leave scope."""
    snapshots = tuple((root, dict(root.__variants__)) for root in roots)

    try:
        yield
    finally:
        for root, snapshot in snapshots:
            root.__variants__.clear()
            root.__variants__.update(snapshot)


class _CallableDescriptorFunction:
    """A callable stored behind a malformed class descriptor."""

    def __call__(self) -> None:
        """Return without producing a reified declaration."""


class _MalformedReifier:
    """Looks function-backed but produces a noncallable on class access."""

    __slots__ = ()

    @property
    def __func__(self) -> _CallableDescriptorFunction:
        """Return a callable resembling a classmethod's stored function."""
        return _CallableDescriptorFunction()

    def __get__[Owner](
        self, instance: Owner | None, owner: type[Owner] | None = None
    ) -> None:
        """Return no callable when read through a class."""
        del instance, owner


def test_each_matching_variant_builds_its_congruence() -> None:
    identity = IdentityMatchingSpec()
    alpha = AlphaMatchingSpec()
    erasure = FeatureErasureMatchingSpec()
    product = ProductMatchingSpec(left=identity, right=alpha)
    subsumption = SubsumptionMatchingSpec()

    identity_runtime = identity.build()
    alpha_runtime = alpha.build()
    erasure_runtime = erasure.build()
    product_runtime = product.build()
    subsumption_runtime = subsumption.build()

    assert isinstance(identity_runtime, Identity)
    assert isinstance(alpha_runtime, Alpha)
    assert isinstance(erasure_runtime, FeatureErasure)
    assert isinstance(product_runtime, Product)
    assert isinstance(subsumption_runtime, Subsumption)
    assert product_runtime.has_key
    assert subsumption_runtime.has_key
    assert isinstance(product_runtime, TermMatching)
    assert isinstance(subsumption_runtime, TermMatching)

    _check_r7(identity, identity_runtime, identity.reify)
    _check_r7(alpha, alpha_runtime, alpha.reify)
    _check_r7(erasure, erasure_runtime, erasure.reify)
    _check_r7(product, product_runtime, product.reify)
    _check_r7(subsumption, subsumption_runtime, subsumption.reify)


def test_a_matching_variant_without_a_reifier_is_refused_at_class_creation() -> None:
    with pytest.raises(TypeError, match=r"IncompleteMatchingSpec.*reify"):

        class IncompleteMatchingSpec(MatchingSpec, extra="forbid"):
            kind: Literal["incomplete-matching"] = "incomplete-matching"

    assert "incomplete-matching" not in MatchingSpec.__variants__


def test_matching_variants_refuse_reifiers_that_are_not_class_level() -> None:
    with pytest.raises(TypeError, match=r"NonCallableMatchingSpec.*reify.*class-level"):

        class NonCallableMatchingSpec(MatchingSpec, extra="forbid"):
            kind: Literal["non-callable-matching"] = "non-callable-matching"
            reify = None

            @override
            def build(self) -> Identity[Term]:
                return Identity[Term]()

    with pytest.raises(TypeError, match=r"InstanceMatchingSpec.*reify.*class-level"):

        class InstanceMatchingSpec(MatchingSpec, extra="forbid"):
            kind: Literal["instance-matching"] = "instance-matching"

            @override
            def build(self) -> Identity[Term]:
                return Identity[Term]()

            def reify(self, matching: Identity[Term], /) -> Self:
                del matching

                return self

    malformed = _MalformedReifier()

    assert callable(malformed.__func__)
    assert malformed.__get__(None, MatchingSpec) is None

    with pytest.raises(TypeError, match=r"MalformedMatchingSpec.*reify.*class-level"):

        class MalformedMatchingSpec(MatchingSpec, extra="forbid"):
            kind: Literal["malformed-matching"] = "malformed-matching"
            reify = malformed

            @override
            def build(self) -> Identity[Term]:
                return Identity[Term]()

    with pytest.raises(TypeError, match=r"WideMatchingSpec.*reify.*exactly"):

        class WideMatchingSpec(MatchingSpec, extra="forbid"):
            kind: Literal["wide-matching"] = "wide-matching"

            @override
            def build(self) -> Identity[Term]:
                return Identity[Term]()

            @classmethod
            def reify(cls, matching: Identity[Term], extra: int, /) -> Self:
                del matching, extra

                return cls()

    with pytest.raises(TypeError, match=r"AsyncMatchingSpec.*reify.*exactly"):

        class AsyncMatchingSpec(MatchingSpec, extra="forbid"):
            kind: Literal["async-matching"] = "async-matching"

            @override
            def build(self) -> Identity[Term]:
                return Identity[Term]()

            @classmethod
            async def reify(cls, matching: Identity[Term], /) -> Self:
                del matching

                return cls()

    with pytest.raises(TypeError, match=r"GeneratorMatchingSpec.*reify.*exactly"):

        class GeneratorMatchingSpec(MatchingSpec, extra="forbid"):
            kind: Literal["generator-matching"] = "generator-matching"

            @override
            def build(self) -> Identity[Term]:
                return Identity[Term]()

            @classmethod
            def reify(cls, matching: Identity[Term], /) -> Iterator[Self]:
                del matching

                yield cls()

    assert "non-callable-matching" not in MatchingSpec.__variants__
    assert "instance-matching" not in MatchingSpec.__variants__
    assert "malformed-matching" not in MatchingSpec.__variants__
    assert "wide-matching" not in MatchingSpec.__variants__
    assert "async-matching" not in MatchingSpec.__variants__
    assert "generator-matching" not in MatchingSpec.__variants__


def test_a_variant_cannot_bypass_its_installed_builder() -> None:
    with (
        _isolated_variants(MatchingSpec),
        pytest.raises(TypeError, match=r"lofi\.matching"),
    ):

        class ClosedMatchingSpec(MatchingSpec, extra="forbid"):
            kind: Literal["closed-matching"] = "closed-matching"

            @override
            def build(self) -> Identity[Term]:
                return Identity[Term]()

            @classmethod
            def reify(cls, matching: Identity[Term], /) -> Self:
                del matching

                return cls()

    assert "closed-matching" not in MatchingSpec.__variants__


def test_the_registration_check_accepts_a_class_level_inverse() -> None:
    require_reifier(IdentityMatchingSpec)

    with pytest.raises(TypeError, match=r"MatchingSpec\.reify"):
        require_reifier(MatchingSpec)


def test_a_variant_rebuilds_from_its_json_through_the_root() -> None:
    for variant in (
        IdentityMatchingSpec(),
        AlphaMatchingSpec(),
        FeatureErasureMatchingSpec(),
        ProductMatchingSpec(left=IdentityMatchingSpec(), right=AlphaMatchingSpec()),
        SubsumptionMatchingSpec(),
    ):
        assert MatchingSpec.model_validate_json(variant.model_dump_json()) == variant

    with pytest.raises(dx.ValidationError):
        MatchingSpec.model_validate({"kind": "unclaimed"})
