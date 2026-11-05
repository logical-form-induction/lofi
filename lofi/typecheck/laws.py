"""Typing-system and clamp-projection law checks."""

from typing import TYPE_CHECKING, Final, NamedTuple, NoReturn

from lofi._checks import LawViolationError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from lofi.deduce import Cell
    from lofi.terms import OpName, Term
    from lofi.typecheck._ports import ClampProjection, Substitution, TypingSystem
    from lofi.typecheck._values import TypingRule

__all__ = [
    "CLAMP_PROJECTION_LAWS",
    "ClampProjectionFixture",
    "CoherenceFixture",
    "CoherenceInstance",
    "CoherenceItem",
    "CoherenceReport",
    "TypingLawFixture",
    "a_meaning_is_read_exactly_when_declared",
    "a_projection_reads_the_same_reading_the_same_way",
    "check_clamp_projection_laws",
    "check_typing_laws",
    "coherence_square",
    "typing_coherence_is_subsumption",
]


class TypingLawFixture[L, C, T](NamedTuple):
    """Inputs for the typing coherence law."""

    logical_form: L
    category: C
    typeof: Callable[[L], T]
    erase: Callable[[C], T]
    subsumes: Callable[[T, T], Mapping[int, T] | None]


class CoherenceItem[C](NamedTuple):
    """One chart item with the logical forms its derivations read.

    Attributes
    ----------
    span : Cell
        The half-open token interval the item covers.
    category : C
        The category the item's derivations conclude.
    logical_forms : tuple[Term, ...]
        The meaning each derivation of the item reads, in derivation order.
    tag : str
        The normal-form tag telling the item apart from others packed on the
        same span and category, or empty when the grammar uses none.
    """

    span: Cell
    category: C
    logical_forms: tuple[Term, ...]
    tag: str = ""


class CoherenceInstance[C](NamedTuple):
    """One commuting instance of the square: a derivation's two readings agree.

    Attributes
    ----------
    span : Cell
        The half-open token interval the derivation covers.
    category : C
        The category the derivation concludes.
    logical_form : Term
        The meaning the derivation reads.
    inferred : Term
        The principal type of the logical form.
    erased : Term
        The erasure of the category.
    substitution : Substitution
        The substitution under which the erased type is an instance of the
        inferred one.
    strict : bool
        Whether the erased type is a proper instance, so the principal type
        is more general than the category commits to.
    tag : str
        The normal-form tag of the item the derivation packs on, or empty
        when the grammar uses none.
    """

    span: Cell
    category: C
    logical_form: Term
    inferred: Term
    erased: Term
    substitution: Substitution
    strict: bool
    tag: str = ""


class CoherenceReport[C](NamedTuple):
    """The instances the square commuted on, over one batch of chart items.

    Attributes
    ----------
    instances : tuple[CoherenceInstance[C], ...]
        One record per derivation of every item, in item then derivation
        order.
    item_count : int
        How many chart items were checked.
    """

    instances: tuple[CoherenceInstance[C], ...]
    item_count: int

    @property
    def strict(self) -> tuple[CoherenceInstance[C], ...]:
        """Return the instances whose erased type is a proper instance."""
        return tuple(instance for instance in self.instances if instance.strict)

    @property
    def strict_items(self) -> frozenset[tuple[Cell, C, str]]:
        """Return the span, category and tag of every item with a strict instance."""
        return frozenset(
            (instance.span, instance.category, instance.tag) for instance in self.strict
        )


class CoherenceFixture[C, S](NamedTuple):
    """Inputs for the commuting square over chart items.

    Attributes
    ----------
    items : tuple[CoherenceItem[C], ...]
        The chart items with the logical forms their derivations read.
    typing : TypingSystem[S]
        The typing system that infers principal types and matches them.
    signature : S
        The lexical environment the typing system checks under.
    rules : Mapping[OpName, TypingRule]
        The typing rules keyed by logical-form operator.
    erase : Callable[[C], Term]
        The erasure from categories to semantic types, into the typing
        system's type store.
    """

    items: tuple[CoherenceItem[C], ...]
    typing: TypingSystem[S]
    signature: S
    rules: Mapping[OpName, TypingRule]
    erase: Callable[[C], Term]


class ClampProjectionFixture[T, R](NamedTuple):
    """Inputs for the clamp-projection laws.

    Attributes
    ----------
    projection : ClampProjection[T, R]
        The projection under test.
    readings : tuple[R, ...]
        Nonempty chart readings of the shape the projection reads.
    """

    projection: ClampProjection[T, R]
    readings: tuple[R, ...]


def _fail(detail: str) -> NoReturn:
    """Raise a typing-law violation.

    Parameters
    ----------
    detail : str
        Failure description.
    """
    raise LawViolationError(detail)


def typing_coherence_is_subsumption[L, C, T](
    fixture: TypingLawFixture[L, C, T],
) -> None:
    """Check that the inferred type subsumes the erased category.

    Parameters
    ----------
    fixture : TypingLawFixture[L, C, T]
        Logical form, category, and typing operations.
    """
    inferred = fixture.typeof(fixture.logical_form)
    erased = fixture.erase(fixture.category)

    if fixture.subsumes(inferred, erased) is None:
        _fail(f"inferred type {inferred!r} does not subsume {erased!r}")


def check_typing_laws[L, C, T](
    fixture: TypingLawFixture[L, C, T],
) -> None:
    """Check the typing coherence law.

    Parameters
    ----------
    fixture : TypingLawFixture[L, C, T]
        Logical form, category, and typing operations.
    """
    typing_coherence_is_subsumption(fixture)


def coherence_square[C, S](
    fixture: CoherenceFixture[C, S],
) -> CoherenceReport[C]:
    """Check that the square commutes on every derivation of every chart item.

    The erased category must instantiate the derivation's principal logical-form
    type. Only principal-type metavariables may bind; a proper instance is
    reported as strict.

    Parameters
    ----------
    fixture : CoherenceFixture[C, S]
        The chart items, the typing system, its environment and rules, and
        the erasure.

    Returns
    -------
    CoherenceReport[C]
        One instance per derivation of every item, with the strict ones
        marked.

    Raises
    ------
    LawViolationError
        If no item was offered, an item reads no logical form, a logical
        form is not well typed, or an erased category is not an instance of
        the principal type. Every failing derivation in the batch is named.
    """
    if not fixture.items:
        _fail("no chart item was offered, so nothing was checked")

    instances: list[CoherenceInstance[C]] = []
    failures: list[str] = []

    for item in fixture.items:
        where = f"item {item.span!r} {item.category!r}"

        if item.tag:
            where = f"{where} {item.tag!r}"

        if not item.logical_forms:
            failures.append(f"{where} reads no logical form")
            continue

        erased = fixture.erase(item.category)

        for logical_form in item.logical_forms:
            result = fixture.typing.check(
                logical_form, fixture.signature, fixture.rules
            )
            scheme = result.scheme

            if scheme is None or not result.is_well_typed:
                failures.append(
                    f"{where} reads {logical_form!r}, which is not well typed: "
                    f"{result.residual!r}"
                )
                continue

            substitution = fixture.typing.subsumes(scheme.body, erased)

            if substitution is None:
                failures.append(
                    f"{where} reads {logical_form!r} of principal type "
                    f"{scheme.body!r}, which {erased!r} does not instantiate"
                )
                continue

            instances.append(
                CoherenceInstance(
                    item.span,
                    item.category,
                    logical_form,
                    scheme.body,
                    erased,
                    substitution,
                    fixture.typing.subsumes(erased, scheme.body) is None,
                    item.tag,
                )
            )

    if failures:
        _fail("\n".join(failures))

    return CoherenceReport(tuple(instances), len(fixture.items))


def a_meaning_is_read_exactly_when_declared[T, R](
    fixture: ClampProjectionFixture[T, R],
) -> None:
    """Check that a reading holds a meaning exactly when the projection says so.

    Parameters
    ----------
    fixture : ClampProjectionFixture[T, R]
        The projection under test and the readings to read.

    Raises
    ------
    LawViolationError
        If no reading was offered, or a reading's meaning is present or absent
        against the projection's declaration.
    """
    if not fixture.readings:
        _fail("no reading was offered, so nothing was checked")

    for reading in fixture.readings:
        held = fixture.projection.meaning(reading) is not None

        if held != fixture.projection.has_meaning:
            _fail(
                f"{fixture.projection!r} declares has_meaning="
                f"{fixture.projection.has_meaning!r} but reads "
                f"{fixture.projection.meaning(reading)!r} off {reading!r}"
            )


def a_projection_reads_the_same_reading_the_same_way[T, R](
    fixture: ClampProjectionFixture[T, R],
) -> None:
    """Check that reading a category or a meaning twice gives the same object.

    Parameters
    ----------
    fixture : ClampProjectionFixture[T, R]
        The projection under test and the readings to read.

    Raises
    ------
    LawViolationError
        If a second read of a reading returns a different object.
    """
    for reading in fixture.readings:
        if fixture.projection.category(reading) is not fixture.projection.category(
            reading
        ):
            _fail(f"{fixture.projection!r} reads two categories off {reading!r}")

        if fixture.projection.meaning(reading) is not fixture.projection.meaning(
            reading
        ):
            _fail(f"{fixture.projection!r} reads two meanings off {reading!r}")


CLAMP_PROJECTION_LAWS: Final = (
    a_meaning_is_read_exactly_when_declared,
    a_projection_reads_the_same_reading_the_same_way,
)
"""Clamp-projection laws in conformance order."""


def check_clamp_projection_laws[T, R](
    fixture: ClampProjectionFixture[T, R],
) -> None:
    """Run every clamp-projection law against one projection.

    Parameters
    ----------
    fixture : ClampProjectionFixture[T, R]
        The projection under test and the readings to read.
    """
    for law in CLAMP_PROJECTION_LAWS:
        law(fixture)
