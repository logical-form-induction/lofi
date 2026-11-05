"""Differential comparison of decoded and inferred span types."""

from typing import TYPE_CHECKING, NamedTuple, Never, Self, final, override

from lofi._checks import ensure_name, ensure_non_negative
from lofi._types import ItemId
from lofi.typecheck._evaluate import TypeAgreement, _agrees

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi.deduce import Cell
    from lofi.terms import OpName, Term
    from lofi.typecheck._ports import TypingSystem
    from lofi.typecheck._values import Constraint, TypingRule

__all__ = [
    "DifferentialItem",
    "DifferentialReport",
    "SpanAddress",
    "differential_report",
]

type SpanAddress = tuple[ItemId, Cell]
"""Dataset item and half-open token span."""


class DifferentialItem(NamedTuple):
    """Decoded and inferred types at one span.

    Attributes
    ----------
    address : SpanAddress
        Dataset item and span.
    predicted : Term
        Neural argmax type.
    inferred : Term
        Native inferred type.
    residual : tuple[Constraint, ...]
        Unsatisfied constraints from native inference.
    agrees : bool
        Whether the decoded and inferred types stand in the declared relation.
    """

    address: SpanAddress
    predicted: Term
    inferred: Term
    residual: tuple[Constraint, ...]
    agrees: bool


@final
class DifferentialReport:
    """Frozen span comparisons under one versioned theory.

    Parameters
    ----------
    theory_version : str
        Theory format version.
    theory_digest : str
        Digest of the canonical theory specification.
    agreement : TypeAgreement
        Relation used to compare decoded and inferred types.
    items : Sequence[DifferentialItem]
        Span comparisons in stable address order.

    Raises
    ------
    TypeError
        If a theory identifier is not a string or the agreement is not declared.
    ValueError
        If a theory identifier is empty.
    """

    __slots__ = ("_agreement", "_items", "_theory_digest", "_theory_version")

    def __init__(
        self,
        theory_version: str,
        theory_digest: str,
        agreement: TypeAgreement,
        items: Sequence[DifferentialItem],
    ) -> None:
        object.__setattr__(
            self,
            "_theory_version",
            ensure_name("a theory version", theory_version),
        )
        object.__setattr__(
            self,
            "_theory_digest",
            ensure_name("a theory digest", theory_digest),
        )
        if not isinstance(agreement, TypeAgreement):
            msg = f"a differential type agreement is not one: {agreement!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_agreement", agreement)
        object.__setattr__(self, "_items", tuple(items))

    @property
    def agreement(self) -> TypeAgreement:
        """Return the declared semantic-type agreement relation."""
        return self._agreement

    @property
    def theory_version(self) -> str:
        """Theory format version."""
        return self._theory_version

    @property
    def theory_digest(self) -> str:
        """Digest of the canonical theory specification."""
        return self._theory_digest

    @property
    def items(self) -> tuple[DifferentialItem, ...]:
        """Span comparisons in stable address order."""
        return self._items

    @property
    def disagreements(self) -> tuple[DifferentialItem, ...]:
        """Well-typed spans whose decoded and inferred types differ."""
        return tuple(
            item for item in self._items if not item.residual and not item.agrees
        )

    @property
    def residuals(self) -> tuple[DifferentialItem, ...]:
        """Spans whose native inference left unsatisfied constraints."""
        return tuple(item for item in self._items if item.residual)

    @property
    def agrees(self) -> bool:
        """Whether every span agrees without a residual."""
        return all(item.agrees for item in self._items)

    def compare(self, other: DifferentialReport, /) -> bool:
        """Compare outcomes produced under the same versioned theory.

        Parameters
        ----------
        other : DifferentialReport
            Report to compare.

        Returns
        -------
        bool
            Whether every span outcome is identical.

        Raises
        ------
        ValueError
            If the reports name different theory versions, digests, or
            agreement relations.
        """
        if (
            self._theory_version != other._theory_version
            or self._theory_digest != other._theory_digest
            or self._agreement is not other._agreement
        ):
            msg = (
                "differential reports use different theories: "
                f"{self._theory_version}@{self._theory_digest} and "
                f"{other._theory_version}@{other._theory_digest}, under "
                f"{self._agreement.value!r} and {other._agreement.value!r}"
            )
            raise ValueError(msg)

        return self._items == other._items

    def __copy__(self) -> Self:
        """Preserve identity when copied."""
        return self

    def __deepcopy__(self, memo: dict[int, Self]) -> Self:
        """Preserve identity when deep-copied."""
        del memo
        return self

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a differential report is immutable"
        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a differential report is immutable"
        raise AttributeError(msg)

    @override
    def __repr__(self) -> str:
        """Return the theory identity and outcome counts."""
        return (
            f"DifferentialReport({self._theory_version!r}, "
            f"{self._theory_digest!r}, agreement={self._agreement.value!r}, "
            f"items={len(self._items)}, "
            f"disagreements={len(self.disagreements)}, "
            f"residuals={len(self.residuals)})"
        )


def _address(address: SpanAddress) -> SpanAddress:
    """Validate and normalize a span address.

    Parameters
    ----------
    address : SpanAddress
        Item and half-open span to validate.

    Raises
    ------
    TypeError
        If the item id is not a string or a span endpoint is not an integer.
    ValueError
        If the item id is empty or the span is negative, empty, or reversed.
    """
    item_id, span = address
    start, end = span
    held_id = ItemId(ensure_name("a differential item id", item_id))
    held_start = ensure_non_negative("a differential span start", start)
    held_end = ensure_non_negative("a differential span end", end)

    if held_end <= held_start:
        msg = f"a differential span is empty or reversed: [{start}, {end})"
        raise ValueError(msg)

    return held_id, (held_start, held_end)


def differential_report[S](
    chart: Mapping[SpanAddress, Term],
    decoded: Mapping[SpanAddress, Term],
    typing: TypingSystem[S],
    signature: S,
    rules: Mapping[OpName, TypingRule],
    /,
    *,
    agreement: TypeAgreement,
    theory_version: str,
    theory_digest: str,
) -> DifferentialReport:
    """Compare decoded types with native inference at each chart span.

    Parameters
    ----------
    chart : Mapping[SpanAddress, Term]
        Selected logical form at each chart span.
    decoded : Mapping[SpanAddress, Term]
        Neural argmax type at each chart span.
    typing : TypingSystem[S]
        Native type checker.
    signature : S
        Symbol environment used by the checker.
    rules : Mapping[OpName, TypingRule]
        Typing rules keyed by logical-form operator.
    agreement : TypeAgreement
        Relation used to compare decoded and inferred types.
    theory_version : str
        Theory format version.
    theory_digest : str
        Digest of the canonical theory specification.

    Returns
    -------
    DifferentialReport
        Frozen per-span outcomes under the named theory.

    Raises
    ------
    TypeError
        If an item id or theory identifier is not a string, a span endpoint is
        not an integer, or the agreement is not declared.
    ValueError
        If no span is supplied, an address or theory identifier is invalid, or
        the chart and decoder cover different spans.
    """
    checked_chart = {_address(address): term for address, term in chart.items()}
    checked_decoded = {_address(address): term for address, term in decoded.items()}

    if not checked_chart:
        msg = "a differential report needs at least one chart span"
        raise ValueError(msg)

    if checked_chart.keys() != checked_decoded.keys():
        missing = tuple(sorted(checked_chart.keys() - checked_decoded.keys()))
        extra = tuple(sorted(checked_decoded.keys() - checked_chart.keys()))
        msg = (
            "decoded spans differ from chart spans; "
            f"missing={missing!r}, extra={extra!r}"
        )
        raise ValueError(msg)

    items = []

    for address in sorted(checked_chart):
        result = typing.check(checked_chart[address], signature, rules)
        items.append(
            DifferentialItem(
                address,
                checked_decoded[address],
                result.assignment[()],
                result.residual,
                not result.residual
                and _agrees(
                    typing,
                    checked_decoded[address],
                    result.assignment[()],
                    agreement,
                ),
            )
        )

    return DifferentialReport(theory_version, theory_digest, agreement, items)
