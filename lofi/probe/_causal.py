"""Causal-abstraction metrics with aligned interventions."""

from enum import Enum, auto
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Protocol, override, runtime_checkable

from lofi._checks import (
    ensure_finite,
    ensure_name,
    ensure_non_negative,
    ensure_positive,
)
from lofi._frozen import FrozenValue
from lofi.probe._provenance import Provenance
from lofi.stats import NullProvenance

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.deduce import Cell

__all__ = [
    "PIECE_POLICIES",
    "SPECIAL_TOKEN_POLICIES",
    "AlignedInput",
    "AlignmentPolicy",
    "CausalPair",
    "CellAssignment",
    "CombinatorAt",
    "CounterfactualPair",
    "HighLevelVariable",
    "IIAResult",
    "JointCausalPair",
    "JointCounterfactualPair",
    "NamedVariable",
    "PiecePolicy",
    "RowAligner",
    "RowEvidence",
    "Side",
    "SpecialTokenPolicy",
    "SplitAt",
    "TokenizerAlignment",
    "TypeAt",
    "strictly_dominates",
    "tokenizer_alignment",
    "variable_label",
]


def _a_cell(field: str, span: Cell) -> Cell:
    """Return a span as a checked half-open cell.

    Parameters
    ----------
    field : str
        The name to report in a message.
    span : Cell
        The span to check.

    Returns
    -------
    Cell
        The span, as a start and an end.

    Raises
    ------
    TypeError
        If the span is not a pair, or a bound is not an integer.
    ValueError
        If a bound is negative, or the end does not lie above the start.
    """
    try:
        given_start, given_end = span
    except (TypeError, ValueError) as unpacking:
        msg = f"{field} is a start and an end, not {span!r}"
        raise TypeError(msg) from unpacking

    start = ensure_non_negative(f"the start of {field}", given_start)
    end = ensure_non_negative(f"the end of {field}", given_end)

    if end <= start:
        msg = f"{field} covers nothing: [{start}, {end})"
        raise ValueError(msg)

    return (start, end)


class TypeAt:
    """Identifies the category assigned to a chart cell.

    Parameters
    ----------
    cell : Cell
        The half-open span the category is assigned over.

    Raises
    ------
    TypeError
        If the cell is not a pair of integers.
    ValueError
        If the cell covers nothing or begins below zero.
    """

    __slots__ = ("_cell",)

    def __init__(self, cell: Cell) -> None:
        self._cell = _a_cell("a variable's cell", cell)

    @property
    def kind(self) -> Literal["type"]:
        """Return which variant of high-level variable this is."""
        return "type"

    @property
    def cell(self) -> Cell:
        """Return the half-open span the variable is about."""
        return self._cell

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value names the category at the same cell.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not TypeAt:
            return NotImplemented

        return self._cell == other._cell

    @override
    def __hash__(self) -> int:
        """Return a hash over the variant and the cell."""
        return hash((TypeAt, self._cell))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this variable."""
        return f"TypeAt({self._cell!r})"


class SplitAt:
    """Identifies the split assigned to a chart cell.

    Parameters
    ----------
    cell : Cell
        The half-open span that is divided.

    Raises
    ------
    TypeError
        If the cell is not a pair of integers.
    ValueError
        If the cell covers nothing or begins below zero.
    """

    __slots__ = ("_cell",)

    def __init__(self, cell: Cell) -> None:
        self._cell = _a_cell("a variable's cell", cell)

    @property
    def kind(self) -> Literal["split"]:
        """Return which variant of high-level variable this is."""
        return "split"

    @property
    def cell(self) -> Cell:
        """Return the half-open span the variable is about."""
        return self._cell

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value names the split of the same cell.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not SplitAt:
            return NotImplemented

        return self._cell == other._cell

    @override
    def __hash__(self) -> int:
        """Return a hash over the variant and the cell."""
        return hash((SplitAt, self._cell))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this variable."""
        return f"SplitAt({self._cell!r})"


class CombinatorAt:
    """Identifies the combinator assigned to a chart cell.

    Parameters
    ----------
    cell : Cell
        The half-open span that was joined.
    split : int
        Where inside it the two halves meet.

    Raises
    ------
    TypeError
        If the cell is not a pair of integers, or the split is not an integer.
    ValueError
        If the cell covers nothing, or the split does not lie strictly inside it.
    """

    __slots__ = ("_cell", "_split")

    def __init__(self, cell: Cell, split: int) -> None:
        checked = _a_cell("a variable's cell", cell)
        at = ensure_non_negative("a variable's split", split)

        if not checked[0] < at < checked[1]:
            msg = f"a split at {at} does not lie inside [{checked[0]}, {checked[1]})"
            raise ValueError(msg)

        self._cell = checked
        self._split = at

    @property
    def kind(self) -> Literal["combinator"]:
        """Return which variant of high-level variable this is."""
        return "combinator"

    @property
    def cell(self) -> Cell:
        """Return the half-open span the variable is about."""
        return self._cell

    @property
    def split(self) -> int:
        """Return where inside the cell the two halves meet."""
        return self._split

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value names the same rule choice.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not CombinatorAt:
            return NotImplemented

        return self._cell == other._cell and self._split == other._split

    @override
    def __hash__(self) -> int:
        """Return a hash over the variant, the cell and the split."""
        return hash((CombinatorAt, self._cell, self._split))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this variable."""
        return f"CombinatorAt({self._cell!r}, {self._split!r})"


class NamedVariable(FrozenValue):
    """Identify a model-facing decision independently of chart geometry.

    Parameters
    ----------
    namespace : str
        The system or inventory within which the variable's name is interpreted.
    name : str
        The variable's name within that namespace.

    Raises
    ------
    TypeError
        If either field is not a string.
    ValueError
        If either field is empty.
    """

    __slots__ = ("_name", "_namespace")

    _noun = "named high-level variable"

    def __init__(self, namespace: str, name: str) -> None:
        object.__setattr__(
            self,
            "_namespace",
            ensure_name("a named variable's namespace", namespace),
        )
        object.__setattr__(
            self,
            "_name",
            ensure_name("a named variable's name", name),
        )

    @property
    def kind(self) -> Literal["named"]:
        """Return which variant of high-level variable this is."""
        return "named"

    @property
    def namespace(self) -> str:
        """Return the inventory in which the variable is named."""
        return self._namespace

    @property
    def name(self) -> str:
        """Return the variable's name within its namespace."""
        return self._name

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value has the same namespace and name."""
        if type(other) is not NamedVariable:
            return NotImplemented

        return self._namespace == other._namespace and self._name == other._name

    @override
    def __hash__(self) -> int:
        """Return a hash over the namespace and name."""
        return hash((NamedVariable, self._namespace, self._name))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this variable."""
        return f"NamedVariable({self._namespace!r}, {self._name!r})"


type HighLevelVariable = TypeAt | SplitAt | CombinatorAt | NamedVariable
"""Defines one decision made by a derivation or an external model."""

type CellAssignment = Mapping[HighLevelVariable, str | int]
"""Specifies what one derivation decided, variable by variable."""


def variable_label(variable: HighLevelVariable, /) -> str:
    """Return a stable report label for one declared high-level variable."""
    if isinstance(variable, NamedVariable):
        return f"named-{variable.namespace}-{variable.name}"

    start, end = variable.cell
    suffix = f"-{variable.split}" if isinstance(variable, CombinatorAt) else ""

    return f"{variable.kind}-{start}-{end}{suffix}"


@runtime_checkable
class CausalPair(Protocol):
    """A pair that changes exactly one high-level decision."""

    @property
    def base(self) -> CellAssignment:
        """Return what the base derivation decided."""

    @property
    def variant(self) -> CellAssignment:
        """Return what the counterfactual derivation decided."""

    @property
    def varied(self) -> HighLevelVariable:
        """Return the one decision on which the two derivations differ."""

    @property
    def span(self) -> Cell:
        """Return the region the intervention targets."""

    @property
    def provenance(self) -> Provenance:
        """Return what generated the pair."""


def strictly_dominates(outer: HighLevelVariable, inner: HighLevelVariable, /) -> bool:
    """Return whether the first variable's cell strictly contains the second's."""
    if isinstance(outer, NamedVariable) or isinstance(inner, NamedVariable):
        msg = "NamedVariable does not participate in structural dominance"
        raise TypeError(msg)
    if type(outer) not in {TypeAt, SplitAt, CombinatorAt}:
        msg = f"a dominance relation starts at a high-level variable, not {outer!r}"
        raise TypeError(msg)
    if type(inner) not in {TypeAt, SplitAt, CombinatorAt}:
        msg = f"a dominance relation ends at a high-level variable, not {inner!r}"
        raise TypeError(msg)

    outer_start, outer_end = outer.cell
    inner_start, inner_end = inner.cell
    return (
        outer.cell != inner.cell
        and outer_start <= inner_start
        and inner_end <= outer_end
    )


@runtime_checkable
class JointCausalPair(Protocol):
    """A structural pair that changes two dominance-related decisions."""

    @property
    def base(self) -> CellAssignment:
        """Return what the base derivation decided."""

    @property
    def variant(self) -> CellAssignment:
        """Return what the counterfactual derivation decided."""

    @property
    def variables(self) -> tuple[HighLevelVariable, HighLevelVariable]:
        """Return the dominant and dominated changed variables."""

    @property
    def span(self) -> Cell:
        """Return the region containing both intervened cells."""

    @property
    def provenance(self) -> Provenance:
        """Return what generated the pair."""


def _an_assignment(field: str, given: CellAssignment) -> CellAssignment:
    """Return one derivation's decisions as a checked read-only mapping.

    Parameters
    ----------
    field : str
        The name to report in a message.
    given : CellAssignment
        The decisions to check.

    Returns
    -------
    CellAssignment
        A copy, behind a read-only view.

    Raises
    ------
    TypeError
        If a key is not a high-level variable, or a value is neither a name nor a count.
    ValueError
        If a name is empty or a count is negative.
    """
    checked: dict[HighLevelVariable, str | int] = {}

    for variable, decided in given.items():
        if type(variable) not in {TypeAt, SplitAt, CombinatorAt, NamedVariable}:
            msg = f"{field} decides high-level variables, not {variable!r}"
            raise TypeError(msg)

        if isinstance(decided, bool) or not isinstance(decided, str | int):
            msg = (
                f"{field} decides {variable!r} as {decided!r}, which is "
                f"neither a name nor a position"
            )
            raise TypeError(msg)

        checked[variable] = (
            ensure_name(f"the decision at {variable!r}", decided)
            if isinstance(decided, str)
            else ensure_non_negative(f"the decision at {variable!r}", decided)
        )

    return MappingProxyType(checked)


class CounterfactualPair:
    """Pairs derivations that differ at one decision.

    Parameters
    ----------
    base : CellAssignment
        What the base derivation decided, variable by variable.
    variant : CellAssignment
        What the counterfactual derivation decided, over the same variables.
    varied : HighLevelVariable
        The one variable the two differ at.
    span : Cell
        The region the intervention targets.
    provenance : Provenance
        What generated the pair, over what data, under what seed.

    Raises
    ------
    TypeError
        If an assignment is not a mapping of variables to names or counts, if the varied
        variable is not one, or if the provenance is not one.
    ValueError
        If assignments cover different variables, differ on zero, multiple, or the
        wrong variable, or place a structural variable's cell outside the span.
    """

    __slots__ = ("_base", "_provenance", "_span", "_variant", "_varied")

    def __init__(
        self,
        base: CellAssignment,
        variant: CellAssignment,
        varied: HighLevelVariable,
        span: Cell,
        provenance: Provenance,
    ) -> None:
        if type(varied) not in {TypeAt, SplitAt, CombinatorAt, NamedVariable}:
            msg = f"a pair varies a high-level variable, not {varied!r}"
            raise TypeError(msg)

        if type(provenance) is not Provenance:
            msg = f"a pair requires a Provenance, not {provenance!r}"
            raise TypeError(msg)

        held_base = _an_assignment("a base", base)
        held_variant = _an_assignment("a variant", variant)
        region = _a_cell("a pair's span", span)

        if set(held_base) != set(held_variant):
            msg = (
                "a base and its variant decide different variables, so nothing "
                "about them is a counterfactual"
            )
            raise ValueError(msg)

        differing = {
            variable
            for variable, decided in held_base.items()
            if held_variant[variable] != decided
        }

        if not differing:
            msg = "a base and its variant decide everything the same way"
            raise ValueError(msg)

        if len(differing) > 1:
            msg = (
                f"a base and its variant differ at {len(differing)} variables, "
                f"so no outcome can be attributed to any one of them"
            )
            raise ValueError(msg)

        if differing != {varied}:
            other = next(iter(differing))
            msg = f"a pair claims to vary {varied!r} but varies {other!r}"
            raise ValueError(msg)

        if not isinstance(varied, NamedVariable) and not (
            region[0] <= varied.cell[0] and varied.cell[1] <= region[1]
        ):
            msg = f"the varied cell {varied.cell!r} falls outside the span {region!r}"
            raise ValueError(msg)

        self._base = held_base
        self._variant = held_variant
        self._varied = varied
        self._span = region
        self._provenance = provenance

    @property
    def base(self) -> CellAssignment:
        """Return what the base derivation decided."""
        return self._base

    @property
    def variant(self) -> CellAssignment:
        """Return what the counterfactual derivation decided."""
        return self._variant

    @property
    def varied(self) -> HighLevelVariable:
        """Return the variable on which the assignments differ."""
        return self._varied

    @property
    def span(self) -> Cell:
        """Return the region the intervention targets."""
        return self._span

    @property
    def provenance(self) -> Provenance:
        """Return what generated the pair, over what data, under what seed."""
        return self._provenance

    @override
    def __repr__(self) -> str:
        """Return what the pair varies, where, and what generated it."""
        return (
            f"CounterfactualPair(varied={self._varied!r}, span={self._span!r}, "
            f"provenance={self._provenance!r})"
        )


class JointCounterfactualPair:
    """Pairs derivations differing at two strictly dominance-related decisions."""

    __slots__ = ("_base", "_provenance", "_span", "_variables", "_variant")

    def __init__(
        self,
        base: CellAssignment,
        variant: CellAssignment,
        variables: tuple[HighLevelVariable, HighLevelVariable],
        span: Cell,
        provenance: Provenance,
    ) -> None:
        if type(variables) is not tuple or len(variables) != 2:
            msg = f"a joint pair names exactly two variables, not {variables!r}"
            raise TypeError(msg)
        first, second = variables
        if isinstance(first, NamedVariable) or isinstance(second, NamedVariable):
            msg = "a joint pair does not admit NamedVariable"
            raise TypeError(msg)
        if type(first) not in {TypeAt, SplitAt, CombinatorAt} or type(second) not in {
            TypeAt,
            SplitAt,
            CombinatorAt,
        }:
            msg = f"a joint pair varies high-level variables, not {variables!r}"
            raise TypeError(msg)
        if first == second:
            msg = "a joint pair changes two distinct variables"
            raise ValueError(msg)
        if type(provenance) is not Provenance:
            msg = f"a joint pair requires a Provenance, not {provenance!r}"
            raise TypeError(msg)

        if strictly_dominates(first, second):
            ordered = (first, second)
        elif strictly_dominates(second, first):
            ordered = (second, first)
        else:
            msg = (
                f"joint variables {first!r} and {second!r} are not related by "
                "strict span dominance"
            )
            raise ValueError(msg)

        held_base = _an_assignment("a joint base", base)
        held_variant = _an_assignment("a joint variant", variant)
        region = _a_cell("a joint pair's span", span)

        if set(held_base) != set(held_variant):
            msg = "a joint base and variant decide different variables"
            raise ValueError(msg)

        differing = tuple(
            variable
            for variable, decided in held_base.items()
            if held_variant[variable] != decided
        )
        if set(differing) != set(ordered) or len(differing) != 2:
            msg = (
                "a joint pair's assignments differ at exactly its two declared "
                f"variables, not {differing!r}"
            )
            raise ValueError(msg)
        if not (region[0] <= ordered[0].cell[0] and ordered[0].cell[1] <= region[1]):
            msg = (
                f"the dominant cell {ordered[0].cell!r} falls outside the span "
                f"{region!r}"
            )
            raise ValueError(msg)

        self._base = held_base
        self._variant = held_variant
        self._variables = ordered
        self._span = region
        self._provenance = provenance

    @property
    def base(self) -> CellAssignment:
        """Return what the base derivation decided."""
        return self._base

    @property
    def variant(self) -> CellAssignment:
        """Return what the joint counterfactual derivation decided."""
        return self._variant

    @property
    def variables(self) -> tuple[HighLevelVariable, HighLevelVariable]:
        """Return the dominant and dominated changed variables."""
        return self._variables

    @property
    def span(self) -> Cell:
        """Return the region containing both intervened cells."""
        return self._span

    @property
    def provenance(self) -> Provenance:
        """Return what generated the pair."""
        return self._provenance

    @override
    def __repr__(self) -> str:
        """Return the changed variables, span, and provenance."""
        return (
            f"JointCounterfactualPair(variables={self._variables!r}, "
            f"span={self._span!r}, provenance={self._provenance!r})"
        )


class Side(Enum):
    """Names one of the two derivations a counterfactual pair holds."""

    #: The derivation whose representation is intervened on.
    BASE = auto()

    #: The derivation the intervened value is taken from.
    VARIANT = auto()


type PiecePolicy = Literal["first", "last", "all"]
"""Specifies which pieces of a tokenized word stand for the word."""

PIECE_POLICIES: Final[tuple[PiecePolicy, ...]] = ("first", "last", "all")
"""Piece policies in report order."""


type SpecialTokenPolicy = Literal["exclude", "include"]
"""Specifies what a span does with the tokens no word of the input produced."""

SPECIAL_TOKEN_POLICIES: Final[tuple[SpecialTokenPolicy, ...]] = ("exclude", "include")
"""Special-token policies in report order."""


class AlignmentPolicy(FrozenValue):
    """Maps derivation spans to model tokens.

    Parameters
    ----------
    pieces : PiecePolicy
        Which pieces of a tokenized word stand for the word.
    specials : SpecialTokenPolicy
        What a span does with tokens no word produced.
    contiguous : bool
        Whether a span is required to land on a contiguous run of tokens.

    Raises
    ------
    TypeError
        If the contiguity flag is not a boolean.
    ValueError
        If the piece policy or the special-token policy is not a declared one.
    """

    __slots__ = ("_contiguous", "_pieces", "_specials")

    _noun = "alignment policy"

    def __init__(
        self,
        pieces: PiecePolicy,
        specials: SpecialTokenPolicy,
        *,
        contiguous: bool,
    ) -> None:
        if pieces not in PIECE_POLICIES:
            msg = f"no piece policy is called {pieces!r}, only {PIECE_POLICIES}"
            raise ValueError(msg)

        if specials not in SPECIAL_TOKEN_POLICIES:
            msg = (
                f"no special-token policy is called {specials!r}, "
                f"only {SPECIAL_TOKEN_POLICIES}"
            )
            raise ValueError(msg)

        if not isinstance(contiguous, bool):
            msg = f"a policy's contiguity is not a boolean: {contiguous!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_pieces", pieces)
        object.__setattr__(self, "_specials", specials)
        object.__setattr__(self, "_contiguous", contiguous)

    @property
    def pieces(self) -> PiecePolicy:
        """Return which pieces of a tokenized word stand for the word."""
        return self._pieces

    @property
    def specials(self) -> SpecialTokenPolicy:
        """Return what a span does with tokens no word produced."""
        return self._specials

    @property
    def contiguous(self) -> bool:
        """Return whether a span must land on a contiguous run of tokens."""
        return self._contiguous

    def _identity(self) -> tuple[PiecePolicy, SpecialTokenPolicy, bool]:
        """Return the three rules a policy is equal on."""
        return (self._pieces, self._specials, self._contiguous)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value states the same three rules.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not AlignmentPolicy:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the three rules the policy is equal on."""
        return hash((AlignmentPolicy, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this policy."""
        return (
            f"AlignmentPolicy({self._pieces!r}, {self._specials!r}, "
            f"contiguous={self._contiguous!r})"
        )


class TokenizerAlignment(FrozenValue):
    """An exact alignment from source tokens to one tokenizer's live rows.

    Parameters
    ----------
    tokenizer : str
        Stable identity of the tokenizer that produced the rows.
    token_rows : tuple[tuple[int, ...], ...]
        Ordered model rows produced by each source token, in source order.
    special_rows : tuple[int, ...]
        Ordered live rows produced by no source token.
    live_rows : int
        Exclusive upper bound of live model rows.

    Raises
    ------
    TypeError
        If an identity or row has the wrong runtime type.
    ValueError
        If the identity is empty, no source token is represented, a source token
        has no row, rows repeat, overlap, occur out of order or lie outside the
        live extent, or the special rows do not complete the live extent.
    """

    __slots__ = ("_live_rows", "_special_rows", "_token_rows", "_tokenizer")

    _noun = "tokenizer alignment"

    def __init__(
        self,
        tokenizer: str,
        token_rows: tuple[tuple[int, ...], ...],
        special_rows: tuple[int, ...],
        live_rows: int,
    ) -> None:
        identity = ensure_name("a tokenizer alignment's identity", tokenizer)
        extent = ensure_positive("a tokenizer alignment's live-row extent", live_rows)

        if not token_rows:
            msg = "a tokenizer alignment maps at least one source token"
            raise ValueError(msg)

        checked_tokens: list[tuple[int, ...]] = []
        mapped: set[int] = set()
        previous = -1

        for source, rows in enumerate(token_rows):
            if not rows:
                msg = f"source token {source} has zero model rows under {identity!r}"
                raise ValueError(msg)

            checked_rows: list[int] = []

            for row in rows:
                checked = ensure_non_negative("a tokenizer alignment row", row)

                if checked >= extent:
                    msg = (
                        f"model row {checked} lies outside the live-row extent "
                        f"[0, {extent}) under {identity!r}"
                    )
                    raise ValueError(msg)

                if checked in mapped:
                    msg = f"model row {checked} repeats or overlaps under {identity!r}"
                    raise ValueError(msg)

                if checked <= previous:
                    msg = (
                        f"model row {checked} is unordered after row {previous} "
                        f"under {identity!r}"
                    )
                    raise ValueError(msg)

                checked_rows.append(checked)
                mapped.add(checked)
                previous = checked

            checked_tokens.append(tuple(checked_rows))

        checked_specials: list[int] = []
        previous_special = -1

        for row in special_rows:
            checked = ensure_non_negative("a special model row", row)

            if checked >= extent:
                msg = (
                    f"special row {checked} lies outside the live-row extent "
                    f"[0, {extent}) under {identity!r}"
                )
                raise ValueError(msg)

            if checked <= previous_special:
                msg = f"special rows are repeated or unordered under {identity!r}"
                raise ValueError(msg)

            if checked in mapped:
                msg = (
                    f"special row {checked} also belongs to a source token "
                    f"under {identity!r}"
                )
                raise ValueError(msg)

            checked_specials.append(checked)
            previous_special = checked

        covered = mapped | set(checked_specials)
        expected = set(range(extent))

        if covered != expected:
            missing = tuple(sorted(expected - covered))
            msg = (
                f"special rows under {identity!r} do not complete the live-row "
                f"extent; missing {missing!r}"
            )
            raise ValueError(msg)

        object.__setattr__(self, "_tokenizer", identity)
        object.__setattr__(self, "_token_rows", tuple(checked_tokens))
        object.__setattr__(self, "_special_rows", tuple(checked_specials))
        object.__setattr__(self, "_live_rows", extent)

    @property
    def tokenizer(self) -> str:
        """Return the tokenizer's stable identity."""
        return self._tokenizer

    @property
    def token_rows(self) -> tuple[tuple[int, ...], ...]:
        """Return ordered model rows for each source token."""
        return self._token_rows

    @property
    def special_rows(self) -> tuple[int, ...]:
        """Return ordered live rows produced by no source token."""
        return self._special_rows

    @property
    def live_rows(self) -> int:
        """Return the exclusive upper bound of live model rows."""
        return self._live_rows

    def model_rows(self, span: Cell, policy: AlignmentPolicy, /) -> tuple[int, ...]:
        """Return the model rows selected for one source-token span.

        Parameters
        ----------
        span : Cell
            Half-open source-token span to map.
        policy : AlignmentPolicy
            Rules selecting pieces and special rows.

        Raises
        ------
        TypeError
            If the policy is not an alignment policy.
        ValueError
            If the source span runs past this alignment.
        """
        if type(policy) is not AlignmentPolicy:
            msg = f"model-row selection requires an AlignmentPolicy, not {policy!r}"
            raise TypeError(msg)

        start, end = _a_cell("a source span", span)

        if end > len(self._token_rows):
            msg = (
                f"source span {(start, end)!r} runs past the "
                f"{len(self._token_rows)} source tokens under {self._tokenizer!r}"
            )
            raise ValueError(msg)

        selected: list[int] = []

        for rows in self._token_rows[start:end]:
            if policy.pieces == "first":
                selected.append(rows[0])
            elif policy.pieces == "last":
                selected.append(rows[-1])
            else:
                selected.extend(rows)

        if policy.specials == "include":
            selected.extend(self._special_rows)

        return tuple(sorted(selected))

    def _identity(
        self,
    ) -> tuple[str, tuple[tuple[int, ...], ...], tuple[int, ...], int]:
        """Return the four fields an alignment is equal on."""
        return (
            self._tokenizer,
            self._token_rows,
            self._special_rows,
            self._live_rows,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same exact alignment."""
        if type(other) is not TokenizerAlignment:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the exact alignment."""
        return hash((TokenizerAlignment, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this alignment."""
        return (
            f"TokenizerAlignment({self._tokenizer!r}, {self._token_rows!r}, "
            f"{self._special_rows!r}, {self._live_rows!r})"
        )


class RowEvidence(Protocol):
    """The live model rows one tokenizer produced for one source-token sequence."""

    @property
    def word_rows(self) -> tuple[tuple[int, ...], ...]:
        """Return the ordered model rows of each source token, in source order."""
        ...

    @property
    def special_rows(self) -> tuple[int, ...]:
        """Return the ordered live rows no source token produced."""
        ...

    @property
    def live_rows(self) -> int:
        """Return how many live model rows the sequence takes."""
        ...


@runtime_checkable
class RowAligner(Protocol):
    """Reads the model rows one named tokenizer assigns to source tokens."""

    @property
    def identity(self) -> str:
        """Return the stable identity of the tokenizer."""
        ...

    def align(self, words: tuple[str, ...], /) -> RowEvidence:
        """Return the rows the tokenizer assigns to each of the words."""
        ...


def tokenizer_alignment(
    aligner: RowAligner, tokens: tuple[str, ...], /
) -> TokenizerAlignment:
    """Return the exact alignment one tokenizer gives a source-token sequence.

    Parameters
    ----------
    aligner : RowAligner
        The tokenizer whose rows are read, under its identity.
    tokens : tuple[str, ...]
        The source tokens, in their declared order.

    Returns
    -------
    TokenizerAlignment
        Every row of every source token and every special row, under the
        aligner's identity.

    Raises
    ------
    ValueError
        If the evidence covers another number of tokens, or is not an exact
        alignment of the live rows.
    """
    held = tuple(tokens)
    evidence = aligner.align(held)

    if len(evidence.word_rows) != len(held):
        msg = (
            f"tokenizer {aligner.identity!r} returned rows for "
            f"{len(evidence.word_rows)} words, not the {len(held)} source tokens"
        )
        raise ValueError(msg)

    return TokenizerAlignment(
        aligner.identity,
        evidence.word_rows,
        evidence.special_rows,
        evidence.live_rows,
    )


class AlignedInput(FrozenValue):
    """One source-token sequence with named tokenizer alignments.

    Parameters
    ----------
    tokens : tuple[str, ...]
        Source tokens in their declared order.
    alignments : tuple[TokenizerAlignment, ...]
        Tokenizer evidence in manifest order.

    Raises
    ------
    TypeError
        If an alignment is not a tokenizer alignment or a token is not a string.
    ValueError
        If the input or alignment collection is empty, a token is empty, tokenizer
        identities repeat, or an alignment omits or adds source-token mappings.
    """

    __slots__ = ("_alignments", "_tokens")

    _noun = "aligned input"

    def __init__(
        self,
        tokens: tuple[str, ...],
        alignments: tuple[TokenizerAlignment, ...],
    ) -> None:
        if not tokens:
            msg = "an aligned input has at least one source token"
            raise ValueError(msg)

        held_tokens = tuple(
            ensure_name(f"source token {position}", token)
            for position, token in enumerate(tokens)
        )

        if not alignments:
            msg = "an aligned input has at least one tokenizer alignment"
            raise ValueError(msg)

        held_alignments: list[TokenizerAlignment] = []
        identities: set[str] = set()

        for alignment in alignments:
            if type(alignment) is not TokenizerAlignment:
                msg = (
                    "an aligned input requires TokenizerAlignment values, "
                    f"not {alignment!r}"
                )
                raise TypeError(msg)

            if alignment.tokenizer in identities:
                msg = (
                    f"tokenizer identity {alignment.tokenizer!r} appears more than once"
                )
                raise ValueError(msg)

            if len(alignment.token_rows) != len(held_tokens):
                msg = (
                    f"tokenizer {alignment.tokenizer!r} maps "
                    f"{len(alignment.token_rows)} source tokens, not the "
                    f"{len(held_tokens)} tokens in the input"
                )
                raise ValueError(msg)

            held_alignments.append(alignment)
            identities.add(alignment.tokenizer)

        object.__setattr__(self, "_tokens", held_tokens)
        object.__setattr__(self, "_alignments", tuple(held_alignments))

    @property
    def tokens(self) -> tuple[str, ...]:
        """Return source tokens in their declared order."""
        return self._tokens

    @property
    def alignments(self) -> tuple[TokenizerAlignment, ...]:
        """Return tokenizer evidence in manifest order."""
        return self._alignments

    @property
    def tokenizers(self) -> tuple[str, ...]:
        """Return tokenizer identities in manifest order."""
        return tuple(alignment.tokenizer for alignment in self._alignments)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value has the same tokens and alignments."""
        if type(other) is not AlignedInput:
            return NotImplemented

        return self._tokens == other._tokens and self._alignments == other._alignments

    @override
    def __hash__(self) -> int:
        """Return a hash over the tokens and alignments."""
        return hash((AlignedInput, self._tokens, self._alignments))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this aligned input."""
        return f"AlignedInput({self._tokens!r}, {self._alignments!r})"


class IIAResult:
    """Stores intervention accuracy and its test context.

    Parameters
    ----------
    accuracy : float
        The proportion of interventions whose output matched what the high-level
        variable predicts, from zero to one.
    variable : HighLevelVariable
        The exact high-level variable that was intervened on.
    rank : int
        The discrete width the featurizer used. Zero reports an unaligned result.
    null : NullProvenance
        The null the accuracy was tested against, in full.

    Raises
    ------
    TypeError
        If a field is given a value of the wrong type.
    ValueError
        If the accuracy is not finite or lies outside zero to one, or if the rank is
        negative.
    """

    __slots__ = ("_accuracy", "_null", "_rank", "_variable")

    def __init__(
        self,
        accuracy: float,
        variable: HighLevelVariable,
        rank: int,
        null: NullProvenance,
    ) -> None:
        if type(variable) not in {TypeAt, SplitAt, CombinatorAt, NamedVariable}:
            msg = f"an accuracy names a high-level variable, not {variable!r}"
            raise TypeError(msg)
        if not isinstance(null, NullProvenance):
            msg = f"an accuracy is tested against a NullProvenance, not {null!r}"
            raise TypeError(msg)

        self._accuracy = ensure_finite("an intervention accuracy", accuracy)
        self._variable = variable
        self._rank = ensure_non_negative("a featurizer's rank", rank)
        self._null = null

        if not 0.0 <= self._accuracy <= 1.0:
            msg = f"an intervention accuracy is not a proportion: {accuracy!r}"
            raise ValueError(msg)

    @property
    def accuracy(self) -> float:
        """Return the proportion of interventions that matched the prediction."""
        return self._accuracy

    @property
    def variable(self) -> HighLevelVariable:
        """Return the exact high-level variable that was intervened on."""
        return self._variable

    @property
    def rank(self) -> int:
        """Return the discrete width, zero for an unaligned result."""
        return self._rank

    @property
    def null(self) -> NullProvenance:
        """Return the null the accuracy was tested against."""
        return self._null

    def _identity(
        self,
    ) -> tuple[float, HighLevelVariable, int, NullProvenance]:
        """Return the four fields a result is equal on."""
        return (self._accuracy, self._variable, self._rank, self._null)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same four fields.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not IIAResult:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the four fields the result is equal on."""
        return hash((IIAResult, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this result."""
        return (
            f"IIAResult({self._accuracy!r}, {self._variable!r}, {self._rank!r}, "
            f"{self._null!r})"
        )
