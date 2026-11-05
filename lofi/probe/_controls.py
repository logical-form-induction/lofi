"""Causal controls for distributed-alignment assessments."""

from collections import Counter
from random import Random
from statistics import fmean
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, override

from lofi._checks import (
    ensure_finite,
    ensure_name,
    ensure_non_negative,
    ensure_positive,
)
from lofi._frozen import FrozenValue
from lofi.probe._causal import (
    AlignmentPolicy,
    CombinatorAt,
    CounterfactualPair,
    NamedVariable,
    Side,
    SplitAt,
    TypeAt,
)
from lofi.probe._das import DistributedAlignment
from lofi.probe._ports import ModelReadyCausalPair
from lofi.probe._sweep import GridCell, Preregistration
from lofi.stats import Bootstrap, BootstrapResult

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from torch import Tensor

    from lofi.deduce import Cell
    from lofi.nn.ports import Reading
    from lofi.probe._causal import (
        CellAssignment,
        HighLevelVariable,
    )
    from lofi.probe._provenance import Provenance

__all__ = [
    "BinaryBracketing",
    "BracketingDelta",
    "BracketingUnavailable",
    "BracketingUnavailableReason",
    "ControlCell",
    "ControlReport",
    "DeltaIIA",
    "DormantMovement",
    "DormantPathway",
    "IIAPopulation",
    "ScrambledDelta",
    "TypeScramble",
    "bracketing_distance",
    "delta_iia",
    "dormant_pathway",
    "matched_binary_bracketings",
    "random_binary_bracketing",
    "scramble_types",
]

type _TypeProfile = tuple[tuple[TypeAt, str | int, str | int], ...]
type BracketingUnavailableReason = Literal["no-span-length-match"]


def _type_variables(pair: ModelReadyCausalPair) -> tuple[TypeAt, ...]:
    """Return the pair's type variables in structural order."""
    return tuple(
        sorted(
            (variable for variable in pair.base if isinstance(variable, TypeAt)),
            key=lambda variable: variable.cell,
        )
    )


def _type_profile(pair: ModelReadyCausalPair) -> _TypeProfile:
    """Return both sides of a pair's complete type assignment."""
    return tuple(
        (variable, pair.base[variable], pair.variant[variable])
        for variable in _type_variables(pair)
    )


def _with_types(
    target: ModelReadyCausalPair, donor: ModelReadyCausalPair
) -> CounterfactualPair:
    """Return the target structure carrying the donor's complete type assignment."""
    type_variables = frozenset(_type_variables(target))
    base: dict[HighLevelVariable, str | int] = {}
    variant: dict[HighLevelVariable, str | int] = {}
    for variable in target.base:
        base[variable] = (
            donor.base[variable]
            if variable in type_variables
            else target.base[variable]
        )
        variant[variable] = (
            donor.variant[variable]
            if variable in type_variables
            else target.variant[variable]
        )
    return CounterfactualPair(
        base,
        variant,
        target.varied,
        target.span,
        target.provenance,
    )


class _ScrambledPair(FrozenValue):
    """A model-ready target carrying another target's type assignment."""

    __slots__ = ("_pair", "_target")

    _noun = "scrambled model pair"

    def __init__(
        self,
        target: ModelReadyCausalPair,
        donor: ModelReadyCausalPair,
    ) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_pair", _with_types(target, donor))

    @property
    def id(self) -> str:
        """Return the target's stable identity."""
        return self._target.id

    @property
    def tokens(self) -> tuple[str, ...]:
        """Return the target's source tokens."""
        return self._target.tokens

    @property
    def base(self) -> CellAssignment:
        """Return the scrambled base assignment."""
        return self._pair.base

    @property
    def variant(self) -> CellAssignment:
        """Return the scrambled variant assignment."""
        return self._pair.variant

    @property
    def varied(self) -> HighLevelVariable:
        """Return the target's varied variable."""
        return self._pair.varied

    @property
    def span(self) -> Cell:
        """Return the target's intervention span."""
        return self._pair.span

    @property
    def provenance(self) -> Provenance:
        """Return the target's provenance."""
        return self._pair.provenance

    @property
    def policy(self) -> AlignmentPolicy:
        """Return the target's alignment policy."""
        return self._target.policy

    @property
    def tokenizer(self) -> str:
        """Return the target's tokenizer identity."""
        return self._target.tokenizer

    def inputs(self, side: Side, /) -> Mapping[str, Tensor]:
        """Return the target's model inputs for one side."""
        return self._target.inputs(side)

    def rows(self, side: Side, /) -> tuple[int, ...]:
        """Return the target's selected rows for one side."""
        return self._target.rows(side)

    def live_rows(self, side: Side, /) -> int:
        """Return the target's live-row extent for one side."""
        return self._target.live_rows(side)


def _is_global_rename(
    original: Sequence[ModelReadyCausalPair],
    scrambled: Sequence[ModelReadyCausalPair],
) -> bool:
    """Return whether the scramble is only one global bijective label renaming."""
    renamed: dict[str | int, str | int] = {}
    reverse: dict[str | int, str | int] = {}
    for source, changed in zip(original, scrambled, strict=True):
        for variable in _type_variables(source):
            for before, after in (
                (source.base[variable], changed.base[variable]),
                (source.variant[variable], changed.variant[variable]),
            ):
                if before in renamed and renamed[before] != after:
                    return False
                if after in reverse and reverse[after] != before:
                    return False
                renamed[before] = after
                reverse[after] = before
    return bool(renamed)


def _derange_stratum(
    pairs: tuple[ModelReadyCausalPair, ...], rng: Random
) -> tuple[ModelReadyCausalPair, ...]:
    """Return donors forming a complete nonidentity profile derangement."""
    if len(pairs) < 2:
        msg = "a type-scramble stratum contains only one pair"
        raise ValueError(msg)
    profiles = tuple(_type_profile(pair) for pair in pairs)
    candidates = [list(range(len(pairs))) for _ in pairs]
    for target, available in enumerate(candidates):
        rng.shuffle(available)
        candidates[target] = [
            donor
            for donor in available
            if donor != target and profiles[donor] != profiles[target]
        ]
        if not candidates[target]:
            msg = "a type-scramble stratum has no nonidentity assignment for a pair"
            raise ValueError(msg)

    donor_to_target: dict[int, int] = {}

    def place(target: int, seen: set[int]) -> bool:
        for donor in candidates[target]:
            if donor in seen:
                continue
            seen.add(donor)
            occupying = donor_to_target.get(donor)
            if occupying is None or place(occupying, seen):
                donor_to_target[donor] = target
                return True
        return False

    order = list(range(len(pairs)))
    rng.shuffle(order)
    if any(not place(target, set()) for target in order):
        msg = "a type-scramble stratum admits no complete nonidentity derangement"
        raise ValueError(msg)
    target_to_donor = {target: donor for donor, target in donor_to_target.items()}
    return tuple(pairs[target_to_donor[target]] for target in range(len(pairs)))


class TypeScramble(FrozenValue):
    """A model-ready type scramble and its target-to-donor manifest."""

    __slots__ = ("_manifest", "_pairs", "_seed")

    _noun = "type scramble"

    def __init__(
        self,
        pairs: Sequence[ModelReadyCausalPair],
        donors: Sequence[ModelReadyCausalPair],
        *,
        seed: int,
    ) -> None:
        held_pairs = tuple(pairs)
        held_donors = tuple(donors)
        if not held_pairs or len(held_pairs) != len(held_donors):
            msg = "a type scramble pairs equal nonempty target and donor populations"
            raise ValueError(msg)
        if any(not isinstance(pair, ModelReadyCausalPair) for pair in held_pairs):
            msg = "a type scramble requires model-ready target pairs"
            raise TypeError(msg)
        if any(not isinstance(pair, ModelReadyCausalPair) for pair in held_donors):
            msg = "a type scramble requires model-ready donor pairs"
            raise TypeError(msg)
        target_ids = tuple(
            ensure_name("a type-scramble target identity", pair.id)
            for pair in held_pairs
        )
        donor_ids = tuple(
            ensure_name("a type-scramble donor identity", pair.id)
            for pair in held_donors
        )
        if len(set(target_ids)) != len(target_ids):
            msg = "a type scramble requires distinct target identities"
            raise ValueError(msg)
        if set(target_ids) != set(donor_ids):
            msg = "a type scramble's donors are a permutation of its targets"
            raise ValueError(msg)
        for target, donor in zip(held_pairs, held_donors, strict=True):
            if target.id == donor.id or _type_profile(target) == _type_profile(donor):
                msg = f"type scramble leaves target {target.id!r} unchanged"
                raise ValueError(msg)
            if target.varied != donor.varied or _type_variables(
                target
            ) != _type_variables(donor):
                msg = "a type scramble crosses an exact variable-and-inventory stratum"
                raise ValueError(msg)
        changed = tuple(
            _ScrambledPair(target, donor)
            for target, donor in zip(held_pairs, held_donors, strict=True)
        )
        if _is_global_rename(held_pairs, changed):
            msg = "a type scramble is only one global label renaming"
            raise ValueError(msg)
        object.__setattr__(self, "_pairs", changed)
        object.__setattr__(
            self,
            "_manifest",
            MappingProxyType(dict(zip(target_ids, donor_ids, strict=True))),
        )
        object.__setattr__(self, "_seed", ensure_non_negative("a scramble seed", seed))

    @property
    def pairs(self) -> tuple[ModelReadyCausalPair, ...]:
        """Return scrambled model-ready pairs in target order."""
        return self._pairs

    @property
    def manifest(self) -> Mapping[str, str]:
        """Return each target identity's donor identity."""
        return self._manifest

    @property
    def seed(self) -> int:
        """Return the local random seed used to choose donors."""
        return self._seed


def scramble_types(
    pairs: Sequence[ModelReadyCausalPair], /, *, seed: int
) -> TypeScramble:
    """Return complete type assignments deranged within exact structural strata."""
    held = tuple(pairs)
    if not held:
        msg = "a type scramble requires at least one model-ready pair"
        raise ValueError(msg)
    if any(not isinstance(pair, ModelReadyCausalPair) for pair in held):
        msg = "a type scramble requires model-ready causal pairs"
        raise TypeError(msg)
    if any(isinstance(pair.varied, NamedVariable) for pair in held):
        msg = "a type scramble does not admit NamedVariable"
        raise TypeError(msg)
    rng = Random(ensure_non_negative("a scramble seed", seed))
    strata: dict[
        tuple[HighLevelVariable, tuple[TypeAt, ...]], list[ModelReadyCausalPair]
    ] = {}
    for pair in held:
        strata.setdefault((pair.varied, _type_variables(pair)), []).append(pair)
    donor_by_id: dict[str, ModelReadyCausalPair] = {}
    for members in strata.values():
        stratum = tuple(members)
        for target, donor in zip(stratum, _derange_stratum(stratum, rng), strict=True):
            donor_by_id[target.id] = donor
    return TypeScramble(held, tuple(donor_by_id[pair.id] for pair in held), seed=seed)


class BinaryBracketing(FrozenValue):
    """A full ordered binary bracketing over a positive sentence length."""

    __slots__ = ("_length", "_spans")

    _noun = "binary bracketing"

    def __init__(self, length: int, spans: Iterable[Cell]) -> None:
        size = ensure_positive("a bracketing length", length)
        held: set[Cell] = set()
        for span in spans:
            try:
                given_start, given_end = span
            except (TypeError, ValueError) as error:
                msg = f"a binary constituent is a start and end, not {span!r}"
                raise TypeError(msg) from error
            start = ensure_non_negative("a constituent start", given_start)
            end = ensure_positive("a constituent end", given_end)
            if end - start < 2 or end > size:
                msg = f"constituent {(start, end)!r} is not internal to length {size}"
                raise ValueError(msg)
            if (start, end) in held:
                msg = f"constituent {(start, end)!r} occurs twice"
                raise ValueError(msg)
            held.add((start, end))
        expected = size - 1
        if len(held) != expected:
            msg = (
                f"a full binary bracketing of length {size} has {expected} internal "
                f"constituents, not {len(held)}"
            )
            raise ValueError(msg)
        used: set[Cell] = set()

        def visit(start: int, end: int) -> None:
            if end - start == 1:
                return
            cell = (start, end)
            if cell not in held:
                msg = f"a full binary bracketing is missing constituent {cell!r}"
                raise ValueError(msg)
            splits = tuple(
                split
                for split in range(start + 1, end)
                if split - start == 1 or (start, split) in held
                if end - split == 1 or (split, end) in held
            )
            if len(splits) != 1:
                msg = f"constituent {cell!r} has {len(splits)} binary decompositions"
                raise ValueError(msg)
            used.add(cell)
            visit(start, splits[0])
            visit(splits[0], end)

        visit(0, size)
        if used != held:
            msg = "a binary bracketing contains constituents outside its rooted tree"
            raise ValueError(msg)
        ordered = tuple(sorted(held, key=lambda span: (span[0], -span[1])))
        object.__setattr__(self, "_length", size)
        object.__setattr__(self, "_spans", ordered)

    @property
    def length(self) -> int:
        """Return the sentence length in leaves."""
        return self._length

    @property
    def spans(self) -> tuple[Cell, ...]:
        """Return internal constituents in deterministic structural order."""
        return self._spans

    @property
    def proper_widths(self) -> tuple[int, ...]:
        """Return widths of internal constituents other than the root."""
        root = (0, self._length)
        return tuple(end - start for start, end in self._spans if (start, end) != root)

    @override
    def __eq__(self, other: object) -> bool:
        if type(other) is not BinaryBracketing:
            return NotImplemented
        return (self._length, self._spans) == (other._length, other._spans)

    @override
    def __hash__(self) -> int:
        return hash((BinaryBracketing, self._length, self._spans))

    @override
    def __repr__(self) -> str:
        return f"BinaryBracketing({self._length!r}, {self._spans!r})"


def _catalan_counts(length: int) -> tuple[int, ...]:
    """Return Catalan counts through the number of trees over ``length`` leaves."""
    counts = [1]
    for nodes in range(1, length):
        counts.append(
            sum(counts[left] * counts[nodes - 1 - left] for left in range(nodes))
        )
    return tuple(counts)


def _draw_bracketing(length: int, rng: Random) -> BinaryBracketing:
    """Draw one uniformly random full binary bracketing from a local generator."""
    counts = _catalan_counts(length)
    spans: list[Cell] = []

    def draw(start: int, end: int) -> None:
        leaves = end - start
        if leaves == 1:
            return
        spans.append((start, end))
        weights = tuple(
            counts[left - 1] * counts[leaves - left - 1] for left in range(1, leaves)
        )
        chosen = rng.randrange(sum(weights))
        cumulative = 0
        split = start + 1
        for left, weight in enumerate(weights, start=1):
            cumulative += weight
            if chosen < cumulative:
                split = start + left
                break
        draw(start, split)
        draw(split, end)

    draw(0, length)
    return BinaryBracketing(length, spans)


def random_binary_bracketing(length: int, /, *, seed: int) -> BinaryBracketing:
    """Return a Catalan-weighted full binary bracketing from one local seed."""
    size = ensure_positive("a bracketing length", length)
    return _draw_bracketing(
        size, Random(ensure_non_negative("a bracketing seed", seed))
    )


def bracketing_distance(
    reference: Sequence[BinaryBracketing],
    candidate: Sequence[BinaryBracketing],
    /,
) -> float:
    """Return total-variation distance between proper-internal width histograms."""
    first = tuple(reference)
    second = tuple(candidate)
    if not first or len(first) != len(second):
        msg = "bracketing distance compares equal nonempty populations"
        raise ValueError(msg)
    if any(type(value) is not BinaryBracketing for value in (*first, *second)):
        msg = "bracketing distance compares BinaryBracketings"
        raise TypeError(msg)
    if tuple(value.length for value in first) != tuple(
        value.length for value in second
    ):
        msg = "bracketing distance requires the same ordered sentence lengths"
        raise ValueError(msg)
    left = Counter(width for value in first for width in value.proper_widths)
    right = Counter(width for value in second for width in value.proper_widths)
    total = sum(left.values())
    if total == 0:
        return 0.0
    return sum(
        abs(left[width] - right[width]) for width in left.keys() | right.keys()
    ) / (2.0 * total)


class BracketingUnavailable(FrozenValue):
    """A typed failure to draw a span-length-matched bracketing population."""

    __slots__ = ("_attempts", "_closest", "_lengths", "_reason", "_seed", "_tolerance")

    _noun = "bracketing unavailability"

    def __init__(
        self,
        reason: BracketingUnavailableReason,
        lengths: Sequence[int],
        tolerance: float,
        *,
        seed: int,
        attempts: int,
        closest: float,
    ) -> None:
        if reason != "no-span-length-match":
            msg = f"no bracketing-unavailability reason is called {reason!r}"
            raise ValueError(msg)
        held_lengths = tuple(
            ensure_positive("a sentence length", value) for value in lengths
        )
        if not held_lengths:
            msg = "bracketing unavailability names a nonempty population"
            raise ValueError(msg)
        held_tolerance = ensure_finite("a bracketing tolerance", tolerance)
        held_closest = ensure_finite("a closest bracketing distance", closest)
        if not 0.0 <= held_tolerance <= 1.0 or not 0.0 <= held_closest <= 1.0:
            msg = "bracketing distances and tolerances lie on the unit interval"
            raise ValueError(msg)
        object.__setattr__(self, "_reason", reason)
        object.__setattr__(self, "_lengths", held_lengths)
        object.__setattr__(self, "_tolerance", held_tolerance)
        object.__setattr__(
            self, "_seed", ensure_non_negative("a bracketing seed", seed)
        )
        object.__setattr__(
            self, "_attempts", ensure_positive("a bracketing attempt count", attempts)
        )
        object.__setattr__(self, "_closest", held_closest)

    @property
    def reason(self) -> BracketingUnavailableReason:
        """Return the stable reason matching failed."""
        return self._reason

    @property
    def lengths(self) -> tuple[int, ...]:
        """Return the ordered reference sentence lengths."""
        return self._lengths

    @property
    def tolerance(self) -> float:
        """Return the caller-declared maximum distance."""
        return self._tolerance

    @property
    def seed(self) -> int:
        """Return the local random seed."""
        return self._seed

    @property
    def attempts(self) -> int:
        """Return the caller-declared draw budget."""
        return self._attempts

    @property
    def closest(self) -> float:
        """Return the smallest distance among rejected populations."""
        return self._closest


def matched_binary_bracketings(
    reference: Sequence[BinaryBracketing],
    /,
    *,
    tolerance: float,
    seed: int,
    attempts: int,
) -> tuple[BinaryBracketing, ...] | BracketingUnavailable:
    """Return a Catalan draw matching proper span widths, or typed unavailability."""
    held = tuple(reference)
    if not held:
        msg = "bracketing matching requires a nonempty reference population"
        raise ValueError(msg)
    if any(type(value) is not BinaryBracketing for value in held):
        msg = "bracketing matching requires BinaryBracketings"
        raise TypeError(msg)
    maximum = ensure_finite("a bracketing tolerance", tolerance)
    if not 0.0 <= maximum <= 1.0:
        msg = f"a bracketing tolerance is not a proportion: {tolerance!r}"
        raise ValueError(msg)
    draw_count = ensure_positive("a bracketing attempt count", attempts)
    held_seed = ensure_non_negative("a bracketing seed", seed)
    lengths = tuple(value.length for value in held)
    rng = Random(held_seed)
    closest = 1.0
    for _ in range(draw_count):
        candidate = tuple(_draw_bracketing(length, rng) for length in lengths)
        distance = bracketing_distance(held, candidate)
        closest = min(closest, distance)
        if distance <= maximum:
            return candidate
    return BracketingUnavailable(
        "no-span-length-match",
        lengths,
        maximum,
        seed=held_seed,
        attempts=draw_count,
        closest=closest,
    )


class IIAPopulation(FrozenValue):
    """Stable identities and paired binary IIA outcomes for one population."""

    __slots__ = ("_outcomes",)

    _noun = "IIA population"

    def __init__(self, outcomes: Iterable[tuple[str, bool]]) -> None:
        held: list[tuple[str, bool]] = []
        for identity, outcome in outcomes:
            named = ensure_name("an IIA pair identity", identity)
            if not isinstance(outcome, bool):
                msg = f"IIA outcome for {named!r} is not boolean: {outcome!r}"
                raise TypeError(msg)
            held.append((named, outcome))
        if not held:
            msg = "an IIA population contains at least one pair"
            raise ValueError(msg)
        identities = tuple(identity for identity, _ in held)
        if len(set(identities)) != len(identities):
            msg = "an IIA population repeats a stable identity"
            raise ValueError(msg)
        object.__setattr__(
            self, "_outcomes", tuple(sorted(held, key=lambda item: item[0]))
        )

    @property
    def outcomes(self) -> tuple[tuple[str, bool], ...]:
        """Return identities and outcomes in canonical stable-ID order."""
        return self._outcomes

    @property
    def ids(self) -> tuple[str, ...]:
        """Return identities in canonical stable-ID order."""
        return tuple(identity for identity, _ in self._outcomes)

    @property
    def accuracy(self) -> float:
        """Return the share of successful interventions."""
        return fmean(float(outcome) for _, outcome in self._outcomes)


class DeltaIIA(FrozenValue):
    """Observed and control IIA with a paired interval for their difference."""

    __slots__ = ("_control", "_difference", "_observed")

    _noun = "delta IIA"

    def __init__(
        self,
        observed: IIAPopulation,
        control: IIAPopulation,
        difference: BootstrapResult,
    ) -> None:
        if type(observed) is not IIAPopulation or type(control) is not IIAPopulation:
            msg = "delta IIA compares two IIAPopulations"
            raise TypeError(msg)
        if not isinstance(difference, BootstrapResult):
            msg = "delta IIA requires a paired BootstrapResult"
            raise TypeError(msg)
        if observed.ids != control.ids:
            msg = "delta IIA requires identical stable identities"
            raise ValueError(msg)
        expected = observed.accuracy - control.accuracy
        if abs(difference.observed - expected) > 1e-12:
            msg = "delta IIA's bootstrap aggregate disagrees with its paired outcomes"
            raise ValueError(msg)
        if difference.units != len(observed.ids):
            msg = "delta IIA's bootstrap unit count disagrees with its population"
            raise ValueError(msg)
        object.__setattr__(self, "_observed", observed)
        object.__setattr__(self, "_control", control)
        object.__setattr__(self, "_difference", difference)

    @property
    def observed(self) -> IIAPopulation:
        """Return observed per-pair outcomes and IIA."""
        return self._observed

    @property
    def control(self) -> IIAPopulation:
        """Return control per-pair outcomes and IIA."""
        return self._control

    @property
    def difference(self) -> BootstrapResult:
        """Return the paired-bootstrap Delta-IIA headline."""
        return self._difference

    @property
    def headline(self) -> float:
        """Return observed Delta-IIA from the paired bootstrap."""
        return self._difference.observed


def delta_iia(
    observed: IIAPopulation,
    control: IIAPopulation,
    /,
    *,
    bootstrap: Bootstrap,
) -> DeltaIIA:
    """Return paired Delta-IIA with the caller's bootstrap provenance."""
    if type(observed) is not IIAPopulation or type(control) is not IIAPopulation:
        msg = "delta IIA compares two IIAPopulations"
        raise TypeError(msg)
    if not isinstance(bootstrap, Bootstrap):
        msg = f"delta IIA requires a Bootstrap, not {bootstrap!r}"
        raise TypeError(msg)
    if observed.ids != control.ids:
        msg = "delta IIA requires identical stable identities"
        raise ValueError(msg)
    differences = tuple(
        float(left) - float(right)
        for (_, left), (_, right) in zip(
            observed.outcomes, control.outcomes, strict=True
        )
    )
    interval = bootstrap.interval(differences, fmean)
    return DeltaIIA(observed, control, interval)


class ScrambledDelta(FrozenValue):
    """Delta-IIA bound to the exact type-scramble population it compares."""

    __slots__ = ("_delta", "_scramble")

    _noun = "scrambled delta IIA"

    def __init__(self, scramble: TypeScramble, delta: DeltaIIA) -> None:
        if type(scramble) is not TypeScramble:
            msg = f"a scrambled delta requires a TypeScramble, not {scramble!r}"
            raise TypeError(msg)
        if type(delta) is not DeltaIIA:
            msg = f"a scrambled delta requires DeltaIIA, not {delta!r}"
            raise TypeError(msg)
        if {pair.id for pair in scramble.pairs} != set(delta.observed.ids):
            msg = "a scrambled delta's manifest and IIA population are incompatible"
            raise ValueError(msg)
        object.__setattr__(self, "_scramble", scramble)
        object.__setattr__(self, "_delta", delta)

    @property
    def scramble(self) -> TypeScramble:
        """Return the exact scrambled population and donor manifest."""
        return self._scramble

    @property
    def delta(self) -> DeltaIIA:
        """Return paired Delta-IIA against the scrambled population."""
        return self._delta

    @property
    def headline(self) -> float:
        """Return observed Delta-IIA from the paired bootstrap."""
        return self._delta.headline


class BracketingDelta(FrozenValue):
    """Delta-IIA bound to an accepted span-length-matched bracketing population."""

    __slots__ = ("_control", "_delta", "_distance", "_reference", "_tolerance")

    _noun = "bracketing delta IIA"

    def __init__(
        self,
        reference: Sequence[BinaryBracketing],
        control: Sequence[BinaryBracketing],
        delta: DeltaIIA,
        *,
        tolerance: float,
    ) -> None:
        if type(delta) is not DeltaIIA:
            msg = f"a bracketing delta requires DeltaIIA, not {delta!r}"
            raise TypeError(msg)
        held_reference = tuple(reference)
        held_control = tuple(control)
        maximum = ensure_finite("a bracketing tolerance", tolerance)
        if not 0.0 <= maximum <= 1.0:
            msg = f"a bracketing tolerance is not a proportion: {tolerance!r}"
            raise ValueError(msg)
        distance = bracketing_distance(held_reference, held_control)
        if distance > maximum:
            msg = (
                f"a bracketing distance {distance!r} exceeds the preregistered "
                f"tolerance {maximum!r}"
            )
            raise ValueError(msg)
        if len(held_reference) != len(delta.observed.ids):
            msg = "a bracketing delta's trees and IIA population are incompatible"
            raise ValueError(msg)
        object.__setattr__(self, "_reference", held_reference)
        object.__setattr__(self, "_control", held_control)
        object.__setattr__(self, "_delta", delta)
        object.__setattr__(self, "_tolerance", maximum)
        object.__setattr__(self, "_distance", distance)

    @property
    def reference(self) -> tuple[BinaryBracketing, ...]:
        """Return the observed ordered bracketing population."""
        return self._reference

    @property
    def control(self) -> tuple[BinaryBracketing, ...]:
        """Return the accepted random ordered bracketing population."""
        return self._control

    @property
    def delta(self) -> DeltaIIA:
        """Return paired Delta-IIA against the bracketing control."""
        return self._delta

    @property
    def tolerance(self) -> float:
        """Return the preregistered maximum total-variation distance."""
        return self._tolerance

    @property
    def distance(self) -> float:
        """Return the observed total-variation distance."""
        return self._distance

    @property
    def headline(self) -> float:
        """Return observed Delta-IIA from the paired bootstrap."""
        return self._delta.headline


class DormantMovement(FrozenValue):
    """One source patch's binary movement of one target output."""

    __slots__ = ("_moved", "_source_id", "_target_id")

    _noun = "dormant movement"

    def __init__(self, target_id: str, source_id: str, *, moved: bool) -> None:
        object.__setattr__(
            self, "_target_id", ensure_name("a dormant target id", target_id)
        )
        object.__setattr__(
            self, "_source_id", ensure_name("a dormant source id", source_id)
        )
        if not isinstance(moved, bool):
            msg = f"a dormant movement is boolean, not {moved!r}"
            raise TypeError(msg)
        object.__setattr__(self, "_moved", moved)

    @property
    def target_id(self) -> str:
        """Return the patched target's stable identity."""
        return self._target_id

    @property
    def source_id(self) -> str:
        """Return the dormant source's stable identity."""
        return self._source_id

    @property
    def moved(self) -> bool:
        """Return whether the target argmax changed."""
        return self._moved


class DormantPathway(FrozenValue):
    """Per-pair output movement and its mean under dormant-source patches."""

    __slots__ = ("_absent", "_movements", "_policy", "_tokenizer", "_variable")

    _noun = "dormant pathway result"

    def __init__(
        self,
        movements: Sequence[DormantMovement],
        variable: HighLevelVariable,
        absent: str | int,
        policy: AlignmentPolicy,
        tokenizer: str,
    ) -> None:
        held = tuple(movements)
        if not held or any(type(value) is not DormantMovement for value in held):
            msg = "a dormant pathway contains at least one DormantMovement"
            raise ValueError(msg)
        if type(variable) not in {TypeAt, SplitAt, CombinatorAt, NamedVariable}:
            msg = f"a dormant pathway names a high-level variable, not {variable!r}"
            raise TypeError(msg)
        if type(policy) is not AlignmentPolicy:
            msg = f"a dormant pathway requires an AlignmentPolicy, not {policy!r}"
            raise TypeError(msg)
        if isinstance(absent, bool) or not isinstance(absent, str | int):
            msg = f"an absent decision is a name or position, not {absent!r}"
            raise TypeError(msg)
        if isinstance(absent, str):
            absent = ensure_name("an absent decision", absent)
        else:
            absent = ensure_non_negative("an absent decision", absent)
        target_ids = tuple(value.target_id for value in held)
        source_ids = tuple(value.source_id for value in held)
        if len(set(target_ids)) != len(target_ids) or len(set(source_ids)) != len(
            source_ids
        ):
            msg = "a dormant pathway uses distinct stable identities on each side"
            raise ValueError(msg)
        if set(target_ids) & set(source_ids):
            msg = "a dormant pathway's target and source identities are disjoint"
            raise ValueError(msg)
        object.__setattr__(
            self,
            "_movements",
            tuple(sorted(held, key=lambda movement: movement.target_id)),
        )
        object.__setattr__(self, "_variable", variable)
        object.__setattr__(self, "_absent", absent)
        object.__setattr__(self, "_policy", policy)
        object.__setattr__(
            self, "_tokenizer", ensure_name("a dormant tokenizer", tokenizer)
        )

    @property
    def movements(self) -> tuple[DormantMovement, ...]:
        """Return paired movement outcomes in canonical target-ID order."""
        return self._movements

    @property
    def target_ids(self) -> tuple[str, ...]:
        """Return target identities in canonical stable-ID order."""
        return tuple(value.target_id for value in self._movements)

    @property
    def source_ids(self) -> tuple[str, ...]:
        """Return source identities paired with canonical target IDs."""
        return tuple(value.source_id for value in self._movements)

    @property
    def movement_rate(self) -> float:
        """Return the mean binary argmax movement."""
        return fmean(float(value.moved) for value in self._movements)

    @property
    def variable(self) -> HighLevelVariable:
        """Return the variable whose declared outcome is absent at every source."""
        return self._variable

    @property
    def absent(self) -> str | int:
        """Return the caller-declared absent decision."""
        return self._absent

    @property
    def policy(self) -> AlignmentPolicy:
        """Return the common alignment policy."""
        return self._policy

    @property
    def tokenizer(self) -> str:
        """Return the common tokenizer identity."""
        return self._tokenizer


def dormant_pathway[
    ReadingT: Reading,
    PairT: ModelReadyCausalPair,
](
    alignment: DistributedAlignment[ReadingT, PairT],
    targets: Sequence[PairT],
    sources: Sequence[PairT],
    /,
    *,
    absent: str | int,
) -> DormantPathway:
    """Return output movement when fitted coordinates receive dormant readings."""
    if not isinstance(alignment, DistributedAlignment):
        msg = f"a dormant pathway requires a DistributedAlignment, not {alignment!r}"
        raise TypeError(msg)
    held_targets = tuple(targets)
    held_sources = tuple(sources)
    if not held_targets or len(held_targets) != len(held_sources):
        msg = "a dormant pathway pairs equal nonempty target and source populations"
        raise ValueError(msg)
    if any(
        not isinstance(pair, ModelReadyCausalPair)
        for pair in (*held_targets, *held_sources)
    ):
        msg = "a dormant pathway requires model-ready causal pairs"
        raise TypeError(msg)
    target_ids = tuple(
        ensure_name("a dormant target identity", pair.id) for pair in held_targets
    )
    source_ids = tuple(
        ensure_name("a dormant source identity", pair.id) for pair in held_sources
    )
    if len(set(target_ids)) != len(target_ids) or len(set(source_ids)) != len(
        source_ids
    ):
        msg = "a dormant pathway requires distinct stable identities on each side"
        raise ValueError(msg)
    if set(target_ids) & set(source_ids):
        msg = "a dormant pathway's target and source identities are disjoint"
        raise ValueError(msg)
    population = (*held_targets, *held_sources)
    if any(pair.varied != alignment.variable for pair in population):
        msg = "a dormant pathway population does not share the alignment's variable"
        raise ValueError(msg)
    if any(pair.policy != alignment.policy for pair in population):
        msg = "a dormant pathway population does not share the alignment's policy"
        raise ValueError(msg)
    tokenizer = ensure_name("a dormant tokenizer", held_targets[0].tokenizer)
    if any(pair.tokenizer != tokenizer for pair in population):
        msg = "a dormant pathway population does not share one tokenizer"
        raise ValueError(msg)
    if any(pair.variant[alignment.variable] != absent for pair in held_sources):
        msg = "every dormant source variant must hold the declared absent decision"
        raise ValueError(msg)
    moved = alignment.moves_output(held_targets, held_sources)
    return DormantPathway(
        tuple(
            DormantMovement(target.id, source.id, moved=outcome)
            for target, source, outcome in zip(
                held_targets, held_sources, moved, strict=True
            )
        ),
        alignment.variable,
        absent,
        alignment.policy,
        tokenizer,
    )


class ControlCell(FrozenValue):
    """All causal controls for one preregistered alignment cell."""

    __slots__ = ("_bracketing", "_cell", "_dormant", "_preregistration", "_scrambled")

    _noun = "control cell"

    def __init__(
        self,
        cell: GridCell,
        scrambled: ScrambledDelta,
        bracketing: BracketingDelta | BracketingUnavailable,
        dormant: DormantPathway,
        preregistration: str,
    ) -> None:
        if type(cell) is not GridCell:
            msg = f"a control result is for a GridCell, not {cell!r}"
            raise TypeError(msg)
        if type(scrambled) is not ScrambledDelta:
            msg = "a control cell requires a manifest-bound ScrambledDelta"
            raise TypeError(msg)
        if not isinstance(bracketing, BracketingDelta | BracketingUnavailable):
            msg = (
                "a control cell requires a matched BracketingDelta or typed "
                "unavailability"
            )
            raise TypeError(msg)
        if type(dormant) is not DormantPathway:
            msg = "a control cell requires a dormant-pathway result"
            raise TypeError(msg)
        population_ids = scrambled.delta.observed.ids
        scramble_pairs = tuple(
            sorted(scrambled.scramble.pairs, key=lambda pair: pair.id)
        )
        if any(pair.varied != cell.variable for pair in scramble_pairs):
            msg = "a control cell's scramble varies another high-level variable"
            raise ValueError(msg)
        if dormant.variable != cell.variable:
            msg = "a control cell's dormant pathway measures another variable"
            raise ValueError(msg)
        if any(pair.policy != dormant.policy for pair in scramble_pairs):
            msg = "a control cell's control populations use incompatible policies"
            raise ValueError(msg)
        if any(pair.tokenizer != dormant.tokenizer for pair in scramble_pairs):
            msg = "a control cell's control populations use incompatible tokenizers"
            raise ValueError(msg)
        sentence_lengths = tuple(len(pair.tokens) for pair in scramble_pairs)
        if (
            isinstance(bracketing, BracketingDelta)
            and bracketing.delta.observed.ids != population_ids
        ):
            msg = (
                "a control cell's bracketing and scrambled populations are incompatible"
            )
            raise ValueError(msg)
        if (
            isinstance(bracketing, BracketingDelta)
            and tuple(tree.length for tree in bracketing.reference) != sentence_lengths
        ):
            msg = "a control cell's bracketing and sentence lengths are incompatible"
            raise ValueError(msg)
        if isinstance(bracketing, BracketingUnavailable) and len(
            bracketing.lengths
        ) != len(population_ids):
            msg = "a control cell's unavailable bracketing population is incompatible"
            raise ValueError(msg)
        if (
            isinstance(bracketing, BracketingUnavailable)
            and bracketing.lengths != sentence_lengths
        ):
            msg = "a control cell's unavailable bracketing lengths are incompatible"
            raise ValueError(msg)
        if dormant.target_ids != population_ids:
            msg = "a control cell's dormant and Delta-IIA populations are incompatible"
            raise ValueError(msg)
        object.__setattr__(self, "_cell", cell)
        object.__setattr__(self, "_scrambled", scrambled)
        object.__setattr__(self, "_bracketing", bracketing)
        object.__setattr__(self, "_dormant", dormant)
        object.__setattr__(
            self,
            "_preregistration",
            ensure_name("a preregistration digest", preregistration),
        )

    @property
    def cell(self) -> GridCell:
        """Return the preregistered grid cell."""
        return self._cell

    @property
    def scrambled(self) -> ScrambledDelta:
        """Return Delta-IIA against the scrambled-grammar control."""
        return self._scrambled

    @property
    def bracketing(self) -> BracketingDelta | BracketingUnavailable:
        """Return matched-bracketing Delta-IIA or typed unavailability."""
        return self._bracketing

    @property
    def dormant(self) -> DormantPathway:
        """Return dormant-source output movement."""
        return self._dormant

    @property
    def preregistration(self) -> str:
        """Return the one commitment digest for the cell."""
        return self._preregistration


class ControlReport(FrozenValue):
    """Complete causal controls for every cell in one preregistered grid."""

    __slots__ = ("_preregistration", "_results")

    _noun = "control report"

    def __init__(
        self,
        preregistration: Preregistration,
        results: Iterable[ControlCell],
    ) -> None:
        if type(preregistration) is not Preregistration:
            msg = (
                f"a control report requires a Preregistration, not {preregistration!r}"
            )
            raise TypeError(msg)
        indexed: dict[GridCell, ControlCell] = {}
        expected = preregistration.subgrid.cells
        for result in results:
            if type(result) is not ControlCell:
                msg = f"a control report contains ControlCells, not {result!r}"
                raise TypeError(msg)
            if result.preregistration != preregistration.digest:
                msg = "a control cell names another preregistration digest"
                raise ValueError(msg)
            if result.cell not in expected:
                msg = f"{result.cell!r} is not a preregistered control cell"
                raise ValueError(msg)
            if result.cell in indexed:
                msg = f"control cell {result.cell!r} is reported twice"
                raise ValueError(msg)
            indexed[result.cell] = result
        if set(indexed) != set(expected):
            missing = tuple(cell for cell in expected if cell not in indexed)
            msg = f"a control report is missing preregistered cells: {missing!r}"
            raise ValueError(msg)
        object.__setattr__(self, "_preregistration", preregistration)
        object.__setattr__(self, "_results", tuple(indexed[cell] for cell in expected))

    @property
    def preregistration(self) -> Preregistration:
        """Return the commitment whose cells the report covers."""
        return self._preregistration

    @property
    def digest(self) -> str:
        """Return the report's sole preregistration digest."""
        return self._preregistration.digest

    @property
    def results(self) -> tuple[ControlCell, ...]:
        """Return controls in preregistered cell order."""
        return self._results
