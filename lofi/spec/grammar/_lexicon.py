"""Lexicon declarations and the static lexicons they build.

A lexicon is declared inline as entries, structured from named categories, or
generated with a declared category-frequency coupling over inferred categories.
"""

from collections import deque
from math import exp, log, sqrt
from random import Random
from typing import TYPE_CHECKING, Literal, Protocol, Self, cast, override

import didactic.api as dx

from lofi._checks import ensure_finite
from lofi._stats import average_ranks
from lofi.lexicon import Entry, StaticLexicon
from lofi.spec._derivation import ensure_data_seed, fixed_seed, resolved_seed
from lofi.spec.grammar._generation import generate_inventory
from lofi.spec.grammar._signature import _all_named, _nonempty

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from lofi.judge import RuleSet
    from lofi.spec.grammar._values import CategoryMeasure
    from lofi.terms import CategoryResolution, TargetSet, Term, TermStore, TermSyntax

__all__ = [
    "AmbiguityBucketSpec",
    "AmbiguityLawSpec",
    "CategoryComplexityBoundSpec",
    "ClassSpec",
    "GeneratedLexiconSpec",
    "HistogramAmbiguityLawSpec",
    "InlineLexiconSpec",
    "JointSpec",
    "LexicalEntrySpec",
    "LexiconSpec",
    "LexiconStructure",
    "ParametricAmbiguityLawSpec",
    "ParametricJointSpec",
    "PowerLawRankLawSpec",
    "RankLawSpec",
    "TabularJointSpec",
    "UniformRankLawSpec",
]


class LexicalEntrySpec(dx.Model, extra="forbid"):
    """One weighted form-to-category assignment and its optional meaning."""

    form: str
    category: str
    meaning: str | None = None
    weight: float = 0.0

    __axioms__ = (
        _nonempty("form", "a lexical entry needs a form"),
        _nonempty("category", "a lexical entry needs a category"),
    )

    @dx.validates("weight")
    @classmethod
    def _weight_is_finite(cls, weight: float) -> float:
        """Return a finite lexical weight.

        Parameters
        ----------
        weight : float
            The declared log-weight.

        Raises
        ------
        ValueError
            If the weight is infinite or not a number.
        """
        return ensure_finite("a lexical entry's weight", weight)

    @dx.validates("meaning")
    @classmethod
    def _meaning_is_nonempty(cls, meaning: str | None) -> str | None:
        """Return an absent meaning or a nonempty logical form."""
        if meaning == "":
            msg = "a lexical entry's meaning needs a logical form"
            raise ValueError(msg)

        return meaning


class _RankLawDeclaration(Protocol):
    """A declaration that assigns relative weights to ranks."""

    def weights(self, count: int) -> tuple[float, ...]:
        """Return weights for the requested rank count."""
        ...


class _AmbiguityLawDeclaration(Protocol):
    """A declaration that assigns ambiguity counts to forms."""

    def counts(self, forms: int) -> tuple[int, ...]:
        """Return one ambiguity count per form."""
        ...


class _JointDeclaration(Protocol):
    """A declaration that supplies a category-mass marginal."""

    def marginal(self) -> tuple[tuple[str, float], ...]:
        """Return categories with their normalized token shares."""
        ...


class LexiconSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """A lexical source declaration."""

    def category_complexity_bounds(self) -> tuple[int | None, int | None]:
        """Return no generated-category restrictions for this lexical source."""
        return (None, None)

    def with_derived_seed(self, derive: Callable[[str], int], path: str, /) -> Self:
        """Return the lexicon with a derived seed replaced by the one derived for it.

        A lexicon drawn under no seed, or under a fixed one, is returned as it is.

        Parameters
        ----------
        derive : Callable[[str], int]
            The seed a point derives for a dotted path.
        path : str
            The dotted path of the lexicon in the run document.
        """
        del derive, path

        return self


class InlineLexiconSpec(LexiconSpec, extra="forbid"):
    """A lexicon listed entry by entry in the declaration."""

    kind: Literal["inline"] = "inline"
    entries: tuple[LexicalEntrySpec, ...]

    __axioms__ = (
        dx.axiom("len(entries) > 0", message="an inline lexicon needs an entry"),
    )

    def meaning_entries(self) -> tuple[tuple[str, str, str], ...]:
        """Return the logical forms declared on lexical entries."""
        return tuple(
            (entry.form, entry.category, meaning)
            for entry in self.entries
            if (meaning := entry.meaning) is not None
        )

    def build(
        self,
        store: TermStore,
        syntax: TermSyntax,
        resolve_category: CategoryResolution,
        rules: RuleSet[Term],
        goal: TargetSet[Term],
        reserved: frozenset[Term],
        category_measure: CategoryMeasure,
        /,
    ) -> StaticLexicon[Term]:
        """Build the static lexicon over the interned categories.

        Parameters
        ----------
        store : TermStore
            The term store used to intern categories.
        syntax : TermSyntax
            The syntax used to parse categories.
        rules : RuleSet[Term]
            The unused rules.
        goal : TargetSet[Term]
            The unused goal.
        reserved : frozenset[Term]
            The unused reserved categories.
        """
        del store, syntax, rules, goal, reserved, category_measure

        return StaticLexicon(
            Entry(entry.form, resolve_category(entry.category), None, entry.weight)
            for entry in self.entries
        )


class RankLawSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """A law giving each rank in a sequence a relative weight."""


class UniformRankLawSpec(RankLawSpec, extra="forbid"):
    """A law weighing every rank alike."""

    kind: Literal["uniform"] = "uniform"

    def weights(self, count: int) -> tuple[float, ...]:
        """Return one for every rank.

        Parameters
        ----------
        count : int
            How many ranks to weigh.
        """
        return (1.0,) * count


class PowerLawRankLawSpec(RankLawSpec, extra="forbid"):
    """A law weighing rank r in proportion to r to the negative exponent."""

    kind: Literal["power-law"] = "power-law"
    exponent: float

    @dx.validates("exponent")
    @classmethod
    def _exponent_is_finite(cls, exponent: float) -> float:
        """Return a finite exponent.

        Parameters
        ----------
        exponent : float
            The declared exponent.

        Raises
        ------
        ValueError
            If the exponent is infinite or not a number.
        """
        return ensure_finite("a power law's exponent", exponent)

    def weights(self, count: int) -> tuple[float, ...]:
        """Return the power-law weights of the ranks, scaled to mean one.

        Parameters
        ----------
        count : int
            How many ranks to weigh.
        """
        raw = tuple(float(rank) ** -self.exponent for rank in range(1, count + 1))
        total = sum(raw)

        return tuple(weight * count / total for weight in raw)


def _apportion(total: int, weights: tuple[float, ...]) -> tuple[int, ...]:
    """Return integer counts totaling ``total`` by largest remainders."""
    scale = total / sum(weights)
    raw = tuple(weight * scale for weight in weights)
    counts = [int(value) for value in raw]
    left = total - sum(counts)
    order = sorted(
        range(len(raw)), key=lambda at: (raw[at] - counts[at], -at), reverse=True
    )

    for at in order[:left]:
        counts[at] += 1

    return tuple(counts)


class AmbiguityLawSpec(dx.TaggedUnion, discriminator="spelling", extra="forbid"):
    """A law over the number of categories borne by each form type.

    ``maximum`` is a hard support bound. ``placement`` is the signed target
    association between ambiguity and a form's frequency rank: negative
    values put ambiguous forms lower in the frequency order, zero assigns the
    orders independently, and positive values put them higher.
    """

    maximum: int
    placement: float

    __axioms__ = (
        dx.axiom("maximum > 0", message="an ambiguity maximum must be positive"),
        dx.axiom(
            "placement >= -1 and placement <= 1",
            message="ambiguity placement must lie in [-1, 1]",
        ),
    )

    @dx.validates("placement")
    @classmethod
    def _placement_is_finite(cls, placement: float) -> float:
        """Return a finite ambiguity-frequency placement."""
        return ensure_finite("an ambiguity-frequency placement", placement)


class ParametricAmbiguityLawSpec(AmbiguityLawSpec, extra="forbid"):
    """A bounded ambiguity law parameterized by its mean and dispersion.

    At zero dispersion, mass lies on the one or two integers bracketing the
    mean. At unit dispersion, it lies on one and ``maximum`` while preserving
    the same mean. Intermediate values mix those two laws. This gives a
    direct many-mild to few-heavy axis without changing mean ambiguity.
    """

    spelling: Literal["parametric"] = "parametric"
    mean: float
    dispersion: float

    __axioms__ = (
        dx.axiom("mean >= 1", message="mean ambiguity must be at least one"),
        dx.axiom(
            "mean <= maximum",
            message="mean ambiguity cannot exceed its maximum",
        ),
        dx.axiom(
            "dispersion >= 0 and dispersion <= 1",
            message="ambiguity dispersion must lie in [0, 1]",
        ),
    )

    @dx.validates("mean", "dispersion")
    @classmethod
    def _shape_is_finite(cls, value: float) -> float:
        """Return a finite ambiguity-law parameter."""
        return ensure_finite("an ambiguity-law parameter", value)

    def counts(self, forms: int) -> tuple[int, ...]:
        """Return a deterministic finite realization of the declared law."""
        lower = int(self.mean)
        upper = lower if float(lower) == self.mean else lower + 1
        mild: dict[int, float] = {lower: float(upper) - self.mean}

        if upper == lower:
            mild[lower] = 1.0
        else:
            mild[upper] = self.mean - float(lower)

        if self.maximum == 1:
            heavy = {1: 1.0}
        else:
            high = (self.mean - 1.0) / (self.maximum - 1.0)
            heavy = {1: 1.0 - high, self.maximum: high}

        probabilities = {
            ambiguity: (1.0 - self.dispersion) * mild.get(ambiguity, 0.0)
            + self.dispersion * heavy.get(ambiguity, 0.0)
            for ambiguity in range(1, self.maximum + 1)
        }
        values = tuple(probabilities)
        allocated = list(
            _apportion(forms, tuple(probabilities[value] for value in values))
        )
        target = round(forms * self.mean)
        realized = sum(
            value * count for value, count in zip(values, allocated, strict=True)
        )
        supported = [
            at for at, value in enumerate(values) if probabilities[value] > 0.0
        ]

        while realized != target:
            moves = [
                (abs(target - (realized + values[to] - values[at])), at, to)
                for at in supported
                for to in supported
                if allocated[at] > 0
                and abs(target - (realized + values[to] - values[at]))
                < abs(target - realized)
            ]

            if not moves:
                break

            _, at, to = min(moves)
            allocated[at] -= 1
            allocated[to] += 1
            realized += values[to] - values[at]

        return tuple(
            value
            for value, count in zip(values, allocated, strict=True)
            for _ in range(count)
        )


class AmbiguityBucketSpec(dx.Model, extra="forbid"):
    """One categories-per-form value and its relative share of form types."""

    categories: int
    share: float

    __axioms__ = (
        dx.axiom("categories > 0", message="an ambiguity count must be positive"),
    )

    @dx.validates("share")
    @classmethod
    def _share_is_positive(cls, share: float) -> float:
        """Return a finite positive bucket share."""
        held = ensure_finite("an ambiguity bucket's share", share)

        if held <= 0.0:
            msg = f"an ambiguity bucket's share is not positive: {share!r}"
            raise ValueError(msg)

        return held


class HistogramAmbiguityLawSpec(AmbiguityLawSpec, extra="forbid"):
    """An ambiguity law spelled as relative shares at explicit counts."""

    spelling: Literal["histogram"] = "histogram"
    buckets: tuple[AmbiguityBucketSpec, ...]

    __axioms__ = (
        dx.axiom("len(buckets) > 0", message="an ambiguity histogram needs a bucket"),
    )

    @dx.validates("buckets")
    @classmethod
    def _buckets_are_distinct(
        cls, buckets: tuple[AmbiguityBucketSpec, ...]
    ) -> tuple[AmbiguityBucketSpec, ...]:
        """Return buckets when no count is repeated."""
        values = [bucket.categories for bucket in buckets]

        if len(set(values)) != len(values):
            msg = f"an ambiguity histogram repeats a count: {values!r}"
            raise ValueError(msg)

        return buckets

    @dx.model_validator()
    def _buckets_fit_the_maximum(self) -> Self:
        """Reject a bucket beyond the declared maximum."""
        if (
            largest := max(bucket.categories for bucket in self.buckets)
        ) > self.maximum:
            msg = (
                f"an ambiguity bucket of {largest} exceeds the maximum of "
                f"{self.maximum}"
            )
            raise ValueError(msg)

        return self

    def counts(self, forms: int) -> tuple[int, ...]:
        """Return a deterministic finite realization of the histogram."""
        buckets = tuple(sorted(self.buckets, key=lambda bucket: bucket.categories))
        allocated = _apportion(forms, tuple(bucket.share for bucket in buckets))

        return tuple(
            bucket.categories
            for bucket, count in zip(buckets, allocated, strict=True)
            for _ in range(count)
        )


def _independent_supports(
    ambiguities: list[int],
    form_masses: list[float],
    shares: tuple[float, ...],
    random: Random,
) -> tuple[tuple[int, ...], ...]:
    """Return category supports without a class-size intervention."""
    supports: list[tuple[int, ...] | None] = [None] * len(ambiguities)
    covered = [0.0] * len(shares)
    form_order = sorted(
        range(len(ambiguities)), key=form_masses.__getitem__, reverse=True
    )
    unseen = set(range(len(shares)))
    slots_left = sum(ambiguities)

    for form in form_order:
        count = ambiguities[form]
        slots_after = slots_left - count
        forced_count = max(0, len(unseen) - slots_after)
        forced = sorted(
            unseen,
            key=lambda at: (shares[at] - covered[at], random.random()),
            reverse=True,
        )[:forced_count]
        candidates = sorted(
            (at for at in range(len(shares)) if at not in forced),
            key=lambda at: (shares[at] - covered[at], random.random()),
            reverse=True,
        )
        chosen = forced + candidates[: count - len(forced)]

        if len(chosen) != count:
            msg = "the ambiguity law cannot realize the category marginal"
            raise ValueError(msg)

        supports[form] = tuple(chosen)

        for at in chosen:
            covered[at] += form_masses[form]
            unseen.discard(at)

        slots_left = slots_after

    if unseen:
        msg = f"the ambiguity law leaves categories unrealized: {sorted(unseen)!r}"
        raise ValueError(msg)

    return tuple(support for support in supports if support is not None)


def _coupled_supports(
    ambiguities: list[int], targets: tuple[int, ...], random: Random
) -> tuple[tuple[int, ...], ...]:
    """Return randomized supports close to declared category class sizes."""
    supports = [set[int]() for _ in ambiguities]
    sizes = [0] * len(targets)
    unseen = set(range(len(targets)))
    slots_left = sum(ambiguities)
    form_order = sorted(
        range(len(ambiguities)),
        key=lambda form: (ambiguities[form], random.random()),
        reverse=True,
    )

    for form in form_order:
        count = ambiguities[form]
        slots_after = slots_left - count
        forced_count = max(0, len(unseen) - slots_after)
        forced = sorted(
            unseen,
            key=lambda at: (targets[at] - sizes[at], random.random()),
            reverse=True,
        )[:forced_count]
        candidates = sorted(
            (at for at in range(len(targets)) if at not in forced),
            key=lambda at: (targets[at] - sizes[at], random.random()),
            reverse=True,
        )
        chosen = forced + candidates[: count - len(forced)]

        if len(chosen) != count:
            msg = "the ambiguity law cannot realize the coupled category supports"
            raise ValueError(msg)

        supports[form].update(chosen)

        for at in chosen:
            sizes[at] += 1
            unseen.discard(at)

        slots_left = slots_after

    if unseen:
        msg = f"the ambiguity law leaves categories unrealized: {sorted(unseen)!r}"
        raise ValueError(msg)

    forms = tuple(range(len(supports)))

    for _ in range(20 * sum(ambiguities)):
        left = random.choice(forms)
        right = random.choice(forms)

        if left == right:
            continue

        left_only = tuple(sorted(supports[left] - supports[right]))
        right_only = tuple(sorted(supports[right] - supports[left]))

        if not left_only or not right_only:
            continue

        left_category = random.choice(left_only)
        right_category = random.choice(right_only)
        supports[left].remove(left_category)
        supports[left].add(right_category)
        supports[right].remove(right_category)
        supports[right].add(left_category)

    return tuple(tuple(sorted(support)) for support in supports)


def _transport_cells(
    supports: tuple[tuple[int, ...], ...],
    form_masses: list[float],
    shares: tuple[float, ...],
) -> tuple[dict[int, float], ...] | None:
    """Return a positive transport plan over supports, if one exists."""
    category_sizes = [0] * len(shares)

    for support in supports:
        for category in support:
            category_sizes[category] += 1

    epsilon = min(
        1e-9,
        *(
            mass / (2.0 * len(support))
            for mass, support in zip(form_masses, supports, strict=True)
        ),
        *(
            share / (2.0 * size)
            for share, size in zip(shares, category_sizes, strict=True)
        ),
    )
    rows = tuple(
        mass - epsilon * len(support)
        for mass, support in zip(form_masses, supports, strict=True)
    )
    columns = tuple(
        share - epsilon * size
        for share, size in zip(shares, category_sizes, strict=True)
    )
    form_offset = 1
    category_offset = form_offset + len(supports)
    sink = category_offset + len(shares)
    nodes = sink + 1
    residual = [[0.0] * nodes for _ in range(nodes)]
    adjacent = [list[int]() for _ in range(nodes)]

    def add_edge(source: int, target: int, capacity: float) -> None:
        adjacent[source].append(target)
        adjacent[target].append(source)
        residual[source][target] = capacity

    for form, mass in enumerate(rows):
        form_node = form_offset + form
        add_edge(0, form_node, mass)

        for category in supports[form]:
            add_edge(form_node, category_offset + category, 1.0)

    for category, share in enumerate(columns):
        add_edge(category_offset + category, sink, share)

    transported = 0.0
    target_mass = sum(rows)

    while transported + 1e-12 < target_mass:
        parent = [-1] * nodes
        parent[0] = 0
        queue = deque([0])

        while queue and parent[sink] == -1:
            source = queue.popleft()

            for target in adjacent[source]:
                if parent[target] == -1 and residual[source][target] > 1e-15:
                    parent[target] = source
                    queue.append(target)

        if parent[sink] == -1:
            return None

        amount = target_mass - transported
        node = sink

        while node != 0:
            source = parent[node]
            amount = min(amount, residual[source][node])
            node = source

        node = sink

        while node != 0:
            source = parent[node]
            residual[source][node] -= amount
            residual[node][source] += amount
            node = source

        transported += amount

    return tuple(
        {
            category: epsilon + residual[category_offset + category][form_offset + form]
            for category in support
        }
        for form, support in enumerate(supports)
    )


def _coupled_cells(
    ambiguities: list[int],
    targets: tuple[int, ...],
    form_masses: list[float],
    shares: tuple[float, ...],
    random: Random,
) -> tuple[dict[int, float], ...]:
    """Return a deterministic coupled joint with both marginals fixed when feasible."""
    last_supports: tuple[tuple[int, ...], ...] | None = None
    singleton_uniform = all(ambiguity == 1 for ambiguity in ambiguities) and (
        max(form_masses) - min(form_masses) <= 1e-15
    )
    compatible_singletons = all(
        abs(share / form_masses[0] - round(share / form_masses[0])) <= 1e-12
        for share in shares
    )
    attempts = 1 if singleton_uniform and not compatible_singletons else 64

    for _ in range(attempts):
        supports = _coupled_supports(ambiguities, targets, random)
        last_supports = supports
        cells = _transport_cells(supports, form_masses, shares)

        if cells is not None:
            return cells

    if last_supports is None:
        msg = "the ambiguity law supplies no category supports"
        raise ValueError(msg)

    if all(len(support) == 1 for support in last_supports):
        return tuple(
            {support[0]: mass}
            for support, mass in zip(last_supports, form_masses, strict=True)
        )

    cells = [dict.fromkeys(support, 1.0) for support in last_supports]

    for _ in range(100):
        totals = [0.0] * len(shares)

        for row in cells:
            for category, value in row.items():
                totals[category] += value

        for row in cells:
            for category in row:
                row[category] *= shares[category] / totals[category]

        for mass, row in zip(form_masses, cells, strict=True):
            scale = mass / sum(row.values())

            for category in row:
                row[category] *= scale

    return tuple(cells)


def _drawn_entries[C](
    classes: Iterable[tuple[C, float]],
    forms: int,
    ambiguity: AmbiguityLawSpec,
    rank_law: RankLawSpec,
    coupling: float | None,
    random: Random,
) -> tuple[Entry[C], ...]:
    """Return entries with declared category, ambiguity and frequency marginals.

    Parameters
    ----------
    classes : Iterable[tuple[C, float]]
        Each category with its share of token mass.
    forms : int
        How many forms the inventory holds.
    ambiguity : AmbiguityLawSpec
        The law over categories per form and their frequency placement.
    rank_law : RankLawSpec
        The law over corpus-level form frequency ranks.
    coupling : float | None
        The fitted category-frequency coupling targeted through category class
        sizes, or nothing to leave those sizes unconstrained.
    random : Random
        The source of seeded tie-breaking and independent rank placement.
    """
    held = tuple(classes)
    categories = tuple(category for category, _ in held)
    shares = tuple(share for _, share in held)
    ambiguities = cast("_AmbiguityLawDeclaration", ambiguity).counts(forms)
    random.shuffle(ambiguities := list(ambiguities))
    slots = sum(ambiguities)

    if slots < len(categories):
        msg = (
            f"the ambiguity law supplies {slots} category slots for "
            f"{len(categories)} categories"
        )
        raise ValueError(msg)

    base = sorted(cast("_RankLawDeclaration", rank_law).weights(forms))
    ambiguity_ranks = average_ranks([float(value) for value in ambiguities])
    mean_rank = sum(ambiguity_ranks) / forms
    centered = [rank - mean_rank for rank in ambiguity_ranks]
    spread = sqrt(sum(value * value for value in centered))
    ambiguity_axis = [value / spread if spread else 0.0 for value in centered]
    noise = [random.random() for _ in range(forms)]
    mean_noise = sum(noise) / forms
    centered_noise = [value - mean_noise for value in noise]
    projection = sum(x * y for x, y in zip(ambiguity_axis, centered_noise, strict=True))
    residual = [
        y - projection * x for x, y in zip(ambiguity_axis, centered_noise, strict=True)
    ]
    noise_spread = sqrt(sum(value * value for value in residual))
    noise_axis = [value / noise_spread if noise_spread else 0.0 for value in residual]
    scores = [
        ambiguity.placement * x
        + sqrt(1.0 - ambiguity.placement * ambiguity.placement) * y
        for x, y in zip(ambiguity_axis, noise_axis, strict=True)
    ]
    frequency_order = sorted(range(forms), key=scores.__getitem__)
    form_masses = [0.0] * forms

    for form, weight in zip(frequency_order, base, strict=True):
        form_masses[form] = weight / forms

    size_targets = (
        None
        if coupling is None
        else _coupled_class_sizes(shares, tuple(form_masses), slots, forms, coupling)
    )
    if size_targets is None:
        supports = _independent_supports(ambiguities, form_masses, shares, random)
        cells = [dict.fromkeys(support, 1.0) for support in supports]

        for _ in range(100):
            totals = [0.0] * len(categories)

            for row in cells:
                for category, value in row.items():
                    totals[category] += value

            for row in cells:
                for category in row:
                    row[category] *= shares[category] / totals[category]

            for mass, row in zip(form_masses, cells, strict=True):
                scale = mass / sum(row.values())

                for category in row:
                    row[category] *= scale

        totals = [0.0] * len(categories)

        for row in cells:
            for category, value in row.items():
                totals[category] += value

        for row in cells:
            for category in row:
                row[category] *= shares[category] / totals[category]
    else:
        cells = list(
            _coupled_cells(ambiguities, size_targets, form_masses, shares, random)
        )

    return tuple(
        Entry(f"w{form}", categories[category], None, log(mass))
        for form, row in enumerate(cells)
        for category, mass in row.items()
    )


def _coupled_class_sizes(
    shares: tuple[float, ...],
    form_masses: tuple[float, ...],
    slots: int,
    forms: int,
    coupling: float,
) -> tuple[int, ...]:
    """Return class-size targets for a category-frequency coupling.

    With category token shares held fixed, coupling zero assigns class sizes in
    proportion to those shares, so mean token mass per form does not depend on
    category. Away from zero, the exponent ``1 / (1 - coupling)`` changes the
    association while leaving both the category-share and form-frequency
    marginals untouched. Integer targets are bounded by one and the number of
    forms; the built and sampled lexicon reports the coupling it actually
    achieves rather than treating this finite target as an observation.
    """
    exponent = float(forms * len(shares)) if coupling == 1.0 else 1.0 / (1.0 - coupling)
    log_weights = tuple(exponent * log(share) for share in shares)
    largest = max(log_weights)
    weights = tuple(exp(weight - largest) for weight in log_weights)
    total_weight = sum(weights)
    ideals = tuple(slots * weight / total_weight for weight in weights)
    descending_masses = sorted(form_masses, reverse=True)
    minimums = []

    for share in shares:
        carried = 0.0

        for at, mass in enumerate(descending_masses, start=1):
            carried += mass

            if carried + 1e-15 >= share:
                minimums.append(at)
                break

    sizes = minimums if sum(minimums) <= slots else [1] * len(shares)

    for _ in range(slots - sum(sizes)):
        available = (at for at, size in enumerate(sizes) if size < forms)
        at = max(
            available,
            key=lambda held: (ideals[held] - sizes[held], weights[held], -held),
        )
        sizes[at] += 1

    return tuple(sizes)


class JointSpec(dx.TaggedUnion, discriminator="spelling", extra="forbid"):
    """A spelling of the category token-mass marginal."""


class ParametricJointSpec(JointSpec, extra="forbid"):
    """A category marginal whose shares follow a rank law."""

    spelling: Literal["parametric"] = "parametric"
    categories: tuple[str, ...]
    category_shares: RankLawSpec

    __axioms__ = (
        dx.axiom("len(categories) > 0", message="a parametric joint needs a category"),
        _all_named("categories", "a category needs a name"),
    )

    @dx.validates("categories")
    @classmethod
    def _categories_are_distinct(cls, categories: tuple[str, ...]) -> tuple[str, ...]:
        """Return the categories when none is named twice.

        Parameters
        ----------
        categories : tuple[str, ...]
            The declared categories.

        Raises
        ------
        ValueError
            If a category is named twice.
        """
        if len(set(categories)) != len(categories):
            msg = f"a parametric joint names a category twice: {categories!r}"
            raise ValueError(msg)

        return categories

    def marginal(self) -> tuple[tuple[str, float], ...]:
        """Return category shares following the declared rank law."""
        weights = cast("_RankLawDeclaration", self.category_shares).weights(
            len(self.categories)
        )
        total = sum(weights)

        return tuple(
            (category, weight / total)
            for category, weight in zip(self.categories, weights, strict=True)
        )


class ClassSpec(dx.Model, extra="forbid"):
    """One category and its relative share of token mass."""

    category: str
    share: float

    __axioms__ = (_nonempty("category", "a class needs a category"),)

    @dx.validates("share")
    @classmethod
    def _share_is_positive(cls, share: float) -> float:
        """Return a finite, positive share.

        Parameters
        ----------
        share : float
            The declared share.

        Raises
        ------
        ValueError
            If the share is infinite, not a number, or not positive.
        """
        held = ensure_finite("a class's share", share)

        if held <= 0.0:
            msg = f"a class's share is not positive: {share!r}"
            raise ValueError(msg)

        return held


class TabularJointSpec(JointSpec, extra="forbid"):
    """A category marginal spelled as an explicit table of relative shares.

    The shares are relative: they are scaled to sum to one.
    """

    spelling: Literal["tabular"] = "tabular"
    rows: tuple[ClassSpec, ...]

    __axioms__ = (dx.axiom("len(rows) > 0", message="a tabular joint needs a row"),)

    @dx.validates("rows")
    @classmethod
    def _categories_are_distinct(
        cls, rows: tuple[ClassSpec, ...]
    ) -> tuple[ClassSpec, ...]:
        """Return the rows when no category has two.

        Parameters
        ----------
        rows : tuple[ClassSpec, ...]
            The declared rows.

        Raises
        ------
        ValueError
            If a category has two rows.
        """
        categories = [row.category for row in rows]

        if len(set(categories)) != len(categories):
            msg = f"a tabular joint names a category twice: {categories!r}"
            raise ValueError(msg)

        return rows

    def marginal(self) -> tuple[tuple[str, float], ...]:
        """Return each category with its normalized token share, in order."""
        total = sum(row.share for row in self.rows)

        return tuple((row.category, row.share / total) for row in self.rows)


class LexiconStructure(LexiconSpec, extra="forbid"):
    """A lexicon with independent category, ambiguity and frequency marginals.

    The ambiguity law fixes how many categories every form bears. The joint
    assigns those incidence slots and their token mass across categories. The
    rank law is corpus-level: it fixes the marginal frequency ranks of forms,
    rather than changing ambiguity through within-class sampling.
    """

    kind: Literal["structured"] = "structured"
    forms: int
    joint: JointSpec
    ambiguity: AmbiguityLawSpec
    rank_law: RankLawSpec
    seed: int | str

    __axioms__ = (dx.axiom("forms > 0", message="a form inventory must be positive"),)

    def meaning_entries(self) -> tuple[tuple[str, str, str], ...]:
        """Return no meanings because this source generates only categories."""
        return ()

    @dx.validates("seed")
    @classmethod
    def _seed_is_a_data_seed(cls, value: int | str) -> int | str:
        """Return a nonnegative seed or the derived spelling."""
        return ensure_data_seed("a lexicon's seed", value)

    @override
    def with_derived_seed(self, derive: Callable[[str], int], path: str, /) -> Self:
        """Return the lexicon drawn under the seed a point derives for it.

        Parameters
        ----------
        derive : Callable[[str], int]
            The seed a point derives for a dotted path.
        path : str
            The dotted path of the lexicon in the run document.
        """
        return self.with_(seed=resolved_seed(self.seed, derive, f"{path}.seed"))

    @dx.model_validator()
    def _ambiguity_fits_the_categories(self) -> Self:
        """Reject ambiguity that the category inventory cannot realize.

        Raises
        ------
        ValueError
            If a form may bear more categories than the joint declares or
            the realized law has too few slots to use every category.
        """
        categories = len(cast("_JointDeclaration", self.joint).marginal())

        if self.ambiguity.maximum > categories:
            msg = (
                f"the ambiguity maximum of {self.ambiguity.maximum} exceeds "
                f"the {categories} declared categories"
            )
            raise ValueError(msg)

        slots = sum(cast("_AmbiguityLawDeclaration", self.ambiguity).counts(self.forms))

        if slots < categories:
            msg = (
                f"the ambiguity law supplies {slots} category slots for "
                f"{categories} declared categories"
            )
            raise ValueError(msg)

        return self

    def build(
        self,
        store: TermStore,
        syntax: TermSyntax,
        resolve_category: CategoryResolution,
        rules: RuleSet[Term],
        goal: TargetSet[Term],
        reserved: frozenset[Term],
        category_measure: CategoryMeasure,
        /,
    ) -> StaticLexicon[Term]:
        """Draw the lexicon the joint declares over the interned categories.

        Parameters
        ----------
        store : TermStore
            The term store used to intern categories.
        syntax : TermSyntax
            The syntax used to parse categories.
        rules : RuleSet[Term]
            The unused rules.
        goal : TargetSet[Term]
            The unused goal.
        reserved : frozenset[Term]
            The unused reserved categories.
        """
        del store, syntax, rules, goal, reserved, category_measure

        return StaticLexicon(
            _drawn_entries(
                (
                    (resolve_category(category), share)
                    for category, share in cast(
                        "_JointDeclaration", self.joint
                    ).marginal()
                ),
                self.forms,
                self.ambiguity,
                self.rank_law,
                None,
                Random(fixed_seed("a lexicon's seed", self.seed)),
            )
        )


class CategoryComplexityBoundSpec(dx.Model, extra="forbid"):
    """One nonnegative upper bound on a lexical-category measurement."""

    value: int

    __axioms__ = (
        dx.axiom("value >= 0", message="a category complexity bound is nonnegative"),
    )


class GeneratedLexiconSpec(LexiconSpec, extra="forbid"):
    """A lexicon over categories generated from the signature and the rules.

    No category is named. The inventory is grown from the goal categories
    through the inverse of the declared rules, as
    :func:`lofi.spec.grammar.generate_inventory` describes: a derivation starts at a
    goal, a node is expanded by residuating one premise of a rule against it
    with the rule's other premises drawn from the primitives and the
    categories already admitted, and the leaves of every admitted derivation
    are the lexical categories. Each category is therefore a leaf of a
    derivation of a goal whose other leaves are lexical too. The primitives
    are the signature's bare nullary categories less those it reserves for its
    rules, such as a CCG signature's conjunction markers, so sweeping the
    signature's primitive list sweeps what the inventory is built from.

    The generated categories take token-share ranks in the order they were
    admitted. Ambiguity and form frequency follow their independent laws as
    in :class:`LexiconStructure`. One seeded generator makes every choice, so
    equal declarations over equal grammars build equal lexicons.

    Attributes
    ----------
    forms : int
        How many forms the drawn lexicon's inventory holds.
    categories : int
        How many distinct lexical categories the inventory holds, exactly.
    category_shares : RankLawSpec
        The law category token shares follow over generated category ranks.
    coupling : float
        The target fitted association between category class size and mean
        token mass per form. The corpus reports the association actually
        achieved by its finite draw.
    ambiguity : AmbiguityLawSpec
        The law over categories borne by each form type.
    rank_law : RankLawSpec
        The corpus-level law over form frequency ranks.
    seed : int | str
        The seed of every generation and draw choice, or ``derived`` to take
        the seed each point derives for the lexicon.
    max_depth : int
        The deepest a lexical category may nest constructors, counting a
        primitive as zero and a constructor as one more than its deepest
        argument, so ``(S/NP)/NP`` has depth two. It also bounds every
        premise drawn or residuated during generation.
    max_arguments : CategoryComplexityBoundSpec | None
        The greatest functor-spine argument count an admitted category may
        have, or no bound.
    max_order : CategoryComplexityBoundSpec | None
        The greatest order under the signature's declared functor roles an
        admitted category may have, or no bound.
    derivation_depth : int
        The most steps between the root of a generating derivation and a
        leaf. The root is expanded by up to ``patience`` attempts, each
        drawing its premises afresh, and is a leaf only if all of them fail.
    expansion : float
        The chance a node strictly between the root and the derivation depth
        is expanded rather than left a lexical leaf.
    patience : int
        How many consecutive generating derivations may be refused, for
        adding no new category, for overfilling the inventory or for a step
        the rules or normal form refuse, before the declaration is refused,
        and how many expansions of a derivation's root are attempted.
    """

    kind: Literal["generated"] = "generated"
    forms: int
    categories: int
    category_shares: RankLawSpec
    coupling: float
    ambiguity: AmbiguityLawSpec
    rank_law: RankLawSpec
    seed: int | str
    max_depth: int = 3
    max_arguments: CategoryComplexityBoundSpec | None = None
    max_order: CategoryComplexityBoundSpec | None = None
    derivation_depth: int = 4
    expansion: float = 0.5
    patience: int = 1_000

    __axioms__ = (
        dx.axiom("forms > 0", message="a form inventory must be positive"),
        dx.axiom("categories > 0", message="a category inventory must be positive"),
        dx.axiom(
            "ambiguity.maximum <= categories",
            message="ambiguity cannot exceed the category inventory",
        ),
        dx.axiom("max_depth >= 0", message="a category depth bound is nonnegative"),
        dx.axiom(
            "derivation_depth > 0", message="a derivation depth bound must be positive"
        ),
        dx.axiom("patience > 0", message="a generation patience must be positive"),
    )

    def meaning_entries(self) -> tuple[tuple[str, str, str], ...]:
        """Return no meanings because this source generates only categories."""
        return ()

    @dx.validates("coupling")
    @classmethod
    def _coupling_is_finite(cls, coupling: float) -> float:
        """Return a finite category-frequency coupling."""
        return ensure_finite("a generated lexicon's coupling", coupling)

    @dx.validates("seed")
    @classmethod
    def _seed_is_a_data_seed(cls, value: int | str) -> int | str:
        """Return a nonnegative seed or the derived spelling."""
        return ensure_data_seed("a lexicon's seed", value)

    @override
    def category_complexity_bounds(self) -> tuple[int | None, int | None]:
        """Return the declared argument-count and category-order bounds."""
        return (
            None if self.max_arguments is None else self.max_arguments.value,
            None if self.max_order is None else self.max_order.value,
        )

    @override
    def with_derived_seed(self, derive: Callable[[str], int], path: str, /) -> Self:
        """Return the lexicon drawn under the seed a point derives for it.

        Parameters
        ----------
        derive : Callable[[str], int]
            The seed a point derives for a dotted path.
        path : str
            The dotted path of the lexicon in the run document.
        """
        return self.with_(seed=resolved_seed(self.seed, derive, f"{path}.seed"))

    @dx.validates("expansion")
    @classmethod
    def _expansion_is_a_chance(cls, expansion: float) -> float:
        """Return an expansion chance lying in the unit interval.

        Parameters
        ----------
        expansion : float
            The declared chance.

        Raises
        ------
        ValueError
            If the chance is not a number or lies outside [0, 1].
        """
        held = ensure_finite("an expansion chance", expansion)

        if not 0.0 <= held <= 1.0:
            msg = f"an expansion chance lies outside [0, 1]: {expansion!r}"
            raise ValueError(msg)

        return held

    def build(
        self,
        store: TermStore,
        syntax: TermSyntax,
        resolve_category: CategoryResolution,
        rules: RuleSet[Term],
        goal: TargetSet[Term],
        reserved: frozenset[Term],
        category_measure: CategoryMeasure,
        /,
    ) -> StaticLexicon[Term]:
        """Generate the category inventory and draw the lexicon over it.

        Parameters
        ----------
        store : TermStore
            The term store categories are interned in.
        syntax : TermSyntax
            The unused category syntax.
        rules : RuleSet[Term]
            The rules the inventory is generated through.
        goal : TargetSet[Term]
            The categories every generating derivation concludes.
        reserved : frozenset[Term]
            The bare categories the signature reserves for its rules, which
            the premise pool leaves out.
        category_measure : CategoryMeasure
            The signature's declared result roles used to bound arguments
            and order.

        Returns
        -------
        StaticLexicon[Term]
            The drawn lexicon over exactly ``categories`` categories.

        Raises
        ------
        InventoryExhaustedError
            If the rules stop yielding new categories within the bounds
            before the inventory is full.
        """
        del resolve_category, syntax
        random = Random(fixed_seed("a lexicon's seed", self.seed))
        sort = goal.categories[0].op.sort
        vocabulary = store.signature.vocabulary(sort)
        primitives = tuple(
            category
            for name in vocabulary.names
            if store.signature.constant_core(sort, name) == name
            and (category := store.const(sort, name)) not in reserved
        )
        inventory = generate_inventory(
            rules,
            goal.categories,
            primitives,
            category_measure=category_measure,
            categories=self.categories,
            max_depth=self.max_depth,
            max_arguments=(
                None if self.max_arguments is None else self.max_arguments.value
            ),
            max_order=None if self.max_order is None else self.max_order.value,
            derivation_depth=self.derivation_depth,
            expansion=self.expansion,
            patience=self.patience,
            random=random,
        )
        weights = cast("_RankLawDeclaration", self.category_shares).weights(
            self.categories
        )
        total = sum(weights)

        return StaticLexicon(
            _drawn_entries(
                (
                    (category, weight / total)
                    for category, weight in zip(
                        inventory.categories, weights, strict=True
                    )
                ),
                self.forms,
                self.ambiguity,
                self.rank_law,
                self.coupling,
                random,
            )
        )
