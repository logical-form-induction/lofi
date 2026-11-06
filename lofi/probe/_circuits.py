"""Circuits and their supporting evidence."""

import json
from hashlib import blake2b
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NewType, override

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi.nn.sites import Site
from lofi.probe._json import a_count, a_name, a_number, an_object, field
from lofi.probe._provenance import Provenance, provenance_from_json

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from lofi._types import JsonValue

__all__ = [
    "AblationEvidence",
    "AlignmentEvidence",
    "AttributionEvidence",
    "Circuit",
    "CircuitEdge",
    "CircuitId",
    "Evidence",
    "StatisticalEvidence",
    "Unit",
    "evidence_from_json",
]

CircuitId = NewType("CircuitId", str)
"""A circuit identifier derived from its units."""


class Unit:
    """Identifies a coordinate within a published value.

    Parameters
    ----------
    site : Site
        The published point the unit indexes into.
    index : Mapping[str, int] | None
        Which entry of the value there, one coordinate per axis, keyed by the axis name
        the site's specification declares.

    Raises
    ------
    TypeError
        If the site is not a site, or a coordinate is not an integer.
    ValueError
        If an axis name is empty, or a coordinate is negative.
    """

    __slots__ = ("_index", "_site")

    def __init__(self, site: Site, index: Mapping[str, int] | None = None) -> None:
        if not isinstance(site, Site):
            msg = f"a unit's site is not a Site: {site!r}"
            raise TypeError(msg)

        coordinates: dict[str, int] = {}

        for axis, coordinate in (index or {}).items():
            named = ensure_name("a unit's axis name", axis)
            coordinates[named] = ensure_non_negative(
                f"the coordinate on axis {named!r}", coordinate
            )

        self._site = site
        self._index = MappingProxyType(coordinates)

    @property
    def site(self) -> Site:
        """Return the published point the unit indexes into."""
        return self._site

    @property
    def index(self) -> Mapping[str, int]:
        """Return the coordinate, keyed by axis name, as a read-only view."""
        return self._index

    def _identity(self) -> tuple[Site, tuple[tuple[str, int], ...]]:
        """Return the two fields a unit is equal on, with the axes ordered."""
        return (self._site, tuple(sorted(self._index.items())))

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value names the same entry at the same point.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not Unit:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the site and the coordinate."""
        return hash((Unit, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this unit."""
        return f"Unit({self._site!r}, {dict(sorted(self._index.items()))!r})"


def _canonical(unit: Unit) -> str:
    """Return one unit as the text its identity is hashed from.

    Parameters
    ----------
    unit : Unit
        The unit to render.

    Returns
    -------
    str
        Its canonical rendering, which contains no newline.
    """
    site = unit.site

    return json.dumps(
        [
            site.component,
            site.role.name,
            list(site.key),
            sorted(unit.index.items()),
        ],
        separators=(",", ":"),
    )


class Circuit:
    """Groups probing units under a content-derived identifier.

    Parameters
    ----------
    units : Iterable[Unit]
        The units the circuit is made of, in any order and with repeats.

    Raises
    ------
    TypeError
        If a member is not a unit.
    """

    __slots__ = ("_id", "_units")

    def __init__(self, units: Iterable[Unit]) -> None:
        held = frozenset(units)

        for unit in held:
            if type(unit) is not Unit:
                msg = f"a circuit holds units, not {unit!r}"
                raise TypeError(msg)

        payload = "\n".join(sorted(_canonical(unit) for unit in held))

        self._units = held
        self._id = CircuitId(blake2b(payload.encode(), digest_size=16).hexdigest())

    @property
    def units(self) -> frozenset[Unit]:
        """Return the units the circuit is made of."""
        return self._units

    @property
    def id(self) -> CircuitId:
        """Return the identifier derived from those units."""
        return self._id

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value contains the same units.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not Circuit:
            return NotImplemented

        return self._units == other._units

    @override
    def __hash__(self) -> int:
        """Return a hash over the units the circuit contains."""
        return hash((Circuit, self._units))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this circuit."""
        return f"Circuit({sorted(self._units, key=_canonical)!r})"


class AttributionEvidence:
    """Records an edge's attributed share of a metric.

    Parameters
    ----------
    effect : float
        The signed change in the metric attributed to the edge.
    provenance : Provenance
        The estimator that produced it, over which data, under which seed.

    Raises
    ------
    TypeError
        If the effect is not a number, or the provenance is not one.
    ValueError
        If the effect is not finite.
    """

    __slots__ = ("_effect", "_provenance")

    def __init__(self, effect: float, provenance: Provenance) -> None:
        if type(provenance) is not Provenance:
            msg = f"evidence requires a Provenance, not {provenance!r}"
            raise TypeError(msg)

        self._effect = ensure_finite("an attributed effect", effect)
        self._provenance = provenance

    @property
    def kind(self) -> Literal["attribution"]:
        """Return which variant of evidence this is."""
        return "attribution"

    @property
    def effect(self) -> float:
        """Return the signed change in the metric attributed to the edge."""
        return self._effect

    @property
    def provenance(self) -> Provenance:
        """Return what produced it, over what data, under what seed."""
        return self._provenance

    def to_json(self) -> dict[str, JsonValue]:
        """Return the evidence as a JSON object with its tag.

        Returns
        -------
        dict[str, JsonValue]
            The tag, the effect and the nested provenance.
        """
        return {
            "kind": self.kind,
            "effect": self._effect,
            "provenance": self._provenance.to_json(),
        }

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same effect and provenance.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not AttributionEvidence:
            return NotImplemented

        return self._effect == other._effect and self._provenance == other._provenance

    @override
    def __hash__(self) -> int:
        """Return a hash over the effect and the provenance."""
        return hash((AttributionEvidence, self._effect, self._provenance))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this evidence."""
        return f"AttributionEvidence({self._effect!r}, {self._provenance!r})"


class AblationEvidence:
    """Records what a metric did when the source of an edge was removed.

    Parameters
    ----------
    effect : float
        The signed change in the metric under the ablation.
    baseline : str
        What the ablation put in the source's place.
    provenance : Provenance
        The estimator that produced it, over which data, under which seed.

    Raises
    ------
    TypeError
        If the effect is not a number, or the provenance is not one.
    ValueError
        If the effect is not finite, or the baseline is empty.
    """

    __slots__ = ("_baseline", "_effect", "_provenance")

    def __init__(self, effect: float, baseline: str, provenance: Provenance) -> None:
        if type(provenance) is not Provenance:
            msg = f"evidence requires a Provenance, not {provenance!r}"
            raise TypeError(msg)

        self._effect = ensure_finite("an ablation effect", effect)
        self._baseline = ensure_name("an ablation baseline", baseline)
        self._provenance = provenance

    @property
    def kind(self) -> Literal["ablation"]:
        """Return which variant of evidence this is."""
        return "ablation"

    @property
    def effect(self) -> float:
        """Return the signed change in the metric under the ablation."""
        return self._effect

    @property
    def baseline(self) -> str:
        """Return what the ablation put in the source's place."""
        return self._baseline

    @property
    def provenance(self) -> Provenance:
        """Return what produced it, over what data, under what seed."""
        return self._provenance

    def to_json(self) -> dict[str, JsonValue]:
        """Return the evidence as a JSON object with its tag.

        Returns
        -------
        dict[str, JsonValue]
            The tag, the effect, the baseline and the nested provenance.
        """
        return {
            "kind": self.kind,
            "effect": self._effect,
            "baseline": self._baseline,
            "provenance": self._provenance.to_json(),
        }

    def _identity(self) -> tuple[float, str, Provenance]:
        """Return the three fields this evidence is equal on."""
        return (self._effect, self._baseline, self._provenance)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same three fields.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not AblationEvidence:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the three fields this evidence is equal on."""
        return hash((AblationEvidence, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this evidence."""
        return (
            f"AblationEvidence({self._effect!r}, {self._baseline!r}, "
            f"{self._provenance!r})"
        )


class StatisticalEvidence:
    """Records an edge statistic and its null comparison.

    Parameters
    ----------
    statistic : float
        The number the test computed over the edge.
    null : str
        The name of the null model the statistic was compared against.
    provenance : Provenance
        The test that produced it, over which data, under which seed.

    Raises
    ------
    TypeError
        If the statistic is not a number, or the provenance is not one.
    ValueError
        If the statistic is not finite, or the null's name is empty.
    """

    __slots__ = ("_null", "_provenance", "_statistic")

    def __init__(self, statistic: float, null: str, provenance: Provenance) -> None:
        if type(provenance) is not Provenance:
            msg = f"evidence requires a Provenance, not {provenance!r}"
            raise TypeError(msg)

        self._statistic = ensure_finite("a test statistic", statistic)
        self._null = ensure_name("a null's name", null)
        self._provenance = provenance

    @property
    def kind(self) -> Literal["statistical"]:
        """Return which variant of evidence this is."""
        return "statistical"

    @property
    def statistic(self) -> float:
        """Return the number the test computed over the edge."""
        return self._statistic

    @property
    def null(self) -> str:
        """Return the name of the null model the statistic was compared against."""
        return self._null

    @property
    def provenance(self) -> Provenance:
        """Return what produced it, over what data, under what seed."""
        return self._provenance

    def to_json(self) -> dict[str, JsonValue]:
        """Return the evidence as a JSON object with its tag.

        Returns
        -------
        dict[str, JsonValue]
            The tag, the statistic, the null's name and the nested provenance.
        """
        return {
            "kind": self.kind,
            "statistic": self._statistic,
            "null": self._null,
            "provenance": self._provenance.to_json(),
        }

    def _identity(self) -> tuple[float, str, Provenance]:
        """Return the three fields this evidence is equal on."""
        return (self._statistic, self._null, self._provenance)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same three fields.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not StatisticalEvidence:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the three fields this evidence is equal on."""
        return hash((StatisticalEvidence, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this evidence."""
        return (
            f"StatisticalEvidence({self._statistic!r}, {self._null!r}, "
            f"{self._provenance!r})"
        )


class AlignmentEvidence:
    """Records an aligned intervention result.

    Parameters
    ----------
    accuracy : float
        The proportion of interventions whose output matched what the high-level
        variable predicts, from zero to one.
    rank : int
        The dimension of the subspace the alignment was found in.
    provenance : Provenance
        The search that produced it, over which data, under which seed.

    Raises
    ------
    TypeError
        If the accuracy is not a number, the rank is not an integer, or the provenance
        is not one.
    ValueError
        If the accuracy is not finite or lies outside zero to one, or the rank is
        negative.
    """

    __slots__ = ("_accuracy", "_provenance", "_rank")

    def __init__(self, accuracy: float, rank: int, provenance: Provenance) -> None:
        if type(provenance) is not Provenance:
            msg = f"evidence requires a Provenance, not {provenance!r}"
            raise TypeError(msg)

        self._accuracy = ensure_finite("an intervention accuracy", accuracy)
        self._rank = ensure_non_negative("an alignment's rank", rank)
        self._provenance = provenance

        if not 0.0 <= self._accuracy <= 1.0:
            msg = f"an intervention accuracy is not a proportion: {accuracy!r}"
            raise ValueError(msg)

    @property
    def kind(self) -> Literal["alignment"]:
        """Return which variant of evidence this is."""
        return "alignment"

    @property
    def accuracy(self) -> float:
        """Return the proportion of interventions that matched the prediction."""
        return self._accuracy

    @property
    def rank(self) -> int:
        """Return the dimension of the subspace the alignment was found in."""
        return self._rank

    @property
    def provenance(self) -> Provenance:
        """Return what produced it, over what data, under what seed."""
        return self._provenance

    def to_json(self) -> dict[str, JsonValue]:
        """Return the evidence as a JSON object with its tag.

        Returns
        -------
        dict[str, JsonValue]
            The tag, the accuracy, the rank and the nested provenance.
        """
        return {
            "kind": self.kind,
            "accuracy": self._accuracy,
            "rank": self._rank,
            "provenance": self._provenance.to_json(),
        }

    def _identity(self) -> tuple[float, int, Provenance]:
        """Return the three fields this evidence is equal on."""
        return (self._accuracy, self._rank, self._provenance)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same three fields.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not AlignmentEvidence:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the three fields this evidence is equal on."""
        return hash((AlignmentEvidence, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this evidence."""
        return (
            f"AlignmentEvidence({self._accuracy!r}, {self._rank!r}, "
            f"{self._provenance!r})"
        )


type Evidence = (
    AttributionEvidence | AblationEvidence | StatisticalEvidence | AlignmentEvidence
)
"""Specifies what an edge between two units rests on."""


def evidence_from_json(value: JsonValue) -> Evidence:
    """Return the evidence a JSON object describes.

    Parameters
    ----------
    value : JsonValue
        The object one variant's ``to_json`` produced.

    Returns
    -------
    Evidence
        The deserialized evidence.

    Raises
    ------
    TypeError
        If the value is not a JSON object, or a field is of the wrong type.
    ValueError
        If a field is missing or out of range, or the tag names no variant.
    """
    fields = an_object(value, "evidence")
    kind = a_name(fields, "kind")
    provenance = provenance_from_json(field(fields, "provenance"))

    if kind == "attribution":
        return AttributionEvidence(a_number(fields, "effect"), provenance)

    if kind == "ablation":
        return AblationEvidence(
            a_number(fields, "effect"), a_name(fields, "baseline"), provenance
        )

    if kind == "statistical":
        return StatisticalEvidence(
            a_number(fields, "statistic"), a_name(fields, "null"), provenance
        )

    if kind == "alignment":
        return AlignmentEvidence(
            a_number(fields, "accuracy"), a_count(fields, "rank"), provenance
        )

    msg = f"no evidence is tagged {kind!r}"
    raise ValueError(msg)


class CircuitEdge:
    """Connects two units with supporting evidence.

    Parameters
    ----------
    source : Unit
        The unit the influence runs from.
    target : Unit
        The unit it runs to.
    evidence : Evidence
        What establishes it.

    Raises
    ------
    TypeError
        If an endpoint is not a unit, or the evidence is not one of the declared
        variants.
    """

    __slots__ = ("_evidence", "_source", "_target")

    def __init__(self, source: Unit, target: Unit, evidence: Evidence) -> None:
        for endpoint in (source, target):
            if type(endpoint) is not Unit:
                msg = f"an edge runs between units, not {endpoint!r}"
                raise TypeError(msg)

        if type(evidence) not in {
            AttributionEvidence,
            AblationEvidence,
            StatisticalEvidence,
            AlignmentEvidence,
        }:
            msg = f"an edge's evidence is not one of the declared kinds: {evidence!r}"
            raise TypeError(msg)

        self._source = source
        self._target = target
        self._evidence = evidence

    @property
    def source(self) -> Unit:
        """Return the unit the influence runs from."""
        return self._source

    @property
    def target(self) -> Unit:
        """Return the unit it runs to."""
        return self._target

    @property
    def evidence(self) -> Evidence:
        """Return what establishes the edge."""
        return self._evidence

    def _identity(self) -> tuple[Unit, Unit, Evidence]:
        """Return the three fields an edge is equal on."""
        return (self._source, self._target, self._evidence)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same three fields.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not CircuitEdge:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the endpoints and the evidence."""
        return hash((CircuitEdge, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this edge."""
        return f"CircuitEdge({self._source!r}, {self._target!r}, {self._evidence!r})"
