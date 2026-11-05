"""Circuit faithfulness under ablation.

A circuit is assessed by scoring one metric, item by item, under ablations of
the published points it is drawn from, against two fixed endpoints: the clean
run, with nothing replaced, and the corrupt run, with every entry at every point
under study replaced. The normalized recovery of a run is where its mean score
lies between those two, zero at the corrupt endpoint and one at the clean one.
Sufficiency is the recovery of the circuit alone, with everything else replaced.

Completeness and minimality are the criteria of Wang, Variengien, Conmy,
Shlegeris and Steinhardt (2023), "Interpretability in the Wild: a Circuit for
Indirect Object Identification in GPT-2 Small", ICLR, in the section validating
the circuit. Completeness there compares removing a subset of the circuit from
the circuit against removing the same subset from the whole model, and is read
over a family of subsets. Here the family is every unit on its own, the whole
circuit, and a seeded number of random subsets. Minimality there asks, for each
unit, whether some subset of the rest of the circuit can be removed so that the
unit's own removal then matters. Here each unit is removed alone from the
circuit alone, which is that question with the empty subset, so a unit whose
work another unit of the circuit can take over reads as contributing nothing.
"""

from random import Random
from statistics import fmean
from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple, override

import torch

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi.nn.sites import Site
from lofi.probe._circuits import AblationEvidence, Circuit, Unit
from lofi.probe._instrument import instrument
from lofi.probe._interventions import Sparing, Within
from lofi.probe._provenance import Provenance
from lofi.stats import Bootstrap, EffectSize, paired_effect_size

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from torch import Tensor

    from lofi.nn.ports import Reading
    from lofi.probe._interventions import Intervention
    from lofi.probe._ports import Probed
    from lofi.stats import Estimator

__all__ = [
    "Ablation",
    "Completeness",
    "Contrast",
    "Endpoints",
    "Minimality",
    "RecoveryPoint",
    "Sufficiency",
    "Summary",
    "completeness",
    "endpoints",
    "minimality",
    "recovery",
    "sufficiency",
]


def recovery(patched: float, *, corrupt: float, clean: float) -> float:
    """Return where a patched run's metric lies between the corrupt and clean runs.

    Parameters
    ----------
    patched : float
        The metric of the run assessed.
    corrupt : float
        The metric of the run with everything replaced, where recovery is zero.
    clean : float
        The metric of the run with nothing replaced, where recovery is one.

    Returns
    -------
    float
        The patched metric less the corrupt one, over the clean metric less the
        corrupt one. Below zero where the patched run does worse than the corrupt
        one and above one where it does better than the clean one.

    Raises
    ------
    TypeError
        If a metric is not a number.
    ValueError
        If a metric is not finite, or the two endpoints coincide, which leaves
        nothing to lie between.
    """
    measured = ensure_finite("a patched run's metric", patched)
    floor = ensure_finite("the corrupt run's metric", corrupt)
    ceiling = ensure_finite("the clean run's metric", clean)

    if ceiling == floor:
        msg = (
            f"the clean and corrupt runs both score {ceiling!r}, so nothing lies "
            "between them"
        )
        raise ValueError(msg)

    return (measured - floor) / (ceiling - floor)


def _scores(field: str, scores: Sequence[float]) -> tuple[float, ...]:
    """Return one run's per-item scores, checked.

    Parameters
    ----------
    field : str
        What the scores are, for a message.
    scores : Sequence[float]
        One score per item.

    Returns
    -------
    tuple[float, ...]
        The scores, as finite floats.

    Raises
    ------
    TypeError
        If a score is not a number.
    ValueError
        If a score is not finite, or there are no scores.
    """
    held = tuple(ensure_finite(f"a score of {field}", score) for score in scores)

    if not held:
        msg = f"{field} scored no items"
        raise ValueError(msg)

    return held


class Endpoints:
    """Records the per-item scores of the clean run and of the corrupt run.

    Parameters
    ----------
    clean : Sequence[float]
        One score per item with nothing replaced.
    corrupt : Sequence[float]
        One score per item, the same items in the same order, with every entry at
        every point under study replaced.

    Raises
    ------
    TypeError
        If a score is not a number.
    ValueError
        If a score is not finite, if there are no items or the two runs scored
        different numbers of them, or if the two runs score alike on average, which
        leaves nothing to lie between.
    """

    __slots__ = ("_clean", "_corrupt")

    def __init__(self, clean: Sequence[float], corrupt: Sequence[float]) -> None:
        self._clean = _scores("the clean run", clean)
        self._corrupt = _scores("the corrupt run", corrupt)

        if len(self._clean) != len(self._corrupt):
            msg = (
                f"the clean run scored {len(self._clean)} items and the corrupt "
                f"run {len(self._corrupt)}"
            )
            raise ValueError(msg)

        if fmean(self._clean) == fmean(self._corrupt):
            msg = (
                f"the clean and corrupt runs both score {fmean(self._clean)!r} on "
                "average, so nothing lies between them"
            )
            raise ValueError(msg)

    @property
    def clean(self) -> tuple[float, ...]:
        """Return the per-item scores with nothing replaced."""
        return self._clean

    @property
    def corrupt(self) -> tuple[float, ...]:
        """Return the per-item scores with everything replaced."""
        return self._corrupt

    @property
    def clean_mean(self) -> float:
        """Return the mean score with nothing replaced."""
        return fmean(self._clean)

    @property
    def corrupt_mean(self) -> float:
        """Return the mean score with everything replaced."""
        return fmean(self._corrupt)

    @property
    def gap(self) -> float:
        """Return the clean mean less the corrupt mean, which is never zero."""
        return self.clean_mean - self.corrupt_mean

    def __len__(self) -> int:
        """Return how many items each run scored."""
        return len(self._clean)

    def _identity(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return the two fields the endpoints are equal on."""
        return (self._clean, self._corrupt)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same two runs."""
        if type(other) is not Endpoints:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the two runs."""
        return hash((Endpoints, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the two means and the item count."""
        return (
            f"Endpoints(clean={self.clean_mean!r}, corrupt={self.corrupt_mean!r}, "
            f"items={len(self._clean)})"
        )


class Contrast:
    """Records one run's per-item scores beside a reference run's.

    Parameters
    ----------
    scores : Sequence[float]
        One score per item under the run assessed.
    reference : Sequence[float]
        One score per item, the same items in the same order, under the run
        compared against.
    bootstrap : Bootstrap
        The resampling the standardized difference's interval is drawn with.
    estimator : Estimator
        Which standardized difference to report.

    Raises
    ------
    TypeError
        If a score is not a number, or the bootstrap is not one.
    ValueError
        If a score is not finite, if there are no items or the two runs scored
        different numbers of them, if the estimator is not one the statistics
        package declares, or if the estimator is undefined on the scores or on a
        resample of them.
    """

    __slots__ = ("_effect_size", "_reference", "_scores")

    def __init__(
        self,
        scores: Sequence[float],
        reference: Sequence[float],
        *,
        bootstrap: Bootstrap,
        estimator: Estimator = "paired-cohens-d",
    ) -> None:
        self._scores = _scores("the run assessed", scores)
        self._reference = _scores("the reference run", reference)

        if len(self._scores) != len(self._reference):
            msg = (
                f"the run assessed scored {len(self._scores)} items and the "
                f"reference run {len(self._reference)}"
            )
            raise ValueError(msg)

        self._effect_size = paired_effect_size(
            self._scores, self._reference, bootstrap, estimator=estimator
        )

    @property
    def scores(self) -> tuple[float, ...]:
        """Return the per-item scores under the run assessed."""
        return self._scores

    @property
    def reference(self) -> tuple[float, ...]:
        """Return the per-item scores under the run compared against."""
        return self._reference

    @property
    def mean(self) -> float:
        """Return the mean score under the run assessed."""
        return fmean(self._scores)

    @property
    def reference_mean(self) -> float:
        """Return the mean score under the run compared against."""
        return fmean(self._reference)

    @property
    def difference(self) -> float:
        """Return the mean of the run assessed less the mean of the reference."""
        return self.mean - self.reference_mean

    @property
    def effect_size(self) -> EffectSize:
        """Return the standardized per-item difference, with its interval."""
        return self._effect_size

    def __len__(self) -> int:
        """Return how many items each run scored."""
        return len(self._scores)

    def _identity(self) -> tuple[tuple[float, ...], tuple[float, ...], EffectSize]:
        """Return the three fields a contrast is equal on."""
        return (self._scores, self._reference, self._effect_size)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same runs and effect size."""
        if type(other) is not Contrast:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the runs and the effect size."""
        return hash((Contrast, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the two means and the standardized difference."""
        return (
            f"Contrast(mean={self.mean!r}, reference={self.reference_mean!r}, "
            f"effect_size={self._effect_size.value!r})"
        )


def _ordered(units: Iterable[Unit]) -> str:
    """Return units rendered in one fixed order.

    Parameters
    ----------
    units : Iterable[Unit]
        The units, in any order.

    Returns
    -------
    str
        The units' representations, sorted, as one list.
    """
    return repr(sorted(units, key=repr))


def _a_circuit(circuit: Circuit) -> Circuit:
    """Return a circuit, checked to be one with at least one unit.

    Parameters
    ----------
    circuit : Circuit
        The value to check.

    Raises
    ------
    TypeError
        If the value is not a circuit.
    ValueError
        If the circuit holds no unit.
    """
    if type(circuit) is not Circuit:
        msg = f"an assessment is of a Circuit, not {circuit!r}"
        raise TypeError(msg)

    if not circuit.units:
        msg = "an assessment is of a circuit with at least one unit"
        raise ValueError(msg)

    return circuit


def _endpoints_of(endpoints: Endpoints) -> Endpoints:
    """Return endpoints, checked to be some.

    Parameters
    ----------
    endpoints : Endpoints
        The value to check.

    Raises
    ------
    TypeError
        If the value is not endpoints.
    """
    if type(endpoints) is not Endpoints:
        msg = f"an assessment is against Endpoints, not {endpoints!r}"
        raise TypeError(msg)

    return endpoints


def _a_contrast(
    field: str, contrast: Contrast, reference: tuple[float, ...]
) -> Contrast:
    """Return a contrast, checked to be one against the expected reference.

    Parameters
    ----------
    field : str
        What the contrast is, for a message.
    contrast : Contrast
        The value to check.
    reference : tuple[float, ...]
        The per-item scores the contrast must be against.

    Raises
    ------
    TypeError
        If the value is not a contrast.
    ValueError
        If the contrast is against other scores than the expected ones.
    """
    if type(contrast) is not Contrast:
        msg = f"{field} is a Contrast, not {contrast!r}"
        raise TypeError(msg)

    if contrast.reference != reference:
        msg = f"{field} is against other scores than the run it is read beside"
        raise ValueError(msg)

    return contrast


def _a_provenance(provenance: Provenance) -> Provenance:
    """Return a provenance, checked to be one.

    Parameters
    ----------
    provenance : Provenance
        The value to check.

    Raises
    ------
    TypeError
        If the value is not a provenance.
    """
    if type(provenance) is not Provenance:
        msg = f"an assessment records a Provenance, not {provenance!r}"
        raise TypeError(msg)

    return provenance


class Sufficiency:
    """Records how far a circuit alone recovers the metric.

    Parameters
    ----------
    circuit : Circuit
        The circuit assessed.
    endpoints : Endpoints
        The clean and corrupt runs the recovery is read between.
    contrast : Contrast
        The run with everything outside the circuit replaced, against the clean run.
    baseline : str
        What the ablation put in place of every entry outside the circuit.
    provenance : Provenance
        What produced the assessment, over what data, under what seed.

    Raises
    ------
    TypeError
        If the circuit, the endpoints, the contrast or the provenance is not one.
    ValueError
        If the circuit holds no unit, the contrast is against other scores than the
        clean run's, or the baseline is empty.
    """

    __slots__ = ("_circuit", "_contrast", "_endpoints", "_evidence", "_recovery")

    def __init__(
        self,
        circuit: Circuit,
        endpoints: Endpoints,
        contrast: Contrast,
        *,
        baseline: str,
        provenance: Provenance,
    ) -> None:
        self._circuit = _a_circuit(circuit)
        self._endpoints = _endpoints_of(endpoints)
        self._contrast = _a_contrast(
            "the circuit's run", contrast, self._endpoints.clean
        )
        self._recovery = recovery(
            self._contrast.mean,
            corrupt=self._endpoints.corrupt_mean,
            clean=self._endpoints.clean_mean,
        )
        self._evidence = AblationEvidence(
            self._contrast.difference, baseline, _a_provenance(provenance)
        )

    @property
    def circuit(self) -> Circuit:
        """Return the circuit assessed."""
        return self._circuit

    @property
    def endpoints(self) -> Endpoints:
        """Return the clean and corrupt runs the recovery is read between."""
        return self._endpoints

    @property
    def contrast(self) -> Contrast:
        """Return the circuit's run against the clean run."""
        return self._contrast

    @property
    def recovery(self) -> float:
        """Return where the circuit's run lies between the corrupt and clean runs."""
        return self._recovery

    @property
    def evidence(self) -> AblationEvidence:
        """Return the change in the metric when everything outside was replaced."""
        return self._evidence

    def _identity(self) -> tuple[Circuit, Endpoints, Contrast, AblationEvidence]:
        """Return the four fields a sufficiency is equal on."""
        return (self._circuit, self._endpoints, self._contrast, self._evidence)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same four fields."""
        if type(other) is not Sufficiency:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the four fields the sufficiency is equal on."""
        return hash((Sufficiency, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the circuit's size, the recovery and the effect."""
        return (
            f"Sufficiency(units={len(self._circuit.units)}, "
            f"recovery={self._recovery!r}, effect={self._evidence.effect!r})"
        )


class Completeness:
    """Records how removing subsets of a circuit from it and from the model disagree.

    A subset removed from the circuit alone and the same subset removed from the
    whole model should score alike if the circuit is complete. The incompleteness
    score is the largest disagreement over the subsets tried, as a share of the gap
    between the endpoints, so zero is complete and larger is less so.

    Parameters
    ----------
    circuit : Circuit
        The circuit assessed.
    endpoints : Endpoints
        The clean and corrupt runs the disagreements are scaled by.
    knockouts : Mapping[frozenset[Unit], Contrast]
        For each subset tried, the run with the circuit alone less that subset,
        against the run with the whole model less that subset.
    baseline : str
        What the ablation put in place of every removed entry.
    provenance : Provenance
        What produced the assessment, over what data, under what seed.

    Raises
    ------
    TypeError
        If the circuit, the endpoints, a contrast or the provenance is not one.
    ValueError
        If the circuit holds no unit, no subset was tried, a subset is empty or
        holds a unit outside the circuit, or the baseline is empty.
    """

    __slots__ = ("_baseline", "_circuit", "_endpoints", "_knockouts", "_provenance")

    def __init__(
        self,
        circuit: Circuit,
        endpoints: Endpoints,
        knockouts: Mapping[frozenset[Unit], Contrast],
        *,
        baseline: str,
        provenance: Provenance,
    ) -> None:
        self._circuit = _a_circuit(circuit)
        self._endpoints = _endpoints_of(endpoints)
        held: dict[frozenset[Unit], Contrast] = {}

        for removed, contrast in knockouts.items():
            if not removed:
                msg = "a completeness assessment removes at least one unit at a time"
                raise ValueError(msg)

            if not removed <= self._circuit.units:
                msg = (
                    f"a completeness assessment removed {_ordered(removed)}, "
                    "which is not within the circuit"
                )
                raise ValueError(msg)

            if type(contrast) is not Contrast:
                msg = f"a knockout is a Contrast, not {contrast!r}"
                raise TypeError(msg)

            held[frozenset(removed)] = contrast

        if not held:
            msg = "a completeness assessment tries at least one subset"
            raise ValueError(msg)

        self._knockouts = MappingProxyType(held)
        self._baseline = ensure_name("an ablation baseline", baseline)
        self._provenance = _a_provenance(provenance)

    @property
    def circuit(self) -> Circuit:
        """Return the circuit assessed."""
        return self._circuit

    @property
    def endpoints(self) -> Endpoints:
        """Return the clean and corrupt runs the disagreements are scaled by."""
        return self._endpoints

    @property
    def knockouts(self) -> Mapping[frozenset[Unit], Contrast]:
        """Return each subset's contrast, as a read-only view."""
        return self._knockouts

    @property
    def baseline(self) -> str:
        """Return what the ablation put in place of every removed entry."""
        return self._baseline

    @property
    def provenance(self) -> Provenance:
        """Return what produced the assessment, over what data, under what seed."""
        return self._provenance

    def disagreement(self, removed: frozenset[Unit]) -> float:
        """Return how far one subset's two runs disagree, as a share of the gap.

        Parameters
        ----------
        removed : frozenset[Unit]
            The subset, one of those tried.

        Returns
        -------
        float
            The circuit's run less the model's run, over the clean mean less the
            corrupt mean. Signed, so a negative value is a circuit that does worse
            than the model without the subset.

        Raises
        ------
        KeyError
            If the subset was not tried.
        """
        return self._knockouts[removed].difference / self._endpoints.gap

    @property
    def incompleteness(self) -> float:
        """Return the largest disagreement in magnitude over the subsets tried."""
        return max(abs(self.disagreement(removed)) for removed in self._knockouts)

    @property
    def worst(self) -> frozenset[Unit]:
        """Return the subset whose two runs disagree most, the first tried on a tie."""
        return max(
            sorted(self._knockouts, key=_ordered),
            key=lambda removed: abs(self.disagreement(removed)),
        )

    def _identity(
        self,
    ) -> tuple[
        Circuit,
        Endpoints,
        frozenset[tuple[frozenset[Unit], Contrast]],
        str,
        Provenance,
    ]:
        """Return the five fields a completeness is equal on."""
        return (
            self._circuit,
            self._endpoints,
            frozenset(self._knockouts.items()),
            self._baseline,
            self._provenance,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same five fields."""
        if type(other) is not Completeness:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the five fields the completeness is equal on."""
        return hash((Completeness, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the circuit's size, how many subsets were tried and the score."""
        return (
            f"Completeness(units={len(self._circuit.units)}, "
            f"subsets={len(self._knockouts)}, "
            f"incompleteness={self.incompleteness!r})"
        )


class Minimality:
    """Records what each unit of a circuit contributes when it alone is removed.

    Parameters
    ----------
    circuit : Circuit
        The circuit assessed.
    endpoints : Endpoints
        The clean and corrupt runs the contributions are scaled by.
    removals : Mapping[Unit, Contrast]
        For each unit of the circuit, the run with the circuit alone less that
        unit, against the run with the circuit alone.
    baseline : str
        What the ablation put in place of every removed entry.
    provenance : Provenance
        What produced the assessment, over what data, under what seed.

    Raises
    ------
    TypeError
        If the circuit, the endpoints, a contrast or the provenance is not one.
    ValueError
        If the circuit holds no unit, the removals are not of exactly the circuit's
        units, two removals are against different runs, or the baseline is empty.
    """

    __slots__ = ("_baseline", "_circuit", "_endpoints", "_provenance", "_removals")

    def __init__(
        self,
        circuit: Circuit,
        endpoints: Endpoints,
        removals: Mapping[Unit, Contrast],
        *,
        baseline: str,
        provenance: Provenance,
    ) -> None:
        self._circuit = _a_circuit(circuit)
        self._endpoints = _endpoints_of(endpoints)

        if frozenset(removals) != self._circuit.units:
            msg = "a minimality assessment removes each unit of the circuit once"
            raise ValueError(msg)

        held: dict[Unit, Contrast] = {}
        reference: tuple[float, ...] | None = None

        for unit in sorted(removals, key=repr):
            contrast = removals[unit]

            if type(contrast) is not Contrast:
                msg = f"a removal is a Contrast, not {contrast!r}"
                raise TypeError(msg)

            if reference is None:
                reference = contrast.reference

            held[unit] = _a_contrast(f"the removal of {unit!r}", contrast, reference)

        self._removals = MappingProxyType(held)
        self._baseline = ensure_name("an ablation baseline", baseline)
        self._provenance = _a_provenance(provenance)

    @property
    def circuit(self) -> Circuit:
        """Return the circuit assessed."""
        return self._circuit

    @property
    def endpoints(self) -> Endpoints:
        """Return the clean and corrupt runs the contributions are scaled by."""
        return self._endpoints

    @property
    def removals(self) -> Mapping[Unit, Contrast]:
        """Return each unit's contrast, as a read-only view."""
        return self._removals

    @property
    def baseline(self) -> str:
        """Return what the ablation put in place of every removed entry."""
        return self._baseline

    @property
    def provenance(self) -> Provenance:
        """Return what produced the assessment, over what data, under what seed."""
        return self._provenance

    def contribution(self, unit: Unit) -> float:
        """Return what removing one unit alone costs, as a share of the gap.

        Parameters
        ----------
        unit : Unit
            The unit, one of the circuit's.

        Returns
        -------
        float
            The circuit's run less the run without the unit, over the clean mean
            less the corrupt mean. Positive where the unit helps.

        Raises
        ------
        KeyError
            If the unit is not one of the circuit's.
        """
        removal = self._removals[unit]

        return (removal.reference_mean - removal.mean) / self._endpoints.gap

    def redundant(self, tolerance: float) -> frozenset[Unit]:
        """Return the units whose removal alone moves the metric less than a tolerance.

        Parameters
        ----------
        tolerance : float
            The share of the gap below which a contribution in magnitude counts as
            nothing.

        Returns
        -------
        frozenset[Unit]
            The units that contribute less than that.

        Raises
        ------
        TypeError
            If the tolerance is not a number.
        ValueError
            If the tolerance is not finite or is negative.
        """
        bound = ensure_finite("a contribution tolerance", tolerance)

        if bound < 0.0:
            msg = f"a contribution tolerance is negative: {tolerance!r}"
            raise ValueError(msg)

        return frozenset(
            unit for unit in self._removals if abs(self.contribution(unit)) < bound
        )

    def _identity(
        self,
    ) -> tuple[Circuit, Endpoints, frozenset[tuple[Unit, Contrast]], str, Provenance]:
        """Return the five fields a minimality is equal on."""
        return (
            self._circuit,
            self._endpoints,
            frozenset(self._removals.items()),
            self._baseline,
            self._provenance,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same five fields."""
        if type(other) is not Minimality:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the five fields the minimality is equal on."""
        return hash((Minimality, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the circuit's size and each unit's contribution."""
        contributions = [
            round(self.contribution(unit), 4)
            for unit in sorted(self._removals, key=repr)
        ]

        return (
            f"Minimality(units={len(self._circuit.units)}, "
            f"contributions={contributions!r})"
        )


class RecoveryPoint(NamedTuple):
    """One circuit's place on a recovery curve read over circuit size.

    Attributes
    ----------
    size : int
        How many units the circuit holds.
    recovery : float
        Where the circuit's run lies between the corrupt and clean runs.
    effect_size : EffectSize
        The standardized per-item difference against the clean run, with the
        interval the bootstrap drew around it.
    """

    size: int
    recovery: float
    effect_size: EffectSize


def _one_setting(records: tuple[Sufficiency, ...]) -> None:
    """Refuse assessments that were not measured under one common setting.

    A recovery says where a run's mean score lies between two endpoints, so two
    recoveries are points on one curve only if they were read between the same
    endpoints, under the same ablation baseline and over the same data. Records
    that disagree on any of the three still plot as a curve, but the curve is an
    artifact of the disagreement rather than of circuit size.

    Parameters
    ----------
    records : tuple[Sufficiency, ...]
        The assessments to aggregate, at least one.

    Raises
    ------
    ValueError
        If the records were not all read between the same endpoints, under the
        same baseline, or over the same dataset.
    """
    between = {record.endpoints for record in records}

    if len(between) > 1:
        rendered = sorted(repr(pair) for pair in between)
        msg = (
            "a summary reads one curve, so its records are read between the same "
            f"endpoints, not {rendered!r}"
        )
        raise ValueError(msg)

    baselines = {record.evidence.baseline for record in records}

    if len(baselines) > 1:
        msg = (
            "a summary reads one curve, so its records share an ablation "
            f"baseline, not {sorted(baselines)!r}"
        )
        raise ValueError(msg)

    datasets = {record.evidence.provenance.dataset for record in records}

    if len(datasets) > 1:
        msg = (
            "a summary reads one curve, so its records run over one dataset, "
            f"not {sorted(datasets)!r}"
        )
        raise ValueError(msg)


class Summary:
    """Reads the sufficiency of several circuits as a curve over circuit size.

    A circuit's recovery only means something beside the size it was reached at,
    since a larger circuit recovers at least as much by holding more of the model.
    What the summary reports is therefore the curve: one point per circuit, each
    carrying the size, the recovery there and the standardized difference with its
    interval. The points are ordered by size alone, with the circuit's identifier
    breaking ties, so the curve is a function of the records and not of the order
    they were handed in. A recovery is published as it was measured, below zero
    where the circuit did worse than the corrupt run and above one where it did
    better than the clean one.

    Parameters
    ----------
    records : Iterable[Sufficiency]
        The assessments, in any order. All of them must be read between the same
        endpoints, under the same ablation baseline and over the same dataset,
        since otherwise their recoveries are not on one curve.

    Raises
    ------
    TypeError
        If a member is not a sufficiency.
    ValueError
        If there are no assessments, or they were not all read between the same
        endpoints, under the same baseline, or over the same dataset.
    """

    __slots__ = ("_records",)

    def __init__(self, records: Iterable[Sufficiency]) -> None:
        held = tuple(records)

        for record in held:
            if type(record) is not Sufficiency:
                msg = f"a summary aggregates Sufficiency records, not {record!r}"
                raise TypeError(msg)

        if not held:
            msg = "a summary aggregates at least one assessment"
            raise ValueError(msg)

        _one_setting(held)
        self._records = tuple(
            sorted(
                held, key=lambda record: (len(record.circuit.units), record.circuit.id)
            )
        )

    @property
    def records(self) -> tuple[Sufficiency, ...]:
        """Return the assessments, by circuit size and then by circuit identifier."""
        return self._records

    @property
    def curve(self) -> tuple[RecoveryPoint, ...]:
        """Return each circuit's size, its measured recovery and its effect size."""
        return tuple(
            RecoveryPoint(
                len(record.circuit.units),
                record.recovery,
                record.contrast.effect_size,
            )
            for record in self._records
        )

    def __len__(self) -> int:
        """Return how many circuits were assessed."""
        return len(self._records)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value aggregates the same assessments."""
        if type(other) is not Summary:
            return NotImplemented

        return self._records == other._records

    @override
    def __hash__(self) -> int:
        """Return a hash over the assessments."""
        return hash((Summary, self._records))

    @override
    def __repr__(self) -> str:
        """Return how many circuits were assessed and the recovery at each size."""
        reached = [(point.size, round(point.recovery, 4)) for point in self.curve]

        return f"Summary(circuits={len(self._records)}, curve={reached!r})"


def _checked_units(field: str, units: Iterable[Unit]) -> frozenset[Unit]:
    """Return units, checked to be some.

    Parameters
    ----------
    field : str
        What the units are, for a message.
    units : Iterable[Unit]
        The units, in any order and with repeats.

    Returns
    -------
    frozenset[Unit]
        The units, each once.

    Raises
    ------
    TypeError
        If a member is not a unit.
    """
    held = frozenset(units)

    for unit in held:
        if type(unit) is not Unit:
            msg = f"{field} are units, not {unit!r}"
            raise TypeError(msg)

    return held


class Ablation[ReadingT: Reading]:
    """Scores one metric over a component with entries at chosen points replaced.

    Parameters
    ----------
    model : Probed[ReadingT]
        The component the points are published by.
    sites : Iterable[Site]
        The points under study, each of which must permit patching. A run with
        everything replaced replaces every entry at each of them.
    metric : Callable[[], Tensor]
        One pass over a fixed batch of items, returning one score per item as a
        tensor of rank one, higher being better.
    intervention : Intervention
        What replaces a removed entry.
    dataset : str
        The handle of the data the pass runs over.

    Raises
    ------
    TypeError
        If a point is not a site.
    ValueError
        If no point is named, or the dataset handle is empty.
    """

    __slots__ = ("_dataset", "_intervention", "_metric", "_model", "_sites")

    def __init__(
        self,
        model: Probed[ReadingT],
        /,
        *,
        sites: Iterable[Site],
        metric: Callable[[], Tensor],
        intervention: Intervention,
        dataset: str,
    ) -> None:
        held = frozenset(sites)

        for site in held:
            if not isinstance(site, Site):
                msg = f"a point under study is not a Site: {site!r}"
                raise TypeError(msg)

        if not held:
            msg = "an ablation studies at least one point"
            raise ValueError(msg)

        self._model = model
        self._sites = held
        self._metric = metric
        self._intervention = intervention
        self._dataset = ensure_name("an ablation's dataset", dataset)

    @property
    def sites(self) -> frozenset[Site]:
        """Return the points under study."""
        return self._sites

    @property
    def intervention(self) -> Intervention:
        """Return what replaces a removed entry."""
        return self._intervention

    @property
    def baseline(self) -> str:
        """Return the tag of what replaces a removed entry."""
        return self._intervention.tag

    @property
    def dataset(self) -> str:
        """Return the handle of the data the pass runs over."""
        return self._dataset

    def _scored(self, patch: Mapping[Site, Intervention]) -> tuple[float, ...]:
        """Run the pass under a patch and return its per-item scores.

        Parameters
        ----------
        patch : Mapping[Site, Intervention]
            What replaces the value at each patched point.

        Returns
        -------
        tuple[float, ...]
            One finite score per item.

        Raises
        ------
        ValueError
            If a patched point is not published or does not permit patching, if the
            pass returns other than one score per item, or if a score is not finite.
        """
        with torch.no_grad(), instrument(self._model, patch=patch) as region:
            scored = region(self._metric)

        if scored.ndim != 1:
            msg = (
                "a metric pass returns one score per item, not a tensor of shape "
                f"{tuple(scored.shape)}"
            )
            raise ValueError(msg)

        return _scores("the pass", scored.tolist())

    def _by_site(self, units: frozenset[Unit]) -> dict[Site, list[Unit]]:
        """Return the units gathered by the point each names, checked to be under study.

        Parameters
        ----------
        units : frozenset[Unit]
            The units.

        Raises
        ------
        ValueError
            If a unit names a point not under study.
        """
        gathered: dict[Site, list[Unit]] = {}

        for unit in units:
            if unit.site not in self._sites:
                msg = f"{unit!r} is at a point this ablation does not study"
                raise ValueError(msg)

            gathered.setdefault(unit.site, []).append(unit)

        return gathered

    def only(self, units: Iterable[Unit]) -> tuple[float, ...]:
        """Score the pass with everything at the points replaced except the units.

        Parameters
        ----------
        units : Iterable[Unit]
            The units whose entries keep their values. With none, every entry at
            every point under study is replaced.

        Returns
        -------
        tuple[float, ...]
            One score per item.

        Raises
        ------
        TypeError
            If a member is not a unit.
        ValueError
            If a unit names a point not under study, or the pass cannot be scored.
        """
        kept = self._by_site(_checked_units("the units kept", units))
        patch = {
            site: Sparing(kept.get(site, ()), self._intervention)
            for site in self._sites
        }

        return self._scored(patch)

    def without(self, units: Iterable[Unit]) -> tuple[float, ...]:
        """Score the pass with the units' entries replaced and nothing else.

        Parameters
        ----------
        units : Iterable[Unit]
            The units whose entries are replaced. With none, nothing is.

        Returns
        -------
        tuple[float, ...]
            One score per item.

        Raises
        ------
        TypeError
            If a member is not a unit.
        ValueError
            If a unit names a point not under study, or the pass cannot be scored.
        """
        removed = self._by_site(_checked_units("the units removed", units))
        patch = {
            site: Within(held, self._intervention) for site, held in removed.items()
        }

        return self._scored(patch)

    def leave_one_out(self) -> Mapping[Site, tuple[float, ...]]:
        """Score the pass once per point under study, with that point replaced whole.

        Each pass opens its own region over the one point it replaces, so the cost
        grows with the number of points and not with its square.

        Returns
        -------
        Mapping[Site, tuple[float, ...]]
            One score per item for each point, keyed by the point replaced.

        Raises
        ------
        ValueError
            If a pass cannot be scored.
        """
        return MappingProxyType(
            {
                site: self._scored({site: self._intervention})
                for site in sorted(self._sites, key=repr)
            }
        )

    @override
    def __repr__(self) -> str:
        """Return how many points are under study, the baseline and the dataset."""
        return (
            f"Ablation(sites={len(self._sites)}, baseline={self.baseline!r}, "
            f"dataset={self._dataset!r})"
        )


def endpoints[ReadingT: Reading](ablation: Ablation[ReadingT], /) -> Endpoints:
    """Score the clean run and the corrupt run.

    Parameters
    ----------
    ablation : Ablation[ReadingT]
        The metric, the points under study and what replaces a removed entry.

    Returns
    -------
    Endpoints
        The per-item scores with nothing replaced and with everything replaced,
        from one pass each.

    Raises
    ------
    ValueError
        If a pass cannot be scored, or the two runs score alike on average.
    """
    return Endpoints(ablation.without(()), ablation.only(()))


def _a_bootstrap(bootstrap: Bootstrap) -> Bootstrap:
    """Return a bootstrap, checked to be one.

    Parameters
    ----------
    bootstrap : Bootstrap
        The value to check.

    Raises
    ------
    TypeError
        If the value is not a bootstrap.
    """
    if not isinstance(bootstrap, Bootstrap):
        msg = f"an assessment's bootstrap is not one: {bootstrap!r}"
        raise TypeError(msg)

    return bootstrap


def sufficiency[ReadingT: Reading](
    ablation: Ablation[ReadingT],
    circuit: Circuit,
    /,
    *,
    endpoints: Endpoints,
    bootstrap: Bootstrap,
    estimator: Estimator = "paired-cohens-d",
) -> Sufficiency:
    """Score the circuit alone, with everything else replaced.

    Parameters
    ----------
    ablation : Ablation[ReadingT]
        The metric, the points under study and what replaces a removed entry.
    circuit : Circuit
        The circuit to assess, drawn from the points under study.
    endpoints : Endpoints
        The clean and corrupt runs of the same ablation.
    bootstrap : Bootstrap
        The resampling the effect size's interval is drawn with.
    estimator : Estimator
        Which standardized difference to report.

    Returns
    -------
    Sufficiency
        The circuit's recovery between the endpoints and its per-item contrast
        against the clean run, from one pass.

    Raises
    ------
    TypeError
        If the circuit, the endpoints or the bootstrap is not one.
    ValueError
        If the circuit holds no unit or names a point not under study, if the pass
        cannot be scored, or if the estimator is undefined on the scores.
    """
    assessed = _a_circuit(circuit)
    between = _endpoints_of(endpoints)
    resampling = _a_bootstrap(bootstrap)
    contrast = Contrast(
        ablation.only(assessed.units),
        between.clean,
        bootstrap=resampling,
        estimator=estimator,
    )

    return Sufficiency(
        assessed,
        between,
        contrast,
        baseline=ablation.baseline,
        provenance=Provenance("sufficiency", ablation.dataset, None),
    )


def _subsets(
    units: frozenset[Unit], draws: int, seed: int
) -> tuple[frozenset[Unit], ...]:
    """Return the subsets a completeness assessment tries.

    Parameters
    ----------
    units : frozenset[Unit]
        The circuit's units.
    draws : int
        How many random subsets to draw, each unit in or out with equal chance.
    seed : int
        The seed the draws run under.

    Returns
    -------
    tuple[frozenset[Unit], ...]
        Every unit on its own, the whole circuit, and the distinct non-empty
        random subsets, in that order without repeats.
    """
    ordered = sorted(units, key=repr)
    family: dict[frozenset[Unit], None] = {frozenset({unit}): None for unit in ordered}
    family[frozenset(ordered)] = None
    random = Random(seed)

    for _ in range(draws):
        drawn = frozenset(unit for unit in ordered if random.random() < 0.5)

        if drawn:
            family[drawn] = None

    return tuple(family)


def completeness[ReadingT: Reading](
    ablation: Ablation[ReadingT],
    circuit: Circuit,
    /,
    *,
    endpoints: Endpoints,
    bootstrap: Bootstrap,
    estimator: Estimator = "paired-cohens-d",
    draws: int,
    seed: int,
) -> Completeness:
    """Compare removing subsets of the circuit from it and from the whole model.

    Parameters
    ----------
    ablation : Ablation[ReadingT]
        The metric, the points under study and what replaces a removed entry.
    circuit : Circuit
        The circuit to assess, drawn from the points under study.
    endpoints : Endpoints
        The clean and corrupt runs of the same ablation.
    bootstrap : Bootstrap
        The resampling each effect size's interval is drawn with.
    estimator : Estimator
        Which standardized difference to report.
    draws : int
        How many random subsets to try, beyond every unit on its own and the whole
        circuit.
    seed : int
        The seed the random subsets are drawn under.

    Returns
    -------
    Completeness
        For each subset tried, the circuit alone less the subset against the whole
        model less the subset, from two passes per subset.

    Raises
    ------
    TypeError
        If the circuit, the endpoints or the bootstrap is not one, or the draw count
        or the seed is not an integer.
    ValueError
        If the circuit holds no unit or names a point not under study, the draw
        count or the seed is negative, a pass cannot be scored, or the estimator is
        undefined on a subset's scores.
    """
    assessed = _a_circuit(circuit)
    between = _endpoints_of(endpoints)
    resampling = _a_bootstrap(bootstrap)
    tried = _subsets(
        assessed.units,
        ensure_non_negative("a completeness draw count", draws),
        ensure_non_negative("a completeness seed", seed),
    )
    knockouts = {
        removed: Contrast(
            ablation.only(assessed.units - removed),
            ablation.without(removed),
            bootstrap=resampling,
            estimator=estimator,
        )
        for removed in tried
    }

    return Completeness(
        assessed,
        between,
        knockouts,
        baseline=ablation.baseline,
        provenance=Provenance("completeness", ablation.dataset, seed),
    )


def minimality[ReadingT: Reading](
    ablation: Ablation[ReadingT],
    circuit: Circuit,
    /,
    *,
    endpoints: Endpoints,
    bootstrap: Bootstrap,
    estimator: Estimator = "paired-cohens-d",
) -> Minimality:
    """Remove each unit of the circuit alone from the circuit alone.

    Parameters
    ----------
    ablation : Ablation[ReadingT]
        The metric, the points under study and what replaces a removed entry.
    circuit : Circuit
        The circuit to assess, drawn from the points under study.
    endpoints : Endpoints
        The clean and corrupt runs of the same ablation.
    bootstrap : Bootstrap
        The resampling each effect size's interval is drawn with.
    estimator : Estimator
        Which standardized difference to report.

    Returns
    -------
    Minimality
        For each unit, the circuit alone less that unit against the circuit alone,
        from one pass for the circuit and one per unit.

    Raises
    ------
    TypeError
        If the circuit, the endpoints or the bootstrap is not one.
    ValueError
        If the circuit holds no unit or names a point not under study, a pass
        cannot be scored, or the estimator is undefined on a unit's scores.
    """
    assessed = _a_circuit(circuit)
    between = _endpoints_of(endpoints)
    resampling = _a_bootstrap(bootstrap)
    whole = ablation.only(assessed.units)
    removals = {
        unit: Contrast(
            ablation.only(assessed.units - {unit}),
            whole,
            bootstrap=resampling,
            estimator=estimator,
        )
        for unit in sorted(assessed.units, key=repr)
    }

    return Minimality(
        assessed,
        between,
        removals,
        baseline=ablation.baseline,
        provenance=Provenance("minimality", ablation.dataset, None),
    )
