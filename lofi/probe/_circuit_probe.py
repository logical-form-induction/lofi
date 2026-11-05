"""Document-driven circuit discovery, pruning, and validation.

The circuit arm consumes a held-out pass that publishes both a computation graph
and the endpoint experiment needed to interrogate it. The endpoint capability is
explicit: it names the observed component, the clean and counterfactual inputs,
the model forward, and the receiver-local unit of every edge. Thus the probe does
not infer private tensor layouts or module paths from a trained point.
"""

from statistics import fmean
from typing import (
    TYPE_CHECKING,
    Final,
    Protocol,
    runtime_checkable,
)

import torch

from lofi._checks import ensure_name
from lofi.nn.sites import Access
from lofi.probe._acdc import PruningBudget, acdc, matched_cardinality_jaccard
from lofi.probe._attribution import attribute_edges
from lofi.probe._boundary import CircuitProbeSettings, CircuitProbeSpec
from lofi.probe._circuitry import MatchedRandomCircuits, circuitry
from lofi.probe._circuits import Circuit
from lofi.probe._faithfulness import (
    Ablation,
    completeness,
    endpoints,
    minimality,
    recovery,
    sufficiency,
)
from lofi.probe._instrument import instrument
from lofi.probe._interventions import Resample
from lofi.probe._probing import (
    GradientMode,
    GraphHeldOutPass,
    ProbeBuildContext,
    ProbeValue,
    UnpublishedSiteError,
)
from lofi.probe._provenance import Provenance
from lofi.probe._sensitivity import AblationMetric, ablation_sensitivity
from lofi.registry import ComponentPlugin
from lofi.stats import BenjaminiHochberg, Bootstrap

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from torch import Tensor

    from lofi.nn.ports import Reading
    from lofi.nn.sites import Site, SiteSpec
    from lofi.probe._circuits import Unit
    from lofi.probe._graph import ComputationGraph, GraphEdge
    from lofi.probe._interventions import Intervention
    from lofi.probe._ports import Probed
    from lofi.probe._probing import TrainedPoint
    from lofi.probe._values import Trace

__all__ = [
    "CircuitExperimentPass",
    "CircuitProbe",
    "CircuitProbeSettings",
    "CircuitProbeSpec",
]


@runtime_checkable
class CircuitExperimentPass(GraphHeldOutPass, Protocol):
    """A graph pass carrying the public endpoint experiment circuit methods need."""

    def circuit_component(self) -> str:
        """Return the trained-point component publishing the graph's sites."""
        ...

    def circuit_inputs(self) -> tuple[Tensor, Tensor]:
        """Return corresponding clean and counterfactual endpoint inputs."""
        ...

    def circuit_forward(self, endpoint: Tensor, /) -> Tensor:
        """Return one behavioral logit difference per item at an endpoint."""
        ...

    def circuit_targets(self) -> Mapping[GraphEdge[object], Unit]:
        """Return one nonoverlapping receiver-local unit for every graph edge."""
        ...

    def circuit_order(self) -> tuple[GraphEdge[object], ...]:
        """Return every graph edge once in total reverse-topological order."""
        ...


def _ordinary(
    measure: str, value: float, count: int, site: Site | None = None
) -> ProbeValue:
    """Return one non-inferential probe value."""
    return ProbeValue(measure, site, value, count, None, None, None, None, None, None)


def _experiment_pass(
    candidate: GraphHeldOutPass, position: int, /
) -> CircuitExperimentPass:
    """Return a graph pass that publishes the endpoint experiment."""
    if not isinstance(candidate, CircuitExperimentPass):
        msg = (
            f"graph held-out pass {position} publishes no circuit endpoint, forward, "
            "receiver-target map, or edge order"
        )
        raise TypeError(msg)

    return candidate


def _counterfactual_trace(
    model: Probed[Reading],
    sites: frozenset[Site],
    forward: Callable[[Tensor], Tensor],
    counterfactual: Tensor,
) -> Trace:
    """Return the closed counterfactual readings at the circuit's sites."""
    with torch.no_grad(), instrument(model, observe=sites) as region:
        region(lambda: forward(counterfactual))

    return region.trace


class CircuitProbe:
    """Run the complete circuit arm over each graph-capable held-out pass."""

    __slots__ = ("_role", "_settings")

    def __init__(self, role: str, settings: CircuitProbeSettings) -> None:
        self._role = ensure_name("a circuit probe role", role)
        self._settings = settings

    def admit(self, published: Mapping[Site, SiteSpec], /) -> None:
        """Refuse a point with no differentiable observable-and-patchable site."""
        if not any(
            spec.differentiable
            and Access.OBSERVE in spec.access
            and Access.PATCH in spec.access
            for spec in published.values()
        ):
            msg = (
                "a circuit probe requires a differentiable published point permitting "
                "observation and patching"
            )
            raise UnpublishedSiteError(msg)

    def _run_pass(
        self, point: TrainedPoint, held: CircuitExperimentPass, position: int, /
    ) -> tuple[ProbeValue, ...]:
        """Run every circuit method over one endpoint experiment."""
        held(GradientMode.ENABLED)
        graph: ComputationGraph[object] = held.computation_graph()
        component = ensure_name(
            "a circuit experiment component", held.circuit_component()
        )

        try:
            model = point.components[component]
        except KeyError:
            msg = f"a circuit experiment names absent component {component!r}"
            raise ValueError(msg) from None

        clean, counterfactual = held.circuit_inputs()
        forward = held.circuit_forward
        targets = held.circuit_targets()
        order = held.circuit_order()
        dataset = f"{self._role}/pass-{position}"
        attribution = attribute_edges(
            model,
            graph,
            targets=targets,
            clean=clean,
            counterfactual=counterfactual,
            forward=lambda endpoint: forward(endpoint).sum(),
            steps=self._settings.attribution_steps,
            dataset=dataset,
        )
        result = acdc(
            model,
            graph,
            targets=targets,
            order=order,
            clean=clean,
            counterfactual=counterfactual,
            forward=forward,
            divergence=lambda reference, trial: (reference - trial).abs().mean(),
            threshold=self._settings.acdc_threshold,
            budget=PruningBudget(self._settings.acdc_trials),
            data_provenance=Provenance("held-out endpoint", dataset, None),
            policy_provenance=Provenance(
                "budgeted-acdc", self._role, self._settings.seed
            ),
        )
        effects = tuple(abs(attribution[edge].effect) for edge in attribution)
        rows = [
            _ordinary(
                "eap-ig/mean-absolute-effect",
                0.0 if not effects else fmean(effects),
                len(effects),
            ),
            _ordinary(
                "acdc/retained-share",
                0.0 if not graph.edges else len(result.retained) / len(graph.edges),
                len(graph.edges),
            ),
            _ordinary(
                "acdc/scheduling-residual", result.scheduling_residual, len(graph.edges)
            ),
            _ordinary(
                "acdc/eap-ig-jaccard",
                matched_cardinality_jaccard(result, attribution),
                len(graph.edges),
            ),
        ]
        circuit = Circuit(targets[edge] for edge in result.retained)

        if not circuit.units:
            msg = (
                "ACDC retained no unit, so the declared faithfulness arm has no circuit"
            )
            raise ValueError(msg)

        sites = frozenset(unit.site for unit in targets.values())
        matched = _counterfactual_trace(model, sites, forward, counterfactual)
        ablation = Ablation(
            model,
            sites=sites,
            metric=lambda: forward(clean),
            intervention=Resample(matched, self._settings.seed),
            dataset=dataset,
        )
        limits = endpoints(ablation)
        bootstrap = Bootstrap(
            unit="item",
            replicates=self._settings.bootstrap_replicates,
            seed=self._settings.seed,
            level=self._settings.bootstrap_level,
        )
        sufficient = sufficiency(
            ablation,
            circuit,
            endpoints=limits,
            bootstrap=bootstrap,
            estimator="paired-cliffs-delta",
        )
        complete = completeness(
            ablation,
            circuit,
            endpoints=limits,
            bootstrap=bootstrap,
            estimator="paired-cliffs-delta",
            draws=self._settings.completeness_draws,
            seed=self._settings.seed,
        )
        minimal = minimality(
            ablation,
            circuit,
            endpoints=limits,
            bootstrap=bootstrap,
            estimator="paired-cliffs-delta",
        )
        rows.extend(
            (
                _ordinary("faithfulness/sufficiency", sufficient.recovery, len(limits)),
                _ordinary(
                    "faithfulness/incompleteness",
                    complete.incompleteness,
                    len(complete.knockouts),
                ),
                _ordinary(
                    "faithfulness/minimum-contribution",
                    min(minimal.contribution(unit) for unit in circuit.units),
                    len(circuit.units),
                ),
            )
        )
        null_report = circuitry(
            ablation,
            circuit,
            endpoints=limits,
            null=MatchedRandomCircuits(
                targets.values(),
                replicates=self._settings.null_replicates,
                seed=self._settings.seed,
            ),
        )
        null_results = tuple(
            null_report.results[criterion] for criterion in null_report.results
        )
        decisions = BenjaminiHochberg().apply(
            tuple(entry.p_value for entry in null_results),
            self._settings.fdr_alpha,
            self._settings.dependence,
        )
        family = f"{self._role}/matched-circuitry"

        for index, entry in enumerate(null_results):
            rows.append(
                ProbeValue(
                    f"circuitry/{entry.criterion.value}",
                    None,
                    entry.statistic,
                    self._settings.null_replicates,
                    entry.p_value.provenance,
                    entry.p_value.value,
                    decisions.adjusted[index],
                    decisions.reject[index],
                    family,
                    decisions.dependence,
                )
            )

        def sensitivity_value(
            intervention: Intervention, metric: AblationMetric
        ) -> float:
            transformed = (
                forward
                if metric is AblationMetric.LOGIT_DIFFERENCE
                else lambda endpoint: forward(endpoint).sigmoid()
            )
            arm = Ablation(
                model,
                sites=sites,
                metric=lambda: transformed(clean),
                intervention=intervention,
                dataset=dataset,
            )
            arm_limits = endpoints(arm)

            return recovery(
                fmean(arm.only(circuit.units)),
                corrupt=arm_limits.corrupt_mean,
                clean=arm_limits.clean_mean,
            )

        sensitivity = ablation_sensitivity(
            matched,
            sensitivity_value,
            lambda _metric, value: value >= self._settings.sensitivity_threshold,
            seed=self._settings.seed,
            include_zero=self._settings.include_zero,
        )

        rows.extend(
            _ordinary(
                f"sensitivity/{cell.semantics.value}/{cell.metric.value}",
                cell.value,
                len(limits),
            )
            for cell in sensitivity.cells
        )

        return tuple(rows)

    def run(self, point: TrainedPoint, /) -> tuple[ProbeValue, ...]:
        """Return circuit rows over every held-out graph endpoint."""
        rows: list[ProbeValue] = []

        for position, candidate in enumerate(point.passes):
            if not isinstance(candidate, GraphHeldOutPass):
                msg = f"held-out pass {position} publishes no computation graph"
                raise TypeError(msg)
            rows.extend(
                self._run_pass(point, _experiment_pass(candidate, position), position)
            )

        return tuple(rows)


def _build(
    settings: CircuitProbeSettings, context: ProbeBuildContext, /
) -> CircuitProbe:
    """Build a circuit probe after validating every pass capability."""
    for position, candidate in enumerate(context.graph_passes()):
        _experiment_pass(candidate, position)

    return CircuitProbe(context.role, settings)


PLUGIN: Final = ComponentPlugin(spec=CircuitProbeSpec, build=_build)
"""The plugin the ``circuit`` probe kind registers."""
