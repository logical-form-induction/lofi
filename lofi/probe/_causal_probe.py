"""Document-driven causal-abstraction sweeps, joint tests, and controls.

The causal arm receives a model-ready population from a held-out pass. The pass
owns tokenization and model input construction; the declaration owns the search,
objective, budget, null, joint, control, and bucketing policies. This separates
the experiment from both private model layouts and paper-specific run glue.
"""

from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Final,
    Protocol,
    runtime_checkable,
)

from lofi._checks import ensure_name
from lofi.nn.sites import Access
from lofi.probe._boundary import (
    CausalProbeSettings,
    CausalProbeSpec,
    CausalSiteDecl,
)
from lofi.probe._bucketing import AbstractionCase, bucket_abstraction
from lofi.probe._causal import variable_label
from lofi.probe._controls import (
    BracketingDelta,
    BracketingUnavailable,
    ControlCell,
    ControlReport,
    IIAPopulation,
    ScrambledDelta,
    delta_iia,
    dormant_pathway,
    matched_binary_bracketings,
    scramble_types,
)
from lofi.probe._das import DistributedAlignment
from lofi.probe._joint import JointDistributedAlignment, select_joint_layer
from lofi.probe._node_vocabulary import node_vocabulary_row
from lofi.probe._probing import (
    CausalHeldOutPass,
    ProbeBuildContext,
    ProbeValue,
    RecordedProbe,
    UnpublishedSiteError,
)
from lofi.probe._sweep import Preregistration, Subgrid, sweep
from lofi.registry import ComponentPlugin
from lofi.stats import BenjaminiHochberg, Bootstrap

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.nn.ports import Reading
    from lofi.nn.sites import Site, SiteSpec
    from lofi.probe._bucketing import AbstractionFeatures
    from lofi.probe._causal import HighLevelVariable
    from lofi.probe._controls import BinaryBracketing
    from lofi.probe._counterfactuals import CounterfactualGeneration
    from lofi.probe._ports import (
        JointSubject,
        ModelReadyCausalPair,
        ModelReadyJointCausalPair,
        Subject,
    )
    from lofi.probe._pricing import CostModel
    from lofi.probe._probing import TrainedPoint
    from lofi.probe._sweep import SweepReport

__all__ = [
    "CausalExperimentPass",
    "CausalProbe",
    "CausalProbeSettings",
    "CausalProbeSpec",
    "CausalSiteDecl",
]


@runtime_checkable
class CausalExperimentPass(CausalHeldOutPass, Protocol):
    """A causal pass carrying model-ready populations and declared control inputs."""

    def causal_subject(self) -> Subject[Reading, ModelReadyCausalPair]:
        """Return the instrumentable subject of the single-variable sweep."""
        ...

    def causal_fitting_pairs(self) -> tuple[ModelReadyCausalPair, ...]:
        """Return the fitting population, with stable identities."""
        ...

    def causal_held_out_pairs(self) -> tuple[ModelReadyCausalPair, ...]:
        """Return the disjoint population assessed and used by controls."""
        ...

    def causal_costs(self) -> CostModel:
        """Return measured prices for each declared search unit."""
        ...

    def causal_joint_variables(
        self,
    ) -> tuple[HighLevelVariable, HighLevelVariable]:
        """Return the strict-dominance variable pair tested jointly."""
        ...

    def causal_joint_subject(
        self,
    ) -> JointSubject[Reading, ModelReadyJointCausalPair]:
        """Return the subject scoring both variables of a joint pair."""
        ...

    def causal_joint_fitting_pairs(self) -> tuple[ModelReadyJointCausalPair, ...]:
        """Return joint fitting pairs, disjoint from development and evaluation."""
        ...

    def causal_joint_held_out_pairs(self) -> tuple[ModelReadyJointCausalPair, ...]:
        """Return joint evaluation pairs."""
        ...

    def causal_features(self, pair: ModelReadyCausalPair, /) -> AbstractionFeatures:
        """Return the declared bucketing features of one held-out pair."""
        ...

    def causal_bracketings(
        self, pairs: Sequence[ModelReadyCausalPair], /
    ) -> tuple[BinaryBracketing, ...]:
        """Return each held-out pair's observed ordered binary bracketing."""
        ...

    def causal_bracketing_pairs(
        self,
        pairs: Sequence[ModelReadyCausalPair],
        bracketings: Sequence[BinaryBracketing],
        /,
    ) -> tuple[ModelReadyCausalPair, ...]:
        """Return the same identities under the matched bracketings."""
        ...

    def causal_dormant_sources(
        self,
        variable: HighLevelVariable,
        targets: Sequence[ModelReadyCausalPair],
        /,
    ) -> tuple[ModelReadyCausalPair, ...]:
        """Return disjoint sources whose variant lacks the declared decision."""
        ...

    def causal_absent_decision(self, variable: HighLevelVariable, /) -> str | int:
        """Return the decision absent from that variable's dormant sources."""
        ...


def _ordinary(
    measure: str, value: float, count: int, site: Site | None = None
) -> ProbeValue:
    """Return one non-inferential causal value."""
    return ProbeValue(measure, site, value, count, None, None, None, None, None, None)


def _experiment_pass(
    candidate: CausalHeldOutPass, position: int, /
) -> CausalExperimentPass:
    """Return a causal pass publishing all populations required by this arm."""
    if not isinstance(candidate, CausalExperimentPass):
        msg = (
            f"causal held-out pass {position} publishes no model-ready sweep, joint, "
            "control, pricing, or bucketing capability"
        )
        raise TypeError(msg)

    return candidate


def _group(
    pairs: Sequence[ModelReadyCausalPair],
    variables: tuple[HighLevelVariable, HighLevelVariable],
    /,
) -> Mapping[HighLevelVariable, tuple[ModelReadyCausalPair, ...]]:
    """Return pairs grouped under the two variables selected for this sweep."""
    grouped = {
        variable: tuple(pair for pair in pairs if pair.varied == variable)
        for variable in variables
    }

    for variable, members in grouped.items():
        if not members:
            msg = f"a causal population has no pair varying {variable!r}"
            raise ValueError(msg)

    return MappingProxyType(grouped)


class CausalProbe:
    """Run the complete causal-abstraction arm over model-ready held-out passes."""

    __slots__ = ("_role", "_settings")

    def __init__(self, role: str, settings: CausalProbeSettings) -> None:
        self._role = ensure_name("a causal probe role", role)
        self._settings = settings

    def admit(self, published: Mapping[Site, SiteSpec], /) -> None:
        """Refuse any declared DAS site not differentiably observable and patchable."""
        for _coordinate, site in self._settings.sites:
            if site not in published:
                msg = f"a causal probe names unpublished site {site!r}"
                raise UnpublishedSiteError(msg)
            spec = published[site]
            if (
                not spec.differentiable
                or Access.OBSERVE not in spec.access
                or Access.PATCH not in spec.access
            ):
                msg = (
                    f"causal site {site!r} is not differentiably observable "
                    "and patchable"
                )
                raise UnpublishedSiteError(msg)

    def _run_pass(
        self, held: CausalExperimentPass, position: int, /
    ) -> tuple[tuple[ProbeValue, ...], JsonValue | None]:
        """Run sweep, joint, controls, and bucketing over one causal pass.

        Returns the pass's rows and, where the probe declares a node-vocabulary
        arm, the record of that arm's complete sweep over the held-out pairs.
        """
        generation: CounterfactualGeneration = held.causal_population()
        subject = held.causal_subject()
        fitting = held.causal_fitting_pairs()
        held_out = held.causal_held_out_pairs()
        if {pair.id for pair in fitting} & {pair.id for pair in held_out}:
            msg = "causal fitting and held-out identities overlap"
            raise ValueError(msg)

        variables = held.causal_joint_variables()
        fitting_by = _group(fitting, variables)
        held_by = _group(held_out, variables)
        sites = dict(self._settings.sites)
        layers = tuple(dict.fromkeys(layer for layer, _position in sites))
        positions = tuple(dict.fromkeys(position for _layer, position in sites))
        subgrid = Subgrid(variables, layers, positions, sites)
        rows: list[ProbeValue] = [
            _ordinary(
                f"pass-{position}/counterfactuals/aligned",
                float(generation.manifest.aligned_count),
                generation.manifest.candidate_count,
            ),
            _ordinary(
                f"pass-{position}/counterfactuals/refused",
                float(len(generation.refusals)),
                generation.manifest.candidate_count,
            ),
        ]
        reports: dict[str, SweepReport] = {}

        for name, search in self._settings.searches:
            preregistration = Preregistration(
                subgrid,
                search,
                costs=held.causal_costs(),
                budget_hours=self._settings.budget_hours,
            )
            report = sweep(subject, preregistration, fitting_by, held_by)
            reports[name] = report
            pvalues = tuple(result.p_value for result in report.results)
            decisions = BenjaminiHochberg().apply(
                pvalues, self._settings.fdr_alpha, self._settings.dependence
            )
            family = f"{self._role}/pass-{position}/{name}/iia"

            for index, result in enumerate(report.results):
                variable = variable_label(result.cell.variable)
                rows.append(
                    ProbeValue(
                        f"pass-{position}/{name}/{self._settings.objective.value}/"
                        f"{variable}/iia",
                        result.site,
                        result.result.accuracy,
                        len(held_by[result.cell.variable]),
                        result.p_value.provenance,
                        result.p_value.value,
                        decisions.adjusted[index],
                        decisions.reject[index],
                        family,
                        decisions.dependence,
                    )
                )

        arm = self._settings.arm
        record = (
            None
            if arm is None
            else node_vocabulary_row(
                arm[0], reports[arm[1]], (pair.id for pair in held_out)
            ).to_record()
        )
        anchor_report = reports[self._settings.joint_search]
        anchor_search = dict(self._settings.searches)[self._settings.joint_search]
        development_ids = tuple(sorted(pair.id for pair in held_out))
        selection = select_joint_layer(anchor_report, development_ids)
        joint = JointDistributedAlignment(
            held.causal_joint_subject(),
            selection,
            variables,
            schedule=self._settings.schedule,
            annealing=self._settings.annealing,
            bootstrap=Bootstrap(
                unit="pair",
                replicates=self._settings.bootstrap_replicates,
                seed=self._settings.seed,
                level=self._settings.bootstrap_level,
            ),
            objective=self._settings.objective,
        ).fit(held.causal_joint_fitting_pairs())
        joint_report = joint.report(held.causal_joint_held_out_pairs())
        rows.extend(
            (
                _ordinary(
                    f"pass-{position}/joint/iia",
                    joint_report.joint_accuracy,
                    len(held.causal_joint_held_out_pairs()),
                    selection.site,
                ),
                _ordinary(
                    f"pass-{position}/joint/minus-marginal-product",
                    joint_report.joint_minus_product.observed,
                    joint_report.joint_minus_product.units,
                    selection.site,
                ),
            )
        )
        control_cells: list[ControlCell] = []
        control_bootstrap = Bootstrap(
            unit="pair",
            replicates=self._settings.bootstrap_replicates,
            seed=self._settings.seed,
            level=self._settings.bootstrap_level,
        )

        for cell in anchor_report.cells:
            pairs = held_by[cell.variable]
            variable = variable_label(cell.variable)
            alignment = DistributedAlignment(
                subject,
                anchor_report.preregistration.subgrid.site_of(cell),
                cell.variable,
                search=anchor_search,
            ).fit(fitting_by[cell.variable])
            observed = IIAPopulation(
                zip((pair.id for pair in pairs), alignment.matches(pairs), strict=True)
            )
            scramble = scramble_types(pairs, seed=self._settings.seed)
            scrambled_population = IIAPopulation(
                zip(
                    (pair.id for pair in scramble.pairs),
                    alignment.matches(scramble.pairs),
                    strict=True,
                )
            )
            scrambled = ScrambledDelta(
                scramble,
                delta_iia(observed, scrambled_population, bootstrap=control_bootstrap),
            )
            reference = held.causal_bracketings(pairs)
            bracketings = matched_binary_bracketings(
                reference,
                tolerance=self._settings.bracketing_tolerance,
                seed=self._settings.seed,
                attempts=self._settings.bracketing_attempts,
            )

            if isinstance(bracketings, BracketingUnavailable):
                bracketing = bracketings
            else:
                changed = held.causal_bracketing_pairs(pairs, bracketings)
                changed_population = IIAPopulation(
                    zip(
                        (pair.id for pair in changed),
                        alignment.matches(changed),
                        strict=True,
                    )
                )
                bracketing = BracketingDelta(
                    reference,
                    bracketings,
                    delta_iia(
                        observed, changed_population, bootstrap=control_bootstrap
                    ),
                    tolerance=self._settings.bracketing_tolerance,
                )

            dormant = dormant_pathway(
                alignment,
                pairs,
                held.causal_dormant_sources(cell.variable, pairs),
                absent=held.causal_absent_decision(cell.variable),
            )
            control_cells.append(
                ControlCell(
                    cell,
                    scrambled,
                    bracketing,
                    dormant,
                    anchor_report.preregistration.digest,
                )
            )
            rows.append(
                _ordinary(
                    f"pass-{position}/control/{variable}/type-scramble-delta-iia",
                    scrambled.headline,
                    len(pairs),
                    anchor_report.preregistration.subgrid.site_of(cell),
                )
            )
            rows.append(
                _ordinary(
                    f"pass-{position}/control/{variable}/dormant-movement",
                    dormant.movement_rate,
                    len(pairs),
                    anchor_report.preregistration.subgrid.site_of(cell),
                )
            )
            rows.append(
                _ordinary(
                    f"pass-{position}/control/{variable}/bracketing-delta-iia",
                    bracketing.closest
                    if isinstance(bracketing, BracketingUnavailable)
                    else bracketing.headline,
                    len(pairs),
                    anchor_report.preregistration.subgrid.site_of(cell),
                )
            )
            cases = tuple(
                AbstractionCase(held.causal_features(pair), matched=matched)
                for pair, matched in zip(pairs, alignment.matches(pairs), strict=True)
            )
            buckets = bucket_abstraction(
                cases,
                well_cutoff=self._settings.well_cutoff,
                min_bucket_size=self._settings.minimum_bucket_size,
                min_accuracy_gap=self._settings.minimum_accuracy_gap,
            )
            rows.extend(
                _ordinary(
                    f"pass-{position}/bucket/{variable}/{bucket.label}/{index}",
                    bucket.accuracy,
                    bucket.size,
                    anchor_report.preregistration.subgrid.site_of(cell),
                )
                for index, bucket in enumerate(buckets.buckets)
            )

        ControlReport(anchor_report.preregistration, control_cells)

        return tuple(rows), record

    def recorded(self, point: TrainedPoint, /) -> RecordedProbe:
        """Return causal rows over every pass and each pass's arm record."""
        rows: list[ProbeValue] = []
        records: dict[str, JsonValue] = {}

        for position, candidate in enumerate(point.passes):
            if not isinstance(candidate, CausalHeldOutPass):
                msg = f"held-out pass {position} publishes no causal population"
                raise TypeError(msg)
            values, record = self._run_pass(
                _experiment_pass(candidate, position), position
            )
            rows.extend(values)

            if record is not None:
                records[f"pass-{position}"] = record

        return RecordedProbe(tuple(rows), MappingProxyType(records))

    def run(self, point: TrainedPoint, /) -> tuple[ProbeValue, ...]:
        """Return complete causal rows over every causal held-out pass."""
        return self.recorded(point).values


def _build(settings: CausalProbeSettings, context: ProbeBuildContext, /) -> CausalProbe:
    """Build a causal probe after validating every pass capability."""
    for position, candidate in enumerate(context.causal_passes()):
        _experiment_pass(candidate, position)

    return CausalProbe(context.role, settings)


PLUGIN: Final = ComponentPlugin(spec=CausalProbeSpec, build=_build)
"""The plugin the ``causal`` probe kind registers."""
