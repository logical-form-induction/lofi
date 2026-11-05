import json
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast, override

import didactic.api as dx
import pytest
from tests.support.installed import install
from tests.unit.spec.components.conftest import (
    INTERVAL,
    ApplicationDecoder,
    ChartParser,
    CoherenceLoss,
    ConsoleSink,
    CosineScheduler,
    ExactEvaluator,
    PinningLoss,
    ReadoutProbe,
    SampleDataset,
    SizedEncoder,
    StepTrainer,
    UniformController,
    WindowFeaturizer,
    complete,
    declaration,
    minimal,
)

from lofi.deduce import (
    DEFAULT_REFINEMENT_PAIRS,
    Boolean,
    CellPruner,
    Counting,
    Expectation,
    Forest,
    ItemKey,
    KBest,
    LogInside,
    ParsePolicy,
    Reachability,
    Retain,
    Viterbi,
)
from lofi.eval._scrambled import SignatureAssignments
from lofi.spec import Dependency, RunSpecCodec, config_search_path, load_plugins
from lofi.spec._errors import (
    ConfigDocumentError,
    DeclarationError,
    SemiringCapabilityError,
)
from lofi.spec.artifacts import ArtifactCacheSpec
from lofi.spec.components.base import RunRole
from lofi.spec.components.data import (
    DatasetSpec,
    RetainedDatasetSpec,
    TrainingAllocationSpec,
    TrainingBatchKind,
)
from lofi.spec.components.loss import LossSpec, WeightedLossSpec
from lofi.spec.components.run import (
    BootstrapIntervalSpec,
    CellPrunerSpec,
    EvaluationData,
    EvaluatorSpec,
    IntervalSettings,
    ParsePlugin,
    ParseSpec,
    RunSpec,
    SignatureAssignmentsSpec,
    SourceComparisonPointSpec,
)
from lofi.spec.components.trainer import TrainerSpec
from lofi.spec.losses import ShapedProofLossSpec
from lofi.spec.readouts import ReadoutSource, TargetSource, TermsReadoutSpec
from lofi.spec.spaces import CombinatorFamilySpec, Merge, SpaceSpec

if TYPE_CHECKING:
    from collections.abc import Sequence
    from importlib.metadata import EntryPoint
    from pathlib import Path

    from lofi._types import JsonValue
    from lofi.deduce import Cell
    from lofi.spec.components.model import ModelSection


class PluginPrunerSpec(CellPrunerSpec, extra="forbid"):
    """Test-only declaration supplied through the open pruner family."""

    kind: Literal["test-first"] = "test-first"

    @property
    def semiring_requirements(self) -> frozenset[str]:
        """Return that keeping the first item reads no weight."""
        return frozenset()


class FirstPruner:
    """Keep only the first item offered by a nonempty cell."""

    def keep[T](
        self,
        span: Cell,
        items: Sequence[tuple[ItemKey[T], float]],
        /,
    ) -> tuple[ItemKey[T], ...]:
        """Return at most the first offered key."""
        del span
        return () if not items else (items[0][0],)


def _first_pruner(declaration: PluginPrunerSpec, /) -> CellPruner[object, float]:
    """Build the test-only first-item pruner."""
    del declaration
    return FirstPruner()


TEST_PRUNER_PLUGIN = ParsePlugin(spec=PluginPrunerSpec, build=_first_pruner)


class _Point:
    """Minimal test entry point."""

    name = "test-first"
    value = "tests.unit.spec.components.test_run:TEST_PRUNER_PLUGIN"
    group = "lofi.cell_pruner"
    dist = None

    def load(self) -> object:
        """Return the test plugin."""
        return TEST_PRUNER_PLUGIN


def test_a_run_is_read_out_of_a_document_slot_by_slot() -> None:
    run = RunSpec.model_validate(
        declaration()
        | {
            "trainer": {"kind": "step-trainer", "steps": 3},
            "scheduler": {"kind": "cosine-scheduler"},
        }
    )

    assert run.model.parser == ChartParser()
    assert run.loss is not None
    assert run.loss.terms == (WeightedLossSpec(objective=CoherenceLoss()),)
    assert run.data.datasets["train"].dataset == SampleDataset()
    assert run.data.datasets["train"].featurizer == WindowFeaturizer()
    assert run.trainer == StepTrainer(steps=3)
    assert run.release.checkpoint == "resume-latest"
    assert run.parse.build().policy == ParsePolicy(Retain.FOREST, Reachability.ALL)
    assert run.parse.policy() == run.parse.build().policy
    assert run.scheduler == CosineScheduler()


def test_a_parse_declaration_selects_every_symbolic_chart_component() -> None:
    parsed = RunSpec.model_validate(
        declaration()
        | {
            "parse": {
                "retain": "forest",
                "reachability": "roots",
                "strategy": {"kind": "cky"},
                "span_index": {"kind": "contiguous"},
                "chart_store": {"kind": "native"},
                "pruner": {"kind": "top-k", "width": 3, "prune_lexical": True},
                "semiring": {"kind": "viterbi"},
            }
        }
    ).parse.build()

    assert parsed.policy == ParsePolicy(Retain.FOREST, Reachability.ROOTS)
    assert parsed.strategy.name == "cky"
    assert parsed.span_index.name == "contiguous"
    assert parsed.chart_store.name == "native"
    assert parsed.pruner is not None
    assert parsed.prune_lexical
    assert isinstance(parsed.semiring, Viterbi)


def test_a_pruner_registered_by_a_plugin_is_selected_from_a_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(monkeypatch, "cell_pruner", (cast("EntryPoint", _Point()),))
    parsed = RunSpec.model_validate(
        declaration()
        | {
            "parse": {
                "retain": "forest",
                "reachability": "all",
                "pruner": {"kind": "test-first"},
            }
        }
    ).parse.build()

    assert isinstance(parsed.pruner, FirstPruner)


def _parse(**fields: JsonValue) -> ParseSpec:
    return ParseSpec.model_validate(
        cast("dict[str, JsonValue]", {"retain": "forest", "reachability": "all"})
        | fields
    )


@pytest.mark.parametrize(
    ("semiring", "algebra"),
    [
        ({"kind": "log-inside"}, LogInside),
        ({"kind": "viterbi"}, Viterbi),
        ({"kind": "counting"}, Counting),
        ({"kind": "boolean"}, Boolean),
        ({"kind": "k-best", "k": 3}, KBest),
        ({"kind": "expectation", "dimensions": 2}, Expectation),
        ({"kind": "forest"}, Forest),
    ],
)
def test_every_implemented_semiring_may_be_declared_as_the_run_parse(
    semiring: dict[str, JsonValue], algebra: type[object]
) -> None:
    declared = _parse(semiring=cast("JsonValue", semiring))
    built = declared.build()

    assert isinstance(built.semiring, algebra)
    assert built.semiring.name == semiring["kind"]
    assert ParseSpec.model_validate_json(declared.model_dump_json()) == declared


def test_a_declared_semiring_carries_its_declared_settings() -> None:
    k_best = _parse(semiring={"kind": "k-best", "k": 3}).build().semiring
    expected = _parse(semiring={"kind": "expectation", "dimensions": 2}).build()

    assert cast("KBest", k_best).k == 3
    assert expected.semiring.one == (1.0, (0.0, 0.0))

    for refused in (
        {"kind": "k-best", "k": 0},
        {"kind": "expectation", "dimensions": 0},
    ):
        with pytest.raises(dx.ValidationError):
            _parse(semiring=cast("JsonValue", refused))


@pytest.mark.parametrize(
    ("kind", "capabilities"),
    [
        ("log-inside", {"log-probability", "marginals"}),
        ("viterbi", {"log-probability"}),
        ("counting", set()),
        ("boolean", set()),
        ("forest", set()),
    ],
)
def test_each_semiring_declares_what_its_weights_can_serve(
    kind: str, capabilities: set[str]
) -> None:
    assert _parse(semiring={"kind": kind}).semiring.capabilities == capabilities


def test_a_semiring_without_a_capability_is_refused_by_name() -> None:
    declared = _parse(semiring={"kind": "counting"})
    refusal = declared.semiring_refusal("log-probability", consumer="the test")

    assert isinstance(refusal, SemiringCapabilityError)
    assert refusal.semiring == "counting"
    assert refusal.capability == "log-probability"
    assert str(refusal) == (
        "the test needs float log-probability weights, which parse.semiring "
        "'counting' does not supply"
    )
    assert isinstance(refusal.prefixed("evaluators.x"), SemiringCapabilityError)
    assert _parse().semiring_refusal("marginals", consumer="the test") is None


@pytest.mark.parametrize("kind", ["counting", "boolean", "forest"])
def test_a_top_k_pruner_refuses_a_semiring_without_log_probabilities(
    kind: str,
) -> None:
    with pytest.raises(
        dx.ValidationError,
        match=(
            "the top-k pruner needs float log-probability weights, which "
            rf"parse\.semiring '{kind}' does not supply"
        ),
    ):
        _parse(semiring={"kind": kind}, pruner={"kind": "top-k", "width": 2})


def test_a_pruner_runs_under_any_semiring_it_declares_no_need_of(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install(monkeypatch, "cell_pruner", (cast("EntryPoint", _Point()),))
    built = _parse(semiring={"kind": "counting"}, pruner={"kind": "test-first"})

    assert isinstance(built.build().pruner, FirstPruner)
    assert isinstance(built.build().semiring, Counting)


def test_a_parse_may_opt_in_to_measuring_the_mass_its_pruner_lost() -> None:
    declared = _parse(pruner={"kind": "top-k", "width": 2}, measure_pruned_mass=True)

    assert declared.build().measure_pruned_mass
    assert not _parse(pruner={"kind": "top-k", "width": 2}).build().measure_pruned_mass


def test_measuring_pruned_mass_needs_a_pruner_and_summed_probabilities() -> None:
    with pytest.raises(dx.ValidationError, match="measures the mass a pruner lost"):
        _parse(measure_pruned_mass=True)

    with pytest.raises(
        dx.ValidationError,
        match=(
            r"measuring pruned mass needs posterior marginals, which "
            r"parse\.semiring 'viterbi' does not supply"
        ),
    ):
        _parse(
            semiring={"kind": "viterbi"},
            pruner={"kind": "top-k", "width": 2},
            measure_pruned_mass=True,
        )


def test_a_parse_declaration_refuses_a_pruner_on_an_inside_chart() -> None:
    with pytest.raises(dx.ValidationError, match="inside chart keeps no items"):
        RunSpec.model_validate(
            declaration()
            | {
                "parse": {
                    "retain": "inside",
                    "reachability": "all",
                    "pruner": {"kind": "top-k", "width": 2},
                }
            }
        )


def test_a_parse_declaration_keeps_an_inside_chart_without_a_pruner() -> None:
    parsed = RunSpec.model_validate(
        declaration() | {"parse": {"retain": "inside", "reachability": "all"}}
    ).parse.build()

    assert parsed.policy == ParsePolicy(Retain.INSIDE, Reachability.ALL)
    assert parsed.pruner is None


def test_a_parse_declaration_caps_the_reachability_refinement() -> None:
    default = ParseSpec(retain="forest", reachability="roots").build()
    capped = _parse(reachability="roots", refinement_pairs=7).build()

    assert default.policy.refinement_pairs == DEFAULT_REFINEMENT_PAIRS
    assert capped.policy == ParsePolicy(
        Retain.FOREST, Reachability.ROOTS, refinement_pairs=7
    )


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"reachability": "roots", "refinement_pairs": -1}, "refinement_pairs"),
        ({"reachability": "all", "refinement_pairs": 7}, "reachability: roots"),
    ],
)
def test_a_parse_declaration_refuses_a_refinement_cap_it_cannot_use(
    fields: dict[str, JsonValue], match: str
) -> None:
    with pytest.raises(dx.ValidationError, match=match):
        _parse(**fields)


class ForestReadingEvaluator(EvaluatorSpec):
    """Test-only evaluator whose decoding parse reads a packed forest."""

    kind: Literal["forest-reading-evaluator"] = "forest-reading-evaluator"

    def parse_refusal(self, parse: ParseSpec, /) -> DeclarationError | None:
        """Refuse a run parse that keeps no forest."""
        if parse.retain == "forest":
            return None
        return DeclarationError("this evaluator decodes a packed forest")


def test_a_run_parse_an_evaluator_cannot_decode_with_is_refused_by_role() -> None:
    evaluators = {
        "evaluators": {
            "scored": {
                "kind": "forest-reading-evaluator",
                "interval": {"replicates": 50, "level": 0.95},
            }
        }
    }
    accepted = RunSpec.model_validate(declaration() | evaluators)

    assert isinstance(accepted.evaluators["scored"], ForestReadingEvaluator)

    with pytest.raises(
        dx.ValidationError,
        match=r"evaluators\.scored was refused: this evaluator decodes a packed forest",
    ):
        RunSpec.model_validate(
            declaration()
            | evaluators
            | {"parse": {"retain": "inside", "reachability": "all"}}
        )


def test_a_run_round_trips_through_json() -> None:
    run = complete()

    assert RunSpec.model_validate_json(run.model_dump_json()) == run


def test_a_run_retains_typed_source_comparison_roles() -> None:
    run = RunSpec.model_validate(
        declaration()
        | {
            "comparison": {
                "source": "source",
                "condition": "primary",
                "direction": "bidirectional",
                "twin": "observed",
            }
        }
    )

    assert run.comparison == SourceComparisonPointSpec(
        source="source",
        condition="primary",
        direction="bidirectional",
        twin="observed",
    )
    assert RunSpec.model_validate_json(run.model_dump_json()) == run


def test_source_comparison_roles_refuse_unknown_or_empty_fields() -> None:
    body = {
        "source": "source",
        "condition": "primary",
        "direction": "bidirectional",
        "twin": "observed",
    }

    with pytest.raises(dx.ValidationError):
        RunSpec.model_validate(declaration() | {"comparison": body | {"capacity": 128}})

    with pytest.raises(dx.ValidationError, match="source-comparison point field"):
        RunSpec.model_validate(declaration() | {"comparison": body | {"condition": ""}})


def test_a_run_retains_typed_signature_assignments() -> None:
    model = cast("dict[str, JsonValue]", declaration()["model"])
    document = declaration()
    document["model"] = cast("JsonValue", model | {"language": {"from": "data"}})
    document["signature_assignment"] = cast(
        "JsonValue",
        {
            "primitives": ["S", "NP", "N"],
            "constructors": ["forward", "backward"],
            "combinators": ["application", "composition"],
        },
    )
    run = RunSpec.model_validate(document)

    assert run.signature_assignment == SignatureAssignmentsSpec(
        primitives=("S", "NP", "N"),
        constructors=("forward", "backward"),
        combinators=("application", "composition"),
    )
    assignments = run.signature_assignment
    assert assignments is not None
    assert assignments.build() == SignatureAssignments(
        ("S", "NP", "N"),
        ("forward", "backward"),
        ("application", "composition"),
    )
    assert RunSpec.model_validate_json(run.model_dump_json()) == run


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("primitives", ["S", "S"]),
        ("constructors", ["forward", ""]),
        ("combinators", ["application", "application"]),
    ],
)
def test_signature_assignments_refuse_empty_or_repeated_names(
    field: str, value: list[str]
) -> None:
    body: dict[str, JsonValue] = {
        "primitives": ["S", "NP"],
        "constructors": ["forward", "backward"],
        "combinators": ["application", "composition"],
    }
    body[field] = cast("JsonValue", value)
    model = cast("dict[str, JsonValue]", declaration()["model"])
    document = declaration()
    document["model"] = cast("JsonValue", model | {"language": {"from": "data"}})
    document["signature_assignment"] = body

    with pytest.raises(dx.ValidationError, match="signature-assignment"):
        RunSpec.model_validate(document)


def test_every_slot_survives_the_round_trip() -> None:
    restored = RunSpec.model_validate_json(complete().model_dump_json())

    assert restored.model.encoders == {"terms": SizedEncoder(width=16)}
    assert restored.model.controller == UniformController()
    assert restored.model.decoders == {"forward": ApplicationDecoder(rank=1)}
    assert restored.model.readouts == {
        "publishing": TermsReadoutSpec(
            space="types",
            reads=ReadoutSource.INSIDE,
            targets=TargetSource.GRAMMAR_GOAL,
            output="grammar-goal",
        )
    }
    assert restored.model.spaces == {
        "types": SpaceSpec(
            sort="Ty",
            encoder="terms",
            identity="forward",
            combinators=CombinatorFamilySpec(
                merge=Merge.CONCATENATE,
                decoders={"apply": "forward"},
            ),
        )
    }
    assert restored.loss is not None
    assert restored.loss.terms == (
        WeightedLossSpec(objective=CoherenceLoss(), weight=0.1),
        WeightedLossSpec(objective=PinningLoss(margin=0.25), weight=0.2),
    )
    assert restored.probes == {"readout": ReadoutProbe()}
    assert restored.sinks == {"console": ConsoleSink()}
    assert restored.evaluators == {"exact": ExactEvaluator(interval=INTERVAL)}
    assert restored.device_policy.device == "cpu"


def test_a_slot_a_run_leaves_unfilled_is_absent() -> None:
    run = minimal()

    assert run.scheduler is None
    assert run.pair_sampler is None
    assert run.type_sampler is None
    assert run.model.encoders == {}
    assert run.model.token_encoder is None
    assert run.model.subword_pooler is None
    assert run.model.controller is None
    assert run.model.decoders == {}
    assert run.model.readouts == {}
    assert run.model.spaces == {}
    assert run.sinks == {}
    assert run.probes == {}
    assert run.evaluators == {}


def test_an_unknown_run_setting_is_refused() -> None:
    with pytest.raises(dx.ValidationError):
        RunSpec.model_validate(declaration() | {"tainer": {"kind": "step-trainer"}})


def test_an_unknown_setting_inside_a_section_is_refused() -> None:
    with pytest.raises(dx.ValidationError):
        RunSpec.model_validate(
            declaration() | {"model": {"parser": {"kind": "chart-parser"}, "beam": 4}}
        )


def test_an_unknown_setting_of_a_variant_is_refused() -> None:
    with pytest.raises(dx.ValidationError):
        RunSpec.model_validate(
            declaration() | {"model": {"parser": {"kind": "chart-parser", "bream": 4}}}
        )


@pytest.mark.parametrize(
    "slot",
    ["model", "data", "trainer", "release", "parse", "optimizer", "device_policy"],
)
def test_a_run_that_leaves_a_required_slot_out_is_refused(slot: str) -> None:
    document = declaration()
    del document[slot]

    with pytest.raises(dx.ValidationError):
        RunSpec.model_validate(document)


@pytest.mark.parametrize(
    ("section", "remaining"),
    [
        ("data", {"featurizer": {"kind": "window-featurizer"}}),
    ],
)
def test_a_section_that_leaves_a_required_slot_out_is_refused(
    section: str, remaining: dict[str, JsonValue]
) -> None:
    with pytest.raises(dx.ValidationError):
        RunSpec.model_validate(declaration() | {section: remaining})


def test_a_slot_refuses_a_variant_of_another_family() -> None:
    with pytest.raises(dx.ValidationError):
        RunSpec.model_validate(declaration() | {"optimizer": {"kind": "step-trainer"}})


def test_two_runs_declared_alike_are_one_value() -> None:
    other = minimal().with_(trainer=StepTrainer(steps=11))

    assert {minimal(), minimal(), other} == {minimal(), other}


def test_a_run_changes_only_by_making_another() -> None:
    run = minimal()
    scheduled = run.with_(scheduler=CosineScheduler())

    assert scheduled.scheduler == CosineScheduler()
    assert run.scheduler is None


def test_a_family_is_stored_in_the_order_of_the_names_it_was_given() -> None:
    run = minimal().with_(sinks={"zeta": ConsoleSink(), "alpha": ConsoleSink()})

    assert list(run.sinks) == ["alpha", "zeta"]


def test_two_runs_naming_their_sinks_in_another_order_are_one_run() -> None:
    one = minimal().with_(sinks={"zeta": ConsoleSink(), "alpha": ConsoleSink()})
    other = minimal().with_(sinks={"alpha": ConsoleSink(), "zeta": ConsoleSink()})

    assert one == other
    assert {one, other} == {one}


def test_the_order_of_a_family_survives_the_round_trip() -> None:
    run = minimal().with_(probes={"zeta": ReadoutProbe(), "alpha": ReadoutProbe()})
    restored = RunSpec.model_validate_json(run.model_dump_json())

    assert list(restored.probes) == ["alpha", "zeta"]


def test_two_runs_naming_their_evaluators_in_another_order_are_one_run() -> None:
    one = minimal().with_(
        evaluators={
            "zeta": ExactEvaluator(interval=INTERVAL),
            "alpha": ExactEvaluator(interval=INTERVAL),
        }
    )
    other = minimal().with_(
        evaluators={
            "alpha": ExactEvaluator(interval=INTERVAL),
            "zeta": ExactEvaluator(interval=INTERVAL),
        }
    )

    assert list(one.evaluators) == ["alpha", "zeta"]
    assert one == other


def test_the_document_a_run_dumps_addresses_a_component_by_its_name() -> None:
    document = json.loads(complete().model_dump_json())

    assert document["sinks"]["console"]["kind"] == "console-sink"
    assert document["probes"]["readout"]["kind"] == "readout-probe"
    assert document["evaluators"]["exact"]["kind"] == "exact-evaluator"
    assert document["model"]["decoders"]["forward"]["rank"] == 1


class CountedDataset(DatasetSpec):
    kind: Literal["counted-dataset"] = "counted-dataset"
    occurrences: int

    @override
    def observed_occurrences(self) -> int:
        return self.occurrences


class AllocatedCountedDataset(DatasetSpec):
    kind: Literal["allocated-counted-dataset"] = "allocated-counted-dataset"
    occurrences: int
    held_out: int

    @override
    def observed_occurrences(self) -> int:
        return self.occurrences

    def training_occurrences(self, total: int, /) -> int:
        return total - self.held_out


class PairedTrainer(TrainerSpec):
    kind: Literal["paired-trainer"] = "paired-trainer"

    @override
    def refuse_epoch(self, occurrences: int, /) -> None:
        if occurrences % 2:
            msg = f"trainer.pairs: {occurrences} occurrences do not pair"
            raise ValueError(msg)

    @override
    def dataset_requirements(
        self,
    ) -> MappingProxyType[str, frozenset[TrainingBatchKind]]:
        return MappingProxyType({"train": frozenset()})


def counted(occurrences: int, trainer: str) -> dict[str, JsonValue]:
    document = declaration()
    document["data"] = {
        "datasets": {
            "train": {
                "dataset": {
                    "kind": "counted-dataset",
                    "occurrences": occurrences,
                }
            }
        }
    }
    document["trainer"] = {"kind": trainer}

    return document


def test_a_run_refuses_a_dataset_its_trainer_cannot_divide_at_composition() -> None:
    assert RunSpec.model_validate(counted(4, "paired-trainer")).trainer == (
        PairedTrainer()
    )

    with pytest.raises(dx.ValidationError, match=r"trainer\.pairs"):
        RunSpec.model_validate(counted(5, "paired-trainer"))


def test_training_allocation_adjusts_the_count_before_epoch_validation() -> None:
    document = counted(5, "paired-trainer")
    document["data"] = {
        "datasets": {
            "train": {
                "dataset": {
                    "kind": "allocated-counted-dataset",
                    "occurrences": 5,
                    "held_out": 1,
                }
            }
        }
    }

    run = RunSpec.model_validate(document)
    dataset = run.data.datasets["train"].dataset

    assert isinstance(dataset, TrainingAllocationSpec)
    assert not isinstance(dataset, RetainedDatasetSpec)
    assert run.trainer == PairedTrainer()

    with pytest.raises(dx.ValidationError, match=r"trainer\.pairs"):
        RunSpec.model_validate(
            document
            | {
                "data": {
                    "datasets": {
                        "train": {
                            "dataset": {
                                "kind": "allocated-counted-dataset",
                                "occurrences": 4,
                                "held_out": 1,
                            }
                        }
                    }
                }
            }
        )


def test_a_dataset_known_only_once_read_and_an_undivided_trainer_are_accepted() -> None:
    assert (
        RunSpec.model_validate(declaration()).data.datasets["train"].dataset
        == SampleDataset()
    )
    assert RunSpec.model_validate(counted(5, "step-trainer")).trainer == StepTrainer()


def test_an_evaluator_declares_the_interval_its_values_carry() -> None:
    declared = ExactEvaluator(interval=INTERVAL)

    assert declared.interval.settings() == IntervalSettings(50, 0.95)

    with pytest.raises(dx.ValidationError, match="interval"):
        ExactEvaluator.model_validate({"kind": "exact-evaluator"})


@pytest.mark.parametrize(
    ("replicates", "level", "message"),
    [
        (0, 0.95, "at least one resample"),
        (50, 0.0, "not a proportion"),
        (50, 1.0, "not a proportion"),
    ],
)
def test_a_malformed_interval_is_refused(
    replicates: int, level: float, message: str
) -> None:
    with pytest.raises(dx.ValidationError, match=message):
        BootstrapIntervalSpec(replicates=replicates, level=level)


def test_an_evaluator_variant_that_declares_no_metric_reports_none() -> None:
    assert ExactEvaluator(interval=INTERVAL).metrics() == ()
    assert ExactEvaluator(interval=INTERVAL).metrics(("yes", "no")) == ()
    assert (
        ExactEvaluator(interval=INTERVAL).evaluation_data()
        is EvaluationData.DERIVATIONS
    )


class SentenceDataset(DatasetSpec):
    kind: Literal["sentence-dataset"] = "sentence-dataset"

    @override
    def training_batches(self) -> frozenset[TrainingBatchKind]:
        return frozenset({TrainingBatchKind.SENTENCES})


class GoldFormDataset(DatasetSpec):
    kind: Literal["gold-form-dataset"] = "gold-form-dataset"

    @override
    def training_batches(self) -> frozenset[TrainingBatchKind]:
        return frozenset({TrainingBatchKind.SENTENCES, TrainingBatchKind.GOLD_FORMS})


class JudgmentDataset(DatasetSpec):
    kind: Literal["judgment-dataset"] = "judgment-dataset"

    @override
    def training_batches(self) -> frozenset[TrainingBatchKind]:
        return frozenset(
            {
                TrainingBatchKind.SENTENCES,
                TrainingBatchKind.INFERENCE_PAIRS,
                TrainingBatchKind.JUDGMENTS,
            }
        )


class GoldFormLoss(LossSpec):
    kind: Literal["gold-form-loss"] = "gold-form-loss"

    @override
    def required_training_batches(self) -> frozenset[TrainingBatchKind]:
        return frozenset({TrainingBatchKind.GOLD_FORMS})


class SentenceTrainer(TrainerSpec):
    kind: Literal["sentence-trainer"] = "sentence-trainer"

    @override
    def required_roles(self) -> frozenset[RunRole]:
        return frozenset({RunRole.PARSER, RunRole.LOSS})

    @override
    def forbidden_roles(self) -> frozenset[RunRole]:
        return frozenset(
            {RunRole.PAIR_SAMPLER, RunRole.TYPE_SAMPLER, RunRole.EVALUATORS}
        )

    @override
    def dataset_requirements(
        self,
    ) -> MappingProxyType[str, frozenset[TrainingBatchKind]]:
        return MappingProxyType({"train": frozenset({TrainingBatchKind.SENTENCES})})


class ReadingLoss(LossSpec):
    kind: Literal["reading-loss"] = "reading-loss"
    readout: str

    @override
    def dependencies(self) -> tuple[Dependency, ...]:
        return (Dependency(self.readout, "form-typing-readout"),)


class ModelNamingLoss(LossSpec):
    kind: Literal["model-naming-loss"] = "model-naming-loss"
    space: str

    @override
    def refuse_model(self, model: ModelSection, path: str, /) -> None:
        if self.space not in model.spaces:
            msg = f"{path}.space: {self.space!r} is not a declared space"
            raise ValueError(msg)


class NamingTrainer(SentenceTrainer):
    kind: Literal["naming-trainer"] = "naming-trainer"
    names: tuple[str, ...] = ()

    @override
    def refuse_loss_terms(self, terms: frozenset[str], /) -> None:
        absent = sorted(set(self.names) - terms)

        if absent:
            msg = f"names absent loss terms {absent}"
            raise ValueError(msg)


def sentence_run(dataset: str) -> dict[str, JsonValue]:
    document = declaration()
    document["data"] = {"datasets": {"train": {"dataset": {"kind": dataset}}}}
    document["trainer"] = {"kind": "sentence-trainer"}

    return document


def test_a_run_whose_dataset_yields_the_trainers_batches_composes() -> None:
    run = RunSpec.model_validate(sentence_run("sentence-dataset"))

    assert run.trainer == SentenceTrainer()


def test_a_loss_requiring_gold_forms_composes_only_with_a_selected_gold_dataset() -> (
    None
):
    document = sentence_run("gold-form-dataset")
    document["loss"] = {"terms": [{"objective": {"kind": "gold-form-loss"}}]}

    run = RunSpec.model_validate(document)

    assert run.loss is not None
    assert isinstance(run.loss.terms[0].objective, GoldFormLoss)

    document = sentence_run("sentence-dataset")
    document["loss"] = {"terms": [{"objective": {"kind": "gold-form-loss"}}]}

    with pytest.raises(dx.ValidationError) as refused:
        RunSpec.model_validate(document)

    message = str(refused.value)
    assert "loss.terms.gold-form-loss" in message
    assert "training-batch capability ['gold-forms']" in message


def test_shaped_proof_is_refused_by_name_for_a_judgment_only_dataset() -> None:
    document = sentence_run("judgment-dataset")
    objective = ShapedProofLossSpec.model_validate(
        {
            "kind": "shaped-proof",
            "readout": "forms",
            "shaping": {
                "ill_formed": 0.0,
                "well_formed": 0.1,
                "type_checks": 0.2,
                "proved": 1.0,
                "length": 0.0,
            },
            "budget": {
                "calls_per_step": 1,
                "wall_per_step": 1.0,
                "cpu_seconds": None,
            },
        }
    )
    document["loss"] = {
        "terms": [{"objective": cast("JsonValue", objective.model_dump())}]
    }

    with pytest.raises(dx.ValidationError) as refused:
        RunSpec.model_validate(document)

    message = str(refused.value)
    assert "loss.terms.shaped-proof" in message
    assert "training-batch capability ['gold-forms']" in message


def test_a_run_whose_dataset_lacks_the_trainers_batches_is_refused_naming_them() -> (
    None
):
    with pytest.raises(dx.ValidationError) as refused:
        RunSpec.model_validate(sentence_run("sample-dataset"))

    message = str(refused.value)
    assert "data.datasets.train.dataset" in message
    assert "training-batch capability ['sentences']" in message
    assert "'sample-dataset' dataset does not declare" in message


@pytest.mark.parametrize(
    ("removed", "role"), [("loss", "loss"), ("model", "model.parser")]
)
def test_a_run_missing_a_role_its_trainer_requires_is_refused(
    removed: str, role: str
) -> None:
    document = sentence_run("sentence-dataset")
    document[removed] = None if removed == "loss" else {"language": {"from": "data"}}

    with pytest.raises(
        dx.ValidationError, match=rf"requires component roles \['{role}'\]"
    ):
        RunSpec.model_validate(document)


@pytest.mark.parametrize(
    ("added", "value", "role"),
    [
        ("type_sampler", {"kind": "finite-sampler"}, "data.type_sampler"),
        ("pair_sampler", {"kind": "rule-sampler"}, "data.pair_sampler"),
        (
            "evaluators",
            {
                "exact": {
                    "kind": "exact-evaluator",
                    "interval": {"replicates": 50, "level": 0.95},
                }
            },
            "output.evaluators",
        ),
    ],
)
def test_a_run_declaring_a_role_its_trainer_does_not_consume_is_refused(
    added: str, value: JsonValue, role: str
) -> None:
    document = sentence_run("sentence-dataset")
    document[added] = value

    with pytest.raises(
        dx.ValidationError, match=rf"does not consume component roles \['{role}'\]"
    ):
        RunSpec.model_validate(document)


def test_an_empty_collection_declares_no_role() -> None:
    document = sentence_run("sentence-dataset")
    document["evaluators"] = {}

    run = RunSpec.model_validate(document)

    assert not run.declares(RunRole.EVALUATORS)


def test_a_role_is_declared_exactly_when_a_component_lies_under_its_path() -> None:
    for run in (minimal(), complete()):
        paths = run.component_roles()

        for role in RunRole:
            assert run.declares(role) == any(
                path == role.value or path.startswith(f"{role.value}.")
                for path in paths
            )

    assert {role for role in RunRole if minimal().declares(role)} == {
        RunRole.DATASETS,
        RunRole.LOSS,
        RunRole.PARSER,
        RunRole.TRAINER,
        RunRole.OPTIMIZER,
    }
    assert {role for role in RunRole if complete().declares(role)} == {
        RunRole.DATASETS,
        RunRole.LOSS,
        RunRole.PARSER,
        RunRole.TERM_ENCODERS,
        RunRole.CONTROLLER,
        RunRole.TERM_DECODERS,
        RunRole.READOUTS,
        RunRole.SINKS,
        RunRole.PROBES,
        RunRole.EVALUATORS,
        RunRole.TRAINER,
        RunRole.OPTIMIZER,
        RunRole.SCHEDULER,
    }


def test_a_loss_term_naming_an_undeclared_readout_is_refused_at_composition() -> None:
    document = sentence_run("sentence-dataset")
    document["loss"] = {
        "terms": [{"objective": {"kind": "reading-loss", "readout": "forms"}}]
    }

    with pytest.raises(dx.ValidationError) as refused:
        RunSpec.model_validate(document)

    message = str(refused.value)
    assert "loss.terms.reading-loss" in message
    assert "readout role 'forms'" in message


def test_a_loss_term_naming_an_undeclared_model_component_is_refused() -> None:
    document = sentence_run("sentence-dataset")
    document["loss"] = {
        "terms": [{"objective": {"kind": "model-naming-loss", "space": "absent"}}]
    }

    with pytest.raises(
        dx.ValidationError,
        match=r"loss\.terms\.model-naming-loss\.space: 'absent' is not a declared",
    ):
        RunSpec.model_validate(document)


def test_the_trainer_is_checked_against_the_declared_loss_terms_at_composition() -> (
    None
):
    document = sentence_run("sentence-dataset")
    document["trainer"] = {"kind": "naming-trainer", "names": ["coherence-loss"]}

    assert RunSpec.model_validate(document).trainer == NamingTrainer(
        names=("coherence-loss",)
    )

    document["trainer"] = {"kind": "naming-trainer", "names": ["simplicity"]}

    with pytest.raises(dx.ValidationError, match=r"absent loss terms \['simplicity'\]"):
        RunSpec.model_validate(document)


def test_a_run_declares_its_counterfactual_population_beside_its_probes() -> None:
    population = {
        "sentences": "candidates.txt",
        "unary_depth": 0,
        "readings": 2,
        "render": "declared",
        "pieces": "all",
        "specials": "exclude",
        "contiguous": True,
        "seed": 0,
    }
    declared = RunSpec.model_validate({**declaration(), "counterfactuals": population})

    assert minimal().counterfactuals is None
    assert declared.counterfactuals is not None
    assert declared.counterfactuals.settings().readings == 2
    assert json.loads(declared.model_dump_json())["counterfactuals"] == {
        **population,
        "filter": None,
    }


def _pretrained_run(tmp_path: Path, *overrides: str) -> RunSpec:
    """Compose the packaged parser over a pretrained token encoder."""
    document = tmp_path / "pretrained.yaml"
    document.write_text(
        "defaults:\n  - run/synthetic-parser\n  - model.token_encoder: pretrained\n",
        encoding="utf-8",
        newline="\n",
    )
    load_plugins()

    return RunSpecCodec(search_path=config_search_path(tmp_path))(
        document, overrides=overrides
    )


def test_a_run_loading_pretrained_weights_declares_the_cache_they_are_fetched_into(
    tmp_path: Path,
) -> None:
    with pytest.raises(ConfigDocumentError, match="artifacts cache"):
        _pretrained_run(tmp_path)

    run = _pretrained_run(tmp_path, f"artifacts.root={tmp_path / 'cache'}")

    assert run.artifacts == ArtifactCacheSpec(root=str(tmp_path / "cache"))
    assert "artifacts" in RunSpec.observation_sections()


def test_a_run_loading_no_pretrained_weights_declares_no_cache(
    tmp_path: Path,
) -> None:
    with pytest.raises(dx.ValidationError, match="no component loads one"):
        RunSpec.model_validate(declaration() | {"artifacts": {"root": str(tmp_path)}})
