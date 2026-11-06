import json
from typing import TYPE_CHECKING, Literal, cast

import pytest

from lofi._types import ItemId
from lofi.data import Annotation, EvalDataset, Item, Token
from lofi.eval import (
    RECALL_HOMOGENEITY_METRIC,
    Approximation,
    ConventionVector,
    MetricResult,
    ReportIdentity,
    ReportRow,
    Resampling,
    read_predictions,
    rescore_declared,
)
from lofi.run import Evaluator
from lofi.spec import (
    ConstantPatternSpec,
    DatasetSpec,
    EvaluatorSpec,
    FeaturizerSpec,
    LossSpec,
    OptimizerSpec,
    ParserSpec,
    PatternRuleSpec,
    ProbeSpec,
    RuleSetSpec,
    RunSpec,
    SchedulerSpec,
    SinkSpec,
    TrainerSpec,
)
from lofi.spec.evaluators.alignment import AlignmentSettings
from lofi.spec.grammar import (
    BudgetSpec,
    GrammarDecl,
    GrammarIso,
    InlineLexiconSpec,
    LexicalEntrySpec,
    RootGoalSpec,
    SignatureSpec,
)
from lofi.spec.language import registered_language

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import JsonValue
    from lofi.formalism import Grammar


class IdentityParser(ParserSpec):
    kind: Literal["e63-parser"] = "e63-parser"
    width: int = 32


class IdentityLoss(LossSpec):
    kind: Literal["e63-loss"] = "e63-loss"
    label: str = "nll"


class IdentityDataset(DatasetSpec):
    kind: Literal["e63-dataset"] = "e63-dataset"
    split: str = "train"


class IdentityFeaturizer(FeaturizerSpec):
    kind: Literal["e63-featurizer"] = "e63-featurizer"
    lowercase: bool = False


class IdentityTrainer(TrainerSpec):
    kind: Literal["e63-trainer"] = "e63-trainer"
    epochs: int = 3


class IdentityOptimizer(OptimizerSpec):
    kind: Literal["e63-optimizer"] = "e63-optimizer"
    rate: float = 0.001


class IdentityScheduler(SchedulerSpec):
    kind: Literal["e63-scheduler"] = "e63-scheduler"
    decay: float = 0.9


class IdentitySink(SinkSpec):
    kind: Literal["e63-sink"] = "e63-sink"
    destination: str = "metrics.json"


class IdentityProbe(ProbeSpec):
    kind: Literal["e63-probe"] = "e63-probe"
    layer: int = 2


class IdentityEvaluator(EvaluatorSpec):
    kind: Literal["e63-evaluator"] = "e63-evaluator"
    threshold: float = 0.5


def _run_spec(**changed: JsonValue) -> RunSpec:
    document: dict[str, JsonValue] = {
        "model": {
            "language": {"from": "data"},
            "parser": {"kind": "e63-parser", "width": 32},
        },
        "loss": {"terms": [{"objective": {"kind": "e63-loss", "label": "nll"}}]},
        "data": {
            "datasets": {
                "train": {
                    "dataset": {"kind": "e63-dataset", "split": "train"},
                    "featurizer": {"kind": "e63-featurizer", "lowercase": False},
                }
            }
        },
        "trainer": {"kind": "e63-trainer", "epochs": 3},
        "optimizer": {"kind": "e63-optimizer", "rate": 0.001},
        "device_policy": {
            "device": "cpu",
            "compute_dtype": "float32",
            "deterministic": True,
            "matmul_precision": "highest",
            "activation_memory_bytes": 8_000_000_000,
        },
        "scheduler": {"kind": "e63-scheduler", "decay": 0.9},
        "sinks": {"metrics": {"kind": "e63-sink", "destination": "metrics.json"}},
        "probes": {"syntax": {"kind": "e63-probe", "layer": 2}},
        "evaluators": {
            "bracketing": {
                "kind": "e63-evaluator",
                "interval": {"replicates": 50, "level": 0.95},
                "threshold": 0.5,
            }
        },
        "parse": {"retain": "forest", "reachability": "all"},
        "release": {"checkpoint": "resume-latest"},
    }

    return RunSpec.model_validate(cast("Mapping[str, JsonValue]", document | changed))


def _grammar(*, alternate: bool = False) -> Grammar:
    first, second = ("B", "A") if alternate else ("A", "B")
    declaration = GrammarDecl(
        name="alternate" if alternate else "baseline",
        signature=SignatureSpec(
            name="cfg", primitives=("S", "A", "B"), syntax="s-expression"
        ),
        rules=RuleSetSpec(
            name="productions",
            rules=(
                PatternRuleSpec(
                    name="s-ba" if alternate else "s-ab",
                    premises=(
                        ConstantPatternSpec(name=first),
                        ConstantPatternSpec(name=second),
                    ),
                    conclusion=ConstantPatternSpec(name="S"),
                    head_policy="pass-through",
                ),
            ),
        ),
        lexicon=InlineLexiconSpec(
            entries=(LexicalEntrySpec(form="token", category='("#const[Cat]" "S")'),)
        ),
        goal=RootGoalSpec(categories=('("#const[Cat]" "S")',)),
        budget=BudgetSpec(unary_rounds=0),
    )
    return (
        GrammarIso().forward(declaration, registered_language("propositional")).grammar
    )


def _conventions() -> ConventionVector:
    return ConventionVector(
        punctuation="strip",
        unary_chains="collapse",
        trivial_spans="drop",
        minimum_length=3,
        aggregation="sentence",
        coverage="zero",
        self_arcs="forbid",
        gold_derivation="canonical",
    )


@pytest.mark.parametrize(
    "changed",
    [
        {
            "model": {
                "language": {"from": "data"},
                "parser": {"kind": "e63-parser", "width": 64},
            }
        },
        {"loss": {"terms": [{"objective": {"kind": "e63-loss", "label": "margin"}}]}},
        {
            "data": {
                "datasets": {
                    "train": {
                        "dataset": {"kind": "e63-dataset", "split": "dev"},
                        "featurizer": {
                            "kind": "e63-featurizer",
                            "lowercase": False,
                        },
                    }
                }
            }
        },
        {"trainer": {"kind": "e63-trainer", "epochs": 4}},
        {"optimizer": {"kind": "e63-optimizer", "rate": 0.002}},
        {"scheduler": {"kind": "e63-scheduler", "decay": 0.8}},
        {
            "sinks": {
                "metrics": {
                    "kind": "e63-sink",
                    "destination": "other.json",
                }
            }
        },
        {"probes": {"syntax": {"kind": "e63-probe", "layer": 3}}},
        {
            "evaluators": {
                "bracketing": {
                    "kind": "e63-evaluator",
                    "interval": {"replicates": 50, "level": 0.95},
                    "threshold": 0.7,
                }
            }
        },
    ],
)
def test_each_run_spec_section_changes_the_config_hash(
    changed: dict[str, JsonValue],
) -> None:
    baseline = ReportIdentity(
        _run_spec(), _grammar(), _conventions(), freeze_id="freeze/1"
    )
    modified = ReportIdentity(
        _run_spec(**changed), _grammar(), _conventions(), freeze_id="freeze/1"
    )

    assert baseline.config_hash != modified.config_hash
    assert baseline.grammar_fingerprint == modified.grammar_fingerprint
    assert baseline.convention_hash == modified.convention_hash


def test_a_compiled_grammar_change_reaches_the_identity_verbatim() -> None:
    first_grammar = _grammar()
    second_grammar = _grammar(alternate=True)
    first = ReportIdentity(
        _run_spec(), first_grammar, _conventions(), freeze_id="freeze/1"
    )
    second = ReportIdentity(
        _run_spec(), second_grammar, _conventions(), freeze_id="freeze/1"
    )

    assert first.grammar_fingerprint == first_grammar.fingerprint
    assert second.grammar_fingerprint == second_grammar.fingerprint
    assert first.grammar_fingerprint != second.grammar_fingerprint


class _Budget:
    depth = 4
    beam = 8
    mass = None


class _ExactMatch:
    conventions = None
    name = "exact-match"
    order_invariant = True

    def __init__(self) -> None:
        self._hits: dict[ItemId, bool] = {}

    def reset(self) -> None:
        self._hits = {}

    def update(self, pred: str, gold: str, *, key: ItemId) -> None:
        self._hits[key] = pred == gold

    def compute(self) -> MetricResult:
        return MetricResult(
            sum(self._hits.values()) / len(self._hits),
            len(self._hits),
            Resampling("item"),
        )

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        return {
            key: MetricResult(float(hit), 1, Resampling("item"))
            for key, hit in self._hits.items()
        }


def test_the_report_row_serializes_identity_and_experiment_columns() -> None:
    identity = ReportIdentity(
        _run_spec(), _grammar(), _conventions(), freeze_id="freeze/1"
    )
    row = ReportRow(
        identity=identity,
        method="cfg",
        metric="f1",
        result=MetricResult(0.75, 8, Resampling("sentence", 3)),
        approximation=Approximation(_Budget(), residual=None),
        suite="grammar-comparison",
        task="bracketing",
        dataset="synthetic",
        split="test",
        seed=3,
    )
    record = row.to_spec()

    assert record["config_hash"] == identity.config_hash
    assert record["grammar_fingerprint"] == identity.grammar_fingerprint
    assert record["convention_hash"] == identity.convention_hash
    assert record["suite"] == "grammar-comparison"
    assert json.loads(json.dumps(record)) == record


def test_current_row_producers_accept_real_run_and_grammar_sources() -> None:
    run_spec = _run_spec()
    grammar = _grammar()
    conventions = _conventions()
    expected = ReportIdentity(run_spec, grammar, conventions, freeze_id="freeze/1")
    dataset = EvalDataset(
        [Item(ItemId("s-1"), (Token(0, "Kim"),))],
        [Annotation(ItemId("s-1"), "yes")],
    )
    evaluator = Evaluator(
        dataset,
        [_ExactMatch()],
        run_spec=run_spec,
        grammar=grammar,
        freeze_id="freeze/1",
        fallback_conventions=conventions,
        suite="identity",
        task="exact-match",
        dataset_name="synthetic",
        split="test",
        seed=0,
        approximation=Approximation(_Budget(), residual=None),
        interval=None,
    )
    evaluated = evaluator.evaluate({ItemId("s-1"): "yes"}).rows[0]

    released = read_predictions(
        json.dumps(
            {
                "system": "released",
                "budget": {"depth": 2, "beam": 4, "mass": None},
                "items": [
                    {
                        "item_id": "d1",
                        "split": "dev",
                        "length": 2,
                        "punctuation": [],
                        "gold": [{"start": 0, "end": 2, "label": "S", "depth": 0}],
                        "spans": [{"start": 0, "end": 2, "label": "c0"}],
                    },
                    {
                        "item_id": "t1",
                        "split": "test",
                        "length": 2,
                        "punctuation": [],
                        "gold": [{"start": 0, "end": 2, "label": "S", "depth": 0}],
                        "spans": [{"start": 0, "end": 2, "label": "c0"}],
                    },
                ],
            }
        )
    )
    rescored = rescore_declared(
        released,
        run_spec=run_spec,
        grammar=grammar,
        freeze_id="freeze/1",
        alignments=(
            AlignmentSettings(
                kind="many-to-one",
                tie_break="least-gold-label",
                fit_on="dev",
                score_on="test",
                floor_replicates=19,
                floor_level=0.9,
                stability_replicates=19,
                stability_level=0.9,
            ),
        ),
        metric=RECALL_HOMOGENEITY_METRIC,
        conventions=(conventions,),
    )[0]

    assert evaluated.identity == expected
    assert rescored.row.identity == ReportIdentity(
        run_spec, grammar, rescored.conventions, freeze_id="freeze/1"
    )
