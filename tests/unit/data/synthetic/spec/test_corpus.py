import json
import subprocess
import sys
from hashlib import sha256
from json import loads
from math import inf, log, nan
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast

import didactic.api as dx
import pytest
from hypothesis import strategies as st
from tests.unit.data.synthetic.conftest import (
    ABSTAINING_TPTP,
    SCOPED_ENTRIES,
    SCOPED_INVENTORY,
    SCOPING,
    grammar,
    interpretation,
    language,
    sample_digest,
    sample_rows,
)
from tests.unit.data.synthetic.spec.conftest import (
    ISO_LAW_NAMES,
    IsoLaw,
    _grammar,
    _interpreted,
    _interpreted_grammar,
    _projection,
)
from tests.unit.spec.grammar.test__signature import slash_views, slashes

from lofi.data.laws import ISO_LAWS, IsoFixture
from lofi.data.synthetic import (
    AmbiguityTarget,
    AmbiguityUnreachableError,
    Histogram,
    LabelNoise,
    NegativeBinomialLength,
    NegativeDataPolicy,
    NegativeDataRegime,
    ScopeTarget,
    ScopeUnreachableError,
    ShapeSchedule,
    ShapeTarget,
    SyntheticDataset,
    SyntheticGenerator,
    TabularLength,
    Target,
    UniformLength,
    estimate,
)
from lofi.data.synthetic.spec.corpus import (
    SOURCE_VERSION,
    TARGET_VERSION,
    AmbiguitySpec,
    CorruptionOperatorsSpec,
    CorruptionProportionSpec,
    CorruptionSpec,
    DerivationSamplerSpec,
    EstimatedClassSpec,
    IllTypingOperatorsSpec,
    IllTypingSpec,
    LabelNoiseRateSpec,
    LabelNoiseSpec,
    LengthMassSpec,
    LengthSpec,
    LexiconEstimateSpec,
    NegativeBinomialLengthSpec,
    ResemblingLexiconSpec,
    ScopeSpec,
    ShapeScheduleSpec,
    ShapeTargetSpec,
    SyntheticCorpusDecl,
    SyntheticCorpusIso,
    SyntheticCorpusLowerer,
    TabularLengthSpec,
    TargetSpec,
    UniformDerivationSamplerSpec,
    UniformLengthSpec,
    WeightedRulesDerivationSamplerSpec,
    canonical_bytes,
    fingerprint,
)
from lofi.data.synthetic.spec.operators import (
    ArgumentSwapSpec,
    DeletionSpec,
    InsertionSpec,
    OverApplicationSpec,
    SortMismatchSpec,
    SwapSpec,
    WeightedCorruptionOperatorSpec,
    WeightedIllTypingOperatorSpec,
)
from lofi.deduce import ParsePolicy, Reachability, Retain
from lofi.lexicon import Entry, StaticLexicon
from lofi.run import (
    RESOLVED_NAME,
    ExperimentSeed,
    ManifestIso,
    open_manifest,
)
from lofi.run._entrypoint import main as run_command
from lofi.spec import (
    DERIVED_SEED,
    DatasetSpec,
    DeclarationError,
    FeaturizerSpec,
    LossSpec,
    OptimizerSpec,
    ParserSpec,
    RunSpec,
    SpecLoader,
    TrainerSpec,
    TrainingBatchKind,
    UnresolvedSeedError,
    fixed_seed,
)
from lofi.spec.grammar import (
    BudgetSpec,
    ConjunctionPairSpec,
    ConjunctionsSpec,
    GrammarDecl,
    GrammarIso,
    InlineLexiconSpec,
    LexicalEntrySpec,
    LexiconSpec,
    OperatorSpec,
    PowerLawRankLawSpec,
    SignatureSpec,
    declare_grammar,
)
from lofi.spec.grammar import canonical_bytes as grammar_bytes
from lofi.spec.language import registered_language

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Protocol

    from lofi._types import JsonValue
    from lofi.data.synthetic import LexiconEstimate
    from lofi.data.synthetic._lengths import LengthDistribution

    class _LengthDeclaration(Protocol):
        """A declaration that builds a token-count distribution."""

        def build(self) -> LengthDistribution:
            """Return the interior distribution."""
            ...

    class _JointDeclaration(Protocol):
        """A declaration that supplies a category-mass marginal."""

        def marginal(self) -> tuple[tuple[str, float], ...]:
            """Return categories with their normalized token shares."""
            ...


type Change = Callable[[SyntheticCorpusDecl], SyntheticCorpusDecl]
TIMEOUT_SECONDS: Final = 120
FOREST_POLICY: Final = ParsePolicy(Retain.FOREST, Reachability.ALL)


def _corruption(
    proportion: float = 0.25,
    *,
    operators: tuple[WeightedCorruptionOperatorSpec, ...] = (
        WeightedCorruptionOperatorSpec(operator=SwapSpec()),
        WeightedCorruptionOperatorSpec(operator=DeletionSpec()),
    ),
    attempts: int = 50,
) -> CorruptionSpec:
    return CorruptionSpec(
        proportion=CorruptionProportionSpec(value=proportion),
        operators=CorruptionOperatorsSpec(entries=operators),
        attempts=attempts,
    )


def _ill_typing(
    proportion: float = 0.5,
    *,
    operators: tuple[WeightedIllTypingOperatorSpec, ...] = (
        WeightedIllTypingOperatorSpec(operator=ArgumentSwapSpec()),
        WeightedIllTypingOperatorSpec(operator=OverApplicationSpec()),
        WeightedIllTypingOperatorSpec(operator=SortMismatchSpec()),
    ),
    attempts: int = 64,
    regime: NegativeDataRegime = "labeled-negatives",
    noise: float = 0.0,
) -> IllTypingSpec:
    return IllTypingSpec(
        proportion=CorruptionProportionSpec(value=proportion),
        operators=IllTypingOperatorsSpec(entries=operators),
        attempts=attempts,
        regime=regime,
        noise=noise,
    )


def _noise(grammatical: float = 0.0, ungrammatical: float = 0.0) -> LabelNoiseSpec:
    return LabelNoiseSpec(
        grammatical=LabelNoiseRateSpec(rate=grammatical),
        ungrammatical=LabelNoiseRateSpec(rate=ungrammatical),
    )


class SpecParser(ParserSpec):
    kind: Literal["synthetic-spec-parser"] = "synthetic-spec-parser"


class SpecLoss(LossSpec):
    kind: Literal["synthetic-spec-loss"] = "synthetic-spec-loss"


class SpecFeaturizer(FeaturizerSpec):
    kind: Literal["synthetic-spec-featurizer"] = "synthetic-spec-featurizer"


class SpecTrainer(TrainerSpec):
    kind: Literal["synthetic-spec-trainer"] = "synthetic-spec-trainer"


class SpecOptimizer(OptimizerSpec):
    kind: Literal["synthetic-spec-optimizer"] = "synthetic-spec-optimizer"


def _declaration(
    *,
    grammar: GrammarDecl | None = None,
    size: int = 12,
    seed: int = 5,
    length: LengthSpec | None = None,
    ambiguity: AmbiguitySpec | None = None,
    scope: ScopeSpec | None = None,
    shape: ShapeScheduleSpec | None = None,
    corruption: CorruptionSpec | None = None,
    noise: LabelNoiseSpec | None = None,
    sampler: DerivationSamplerSpec | None = None,
) -> SyntheticCorpusDecl:
    return SyntheticCorpusDecl(
        retained={"validation": 0.25},
        grammar=_grammar() if grammar is None else grammar,
        language=language(),
        size=size,
        seed=seed,
        length=UniformLengthSpec(min_length=1, max_length=4)
        if length is None
        else length,
        sampler=(UniformDerivationSamplerSpec() if sampler is None else sampler),
        ambiguity=ambiguity,
        scope=scope,
        shape=shape,
        corruption=corruption,
        noise=noise,
    )


AMBIGUITY: Final = AmbiguitySpec(
    derivations=TargetSpec(value=1.0, tolerance=1.0), attempts=20
)
CORRUPTION: Final = _corruption()
NOISE: Final = _noise(0.2, 0.2)


def _full() -> SyntheticCorpusDecl:
    return _declaration(
        size=40, ambiguity=AMBIGUITY, corruption=CORRUPTION, noise=NOISE
    )


def test_a_synthetic_corpus_declares_exact_retained_allocations() -> None:
    declaration = _declaration().with_(retained={"dev": 0.2, "test": 0.2})

    assert declaration.retained_splits() == frozenset(("dev", "test"))

    with pytest.raises(dx.ValidationError, match="retained declares"):
        _declaration().with_(retained={})


def test_a_synthetic_corpus_declares_its_training_split() -> None:
    assert _declaration().training_split == "train"

    declaration = _declaration().with_(training_split="fit")
    lowerer = SyntheticCorpusLowerer()
    dataset = lowerer.lower(lowerer.check(declaration), FOREST_POLICY)

    assert {example.item.split for example in dataset.corpus.examples} <= {
        "fit",
        "validation",
    }
    assert "fit" in {example.item.split for example in dataset.corpus.examples}

    with pytest.raises(dx.ValidationError, match="is the training split"):
        _declaration().with_(training_split="validation")


def _synthetic(spec: DatasetSpec) -> SyntheticCorpusDecl:
    assert isinstance(spec, SyntheticCorpusDecl)

    return spec


def _run_document(dataset: JsonValue | None) -> dict[str, JsonValue]:
    train: dict[str, JsonValue] = {"featurizer": {"kind": "synthetic-spec-featurizer"}}

    if dataset is not None:
        train["dataset"] = dataset

    return {
        "model": {
            "language": {"from": "data"},
            "parser": {"kind": "synthetic-spec-parser"},
        },
        "loss": {"terms": [{"objective": {"kind": "synthetic-spec-loss"}}]},
        "data": {"datasets": {"train": train}},
        "trainer": {"kind": "synthetic-spec-trainer"},
        "optimizer": {"kind": "synthetic-spec-optimizer"},
        "parse": {"retain": "forest", "reachability": "all"},
        "release": {"checkpoint": "validation-best"},
        "device_policy": {
            "device": "cpu",
            "compute_dtype": "float32",
            "deterministic": True,
            "matmul_precision": "highest",
            "activation_memory_bytes": 8_000_000_000,
        },
    }


def _write(path: Path, document: JsonValue) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")

    return path


def _fragment(declaration: SyntheticCorpusDecl) -> dict[str, JsonValue]:
    return cast("dict[str, JsonValue]", loads(canonical_bytes(declaration)))


PROBE: Final = """
import sys
from lofi.data.synthetic.spec.corpus import SyntheticCorpusDecl, SyntheticCorpusIso
from lofi.deduce import ParsePolicy, Reachability, Retain

sys.path.insert(0, sys.argv[2])
from conftest import _projection

declaration = SyntheticCorpusDecl.model_validate_json(open(sys.argv[1], "rb").read())
policy = ParsePolicy(Retain.FOREST, Reachability.ALL)
print(_projection(SyntheticCorpusIso(policy=policy).forward(declaration)))
"""


def _declarations() -> st.SearchStrategy[SyntheticCorpusDecl]:
    return st.builds(
        _declaration,
        size=st.integers(min_value=0, max_value=6),
        seed=st.integers(min_value=0, max_value=1_000),
        noise=st.one_of(st.none(), st.just(_noise(0.5, 0.5))),
    )


@pytest.mark.conformance
@pytest.mark.parametrize("law", ISO_LAWS, ids=ISO_LAW_NAMES)
def test_the_lowering_holds_to_each_iso_law(law: IsoLaw) -> None:
    law(
        IsoFixture(
            SyntheticCorpusIso(policy=FOREST_POLICY),
            sources=_declarations(),
            targets=_declarations().map(
                SyntheticCorpusIso(policy=FOREST_POLICY).forward
            ),
            max_examples=6,
        )
    )


def test_the_declaration_is_the_synthetic_dataset_variant() -> None:
    declaration = _declaration()

    assert isinstance(declaration, DatasetSpec)
    assert declaration.kind == "synthetic"
    assert DatasetSpec.model_validate_json(canonical_bytes(declaration)) == declaration

    with pytest.raises(TypeError, match="already registered"):

        class Duplicate(DatasetSpec):
            kind: Literal["synthetic"] = "synthetic"


def test_the_declaration_yields_sentences_to_a_trainer() -> None:
    assert _declaration().training_batches() == frozenset({TrainingBatchKind.SENTENCES})


def test_an_interpreted_declaration_yields_its_certified_gold_forms() -> None:
    assert _interpreted().training_batches() == frozenset(
        {TrainingBatchKind.SENTENCES, TrainingBatchKind.GOLD_FORMS}
    )


def test_canonical_bytes_are_compact_ascii_and_rebuild_the_declaration() -> None:
    declaration = _full()
    encoded = canonical_bytes(declaration)

    assert b'": ' not in encoded
    assert b", " not in encoded
    assert b"\n" not in encoded
    assert encoded.isascii()
    assert SyntheticCorpusDecl.model_validate_json(encoded) == declaration
    assert canonical_bytes(SyntheticCorpusDecl.model_validate_json(encoded)) == encoded
    assert fingerprint(declaration) == sha256(encoded).hexdigest()


def test_the_same_declaration_lowers_to_the_same_corpus_in_two_lowerings() -> None:
    declaration = _full()
    first = SyntheticCorpusIso(policy=FOREST_POLICY).forward(declaration)
    again = SyntheticCorpusIso(policy=FOREST_POLICY).forward(declaration)

    assert first == again
    assert hash(first) == hash(again)
    assert first.fingerprint == again.fingerprint == fingerprint(declaration)
    assert first.declaration == again.declaration == canonical_bytes(declaration)
    assert _projection(first) == _projection(again)
    assert first.corpus.corrupted
    assert first.corpus.noise is not None
    assert first.corpus.noise.flip_count > 0


def test_two_runs_lower_the_same_declaration_to_byte_identical_corpora(
    tmp_path: Path,
) -> None:
    declaration = _full()
    path = tmp_path / "declaration.json"
    path.write_bytes(canonical_bytes(declaration))
    here = str(Path(__file__).parent)

    def run() -> str:
        completed = subprocess.run(
            [sys.executable, "-c", PROBE, str(path), here],
            check=True,
            capture_output=True,
            encoding="utf-8",
            timeout=TIMEOUT_SECONDS,
        )

        return completed.stdout.strip()

    first = run()
    again = run()

    assert first == again
    assert first == _projection(
        SyntheticCorpusIso(policy=FOREST_POLICY).forward(declaration)
    )


CHANGES: Final[dict[str, Change]] = {
    "seed": lambda held: held.with_(seed=fixed_seed("seed", held.seed) + 1),
    "size": lambda held: held.with_(size=held.size + 1),
    "length-bound": lambda held: held.with_(
        length=UniformLengthSpec(min_length=1, max_length=5)
    ),
    "length-kind": lambda held: held.with_(
        length=NegativeBinomialLengthSpec(r=2.0, p=0.6, min_length=1, max_length=4)
    ),
    "ambiguity-attempts": lambda held: held.with_(
        ambiguity=AmbiguitySpec(
            derivations=TargetSpec(value=1.0, tolerance=1.0), attempts=21
        )
    ),
    "ambiguity-target": lambda held: held.with_(
        ambiguity=AmbiguitySpec(
            derivations=TargetSpec(value=1.0, tolerance=0.0), attempts=20
        )
    ),
    "corruption-proportion": lambda held: held.with_(corruption=_corruption(0.5)),
    "corruption-kinds": lambda held: held.with_(
        corruption=_corruption(
            operators=(WeightedCorruptionOperatorSpec(operator=SwapSpec()),)
        )
    ),
    "corruption-attempts": lambda held: held.with_(corruption=_corruption(attempts=51)),
    "noise-grammatical": lambda held: held.with_(noise=_noise(0.0, 0.2)),
    "noise-ungrammatical": lambda held: held.with_(noise=_noise(0.2, 0.0)),
    "grammar-name": lambda held: held.with_(grammar=_grammar("renamed")),
    "grammar-lexicon": lambda held: held.with_(grammar=_grammar(forms=5)),
}


@pytest.mark.parametrize("change", CHANGES.values(), ids=CHANGES.keys())
def test_changing_any_field_changes_the_fingerprint_and_the_corpus(
    change: Change,
) -> None:
    declaration = _full()
    changed = change(declaration)
    iso = SyntheticCorpusIso(policy=FOREST_POLICY)
    before = iso.forward(declaration)
    after = iso.forward(changed)

    assert changed != declaration
    assert canonical_bytes(changed) != canonical_bytes(declaration)
    assert after.fingerprint != before.fingerprint
    assert after != before
    assert after.corpus != before.corpus


def test_every_report_is_reachable_from_the_lowered_dataset() -> None:
    declaration = _full()
    dataset = SyntheticCorpusIso(policy=FOREST_POLICY).forward(declaration)
    corpus = dataset.corpus

    assert len(dataset) == declaration.size
    assert corpus.seed == declaration.seed
    assert (
        corpus.length.distribution
        == cast("_LengthDeclaration", declaration.length).build()
    )
    assert corpus.length.achieved.total == len(corpus.examples)
    assert corpus.ambiguity.target == AMBIGUITY.build()
    assert corpus.ambiguity.derivations.total == len(corpus.examples)
    assert corpus.corruption is not None
    assert (
        corpus.corruption.requested
        == SyntheticCorpusLowerer().check(declaration).corruption
    )
    assert corpus.corruption.sample_size == declaration.size
    assert corpus.noise is not None
    assert corpus.noise.requested == NOISE.build()
    assert corpus.noise.sample_size == declaration.size
    assert set(corpus.lexicon.class_sizes) == {
        "NP",
        "S\\NP",
        "S\\NP/NP",
        "NP/NP",
    }
    assert corpus.lexicon.rank_frequency_exponent is not None
    assert (
        corpus.knobs
        == SyntheticGenerator(dataset.source_grammar, policy=FOREST_POLICY).knobs
    )


def test_an_undeclared_control_leaves_its_report_absent() -> None:
    corpus = SyntheticCorpusIso(policy=FOREST_POLICY).forward(_declaration()).corpus

    assert corpus.ambiguity.target is None
    assert corpus.corruption is None
    assert corpus.corrupted == ()
    assert corpus.noise is None
    assert all(not label.flipped for label in corpus.labels)


def test_the_iso_serves_one_dataset_per_declaration_and_shares_grammars() -> None:
    grammars = GrammarIso()
    iso = SyntheticCorpusIso(grammars, policy=FOREST_POLICY)
    declaration = _declaration()
    dataset = iso.forward(declaration)

    assert iso.forward(_declaration()) is dataset
    assert iso.forward(_declaration(seed=6)) is not dataset
    assert dataset.source_grammar is grammars.forward(
        declare_grammar(declaration.grammar), declaration.language
    )
    assert iso.forward(_declaration(seed=6)).source_grammar is dataset.source_grammar
    assert iso.backward(dataset) == declaration
    assert iso.backward(dataset) is not declaration
    assert SyntheticCorpusIso(policy=FOREST_POLICY).grammars is not grammars
    assert iso.grammars is grammars


def test_the_lowerer_names_its_route() -> None:
    lowerer = SyntheticCorpusLowerer()

    assert lowerer.route == dx.LoweringRoute(SOURCE_VERSION, TARGET_VERSION)
    assert SOURCE_VERSION != TARGET_VERSION

    dataset = lowerer.lower(lowerer.check(_declaration()), FOREST_POLICY)

    assert dataset.source_version == SOURCE_VERSION
    assert dataset.target_version == TARGET_VERSION


def test_the_lowerer_stages_build_the_request_values_once() -> None:
    lowerer = SyntheticCorpusLowerer()
    declaration = _full()
    checked = lowerer.check(declaration)

    assert checked.source is declaration
    assert checked.declaration == canonical_bytes(declaration)
    assert checked.source_grammar is lowerer.grammars.forward(
        declare_grammar(declaration.grammar), declaration.language
    )
    assert checked.length == UniformLength(1, 4)
    assert checked.ambiguity == AmbiguityTarget(
        derivations=Target(1.0, 1.0), attempts=20
    )
    assert checked.corruption is not None
    assert checked.corruption.proportion == 0.25
    assert checked.corruption.operator_names == ("swap", "deletion")
    assert checked.corruption.attempts == 50
    assert checked.noise == LabelNoise(grammatical=0.2, ungrammatical=0.2)

    dataset = lowerer.lower(checked, FOREST_POLICY)

    assert dataset == SyntheticCorpusIso(policy=FOREST_POLICY).forward(declaration)

    lowerer.validate(dataset)


@pytest.mark.parametrize(
    ("seed", "size", "length", "reason"),
    [
        (6, 12, UniformLength(1, 4), "another seed"),
        (5, 11, UniformLength(1, 4), "another number of items"),
        (5, 12, UniformLength(1, 3), "another length distribution"),
    ],
)
def test_the_lowerer_refuses_a_dataset_that_disagrees_with_its_declaration(
    seed: int, size: int, length: UniformLength, reason: str
) -> None:
    lowerer = SyntheticCorpusLowerer()
    declaration = _declaration()
    assembly = lowerer.grammars.forward(
        declare_grammar(declaration.grammar), declaration.language
    )
    corpus = SyntheticGenerator(assembly, policy=FOREST_POLICY).sample(
        size, seed=seed, length=length
    )
    dataset = SyntheticDataset(
        canonical_bytes(declaration),
        assembly,
        corpus,
        source_version=SOURCE_VERSION,
        target_version=TARGET_VERSION,
    )

    with pytest.raises(DeclarationError, match=reason):
        lowerer.validate(dataset)


def test_the_lowerer_refuses_a_dataset_on_another_route() -> None:
    lowerer = SyntheticCorpusLowerer()
    checked = lowerer.check(_declaration())
    lowered = lowerer.lower(checked, FOREST_POLICY)
    dataset = SyntheticDataset(
        lowered.declaration,
        lowered.source_grammar,
        lowered.corpus,
        source_version=SOURCE_VERSION,
        target_version="lofi-synthetic-corpus-native-0",
    )

    with pytest.raises(DeclarationError, match="another route"):
        lowerer.validate(dataset)


def test_a_declaration_fragment_composes_through_the_config_engine(
    tmp_path: Path,
) -> None:
    declaration = _full()
    _write(
        tmp_path / "data" / "datasets" / "train" / "dataset" / "tiny.json",
        _fragment(declaration),
    )
    _write(
        tmp_path / "run.json",
        {"defaults": [{"data.datasets.train.dataset": "tiny"}], **_run_document(None)},
    )
    _write(
        tmp_path / "profiles" / "reseeded.json",
        {"data": {"datasets": {"train": {"dataset": {"seed": 9}}}}},
    )
    loader = SpecLoader(search_path=(tmp_path,))

    composed = loader.compose_traced(tmp_path / "run.json", schema=RunSpec).spec

    assert composed.data.datasets["train"].dataset == declaration

    reseeded = loader.compose_traced(
        tmp_path / "run.json",
        schema=RunSpec,
        profile="reseeded",
        overrides=(
            "data.datasets.train.dataset.size=8",
            "data.datasets.train.dataset.noise.ungrammatical.rate=0.1",
        ),
    ).spec

    assert reseeded.data.datasets["train"].dataset == declaration.with_(
        seed=9, size=8, noise=_noise(0.2, 0.1)
    )

    dataset = SyntheticCorpusIso(policy=FOREST_POLICY).forward(
        _synthetic(reseeded.data.datasets["train"].dataset)
    )

    assert len(dataset) == 8
    assert dataset.corpus.seed == 9


def test_an_inline_declaration_composes_alongside_the_run() -> None:
    declaration = _declaration()
    composed = RunSpec.model_validate(_run_document(_fragment(declaration)))

    assert composed.data.datasets["train"].dataset == declaration


def test_the_fingerprint_enters_the_experiment_manifest() -> None:
    declaration = _full()
    dataset = SyntheticCorpusIso(policy=FOREST_POLICY).forward(declaration)
    run_spec = RunSpec.model_validate(_run_document(_fragment(declaration)))
    derived = ExperimentSeed(fixed_seed("seed", declaration.seed)).point(
        "0123456789abcdef", 0
    )
    manifest = open_manifest(
        run_spec=run_spec,
        component_fingerprints={
            role: f"fingerprint:{role}" for role in run_spec.component_roles()
        },
        grammar_fingerprint=dataset.source_grammar.grammar.fingerprint,
        signature_fingerprint=dataset.source_grammar.language.codec.index.digest,
        corpus_revision=dataset.fingerprint,
        seed_policy=derived.policy(accelerator=None, samplers=()),
        seed_derivation=derived,
        package_versions={"lofi": "0.1.0"},
        git_revision="0123456789abcdef",
        accelerator="cpu",
    )
    encoded = ManifestIso().forward(manifest)
    decoded = ManifestIso().backward(encoded)

    named = _synthetic(decoded.run_spec.data.datasets["train"].dataset)

    assert decoded.corpus_revision == dataset.fingerprint
    assert named == declaration
    assert canonical_bytes(named) == dataset.declaration
    assert fingerprint(named) == decoded.corpus_revision
    assert SyntheticCorpusIso(policy=FOREST_POLICY).forward(named) == dataset


def test_the_length_variants_lower_to_the_interior_distributions() -> None:
    uniform = UniformLengthSpec(min_length=2, max_length=5)
    negative_binomial = NegativeBinomialLengthSpec(
        r=1.5,
        p=0.4,
        min_length=2,
        max_length=9,
        out_of_support="clip",
    )
    tabular = TabularLengthSpec(
        masses=(
            LengthMassSpec(length=2, mass=1.0),
            LengthMassSpec(length=5, mass=3.0),
        )
    )

    assert uniform.build() == UniformLength(2, 5)
    assert negative_binomial.build() == NegativeBinomialLength(
        1.5, 0.4, min_length=2, max_length=9, out_of_support="clip"
    )
    assert tabular.build() == TabularLength({2: 1.0, 5: 3.0})
    assert loads(uniform.model_dump_json())["kind"] == "uniform"
    assert loads(negative_binomial.model_dump_json())["kind"] == "negative-binomial"
    assert loads(tabular.model_dump_json()) == {
        "kind": "tabular",
        "masses": [
            {"length": 2, "mass": 1.0},
            {"length": 5, "mass": 3.0},
        ],
    }
    assert LengthSpec.model_validate(
        {"kind": "negative-binomial", "r": 1.5, "p": 0.4, "max_length": 9}
    ) == negative_binomial.with_(min_length=1, out_of_support="resample")
    assert (
        LengthSpec.model_validate(
            {
                "kind": "tabular",
                "masses": [
                    {"length": 2, "mass": 1.0},
                    {"length": 5, "mass": 3.0},
                ],
            }
        )
        == tabular
    )


@pytest.mark.parametrize(
    "masses",
    [
        (),
        (LengthMassSpec(length=2, mass=0.0),),
        (
            LengthMassSpec(length=2, mass=1.0),
            LengthMassSpec(length=2, mass=2.0),
        ),
    ],
)
def test_a_tabular_length_declaration_refuses_an_invalid_table(
    masses: tuple[LengthMassSpec, ...],
) -> None:
    with pytest.raises(dx.ValidationError):
        TabularLengthSpec(masses=masses)


def test_the_derivation_sampler_variants_supply_rule_application_mass() -> None:
    grammar = GrammarIso().forward(_grammar(), language()).grammar
    uniform = UniformDerivationSamplerSpec()
    weighted = WeightedRulesDerivationSamplerSpec(
        weights={"application:/": 3.0, "application:\\": 1.0}
    )
    neutral = uniform.build(grammar.rules)
    weighting = weighted.build(grammar.rules)

    assert neutral.weight("application:/") == 1.0
    assert neutral.weight("application:\\") == 1.0
    assert weighting.weight("application:/") == 3.0
    assert weighting.weight("application:\\") == 1.0
    assert loads(uniform.model_dump_json()) == {"kind": "uniform"}
    assert (
        DerivationSamplerSpec.model_validate(
            {
                "kind": "weighted-rules",
                "weights": {"application:/": 3.0, "application:\\": 1.0},
            }
        )
        == weighted
    )


@pytest.mark.parametrize(
    "weights",
    [
        {},
        {"application:/": 0.0},
        {"application:/": -1.0},
        {"application:/": inf},
        {"application:/": nan},
        {"": 1.0},
    ],
)
def test_weighted_rule_sampling_refuses_invalid_local_mass(
    weights: dict[str, float],
) -> None:
    with pytest.raises(dx.ValidationError):
        WeightedRulesDerivationSamplerSpec(weights=weights)


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({"application:/": 1.0}, "missing"),
        (
            {
                "application:/": 1.0,
                "application:\\": 1.0,
                "composition:/": 1.0,
            },
            "unknown 'composition:/'",
        ),
    ],
)
def test_weighted_rule_sampling_names_the_compiled_inventory_exactly(
    weights: dict[str, float], message: str
) -> None:
    declaration = _declaration(
        sampler=WeightedRulesDerivationSamplerSpec(weights=weights)
    )

    with pytest.raises(DeclarationError, match=message):
        SyntheticCorpusLowerer().check(declaration)


def test_the_control_specs_lower_to_the_interior_requests() -> None:
    assert AmbiguitySpec(attempts=3).build() == AmbiguityTarget(attempts=3)
    assert AmbiguitySpec(
        lexical=TargetSpec(value=1.5, tolerance=0.5),
        derivations=TargetSpec(value=2.0, tolerance=1.0),
        attempts=3,
    ).build() == AmbiguityTarget(
        lexical=Target(1.5, 0.5), derivations=Target(2.0, 1.0), attempts=3
    )
    request = (
        SyntheticCorpusLowerer()
        .check(_declaration(corruption=_corruption(0.1, attempts=4)))
        .corruption
    )
    assert request is not None
    assert request.proportion == 0.1
    assert request.operator_names == ("swap", "deletion")
    assert request.attempts == 4

    insertion = (
        SyntheticCorpusLowerer()
        .check(
            _declaration(
                corruption=_corruption(
                    0.1,
                    operators=(
                        WeightedCorruptionOperatorSpec(operator=InsertionSpec()),
                    ),
                    attempts=4,
                )
            )
        )
        .corruption
    )
    assert insertion is not None
    assert insertion.operator_names == ("insertion",)
    assert _noise(0.3, 0.3).build() == LabelNoise(grammatical=0.3, ungrammatical=0.3)
    assert _noise(ungrammatical=0.3).build() == LabelNoise(
        grammatical=0.0, ungrammatical=0.3
    )
    assert ScopeSpec(derivations=8, attempts=3).build() == ScopeTarget(
        derivations=8, attempts=3
    )
    assert ScopeSpec(
        readings=TargetSpec(value=2.0, tolerance=0.0),
        derivations=8,
        attempts=3,
        undecided="equivalent",
    ).build() == ScopeTarget(
        readings=Target(2.0, 0.0),
        derivations=8,
        attempts=3,
        undecided="equivalent",
    )
    assert ShapeTargetSpec(
        depth=TargetSpec(value=4.0, tolerance=1.0),
        right_branching=TargetSpec(value=0.75, tolerance=0.25),
        slash_balance=TargetSpec(value=0.5, tolerance=0.5),
        arity=TargetSpec(value=2.0, tolerance=0.0),
        attempts=20,
    ).build() == ShapeTarget(
        depth=Target(4.0, 1.0),
        right_branching=Target(0.75, 0.25),
        slash_balance=Target(0.5, 0.5),
        arity=Target(2.0, 0.0),
        attempts=20,
    )
    schedule = ShapeScheduleSpec(
        default=None,
        splits={
            "train": ShapeTargetSpec(
                depth=TargetSpec(value=2.0, tolerance=0.0), attempts=20
            ),
            "validation": ShapeTargetSpec(
                depth=TargetSpec(value=4.0, tolerance=0.0), attempts=20
            ),
        },
    ).build()
    assert schedule.target("train") == ShapeTarget(depth=Target(2.0, 0.0), attempts=20)
    assert schedule.target("validation") == ShapeTarget(
        depth=Target(4.0, 0.0), attempts=20
    )


def test_the_shape_target_reaches_the_draw_and_its_report() -> None:
    declaration = _declaration(
        shape=ShapeScheduleSpec(
            default=ShapeTargetSpec(
                arity=TargetSpec(value=2.0, tolerance=0.0), attempts=5
            )
        )
    )
    lowerer = SyntheticCorpusLowerer()
    checked = lowerer.check(declaration)
    dataset = lowerer.lower(checked, FOREST_POLICY)

    assert checked.shape == ShapeSchedule(
        ShapeTarget(arity=Target(2.0, 0.0), attempts=5), {}
    )
    assert dataset.corpus.shape is not None
    assert dataset.corpus.shape.target == checked.shape
    assert dataset.corpus.shape.arity == 2.0


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: TargetSpec(value=1.0, tolerance=-0.1), "tolerance"),
        (lambda: AmbiguitySpec(attempts=0), "attempts"),
        (lambda: _corruption(1.5, attempts=1), "proportion"),
        (lambda: _corruption(0.5, attempts=0), "attempts"),
        (
            lambda: _corruption(0.5, operators=(), attempts=1),
            "operator",
        ),
        (
            lambda: _corruption(
                0.5,
                operators=(
                    WeightedCorruptionOperatorSpec(operator=SwapSpec()),
                    WeightedCorruptionOperatorSpec(operator=SwapSpec()),
                ),
                attempts=1,
            ),
            "twice",
        ),
        (lambda: _noise(grammatical=-0.5), "rate must"),
        (lambda: _noise(ungrammatical=1.5), "rate must"),
        (lambda: _noise(ungrammatical=nan), "not finite"),
        (lambda: UniformLengthSpec(min_length=0, max_length=3), "min_length"),
        (lambda: UniformLengthSpec(min_length=4, max_length=3), "max_length"),
        (
            lambda: NegativeBinomialLengthSpec(r=0.0, p=0.5, max_length=3),
            "r must",
        ),
        (
            lambda: NegativeBinomialLengthSpec(r=1.0, p=1.0, max_length=3),
            "p must",
        ),
        (lambda: _declaration(size=-1), "size"),
        (lambda: ScopeSpec(derivations=0, attempts=1), "derivations"),
        (lambda: ScopeSpec(derivations=1, attempts=0), "attempts"),
        (lambda: ShapeTargetSpec(attempts=1), "names no measurement"),
        (
            lambda: ShapeTargetSpec(
                right_branching=TargetSpec(value=1.1, tolerance=0.0), attempts=1
            ),
            "right_branching",
        ),
        (
            lambda: ShapeTargetSpec(
                slash_balance=TargetSpec(value=-0.1, tolerance=0.0), attempts=1
            ),
            "slash_balance",
        ),
        (
            lambda: _declaration(scope=ScopeSpec(derivations=8, attempts=2)),
            "readings are read under an interpretation",
        ),
    ],
)
def test_the_declaration_refuses_a_setting_outside_its_range(
    build: Callable[[], dx.Model], message: str
) -> None:
    with pytest.raises(dx.ValidationError, match=message):
        build()


GENERATED: Final[dict[str, JsonValue]] = {
    "kind": "synthetic",
    "retained": {"validation": 0.1},
    "language": json.loads(registered_language("propositional").model_dump_json()),
    "grammar": {
        "formalism": "declared",
        "name": "generated",
        "signature": {
            "constructors": [
                {
                    "name": "/",
                    "kind": ["Cat", "Cat", "Cat"],
                    "notation": {
                        "style": "infix",
                        "spelling": "/",
                        "precedence": 10,
                        "associativity": "left",
                    },
                },
                {
                    "name": "\\",
                    "kind": ["Cat", "Cat", "Cat"],
                    "notation": {
                        "style": "infix",
                        "spelling": "\\",
                        "precedence": 10,
                        "associativity": "left",
                    },
                },
            ],
            "functors": [
                {
                    "name": "/",
                    "constructor": "/",
                    "argument_side": "right",
                    "opposite": "\\",
                },
                {
                    "name": "\\",
                    "constructor": "\\",
                    "argument_side": "left",
                    "opposite": "/",
                },
            ],
            "primitives": ["S", "P1", "P2", "P3"],
        },
        "rules": {
            "name": "application",
            "rules": [
                {
                    "rule": "application",
                    "view": "/",
                },
                {
                    "rule": "application",
                    "view": "\\",
                },
            ],
        },
        "lexicon": {
            "kind": "generated",
            "forms": 40,
            "categories": 8,
            "category_shares": {"kind": "power-law", "exponent": 1.0},
            "coupling": 0.0,
            "ambiguity": {
                "spelling": "parametric",
                "mean": 2.0,
                "dispersion": 0.5,
                "maximum": 4,
                "placement": 0.0,
            },
            "rank_law": {"kind": "power-law", "exponent": 1.1},
            "seed": 3,
            "max_depth": 2,
        },
        "goal": {"kind": "root", "categories": ["S"]},
        "matching": {"kind": "feature-erasure"},
        "budget": {"unary_rounds": 1},
    },
    "size": 120,
    "seed": 5,
    "length": {
        "kind": "negative-binomial",
        "min_length": 2,
        "max_length": 9,
        "r": 3.0,
        "p": 0.6,
    },
    "ambiguity": {"lexical": {"value": 1.5, "tolerance": 0.1}, "attempts": 200},
}


def test_a_literal_registered_language_name_is_converted_at_the_field() -> None:
    declared = SyntheticCorpusDecl.model_validate(
        {**GENERATED, "language": "propositional"}
    )

    assert declared.language == registered_language("propositional")


def test_a_declared_lexical_band_selects_the_drawn_strings(tmp_path: Path) -> None:
    _write(
        tmp_path / "data" / "datasets" / "train" / "dataset" / "generated.json",
        GENERATED,
    )
    _write(
        tmp_path / "run.json",
        {
            "defaults": [{"data.datasets.train.dataset": "generated"}],
            **_run_document(None),
        },
    )
    loader = SpecLoader(search_path=(tmp_path,))
    iso = SyntheticCorpusIso(policy=FOREST_POLICY)
    means: dict[float, float] = {}

    for value in (1.4, 2.0, 3.0):
        composed = loader.compose_traced(
            tmp_path / "run.json",
            schema=RunSpec,
            overrides=(f"data.datasets.train.dataset.ambiguity.lexical.value={value}",),
        ).spec
        corpus = iso.forward(_synthetic(composed.data.datasets["train"].dataset)).corpus
        means[value] = corpus.ambiguity.lexical.mean

        assert Target(value, 0.1).admits(means[value])
        assert corpus.ambiguity.rejected > 0
        assert corpus.ambiguity.target == AmbiguityTarget(
            lexical=Target(value, 0.1), attempts=200
        )

    free = iso.forward(
        _synthetic(
            loader.compose_traced(
                tmp_path / "run.json",
                schema=RunSpec,
                overrides=("data.datasets.train.dataset.ambiguity=null",),
            )
            .spec.data.datasets["train"]
            .dataset
        )
    ).corpus

    assert len(set(means.values())) == len(means)
    assert not any(
        Target(value, 0.1).admits(free.ambiguity.lexical.mean) for value in means
    )

    unmet = loader.compose_traced(
        tmp_path / "run.json",
        schema=RunSpec,
        overrides=(
            "data.datasets.train.dataset.ambiguity.lexical.value=1.2",
            "data.datasets.train.dataset.size=400",
        ),
    ).spec

    with pytest.raises(
        AmbiguityUnreachableError,
        match=(
            r"200 outside the lexical target Target\(1\.2, 0\.1\) at a mean of "
            r"[0-9.]+ categories per token"
        ),
    ):
        iso.forward(_synthetic(unmet.data.datasets["train"].dataset))


def test_the_lowerer_refuses_a_corpus_whose_lexical_mean_misses_its_band() -> None:
    lowerer = SyntheticCorpusLowerer()
    entries = (
        ("w0", "NP"),
        ("w1", "S\\NP"),
        ("w2", "(S\\NP)/NP"),
        ("w3", "NP/NP"),
        ("w3", "NP"),
    )
    declaration = _declaration(
        grammar=_grammar().with_(
            lexicon=InlineLexiconSpec(
                entries=tuple(
                    LexicalEntrySpec(form=form, category=category)
                    for form, category in entries
                )
            )
        ),
        ambiguity=AmbiguitySpec(
            lexical=TargetSpec(value=1.0, tolerance=0.0), attempts=200
        ),
    )
    checked = lowerer.check(declaration)
    lowered = lowerer.lower(checked, FOREST_POLICY)

    assert lowered.corpus.ambiguity.lexical.mean == 1
    assert lowered.corpus.ambiguity.rejected > 0

    lowerer.validate(lowered)

    drawn = SyntheticGenerator(checked.source_grammar, policy=FOREST_POLICY).sample(
        declaration.size,
        seed=fixed_seed("seed", declaration.seed),
        length=checked.length,
    )
    mean = drawn.ambiguity.lexical.mean
    dataset = SyntheticDataset(
        checked.declaration,
        checked.source_grammar,
        drawn,
        source_version=SOURCE_VERSION,
        target_version=TARGET_VERSION,
    )

    assert mean > 1

    with pytest.raises(
        DeclarationError,
        match=(
            f"the corpus has a mean of {mean} lexical categories per token, "
            r"outside the lexical target Target\(1\.0, 0\.0\)"
        ),
    ):
        lowerer.validate(dataset)


def test_a_contamination_only_declaration_flips_only_corrupted_items() -> None:
    document = _fragment(
        _declaration(
            size=400,
            seed=11,
            corruption=_corruption(0.5, attempts=100),
            noise=_noise(ungrammatical=0.3),
        )
    )
    corpus = (
        SyntheticCorpusIso(policy=FOREST_POLICY)
        .forward(SyntheticCorpusDecl.model_validate(document))
        .corpus
    )
    report = corpus.noise
    negatives = len(corpus.corrupted)

    assert report is not None
    assert report.requested == LabelNoise(grammatical=0.0, ungrammatical=0.3)
    assert negatives == 200
    assert report.flipped["grammatical"] == 0
    assert all(
        not label.flipped
        for label in corpus.labels
        if label.true_label == "grammatical"
    )
    assert abs(report.achieved_rates["ungrammatical"] - 0.3) <= 4 * (
        (0.3 * 0.7 / negatives) ** 0.5
    )
    assert report.flipped["ungrammatical"] == sum(
        1 for label in corpus.labels if label.flipped
    )


PARSER_RUN: Final = (
    Path(__file__).resolve().parents[5] / "lofi/configs/run/synthetic-parser.yaml"
)


def test_an_unlabeled_mixture_trains_through_the_run_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mixture = _write(
        tmp_path / "mixture.json",
        {
            "data": {
                "datasets": {
                    "train": {
                        "dataset": {
                            "size": 16,
                            "corruption": {
                                "proportion": {"value": 0.5, "draw": "exact"},
                                "operators": {
                                    "entries": [
                                        {
                                            "operator": {
                                                "kind": "swap",
                                                "edits": 1,
                                            },
                                            "weight": 1.0,
                                        },
                                        {
                                            "operator": {
                                                "kind": "deletion",
                                                "edits": 1,
                                            },
                                            "weight": 1.0,
                                        },
                                    ]
                                },
                                "attempts": 50,
                            },
                            "noise": {"ungrammatical": {"rate": 1.0}},
                        }
                    }
                }
            }
        },
    )
    status = run_command(
        [
            "run",
            "--config",
            str(PARSER_RUN),
            "--extra",
            str(mixture),
            "--root",
            str(tmp_path / "store"),
            "--experiment-seed",
            "5",
        ]
    )
    printed = capsys.readouterr()

    _, outcome, directory = printed.out.split()

    assert status == 0, printed.err
    assert outcome == "completed"

    resolved = RunSpec.model_validate_json(
        (Path(directory) / RESOLVED_NAME).read_text(encoding="utf-8")
    )
    corpus = (
        SyntheticCorpusIso(policy=FOREST_POLICY)
        .forward(_synthetic(resolved.data.datasets["train"].dataset))
        .corpus
    )

    assert len(corpus.corrupted) == 8
    assert corpus.noise is not None
    assert corpus.noise.flipped == {"grammatical": 0, "ungrammatical": 8}
    assert {label.observed_label for label in corpus.labels} == {"grammatical"}
    assert {label.true_label for label in corpus.labels} == {
        "grammatical",
        "ungrammatical",
    }


def test_the_declaration_refuses_an_unknown_field() -> None:
    with pytest.raises(dx.ValidationError):
        SyntheticCorpusDecl.model_validate(
            {**_fragment(_declaration()), "vocabulary_size": 4}
        )


def _scoped_declaration(
    *, readings: TargetSpec | None = None, size: int = 6
) -> SyntheticCorpusDecl:
    return SyntheticCorpusDecl(
        retained={"validation": 0.25},
        grammar=grammar("scoped", SCOPED_ENTRIES, SCOPING),
        language=language("forall", "exists").with_(
            logic=language("forall", "exists").logic.with_(prover=ABSTAINING_TPTP)
        ),
        size=size,
        seed=3,
        length=UniformLengthSpec(min_length=3, max_length=3),
        scope=ScopeSpec(readings=readings, derivations=32, attempts=6),
        interpretation=interpretation(SCOPED_INVENTORY),
    )


def test_a_declared_readings_target_reaches_the_drawn_corpus() -> None:
    declaration = _scoped_declaration(readings=TargetSpec(value=2.0, tolerance=0.0))
    checked = SyntheticCorpusLowerer().check(declaration)
    corpus = SyntheticCorpusIso(policy=FOREST_POLICY).forward(declaration).corpus
    report = corpus.scope

    assert checked.scope == ScopeTarget(
        readings=Target(2.0, 0.0), derivations=32, attempts=6
    )
    assert report is not None
    assert report.target == checked.scope
    assert report.readings == Histogram({2: 6})
    assert len(report.counts) == len(corpus.examples) == 6
    assert (
        SyntheticCorpusDecl.model_validate_json(canonical_bytes(declaration))
        == declaration
    )


def test_a_declaration_without_a_readings_band_still_reports_the_readings() -> None:
    corpus = (
        SyntheticCorpusIso(policy=FOREST_POLICY).forward(_scoped_declaration()).corpus
    )
    report = corpus.scope

    assert report is not None
    assert report.target.readings is None
    assert report.rejected == 0
    assert all(count.readings == 2 for count in report.counts)
    assert report.undecided > 0


def test_an_equivalent_undecided_policy_admits_what_the_default_refuses() -> None:
    target = TargetSpec(value=1.0, tolerance=0.0)
    distinct = _scoped_declaration(readings=target, size=1)

    with pytest.raises(ScopeUnreachableError):
        SyntheticCorpusIso(policy=FOREST_POLICY).forward(distinct)

    scope = distinct.scope
    if scope is None:
        pytest.fail("the scoped fixture declares no scope target")
    equivalent = distinct.with_(scope=scope.with_(undecided="equivalent"))
    report = SyntheticCorpusIso(policy=FOREST_POLICY).forward(equivalent).corpus.scope

    assert report is not None
    assert report.readings == Histogram({1: 1})
    assert report.rejected == 0
    assert report.undecided > 0


@pytest.mark.conformance
@pytest.mark.parametrize("law", ISO_LAWS, ids=ISO_LAW_NAMES)
def test_the_lowering_holds_to_each_iso_law_under_a_readings_target(
    law: IsoLaw,
) -> None:
    declarations = st.builds(
        _scoped_declaration, size=st.integers(min_value=1, max_value=3)
    )
    law(
        IsoFixture(
            SyntheticCorpusIso(policy=FOREST_POLICY),
            sources=declarations,
            targets=declarations.map(SyntheticCorpusIso(policy=FOREST_POLICY).forward),
            max_examples=3,
        )
    )


def test_an_ill_typing_declaration_lowers_to_the_injection_request() -> None:
    request = (
        SyntheticCorpusLowerer()
        .check(_interpreted(ill_typing=_ill_typing(attempts=9)))
        .ill_typing
    )
    assert request is not None
    assert request.proportion == 0.5
    assert request.operator_names == (
        "argument-swap",
        "over-application",
        "sort-mismatch",
    )
    assert request.attempts == 9

    noised = (
        SyntheticCorpusLowerer()
        .check(
            _interpreted(
                ill_typing=_ill_typing(
                    operators=(
                        WeightedIllTypingOperatorSpec(operator=SortMismatchSpec()),
                    ),
                    regime="label-flips",
                    noise=0.3,
                )
            )
        )
        .ill_typing
    )
    assert noised is not None
    assert noised.operator_names == ("sort-mismatch",)
    assert noised.policy == NegativeDataPolicy("label-flips", noise=0.3)


@pytest.mark.parametrize(
    ("build", "fault"),
    [
        (lambda: _ill_typing(1.5, attempts=1), "between 0 and 1"),
        (lambda: _ill_typing(nan, attempts=1), "not finite"),
        (lambda: _ill_typing(0.5, attempts=0), "must be positive"),
        (
            lambda: _ill_typing(0.5, operators=(), attempts=1),
            "names no operator",
        ),
        (
            lambda: _ill_typing(
                0.5,
                operators=(
                    WeightedIllTypingOperatorSpec(operator=SortMismatchSpec()),
                    WeightedIllTypingOperatorSpec(operator=SortMismatchSpec()),
                ),
                attempts=1,
            ),
            "names one operator twice",
        ),
        (
            lambda: _ill_typing(0.5, regime="positive-only", noise=0.2),
            "does not take label noise",
        ),
    ],
)
def test_an_ill_typing_declaration_refuses_a_bad_share_operator_list_or_budget(
    build: Callable[[], IllTypingSpec], fault: str
) -> None:
    with pytest.raises(dx.ValidationError, match=fault):
        build()


@pytest.mark.parametrize(
    "field", [{"regime": "label-flips", "noise": 0.2}, {"regime": "positive-only"}]
)
def test_a_string_corruption_refuses_a_negative_data_regime(
    field: dict[str, JsonValue],
) -> None:
    document: dict[str, JsonValue] = {
        "proportion": {"value": 0.25},
        "operators": {"entries": [{"operator": {"kind": "swap"}}]},
        **field,
    }

    with pytest.raises(dx.ValidationError, match="regime"):
        CorruptionSpec.model_validate(document)


def test_injected_forms_are_declared_only_beside_an_interpretation() -> None:
    with pytest.raises(dx.ValidationError, match="injected under an interpretation"):
        SyntheticCorpusDecl(
            retained={"validation": 0.25},
            grammar=_interpreted_grammar(),
            language=language(),
            size=4,
            seed=1,
            length=UniformLengthSpec(min_length=2, max_length=5),
            ill_typing=_ill_typing(attempts=9),
        )


def test_a_declared_injection_lowers_into_the_corpus_and_its_report() -> None:
    declaration = _interpreted(
        ill_typing=_ill_typing(
            operators=(
                WeightedIllTypingOperatorSpec(operator=ArgumentSwapSpec()),
                WeightedIllTypingOperatorSpec(operator=OverApplicationSpec()),
            ),
            attempts=40,
        )
    )
    lowerer = SyntheticCorpusLowerer()
    checked = lowerer.check(declaration)
    corpus = lowerer.lower(checked, FOREST_POLICY).corpus
    report = corpus.ill_typing

    assert checked.ill_typing is not None
    assert checked.ill_typing.proportion == 0.5
    assert checked.ill_typing.operator_names == (
        "argument-swap",
        "over-application",
    )
    assert checked.ill_typing.attempts == 40
    assert report is not None
    assert report.injection.requested == checked.ill_typing
    assert len(corpus.ill_typed) == 5
    assert report.injection.achieved_proportion == 0.5
    assert report.certificates != {}
    assert (
        corpus == SyntheticCorpusIso(policy=FOREST_POLICY).forward(declaration).corpus
    )


@pytest.mark.conformance
@pytest.mark.parametrize("law", ISO_LAWS, ids=ISO_LAW_NAMES)
def test_the_lowering_holds_to_each_iso_law_under_an_injection(law: IsoLaw) -> None:
    declarations = st.builds(
        _interpreted,
        size=st.integers(min_value=2, max_value=4),
        seed=st.integers(min_value=0, max_value=1_000),
        ill_typing=st.just(_ill_typing(attempts=40)),
    )
    law(
        IsoFixture(
            SyntheticCorpusIso(policy=FOREST_POLICY),
            sources=declarations,
            targets=declarations.map(SyntheticCorpusIso(policy=FOREST_POLICY).forward),
            max_examples=3,
        )
    )


def _sample_lexicon() -> StaticLexicon[str]:
    return StaticLexicon(
        Entry(form, category, None, log(count))
        for form, category, count in sample_rows()
    )


def _sample_estimate() -> LexiconEstimateSpec:
    return LexiconEstimateSpec.measured(_sample_lexicon(), str, sample_digest())


def _resembling_lexicon(
    forms: int, *, source: LexiconEstimateSpec | None = None, seed: int = 5
) -> ResemblingLexiconSpec:
    """Declare a lexicon drawn at a size to resemble the committed sample."""
    return ResemblingLexiconSpec(
        forms=forms,
        estimate=_sample_estimate() if source is None else source,
        seed=seed,
    )


def _resembling(
    forms: int, *, source: LexiconEstimateSpec | None = None, seed: int = 5
) -> GrammarDecl:
    """Declare a CCG whose lexicon is drawn to resemble the committed sample."""
    return _grammar("resembling").with_(
        signature=SignatureSpec(
            constructors=slashes(),
            functors=slash_views(),
            primitives=("S", "NP", "N", "conj"),
            operators=(OperatorSpec(name="NP[conj]", sort="Cat", erases_to="NP"),),
            conjunctions=ConjunctionsSpec(
                markers=("conj",),
                pairs=(ConjunctionPairSpec(unmarked="NP", marked="NP[conj]"),),
            ),
        ),
        lexicon=_resembling_lexicon(forms, source=source, seed=seed),
        budget=BudgetSpec(unary_rounds=0),
    )


def _drawn(forms: int, *, seed: int = 5) -> LexiconEstimate:
    """Measure the joint of the lexicon a resembling declaration draws."""
    assembly = GrammarIso().forward(_resembling(forms, seed=seed), language())

    return estimate(
        assembly.lexicon,
        lambda category: assembly.syntax.render(assembly.store, category),
        "drawn",
    )


def test_a_resembling_lexicon_crosses_the_native_boundary_without_rewriting() -> None:
    lexicon = _resembling_lexicon(11)
    source = _resembling(11, source=lexicon.estimate)
    assembly = GrammarIso().forward(source, language())
    categories = {
        assembly.syntax.render(assembly.store, entry.category)
        for form in assembly.lexicon.forms
        for entry in assembly.lexicon.entries(form)
    }

    assert assembly.declaration == grammar_bytes(source)
    assert len(assembly.lexicon.forms) == lexicon.forms
    assert categories <= {row.category for row in lexicon.estimate.rows}


@pytest.mark.parametrize("forms", [11, 71])
def test_a_resembling_lexicon_keeps_source_ambiguity_independent_of_form_count(
    forms: int,
) -> None:
    source = _sample_estimate().build()
    drawn = _drawn(forms)
    source_mean = sum(size for _, size, _ in source.rows) / source.forms
    drawn_mean = sum(size for _, size, _ in drawn.rows) / drawn.forms

    assert drawn.forms == forms
    assert drawn_mean == pytest.approx(source_mean, abs=1 / forms)


@pytest.mark.parametrize("forms", [11, 71])
def test_a_resembling_lexicon_keeps_the_source_s_token_shares(forms: int) -> None:
    source = _sample_estimate().build()
    drawn = _drawn(forms)

    assert {category: share for category, _, share in drawn.rows} == pytest.approx(
        {category: share for category, _, share in source.rows}
    )


def test_a_resembling_lexicon_holds_the_declared_number_of_forms() -> None:
    assembly = GrammarIso().forward(_resembling(71), language())
    source = _sample_estimate().build()
    expected_mean = sum(size for _, size, _ in source.rows) / source.forms

    assert len(assembly.lexicon.forms) == 71
    assert sum(
        len(assembly.lexicon.entries(form)) for form in assembly.lexicon.forms
    ) / 71 == pytest.approx(expected_mean, abs=1 / 71)


def test_a_size_below_the_source_drops_the_classes_it_cannot_hold() -> None:
    source = _sample_estimate().build()
    drawn = {category for category, _, _ in _drawn(5).rows}

    assert drawn < {category for category, _, _ in source.rows}


def test_the_structure_carries_the_scaled_table_and_the_fitted_rank_law() -> None:
    source = _sample_estimate()
    structure = _resembling_lexicon(71).structure()
    scaled = source.build().at(71)
    marginal = cast("_JointDeclaration", structure.joint).marginal()

    assert structure.forms == 71
    assert structure.seed == 5
    assert structure.rank_law == PowerLawRankLawSpec(exponent=source.exponent)
    assert [category for category, _ in marginal] == [
        category for category, _, _ in scaled
    ]
    assert [share for _, share in marginal] == pytest.approx(
        [share for _, _, share in scaled]
    )
    assert structure.ambiguity.mean == pytest.approx(
        sum(row.size for row in source.rows) / source.forms
    )


def test_equal_seeds_draw_equal_lexicons_and_other_seeds_do_not() -> None:
    assert _drawn(71, seed=5) == _drawn(71, seed=5)

    first = GrammarIso().forward(_resembling(71, seed=5), language()).lexicon
    second = GrammarIso().forward(_resembling(71, seed=6), language()).lexicon

    assert [first.entries(form) for form in first.forms] != [
        second.entries(form) for form in second.forms
    ]


def test_the_declaration_bytes_change_when_the_source_digest_changes() -> None:
    seeded = _declaration(grammar=_resembling(11))
    elsewhere = _declaration(
        grammar=_resembling(11, source=_sample_estimate().with_(digest="0" * 64))
    )

    assert sample_digest().encode("utf-8") in canonical_bytes(seeded)
    assert canonical_bytes(seeded) != canonical_bytes(elsewhere)
    assert fingerprint(seeded) != fingerprint(elsewhere)
    assert SyntheticCorpusDecl.model_validate_json(canonical_bytes(seeded)) == seeded


def test_a_resembling_lexicon_rebuilds_from_its_json() -> None:
    declared = _resembling_lexicon(11)

    assert LexiconSpec.model_validate_json(declared.model_dump_json()) == declared


def test_a_size_no_class_of_the_source_survives_is_refused() -> None:
    sparse = LexiconEstimateSpec(
        digest="sparse",
        forms=10,
        exponent=1.0,
        rows=(
            EstimatedClassSpec(category="N", size=4, share=0.6),
            EstimatedClassSpec(category="NP", size=3, share=0.4),
        ),
    )

    with pytest.raises(DeclarationError, match="cannot be resembled"):
        ResemblingLexiconSpec(forms=1, estimate=sparse, seed=1).structure()


def test_a_source_with_no_rank_frequency_curve_is_refused() -> None:
    with pytest.raises(DeclarationError, match="cannot be estimated"):
        LexiconEstimateSpec.measured(
            StaticLexicon((Entry("dog", "N", None, 0.0),)), str, "one"
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda source: source.with_(digest=""), "digest of its source"),
        (lambda source: source.with_(forms=0), "source lexicon holds a form"),
        (lambda source: source.with_(rows=()), "estimate needs a category"),
        (lambda source: source.with_(exponent=inf), "not finite"),
        (
            lambda source: source.with_(
                rows=(
                    EstimatedClassSpec(category="N", size=1, share=1.0),
                    EstimatedClassSpec(category="N", size=2, share=1.0),
                )
            ),
            "names a category twice",
        ),
        (
            lambda source: source.with_(
                rows=(EstimatedClassSpec(category="N", size=99, share=1.0),)
            ),
            "of the source's 34 forms",
        ),
    ],
)
def test_the_estimate_refuses_a_joint_it_cannot_hold(
    change: Callable[[LexiconEstimateSpec], LexiconEstimateSpec], message: str
) -> None:
    with pytest.raises(dx.ValidationError, match=message):
        change(_sample_estimate())


def test_a_resembling_lexicon_refuses_an_empty_inventory() -> None:
    with pytest.raises(dx.ValidationError, match="form inventory must be positive"):
        ResemblingLexiconSpec(forms=0, estimate=_sample_estimate(), seed=1)


def _derivations() -> tuple[list[str], Callable[[str], int]]:
    asked: list[str] = []

    def derive(path: str) -> int:
        asked.append(path)

        return 100 + len(asked)

    return asked, derive


def test_a_corpus_seed_is_an_integer_or_derived() -> None:
    assert _declaration().with_(seed=DERIVED_SEED).seed == "derived"

    with pytest.raises(dx.ValidationError, match="'random'"):
        _declaration().with_(seed="random")

    with pytest.raises(dx.ValidationError, match="negative"):
        _declaration().with_(seed=-2)


def test_a_derived_corpus_takes_the_seeds_derived_for_its_paths() -> None:
    asked, derive = _derivations()
    declared = _declaration(grammar=_resembling(12)).with_(seed=DERIVED_SEED)
    declared = declared.with_(
        grammar=declared.grammar.with_(
            lexicon=_resembling_lexicon(12).with_(seed=DERIVED_SEED)
        )
    )
    resolved = declared.with_derived_seeds(derive, "data.datasets.train.dataset")

    assert asked == [
        "data.datasets.train.dataset.seed",
        "data.datasets.train.dataset.grammar.lexicon.seed",
    ]
    assert resolved.seed == 101
    assert resolved.grammar.lexicon == _resembling_lexicon(12, seed=102)
    assert (
        resolved.with_derived_seeds(derive, "data.datasets.train.dataset") == resolved
    )
    assert len(asked) == 2


def test_a_fixed_corpus_is_kept_whatever_a_point_derives() -> None:
    asked, derive = _derivations()
    declared = _declaration(grammar=_resembling(12))

    assert (
        declared.with_derived_seeds(derive, "data.datasets.train.dataset") == declared
    )
    assert asked == []


def test_a_corpus_drawn_under_an_unresolved_derived_seed_is_refused() -> None:
    lowerer = SyntheticCorpusLowerer()

    with pytest.raises(UnresolvedSeedError, match="corpus's seed is derived"):
        lowerer.check(_declaration().with_(seed=DERIVED_SEED))


def test_derived_corpora_resolved_for_two_points_differ_and_fixed_ones_agree() -> None:
    experiment = ExperimentSeed(3)
    points = (experiment.point("aaaa", 0), experiment.point("bbbb", 1))
    derived = _declaration().with_(seed=DERIVED_SEED)
    fixed = _declaration()

    assert len(
        {
            fingerprint(
                derived.with_derived_seeds(
                    point.data_seed, "data.datasets.train.dataset"
                )
            )
            for point in points
        }
    ) == len(points)
    assert {
        fingerprint(
            fixed.with_derived_seeds(point.data_seed, "data.datasets.train.dataset")
        )
        for point in points
    } == {fingerprint(fixed)}
