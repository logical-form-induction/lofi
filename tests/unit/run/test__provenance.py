import json
from hashlib import sha256
from typing import TYPE_CHECKING, Final, Literal, cast

import didactic.api as dx
import pytest

from lofi.run import CommittedRun, RunHistory, TamperedRecordError
from lofi.run._provenance import identify
from lofi.spec import (
    ArchivedLanguage,
    BoundaryMismatchError,
    ConstantPatternSpec,
    DatasetSpec,
    FeaturizerSpec,
    FingerprintIso,
    IndexedLanguage,
    LossSpec,
    OptimizerSpec,
    ParserSpec,
    RunFingerprint,
    RunSpec,
    TrainerSpec,
    fingerprint_run,
)
from lofi.spec._rule import (
    PatternRuleSpec,
    RuleSetSpec,
)
from lofi.spec.grammar import (
    GRAMMAR_SOURCE_VERSION,
    GRAMMAR_TARGET_VERSION,
    BudgetSpec,
    GrammarDecl,
    GrammarIso,
    InlineLexiconSpec,
    LexicalEntrySpec,
    RootGoalSpec,
    SignatureSpec,
    archive_bytes,
)
from lofi.spec.language import registered_language

if TYPE_CHECKING:
    from pathlib import Path

ROUTE: Final = dx.LoweringRoute("decl-1", "native-1")


class ProvenanceParser(ParserSpec):
    kind: Literal["provenance-unit-parser"] = "provenance-unit-parser"


class ProvenanceLoss(LossSpec):
    kind: Literal["provenance-unit-loss"] = "provenance-unit-loss"


class ProvenanceDataset(DatasetSpec):
    kind: Literal["provenance-unit-dataset"] = "provenance-unit-dataset"


class ProvenanceFeaturizer(FeaturizerSpec):
    kind: Literal["provenance-unit-featurizer"] = "provenance-unit-featurizer"


class ProvenanceTrainer(TrainerSpec):
    kind: Literal["provenance-unit-trainer"] = "provenance-unit-trainer"


class ProvenanceOptimizer(OptimizerSpec):
    kind: Literal["provenance-unit-optimizer"] = "provenance-unit-optimizer"


CONFIGURATION: Final = (
    json.dumps(
        json.loads(
            RunSpec.model_validate(
                {
                    "model": {
                        "language": {"from": "data"},
                        "parser": {"kind": "provenance-unit-parser"},
                    },
                    "loss": {
                        "terms": [{"objective": {"kind": "provenance-unit-loss"}}]
                    },
                    "data": {
                        "datasets": {
                            "train": {
                                "dataset": {"kind": "provenance-unit-dataset"},
                                "featurizer": {"kind": "provenance-unit-featurizer"},
                            }
                        }
                    },
                    "trainer": {"kind": "provenance-unit-trainer"},
                    "parse": {"retain": "forest", "reachability": "all"},
                    "release": {"checkpoint": "resume-latest"},
                    "optimizer": {"kind": "provenance-unit-optimizer"},
                    "device_policy": {
                        "device": "cpu",
                        "compute_dtype": "float32",
                        "deterministic": True,
                        "matmul_precision": "highest",
                        "activation_memory_bytes": 8_000_000_000,
                    },
                }
            ).model_dump_json()
        ),
        indent=2,
    )
    + "\n"
).encode()


def theory(specification: bytes) -> IndexedLanguage:
    return IndexedLanguage(
        "decl-1->native-1", specification, sha256(specification).hexdigest(), ROUTE
    )


def language(specification: bytes = b"theory one") -> ArchivedLanguage:
    return ArchivedLanguage(b'{"name":"toy"}', theory(specification))


def boundary(languages: dict[str, ArchivedLanguage]) -> RunFingerprint:
    return fingerprint_run({name: held.theory for name, held in languages.items()})


def fingerprint_bytes(languages: dict[str, ArchivedLanguage]) -> bytes:
    return FingerprintIso().forward(boundary(languages))


def a_run(
    *,
    configuration: bytes = CONFIGURATION,
    languages: dict[str, ArchivedLanguage] | None = None,
    point: str = "0123456789abcdef",
) -> CommittedRun:
    held = {"grammar": language()} if languages is None else languages

    return CommittedRun(
        point,
        configuration=configuration,
        fingerprint=fingerprint_bytes(held),
        languages=held,
    )


def test_an_identifier_is_the_sha256_of_the_stored_bytes() -> None:
    assert identify(CONFIGURATION) == sha256(CONFIGURATION).hexdigest()
    assert identify(CONFIGURATION) != identify(CONFIGURATION + b" ")


def test_a_run_holds_its_bytes_exactly_as_given() -> None:
    run = a_run()

    assert run.point_id == "0123456789abcdef"
    assert run.configuration == CONFIGURATION
    assert run.fingerprint == fingerprint_bytes({"grammar": language()})
    assert run.configuration_identifier == sha256(CONFIGURATION).hexdigest()
    assert run.fingerprint_identifier == sha256(run.fingerprint).hexdigest()
    assert run.boundary == boundary({"grammar": language()})
    assert run.languages == {"grammar": language()}


def test_whitespace_alone_changes_the_configuration_identifier() -> None:
    compact = json.dumps(json.loads(CONFIGURATION)).encode()

    assert (
        a_run(configuration=compact).configuration_identifier
        != a_run().configuration_identifier
    )


def test_a_run_orders_its_languages_by_name() -> None:
    held = {"zeta": language(b"z"), "alpha": language(b"a")}

    assert list(a_run(languages=held).languages) == ["alpha", "zeta"]


def test_a_run_hands_out_languages_its_state_does_not_share() -> None:
    run = a_run()
    view = cast("dict[str, ArchivedLanguage]", run.languages)
    view.clear()

    assert run.languages == {"grammar": language()}


@pytest.mark.parametrize(
    ("configuration", "fingerprint", "error"),
    [
        (b"", None, ValueError),
        (cast("bytes", "text"), None, TypeError),
        (CONFIGURATION, b"", ValueError),
        (CONFIGURATION, b"{}", ValueError),
        (CONFIGURATION, b"not json", ValueError),
    ],
)
def test_a_run_refuses_bytes_that_are_missing_or_unreadable(
    configuration: bytes, fingerprint: bytes | None, error: type[Exception]
) -> None:
    held = {"grammar": language()}

    with pytest.raises(error):
        CommittedRun(
            "0123456789abcdef",
            configuration=configuration,
            fingerprint=fingerprint_bytes(held) if fingerprint is None else fingerprint,
            languages=held,
        )


def test_a_run_refuses_an_empty_point_identifier() -> None:
    with pytest.raises(ValueError, match="point identifier"):
        a_run(point="")


def test_a_run_refuses_languages_its_fingerprint_does_not_name() -> None:
    held = {"grammar": language()}

    with pytest.raises(ValueError, match="commits languages"):
        CommittedRun(
            "p",
            configuration=CONFIGURATION,
            fingerprint=fingerprint_bytes(held),
            languages={"grammar": language(), "extra": language(b"x")},
        )

    with pytest.raises(ValueError, match="commits languages"):
        CommittedRun(
            "p",
            configuration=CONFIGURATION,
            fingerprint=fingerprint_bytes(held),
            languages={},
        )


@pytest.mark.parametrize(
    "archived",
    [
        theory(b"theory two"),
        IndexedLanguage(
            "decl-1->native-2", b"theory one", sha256(b"theory one").hexdigest(), ROUTE
        ),
        IndexedLanguage(
            "decl-1->native-1",
            b"theory one",
            sha256(b"theory one").hexdigest(),
            dx.LoweringRoute("decl-1", "native-2"),
        ),
        IndexedLanguage(
            "decl-1->native-1",
            b"theory one",
            sha256(b"theory one").hexdigest(),
            dx.LoweringRoute("decl-0", "native-1"),
        ),
    ],
    ids=["specification", "version", "route-target", "route-source"],
)
def test_a_run_refuses_a_theory_its_fingerprint_does_not_name(
    archived: IndexedLanguage,
) -> None:
    with pytest.raises(ValueError, match="not the theory"):
        CommittedRun(
            "p",
            configuration=CONFIGURATION,
            fingerprint=fingerprint_bytes({"grammar": language(b"theory one")}),
            languages={"grammar": ArchivedLanguage(b'{"name":"toy"}', archived)},
        )


def test_a_run_refuses_a_language_that_is_not_archived() -> None:
    with pytest.raises(TypeError, match="ArchivedLanguage"):
        CommittedRun(
            "p",
            configuration=CONFIGURATION,
            fingerprint=fingerprint_bytes({"grammar": language()}),
            languages={"grammar": cast("ArchivedLanguage", theory(b"theory one"))},
        )


def test_matching_identifiers_pass_and_either_mismatch_is_named() -> None:
    run = a_run()
    run.require_identifiers(
        configuration=run.configuration_identifier,
        fingerprint=run.fingerprint_identifier,
    )

    with pytest.raises(TamperedRecordError) as configuration:
        run.require_identifiers(
            configuration="0" * 64, fingerprint=run.fingerprint_identifier
        )

    with pytest.raises(TamperedRecordError) as fingerprint:
        run.require_identifiers(
            configuration=run.configuration_identifier, fingerprint="0" * 64
        )

    assert configuration.value.subject == "configuration"
    assert configuration.value.recorded == "0" * 64
    assert configuration.value.computed == run.configuration_identifier
    assert configuration.value.point_id == run.point_id
    assert fingerprint.value.subject == "fingerprint"
    assert fingerprint.value.computed == run.fingerprint_identifier


def test_persisted_bytes_are_checked_byte_for_byte() -> None:
    run = a_run()
    run.require_persisted(configuration=CONFIGURATION, fingerprint=run.fingerprint)

    with pytest.raises(TamperedRecordError) as configuration:
        run.require_persisted(
            configuration=CONFIGURATION + b" ", fingerprint=run.fingerprint
        )

    with pytest.raises(TamperedRecordError) as fingerprint:
        run.require_persisted(
            configuration=CONFIGURATION, fingerprint=run.fingerprint + b"\n"
        )

    assert configuration.value.subject == "configuration"
    assert configuration.value.recorded == run.configuration_identifier
    assert configuration.value.computed == identify(CONFIGURATION + b" ")
    assert fingerprint.value.subject == "fingerprint"


def test_a_tampered_record_error_is_a_value_error() -> None:
    error = TamperedRecordError(
        point_id="p", subject="configuration", recorded="a", computed="b"
    )

    assert isinstance(error, ValueError)
    assert "configuration of point p" in str(error)


def test_runs_compare_by_point_bytes_and_languages() -> None:
    assert a_run() == a_run()
    assert hash(a_run()) == hash(a_run())
    assert a_run() != a_run(point="fedcba9876543210")
    assert a_run() != a_run(configuration=CONFIGURATION + b"\n")
    assert a_run() != a_run(languages={"grammar": language(b"theory two")})
    assert a_run() != a_run(
        languages={
            "grammar": ArchivedLanguage(b'{"name":"other"}', theory(b"theory one"))
        }
    )
    assert a_run() != "run"
    assert "0123456789abcdef" in repr(a_run())


def node(name: str) -> ConstantPatternSpec:
    return ConstantPatternSpec(name=name)


def declaration(*, determined: bool) -> GrammarDecl:
    sentence = PatternRuleSpec(
        name="s",
        premises=(node("NP"), node("VP")),
        conclusion=node("S"),
        head_policy="pass-through",
    )
    determiner = PatternRuleSpec(
        name="np",
        premises=(node("D"), node("NP")),
        conclusion=node("NP"),
        head_policy="pass-through",
    )
    entries = (
        LexicalEntrySpec(form="john", category='("#const[Cat]" "NP")'),
        LexicalEntrySpec(form="sleeps", category='("#const[Cat]" "VP")'),
    )

    return GrammarDecl(
        name="provenance-cfg",
        signature=SignatureSpec(
            name="cfg", primitives=("S", "NP", "VP", "D"), syntax="s-expression"
        ),
        rules=RuleSetSpec(
            name="productions",
            rules=(sentence, determiner) if determined else (sentence,),
        ),
        lexicon=InlineLexiconSpec(
            entries=(
                (
                    *entries,
                    LexicalEntrySpec(form="the", category='("#const[Cat]" "D")'),
                )
                if determined
                else entries
            )
        ),
        goal=RootGoalSpec(categories=('("#const[Cat]" "S")',)),
        budget=BudgetSpec(unary_rounds=0, closure_rounds=1, closure_nodes=16),
    )


def lowered(source: GrammarDecl) -> ArchivedLanguage:
    assembly = GrammarIso().forward(source, registered_language("propositional"))
    artifact = assembly.grammar

    return ArchivedLanguage(
        archive_bytes(assembly),
        IndexedLanguage(
            artifact.theory_version,
            artifact.theory_specification,
            artifact.theory_digest,
            dx.LoweringRoute(GRAMMAR_SOURCE_VERSION, GRAMMAR_TARGET_VERSION),
        ),
    )


def test_a_replayed_run_builds_its_grammar_at_the_identical_boundary(
    tmp_path: Path,
) -> None:
    language = lowered(declaration(determined=True))
    history = RunHistory(tmp_path)
    history.commit(a_run(languages={"grammar": language}))
    replayed = history.replay("0123456789abcdef")

    assembly = replayed.grammar("grammar", boundary({"grammar": language}))

    assert {rule.name for rule in assembly.grammar.rules} == {"s", "np"}
    assert set(assembly.lexicon.forms) == {"john", "sleeps", "the"}


def test_a_replayed_run_refuses_grammar_reconstruction_at_another_boundary(
    tmp_path: Path,
) -> None:
    before = lowered(declaration(determined=False))
    after = lowered(declaration(determined=True))
    history = RunHistory(tmp_path)
    history.commit(a_run(languages={"grammar": before}))
    replayed = history.replay("0123456789abcdef")

    with pytest.raises(BoundaryMismatchError) as refused:
        replayed.grammar("grammar", boundary({"grammar": after}))

    assert refused.value.committed == replayed.boundary
    assert "languages.grammar" in " ".join(refused.value.members)


def test_a_replayed_run_refuses_an_unknown_language_at_its_boundary(
    tmp_path: Path,
) -> None:
    history = RunHistory(tmp_path)
    history.commit(a_run())
    replayed = history.replay("0123456789abcdef")

    with pytest.raises(KeyError):
        replayed.grammar("lexicon", replayed.boundary)
