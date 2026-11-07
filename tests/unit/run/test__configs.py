import json
from itertools import product
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Final

import pytest
from didactic.settings import UnknownKeyError, compose, load_document

from lofi.data.synthetic.spec.corpus import SyntheticCorpusDecl
from lofi.run import (
    UnknownConfigGroupError,
    configuration_documents,
    packaged_groups,
    packaged_options,
    section_root,
)
from lofi.spec import DataSection, ModelSection, RunSpec, TrainerSpec
from lofi.spec._plugins import load_plugins
from lofi.spec.grammar import GrammarIso, declare_grammar
from lofi.spec.language import registered_language
from lofi.spec.policy import DevicePolicySpec
from lofi.spec.trainers import DistanceTrainerSpec, ParserTrainerSpec, TermsTrainerSpec

if TYPE_CHECKING:
    from lofi.spec import ComponentSpec, DatasetSpec

ENCODER_SLOT: Final = "model.encoders.terms"
IDENTITY_DECODER_SLOT: Final = "model.decoders.identity"
APPLICATION_DECODER_SLOT: Final = "model.decoders.application"
TOKENS_SLOT: Final = "model.token_encoder"
PARSER_SLOT: Final = "model.parser"
POOLER_SLOT: Final = "model.subword_pooler"

DATASET_SLOT: Final = "data.datasets.train.dataset"
GRAMMAR_SLOT: Final = "data.datasets.train.dataset.grammar"
SIGNATURE_SLOT: Final = "data.datasets.train.dataset.grammar.signature"
RULES_SLOT: Final = "data.datasets.train.dataset.grammar.rules"
LEXICON_SLOT: Final = "data.datasets.train.dataset.grammar.lexicon"
LANGUAGE_SLOT: Final = "data.datasets.train.dataset.language"
CORRUPTION_OPERATORS_SLOT: Final = "data.datasets.train.dataset.corruption.operators"
CORRUPTION_PROPORTION_SLOT: Final = "data.datasets.train.dataset.corruption.proportion"
GRAMMATICAL_NOISE_SLOT: Final = "data.datasets.train.dataset.noise.grammatical"
UNGRAMMATICAL_NOISE_SLOT: Final = "data.datasets.train.dataset.noise.ungrammatical"
FEATURIZER_SLOT: Final = "data.datasets.train.featurizer"

DEVICE_SLOT: Final = "device_policy"
TRAINER_SLOT: Final = "trainer"

BLOCK_LINES: Final = 5
DIRECTED: Final = "directed"
CURRENT_CONFIG: Final = Path(__file__).parent / "fixtures" / "config_surface"


@pytest.fixture(autouse=True)
def _declarations() -> None:
    """Make every packaged component declaration available to composition."""
    load_plugins()


def tag(spec: ComponentSpec) -> str:
    """Return the registry tag a composed component was validated under."""
    return str(spec.model_dump()["kind"])


def declared_kind(slot: str, option: str) -> str:
    """Return the tag one packaged fragment writes for its slot."""
    document = load_document(section_root(slot) / f"{option}.yaml")
    kind = document.get("kind")

    if isinstance(kind, str):
        return kind

    defaults = document.get("defaults")

    if isinstance(defaults, list):
        for parent in defaults:
            if isinstance(parent, str):
                return declared_kind(slot, parent)

    msg = f"{slot}={option} declares neither a kind nor a parent fragment"
    raise ValueError(msg)


def declared_names(slot: str, option: str, field: str) -> tuple[str, ...]:
    """Return the names one packaged fragment lists under a field."""
    listed = load_document(section_root(slot) / f"{option}.yaml")[field]

    if not isinstance(listed, list):
        msg = f"{field!r} of {slot}={option} is not a list"
        raise TypeError(msg)

    return tuple(str(name) for name in listed)


def by_size(counts: dict[str, int]) -> tuple[str, ...]:
    """Return the keys of a mapping ordered by the count each holds."""
    return tuple(sorted(counts, key=lambda option: (counts[option], option)))


def synthetic(spec: DatasetSpec) -> SyntheticCorpusDecl:
    """Return a composed dataset as the synthetic corpus declaration it is."""
    if not isinstance(spec, SyntheticCorpusDecl):
        msg = f"the composed dataset is not a synthetic corpus: {tag(spec)}"
        raise TypeError(msg)

    return spec


def model_section(**selections: str) -> ModelSection:
    """Compose a model section from packaged group fragments alone."""
    groups = {"language": "data"}
    groups.update({key.replace("__", "."): name for key, name in selections.items()})
    return compose(
        None,
        schema=ModelSection,
        groups=groups,
        search_path=(section_root("model"),),
    )


def data_section(**selections: str) -> DataSection:
    """Compose a data section from packaged group fragments alone."""
    return compose(
        None,
        schema=DataSection,
        groups={
            f"datasets.train.{key.replace('__', '.')}": name
            for key, name in selections.items()
        },
        search_path=(section_root("data"),),
    )


def rooted(leaf: str) -> bool:
    """Return whether a string names a location outside the declared root."""
    posix = PurePosixPath(leaf)
    windows = PureWindowsPath(leaf)

    return (
        (leaf.startswith("~") and ("/" in leaf or "\\" in leaf))
        or windows.is_absolute()
        or (posix.is_absolute() and len(posix.parts) > 1)
    )


@pytest.mark.parametrize(
    ("leaf", "expected"),
    [
        ("s:dcl", False),
        (r"\x. p(x)", False),
        (r"C:\foo", True),
        (r"\\server\share\foo", True),
    ],
)
def test_path_detection_distinguishes_symbols_from_absolute_paths(
    leaf: str, *, expected: bool
) -> None:
    assert rooted(leaf) is expected


def leaf_strings(value: object) -> tuple[str, ...]:
    """Return every string a loaded document holds, at any depth."""
    if isinstance(value, str):
        return (value,)

    if isinstance(value, dict):
        return tuple(leaf for held in value.values() for leaf in leaf_strings(held))

    if isinstance(value, list):
        return tuple(leaf for held in value for leaf in leaf_strings(held))

    return ()


def blocks(document: Path) -> tuple[tuple[str, ...], ...]:
    """Return every window of consecutive nonblank lines of one document."""
    lines = tuple(
        line.rstrip()
        for line in document.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )

    return tuple(
        lines[start : start + BLOCK_LINES]
        for start in range(len(lines) - BLOCK_LINES + 1)
    )


def test_the_library_holds_a_group_for_every_directory_of_fragments() -> None:
    slots = packaged_groups()
    directories = {
        document.parent for document in configuration_documents(section_root(""))
    }

    assert {section_root(slot) for slot in slots} == directories
    assert list(slots) == sorted(slots)


def test_the_current_configuration_fixture_has_one_document_shape() -> None:
    (document,) = configuration_documents(CURRENT_CONFIG)
    loaded = load_document(document)

    assert document == CURRENT_CONFIG / "run.yaml"
    assert loaded["loss"] == {"terms": [{"objective": {"kind": "launcher-unit-loss"}}]}
    assert RunSpec.model_validate_json(json.dumps(loaded)).trainer.epochs == 3


def test_a_slot_the_library_does_not_hold_is_refused_with_what_it_does_hold() -> None:
    with pytest.raises(UnknownConfigGroupError) as refusal:
        packaged_options("model.optimizer")

    assert refusal.value.slot == "model.optimizer"
    assert refusal.value.available == packaged_groups()


def test_an_option_is_a_fragment_the_group_directory_holds() -> None:
    for slot in packaged_groups():
        held = {
            document.stem
            for document in configuration_documents(section_root(slot))
            if document.parent == section_root(slot)
        }

        assert set(packaged_options(slot)) == held
        assert held


def test_every_encoder_by_decoder_combination_composes_from_group_fragments() -> None:
    combinations = tuple(
        product(
            packaged_options(ENCODER_SLOT),
            packaged_options(IDENTITY_DECODER_SLOT),
            packaged_options(APPLICATION_DECODER_SLOT),
            packaged_options(TOKENS_SLOT),
        )
    )
    composed = {
        (encoder, identity, application, tokens): model_section(
            parser="inside-outside",
            subword_pooler="mean",
            encoders__terms=encoder,
            decoders__identity=identity,
            decoders__application=application,
            token_encoder=tokens,
        )
        for encoder, identity, application, tokens in combinations
    }

    assert len(composed) == len(combinations)

    for (encoder, identity, application, tokens), section in composed.items():
        assert section.token_encoder is not None
        assert tag(section.encoders["terms"]) == declared_kind(ENCODER_SLOT, encoder)
        assert tag(section.decoders["identity"]) == declared_kind(
            IDENTITY_DECODER_SLOT, identity
        )
        assert tag(section.decoders["application"]) == declared_kind(
            APPLICATION_DECODER_SLOT, application
        )
        assert tag(section.token_encoder) == declared_kind(TOKENS_SLOT, tokens)


def test_every_packaged_parser_and_pooler_option_composes() -> None:
    for parser, pooler in product(
        packaged_options(PARSER_SLOT), packaged_options(POOLER_SLOT)
    ):
        section = model_section(
            parser=parser,
            subword_pooler=pooler,
            encoders__terms="hrr",
            decoders__identity="hrr",
            decoders__application="hrr",
        )

        assert section.parser is not None
        assert section.subword_pooler is not None
        assert tag(section.parser) == declared_kind(PARSER_SLOT, parser)
        assert tag(section.subword_pooler) == declared_kind(POOLER_SLOT, pooler)


def test_each_grammar_manipulation_composes_and_compiles() -> None:
    iso = GrammarIso()
    manipulations = product(
        packaged_options(SIGNATURE_SLOT),
        packaged_options(RULES_SLOT),
        packaged_options(LEXICON_SLOT),
    )
    operators: dict[str, int] = {}
    digests: set[str] = set()
    crossings = 0

    for signature, rules, lexicon in manipulations:
        section = data_section(
            dataset="synthetic",
            featurizer="span-constraint",
            dataset__grammar=DIRECTED,
            dataset__grammar__signature=signature,
            dataset__grammar__rules=rules,
            dataset__grammar__lexicon=lexicon,
        )
        declared = synthetic(section.datasets["train"].dataset)
        grammar = declare_grammar(declared.grammar)
        built = iso.forward(grammar, declared.language)
        operators[signature] = len(built.grammar.signature.operators)
        digests.add(built.declaration_digest)
        crossings += 1

        assert grammar.rules.name == str(
            load_document(section_root(RULES_SLOT) / f"{rules}.yaml")["name"]
        )

    assert len(digests) == crossings
    assert by_size(operators) == by_size(
        {
            option: len(declared_names(SIGNATURE_SLOT, option, "primitives"))
            for option in operators
        }
    )


def test_each_constructor_inventory_composes_and_compiles() -> None:
    iso = GrammarIso()

    for option in packaged_options(GRAMMAR_SLOT):
        selections = {
            "datasets.train.dataset": "synthetic",
            "datasets.train.featurizer": "span-constraint",
            "datasets.train.dataset.grammar": option,
        }

        if option == DIRECTED:
            selections |= {
                "datasets.train.dataset.grammar.signature": "narrow",
                "datasets.train.dataset.grammar.rules": "application",
                "datasets.train.dataset.grammar.lexicon": "unambiguous",
            }

        section = compose(
            None,
            schema=DataSection,
            groups=selections,
            search_path=(section_root("data"),),
        )
        declared = synthetic(section.datasets["train"].dataset)
        built = iso.forward(declare_grammar(declared.grammar), declared.language)

        assert built.grammar.rules.rules


def test_each_packaged_language_composes_field_by_field() -> None:
    languages = {
        option: synthetic(
            data_section(
                dataset="synthetic",
                featurizer="span-constraint",
                dataset__grammar="undirected",
                dataset__language=option,
            )
            .datasets["train"]
            .dataset
        ).language
        for option in packaged_options(LANGUAGE_SLOT)
    }

    registered = ("neo-davidsonian", "propositional")

    assert {name: language.name for name, language in languages.items()} == {
        name: name for name in (*registered, "choice", "dynamic", "plurals")
    }
    assert {name: languages[name] for name in registered} == {
        name: registered_language(name) for name in registered
    }


def test_the_self_contained_dataset_composes_with_every_featurizer() -> None:
    for dataset, featurizer in product(
        ("synthetic",), packaged_options(FEATURIZER_SLOT)
    ):
        selections = {"dataset": dataset, "featurizer": featurizer}
        if dataset == "synthetic":
            selections["dataset__grammar"] = "undirected"

        section = data_section(**selections)

        training = section.datasets["train"]
        assert training.featurizer is not None
        assert tag(training.dataset) == declared_kind(DATASET_SLOT, dataset)
        assert tag(training.featurizer) == declared_kind(FEATURIZER_SLOT, featurizer)


def test_every_synthetic_corruption_and_noise_option_composes() -> None:
    options = product(
        packaged_options(CORRUPTION_OPERATORS_SLOT),
        packaged_options(CORRUPTION_PROPORTION_SLOT),
        packaged_options(GRAMMATICAL_NOISE_SLOT),
        packaged_options(UNGRAMMATICAL_NOISE_SLOT),
        packaged_options(FEATURIZER_SLOT),
    )

    for operators, proportion, grammatical, ungrammatical, featurizer in options:
        section = data_section(
            dataset="synthetic",
            featurizer=featurizer,
            dataset__grammar="undirected",
            dataset__corruption__operators=operators,
            dataset__corruption__proportion=proportion,
            dataset__noise__grammatical=grammatical,
            dataset__noise__ungrammatical=ungrammatical,
        )

        training = section.datasets["train"]
        drawn = synthetic(training.dataset)

        assert drawn.corruption is not None
        assert drawn.noise is not None
        assert training.featurizer is not None
        assert tag(training.featurizer) == declared_kind(FEATURIZER_SLOT, featurizer)
        assert drawn.corruption.attempts > 0


def test_a_fragment_for_another_dataset_variant_is_refused() -> None:
    with pytest.raises(
        UnknownKeyError,
        match="belongs to variants 'ccgbank-reskin', 'synthetic'",
    ):
        data_section(
            dataset="synthetic-inference",
            featurizer="span-constraint",
            dataset__corruption__proportion="half",
        )


def test_a_dataset_without_a_corruption_or_a_noise_selection_holds_neither() -> None:
    section = data_section(
        dataset="synthetic",
        featurizer="span-constraint",
        dataset__grammar="undirected",
    )

    drawn = synthetic(section.datasets["train"].dataset)

    assert drawn.corruption is None
    assert drawn.noise is None


def test_every_device_policy_option_validates() -> None:
    for option in packaged_options(DEVICE_SLOT):
        document = load_document(section_root(DEVICE_SLOT) / f"{option}.yaml")
        policy = DevicePolicySpec.model_validate_json(json.dumps(document))

        assert policy.device == document["device"]


def test_every_trainer_option_is_a_complete_current_declaration() -> None:
    trainers = {
        option: TrainerSpec.model_validate_json(
            json.dumps(load_document(section_root(TRAINER_SLOT) / f"{option}.yaml"))
        )
        for option in packaged_options(TRAINER_SLOT)
    }

    assert set(trainers) == {"distance", "parser", "terms"}
    parser = trainers["parser"]
    distance = trainers["distance"]
    terms = trainers["terms"]
    assert isinstance(parser, ParserTrainerSpec)
    assert isinstance(distance, DistanceTrainerSpec)
    assert isinstance(terms, TermsTrainerSpec)
    assert terms.spaces == ("categories",)
    assert parser.early_stopping.patience == 10
    assert parser.accumulation_steps == 1
    assert parser.max_grad_norm is None
    assert parser.active_groups is None
    assert distance.checkpoints.interval == 100
    assert distance.checkpoints.keep == 2
    assert parser.early_stopping.keep_best == 1


def test_no_packaged_fragment_repeats_a_block_that_appears_elsewhere() -> None:
    seen: dict[tuple[str, ...], Path] = {}
    repeated: list[tuple[Path, Path]] = []

    for document in configuration_documents(section_root("")):
        for block in blocks(document):
            if block in seen:
                repeated.append((seen[block], document))
            else:
                seen[block] = document

    assert repeated == []


def test_no_packaged_configuration_carries_an_absolute_or_home_relative_path() -> None:
    carried = [
        (document, leaf)
        for document in configuration_documents(section_root(""))
        for leaf in leaf_strings(load_document(document))
        if rooted(leaf)
    ]

    assert carried == []
