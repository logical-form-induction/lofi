from typing import TYPE_CHECKING, Literal, NamedTuple, cast

import pytest

from lofi.data import HashedResources, Inventory, ItemHasher, ItemLookup
from lofi.data.datasets import (
    DatasetPlugin,
    DatasetVocabulary,
    RefusedTraining,
    ReservedForms,
    SentenceTraining,
    artifact_dataset_plugin,
)
from lofi.deduce import ParsePolicy, Reachability, Retain
from lofi.nn._tensorize import TorchCodeTensorizer
from lofi.registry import Registry, RegistryError
from lofi.spec import DatasetSpec

if TYPE_CHECKING:
    from lofi.data._ports import CodeTensorizer, FeaturizerResources
    from lofi.spec.grammar import GrammarAssembly


class ExampleDatasetSpec(DatasetSpec):
    kind: Literal["dataset-plugin-example"] = "dataset-plugin-example"


class OtherDatasetSpec(DatasetSpec):
    kind: Literal["dataset-plugin-other"] = "dataset-plugin-other"


class CheckedExample(NamedTuple):
    declaration: bytes
    source_grammar: GrammarAssembly
    source_interpretation: None
    source_interpretation_declaration: None


class LoweredExample(NamedTuple):
    declaration: bytes
    fingerprint: str
    source_grammar: GrammarAssembly


GRAMMAR = cast("GrammarAssembly", object())
POLICY = ParsePolicy(Retain.FOREST, Reachability.ALL)
RESERVED = ReservedForms(("<pad>", "<unk>"), unknown="<unk>")
SENTENCES = cast("SentenceTraining", object())
RESOURCES = HashedResources(
    ItemHasher(Inventory("lexicon", ("known",)), Inventory("categories", ())),
    TorchCodeTensorizer(),
    ItemLookup(()),
)


def _plugin() -> DatasetPlugin[ExampleDatasetSpec, CheckedExample, LoweredExample]:
    def check(declaration: ExampleDatasetSpec) -> CheckedExample:
        return CheckedExample(
            declaration.model_dump_json().encode(), GRAMMAR, None, None
        )

    def lower(checked: CheckedExample, policy: ParsePolicy[Retain]) -> LoweredExample:
        assert policy is POLICY
        return LoweredExample(
            checked.declaration, "example-fingerprint", checked.source_grammar
        )

    def validate(dataset: LoweredExample) -> None:
        if dataset.source_grammar is not GRAMMAR:
            msg = "the lowered dataset changed grammars"
            raise ValueError(msg)

    def vocabulary(
        dataset: LoweredExample, reserved: ReservedForms
    ) -> DatasetVocabulary:
        del dataset
        return DatasetVocabulary(
            ("known",), reserved=reserved.indices, unknown=reserved.unknown
        )

    def resources(
        dataset: LoweredExample, reserved: ReservedForms, tensorizer: CodeTensorizer
    ) -> FeaturizerResources:
        del dataset, reserved
        assert isinstance(tensorizer, TorchCodeTensorizer)
        return RESOURCES

    def sentences(dataset: LoweredExample) -> SentenceTraining:
        del dataset
        return SENTENCES

    def streams(declaration: ExampleDatasetSpec) -> tuple[str, ...]:
        del declaration
        return ("example-draws",)

    return DatasetPlugin(
        spec=ExampleDatasetSpec,
        checked=CheckedExample,
        dataset=LoweredExample,
        check=check,
        lower=lower,
        validate=validate,
        vocabulary=vocabulary,
        resources=resources,
        sentences=sentences,
        pairs=RefusedTraining("the example route supplies no inference pairs"),
        streams=streams,
    )


def test_a_dataset_plugin_runs_one_checked_lowering_route() -> None:
    plugin = _plugin()
    declaration = ExampleDatasetSpec()
    checked = plugin.check(declaration)
    dataset = plugin.lower(checked, POLICY)

    plugin.validate(dataset)

    assert dataset.declaration == declaration.model_dump_json().encode()
    assert plugin.sampler_streams(declaration) == ("example-draws",)
    vocabulary = plugin.vocabulary(dataset, RESERVED)
    assert vocabulary.forms == ("known",)
    assert dict(vocabulary.reserved) == {"<pad>": 0, "<unk>": 1}
    assert vocabulary.unknown == "<unk>"
    assert plugin.resources(dataset, RESERVED, TorchCodeTensorizer()) is RESOURCES
    assert plugin.sentences(dataset) is SENTENCES


def test_a_dataset_plugin_refuses_values_from_another_route() -> None:
    plugin = _plugin()

    with pytest.raises(RegistryError, match="handles ExampleDatasetSpec"):
        plugin.check(OtherDatasetSpec())

    with pytest.raises(RegistryError, match="lowers CheckedExample"):
        plugin.lower(cast("CheckedExample", object()), POLICY)

    with pytest.raises(RegistryError, match="validates LoweredExample"):
        plugin.validate(cast("LoweredExample", object()))

    with pytest.raises(RegistryError, match="featurizer resources from"):
        plugin.resources(
            cast("LoweredExample", object()), RESERVED, TorchCodeTensorizer()
        )

    with pytest.raises(RegistryError, match="sentence training from"):
        plugin.sentences(cast("LoweredExample", object()))

    with pytest.raises(RegistryError, match="inference-pair training from"):
        plugin.pairs(cast("LoweredExample", object()))


def test_the_artifact_resolver_refuses_a_training_dataset_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = _plugin()
    monkeypatch.setattr(Registry, "get", lambda _registry, _tag: plugin)

    with pytest.raises(RegistryError, match="not name an artifact dataset plugin"):
        artifact_dataset_plugin(ExampleDatasetSpec())


def test_a_dataset_vocabulary_validates_its_owned_indices() -> None:
    with pytest.raises(ValueError, match="not reserved"):
        DatasetVocabulary(("known",), reserved={"<pad>": 0}, unknown="<unk>")

    with pytest.raises(ValueError, match="one index"):
        DatasetVocabulary(
            ("known",), reserved={"<pad>": 0, "<unk>": 0}, unknown="<unk>"
        )


def test_reserved_forms_index_their_declared_forms_in_order() -> None:
    reserved = ReservedForms(("<p>", "<bos>", "<u>"), unknown="<u>")

    assert reserved.forms == ("<p>", "<bos>", "<u>")
    assert dict(reserved.indices) == {"<p>": 0, "<bos>": 1, "<u>": 2}
    assert reserved.unknown == "<u>"
    assert reserved == ReservedForms(("<p>", "<bos>", "<u>"), unknown="<u>")


def test_a_dataset_plugin_refuses_a_training_route_it_does_not_supply() -> None:
    plugin = _plugin()
    dataset = plugin.lower(plugin.check(ExampleDatasetSpec()), POLICY)

    with pytest.raises(ValueError, match="supplies no inference pairs"):
        plugin.pairs(dataset)

    assert repr(RefusedTraining("no pairs")) == "RefusedTraining('no pairs')"
