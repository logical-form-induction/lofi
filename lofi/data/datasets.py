"""Typed dataset plugins and the values shared by their lowering routes.

A dataset plugin owns the complete route from one declaration variant to its
checked, allocation-free plan and then to its runtime dataset.  Planning sees
only the checked protocol; training asks the same plugin to lower and validate
that checked value.  Thus adding a dataset kind does not add an identity branch
to the run or training packages.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Never, Protocol, cast, final, override

from lofi._checks import ensure_name
from lofi._frozen import FrozenValue
from lofi.registry import Registry, RegistryError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import JsonValue
    from lofi.data._epochs import EpochBatching, EpochSource
    from lofi.data._ports import (
        CodeTensorizer,
        ConstraintBatcher,
        Featurizer,
        FeaturizerResources,
    )
    from lofi.data._supervision import Relational, Unlabeled
    from lofi.data.featurizers._core import FeaturizedBatch
    from lofi.data.synthetic import (
        InterpretationFunction,
        SyntheticCorpus,
    )
    from lofi.data.synthetic.training import (
        InferenceTrainingBatch,
        PairCorpus,
        SyntheticTrainingBatch,
    )
    from lofi.deduce import ParsePolicy, Retain
    from lofi.lf import InferenceLabeling
    from lofi.spec import DatasetSpec
    from lofi.spec.grammar import GrammarAssembly

__all__ = [
    "ArtifactDatasetLoader",
    "ArtifactDatasetPlugin",
    "CheckedDataset",
    "DatasetChecker",
    "DatasetLowerer",
    "DatasetPairsDeclaration",
    "DatasetPlugin",
    "DatasetResourcesDeclaration",
    "DatasetSentencesDeclaration",
    "DatasetStreamDeclaration",
    "DatasetValidator",
    "DatasetVocabulary",
    "DatasetVocabularyDeclaration",
    "LoweredDataset",
    "PairTraining",
    "RefusedTraining",
    "ReservedForms",
    "SentenceTraining",
    "artifact_dataset_plugin",
    "dataset_plugin",
]


class ArtifactDatasetLoader[S, A](Protocol):
    """Load one declared evaluator artifact at its data boundary."""

    def __call__(self, declaration: S, /) -> A:
        """Return the validated artifact named by the declaration."""
        ...


class ArtifactDatasetPlugin[S, A](FrozenValue):
    """Pair an artifact-dataset declaration with its one boundary loader."""

    __slots__ = ("_artifact_type", "_load", "_spec")

    _noun = "artifact dataset plugin"

    def __init__(
        self,
        *,
        spec: type[S],
        artifact: type[A],
        load: ArtifactDatasetLoader[S, A],
    ) -> None:
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_artifact_type", artifact)
        object.__setattr__(self, "_load", load)

    @property
    def spec(self) -> type[S]:
        """Return the declaration variant this plugin implements."""
        return self._spec

    @property
    def artifact(self) -> type[A]:
        """Return the exact artifact type this plugin loads."""
        return self._artifact_type

    def load(self, declaration: DatasetSpec, /) -> A:
        """Load a declaration of this plugin's exact variant."""
        if type(declaration) is not self._spec:
            msg = (
                f"the artifact dataset plugin handles {self._spec.__name__}, not "
                f"{type(declaration).__name__}"
            )
            raise RegistryError(msg, family="dataset")

        artifact = self._load(cast("S", declaration))

        if type(artifact) is not self._artifact_type:
            msg = (
                f"the artifact dataset plugin loads {self._artifact_type.__name__}, "
                f"not {type(artifact).__name__}"
            )
            raise RegistryError(msg, family="dataset")

        return artifact

    @override
    def __repr__(self) -> str:
        """Return the declaration and artifact types of the plugin."""
        return (
            f"ArtifactDatasetPlugin(spec={self._spec.__name__}, "
            f"artifact={self._artifact_type.__name__})"
        )


class CheckedDataset(Protocol):
    """Allocation-free dataset facts shared by planning and lowering."""

    @property
    def declaration(self) -> bytes:
        """Return the canonical declaration bytes."""
        ...

    @property
    def source_grammar(self) -> GrammarAssembly:
        """Return the compiled grammar the dataset source is read under."""
        ...

    @property
    def source_interpretation(self) -> InterpretationFunction | None:
        """Return the source interpretation, when the source declares one."""
        ...

    @property
    def source_interpretation_declaration(self) -> dict[str, JsonValue] | None:
        """Return the source interpretation declaration as typed JSON."""
        ...


class LoweredDataset(Protocol):
    """A runtime dataset produced by a dataset plugin."""

    @property
    def declaration(self) -> bytes:
        """Return the canonical declaration bytes."""
        ...

    @property
    def fingerprint(self) -> str:
        """Return the digest of the declaration bytes."""
        ...

    @property
    def source_grammar(self) -> GrammarAssembly:
        """Return the compiled grammar the dataset source was read under."""
        ...


class ReservedForms(FrozenValue):
    """The token forms a dataset section reserves ahead of its dataset's forms.

    Parameters
    ----------
    forms : tuple[str, ...]
        The reserved forms in index order.
    unknown : str
        The reserved form a token outside the vocabulary is read as.
    """

    __slots__ = ("_forms", "_unknown")

    _noun = "reserved forms"

    def __init__(self, forms: tuple[str, ...], /, *, unknown: str) -> None:
        object.__setattr__(self, "_forms", forms)
        object.__setattr__(self, "_unknown", unknown)

    @property
    def forms(self) -> tuple[str, ...]:
        """Return the reserved forms in index order."""
        return self._forms

    @property
    def indices(self) -> Mapping[str, int]:
        """Return each reserved form keyed to its index."""
        return MappingProxyType({form: index for index, form in enumerate(self._forms)})

    @property
    def unknown(self) -> str:
        """Return the reserved form an unseen token is read as."""
        return self._unknown

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether both values reserve the same forms and unknown form."""
        if type(other) is not ReservedForms:
            return NotImplemented

        return (self._forms, self._unknown) == (other._forms, other._unknown)

    @override
    def __hash__(self) -> int:
        """Return a hash consistent with equality."""
        return hash((self._forms, self._unknown))

    @override
    def __repr__(self) -> str:
        """Return the reserved forms and the unknown-token form."""
        return f"ReservedForms({self._forms!r}, unknown={self._unknown!r})"


class DatasetVocabulary(FrozenValue):
    """The token vocabulary a lowered dataset makes available to a model."""

    __slots__ = ("_forms", "_reserved", "_unknown")

    _noun = "dataset vocabulary"

    def __init__(
        self,
        forms: tuple[str, ...],
        /,
        *,
        reserved: Mapping[str, int],
        unknown: str,
    ) -> None:
        named_forms = tuple(ensure_name("a vocabulary form", form) for form in forms)

        if len(set(named_forms)) != len(named_forms):
            msg = "a dataset vocabulary names one form twice"
            raise ValueError(msg)

        named_unknown = ensure_name("an unknown-token form", unknown)
        held = {
            ensure_name("a reserved-token form", form): index
            for form, index in reserved.items()
        }

        if any(type(index) is not int or index < 0 for index in held.values()):
            msg = "a reserved-token index is not a nonnegative integer"
            raise ValueError(msg)

        if len(set(held.values())) != len(held):
            msg = "two reserved-token forms have one index"
            raise ValueError(msg)

        if named_unknown not in held:
            msg = f"unknown-token form {named_unknown!r} is not reserved"
            raise ValueError(msg)

        object.__setattr__(self, "_forms", named_forms)
        object.__setattr__(self, "_reserved", MappingProxyType(held))
        object.__setattr__(self, "_unknown", named_unknown)

    @property
    def forms(self) -> tuple[str, ...]:
        """Return the ordinary forms in stable order."""
        return self._forms

    @property
    def reserved(self) -> Mapping[str, int]:
        """Return the reserved forms keyed by their indices."""
        return self._reserved

    @property
    def unknown(self) -> str:
        """Return the reserved form used for an unseen token."""
        return self._unknown

    @override
    def __repr__(self) -> str:
        """Return the complete vocabulary declaration."""
        return (
            f"DatasetVocabulary({self._forms!r}, reserved={dict(self._reserved)!r}, "
            f"unknown={self._unknown!r})"
        )


class DatasetChecker[S, C: CheckedDataset](Protocol):
    """Check one dataset declaration without drawing or reading its items."""

    def __call__(self, declaration: S, /) -> C:
        """Return the checked declaration."""
        ...


class DatasetLowerer[C: CheckedDataset, D: LoweredDataset](Protocol):
    """Lower one checked declaration to its runtime dataset."""

    def __call__(self, checked: C, policy: ParsePolicy[Retain], /) -> D:
        """Return the dataset lowered under one symbolic parse policy."""
        ...


class DatasetValidator[D: LoweredDataset](Protocol):
    """Validate one lowered dataset against its checked route."""

    def __call__(self, dataset: D, /) -> None:
        """Refuse a dataset inconsistent with its declaration."""
        ...


class DatasetVocabularyDeclaration[D: LoweredDataset](Protocol):
    """Declare the token vocabulary of one lowered dataset."""

    def __call__(self, dataset: D, reserved: ReservedForms, /) -> DatasetVocabulary:
        """Return the dataset's token vocabulary after the reserved forms."""
        ...


class DatasetResourcesDeclaration[D: LoweredDataset](Protocol):
    """Declare the resources a featurizer over one lowered dataset is built from."""

    def __call__(
        self, dataset: D, reserved: ReservedForms, tensorizer: CodeTensorizer, /
    ) -> FeaturizerResources:
        """Return the dataset's encoder and item lookup over the given tensorizer."""
        ...


class SentenceTraining(Protocol):
    """The sentence occurrences one lowered dataset trains a parser on."""

    @property
    def corpus(self) -> SyntheticCorpus:
        """Return the sentence corpus the epochs are drawn from."""
        ...

    @property
    def source_grammar(self) -> GrammarAssembly:
        """Return the compiled grammar the corpus was read under."""
        ...

    @property
    def source_interpretation(self) -> InterpretationFunction | None:
        """Return the interpretation the corpus was read under, when declared."""
        ...

    def epochs[ConstraintT](
        self,
        featurizer: Featurizer[Unlabeled, FeaturizedBatch[Unlabeled]],
        constraints: ConstraintBatcher[ConstraintT] | None,
        interpretation: InterpretationFunction | None,
        batching: EpochBatching,
        /,
    ) -> EpochSource[SyntheticTrainingBatch[ConstraintT]]:
        """Return the seeded parser epochs over the corpus's split occurrences."""
        ...


class PairTraining(Protocol):
    """The labeled inference pairs one lowered dataset trains a grounding on."""

    @property
    def corpus(self) -> PairCorpus:
        """Return the pair corpus the epochs are drawn from."""
        ...

    @property
    def labeling(self) -> InferenceLabeling:
        """Return the label vocabulary shared by training and evaluation."""
        ...

    def epochs(
        self,
        featurizer: Featurizer[Relational, FeaturizedBatch[Relational]],
        batching: EpochBatching,
        /,
    ) -> EpochSource[InferenceTrainingBatch]:
        """Return the seeded epochs over the decided pairs of each split."""
        ...


class DatasetSentencesDeclaration[D: LoweredDataset](Protocol):
    """Declare the sentence training one lowered dataset supplies."""

    def __call__(self, dataset: D, /) -> SentenceTraining:
        """Return the dataset's sentence training."""
        ...


class DatasetPairsDeclaration[D: LoweredDataset](Protocol):
    """Declare the inference-pair training one lowered dataset supplies."""

    def __call__(self, dataset: D, /) -> PairTraining:
        """Return the dataset's inference-pair training."""
        ...


@final
class RefusedTraining(FrozenValue):
    """Refuse a training route a dataset kind does not supply, naming why.

    Parameters
    ----------
    reason : str
        The refusal message.
    """

    __slots__ = ("_reason",)

    _noun = "refused training route"

    def __init__(self, reason: str, /) -> None:
        object.__setattr__(self, "_reason", reason)

    def __call__(self, dataset: LoweredDataset, /) -> Never:
        """Raise the declared refusal.

        Raises
        ------
        ValueError
            Always.
        """
        del dataset
        raise ValueError(self._reason)

    @override
    def __repr__(self) -> str:
        """Return the refusal message."""
        return f"RefusedTraining({self._reason!r})"


class DatasetStreamDeclaration[S](Protocol):
    """Declare any random streams consumed while lowering a dataset."""

    def __call__(self, declaration: S, /) -> tuple[str, ...]:
        """Return stream names in consumption order."""
        ...


class DatasetPlugin[S, C: CheckedDataset, D: LoweredDataset](FrozenValue):
    """Pair a dataset declaration with its complete checked lowering route."""

    __slots__ = (
        "_check",
        "_checked_type",
        "_dataset_type",
        "_lower",
        "_pairs",
        "_resources",
        "_sentences",
        "_spec",
        "_streams",
        "_validate",
        "_vocabulary",
    )

    _noun = "dataset plugin"

    def __init__(
        self,
        *,
        spec: type[S],
        checked: type[C],
        dataset: type[D],
        check: DatasetChecker[S, C],
        lower: DatasetLowerer[C, D],
        validate: DatasetValidator[D],
        vocabulary: DatasetVocabularyDeclaration[D],
        resources: DatasetResourcesDeclaration[D],
        sentences: DatasetSentencesDeclaration[D],
        pairs: DatasetPairsDeclaration[D],
        streams: DatasetStreamDeclaration[S],
    ) -> None:
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_checked_type", checked)
        object.__setattr__(self, "_dataset_type", dataset)
        object.__setattr__(self, "_check", check)
        object.__setattr__(self, "_lower", lower)
        object.__setattr__(self, "_validate", validate)
        object.__setattr__(self, "_vocabulary", vocabulary)
        object.__setattr__(self, "_resources", resources)
        object.__setattr__(self, "_sentences", sentences)
        object.__setattr__(self, "_pairs", pairs)
        object.__setattr__(self, "_streams", streams)

    @property
    def spec(self) -> type[S]:
        """Return the declaration variant this plugin implements."""
        return self._spec

    def _declaration(self, declaration: DatasetSpec, /) -> S:
        """Return a declaration of this plugin's exact variant."""
        if type(declaration) is not self._spec:
            msg = (
                f"the dataset plugin handles {self._spec.__name__}, not "
                f"{type(declaration).__name__}"
            )
            raise RegistryError(msg, family="dataset")

        return cast("S", declaration)

    def check(self, declaration: DatasetSpec, /) -> CheckedDataset:
        """Check a declaration of this plugin's exact variant."""
        return self._check(self._declaration(declaration))

    def lower(
        self, checked: CheckedDataset, policy: ParsePolicy[Retain], /
    ) -> LoweredDataset:
        """Lower a checked value of this plugin's exact route."""
        if type(checked) is not self._checked_type:
            msg = (
                f"the dataset plugin lowers {self._checked_type.__name__}, not "
                f"{type(checked).__name__}"
            )
            raise RegistryError(msg, family="dataset")

        return self._lower(cast("C", checked), policy)

    def validate(self, dataset: LoweredDataset, /) -> None:
        """Validate a dataset of this plugin's exact route."""
        if type(dataset) is not self._dataset_type:
            msg = (
                f"the dataset plugin validates {self._dataset_type.__name__}, not "
                f"{type(dataset).__name__}"
            )
            raise RegistryError(msg, family="dataset")

        self._validate(cast("D", dataset))

    def vocabulary(
        self, dataset: LoweredDataset, reserved: ReservedForms, /
    ) -> DatasetVocabulary:
        """Return the vocabulary of a dataset of this plugin's exact route."""
        if type(dataset) is not self._dataset_type:
            msg = (
                f"the dataset plugin reads vocabulary from "
                f"{self._dataset_type.__name__}, not {type(dataset).__name__}"
            )
            raise RegistryError(msg, family="dataset")

        return self._vocabulary(cast("D", dataset), reserved)

    def resources(
        self,
        dataset: LoweredDataset,
        reserved: ReservedForms,
        tensorizer: CodeTensorizer,
        /,
    ) -> FeaturizerResources:
        """Return the featurizer resources of a dataset of this plugin's route."""
        if type(dataset) is not self._dataset_type:
            msg = (
                f"the dataset plugin builds featurizer resources from "
                f"{self._dataset_type.__name__}, not {type(dataset).__name__}"
            )
            raise RegistryError(msg, family="dataset")

        return self._resources(cast("D", dataset), reserved, tensorizer)

    def sentences(self, dataset: LoweredDataset, /) -> SentenceTraining:
        """Return the sentence training a dataset of this plugin's route supplies."""
        return self._sentences(self._dataset(dataset, "sentence training"))

    def pairs(self, dataset: LoweredDataset, /) -> PairTraining:
        """Return the pair training a dataset of this plugin's route supplies."""
        return self._pairs(self._dataset(dataset, "inference-pair training"))

    def _dataset(self, dataset: LoweredDataset, purpose: str, /) -> D:
        """Return a dataset of this plugin's exact lowered type."""
        if type(dataset) is not self._dataset_type:
            msg = (
                f"the dataset plugin builds {purpose} from "
                f"{self._dataset_type.__name__}, not {type(dataset).__name__}"
            )
            raise RegistryError(msg, family="dataset")

        return cast("D", dataset)

    def sampler_streams(self, declaration: DatasetSpec, /) -> tuple[str, ...]:
        """Return the random streams this dataset route consumes."""
        return self._streams(self._declaration(declaration))

    @override
    def __repr__(self) -> str:
        """Return the declaration and target types of the plugin."""
        return (
            f"DatasetPlugin(spec={self._spec.__name__}, "
            f"dataset={self._dataset_type.__name__})"
        )


def dataset_plugin(
    declaration: DatasetSpec, /
) -> DatasetPlugin[DatasetSpec, CheckedDataset, LoweredDataset]:
    """Return the installed training plugin for one dataset declaration."""
    tag = cast("str", declaration.kind)
    member = Registry[object]("dataset").get(tag)

    if not isinstance(member, DatasetPlugin):
        msg = f"the dataset entry {tag!r} does not name a training dataset plugin"
        raise RegistryError(msg, family="dataset")

    plugin = cast("DatasetPlugin[DatasetSpec, CheckedDataset, LoweredDataset]", member)

    if plugin.spec is not type(declaration):
        msg = (
            f"the dataset entry {tag!r} handles {plugin.spec.__name__}, not "
            f"{type(declaration).__name__}"
        )
        raise RegistryError(msg, family="dataset")

    return plugin


def artifact_dataset_plugin(
    declaration: DatasetSpec, /
) -> ArtifactDatasetPlugin[DatasetSpec, object]:
    """Return the installed artifact plugin for one dataset declaration."""
    tag = cast("str", declaration.kind)
    member = Registry[object]("dataset").get(tag)

    if not isinstance(member, ArtifactDatasetPlugin):
        msg = f"the dataset entry {tag!r} does not name an artifact dataset plugin"
        raise RegistryError(msg, family="dataset")

    plugin = cast("ArtifactDatasetPlugin[DatasetSpec, object]", member)

    if plugin.spec is not type(declaration):
        msg = (
            f"the dataset entry {tag!r} handles {plugin.spec.__name__}, not "
            f"{type(declaration).__name__}"
        )
        raise RegistryError(msg, family="dataset")

    return plugin
