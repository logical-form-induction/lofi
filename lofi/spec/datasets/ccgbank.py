"""Boundary declarations for a CCGbank dataset."""

from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, override

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.spec._interpolation import lower_package_resource
from lofi.spec._loader import SpecLoader
from lofi.spec.components.data import DatasetSpec, TrainingBatchKind
from lofi.spec.grammar import GrammarDecl
from lofi.spec.language import LANGUAGE_FIELD, LanguageDecl

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self

__all__ = [
    "AssignUnknownWordSpec",
    "CcgbankDatasetSpec",
    "SkipUnknownWordSpec",
    "UnknownWordSpec",
    "ccgbank_grammar",
]

_GRAMMAR_RESOURCE: Final = (
    "lofi.resource:data/datasets/train/dataset/grammar/ccgbank.yaml"
)
DEFAULT_CCGBANK_VALIDATION_SPLIT: Final = "validation"


class UnknownWordSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare how a held-out word unseen in sections 02--21 is handled."""

    def policy(self) -> tuple[Literal["skip", "assign"], str | None]:
        """Return the interior reader policy and its optional category."""
        msg = f"{type(self).__name__} is the unknown-word union, not a policy"
        raise TypeError(msg)


class SkipUnknownWordSpec(UnknownWordSpec):
    """Skip and report a sentence containing a training-unseen word."""

    kind: Literal["skip"] = "skip"

    @override
    def policy(self) -> tuple[Literal["skip"], None]:
        """Return the skip policy."""
        return "skip", None


class AssignUnknownWordSpec(UnknownWordSpec):
    """Assign one declared category to every training-unseen word."""

    kind: Literal["assign"] = "assign"
    category: str

    __axioms__ = (
        dx.axiom("len(category) > 0", message="an unknown category needs a name"),
    )

    @override
    def policy(self) -> tuple[Literal["assign"], str]:
        """Return the assignment policy and its category."""
        return "assign", self.category


class CcgbankDatasetSpec(DatasetSpec):
    """Declare a CCGbank source read under one grammar.

    The conventional sections are training 02--21, validation 00,
    development 22 and test 23. A single AUTO file is treated as training
    data, which permits small declared fixtures to exercise the same route.
    A parser trains on the gold derivations of the declared training split.
    """

    kind: Literal["ccgbank"] = "ccgbank"
    source: str
    splits: tuple[Literal["train", "validation", "development", "test"], ...] = (
        "train",
        "validation",
        "development",
        "test",
    )
    training_split: Literal["train", "validation", "development", "test"] = "train"
    validation_split: str = DEFAULT_CCGBANK_VALIDATION_SPLIT
    grammar: GrammarDecl
    language: LanguageDecl = LANGUAGE_FIELD
    unknown_words: UnknownWordSpec = dx.field(default_factory=SkipUnknownWordSpec)

    __axioms__ = (
        dx.axiom("len(source) > 0", message="a CCGbank dataset needs a source"),
        dx.axiom("len(splits) > 0", message="a CCGbank dataset needs a split"),
    )

    @dx.model_validator()
    def _splits_are_distinct(self) -> Self:
        """Return a declaration that names each selected split once."""
        if len(set(self.splits)) != len(self.splits):
            msg = "a CCGbank dataset names each split once"
            raise ValueError(msg)
        if self.training_split not in self.splits:
            msg = (
                f"a CCGbank dataset trains on {self.training_split!r}, which it "
                f"does not read from {self.splits!r}"
            )
            raise ValueError(msg)
        return self

    @dx.validates("validation_split")
    @classmethod
    def _validation_split_is_named(cls, value: str) -> str:
        """Return the name under which held-out rows are reported."""
        return ensure_name("a CCGbank corpus's validation split", value)

    def held_out_split(self) -> str:
        """Return the declared validation split name."""
        return self.validation_split

    def retained_splits(self) -> frozenset[str]:
        """Return selected corpus partitions withheld from training."""
        return frozenset(split for split in self.splits if split != self.training_split)

    @override
    def with_derived_seeds(self, derive: Callable[[str], int], path: str, /) -> Self:
        """Return the declaration with a derived lexicon seed resolved."""
        grammar = self.grammar
        return self.with_(
            grammar=grammar.with_(
                lexicon=grammar.lexicon.with_derived_seed(
                    derive, f"{path}.grammar.lexicon"
                )
            )
        )

    @override
    def training_batches(self) -> frozenset[TrainingBatchKind]:
        """Return the sentence batches this dataset yields."""
        return frozenset({TrainingBatchKind.SENTENCES})


def ccgbank_grammar() -> GrammarDecl:
    """Load the packaged CCGbank grammar declaration."""
    path = Path(lower_package_resource(_GRAMMAR_RESOURCE))
    return SpecLoader().compose_traced(path, schema=GrammarDecl).spec
