"""Boundary declaration for a derivation-preserving CCGbank reskin."""

from typing import TYPE_CHECKING, Literal, Self, override

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.data.synthetic.spec.corpus import CorruptionSpec, LabelNoiseSpec
from lofi.spec import ensure_data_seed, resolved_seed
from lofi.spec.components.data import DatasetSpec, TrainingBatchKind
from lofi.spec.datasets.ccgbank import DEFAULT_CCGBANK_VALIDATION_SPLIT
from lofi.spec.grammar import GrammarDecl
from lofi.spec.language import LANGUAGE_FIELD, LanguageDecl

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["CcgbankReskinDatasetSpec", "FrequencyReskinSpec"]


class FrequencyReskinSpec(dx.Model, extra="forbid"):
    """Map frequent feature-erased leaf categories to synthetic forms."""

    prefix: str = "type"
    minimum_frequency: int = 1

    __axioms__ = (
        dx.axiom(
            "minimum_frequency > 0",
            message="minimum_frequency must be positive",
        ),
    )

    @dx.validates("prefix")
    @classmethod
    def _prefix_is_named(cls, prefix: str) -> str:
        return ensure_name("a reskin form prefix", prefix)

    def settings(self) -> tuple[str, int]:
        """Return the validated interior mapping settings."""
        return self.prefix, self.minimum_frequency


class CcgbankReskinDatasetSpec(DatasetSpec):
    """Preserve CCGbank derivations while replacing frequent lexical forms."""

    kind: Literal["ccgbank-reskin"] = "ccgbank-reskin"
    source: str
    splits: tuple[Literal["train", "validation", "development", "test"], ...] = (
        "train",
    )
    training_split: Literal["train", "validation", "development", "test"] = "train"
    validation_split: str = DEFAULT_CCGBANK_VALIDATION_SPLIT
    grammar: GrammarDecl
    language: LanguageDecl = LANGUAGE_FIELD
    mapping: FrequencyReskinSpec = dx.field(default_factory=FrequencyReskinSpec)
    seed: int | str
    corruption: CorruptionSpec | None = None
    noise: LabelNoiseSpec | None = None

    __axioms__ = (
        dx.axiom("len(source) > 0", message="a CCGbank reskin needs a source"),
        dx.axiom("len(splits) > 0", message="a CCGbank reskin needs a split"),
    )

    @dx.model_validator()
    def _splits_are_distinct(self) -> Self:
        if len(set(self.splits)) != len(self.splits):
            msg = "a CCGbank reskin names each split once"
            raise ValueError(msg)
        if self.training_split not in self.splits:
            msg = (
                f"a CCGbank reskin trains on {self.training_split!r}, which it "
                f"does not read from {self.splits!r}"
            )
            raise ValueError(msg)
        return self

    @dx.validates("validation_split")
    @classmethod
    def _validation_split_is_named(cls, value: str) -> str:
        return ensure_name("a CCGbank reskin's validation split", value)

    @dx.validates("seed")
    @classmethod
    def _seed_is_a_data_seed(cls, value: int | str) -> int | str:
        return ensure_data_seed("a CCGbank reskin's seed", value)

    def held_out_split(self) -> str:
        """Return the declared validation split name."""
        return self.validation_split

    def retained_splits(self) -> frozenset[str]:
        """Return selected source partitions withheld from training."""
        return frozenset(split for split in self.splits if split != self.training_split)

    @override
    def with_derived_seeds(self, derive: Callable[[str], int], path: str, /) -> Self:
        """Return the declaration with its reskin stream seed resolved."""
        return self.with_(seed=resolved_seed(self.seed, derive, f"{path}.seed"))

    @override
    def training_batches(self) -> frozenset[TrainingBatchKind]:
        """Return the sentence batches this dataset yields."""
        return frozenset({TrainingBatchKind.SENTENCES})
