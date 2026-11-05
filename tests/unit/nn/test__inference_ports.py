"""Tests for grounded-inference batch capabilities."""

from typing import NamedTuple

from lofi._types import ReadoutId
from lofi.lf import ArgumentRenaming, InferenceLabel
from lofi.nn._inference_ports import InferencePairs


class Paired(NamedTuple):
    """A batch implementing the inference-pairs protocol."""

    premises: tuple[ReadoutId, ...]
    hypotheses: tuple[ReadoutId, ...]
    alignments: tuple[tuple[ArgumentRenaming, ...], ...]
    labels: tuple[InferenceLabel, ...]


class Unpaired(NamedTuple):
    """A batch that names premises and labels but no hypothesis."""

    premises: tuple[ReadoutId, ...]
    alignments: tuple[tuple[ArgumentRenaming, ...], ...]
    labels: tuple[InferenceLabel, ...]


def test_a_batch_naming_both_sides_of_each_pair_satisfies_the_protocol() -> None:
    paired = Paired(
        (ReadoutId("premise"),),
        (ReadoutId("hypothesis"),),
        ((ArgumentRenaming.identity(),),),
        (InferenceLabel.ENTAILS,),
    )
    unpaired = Unpaired(
        (ReadoutId("premise"),),
        ((ArgumentRenaming.identity(),),),
        (InferenceLabel.ENTAILS,),
    )

    assert isinstance(paired, InferencePairs)
    assert not isinstance(unpaired, InferencePairs)
    assert not isinstance((paired.premises, paired.labels), InferencePairs)
