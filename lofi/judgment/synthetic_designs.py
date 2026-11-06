"""Synthetic rating designs whose participant effects are known.

A design crosses every participant with every item. Each item carries one
score, and each participant an intercept and a set of cutpoint offsets. The
responses are drawn through the compiled program of the generating
specification, at a parameter vector whose population values are declared
here and whose participant offsets and item scores are drawn from the
design's seed. The population cutpoints divide a declared latent range into
as many equal intervals as the design has response levels, one cutpoint at
each interior boundary. The same seed therefore reproduces the same design, and the
vector the responses were drawn at is the known truth a fit recovers.
"""

import json
import math
from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, final, override

import torch

from lofi.judgment._design import DesignCodec, JudgmentRow, RowColumns
from lofi.judgment._spec import (
    ConditionalLikelihood,
    Cumulative,
    DesignProvenance,
    FixedDiscrimination,
    LinkingSpec,
    LinkTerm,
    PriorBlock,
    RandomIntercept,
    ReadDesign,
    ResponseSpec,
    ThresholdPriors,
)
from lofi.judgment.compiler import PARTICIPANT, ResponseCompiler
from lofi.judgment.linear import LINEAR
from lofi.judgment.simulate import sample_responses, simulated_design

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import Tensor

    from lofi.judgment._design import JudgmentDesign
    from lofi.judgment.compiler import BoundProgram

__all__ = [
    "KNOWN_RANDOM_EFFECTS",
    "KNOWN_RANDOM_EFFECTS_TAG",
    "PRIOR_PREDICTIVE_REFERENCE",
    "PRIOR_PREDICTIVE_REFERENCE_TAG",
    "SCORE",
    "KnownEffectsDesign",
    "generating_spec",
]

SCORE: Final = "score"
"""The readout column holding each item's score."""


_POPULATION: Final[Mapping[str, float]] = MappingProxyType(
    {
        "score_slope": 1.0,
        "participant_intercept_scale": 0.8,
        "participant_cutpoint_spread": 0.3,
    }
)
"""The value of every population site but the cutpoints, each element alike."""

_PARTICIPANT_EFFECTS: Final = (
    "participant_intercept_offsets",
    "participant_cutpoint_offsets",
)
"""The standard normal participant sites, drawn from the design's seed."""


def _population(levels: int, latent_range: tuple[float, float], /) -> dict[str, float]:
    """Return every population site's value, the cutpoints dividing the range.

    The program places cutpoint ``k`` at the base plus the running sum of the
    softplus of the first ``k`` increments, so the base is the range's lower
    end and each increment, held before the softplus, is one interval width.
    """
    lower, upper = latent_range
    width = (upper - lower) / levels
    return {
        **_POPULATION,
        "cutpoint_base": lower,
        "cutpoint_increments": math.log(math.expm1(width)),
    }


def generating_spec(levels: int, /) -> ResponseSpec:
    """Return the response model a synthetic design's responses are drawn from.

    The latent response is the item's score plus the participant's
    intercept, observed through participant-specific cutpoints.

    Parameters
    ----------
    levels : int
        Number of ordered response levels.
    """
    return ResponseSpec(
        family=Cumulative(),
        levels=levels,
        objective=ConditionalLikelihood(),
        predictor=(
            LinkTerm(readout=SCORE, linking=LinkingSpec(family=LINEAR)),
            RandomIntercept(group=PARTICIPANT),
        ),
        identification=FixedDiscrimination(value=1.0),
        priors=PriorBlock(thresholds=ThresholdPriors()),
    )


@final
class KnownEffectsDesign:
    """A crossed rating design drawn at known participant effects.

    Calling the design returns it. Unless the item scores are declared, they
    are drawn first from the seed; then each participant site is drawn in
    program order, then the responses.

    Parameters
    ----------
    tag : str
        The tag the design is registered under.
    participants : int
        Number of participants, each of whom rates every item.
    items : int
        Number of items.
    levels : int
        Number of ordered response levels.
    seed : int
        Seed of every draw the design is made from.
    latent_range : tuple[float, float]
        Lower and upper end of the latent interval the population cutpoints
        divide into ``levels`` equal intervals.
    scores : Sequence[float] | None
        Score of each item, ``item-0`` first, or None to draw them.

    Raises
    ------
    ValueError
        If declared scores do not number the items, or the latent range has
        no positive finite width.
    """

    __slots__ = (
        "_items",
        "_latent_range",
        "_levels",
        "_participants",
        "_scores",
        "_seed",
        "_tag",
    )

    def __init__(
        self,
        tag: str,
        *,
        participants: int,
        items: int,
        levels: int,
        seed: int,
        latent_range: tuple[float, float],
        scores: Sequence[float] | None = None,
    ) -> None:
        if scores is not None and len(scores) != items:
            msg = f"a design declares {len(scores)} scores for {items} items"
            raise ValueError(msg)
        lower, upper = latent_range
        if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
            msg = f"a latent range has no positive finite width: {latent_range!r}"
            raise ValueError(msg)
        self._latent_range = (float(lower), float(upper))
        self._tag = tag
        self._participants = participants
        self._items = items
        self._levels = levels
        self._seed = seed
        self._scores = None if scores is None else tuple(map(float, scores))

    @property
    def tag(self) -> str:
        """Return the tag the design is registered under."""
        return self._tag

    @property
    def participants(self) -> int:
        """Return the number of participants."""
        return self._participants

    @property
    def items(self) -> int:
        """Return the number of items."""
        return self._items

    @property
    def levels(self) -> int:
        """Return the number of ordered response levels."""
        return self._levels

    @property
    def seed(self) -> int:
        """Return the seed every draw is made from."""
        return self._seed

    @property
    def latent_range(self) -> tuple[float, float]:
        """Return the latent interval the population cutpoints divide."""
        return self._latent_range

    @property
    def population_digest(self) -> str:
        """Return the SHA-256 digest of the population values, as sorted JSON."""
        population = _population(self._levels, self._latent_range)
        return sha256(
            json.dumps(population, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    @property
    def spec(self) -> ResponseSpec:
        """Return the response model the responses are drawn from."""
        return generating_spec(self._levels)

    def _structure(self, generator: torch.Generator) -> JudgmentDesign:
        """Return the crossed rows, each at the lowest level, with their scores."""
        scores = (
            torch.randn(self._items, generator=generator, dtype=torch.float64)
            if self._scores is None
            else torch.tensor(self._scores, dtype=torch.float64)
        )
        codec = DesignCodec(self._levels, factors=(), readouts=(SCORE,))

        return codec.encode(
            JudgmentRow(
                f"participant-{participant}",
                f"item-{item}",
                0,
                RowColumns({}, {SCORE: float(scores[item])}),
            )
            for participant in range(self._participants)
            for item in range(self._items)
        )

    def _bound(self, structure: JudgmentDesign) -> BoundProgram:
        """Return the generating program bound to a design."""
        return ResponseCompiler.installed().emit(self.spec, structure)

    def _drawn(self) -> tuple[JudgmentDesign, BoundProgram, Tensor]:
        """Return the design, its generating program and the generating vector."""
        generator = torch.Generator().manual_seed(self._seed)
        structure = self._structure(generator)
        program = self._bound(structure)
        values: list[Tensor] = []
        population = _population(self._levels, self._latent_range)

        for slot in program.layout:
            if slot.site in _PARTICIPANT_EFFECTS:
                values.append(
                    torch.randn(slot.size, generator=generator, dtype=torch.float64)
                )
            else:
                values.append(
                    torch.full((slot.size,), population[slot.site], dtype=torch.float64)
                )

        generating = torch.cat(values)
        responses = sample_responses(program, generating, generator=generator)
        design = simulated_design(structure, responses)

        return design, self._bound(design), generating

    def program(self) -> BoundProgram:
        """Return the generating program bound to the drawn design."""
        _, program, _ = self._drawn()

        return program

    def generating(self) -> Tensor:
        """Return the parameter vector the responses were drawn at.

        The vector is laid out as the generating program lays out its sites.
        """
        _, _, generating = self._drawn()

        return generating

    def __call__(self) -> JudgmentDesign:
        """Return the design drawn from the seed."""
        design, _, _ = self._drawn()

        return design

    def read(self) -> ReadDesign:
        """Return the design drawn from the seed with what it was drawn at.

        The provenance records the tag, the seed, the participant, item and
        level counts, and the digest of the population values, the cutpoints
        among them.
        """
        provenance = DesignProvenance(
            tag=self._tag,
            checkpoint=None,
            checkpoint_dtype=None,
            unigram_digest=None,
            corpus_digests={},
            seed=self._seed,
            participants=self._participants,
            items=self._items,
            levels=self._levels,
            population_digest=self.population_digest,
            producer_run=None,
            producer_point=None,
        )

        return ReadDesign(self(), provenance)

    @override
    def __repr__(self) -> str:
        """Return the declared counts and seed."""
        return (
            f"KnownEffectsDesign({self._tag!r}, participants={self._participants}, "
            f"items={self._items}, levels={self._levels}, seed={self._seed}, "
            f"latent_range={self._latent_range!r})"
        )


KNOWN_RANDOM_EFFECTS_TAG: Final = "known-random-effects"
"""The tag the design for comparing likelihood objectives registers under."""

KNOWN_RANDOM_EFFECTS: Final = KnownEffectsDesign(
    KNOWN_RANDOM_EFFECTS_TAG,
    participants=40,
    items=30,
    levels=7,
    seed=20_261_012,
    latent_range=(-2.5, 4.5),
)
"""The design the conditional and marginal objectives are compared on."""

PRIOR_PREDICTIVE_REFERENCE_TAG: Final = "prior-predictive-reference"
"""The tag the design for simulation-based calibration registers under."""

PRIOR_PREDICTIVE_REFERENCE: Final = KnownEffectsDesign(
    PRIOR_PREDICTIVE_REFERENCE_TAG,
    participants=20,
    items=25,
    levels=7,
    seed=20_261_013,
    latent_range=(-2.5, 4.5),
)
"""The design whose rows simulation-based calibration replicates are drawn over."""
