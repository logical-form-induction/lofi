"""Evidence-typed scoring of premise and hypothesis forms."""

from typing import TYPE_CHECKING, Protocol, final, runtime_checkable

import torch

from lofi._checks import ensure_finite, ensure_non_negative
from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.lf import InferenceLabel, LogicFragment, Verdict
    from lofi.lf.provers import Questions
    from lofi.nn.grounding._values import EvidenceLedger, ProofMetadata

__all__ = [
    "AuxiliaryScorer",
    "Evidence",
    "PairScorer",
    "PositiveQuestionScorer",
    "ProofSearch",
    "Satisfaction",
    "Scores",
]


@final
class ProofSearch(FrozenValue):
    """A guided or neural proof search and its checked verdict."""

    _noun = "proof-search evidence"
    __slots__ = ("_found", "_score", "_verdict", "_verified")

    def __init__(
        self,
        verdict: Verdict,
        /,
        *,
        found: bool,
        verified: bool,
        score: float,
    ) -> None:
        held = ensure_finite("a proof-search score", score)

        if not 0.0 <= held <= 1.0:
            msg = f"a proof-search score lies in the unit interval, not {score!r}"
            raise ValueError(msg)

        if not isinstance(found, bool) or not isinstance(verified, bool):
            msg = "proof-search found and verified flags are bools"
            raise TypeError(msg)

        object.__setattr__(self, "_verdict", verdict)
        object.__setattr__(self, "_found", found)
        object.__setattr__(self, "_verified", verified)
        object.__setattr__(self, "_score", held)

    @property
    def verdict(self) -> Verdict:
        return self._verdict

    @property
    def found(self) -> bool:
        return self._found

    @property
    def verified(self) -> bool:
        return self._verified

    @property
    def score(self) -> float:
        return self._score


@final
class Satisfaction(FrozenValue):
    """Conditional satisfaction evidence from a declared structure source."""

    _noun = "satisfaction evidence"
    __slots__ = ("_both", "_premise", "_source", "_structures")

    def __init__(
        self,
        source: str,
        /,
        *,
        structures: int,
        premise_holding: int,
        both_holding: int,
    ) -> None:
        held_structures = ensure_non_negative("sampled structures", structures)
        premise = ensure_non_negative("premise-holding structures", premise_holding)
        both = ensure_non_negative("jointly holding structures", both_holding)

        if premise > held_structures or both > premise:
            msg = (
                "jointly holding structures are a subset of premise-holding structures"
            )
            raise ValueError(msg)

        if not source:
            msg = "a structure source has a name"
            raise ValueError(msg)

        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_structures", held_structures)
        object.__setattr__(self, "_premise", premise)
        object.__setattr__(self, "_both", both)

    @property
    def source(self) -> str:
        return self._source

    @property
    def structures(self) -> int:
        return self._structures

    @property
    def premise_holding(self) -> int:
        return self._premise

    @property
    def both_holding(self) -> int:
        return self._both


if TYPE_CHECKING:
    type Evidence = Verdict | ProofSearch | Satisfaction
else:
    type Evidence = object


@final
class Scores[E](FrozenValue):
    """One score, decision flag, and evidence value per distinct question."""

    _noun = "pair scores"
    __slots__ = ("_decided", "_evidence", "_ledger", "_values")

    def __init__(
        self,
        values: Tensor,
        decided: Tensor,
        evidence: tuple[E, ...],
        ledger: EvidenceLedger,
    ) -> None:
        if not values.is_floating_point():
            msg = f"pair scores are floating point, not {values.dtype}"
            raise TypeError(msg)

        if decided.dtype is not torch.bool:
            msg = f"pair decisions are boolean, not {decided.dtype}"
            raise TypeError(msg)

        shape = (len(evidence),)

        if values.shape != shape or decided.shape != shape:
            msg = (
                "pair scores align one value and decision with each evidence item; "
                f"expected {shape}, got {tuple(values.shape)} and "
                f"{tuple(decided.shape)}"
            )
            raise ValueError(msg)

        held = values.detach()[decided]

        if held.numel() and (
            not bool(held.isfinite().all()) or bool(((held < 0.0) | (held > 1.0)).any())
        ):
            msg = "each decided pair score lies in the unit interval"
            raise ValueError(msg)

        object.__setattr__(self, "_values", values)
        object.__setattr__(self, "_decided", decided)
        object.__setattr__(self, "_evidence", tuple(evidence))
        object.__setattr__(self, "_ledger", ledger)

    @property
    def values(self) -> Tensor:
        return self._values

    @property
    def decided(self) -> Tensor:
        return self._decided

    @property
    def evidence(self) -> tuple[E, ...]:
        return self._evidence

    @property
    def ledger(self) -> EvidenceLedger:
        return self._ledger


@runtime_checkable
class PairScorer[E](Protocol):
    """Score distinct entailment questions with one evidence kind."""

    @property
    def scorer_kind(self) -> str: ...

    @property
    def source_identity(self) -> str: ...

    @property
    def fragment(self) -> LogicFragment: ...

    @property
    def differentiable(self) -> bool: ...

    @property
    def certified(self) -> bool: ...

    @property
    def proof_metadata(self) -> ProofMetadata | None: ...

    def score(self, questions: Questions, /) -> Scores[E]: ...

    def open_step(self) -> None: ...


@runtime_checkable
class PositiveQuestionScorer(Protocol):
    """Declare a scorer whose label model needs only the positive question."""

    @property
    def positive_questions_only(self) -> bool: ...

    def validate_labels(self, labels: tuple[InferenceLabel, ...], /) -> None: ...


@runtime_checkable
class AuxiliaryScorer(Protocol):
    """Publish a scorer-owned loss without widening every scorer contract."""

    def auxiliary(self) -> Tensor | None:
        """Return the current auxiliary loss term, when one is declared."""
        ...
