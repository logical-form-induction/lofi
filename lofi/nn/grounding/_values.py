"""Values published by inference groundings."""

from enum import Enum, StrEnum
from math import log
from typing import TYPE_CHECKING, Never, final, override

import torch

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.control_variate import ControlVariate
from lofi.lf import ABSTENTION_CAUSES, Abstain, InferenceCell, InferenceLabel
from lofi.lf.provers import ProofBudget
from lofi.reward import InferenceReward

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import Tensor

    from lofi.lf import AbstentionCause
    from lofi.lf.provers import ServiceReport
    from lofi.nn._values import LogDistribution

__all__ = [
    "Best",
    "CellMass",
    "Composition",
    "EmbeddingVectors",
    "EvidenceLedger",
    "GradientPath",
    "GroundedBatch",
    "GroundingMetadata",
    "GroundingTelemetry",
    "Marginal",
    "Phase",
    "ProofMetadata",
    "Tempered",
    "checked_composition",
    "composition_from_settings",
]


class EmbeddingVectors(StrEnum):
    """Name the vector source consumed by an embedding grounding."""

    EXPECTED_FORM = "expected-form"
    SENTENCE = "sentence"


class GradientPath(Enum):
    """How a grounded loss reaches the decoder and scorer parameters."""

    PATHWISE = "pathwise"
    SCORE_FUNCTION = "score-function"
    MIXED = "mixed"


@final
class ProofMetadata(FrozenValue):
    """Stable proof-service identity and the scorer's use-site budget."""

    _noun = "proof metadata"
    __slots__ = ("_budget", "_prover", "_version")

    def __init__(
        self,
        prover: str,
        budget: ProofBudget,
        /,
        *,
        version: str | None,
    ) -> None:
        if not isinstance(budget, ProofBudget):
            msg = f"proof metadata budget is ProofBudget, not {budget!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_prover", ensure_name("prover", prover))
        object.__setattr__(self, "_budget", budget)
        object.__setattr__(
            self,
            "_version",
            None if version is None else ensure_name("prover version", version),
        )

    @property
    def prover(self) -> str:
        return self._prover

    @property
    def version(self) -> str | None:
        return self._version

    @property
    def budget(self) -> ProofBudget:
        return self._budget


@final
class GroundingMetadata(FrozenValue):
    """Stable grounding, scorer, source, and proof-use checkpoint identity."""

    _noun = "grounding metadata"
    __slots__ = (
        "_budget",
        "_prover",
        "_prover_version",
        "_scorer",
        "_source",
        "_variant",
    )

    def __init__(
        self,
        variant: str,
        /,
        *,
        scorer: str,
        source: str,
        prover: str | None,
        prover_version: str | None,
        budget: ProofBudget | None,
    ) -> None:
        object.__setattr__(self, "_variant", ensure_name("grounding variant", variant))
        object.__setattr__(self, "_scorer", ensure_name("grounding scorer", scorer))
        object.__setattr__(self, "_source", ensure_name("grounding source", source))
        object.__setattr__(
            self,
            "_prover",
            None if prover is None else ensure_name("grounding prover", prover),
        )
        object.__setattr__(
            self,
            "_prover_version",
            (
                None
                if prover_version is None
                else ensure_name("grounding prover version", prover_version)
            ),
        )

        if budget is not None and not isinstance(budget, ProofBudget):
            msg = f"a grounding proof budget is ProofBudget, not {budget!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_budget", budget)

    @property
    def variant(self) -> str:
        return self._variant

    @property
    def scorer(self) -> str:
        return self._scorer

    @property
    def source(self) -> str:
        return self._source

    @property
    def prover(self) -> str | None:
        return self._prover

    @property
    def prover_version(self) -> str | None:
        return self._prover_version

    @property
    def budget(self) -> ProofBudget | None:
        return self._budget


@final
class GroundingTelemetry(FrozenValue):
    """Tensor-free telemetry measured from one grounded batch."""

    _noun = "grounding telemetry"
    __slots__ = (
        "_abstentions",
        "_calls",
        "_cpu_seconds",
        "_degenerate_mass",
        "_gradient_path",
        "_label_agreement",
        "_malformed_mass",
        "_memo_hit_rate",
        "_memo_hits",
        "_out_of_fragment_share",
        "_process_starts",
        "_questions",
        "_questions_per_second",
        "_reward_mean",
        "_variance",
        "_verified_share",
        "_wall_seconds",
    )

    def __init__(
        self,
        gradient_path: GradientPath,
        abstentions: Mapping[str, int],
        /,
        *,
        calls: int,
        questions: int,
        questions_per_second: float,
        memo_hits: int,
        memo_hit_rate: float,
        cpu_seconds: float,
        wall_seconds: float,
        process_starts: int | None,
        label_agreement: float | None,
        degenerate_mass: float,
        malformed_mass: float,
        reward_mean: float | None,
        variance: float | None,
        verified_share: float | None,
        out_of_fragment_share: float,
    ) -> None:
        if not isinstance(gradient_path, GradientPath):
            msg = f"a grounding telemetry path is GradientPath, not {gradient_path!r}"
            raise TypeError(msg)

        if label_agreement is not None:
            agreement = ensure_finite("grounded label agreement", label_agreement)

            if not 0.0 <= agreement <= 1.0:
                msg = (
                    "grounded label agreement is a probability, "
                    f"not {label_agreement!r}"
                )
                raise ValueError(msg)
        else:
            agreement = None

        object.__setattr__(self, "_gradient_path", gradient_path)
        object.__setattr__(self, "_abstentions", tuple(sorted(abstentions.items())))
        object.__setattr__(self, "_calls", calls)
        object.__setattr__(self, "_questions", questions)
        object.__setattr__(self, "_questions_per_second", questions_per_second)
        object.__setattr__(self, "_memo_hits", memo_hits)
        object.__setattr__(self, "_memo_hit_rate", memo_hit_rate)
        object.__setattr__(self, "_cpu_seconds", cpu_seconds)
        object.__setattr__(self, "_wall_seconds", wall_seconds)
        object.__setattr__(self, "_process_starts", process_starts)
        object.__setattr__(self, "_label_agreement", agreement)
        object.__setattr__(self, "_degenerate_mass", degenerate_mass)
        object.__setattr__(self, "_malformed_mass", malformed_mass)
        object.__setattr__(self, "_reward_mean", reward_mean)
        object.__setattr__(self, "_variance", variance)
        object.__setattr__(self, "_verified_share", verified_share)
        outside = ensure_finite(
            "an out-of-fragment share",
            out_of_fragment_share,
        )

        if not 0.0 <= outside <= 1.0:
            msg = (
                "an out-of-fragment share is a probability, "
                f"not {out_of_fragment_share!r}"
            )
            raise ValueError(msg)

        object.__setattr__(self, "_out_of_fragment_share", outside)

    @property
    def gradient_path(self) -> GradientPath:
        return self._gradient_path

    @property
    def abstentions(self) -> Mapping[str, int]:
        return dict(self._abstentions)

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def questions(self) -> int:
        return self._questions

    @property
    def questions_per_second(self) -> float:
        return self._questions_per_second

    @property
    def memo_hits(self) -> int:
        return self._memo_hits

    @property
    def memo_hit_rate(self) -> float:
        return self._memo_hit_rate

    @property
    def cpu_seconds(self) -> float:
        return self._cpu_seconds

    @property
    def wall_seconds(self) -> float:
        return self._wall_seconds

    @property
    def process_starts(self) -> int | None:
        return self._process_starts

    @property
    def label_agreement(self) -> float | None:
        """Return decided argmax agreement with the symbolic data labels."""
        return self._label_agreement

    @property
    def degenerate_mass(self) -> float:
        return self._degenerate_mass

    @property
    def malformed_mass(self) -> float:
        return self._malformed_mass

    @property
    def reward_mean(self) -> float | None:
        return self._reward_mean

    @property
    def variance(self) -> float | None:
        return self._variance

    @property
    def verified_share(self) -> float | None:
        return self._verified_share

    @property
    def out_of_fragment_share(self) -> float:
        """Return the question share outside the scorer's served fragment."""
        return self._out_of_fragment_share


class Phase(Enum):
    """Whether a grounding is serving optimization or complete evaluation."""

    TRAIN = "train"
    EVALUATE = "evaluate"


class _CompositionValue(FrozenValue):
    """Shared frozen behavior for composition rules."""

    _noun = "grounding composition"
    __slots__ = ()


def _decided(log_probabilities: Tensor) -> Tensor:
    return log_probabilities[log_probabilities > -torch.inf]


def _log_mean_exp(log_probabilities: Tensor) -> Tensor:
    finite = _decided(log_probabilities)

    if finite.numel() == 0:
        return log_probabilities.new_full((), -torch.inf)

    return finite.logsumexp(dim=0) - log(log_probabilities.shape[0])


@final
class Marginal(_CompositionValue):
    """Compose form pairs by their normalized probability mass."""

    __slots__ = ()

    def log_shares(self, log_joint: Tensor, /) -> Tensor:
        return log_joint - log_joint.logsumexp(dim=0)

    def over_alignments(self, log_probabilities: Tensor, /) -> Tensor:
        return _log_mean_exp(log_probabilities)

    def over_residuals(self, residuals: Tensor, /) -> Tensor:
        return residuals.mean()

    @override
    def __eq__(self, other: object) -> bool:
        return type(other) is Marginal

    @override
    def __hash__(self) -> int:
        return hash(Marginal)

    @override
    def __repr__(self) -> str:
        return "Marginal()"


@final
class Best(_CompositionValue):
    """Compose form pairs through the heaviest retained alternative."""

    __slots__ = ()

    def log_shares(self, log_joint: Tensor, /) -> Tensor:
        chosen = torch.zeros_like(log_joint, dtype=torch.bool)
        chosen[log_joint.argmax()] = True
        return torch.where(
            chosen,
            log_joint - log_joint.detach(),
            log_joint.new_full((), -torch.inf),
        )

    def over_alignments(self, log_probabilities: Tensor, /) -> Tensor:
        return log_probabilities.max()

    def over_residuals(self, residuals: Tensor, /) -> Tensor:
        return residuals.max()

    @override
    def __eq__(self, other: object) -> bool:
        return type(other) is Best

    @override
    def __hash__(self) -> int:
        return hash(Best)

    @override
    def __repr__(self) -> str:
        return "Best()"


@final
class Tempered(_CompositionValue):
    """Compose form pairs by a power mean at a positive temperature."""

    __slots__ = ("_temperature",)

    def __init__(self, temperature: float) -> None:
        held = ensure_finite("a composition temperature", temperature)

        if held <= 0.0:
            msg = f"a composition temperature is positive, not {temperature!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_temperature", held)

    @property
    def temperature(self) -> float:
        return self._temperature

    def log_shares(self, log_joint: Tensor, /) -> Tensor:
        return (log_joint / self._temperature).log_softmax(dim=0)

    def over_alignments(self, log_probabilities: Tensor, /) -> Tensor:
        return self._temperature * _log_mean_exp(log_probabilities / self._temperature)

    def over_residuals(self, residuals: Tensor, /) -> Tensor:
        return residuals.max()

    @override
    def __eq__(self, other: object) -> bool:
        return type(other) is Tempered and self._temperature == other._temperature

    @override
    def __hash__(self) -> int:
        return hash((Tempered, self._temperature))

    @override
    def __repr__(self) -> str:
        return f"Tempered({self._temperature!r})"


type Composition = Marginal | Best | Tempered


def composition_from_settings(rule: str, temperature: float | None, /) -> Composition:
    """Lower one validated composition declaration at the neural boundary."""
    if rule == "marginal" and temperature is None:
        return Marginal()

    if rule == "best" and temperature is None:
        return Best()

    if rule == "tempered" and temperature is not None:
        return Tempered(temperature)

    msg = f"invalid composition settings: rule={rule!r}, temperature={temperature!r}"
    raise ValueError(msg)


def checked_composition(value: Composition) -> Composition:
    """Return a declared composition rule."""
    if not isinstance(value, Marginal | Best | Tempered):
        msg = f"a grounding composition is not a declared rule: {value!r}"
        raise TypeError(msg)

    return value


@final
class CellMass(FrozenValue):
    """Probability mass on each inference cell for every data pair."""

    _noun = "inference-cell mass"
    __slots__ = ("_tensor",)

    def __init__(self, tensor: Tensor) -> None:
        if not tensor.is_floating_point():
            msg = f"inference-cell mass is floating point, not {tensor.dtype}"
            raise TypeError(msg)

        expected = len(InferenceCell)

        if tensor.ndim != 2 or tensor.shape[1] != expected:
            msg = (
                "inference-cell mass has axes ('pair', 'cell') with "
                f"{expected} cells, not shape {tuple(tensor.shape)}"
            )
            raise ValueError(msg)

        held = tensor.detach()

        if not bool(held.isfinite().all()) or bool((held < 0.0).any()):
            msg = "inference-cell mass is finite and non-negative"
            raise ValueError(msg)

        object.__setattr__(self, "_tensor", tensor)

    @property
    def tensor(self) -> Tensor:
        return self._tensor

    @property
    def order(self) -> tuple[InferenceCell, ...]:
        return tuple(InferenceCell)

    def __len__(self) -> int:
        return int(self._tensor.shape[0])

    @override
    def __eq__(self, other: object) -> bool:
        return type(other) is CellMass and bool(
            torch.equal(self._tensor, other._tensor)
        )

    @override
    def __hash__(self) -> Never:
        msg = "inference-cell mass contains a tensor and is unhashable"
        raise TypeError(msg)


@final
class EvidenceLedger(FrozenValue):
    """Counts evidence, abstentions, calls, memo hits, and prover time."""

    _noun = "grounding evidence ledger"
    __slots__ = (
        "_abstentions",
        "_calls",
        "_cpu_seconds",
        "_evidence",
        "_memo_hits",
        "_process_starts",
        "_questions",
        "_verified_share",
        "_wall_seconds",
    )

    def __init__(
        self,
        evidence: Mapping[str, int],
        abstentions: Mapping[AbstentionCause, int],
        /,
        *,
        calls: int,
        memo_hits: int,
        cpu_seconds: float,
        questions: int,
        wall_seconds: float,
        process_starts: int | None,
        verified_share: float | None,
    ) -> None:
        object.__setattr__(
            self,
            "_evidence",
            tuple(
                sorted(
                    (name, ensure_non_negative(name, count))
                    for name, count in evidence.items()
                )
            ),
        )
        object.__setattr__(
            self,
            "_abstentions",
            tuple(
                (
                    cause,
                    ensure_non_negative(cause, abstentions.get(cause, 0)),
                )
                for cause in sorted(set(ABSTENTION_CAUSES) | set(abstentions))
            ),
        )
        object.__setattr__(self, "_calls", ensure_non_negative("evidence calls", calls))
        object.__setattr__(
            self, "_memo_hits", ensure_non_negative("evidence memo hits", memo_hits)
        )
        object.__setattr__(
            self, "_questions", ensure_non_negative("evidence questions", questions)
        )
        object.__setattr__(
            self,
            "_process_starts",
            (
                None
                if process_starts is None
                else ensure_non_negative("evidence process starts", process_starts)
            ),
        )
        seconds = ensure_finite("evidence CPU seconds", cpu_seconds)
        wall = ensure_finite("evidence wall seconds", wall_seconds)

        if seconds < 0.0:
            msg = f"evidence CPU seconds are non-negative, not {cpu_seconds!r}"
            raise ValueError(msg)

        if wall < 0.0:
            msg = f"evidence wall seconds are non-negative, not {wall_seconds!r}"
            raise ValueError(msg)

        if verified_share is not None:
            share = ensure_finite("a verified share", verified_share)

            if not 0.0 <= share <= 1.0:
                msg = f"a verified share is a probability, not {verified_share!r}"
                raise ValueError(msg)
        else:
            share = None

        object.__setattr__(self, "_cpu_seconds", seconds)
        object.__setattr__(self, "_wall_seconds", wall)
        object.__setattr__(self, "_verified_share", share)

    @classmethod
    def empty(cls) -> EvidenceLedger:
        return cls(
            {},
            {},
            calls=0,
            memo_hits=0,
            cpu_seconds=0.0,
            questions=0,
            wall_seconds=0.0,
            process_starts=None,
            verified_share=None,
        )

    @classmethod
    def from_evidence(cls, evidence: Sequence[object], /) -> EvidenceLedger:
        counts: dict[str, int] = {}
        abstentions: dict[AbstentionCause, int] = {}

        for item in evidence:
            name = (
                "verdict"
                if type(item).__module__.startswith("lofi.lf")
                else type(item).__name__.lower()
            )
            counts[name] = counts.get(name, 0) + 1

            if isinstance(item, Abstain):
                abstentions[item.cause] = abstentions.get(item.cause, 0) + 1

        return cls(
            counts,
            abstentions,
            calls=len(evidence),
            memo_hits=0,
            cpu_seconds=0.0,
            questions=len(evidence),
            wall_seconds=0.0,
            process_starts=None,
            verified_share=None,
        )

    @classmethod
    def from_service(
        cls,
        evidence: Sequence[object],
        report: ServiceReport,
        /,
    ) -> EvidenceLedger:
        """Build a ledger from evidence and the proof service's step report."""
        counts: dict[str, int] = {}

        for item in evidence:
            name = (
                "verdict"
                if type(item).__module__.startswith("lofi.lf")
                else type(item).__name__.lower()
            )
            counts[name] = counts.get(name, 0) + 1

        return cls(
            counts,
            report.abstentions,
            calls=report.calls,
            memo_hits=report.memo_hits,
            cpu_seconds=report.cpu_seconds,
            questions=report.questions,
            wall_seconds=report.wall_seconds,
            process_starts=report.process_starts,
            verified_share=None,
        )

    @property
    def evidence(self) -> Mapping[str, int]:
        return dict(self._evidence)

    @property
    def abstentions(self) -> Mapping[AbstentionCause, int]:
        return dict(self._abstentions)

    @property
    def calls(self) -> int:
        return self._calls

    @property
    def memo_hits(self) -> int:
        return self._memo_hits

    @property
    def cpu_seconds(self) -> float:
        return self._cpu_seconds

    @property
    def questions(self) -> int:
        return self._questions

    @property
    def wall_seconds(self) -> float:
        return self._wall_seconds

    @property
    def questions_per_second(self) -> float:
        return (
            0.0 if self._wall_seconds == 0.0 else self._questions / self._wall_seconds
        )

    @property
    def memo_hit_rate(self) -> float:
        decided = self._calls + self._memo_hits
        return 0.0 if decided == 0 else self._memo_hits / decided

    @property
    def process_starts(self) -> int | None:
        return self._process_starts

    @property
    def verified_share(self) -> float | None:
        return self._verified_share

    def _identity(self) -> tuple[object, ...]:
        return (
            self._evidence,
            self._abstentions,
            self._calls,
            self._memo_hits,
            self._cpu_seconds,
            self._questions,
            self._wall_seconds,
            self._process_starts,
            self._verified_share,
        )

    @override
    def __eq__(self, other: object) -> bool:
        return type(other) is EvidenceLedger and self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        return hash((EvidenceLedger, self._identity()))


def _pair_vector(field: str, value: Tensor, pairs: int, /, *, boolean: bool) -> Tensor:
    if boolean:
        if value.dtype is not torch.bool:
            msg = f"{field} is boolean, not {value.dtype}"
            raise TypeError(msg)
    elif not value.is_floating_point():
        msg = f"{field} is floating point, not {value.dtype}"
        raise TypeError(msg)

    if value.shape != (pairs,):
        msg = f"one {field} per pair gives shape ({pairs},), not {tuple(value.shape)}"
        raise ValueError(msg)

    return value


@final
class GroundedBatch(FrozenValue):
    """The sole batched prediction published by an inference grounding."""

    _noun = "grounded batch"
    __slots__ = (
        "_auxiliary",
        "_baseline",
        "_cells",
        "_complete",
        "_decided",
        "_gradient_path",
        "_labels",
        "_ledger",
        "_log_probabilities",
        "_residual",
        "_reward",
        "_variance",
    )

    def __init__(
        self,
        log_probabilities: LogDistribution,
        /,
        *,
        labels: tuple[InferenceLabel, ...],
        complete: bool,
        residual: Tensor,
        decided: Tensor,
        cells: CellMass,
        gradient_path: GradientPath,
        baseline: ControlVariate | None,
        auxiliary: Tensor | None,
        ledger: EvidenceLedger,
        variance: Tensor | None,
        reward: InferenceReward | None,
    ) -> None:
        if log_probabilities.axes != ("pair", "label"):
            msg = "grounded log probabilities have axes ('pair', 'label')"
            raise ValueError(msg)

        pairs, width = log_probabilities.tensor.shape

        if len(labels) < 2 or len(set(labels)) != len(labels):
            msg = f"a grounded label inventory is distinct and nontrivial: {labels!r}"
            raise ValueError(msg)

        if any(not isinstance(label, InferenceLabel) for label in labels):
            msg = (
                f"a grounded label inventory contains InferenceLabel values: {labels!r}"
            )
            raise TypeError(msg)

        if width != len(labels):
            msg = (
                f"a grounded distribution has {width} columns for {len(labels)} labels"
            )
            raise ValueError(msg)

        if not isinstance(complete, bool):
            msg = f"grounded completeness is a bool, not {complete!r}"
            raise TypeError(msg)

        if not isinstance(cells, CellMass) or len(cells) != pairs:
            msg = f"one inference-cell row per pair gives {pairs} rows"
            raise ValueError(msg)

        if not isinstance(gradient_path, GradientPath):
            msg = f"a gradient path is a GradientPath, not {gradient_path!r}"
            raise TypeError(msg)

        if baseline is not None and not isinstance(baseline, ControlVariate):
            msg = f"a grounded baseline implements ControlVariate, not {baseline!r}"
            raise TypeError(msg)

        if not isinstance(ledger, EvidenceLedger):
            msg = f"a grounding ledger is EvidenceLedger, not {ledger!r}"
            raise TypeError(msg)

        if reward is not None and not isinstance(reward, InferenceReward):
            msg = f"a grounded reward is InferenceReward, not {reward!r}"
            raise TypeError(msg)

        held_residual = _pair_vector(
            "residual", residual, pairs, boolean=False
        ).detach()

        if not bool(held_residual.isfinite().all()) or bool(
            ((held_residual < 0.0) | (held_residual > 1.0)).any()
        ):
            msg = "each residual is a probability"
            raise ValueError(msg)

        held_decided = _pair_vector("decision", decided, pairs, boolean=True)
        held_variance = None

        if variance is not None:
            held_variance = _pair_vector(
                "variance", variance, pairs, boolean=False
            ).detach()

            if not bool(held_variance.isfinite().all()) or bool(
                (held_variance < 0.0).any()
            ):
                msg = "each estimator variance is finite and non-negative"
                raise ValueError(msg)

        object.__setattr__(self, "_log_probabilities", log_probabilities)
        object.__setattr__(self, "_labels", tuple(labels))
        object.__setattr__(self, "_complete", complete)
        object.__setattr__(self, "_residual", held_residual)
        object.__setattr__(self, "_decided", held_decided)
        object.__setattr__(self, "_cells", cells)
        object.__setattr__(self, "_gradient_path", gradient_path)
        object.__setattr__(self, "_baseline", baseline)
        object.__setattr__(self, "_auxiliary", auxiliary)
        object.__setattr__(self, "_ledger", ledger)
        object.__setattr__(self, "_variance", held_variance)
        object.__setattr__(self, "_reward", reward)

    @property
    def log_probabilities(self) -> LogDistribution:
        return self._log_probabilities

    @property
    def labels(self) -> tuple[InferenceLabel, ...]:
        """Return the label carried by each distribution column."""
        return self._labels

    @property
    def complete(self) -> bool:
        return self._complete

    @property
    def residual(self) -> Tensor:
        return self._residual

    @property
    def decided(self) -> Tensor:
        return self._decided

    @property
    def cells(self) -> CellMass:
        return self._cells

    @property
    def gradient_path(self) -> GradientPath:
        return self._gradient_path

    @property
    def auxiliary(self) -> Tensor | None:
        return self._auxiliary

    def observe_payments(self, payments: Tensor, /) -> None:
        """Update the declared control variate from decided response payments."""
        baseline = self._baseline

        if baseline is None:
            return

        if payments.ndim != 1 or not payments.is_floating_point():
            msg = "grounded response payments are a rank-1 floating tensor"
            raise ValueError(msg)

        held = payments.detach()

        if not bool(held.isfinite().all()):
            msg = "grounded response payments are finite"
            raise ValueError(msg)

        baseline.observe(tuple(float(one) for one in held))

    @property
    def ledger(self) -> EvidenceLedger:
        return self._ledger

    @property
    def variance(self) -> Tensor | None:
        return self._variance

    @property
    def reward(self) -> InferenceReward | None:
        """Return declared proof payoffs, if this batch supports reward training."""
        return self._reward

    @override
    def __eq__(self, other: object) -> bool:
        if type(other) is not GroundedBatch:
            return False

        return (
            self._complete == other._complete
            and self._labels == other._labels
            and self._gradient_path is other._gradient_path
            and self._reward == other._reward
            and self._ledger == other._ledger
            and self._cells == other._cells
            and bool(
                torch.equal(
                    self._log_probabilities.tensor, other._log_probabilities.tensor
                )
            )
            and bool(torch.equal(self._residual, other._residual))
            and bool(torch.equal(self._decided, other._decided))
            and (
                (self._auxiliary is None and other._auxiliary is None)
                or (
                    self._auxiliary is not None
                    and other._auxiliary is not None
                    and bool(torch.equal(self._auxiliary, other._auxiliary))
                )
            )
            and (
                (self._variance is None and other._variance is None)
                or (
                    self._variance is not None
                    and other._variance is not None
                    and bool(torch.equal(self._variance, other._variance))
                )
            )
        )

    @override
    def __hash__(self) -> Never:
        msg = "a grounded batch contains tensors and is unhashable"
        raise TypeError(msg)
