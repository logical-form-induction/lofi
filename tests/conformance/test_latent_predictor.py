from typing import Final, Literal, override

import pytest
import torch
from tests.conformance.latent_predictor import (
    LatentPredictorSubject,
    LatentPredictorSuite,
    declared_terms_are_stable,
    latent_scores_are_repeatable,
    latent_scores_cover_every_row,
)

from lofi._checks import LawViolationError
from lofi.judgment import (
    DesignCodec,
    JudgmentDesign,
    JudgmentRow,
    Predictor,
    RowColumns,
)

READOUT = "p_gram"
LEVELS = 5


class Readout(Predictor, extra="forbid"):
    kind: Literal["conformance_readout"] = "conformance_readout"
    name: str


class ReadoutPredictor:
    def __init__(self, name: str) -> None:
        self._terms = (Readout(name=name),)
        self._name = name

    @property
    def terms(self) -> tuple[Predictor, ...]:
        return self._terms

    def latent_scores(self, design: JudgmentDesign, /) -> torch.Tensor:
        return design.readouts[self._name]


class UntermedPredictor(ReadoutPredictor):
    @property
    @override
    def terms(self) -> tuple[Predictor, ...]:
        return ()


class UnstableTermsPredictor(ReadoutPredictor):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._reads = 0

    @property
    @override
    def terms(self) -> tuple[Predictor, ...]:
        self._reads += 1

        return (Readout(name=f"{self._name}-{self._reads}"),)


class ShortPredictor(ReadoutPredictor):
    @override
    def latent_scores(self, design: JudgmentDesign, /) -> torch.Tensor:
        return super().latent_scores(design)[:-1]


class IntegerPredictor(ReadoutPredictor):
    @override
    def latent_scores(self, design: JudgmentDesign, /) -> torch.Tensor:
        return super().latent_scores(design).to(torch.int64)


class DivergentPredictor(ReadoutPredictor):
    @override
    def latent_scores(self, design: JudgmentDesign, /) -> torch.Tensor:
        return super().latent_scores(design) / 0.0


class DriftingPredictor(ReadoutPredictor):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._calls = 0

    @override
    def latent_scores(self, design: JudgmentDesign, /) -> torch.Tensor:
        self._calls += 1

        return super().latent_scores(design) + float(self._calls)


def design() -> JudgmentDesign:
    rows = tuple(
        JudgmentRow(
            f"participant-{index % 2}",
            f"item-{index}",
            index % LEVELS,
            RowColumns({}, {READOUT: float(index) / 4.0}),
        )
        for index in range(6)
    )

    return DesignCodec(LEVELS, readouts=(READOUT,)).encode(rows)


DESIGN: Final = design()


class TestAConformingPredictorObeysTheLaws(LatentPredictorSuite):
    @pytest.fixture
    def latent_predictor_subject(self) -> LatentPredictorSubject:
        return LatentPredictorSubject(ReadoutPredictor(READOUT), DESIGN)


def test_the_terms_law_catches_an_empty_term_tuple() -> None:
    with pytest.raises(LawViolationError, match="terms are empty"):
        declared_terms_are_stable(
            LatentPredictorSubject(UntermedPredictor(READOUT), DESIGN)
        )


def test_the_terms_law_catches_terms_that_change_between_reads() -> None:
    subject = LatentPredictorSubject(UnstableTermsPredictor(READOUT), DESIGN)

    with pytest.raises(LawViolationError, match="two reads of the terms disagree"):
        declared_terms_are_stable(subject)


def test_the_coverage_law_catches_a_missing_row() -> None:
    with pytest.raises(LawViolationError, match="scores have shape"):
        latent_scores_cover_every_row(
            LatentPredictorSubject(ShortPredictor(READOUT), DESIGN)
        )


def test_the_coverage_law_catches_a_nonfloating_kind() -> None:
    with pytest.raises(LawViolationError, match="not a floating-point kind"):
        latent_scores_cover_every_row(
            LatentPredictorSubject(IntegerPredictor(READOUT), DESIGN)
        )


def test_the_coverage_law_catches_a_nonfinite_score() -> None:
    with pytest.raises(LawViolationError, match="nonfinite value"):
        latent_scores_cover_every_row(
            LatentPredictorSubject(DivergentPredictor(READOUT), DESIGN)
        )


def test_the_repeatability_law_catches_a_predictor_that_drifts() -> None:
    subject = LatentPredictorSubject(DriftingPredictor(READOUT), DESIGN)

    with pytest.raises(LawViolationError, match="two scorings differ"):
        latent_scores_are_repeatable(subject)
