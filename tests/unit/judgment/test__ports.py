import annotationlib
from typing import TYPE_CHECKING, Literal, NamedTuple

import pytest
import torch

from lofi.judgment import (
    BoundedScale,
    ClosedScale,
    DesignCodec,
    InferenceMethod,
    JudgedItem,
    JudgmentRow,
    JudgmentSource,
    Linking,
    LinkingCompiler,
    LinkingDomain,
    LinkingSpec,
    LinkingSteps,
    LinkTerm,
    LmScore,
    Name,
    NutsPlan,
    OrdinalScale,
    PosteriorFit,
    PosteriorFitter,
    ProgramRuntime,
    RatedItem,
    Rating,
    ReadoutColumns,
    RelatedItem,
    Relation,
    ResponseBackend,
    RowColumns,
    SentenceScore,
    SpecialCase,
    ensure_nuts_fitter,
)
from lofi.stats import Interval

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from types import FunctionType

    from torch import Tensor

    from lofi.judgment import (
        BoundProgram,
        ResponseDeclaration,
        ResponseDesign,
        ResponseProgram,
    )


class CompiledResponse(NamedTuple):
    levels: int


class Backend:
    def compile(self, spec: ResponseDeclaration, /) -> CompiledResponse:
        del spec

        return CompiledResponse(0)

    def log_likelihood(
        self,
        compiled: CompiledResponse,
        params: Tensor,
        design: ResponseDesign,
        /,
    ) -> Tensor:
        del compiled, params

        return torch.zeros(design.n_rows)


class CompileOnly:
    def compile(self, spec: ResponseDeclaration, /) -> CompiledResponse:
        del spec

        return CompiledResponse(0)


def annotations_of(function: FunctionType) -> dict[str, str]:
    return annotationlib.get_annotations(function, format=annotationlib.Format.STRING)


def test_a_complete_structural_implementation_satisfies_the_protocol() -> None:
    assert isinstance(Backend(), ResponseBackend)


def test_a_closed_scale_satisfies_the_bounded_scale_port() -> None:
    assert isinstance(ClosedScale(0.0, 100.0), BoundedScale)


def test_a_backend_without_an_evaluator_does_not_satisfy_the_protocol() -> None:
    assert not isinstance(CompileOnly(), ResponseBackend)


def test_compile_preserves_the_backend_specific_representation() -> None:
    annotations = annotations_of(ResponseBackend.compile)

    assert annotations["spec"] == "ResponseDeclaration"
    assert annotations["return"] == "Compiled"


def test_log_likelihood_takes_the_same_compiled_representation() -> None:
    annotations = annotations_of(ResponseBackend.log_likelihood)

    assert annotations["compiled"] == "Compiled"
    assert annotations["params"] == "Tensor"
    assert annotations["design"] == "ResponseDesign"
    assert annotations["return"] == "Tensor"


class ScoringRuntime:
    def load(self, program: ResponseProgram, /) -> None:
        del program

    def observed_log_probabilities(
        self, program: BoundProgram, params: Tensor, /
    ) -> Tensor:
        del params

        return torch.zeros(program.rows)


class LoadingRuntime:
    def load(self, program: ResponseProgram, /) -> None:
        del program


class ScoringOnlyRuntime:
    def observed_log_probabilities(
        self, program: BoundProgram, params: Tensor, /
    ) -> Tensor:
        del params

        return torch.zeros(program.rows)


def test_a_runtime_loading_and_scoring_a_bound_program_answers_the_port() -> None:
    assert isinstance(ScoringRuntime(), ProgramRuntime)


def test_a_runtime_that_only_loads_or_only_scores_does_not_answer_the_port() -> None:
    assert not isinstance(LoadingRuntime(), ProgramRuntime)
    assert not isinstance(ScoringOnlyRuntime(), ProgramRuntime)


def test_a_runtime_loads_a_compiled_program_and_scores_it_bound_at_a_vector() -> None:
    loading = annotations_of(ProgramRuntime.load)
    scoring = annotations_of(ProgramRuntime.observed_log_probabilities)

    assert loading["program"] == "ResponseProgram"
    assert loading["return"] == "None"
    assert scoring["program"] == "BoundProgram"
    assert scoring["params"] == "Tensor"
    assert scoring["return"] == "Tensor"


class Fitter:
    @property
    def method(self) -> Literal[InferenceMethod.NUTS]:
        return InferenceMethod.NUTS

    @property
    def plan(self) -> NutsPlan:
        return NutsPlan(warmup=1, retained=4, chains=2, seed=0)

    def fit(self, program: BoundProgram, /) -> PosteriorFit:
        del program
        return PosteriorFit(
            draws={"parameter": torch.zeros((2, 4, 1))},
            log_densities=torch.zeros((2, 4)),
            acceptance_rates=torch.ones(2),
            divergence_counts=torch.zeros(2, dtype=torch.int64),
            r_hat={"parameter": torch.ones(1)},
            effective_sample_sizes={"parameter": torch.ones(1)},
        )


class InferenceStandIn:
    def __init__(self, method: InferenceMethod) -> None:
        self._method = method

    @property
    def method(self) -> InferenceMethod:
        return self._method

    @property
    def plan(self) -> NutsPlan:
        return NutsPlan(warmup=1, retained=4, chains=2, seed=0)

    def fit(self, program: BoundProgram, /) -> PosteriorFit:
        return Fitter().fit(program)


def test_a_complete_nuts_fitter_satisfies_the_posterior_port() -> None:
    fitter = Fitter()

    assert isinstance(fitter, PosteriorFitter)
    assert ensure_nuts_fitter(fitter) is fitter


@pytest.mark.parametrize(
    "method",
    [InferenceMethod.MAP, InferenceMethod.FULL_RANK],
)
def test_a_full_shape_inference_stand_in_is_refused_at_the_nuts_boundary(
    method: InferenceMethod,
) -> None:
    stand_in = InferenceStandIn(method)

    assert isinstance(stand_in, PosteriorFitter)
    with pytest.raises(ValueError, match="must run genuine NUTS"):
        ensure_nuts_fitter(stand_in)


def test_the_posterior_port_requires_a_bound_program_and_returns_a_fit() -> None:
    annotations = annotations_of(PosteriorFitter.fit)

    assert annotations["program"] == "BoundProgram"
    assert annotations["return"] == "PosteriorFit"


class Columns:
    def __init__(self, rows: int) -> None:
        self._readouts = {"score": torch.zeros(rows)}

    @property
    def n_rows(self) -> int:
        return len(self._readouts["score"])

    @property
    def readouts(self) -> Mapping[str, Tensor]:
        return self._readouts


class Link:
    @property
    def domain(self) -> LinkingDomain:
        return LinkingDomain("score", {"score": Interval(-1.0, 1.0)})

    @property
    def special_cases(self) -> tuple[SpecialCase, ...]:
        return ()

    def contribution(self, columns: ReadoutColumns, parameters: Tensor, /) -> Tensor:
        del parameters

        return columns.readouts["score"]


class DomainOnly:
    @property
    def domain(self) -> LinkingDomain:
        return LinkingDomain("score", {"score": Interval(-1.0, 1.0)})


def test_readout_columns_is_answered_by_a_design_and_by_a_column_source() -> None:
    assert isinstance(Columns(3), ReadoutColumns)
    assert isinstance(
        DesignCodec(2, readouts=("score",)).encode([]),
        ReadoutColumns,
    )


def test_a_linking_with_a_domain_cases_and_a_contribution_answers_the_port() -> None:
    assert isinstance(Link(), Linking)


def test_a_linking_without_a_contribution_does_not_answer_the_port() -> None:
    assert not isinstance(DomainOnly(), Linking)


def column_steps(
    term: LinkTerm,
    prefix: str,
    /,
    *,
    unit_slope: bool,
    signed_slope: bool,
) -> LinkingSteps:
    del prefix, signed_slope, unit_slope

    return LinkingSteps({}, Name(term.readout))


def test_a_linking_compiler_takes_a_term_a_prefix_and_the_slope_identification() -> (
    None
):
    term = LinkTerm(readout="score", linking=LinkingSpec(family="column"))
    compile_linking: LinkingCompiler = column_steps

    assert isinstance(column_steps, LinkingCompiler)
    assert compile_linking(
        term, "score", unit_slope=True, signed_slope=False
    ).expression == Name("score")
    assert (
        compile_linking(term, "score", unit_slope=False, signed_slope=False).draws == {}
    )


class LengthScorer:
    def score(self, sentences: Sequence[str], /) -> tuple[SentenceScore, ...]:
        scores = []

        for sentence in sentences:
            tokens = tuple(sentence.split())
            per_token = torch.full((len(tokens),), -1.0)
            scores.append(SentenceScore(tokens, per_token, per_token))

        return tuple(scores)


class Tokenizer:
    def tokenize(self, sentence: str, /) -> tuple[str, ...]:
        return tuple(sentence.split())


def test_a_scorer_returning_one_sentence_score_per_sentence_answers_the_port() -> None:
    assert isinstance(LengthScorer(), LmScore)


def test_a_tokenizer_without_a_score_method_does_not_answer_the_port() -> None:
    assert not isinstance(Tokenizer(), LmScore)


def test_score_takes_sentences_and_returns_their_scores() -> None:
    annotations = annotations_of(LmScore.score)

    assert annotations["sentences"] == "Sequence[str]"
    assert annotations["return"] == "tuple[SentenceScore, ...]"


class Sentence(NamedTuple):
    item_id: str
    languages: tuple[str, ...]


class SevenPoint:
    @property
    def levels(self) -> int | None:
        return 7

    def level_of(self, value: float, /) -> int:
        return int(value) - 1


class Response(NamedTuple):
    value: float | None
    scale: SevenPoint
    annotator: str | None


class Rated(NamedTuple):
    item: Sentence
    supervision: Response


class Entailment(NamedTuple):
    other: str
    relation: str
    annotator: str | None


class Related(NamedTuple):
    item: Sentence
    supervision: Entailment


class Rows:
    @property
    def response_levels(self) -> int:
        return 7

    @property
    def factors(self) -> tuple[str, ...]:
        return ("language",)

    @property
    def skips(self) -> Mapping[str, int]:
        return {}

    def rows(self) -> Iterable[JudgmentRow]:
        return (JudgmentRow("p1", "i1", 3, RowColumns({"language": "en"}, {})),)


class RowsWithoutSkips:
    @property
    def response_levels(self) -> int:
        return 7

    @property
    def factors(self) -> tuple[str, ...]:
        return ()

    def rows(self) -> Iterable[JudgmentRow]:
        return ()


def test_the_record_shapes_a_reader_emits_answer_the_record_ports() -> None:
    sentence = Sentence("s1", ("en",))
    rated = Rated(sentence, Response(5.0, SevenPoint(), "p1"))
    related = Related(sentence, Entailment("s2", "entails", None))

    assert isinstance(sentence, JudgedItem)
    assert isinstance(SevenPoint(), OrdinalScale)
    assert isinstance(rated.supervision, Rating)
    assert isinstance(rated, RatedItem)
    assert isinstance(related.supervision, Relation)
    assert isinstance(related, RelatedItem)


def test_a_rating_is_not_a_relation_and_a_relation_is_not_a_rating() -> None:
    assert not isinstance(Response(5.0, SevenPoint(), "p1"), Relation)
    assert not isinstance(Entailment("s2", "entails", None), Rating)


def test_a_source_with_levels_factors_skips_and_rows_answers_the_port() -> None:
    assert isinstance(Rows(), JudgmentSource)


def test_a_source_without_skips_does_not_answer_the_port() -> None:
    assert not isinstance(RowsWithoutSkips(), JudgmentSource)
