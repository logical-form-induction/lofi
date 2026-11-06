"""Response-model protocols."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from torch import Generator, Tensor

    from lofi.judgment._design import JudgmentDesign, JudgmentRow, ResponseDesign
    from lofi.judgment._program import LinkingSteps
    from lofi.judgment._results import NutsPlan, PosteriorFit
    from lofi.judgment._spec import (
        AgreementBin,
        InferenceMethod,
        LinkTerm,
        MapInference,
        Predictor,
        ResponseDeclaration,
    )
    from lofi.judgment._values import LinkingDomain, SentenceScore, SpecialCase
    from lofi.judgment.compiler import BoundProgram, ResponseProgram, Slot
    from lofi.judgment.crossed import CrossedModel, CrossedPosterior
    from lofi.judgment.fitting import PointFit

__all__ = [
    "BoundedRatedItem",
    "BoundedRating",
    "BoundedScale",
    "ContinuousScale",
    "JudgedItem",
    "JudgmentSource",
    "LatentPredictor",
    "Linking",
    "LinkingCompiler",
    "LmScore",
    "MarginalLikelihoodError",
    "OrdinalScale",
    "PointFitter",
    "PointFittingError",
    "PointObjective",
    "PosteriorFitter",
    "PosteriorFittingError",
    "ProgramRuntime",
    "RatedItem",
    "Rating",
    "ReadoutColumns",
    "RelatedItem",
    "Relation",
    "ResponseBackend",
    "StratifiedTarget",
    "StratifiedTargets",
    "VariationalGuide",
]


class MarginalLikelihoodError(ValueError):
    """A response program cannot be evaluated as a truncated marginal likelihood."""


class PosteriorFittingError(RuntimeError):
    """A posterior sampler reported a numerical condition in a declared fit."""


class PointFittingError(RuntimeError):
    """A point optimizer reported a numerical condition in a declared fit."""


class StratifiedTarget(Protocol):
    """A scored row assigned to an item-level agreement stratum."""

    @property
    def item(self) -> str:
        """Return the item identifier."""
        ...

    @property
    def stratum(self) -> AgreementBin:
        """Return the item's agreement stratum."""
        ...


class StratifiedTargets(Protocol):
    """Item-keyed rows assigned to agreement strata."""

    @property
    def items(self) -> Mapping[str, StratifiedTarget]:
        """Return the target values by item identifier."""
        ...

    @property
    def rows(self) -> Sequence[StratifiedTarget]:
        """Return the target rows in scoring order."""
        ...


class ContinuousScale(Protocol):
    """A finite continuous response scale before dense encoding."""

    @property
    def kind(self) -> str:
        """Return the scale kind."""
        ...

    @property
    def minimum(self) -> float:
        """Return the inclusive lower endpoint."""
        ...

    @property
    def maximum(self) -> float | None:
        """Return the inclusive upper endpoint, if declared."""
        ...


class BoundedRating(Protocol):
    """A scalar response together with the items it relates."""

    @property
    def value(self) -> float | None:
        """Return the response, or None when missing."""
        ...

    @property
    def scale(self) -> ContinuousScale:
        """Return the response scale."""
        ...

    @property
    def annotator(self) -> str | None:
        """Return the participant identifier, if recorded."""
        ...

    @property
    def related_items(self) -> tuple[str, ...]:
        """Return the other items this response refers to, in semantic order."""
        ...


class BoundedRatedItem(Protocol):
    """An item paired with one bounded scalar response."""

    @property
    def item(self) -> JudgedItem:
        """Return the rated item."""
        ...

    @property
    def supervision(self) -> BoundedRating:
        """Return the bounded response and its item references."""
        ...


@runtime_checkable
class ResponseBackend[Compiled](Protocol):
    """A backend for compiling and evaluating response models."""

    def compile(self, spec: ResponseDeclaration, /) -> Compiled:
        """Compile a response specification.

        Parameters
        ----------
        spec : ResponseDeclaration
            Response model to compile.

        Returns
        -------
        Compiled
            Backend-specific compiled model.
        """
        ...

    def log_likelihood(
        self,
        compiled: Compiled,
        params: Tensor,
        design: ResponseDesign,
        /,
    ) -> Tensor:
        """Return one log likelihood per response.

        Parameters
        ----------
        compiled : Compiled
            Compiled response model.
        params : Tensor
            Model parameters.
        design : ResponseDesign
            Response rows and predictor columns.

        Returns
        -------
        Tensor
            Log likelihood for each response row.
        """
        ...


@runtime_checkable
class ProgramRuntime(Protocol):
    """A runtime that builds a compiled program and scores its rows."""

    def load(self, program: ResponseProgram, /) -> None:
        """Check that the runtime can build a program, refusing one it cannot.

        Parameters
        ----------
        program : ResponseProgram
            Compiled program.

        Raises
        ------
        ValueError
            If the program has a step the runtime has no form for.
        """
        ...

    def observed_log_probabilities(
        self,
        program: BoundProgram,
        params: Tensor,
        /,
    ) -> Tensor:
        """Return the log probability of each row's recorded level.

        The program is run over its host data with every drawn site held
        at the value the parameter vector gives it, and the observation is
        scored one row at a time.

        Parameters
        ----------
        program : BoundProgram
            Program bound to host data.
        params : Tensor
            Drawn-site values in program layout order.

        Returns
        -------
        Tensor
            Log probability for each response row.
        """
        ...


@runtime_checkable
class PosteriorFitter(Protocol):
    """A declared posterior fitter for a compiled response program."""

    @property
    def method(self) -> InferenceMethod:
        """Return the sampling method recorded for the fit."""
        ...

    @property
    def plan(self) -> NutsPlan:
        """Return the complete immutable sampling plan."""
        ...

    def fit(self, program: BoundProgram, /) -> PosteriorFit:
        """Draw and diagnose the posterior of a bound response program.

        Parameters
        ----------
        program : BoundProgram
            Compiled response program bound to its observed design.

        Returns
        -------
        PosteriorFit
            Chain-preserving posterior draws and diagnostics.

        Raises
        ------
        PosteriorFittingError
            If the sampler reports a numerical condition: a nonfinite
            position, draw, density or convergence diagnostic, or a chain
            whose every retained transition diverged.
        MemoryError
            If the accelerator runs out of memory.
        """
        ...


@runtime_checkable
class PointObjective(Protocol):
    """A support-parameterized scalar objective with one owner-derived layout."""

    @property
    def layout(self) -> tuple[Slot, ...]:
        """Return the constrained parameter layout in owner order."""
        ...

    def initial_position(self, *, generator: Generator, scale: float) -> Tensor:
        """Draw one flat unconstrained position from a caller-owned stream."""
        ...

    def parameters(self, position: Tensor, /) -> Tensor:
        """Map one unconstrained position into the constrained owner layout."""
        ...

    def log_density(self, parameters: Tensor, /) -> Tensor:
        """Return the scalar log density of constrained parameters."""
        ...


@runtime_checkable
class PointFitter(Protocol):
    """A declared maximum a posteriori fitter for a point objective."""

    @property
    def method(self) -> InferenceMethod:
        """Return maximum a posteriori inference."""
        ...

    @property
    def plan(self) -> MapInference:
        """Return the complete immutable optimization declaration."""
        ...

    def fit(self, objective: PointObjective, /) -> PointFit:
        """Optimize and return one point in the objective's declared layout."""
        ...


@runtime_checkable
class LatentPredictor(Protocol):
    """An evaluator for the terms a response specification declares."""

    @property
    def terms(self) -> tuple[Predictor, ...]:
        """Return the declared terms this evaluator implements."""
        ...

    def latent_scores(self, design: JudgmentDesign, /) -> Tensor:
        """Return one latent score per judgment row.

        Parameters
        ----------
        design : JudgmentDesign
            Rows and columns to evaluate.

        Returns
        -------
        Tensor
            Latent score for each row.
        """
        ...


@runtime_checkable
class ReadoutColumns(Protocol):
    """The readout columns of a design, one value per row."""

    @property
    def n_rows(self) -> int:
        """Return the number of rows each column has."""
        ...

    @property
    def readouts(self) -> Mapping[str, Tensor]:
        """Return the readout columns by name."""
        ...


@runtime_checkable
class LinkingCompiler(Protocol):
    """The program form of one linking family, from a link term declaring it."""

    def __call__(
        self,
        term: LinkTerm,
        prefix: str,
        /,
        *,
        unit_slope: bool,
        signed_slope: bool,
    ) -> LinkingSteps:
        """Return a link term's parameter draws and its expression.

        Under a unit slope the linking's slope is held at one rather than
        drawn. Under a signed slope a free slope may cross zero. A linking
        without a slope ignores both choices.

        Parameters
        ----------
        term : LinkTerm
            Link term to compile.
        prefix : str
            Prefix for generated site names.
        unit_slope : bool
            Whether to hold the slope at one.
        signed_slope : bool
            Whether a fitted slope may be negative.

        Returns
        -------
        LinkingSteps
            Parameter draws and latent-scale expression.
        """
        ...


@runtime_checkable
class Linking(Protocol):
    """A map from readout columns and fitted parameters to the latent scale.

    A linking reads the readout columns its domain declares and no other
    column of a design. The recorded responses are not among what it is
    given.
    """

    @property
    def domain(self) -> LinkingDomain:
        """Return the columns read, the argument, and the fitted parameters."""
        ...

    @property
    def special_cases(self) -> tuple[SpecialCase, ...]:
        """Return the parameter values at which the linking reduces to a closed form."""
        ...

    def contribution(self, columns: ReadoutColumns, parameters: Tensor, /) -> Tensor:
        """Return one latent-scale contribution per row.

        Parameters
        ----------
        columns : ReadoutColumns
            Declared input columns.
        parameters : Tensor
            Fitted parameters in domain order.

        Returns
        -------
        Tensor
            Latent-scale contribution for each row.
        """
        ...


@runtime_checkable
class LmScore(Protocol):
    """Direct per-token language model scores for sentences.

    Implementations read token probabilities without adding a prompt or
    instruction.
    """

    def score(self, sentences: Sequence[str], /) -> tuple[SentenceScore, ...]:
        """Return one score per sentence, in the sentences' order.

        A sentence's score depends on that sentence alone: scored on its own
        or in any batch, at any padded length, it has the same tokens and
        the same values, and scoring it again at a fixed seed repeats them.

        Parameters
        ----------
        sentences : Sequence[str]
            Sentences to score.

        Returns
        -------
        tuple[SentenceScore, ...]
            Scores in input order.
        """
        ...


@runtime_checkable
class JudgedItem(Protocol):
    """An item a judgment was given to."""

    @property
    def item_id(self) -> str:
        """Return the identifier the item is joined on."""
        ...


@runtime_checkable
class OrdinalScale(Protocol):
    """A rating scale that may have ordered levels."""

    @property
    def levels(self) -> int | None:
        """Return the number of ordered levels, or None for another scale kind."""
        ...

    def level_of(self, value: float, /) -> int:
        """Return a response's zero-based level.

        Parameters
        ----------
        value : float
            Recorded response.

        Returns
        -------
        int
            Zero-based ordinal level.
        """
        ...


@runtime_checkable
class BoundedScale(Protocol):
    """A finite response interval with both endpoints included."""

    @property
    def lower(self) -> float:
        """Return the inclusive lower endpoint."""
        ...

    @property
    def upper(self) -> float:
        """Return the inclusive upper endpoint."""
        ...


@runtime_checkable
class Rating(Protocol):
    """One participant's response on a scale."""

    @property
    def value(self) -> float | None:
        """Return the response, or None when it is missing."""
        ...

    @property
    def scale(self) -> OrdinalScale:
        """Return the scale the response was given on."""
        ...

    @property
    def annotator(self) -> str | None:
        """Return the participant's identifier, if recorded."""
        ...


@runtime_checkable
class RatedItem(Protocol):
    """An item paired with one rating of it."""

    @property
    def item(self) -> JudgedItem:
        """Return the rated item."""
        ...

    @property
    def supervision(self) -> Rating:
        """Return the rating."""
        ...


@runtime_checkable
class Relation(Protocol):
    """One response naming the relation of an item to another."""

    @property
    def other(self) -> str:
        """Return the identifier of the item the relation runs to."""
        ...

    @property
    def relation(self) -> str:
        """Return the relation's name."""
        ...

    @property
    def annotator(self) -> str | None:
        """Return the participant's identifier, if recorded."""
        ...


@runtime_checkable
class RelatedItem(Protocol):
    """An item paired with one relation asserted from it."""

    @property
    def item(self) -> JudgedItem:
        """Return the item the relation runs from."""
        ...

    @property
    def supervision(self) -> Relation:
        """Return the relation."""
        ...


@runtime_checkable
class JudgmentSource(Protocol):
    """A collection of judgments read as design rows.

    Every judgment of the collection becomes one row or one counted skip,
    so the rows and the skips together account for the collection.
    """

    @property
    def response_levels(self) -> int:
        """Return the number of ordered response levels every row is over."""
        ...

    @property
    def factors(self) -> tuple[str, ...]:
        """Return the factors every row has a level of, in column order."""
        ...

    @property
    def skips(self) -> Mapping[str, int]:
        """Return how many judgments became no row, by reason."""
        ...

    def rows(self) -> Iterable[JudgmentRow]:
        """Return one row per judgment that became one."""
        ...


@runtime_checkable
class VariationalGuide[State](Protocol):
    """A variational family over a crossed model, fitted by coordinate ascent.

    A guide holds no state of its own. One call returns the state a fit
    starts from, each sweep returns the next state, and the state alone says
    what the approximate posterior is.
    """

    @property
    def method(self) -> InferenceMethod:
        """Return the inference a fit under this guide is recorded as."""
        ...

    def initialize(self, model: CrossedModel, /) -> State:
        """Return the state a fit starts from.

        Parameters
        ----------
        model : CrossedModel
            Model to fit.

        Returns
        -------
        State
            The starting state.
        """
        ...

    def update(self, model: CrossedModel, state: State, /) -> State:
        """Return the state after one coordinate ascent sweep.

        A sweep never lowers the evidence lower bound.

        Parameters
        ----------
        model : CrossedModel
            Model to fit.
        state : State
            State to improve on.

        Returns
        -------
        State
            The state after the sweep.
        """
        ...

    def evidence_bound(self, model: CrossedModel, state: State, /) -> Tensor:
        """Return the evidence lower bound at a state.

        Parameters
        ----------
        model : CrossedModel
            Model the state approximates the posterior of.
        state : State
            State to score.

        Returns
        -------
        Tensor
            The bound, as a scalar.
        """
        ...

    def posterior(self, state: State, /) -> CrossedPosterior:
        """Return the marginal means and variances a state implies.

        Parameters
        ----------
        state : State
            State to read.

        Returns
        -------
        CrossedPosterior
            One mean and one variance per parameter, in layout order.
        """
        ...
