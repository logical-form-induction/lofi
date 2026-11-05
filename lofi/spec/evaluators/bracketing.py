"""Held-out bracketing evaluator declaration."""

from typing import Literal, NamedTuple, override

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.spec.components.run import EvaluatorSpec
from lofi.spec.evaluators.conventions import ConventionSettings, ConventionSpec

__all__ = [
    "BracketingEvaluatorSettings",
    "BracketingEvaluatorSpec",
    "NoPunctuationSourceSpec",
    "PunctuationSourceSpec",
    "SurfacePunctuationSourceSpec",
]

type BracketingDecoder = Literal["minimum-risk", "viterbi"]
"""A declared tree decoder."""


class PunctuationSourceSpec(dx.TaggedUnion, discriminator="source", extra="forbid"):
    """Declare how an item's punctuation positions are identified."""

    source: Literal["none", "surface"]
    forms: tuple[str, ...]

    def settings(self) -> tuple[Literal["none", "surface"], tuple[str, ...]]:
        """Return the source and its declared surface inventory."""
        return self.source, self.forms


class NoPunctuationSourceSpec(PunctuationSourceSpec):
    """Declare that the dataset supplies no punctuation positions."""

    source: Literal["none"] = "none"

    @dx.validates("forms")
    @classmethod
    def _forms_are_empty(cls, forms: tuple[str, ...]) -> tuple[str, ...]:
        """Return the empty surface inventory."""
        if forms:
            msg = f"a source of no punctuation declares surface forms: {forms!r}"
            raise ValueError(msg)

        return forms


class SurfacePunctuationSourceSpec(PunctuationSourceSpec):
    """Identify punctuation by exact token surface forms."""

    source: Literal["surface"] = "surface"
    forms: tuple[str, ...]

    @dx.validates("forms")
    @classmethod
    def _forms_are_named_distinct_and_nonempty(
        cls, forms: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Return a nonempty sequence of distinct surface forms."""
        held = tuple(ensure_name("a punctuation surface form", form) for form in forms)

        if not held:
            msg = "a surface punctuation source declares no forms"
            raise ValueError(msg)

        if len(set(held)) != len(held):
            msg = f"a surface punctuation source repeats a form: {held!r}"
            raise ValueError(msg)

        return held


class BracketingEvaluatorSettings(NamedTuple):
    """The settings a bracketing evaluator is built from.

    Attributes
    ----------
    conventions : ConventionSettings
        The conventions spans are scored and reported under.
    punctuation_source : Literal["none", "surface"]
        How punctuation positions are identified.
    punctuation_forms : tuple[str, ...]
        Exact surface forms used by the surface source.
    decoder : BracketingDecoder
        Tree-decoding objective.
    """

    conventions: ConventionSettings
    punctuation_source: Literal["none", "surface"]
    punctuation_forms: tuple[str, ...]
    decoder: BracketingDecoder


class BracketingEvaluatorSpec(EvaluatorSpec):
    """Declare unlabeled bracketing F1 over the held-out split.

    Parameters
    ----------
    conventions : ConventionSpec
        The conventions spans are scored and reported under.
    punctuation : PunctuationSourceSpec
        How punctuation positions are identified.
    decoder : BracketingDecoder
        Whether prediction maximizes constituent mass or tree probability.
    """

    kind: Literal["bracketing"] = "bracketing"
    conventions: ConventionSpec
    punctuation: PunctuationSourceSpec
    decoder: BracketingDecoder

    @override
    def metrics(self, labels: tuple[str, ...] = (), /) -> tuple[str, ...]:
        """Return the one metric the evaluator reports."""
        del labels
        return ("unlabeled-bracketing-f1",)

    def settings(self) -> BracketingEvaluatorSettings:
        """Return the settings the plugin builds this evaluator from."""
        punctuation_source, punctuation_forms = self.punctuation.settings()

        return BracketingEvaluatorSettings(
            conventions=self.conventions.settings(),
            punctuation_source=punctuation_source,
            punctuation_forms=punctuation_forms,
            decoder=self.decoder,
        )
