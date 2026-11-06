"""Declared supertag, lexicon, case-study, and clustering diagnostics."""

from typing import Literal, NamedTuple, Self, override

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi.spec._ports import Dependency
from lofi.spec.components.run import EvaluatorSpec
from lofi.spec.evaluators.conventions import ConventionSettings, ConventionSpec

__all__ = [
    "CountStratumSettings",
    "CountStratumSpec",
    "LexicalDiagnosticsSettings",
    "LexicalDiagnosticsSpec",
]


class CountStratumSettings(NamedTuple):
    """Resolved inclusive band over a nonnegative empirical count."""

    at_least: int
    at_most: int | None


class CountStratumSpec(dx.Model, extra="forbid"):
    """Declare one inclusive band over a nonnegative empirical count."""

    at_least: int
    at_most: int | None

    @dx.validates("at_least")
    @classmethod
    def _floor_is_nonnegative(cls, value: int) -> int:
        """Return a nonnegative inclusive floor."""
        return ensure_non_negative("a diagnostic stratum's floor", value)

    @dx.validates("at_most")
    @classmethod
    def _ceiling_is_nonnegative_or_absent(cls, value: int | None) -> int | None:
        """Return a nonnegative inclusive ceiling, or no ceiling."""
        if value is None:
            return None
        return ensure_non_negative("a diagnostic stratum's ceiling", value)

    @dx.model_validator()
    def _ceiling_follows_floor(self) -> Self:
        """Return a nonempty inclusive count band."""
        if self.at_most is not None and self.at_most < self.at_least:
            msg = (
                f"a diagnostic stratum runs from {self.at_least} to "
                f"{self.at_most}, which admits none"
            )
            raise ValueError(msg)
        return self

    def settings(self) -> CountStratumSettings:
        """Return immutable count-band settings."""
        return CountStratumSettings(self.at_least, self.at_most)


def _count_partition(
    field: str, values: dict[str, CountStratumSpec], /
) -> dict[str, CountStratumSpec]:
    """Return named bands that partition every nonnegative integer exactly once."""
    if not values:
        msg = f"lexical diagnostics declare no {field} stratum"
        raise ValueError(msg)

    held = {
        ensure_name(f"a lexical diagnostic {field} stratum", name): band
        for name, band in sorted(values.items())
    }
    ordered = sorted(held.items(), key=lambda entry: (entry[1].at_least, entry[0]))
    expected = 0
    for position, (name, band) in enumerate(ordered):
        if band.at_least != expected:
            msg = (
                f"lexical diagnostic {field} strata leave a gap or overlap at "
                f"count {expected}: {name!r} starts at {band.at_least}"
            )
            raise ValueError(msg)
        if band.at_most is None:
            if position != len(ordered) - 1:
                msg = (
                    f"unbounded lexical diagnostic {field} stratum {name!r} "
                    "precedes another stratum"
                )
                raise ValueError(msg)
            return held
        expected = band.at_most + 1

    msg = f"lexical diagnostic {field} strata stop at count {expected - 1}"
    raise ValueError(msg)


class LexicalDiagnosticsSettings(NamedTuple):
    """Resolved lexical diagnostic settings."""

    readout: str
    beta: float
    clustering_beta: float
    top_k: int
    frequency_strata: tuple[tuple[str, CountStratumSettings], ...]
    ambiguity_strata: tuple[tuple[str, CountStratumSettings], ...]
    word_classes: tuple[tuple[str, tuple[str, ...]], ...]
    case_studies: tuple[tuple[str, tuple[tuple[str, tuple[str, ...]], ...]], ...]
    conventions: ConventionSettings
    metrics: tuple[str, ...]


class LexicalDiagnosticsSpec(EvaluatorSpec):
    """Declare lexical reports over decoded term posteriors."""

    kind: Literal["lexical-diagnostics"] = "lexical-diagnostics"
    readout: str
    beta: float
    clustering_beta: float
    top_k: int
    frequency_strata: dict[str, CountStratumSpec]
    ambiguity_strata: dict[str, CountStratumSpec]
    word_classes: dict[str, tuple[str, ...]]
    case_studies: dict[str, dict[str, tuple[str, ...]]]
    conventions: ConventionSpec

    @dx.validates("readout")
    @classmethod
    def _readout_is_named(cls, value: str) -> str:
        """Return a named complete term readout role."""
        return ensure_name("a lexical diagnostic readout", value)

    @dx.validates("beta")
    @classmethod
    def _beta_is_a_threshold(cls, value: float) -> float:
        """Return a finite C&C-style pruning threshold."""
        held = ensure_finite("a lexical diagnostic beta", value)
        if not 0.0 <= held < 1.0:
            msg = f"a lexical diagnostic beta is outside [0, 1): {value!r}"
            raise ValueError(msg)
        return held

    @dx.model_validator()
    def _case_studies_are_binary_and_named(self) -> Self:
        """Return declared binary contrasts with distinct exact surface forms."""
        held: dict[str, dict[str, tuple[str, ...]]] = {}
        for contrast, groups in sorted(self.case_studies.items()):
            contrast_name = ensure_name("a lexical case-study contrast", contrast)
            if len(groups) != 2:
                msg = (
                    f"case study {contrast_name!r} needs exactly two groups, "
                    f"not {len(groups)}"
                )
                raise ValueError(msg)

            declared: dict[str, tuple[str, ...]] = {}
            owners: dict[str, str] = {}
            for group, forms in sorted(groups.items()):
                group_name = ensure_name(f"case study {contrast_name!r} group", group)
                surfaces = tuple(
                    ensure_name(
                        f"case study {contrast_name!r} group {group_name!r} form",
                        form,
                    )
                    for form in forms
                )
                if not surfaces or len(set(surfaces)) != len(surfaces):
                    msg = (
                        f"case study {contrast_name!r} group {group_name!r} "
                        "is empty or repeats forms"
                    )
                    raise ValueError(msg)
                for surface in surfaces:
                    previous = owners.setdefault(surface, group_name)
                    if previous != group_name:
                        msg = (
                            f"case study {contrast_name!r} form {surface!r} belongs "
                            f"to both {previous!r} and {group_name!r}"
                        )
                        raise ValueError(msg)
                declared[group_name] = surfaces
            held[contrast_name] = declared

        del held
        return self

    @dx.validates("clustering_beta")
    @classmethod
    def _clustering_beta_is_nonnegative(cls, value: float) -> float:
        """Return a finite nonnegative completeness weight for V-measure."""
        held = ensure_finite("a clustering beta", value)
        if held < 0.0:
            msg = f"a clustering beta is negative: {value!r}"
            raise ValueError(msg)
        return held

    @dx.validates("top_k")
    @classmethod
    def _top_k_is_positive(cls, value: int) -> int:
        """Return a positive lexicon-mass report width."""
        if value < 1:
            msg = f"a lexical mass report keeps a positive number, not {value}"
            raise ValueError(msg)
        return value

    @dx.validates("frequency_strata")
    @classmethod
    def _frequency_strata_partition_counts(
        cls, values: dict[str, CountStratumSpec]
    ) -> dict[str, CountStratumSpec]:
        """Return named training-frequency bands covering every count."""
        return _count_partition("frequency", values)

    @dx.validates("ambiguity_strata")
    @classmethod
    def _ambiguity_strata_partition_counts(
        cls, values: dict[str, CountStratumSpec]
    ) -> dict[str, CountStratumSpec]:
        """Return named gold-ambiguity bands covering every count."""
        return _count_partition("ambiguity", values)

    @dx.validates("word_classes")
    @classmethod
    def _classes_are_disjoint_and_named(
        cls, values: dict[str, tuple[str, ...]]
    ) -> dict[str, tuple[str, ...]]:
        """Return nonempty disjoint word classes in canonical name order."""
        if not values:
            msg = "lexical diagnostics declare no word class"
            raise ValueError(msg)
        held: dict[str, tuple[str, ...]] = {}
        owners: dict[str, str] = {}
        for name, forms in sorted(values.items()):
            class_name = ensure_name("a lexical diagnostic word class", name)
            declared = tuple(
                ensure_name(f"word class {class_name!r} form", form) for form in forms
            )
            if not declared or len(set(declared)) != len(declared):
                msg = f"word class {class_name!r} is empty or repeats forms"
                raise ValueError(msg)
            for form in declared:
                previous = owners.setdefault(form, class_name)
                if previous != class_name:
                    msg = (
                        f"form {form!r} belongs to both {previous!r} and {class_name!r}"
                    )
                    raise ValueError(msg)
            held[class_name] = declared
        return held

    @override
    def dependencies(self) -> tuple[Dependency, ...]:
        """Return the complete term readout dependency."""
        return (Dependency(self.readout, "terms-readout"),)

    @override
    def metrics(self, labels: tuple[str, ...] = (), /) -> tuple[str, ...]:
        """Return every row in the declared diagnostic battery."""
        del labels
        classes = tuple(sorted(self.word_classes))
        case_studies = tuple(
            (contrast, group)
            for contrast, groups in sorted(self.case_studies.items())
            for group in sorted(groups)
        )
        frequency_strata = tuple(sorted(self.frequency_strata))
        ambiguity_strata = tuple(sorted(self.ambiguity_strata))
        return (
            "supertag-accuracy",
            "supertag-multi-accuracy",
            "supertag-ambiguity",
            "supertag-confusion",
            "clustering-homogeneity",
            "clustering-completeness",
            "clustering-v-measure",
            "clustering-adjusted-mutual-information",
            "clustering-adjusted-rand",
            *(f"supertag-accuracy-frequency-{name}" for name in frequency_strata),
            *(f"supertag-accuracy-ambiguity-{name}" for name in ambiguity_strata),
            *(f"lexicon-mass-{name}" for name in classes),
            *(f"case-study-{contrast}-{group}" for contrast, group in case_studies),
            "collapse-rate",
            "decoded-residual-mass",
            "decode-refusal-rate",
        )

    def settings(self) -> LexicalDiagnosticsSettings:
        """Return immutable settings for the evaluator plugin."""
        return LexicalDiagnosticsSettings(
            self.readout,
            self.beta,
            self.clustering_beta,
            self.top_k,
            tuple(
                (name, band.settings()) for name, band in self.frequency_strata.items()
            ),
            tuple(
                (name, band.settings()) for name, band in self.ambiguity_strata.items()
            ),
            tuple((name, forms) for name, forms in self.word_classes.items()),
            tuple(
                (
                    contrast,
                    tuple((group, groups[group]) for group in sorted(groups)),
                )
                for contrast, groups in sorted(self.case_studies.items())
            ),
            self.conventions.settings(),
            self.metrics(),
        )
