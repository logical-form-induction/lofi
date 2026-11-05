"""Semantic lexicon declarations and the lexicons they build.

A semantic lexicon assigns logical-form classes and arities to the entries of
a grammar's lexicon, either from a declared joint and assignment or by
resembling the logical forms of a source at a declared size. Building one
against a grammar assembly returns the interior semantic lexicon.
"""

from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol, Self, cast

import didactic.api as dx

from lofi._checks import ensure_finite
from lofi.data.synthetic._resemblance import SemanticLexiconEstimate, semantic_estimate
from lofi.data.synthetic._semantics import (
    LogicalClass,
    SemanticLexicon,
    assign,
    inventory,
)
from lofi.data.synthetic.spec._validation import all_named as _all_named
from lofi.data.synthetic.spec._validation import nonempty as _nonempty
from lofi.lf import LogicFragment
from lofi.spec import DeclarationError, config_search_path
from lofi.spec.grammar import (
    AmbiguityLawSpec,
    ParametricAmbiguityLawSpec,
    PowerLawRankLawSpec,
    RankLawSpec,
    UniformRankLawSpec,
)
from lofi.terms import TermError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from lofi.data.synthetic._ports import SemanticSource
    from lofi.data.synthetic._semantics import AssignmentRow
    from lofi.spec.grammar import GrammarAssembly
    from lofi.spec.language import LanguageAssembly


__all__ = [
    "ArityCountSpec",
    "ArityLawSpec",
    "AritySpec",
    "AssignmentCountSpec",
    "AssignmentRowSpec",
    "AssignmentSpec",
    "ClassWeightSpec",
    "IndependentAssignmentSpec",
    "ParametricSemanticJointSpec",
    "QuantifierMeaningSpec",
    "ResemblingSemanticLexiconSpec",
    "SemanticClassSpec",
    "SemanticFragmentSpec",
    "SemanticJointSpec",
    "SemanticLexiconEstimateSpec",
    "SemanticLexiconStructure",
    "TabularAssignmentSpec",
    "TabularSemanticJointSpec",
    "preset",
]


class QuantifierMeaningSpec(dx.Model, extra="forbid"):
    """One quantifier and the lexical meaning generated from it."""

    name: str
    template: str | None = None

    __axioms__ = (_nonempty("name", "a quantifier meaning needs a name"),)

    @dx.validates("template")
    @classmethod
    def _template_is_nonempty(cls, template: str | None) -> str | None:
        """Return no override or a nonempty logical-form template."""
        if template == "":
            msg = "a quantifier meaning template is empty"
            raise ValueError(msg)
        return template


class SemanticFragmentSpec(dx.Model, extra="forbid"):
    """The logical-form classes a generated semantic lexicon may draw."""

    order: Literal["first", "second", "higher"] | None = None
    quantifiers: tuple[QuantifierMeaningSpec, ...] | None = None
    connectives: tuple[str, ...] | None = None
    functions: bool | None = None
    equality: bool | None = None

    @dx.validates("connectives")
    @classmethod
    def _entries_are_distinct(
        cls, values: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        """Reject a form listed twice in the same class."""
        if values is None:
            return None

        if any(not value for value in values):
            msg = "a semantic fragment form needs a name"
            raise ValueError(msg)

        if len(set(values)) != len(values):
            msg = "a semantic fragment lists a form twice"
            raise ValueError(msg)

        return values

    @dx.validates("quantifiers")
    @classmethod
    def _quantifier_names_are_distinct(
        cls, values: tuple[QuantifierMeaningSpec, ...] | None
    ) -> tuple[QuantifierMeaningSpec, ...] | None:
        """Reject a quantifier listed twice while preserving declaration order."""
        if values is None:
            return None
        names = tuple(value.name for value in values)
        if len(set(names)) != len(names):
            msg = "a semantic fragment lists a quantifier twice"
            raise ValueError(msg)
        return values

    def build(self, language: LanguageAssembly, /) -> LogicFragment:
        """Build the interior fragment used by the semantic sampler."""
        source = language.profile.fragment
        return LogicFragment(
            order=source.order if self.order is None else self.order,
            quantifiers=(
                source.quantifiers
                if self.quantifiers is None
                else tuple(row.name for row in self.quantifiers)
            ),
            connectives=(
                source.connectives if self.connectives is None else self.connectives
            ),
            functions=source.functions if self.functions is None else self.functions,
            equality=source.equality if self.equality is None else self.equality,
        )

    def quantifier_rows(
        self, language: LanguageAssembly, /
    ) -> tuple[tuple[str, str | None], ...]:
        """Return quantifier names and templates in declaration order."""
        if self.quantifiers is None:
            return tuple((name, None) for name in language.profile.fragment.quantifiers)
        return tuple((row.name, row.template) for row in self.quantifiers)


class SemanticClassSpec(dx.Model, extra="forbid"):
    """One logical-form class with the number of forms in it and its share of tokens.

    Attributes
    ----------
    logical_class : LogicalClass
        The class the row sizes.
    size : int
        How many forms the class has.
    share : float
        The class's share of token mass, relative to the other rows.
    """

    logical_class: LogicalClass
    size: int
    share: float

    __axioms__ = (dx.axiom("size > 0", message="a class size must be positive"),)

    @dx.validates("share")
    @classmethod
    def _share_is_positive(cls, share: float) -> float:
        """Return a finite, positive share.

        Raises
        ------
        ValueError
            If the share is infinite, not a number, or not positive.
        """
        held = ensure_finite("a class's share", share)

        if held <= 0.0:
            msg = f"a class's share is not positive: {share!r}"
            raise ValueError(msg)

        return held


class _SemanticJointDeclaration(Protocol):
    """A declaration that supplies logical-form classes and their mass."""

    def classes(self) -> tuple[tuple[LogicalClass, int, float], ...]:
        """Return each class with its size and token share."""
        ...


class _RankLawDeclaration(Protocol):
    """A declaration that assigns relative weights to ranks."""

    def weights(self, count: int) -> tuple[float, ...]:
        """Return weights for the requested rank count."""
        ...


class _AmbiguityLawDeclaration(Protocol):
    """A declaration that assigns ambiguity counts to forms."""

    def counts(self, forms: int) -> tuple[int, ...]:
        """Return one ambiguity count per form."""
        ...


class _AssignmentDeclaration(Protocol):
    """A declaration that assigns logical-form classes to categories."""

    def categories(self) -> tuple[str, ...]:
        """Return the category texts named by the declaration."""
        ...

    def conditional(
        self,
        lexicon: Iterable[str],
        canonical: Mapping[str, str],
        classes: tuple[LogicalClass, ...],
    ) -> Mapping[str, tuple[AssignmentRow, ...]]:
        """Return the assignment rows available to every category."""
        ...


class SemanticJointSpec(dx.TaggedUnion, discriminator="spelling", extra="forbid"):
    """A spelling of the joint distribution of logical-form class and token mass."""


class ParametricSemanticJointSpec(SemanticJointSpec, extra="forbid"):
    """A joint spelled as a class-size law and a coupling exponent.

    Class sizes follow the declared law over the members in declaration
    order, scaled so the heaviest rank holds ``largest_class`` forms and no
    class is empty. A class's share of token mass is proportional to its size
    raised to one minus the coupling, so at a coupling of zero every form has
    the same mean mass whatever its class, a positive coupling gives the forms
    of small classes more mass, and a negative one gives the forms of large
    classes more.

    Attributes
    ----------
    members : tuple[LogicalClass, ...]
        The classes in rank order, the first the largest.
    largest_class : int
        How many forms the first member has.
    class_sizes : RankLawSpec
        The law the sizes fall by along the members.
    coupling : float
        The exponent relating a class's share of token mass to its size.
    """

    spelling: Literal["parametric"] = "parametric"
    members: tuple[LogicalClass, ...]
    largest_class: int
    class_sizes: RankLawSpec
    coupling: float

    __axioms__ = (
        dx.axiom("len(members) > 0", message="a parametric joint needs a class"),
        dx.axiom("largest_class > 0", message="the largest class must be positive"),
    )

    @dx.validates("members")
    @classmethod
    def _members_are_distinct(
        cls, members: tuple[LogicalClass, ...]
    ) -> tuple[LogicalClass, ...]:
        """Return the members when none is listed twice.

        Raises
        ------
        ValueError
            If a class is listed twice.
        """
        if len(set(members)) != len(members):
            msg = f"a parametric joint lists a class twice: {members!r}"
            raise ValueError(msg)

        return members

    @dx.validates("coupling")
    @classmethod
    def _coupling_is_finite(cls, coupling: float) -> float:
        """Return a finite coupling.

        Raises
        ------
        ValueError
            If the coupling is infinite or not a number.
        """
        return ensure_finite("a joint's coupling", coupling)

    def classes(self) -> tuple[tuple[LogicalClass, int, float], ...]:
        """Return each class with its size and token share, in order."""
        weights = cast("_RankLawDeclaration", self.class_sizes).weights(
            len(self.members)
        )
        heaviest = max(weights)
        sizes = tuple(
            max(1, round(self.largest_class * weight / heaviest)) for weight in weights
        )
        masses = tuple(float(size) ** (1.0 - self.coupling) for size in sizes)
        total = sum(masses)

        return tuple(
            (member, size, mass / total)
            for member, size, mass in zip(self.members, sizes, masses, strict=True)
        )


class TabularSemanticJointSpec(SemanticJointSpec, extra="forbid"):
    """A joint spelled as an explicit table of size and share per class.

    The shares are relative: they are scaled to sum to one.

    Attributes
    ----------
    rows : tuple[SemanticClassSpec, ...]
        One row per class.
    """

    spelling: Literal["tabular"] = "tabular"
    rows: tuple[SemanticClassSpec, ...]

    __axioms__ = (dx.axiom("len(rows) > 0", message="a tabular joint needs a row"),)

    @dx.validates("rows")
    @classmethod
    def _classes_are_distinct(
        cls, rows: tuple[SemanticClassSpec, ...]
    ) -> tuple[SemanticClassSpec, ...]:
        """Return the rows when no class has two.

        Raises
        ------
        ValueError
            If a class has two rows.
        """
        classes = [row.logical_class for row in rows]

        if len(set(classes)) != len(classes):
            msg = f"a tabular joint lists a class twice: {classes!r}"
            raise ValueError(msg)

        return rows

    def classes(self) -> tuple[tuple[LogicalClass, int, float], ...]:
        """Return each class with its size and token share, in order."""
        total = sum(row.share for row in self.rows)

        return tuple(
            (row.logical_class, row.size, row.share / total) for row in self.rows
        )


class AritySpec(dx.Model, extra="forbid"):
    """One predicate arity with its share of the predicate forms and its rank law.

    Attributes
    ----------
    arity : int
        How many arguments the predicates take.
    share : float
        The arity's share of the predicate forms, relative to the other rows.
    rank_law : RankLawSpec
        The law token mass falls by along the forms of this arity.
    """

    arity: int
    share: float
    rank_law: RankLawSpec = dx.field(default_factory=UniformRankLawSpec)

    __axioms__ = (dx.axiom("arity > 0", message="an arity must be positive"),)

    @dx.validates("share")
    @classmethod
    def _share_is_positive(cls, share: float) -> float:
        """Return a finite, positive share.

        Raises
        ------
        ValueError
            If the share is infinite, not a number, or not positive.
        """
        held = ensure_finite("an arity's share", share)

        if held <= 0.0:
            msg = f"an arity's share is not positive: {share!r}"
            raise ValueError(msg)

        return held


class ArityLawSpec(dx.Model, extra="forbid"):
    """How the predicate forms split across arities, with a rank law within each.

    Attributes
    ----------
    arities : tuple[AritySpec, ...]
        One row per arity, in the order the forms are numbered.
    """

    arities: tuple[AritySpec, ...]

    __axioms__ = (dx.axiom("len(arities) > 0", message="an arity law needs a row"),)

    @dx.validates("arities")
    @classmethod
    def _arities_are_distinct(
        cls, arities: tuple[AritySpec, ...]
    ) -> tuple[AritySpec, ...]:
        """Return the rows when no arity has two.

        Raises
        ------
        ValueError
            If an arity has two rows.
        """
        held = [row.arity for row in arities]

        if len(set(held)) != len(held):
            msg = f"an arity law lists an arity twice: {held!r}"
            raise ValueError(msg)

        return arities

    def counts(self, size: int) -> tuple[tuple[int, int], ...]:
        """Return how many of a number of predicate forms bear each arity.

        Every row keeps one form, and the rest are apportioned to the rows in
        proportion to their shares by largest remainder, an earlier row
        winning a tie.

        Parameters
        ----------
        size : int
            How many predicate forms to apportion.

        Raises
        ------
        ValueError
            If there are fewer forms than rows.
        """
        rows = self.arities

        if size < len(rows):
            msg = f"{size} predicate forms cannot bear {len(rows)} arities"
            raise ValueError(msg)

        total = sum(row.share for row in rows)
        spare = size - len(rows)
        quotas = [row.share / total * spare for row in rows]
        held = [1 + int(quota) for quota in quotas]
        remainders = sorted(
            range(len(rows)), key=lambda at: (int(quotas[at]) - quotas[at], at)
        )

        for at in remainders[: size - sum(held)]:
            held[at] += 1

        return tuple((row.arity, count) for row, count in zip(rows, held, strict=True))

    def weights(self, size: int) -> tuple[float, ...]:
        """Return one rank weight per predicate form, in arity-row order.

        Parameters
        ----------
        size : int
            How many predicate forms to weigh.
        """
        return tuple(
            weight
            for row, (_, count) in zip(self.arities, self.counts(size), strict=True)
            for weight in cast("_RankLawDeclaration", row.rank_law).weights(count)
        )


class ClassWeightSpec(dx.Model, extra="forbid"):
    """One class a category may draw, with its weight and the arity it forces.

    Attributes
    ----------
    logical_class : LogicalClass
        The class the category may draw.
    arity : int | None
        The arity a drawn predicate or connective bears, or nothing to draw
        without filtering by arity.
    weight : float
        The row's weight, relative to the category's other rows.
    """

    logical_class: LogicalClass
    arity: int | None = None
    weight: float = 1.0

    @dx.model_validator()
    def _arity_goes_with_functional_forms(self) -> Self:
        """Reject arity filters on nonfunctional rows and nonpositive arities.

        Raises
        ------
        ValueError
            If a row other than a predicate or connective row forces an arity,
            or the arity is not positive.
        """
        if self.arity is None:
            return self

        if self.logical_class not in {"predicate", "connective"}:
            msg = (
                "only a predicate or connective row forces an arity, not a "
                f"{self.logical_class} row"
            )
            raise ValueError(msg)

        if self.arity <= 0:
            msg = f"an arity must be positive, not {self.arity}"
            raise ValueError(msg)

        return self

    @dx.validates("weight")
    @classmethod
    def _weight_is_positive(cls, weight: float) -> float:
        """Return a finite, positive weight.

        Raises
        ------
        ValueError
            If the weight is infinite, not a number, or not positive.
        """
        held = ensure_finite("a class weight", weight)

        if held <= 0.0:
            msg = f"a class weight is not positive: {weight!r}"
            raise ValueError(msg)

        return held


class AssignmentRowSpec(dx.Model, extra="forbid"):
    """The classes one category may draw.

    Attributes
    ----------
    category : str
        The category text.
    classes : tuple[ClassWeightSpec, ...]
        The classes the category may draw, each with its weight.
    """

    category: str
    classes: tuple[ClassWeightSpec, ...]

    __axioms__ = (
        _nonempty("category", "an assignment row needs a category"),
        dx.axiom("len(classes) > 0", message="an assignment row needs a class"),
    )


class AssignmentSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """A spelling of the conditional distribution of class given category."""


class IndependentAssignmentSpec(AssignmentSpec, extra="forbid"):
    """A class drawn without regard to the category.

    Every category may draw every class of the joint with the same weight, so
    the class is drawn from the joint's token shares alone.
    """

    kind: Literal["independent"] = "independent"

    def categories(self) -> tuple[str, ...]:
        """Return no category, since every one draws alike."""
        return ()

    def conditional(
        self,
        lexicon: Iterable[str],
        canonical: Mapping[str, str],
        classes: tuple[LogicalClass, ...],
    ) -> Mapping[str, tuple[AssignmentRow, ...]]:
        """Return the same rows, one per class, for every category."""
        del canonical
        shared = tuple((logical_class, None, 1.0) for logical_class in classes)

        return MappingProxyType(dict.fromkeys(lexicon, shared))


class TabularAssignmentSpec(AssignmentSpec, extra="forbid"):
    """A conditional spelled as an explicit table of rows per category.

    Attributes
    ----------
    rows : tuple[AssignmentRowSpec, ...]
        One row per category. A row for a category the lexicon lacks is
        never drawn from.
    """

    kind: Literal["tabular"] = "tabular"
    rows: tuple[AssignmentRowSpec, ...]

    __axioms__ = (
        dx.axiom("len(rows) > 0", message="a tabular assignment needs a row"),
    )

    @dx.validates("rows")
    @classmethod
    def _categories_are_distinct(
        cls, rows: tuple[AssignmentRowSpec, ...]
    ) -> tuple[AssignmentRowSpec, ...]:
        """Return the rows when no category has two.

        Raises
        ------
        ValueError
            If a category has two rows.
        """
        categories = [row.category for row in rows]

        if len(set(categories)) != len(categories):
            msg = f"a tabular assignment lists a category twice: {categories!r}"
            raise ValueError(msg)

        return rows

    def categories(self) -> tuple[str, ...]:
        """Return the category text of each row."""
        return tuple(row.category for row in self.rows)

    def conditional(
        self,
        lexicon: Iterable[str],
        canonical: Mapping[str, str],
        classes: tuple[LogicalClass, ...],
    ) -> Mapping[str, tuple[AssignmentRow, ...]]:
        """Return each lexicon category's declared rows."""
        del classes
        declared = {
            canonical[row.category]: tuple(
                (held.logical_class, held.arity, held.weight) for held in row.classes
            )
            for row in self.rows
        }
        held = tuple(lexicon)
        missing = sorted(set(held) - set(declared))

        if missing:
            msg = f"the assignment has no row for the categories {missing!r}"
            raise DeclarationError(msg)

        return MappingProxyType({category: declared[category] for category in held})


class SemanticLexiconStructure(dx.Model, extra="forbid"):
    """The joint of logical-form class and token mass drawn over a lexicon.

    Each class of the joint has an inventory of forms: the fragment's
    quantifiers or connectives, fresh predicate symbols split across arities
    by the arity law, the one identity, or fresh individual constants. Every
    lexical item of the grammar's lexicon draws a class from the rows its
    category is assigned, each row weighted by its own weight times the
    class's token share, then a form of the class in proportion to the rank
    weights. The lexicon's strings and masses are untouched: the structure
    assigns forms to items and nothing else.

    Attributes
    ----------
    joint : SemanticJointSpec
        The class sizes and token shares.
    rank_law : RankLawSpec
        The law token mass falls by along the forms of a class other than the
        predicate class.
    arities : ArityLawSpec | None
        How the predicate forms split across arities and the rank law within
        each, required exactly when the joint has a predicate class.
    assignment : AssignmentSpec
        The conditional distribution of class given category.
    fragment : SemanticFragmentSpec
        The logic the forms are drawn within, which bounds the quantifier and
        connective classes.
    seed : int
        The seed of the draw. Equal seeds over equal lexicons draw equal
        assignments.
    constants : int | None
        The total inventory size, or nothing to use the joint's row sizes.
    polysemy : AmbiguityLawSpec
        The law over meanings per form-category item and their frequency placement.
    """

    joint: SemanticJointSpec
    rank_law: RankLawSpec = dx.field(default_factory=UniformRankLawSpec)
    arities: ArityLawSpec | None = None
    assignment: AssignmentSpec
    fragment: SemanticFragmentSpec = dx.field(default_factory=SemanticFragmentSpec)
    seed: int
    constants: int | None = None
    polysemy: AmbiguityLawSpec = dx.field(
        default_factory=lambda: ParametricAmbiguityLawSpec(
            mean=1.0, dispersion=0.0, maximum=1, placement=0.0
        )
    )

    @dx.validates("constants")
    @classmethod
    def _constant_count_is_positive(cls, count: int | None) -> int | None:
        """Return no count or a positive total inventory size."""
        if count is not None and count <= 0:
            msg = "a semantic constant inventory must be positive"
            raise ValueError(msg)

        return count

    @dx.model_validator()
    def _arities_go_with_predicates(self) -> Self:
        """Reject an arity law without a predicate class, or the reverse.

        Raises
        ------
        ValueError
            If the joint has a predicate class and no arity law, or an arity
            law and no predicate class.
        """
        predicates = any(
            logical_class == "predicate"
            for logical_class, _, _ in cast(
                "_SemanticJointDeclaration", self.joint
            ).classes()
        )

        if predicates and self.arities is None:
            msg = "a joint with a predicate class needs an arity law"
            raise ValueError(msg)

        if self.arities is not None and not predicates:
            msg = "an arity law needs a predicate class in the joint"
            raise ValueError(msg)

        return self

    def build(self, assembly: GrammarAssembly, /) -> SemanticLexicon:
        """Draw the assignment of forms to the lexical items of a compiled grammar.

        Parameters
        ----------
        assembly : GrammarAssembly
            The compiled grammar whose lexicon's items draw forms.

        Raises
        ------
        DeclarationError
            If an assignment row names a category the grammar's syntax does
            not read, a closed class is larger than the fragment's inventory,
            a standard quantifier restricts by a connective the fragment
            lacks, the identity class is sized at more than one form, the
            arity law cannot apportion the predicate forms, a category of the
            lexicon has no assignment row, a row names a class the joint does
            not declare, or a row forces an arity no predicate form bears.
        """
        fragment = self.fragment.build(assembly.language)
        classes = self._classes()
        store = assembly.store
        syntax = assembly.syntax
        lexicon = assembly.lexicon
        render = {
            entry.category: syntax.render(store, entry.category)
            for form in lexicon.forms
            for entry in lexicon.entries(form)
        }
        declared = tuple(logical_class for logical_class, _, _ in classes)

        try:
            canonical = {
                text: syntax.render(store, syntax.parse(store, text))
                for text in cast("_AssignmentDeclaration", self.assignment).categories()
            }
        except TermError as error:
            msg = (
                f"an assignment row names a category the grammar does not read: {error}"
            )
            raise DeclarationError(msg) from error

        rows = cast("_AssignmentDeclaration", self.assignment).conditional(
            dict.fromkeys(render.values()), canonical, declared
        )

        for category, held in rows.items():
            for logical_class, _, _ in held:
                if logical_class not in declared:
                    msg = (
                        f"category {category!r} is assigned the {logical_class} "
                        f"class, which the joint does not declare"
                    )
                    raise DeclarationError(msg)

        try:
            forms = inventory(
                tuple(
                    (logical_class, size, self._weights(logical_class, size))
                    for logical_class, size, _ in classes
                ),
                language=assembly.language,
                quantifiers=self.fragment.quantifier_rows(assembly.language),
                connectives=fragment.connectives,
                arities=self._arity_counts(classes),
            )

            return assign(
                lexicon,
                render,
                forms,
                {logical_class: share for logical_class, _, share in classes},
                rows,
                seed=self.seed,
                meanings_per_item=cast(
                    "_AmbiguityLawDeclaration", self.polysemy
                ).counts(sum(len(lexicon.entries(form)) for form in lexicon.forms)),
                placement=self.polysemy.placement,
            )
        except ValueError as error:
            msg = f"the semantic lexicon structure cannot be drawn: {error}"
            raise DeclarationError(msg) from error

    def _weights(self, logical_class: LogicalClass, size: int) -> tuple[float, ...]:
        """Return the rank weights of a class's forms."""
        if self.arities is not None and logical_class == "predicate":
            return self.arities.weights(size)

        return cast("_RankLawDeclaration", self.rank_law).weights(size)

    def _classes(self) -> tuple[tuple[LogicalClass, int, float], ...]:
        """Return the joint's classes scaled to the declared total, if any."""
        classes = cast("_SemanticJointDeclaration", self.joint).classes()
        total = self.constants

        if total is None:
            return classes

        scalable = tuple(
            at
            for at, (logical_class, _, _) in enumerate(classes)
            if logical_class in {"predicate", "constant"}
        )
        fixed = tuple(at for at in range(len(classes)) if at not in scalable)
        reserved = sum(classes[at][1] for at in fixed)
        required = reserved + len(scalable)

        if total < required:
            msg = (
                f"an inventory of {total} constants cannot keep the {reserved} "
                f"closed-class forms and {len(scalable)} open classes"
            )
            raise DeclarationError(msg)

        if not scalable:
            if total != reserved:
                msg = (
                    f"an inventory of {total} constants cannot scale a joint "
                    "with no open class"
                )
                raise DeclarationError(msg)

            return classes

        available = total - reserved
        weight = sum(classes[at][1] for at in scalable)
        quotas = tuple(classes[at][1] / weight * available for at in scalable)
        scaled = [max(1, int(quota)) for quota in quotas]
        remainders = sorted(
            range(len(scalable)),
            key=lambda at: (int(quotas[at]) - quotas[at], at),
        )

        for at in remainders[: max(0, available - sum(scaled))]:
            scaled[at] += 1

        while sum(scaled) > available:
            at = max(
                (index for index, size in enumerate(scaled) if size > 1),
                key=lambda index: (scaled[index], -index),
            )
            scaled[at] -= 1

        sizes = [size for _, size, _ in classes]

        for at, size in zip(scalable, scaled, strict=True):
            sizes[at] = size

        return tuple(
            (logical_class, size, share)
            for (logical_class, _, share), size in zip(classes, sizes, strict=True)
        )

    def _arity_counts(
        self, classes: tuple[tuple[LogicalClass, int, float], ...]
    ) -> tuple[tuple[int, int], ...]:
        """Return how many predicate forms bear each arity, none without predicates."""
        if self.arities is None:
            return ()

        return self.arities.counts(
            next(
                size
                for logical_class, size, _ in classes
                if logical_class == "predicate"
            )
        )


class ArityCountSpec(dx.Model, extra="forbid"):
    """One predicate arity with how many of a source's predicate constants bear it.

    Attributes
    ----------
    arity : int
        How many arguments the predicates take.
    constants : int
        How many of the source's predicate constants bear the arity.
    """

    arity: int
    constants: int

    __axioms__ = (
        dx.axiom("arity > 0", message="an arity must be positive"),
        dx.axiom("constants > 0", message="an arity needs a predicate constant"),
    )


class AssignmentCountSpec(dx.Model, extra="forbid"):
    """One class the items of a category bore in a source, with how many bore it.

    Attributes
    ----------
    category : str
        The category text.
    logical_class : LogicalClass
        The class the items bore.
    items : int
        How many of the source's items of the category bore the class.
    arity : int | None
        The arity the borne predicates take, or nothing for another class.
    """

    category: str
    logical_class: LogicalClass
    items: int
    arity: int | None = None

    __axioms__ = (
        _nonempty("category", "an assigned category needs a name"),
        dx.axiom("items > 0", message="an assigned class needs an item"),
    )

    @dx.model_validator()
    def _arity_goes_with_predicates(self) -> Self:
        """Reject an arity on a row of another class, or one that is not positive.

        Raises
        ------
        ValueError
            If a row other than a predicate row records an arity, a predicate
            row records none, or the arity is not positive.
        """
        if (self.arity is None) == (self.logical_class == "predicate"):
            msg = (
                f"only a predicate row records an arity, but a "
                f"{self.logical_class} row records {self.arity!r}"
            )
            raise ValueError(msg)

        if self.arity is not None and self.arity <= 0:
            msg = f"an arity must be positive, not {self.arity}"
            raise ValueError(msg)

        return self


class SemanticLexiconEstimateSpec(dx.Model, extra="forbid"):
    """The logical forms a source lexicon was measured to carry, beside its digest.

    Attributes
    ----------
    digest : str
        The digest of the source the forms were read from. Two declarations
        seeded from different sources differ here, and so in their bytes.
    exponent : float
        The exponent fitted to the rank-frequency curve of the source's
        logical constants.
    rows : tuple[SemanticClassSpec, ...]
        Each class with how many of the source's logical constants belong to
        it and the fraction of the source's token mass its items carry.
    assignment : tuple[AssignmentCountSpec, ...]
        Each category with the classes its items bore and how many bore each.
    arities : tuple[ArityCountSpec, ...]
        Each arity the source's predicates take, with how many predicate
        constants bear it.
    quantifiers : tuple[str, ...]
        The quantifiers the source's quantifier forms are built on.
    connectives : tuple[str, ...]
        The connectives the source's connective forms are built on.
    """

    digest: str
    exponent: float
    rows: tuple[SemanticClassSpec, ...]
    assignment: tuple[AssignmentCountSpec, ...]
    arities: tuple[ArityCountSpec, ...] = ()
    quantifiers: tuple[str, ...] = ()
    connectives: tuple[str, ...] = ()

    __axioms__ = (
        _nonempty("digest", "an estimate needs the digest of its source"),
        dx.axiom("len(rows) > 0", message="an estimate needs a class"),
        dx.axiom("len(assignment) > 0", message="an estimate needs an assigned item"),
        _all_named("quantifiers", "a quantifier needs a name"),
        _all_named("connectives", "a connective needs a name"),
    )

    @dx.validates("exponent")
    @classmethod
    def _exponent_is_finite(cls, exponent: float) -> float:
        """Return a finite exponent.

        Parameters
        ----------
        exponent : float
            The measured exponent.

        Raises
        ------
        ValueError
            If the exponent is infinite or not a number.
        """
        return ensure_finite("an estimated exponent", exponent)

    @classmethod
    def measured(cls, source: SemanticSource, digest: str, /) -> Self:
        """Return the estimate read from a source lexicon carrying logical forms.

        Parameters
        ----------
        source : SemanticSource
            The lexicon to read.
        digest : str
            The digest of the source.

        Returns
        -------
        Self
            The declaration of the logical forms the source was measured to
            carry.

        Raises
        ------
        DeclarationError
            If the source holds no item, holds an item bearing no logical
            form or no token mass, or holds too few logical constants for a
            rank-frequency curve to be fitted.
        """
        try:
            measurement = semantic_estimate(source, digest)
        except ValueError as error:
            msg = f"the source's logical forms cannot be estimated: {error}"
            raise DeclarationError(msg) from error

        return cls(
            digest=measurement.digest,
            exponent=measurement.exponent,
            rows=tuple(
                SemanticClassSpec(logical_class=logical_class, size=size, share=share)
                for logical_class, size, share in measurement.rows
            ),
            assignment=tuple(
                AssignmentCountSpec(
                    category=category,
                    logical_class=logical_class,
                    items=items,
                    arity=arity,
                )
                for category, logical_class, arity, items in measurement.assignment
            ),
            arities=tuple(
                ArityCountSpec(arity=arity, constants=constants)
                for arity, constants in measurement.arities
            ),
            quantifiers=measurement.quantifiers,
            connectives=measurement.connectives,
        )

    def build(self) -> SemanticLexiconEstimate:
        """Return the interior estimate.

        Raises
        ------
        ValueError
            If the declared rows, arities and assignment do not read as one
            joint over the source's logical constants.
        """
        return SemanticLexiconEstimate(
            self.digest,
            tuple((row.logical_class, row.size, row.share) for row in self.rows),
            tuple((row.arity, row.constants) for row in self.arities),
            tuple(
                (row.category, row.logical_class, row.arity, row.items)
                for row in self.assignment
            ),
            quantifiers=self.quantifiers,
            connectives=self.connectives,
            exponent=self.exponent,
        )


class ResemblingSemanticLexiconSpec(dx.Model, extra="forbid"):
    """A semantic lexicon drawn at a declared size from a source's logical forms.

    The classes, their token shares, the arity law and the conditional
    distribution of class given category are read from the source rather
    than declared. Below the source's inventory each class is subsampled in
    proportion, so the constants drawn are a subset of the source's; above it
    the predicate and individual classes are extrapolated, while a closed
    class is held at the constants the source names. A row's weight is the
    count of items divided by its class's token share, since a draw weighs a
    row by its weight times that share, so the quotient is what makes the
    drawn conditional the source's.

    A source whose quantifiers restrict by a connective it never uses cannot
    be resembled, since the drawn fragment declares the source's connectives
    and no others.

    Attributes
    ----------
    constants : int
        How many logical constants the drawn inventory holds.
    estimate : SemanticLexiconEstimateSpec
        The logical forms the source was measured to carry, with its digest.
    seed : int
        The seed of the draw. Equal seeds over equal lexicons draw equal
        assignments.
    order : Literal["first", "second", "higher"]
        The order of the fragment the forms are drawn within, which a source
        using a generalized quantifier needs above the first.
    """

    constants: int
    estimate: SemanticLexiconEstimateSpec
    seed: int
    order: Literal["first", "second", "higher"] = "first"

    __axioms__ = (
        dx.axiom("constants > 0", message="a constant inventory must be positive"),
    )

    def structure(self) -> SemanticLexiconStructure:
        """Return the structured declaration the drawn semantic lexicon comes from.

        Returns
        -------
        SemanticLexiconStructure
            The estimate's joint scaled to the declared inventory, spelled as
            a table, under the rank law the source's exponent fits, over the
            fragment the source's logical constants name.

        Raises
        ------
        DeclarationError
            If the estimate does not read as one joint, the inventory is
            larger than a source of closed classes alone can fill, or it
            drops every class some category of the source bears.
        """
        try:
            scaled = self.estimate.build().at(self.constants)
        except ValueError as error:
            msg = f"the source's logical forms cannot be resembled: {error}"
            raise DeclarationError(msg) from error

        shares = {logical_class: share for logical_class, _, share in scaled}
        drawn: dict[str, list[ClassWeightSpec]] = {}

        for row in self.estimate.assignment:
            share = shares.get(row.logical_class)

            if share is not None:
                drawn.setdefault(row.category, []).append(
                    ClassWeightSpec(
                        logical_class=row.logical_class,
                        arity=row.arity,
                        weight=row.items / share,
                    )
                )

        dropped = sorted(
            {row.category for row in self.estimate.assignment} - set(drawn)
        )

        if dropped:
            msg = (
                f"an inventory of {self.constants} constants drops every class "
                f"the categories {dropped!r} bear"
            )
            raise DeclarationError(msg)

        return SemanticLexiconStructure(
            joint=TabularSemanticJointSpec(
                rows=tuple(
                    SemanticClassSpec(
                        logical_class=logical_class, size=size, share=share
                    )
                    for logical_class, size, share in scaled
                )
            ),
            rank_law=PowerLawRankLawSpec(exponent=self.estimate.exponent),
            arities=self._arity_law() if "predicate" in shares else None,
            assignment=TabularAssignmentSpec(
                rows=tuple(
                    AssignmentRowSpec(category=category, classes=tuple(held))
                    for category, held in drawn.items()
                )
            ),
            fragment=SemanticFragmentSpec(
                order=self.order,
                quantifiers=tuple(
                    QuantifierMeaningSpec(name=name)
                    for name in self.estimate.quantifiers
                ),
                connectives=self.estimate.connectives,
            ),
            seed=self.seed,
        )

    def build(self, assembly: GrammarAssembly, /) -> SemanticLexicon:
        """Draw the assignment of the scaled inventory to a compiled grammar's items.

        Parameters
        ----------
        assembly : GrammarAssembly
            The compiled grammar whose lexicon's items draw forms.

        Returns
        -------
        SemanticLexicon
            The drawn semantic lexicon.

        Raises
        ------
        DeclarationError
            If the scaled joint cannot be drawn, or a category of the
            grammar's lexicon is one the source never held.
        """
        return self.structure().build(assembly)

    def _arity_law(self) -> ArityLawSpec:
        """Return the law splitting the drawn predicates across the source's arities."""
        return ArityLawSpec(
            arities=tuple(
                AritySpec(
                    arity=row.arity,
                    share=float(row.constants),
                    rank_law=PowerLawRankLawSpec(exponent=self.estimate.exponent),
                )
                for row in self.estimate.arities
            )
        )


def preset(name: str, /) -> SemanticLexiconStructure:
    """Return a semantic-lexicon preset from the configured search path.

    Parameters
    ----------
    name : str
        The preset's file name without its extension, under the
        ``semantic_lexicons`` group of a project, package or installed
        ``lofi.config_dirs`` provider.

    Raises
    ------
    FileNotFoundError
        If no preset has the name.
    """
    if Path(name).name != name or name in {"", ".", ".."}:
        msg = f"a semantic-lexicon preset needs a file name, not {name!r}"
        raise ValueError(msg)
    relative = Path("semantic_lexicons") / f"{name}.json"
    roots = config_search_path(Path.cwd())
    for root in roots:
        document = root / relative
        if document.is_file():
            return SemanticLexiconStructure.model_validate_json(
                document.read_text(encoding="utf-8")
            )
    tried = tuple(root / relative for root in roots)
    msg = f"no semantic-lexicon preset {name!r}; tried {tried!r}"
    raise FileNotFoundError(msg)
