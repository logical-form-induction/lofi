"""Neutral declarations for a model-owned learner interpretation."""

from typing import Literal, Protocol, Self, cast

import didactic.api as dx

from lofi._checks import ensure_finite
from lofi.interpret import InhabitantSelection
from lofi.judge import PROOF_LOGICS, ProofLogic
from lofi.spec.countermodels import CountermodelSpec
from lofi.spec.grammar import (
    AmbiguityLawSpec,
    ErasureSpec,
    ParametricAmbiguityLawSpec,
    RankLawSpec,
    UniformRankLawSpec,
)
from lofi.spec.language import ConstantSpec
from lofi.spec.normalization import NormalizationSpec

__all__ = [
    "LearnerInterpretationSpec",
    "LearnerMeaningSourceSpec",
    "LearnerMeaningSpec",
    "LearnerProofSearchSpec",
    "LearnerResemblingMeaningSourceSpec",
    "LexicalLearnerMeaningSourceSpec",
    "StructuredLearnerMeaningSourceSpec",
]


class LearnerQuantifierSpec(dx.Model, extra="forbid"):
    """Declare one quantifier available to generated learner meanings."""

    name: str
    template: str | None = None

    @dx.model_validator()
    def _named(self) -> Self:
        """Require a named quantifier and a nonempty supplied template."""
        if not self.name:
            msg = "a quantifier meaning needs a name"
            raise ValueError(msg)
        if self.template == "":
            msg = "a quantifier meaning template is empty"
            raise ValueError(msg)
        return self


class LearnerSemanticFragmentSpec(dx.Model, extra="forbid"):
    """Declare the logical fragment generated learner meanings inhabit."""

    order: Literal["first", "second", "higher"] | None = None
    quantifiers: tuple[LearnerQuantifierSpec, ...] | None = None
    connectives: tuple[str, ...] | None = None
    functions: bool | None = None
    equality: bool | None = None

    @dx.model_validator()
    def _distinct_named_forms(self) -> Self:
        """Require named, unrepeated quantifiers and connectives."""
        quantifiers = () if self.quantifiers is None else self.quantifiers
        names = tuple(row.name for row in quantifiers)
        connectives = () if self.connectives is None else self.connectives
        if len(set(names)) != len(names):
            msg = "a semantic fragment lists a quantifier twice"
            raise ValueError(msg)
        if any(not name for name in connectives):
            msg = "a semantic fragment form needs a name"
            raise ValueError(msg)
        if len(set(connectives)) != len(connectives):
            msg = "a semantic fragment lists a form twice"
            raise ValueError(msg)
        return self


class LearnerSemanticClassSpec(dx.Model, extra="forbid"):
    """Declare one generated logical form class and its mass."""

    logical_class: str
    size: int
    share: float

    __axioms__ = (dx.axiom("size > 0", message="a class size must be positive"),)

    @dx.validates("share")
    @classmethod
    def _share_is_positive(cls, share: float) -> float:
        """Return a finite, positive class share."""
        held = ensure_finite("a class's share", share)
        if held <= 0.0:
            msg = f"a class's share is not positive: {share!r}"
            raise ValueError(msg)
        return held


class _LearnerSemanticJointDeclaration(Protocol):
    """A semantic joint that reports its declared classes."""

    def logical_classes(self) -> tuple[str, ...]:
        """Return the classes named by the joint."""
        ...


class LearnerSemanticJointSpec(
    dx.TaggedUnion, discriminator="spelling", extra="forbid"
):
    """Declare the joint distribution of logical form class and token mass."""


class LearnerParametricSemanticJointSpec(LearnerSemanticJointSpec, extra="forbid"):
    """Declare a rank-parametric semantic joint."""

    spelling: Literal["parametric"] = "parametric"
    members: tuple[str, ...]
    largest_class: int
    class_sizes: RankLawSpec
    coupling: float

    __axioms__ = (
        dx.axiom("len(members) > 0", message="a parametric joint needs a class"),
        dx.axiom("largest_class > 0", message="the largest class must be positive"),
    )

    @dx.model_validator()
    def _valid_joint(self) -> Self:
        """Require distinct members and a finite coupling."""
        if len(set(self.members)) != len(self.members):
            msg = f"a parametric joint lists a class twice: {self.members!r}"
            raise ValueError(msg)
        ensure_finite("a joint's coupling", self.coupling)
        return self

    def logical_classes(self) -> tuple[str, ...]:
        """Return the class members in declaration order."""
        return self.members


class LearnerTabularSemanticJointSpec(LearnerSemanticJointSpec, extra="forbid"):
    """Declare a semantic joint as an explicit table."""

    spelling: Literal["tabular"] = "tabular"
    rows: tuple[LearnerSemanticClassSpec, ...]

    __axioms__ = (dx.axiom("len(rows) > 0", message="a tabular joint needs a row"),)

    @dx.model_validator()
    def _distinct_classes(self) -> Self:
        """Require one row per class."""
        classes = self.logical_classes()
        if len(set(classes)) != len(classes):
            msg = f"a tabular joint lists a class twice: {classes!r}"
            raise ValueError(msg)
        return self

    def logical_classes(self) -> tuple[str, ...]:
        """Return the table's classes in row order."""
        return tuple(row.logical_class for row in self.rows)


class LearnerAritySpec(dx.Model, extra="forbid"):
    """Declare one generated predicate arity and its mass."""

    arity: int
    share: float
    rank_law: RankLawSpec = dx.field(default_factory=UniformRankLawSpec)

    __axioms__ = (dx.axiom("arity > 0", message="an arity must be positive"),)

    @dx.validates("share")
    @classmethod
    def _share_is_positive(cls, share: float) -> float:
        """Return a finite, positive arity share."""
        held = ensure_finite("an arity's share", share)
        if held <= 0.0:
            msg = f"an arity's share is not positive: {share!r}"
            raise ValueError(msg)
        return held


class LearnerArityLawSpec(dx.Model, extra="forbid"):
    """Declare the generated predicate-arity distribution."""

    arities: tuple[LearnerAritySpec, ...]

    __axioms__ = (dx.axiom("len(arities) > 0", message="an arity law needs a row"),)

    @dx.model_validator()
    def _distinct_arities(self) -> Self:
        """Require one row per arity."""
        arities = tuple(row.arity for row in self.arities)
        if len(set(arities)) != len(arities):
            msg = f"an arity law lists an arity twice: {arities!r}"
            raise ValueError(msg)
        return self


class LearnerClassWeightSpec(dx.Model, extra="forbid"):
    """Declare one logical form class available to a category."""

    logical_class: str
    arity: int | None = None
    weight: float = 1.0

    @dx.model_validator()
    def _arity_goes_with_functional_forms(self) -> Self:
        """Require positive arities only on filterable functional-form rows."""
        filterable = {"predicate", "connective"}
        if self.arity is not None and self.logical_class not in filterable:
            msg = (
                "only a predicate or connective row forces an arity, not "
                f"{self.logical_class!r}"
            )
            raise ValueError(msg)
        if self.arity is not None and self.arity <= 0:
            msg = f"an arity must be positive, not {self.arity}"
            raise ValueError(msg)
        held = ensure_finite("a class weight", self.weight)
        if held <= 0.0:
            msg = f"a class weight is not positive: {self.weight!r}"
            raise ValueError(msg)
        return self


class LearnerAssignmentRowSpec(dx.Model, extra="forbid"):
    """Declare the logical form classes available to one category."""

    category: str
    classes: tuple[LearnerClassWeightSpec, ...]

    @dx.model_validator()
    def _nonempty_row(self) -> Self:
        """Require a named category with at least one class."""
        if not self.category:
            msg = "an assignment row needs a category"
            raise ValueError(msg)
        if not self.classes:
            msg = "an assignment row needs a class"
            raise ValueError(msg)
        return self


class LearnerAssignmentSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare how generated logical form classes depend on categories."""


class LearnerIndependentAssignmentSpec(LearnerAssignmentSpec, extra="forbid"):
    """Draw a logical form class independently of category."""

    kind: Literal["independent"] = "independent"


class LearnerTabularAssignmentSpec(LearnerAssignmentSpec, extra="forbid"):
    """Declare class availability by category."""

    kind: Literal["tabular"] = "tabular"
    rows: tuple[LearnerAssignmentRowSpec, ...]

    @dx.model_validator()
    def _distinct_categories(self) -> Self:
        """Require one row per category."""
        categories = tuple(row.category for row in self.rows)
        if not categories:
            msg = "a tabular assignment needs a row"
            raise ValueError(msg)
        if len(set(categories)) != len(categories):
            msg = f"a tabular assignment lists a category twice: {categories!r}"
            raise ValueError(msg)
        return self


class LearnerSemanticLexiconStructureSpec(dx.Model, extra="forbid"):
    """Declare one generated semantic lexicon without importing data code."""

    joint: LearnerSemanticJointSpec
    rank_law: RankLawSpec = dx.field(default_factory=UniformRankLawSpec)
    arities: LearnerArityLawSpec | None = None
    assignment: LearnerAssignmentSpec
    fragment: LearnerSemanticFragmentSpec = dx.field(
        default_factory=LearnerSemanticFragmentSpec
    )
    seed: int
    constants: int | None = None
    polysemy: AmbiguityLawSpec = dx.field(
        default_factory=lambda: ParametricAmbiguityLawSpec(
            mean=1.0, dispersion=0.0, maximum=1, placement=0.0
        )
    )

    @dx.model_validator()
    def _valid_inventory(self) -> Self:
        """Require positive inventory size and arities exactly for predicates."""
        if self.constants is not None and self.constants <= 0:
            msg = "a semantic constant inventory must be positive"
            raise ValueError(msg)
        classes = cast("_LearnerSemanticJointDeclaration", self.joint).logical_classes()
        predicates = "predicate" in classes
        if predicates and self.arities is None:
            msg = "a joint with a predicate class needs an arity law"
            raise ValueError(msg)
        if not predicates and self.arities is not None:
            msg = "an arity law needs a predicate class in the joint"
            raise ValueError(msg)
        return self


class LearnerArityCountSpec(dx.Model, extra="forbid"):
    """Record one measured predicate arity count."""

    arity: int
    constants: int

    __axioms__ = (
        dx.axiom("arity > 0", message="an arity must be positive"),
        dx.axiom("constants > 0", message="an arity needs a predicate constant"),
    )


class LearnerAssignmentCountSpec(dx.Model, extra="forbid"):
    """Record one measured category-to-class assignment count."""

    category: str
    logical_class: str
    items: int
    arity: int | None = None

    @dx.model_validator()
    def _valid_count(self) -> Self:
        """Require a named category, positive count, and predicate arity."""
        if not self.category:
            msg = "an assigned category needs a name"
            raise ValueError(msg)
        if self.items <= 0:
            msg = "an assigned class needs an item"
            raise ValueError(msg)
        if (self.arity is None) == (self.logical_class == "predicate"):
            msg = "only a predicate assignment records an arity"
            raise ValueError(msg)
        if self.arity is not None and self.arity <= 0:
            msg = f"an arity must be positive, not {self.arity}"
            raise ValueError(msg)
        return self


class LearnerSemanticLexiconEstimateSpec(dx.Model, extra="forbid"):
    """Declare the sufficient statistics of a measured semantic lexicon."""

    digest: str
    exponent: float
    rows: tuple[LearnerSemanticClassSpec, ...]
    assignment: tuple[LearnerAssignmentCountSpec, ...]
    arities: tuple[LearnerArityCountSpec, ...] = ()
    quantifiers: tuple[str, ...] = ()
    connectives: tuple[str, ...] = ()

    @dx.model_validator()
    def _valid_estimate(self) -> Self:
        """Require named source statistics and a finite fitted exponent."""
        if not self.digest:
            msg = "an estimate needs the digest of its source"
            raise ValueError(msg)
        if not self.rows:
            msg = "an estimate needs a class"
            raise ValueError(msg)
        if not self.assignment:
            msg = "an estimate needs an assigned item"
            raise ValueError(msg)
        if any(not name for name in (*self.quantifiers, *self.connectives)):
            msg = "an estimated logical form needs a name"
            raise ValueError(msg)
        ensure_finite("an estimated exponent", self.exponent)
        return self


class LearnerResemblingSemanticLexiconSpec(dx.Model, extra="forbid"):
    """Declare a semantic lexicon drawn to resemble measured statistics."""

    constants: int
    estimate: LearnerSemanticLexiconEstimateSpec
    seed: int
    order: Literal["first", "second", "higher"] = "first"

    __axioms__ = (
        dx.axiom("constants > 0", message="a constant inventory must be positive"),
    )


class LearnerMeaningSourceSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare where a learner interpretation obtains lexical meanings."""


class LexicalLearnerMeaningSourceSpec(LearnerMeaningSourceSpec, extra="forbid"):
    """Read meanings declared on the learner grammar's lexical entries."""

    kind: Literal["lexical"] = "lexical"


class StructuredLearnerMeaningSourceSpec(LearnerMeaningSourceSpec, extra="forbid"):
    """Read an inline or packaged semantic lexicon structure."""

    kind: Literal["structured"] = "structured"
    structure: LearnerSemanticLexiconStructureSpec | None = None
    preset: str | None = None

    @dx.validates("preset")
    @classmethod
    def _preset_is_named(cls, name: str | None) -> str | None:
        """Return a nonempty preset name or no preset."""
        if name == "":
            msg = "a learner semantic-lexicon preset needs a name"
            raise ValueError(msg)
        return name

    @dx.model_validator()
    def _one_structure_spelling(self) -> Self:
        """Require exactly one inline structure or packaged preset."""
        if (self.structure is None) == (self.preset is None):
            msg = "structured learner meanings need one structure or preset"
            raise ValueError(msg)
        return self


class LearnerResemblingMeaningSourceSpec(LearnerMeaningSourceSpec, extra="forbid"):
    """Draw meanings resembling a measured semantic lexicon."""

    kind: Literal["resembling"] = "resembling"
    source: LearnerResemblingSemanticLexiconSpec


class LearnerMeaningSpec(dx.Model, extra="forbid"):
    """Declare learner lexical meanings and proof-inhabitant selection."""

    source: LearnerMeaningSourceSpec
    inhabitant_selection: InhabitantSelection = "unique"
    search_budget: int = 64

    __axioms__ = (
        dx.axiom(
            "search_budget > 0", message="a meaning search budget must be positive"
        ),
    )


class LearnerProofSearchSpec(dx.Model, extra="forbid"):
    """Declare learner structural proof search."""

    logic: ProofLogic = "relevant"
    max_uses: int = 1

    __axioms__ = (dx.axiom("max_uses > 0", message="max_uses must be positive"),)

    @dx.model_validator()
    def _structural_use_bound(self) -> Self:
        """Require substructural logics to use each assumption at most once."""
        if not PROOF_LOGICS[self.logic].contraction and self.max_uses != 1:
            msg = "linear and affine proof search use each assumption at most once"
            raise ValueError(msg)
        return self


class LearnerInterpretationSpec(dx.Model, extra="forbid"):
    """Declare an interpretation of an inline learner grammar."""

    entity: str
    truth: str
    type_syntax: Literal["angle", "s-expression"] = "angle"
    erasure: ErasureSpec
    constants: tuple[ConstantSpec, ...]
    meanings: LearnerMeaningSpec
    normalization: NormalizationSpec = dx.field(default_factory=NormalizationSpec)
    countermodels: CountermodelSpec | None = None
    actions: dict[str, str] = dx.field(default_factory=dict)
    proof_search: LearnerProofSearchSpec = dx.field(
        default_factory=LearnerProofSearchSpec
    )

    @dx.model_validator()
    def _distinct_semantic_types(self) -> Self:
        """Require named, distinct entity and truth types."""
        if not self.entity:
            msg = "an interpretation needs an entity type"
            raise ValueError(msg)
        if not self.truth:
            msg = "an interpretation needs a truth type"
            raise ValueError(msg)
        if self.entity == self.truth:
            msg = "entity and truth are two types"
            raise ValueError(msg)
        return self
