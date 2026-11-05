"""Values an ill-typed logical form is built from and certified with.

An injected form is a corruption of a well-typed one that the declared type
checker refuses. The refusal itself is the certificate: the constraints the
checker left unsatisfied, written in the semantic-type syntax, beside the
digest of the language declaration they were left under.
"""

from typing import TYPE_CHECKING, Final, Literal, override

from lofi._checks import ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from lofi._types import ItemId
    from lofi.terms import Path
    from lofi.typecheck import TypingFailureKind

__all__ = [
    "FORM_TYPING_LABELS",
    "FormTypingExample",
    "FormTypingLabel",
    "IllTyped",
    "TypingConflict",
    "TypingFailure",
]

type FormTypingLabel = Literal["well-typed", "ill-typed"]
"""Whether a logical form is presented as satisfying its type system."""

FORM_TYPING_LABELS: Final[tuple[FormTypingLabel, ...]] = (
    "well-typed",
    "ill-typed",
)
"""The logical-form typing labels in report order."""


class FormTypingExample(FrozenValue):
    """One logical form with its certified and observed typing labels.

    Parameters
    ----------
    item_id : ItemId
        The surface item the logical form belongs to.
    text : str
        The logical form in the declared surface syntax.
    true_label : FormTypingLabel
        Whether the declared type checker accepts the form.
    observed_label : FormTypingLabel
        The label presented to training after applying the negative-data policy.

    Raises
    ------
    ValueError
        If the form is empty or either label is unknown.
    """

    _noun = "logical-form typing example"

    __slots__ = ("_item_id", "_observed_label", "_text", "_true_label")

    def __init__(
        self,
        item_id: ItemId,
        text: str,
        true_label: FormTypingLabel,
        observed_label: FormTypingLabel,
    ) -> None:
        for field, label in (
            ("true", true_label),
            ("observed", observed_label),
        ):
            if label not in FORM_TYPING_LABELS:
                msg = (
                    f"the {field} label of the logical form for {item_id!r} "
                    f"is unknown: {label!r}"
                )
                raise ValueError(msg)

        object.__setattr__(self, "_item_id", item_id)
        object.__setattr__(
            self, "_text", ensure_name(f"the logical form of {item_id!r}", text)
        )
        object.__setattr__(self, "_true_label", true_label)
        object.__setattr__(self, "_observed_label", observed_label)

    @property
    def item_id(self) -> ItemId:
        """Return the surface item the form belongs to."""
        return self._item_id

    @property
    def text(self) -> str:
        """Return the logical form in the declared surface syntax."""
        return self._text

    @property
    def true_label(self) -> FormTypingLabel:
        """Return whether the declared type checker accepts the form."""
        return self._true_label

    @property
    def observed_label(self) -> FormTypingLabel:
        """Return the label presented to training."""
        return self._observed_label

    @property
    def flipped(self) -> bool:
        """Return whether the observed label differs from the certified one."""
        return self._observed_label != self._true_label

    def _fields(self) -> tuple[ItemId, str, FormTypingLabel, FormTypingLabel]:
        """Return the fields the example is equal on."""
        return (
            self._item_id,
            self._text,
            self._true_label,
            self._observed_label,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another example presents the same labeled form."""
        return type(other) is FormTypingExample and self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the item, form and labels."""
        return hash((FormTypingExample, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this example."""
        return (
            f"FormTypingExample({self._item_id!r}, {self._text!r}, "
            f"{self._true_label!r}, {self._observed_label!r})"
        )


class TypingConflict:
    """One constraint the type checker left unsatisfied, at the node it arose at.

    Parameters
    ----------
    kind : TypingFailureKind
        Why the constraint remained unsatisfied.
    left : str
        One side of the constraint, in the semantic-type syntax.
    right : str
        The other side, in the semantic-type syntax.
    path : Path
        The node of the logical form the constraint arose at.

    Raises
    ------
    TypeError
        If a path step is not an integer.
    ValueError
        If either side is empty or a path step is negative.
    """

    __slots__ = ("_kind", "_left", "_path", "_right")

    def __init__(
        self, kind: TypingFailureKind, left: str, right: str, path: Path
    ) -> None:
        self._kind = kind
        self._left = ensure_name("a conflict's left type", left)
        self._right = ensure_name("a conflict's right type", right)
        self._path = tuple(
            ensure_non_negative(f"path[{at}]", step) for at, step in enumerate(path)
        )

    @property
    def kind(self) -> TypingFailureKind:
        """Return why the constraint remained unsatisfied."""
        return self._kind

    @property
    def left(self) -> str:
        """Return one side of the constraint in the semantic-type syntax."""
        return self._left

    @property
    def right(self) -> str:
        """Return the other side in the semantic-type syntax."""
        return self._right

    @property
    def path(self) -> Path:
        """Return the node of the logical form the constraint arose at."""
        return self._path

    def _fields(self) -> tuple[TypingFailureKind, str, str, Path]:
        """Return the fields the conflict is equal on."""
        return (self._kind, self._left, self._right, self._path)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another conflict records the same unsatisfied constraint."""
        if type(other) is not TypingConflict:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the kind, the two sides and the node."""
        return hash((TypingConflict, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this conflict."""
        return (
            f"TypingConflict({self._kind!r}, {self._left!r}, {self._right!r}, "
            f"{self._path!r})"
        )


class TypingFailure:
    """The refusal the type checker returned, standing as an ill-typed form's warrant.

    Parameters
    ----------
    conflicts : tuple[TypingConflict, ...]
        The constraints the checker left unsatisfied, in the order it
        returned them.
    language_digest : str
        The SHA-256 of the declaration bytes of the logical-form language the
        form was checked under.

    Raises
    ------
    ValueError
        If no conflict is named or the digest is empty.
    """

    __slots__ = ("_conflicts", "_language_digest")

    def __init__(
        self, conflicts: tuple[TypingConflict, ...], *, language_digest: str
    ) -> None:
        if not conflicts:
            msg = "a typing failure names no unsatisfied constraint"
            raise ValueError(msg)

        self._conflicts = conflicts
        self._language_digest = ensure_name(
            "a typing failure's language digest", language_digest
        )

    @property
    def conflicts(self) -> tuple[TypingConflict, ...]:
        """Return the constraints the checker left unsatisfied."""
        return self._conflicts

    @property
    def language_digest(self) -> str:
        """Return the digest of the language declaration the form was checked under."""
        return self._language_digest

    @property
    def kinds(self) -> tuple[TypingFailureKind, ...]:
        """Return the kind of each unsatisfied constraint, in the checker's order."""
        return tuple(conflict.kind for conflict in self._conflicts)

    def _fields(self) -> tuple[tuple[TypingConflict, ...], str]:
        """Return the fields the failure is equal on."""
        return (self._conflicts, self._language_digest)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another failure records the same refusal."""
        if type(other) is not TypingFailure:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the conflicts and the language digest."""
        return hash((TypingFailure, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this failure."""
        return (
            f"TypingFailure({self._conflicts!r}, "
            f"language_digest={self._language_digest!r})"
        )


class IllTyped:
    """A refused logical form, with the well-typed one it was corrupted from.

    Parameters
    ----------
    item_id : ItemId
        The grammatical item whose logical form was corrupted.
    kind : str
        The operator that produced the form.
    source : str
        The well-typed form the operator was applied to, in the logical-form
        surface syntax.
    text : str
        The refused form, in the logical-form surface syntax.
    certificate : TypingFailure
        The refusal the checker returned on the form.
    observed_label : FormTypingLabel
        The label training observes after applying the negative-data regime.

    Raises
    ------
    ValueError
        If the kind is unknown, either form is empty, or the two forms are
        the same text.
    """

    __slots__ = (
        "_certificate",
        "_item_id",
        "_kind",
        "_observed_label",
        "_source",
        "_text",
    )

    def __init__(
        self,
        item_id: ItemId,
        kind: str,
        source: str,
        text: str,
        certificate: TypingFailure,
        *,
        observed_label: FormTypingLabel,
    ) -> None:
        self._source = ensure_name(f"the source form of {item_id!r}", source)
        self._text = ensure_name(f"the ill-typed form of {item_id!r}", text)

        if self._source == self._text:
            msg = f"the ill-typed form of {item_id!r} is its own source"
            raise ValueError(msg)

        if observed_label not in FORM_TYPING_LABELS:
            msg = (
                f"the ill-typed form of {item_id!r} has an unknown observed label: "
                f"{observed_label!r}"
            )
            raise ValueError(msg)

        self._item_id = item_id
        self._kind = ensure_name("an ill-typing operator", kind)
        self._certificate = certificate
        self._observed_label = observed_label

    @property
    def item_id(self) -> ItemId:
        """Return the grammatical item whose logical form was corrupted."""
        return self._item_id

    @property
    def kind(self) -> str:
        """Return the operator that produced the form."""
        return self._kind

    @property
    def source(self) -> str:
        """Return the well-typed form the operator was applied to."""
        return self._source

    @property
    def text(self) -> str:
        """Return the refused form in the logical-form surface syntax."""
        return self._text

    @property
    def certificate(self) -> TypingFailure:
        """Return the refusal the checker returned on the form."""
        return self._certificate

    @property
    def true_label(self) -> FormTypingLabel:
        """Return the certified label, which is ill-typed by construction."""
        return "ill-typed"

    @property
    def observed_label(self) -> FormTypingLabel:
        """Return the label presented to logical-form training."""
        return self._observed_label

    @property
    def flipped(self) -> bool:
        """Return whether the observed label differs from the certified one."""
        return self._observed_label != self.true_label

    def _fields(
        self,
    ) -> tuple[ItemId, str, str, str, TypingFailure, FormTypingLabel]:
        """Return the fields the record is equal on."""
        return (
            self._item_id,
            self._kind,
            self._source,
            self._text,
            self._certificate,
            self._observed_label,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another record holds the same corruption of one form."""
        if type(other) is not IllTyped:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the item, the kind, the forms and the certificate."""
        return hash((IllTyped, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this record."""
        return (
            f"IllTyped({self._item_id!r}, {self._kind!r}, {self._source!r}, "
            f"{self._text!r}, {self._certificate!r}, "
            f"observed_label={self._observed_label!r})"
        )
