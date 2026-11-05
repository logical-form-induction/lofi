"""The checked bridge from logical forms to lexical categories."""

from typing import TYPE_CHECKING, Protocol, final

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.terms import OpName, Term
    from lofi.typecheck._erase import Erasure
    from lofi.typecheck._lexsig import LexicalSignature
    from lofi.typecheck._ports import TypingSystem
    from lofi.typecheck._values import TypingRule

__all__ = ["CategoryTypeMismatchError", "check_logical_form_category"]


class _BridgeLanguage(Protocol):
    """The language capabilities needed by the category bridge."""

    @property
    def checker(self) -> TypingSystem[LexicalSignature]:
        """Return the logical-form type checker."""
        ...

    @property
    def lexicon(self) -> LexicalSignature:
        """Return the declared term-constant signature."""
        ...

    @property
    def rules(self) -> Mapping[OpName, TypingRule]:
        """Return the structural logical-form typing rules."""
        ...

    def print_type(self, term: Term, /) -> str:
        """Render one semantic type in the language's notation."""
        ...


@final
class CategoryTypeMismatchError(ValueError):
    """A logical form whose type cannot instantiate to a category's erasure."""

    __slots__ = ("category_type", "logical_form_type")

    def __init__(
        self,
        logical_form_type: Term,
        category_type: Term,
        /,
        *,
        language: _BridgeLanguage,
    ) -> None:
        """Name both incompatible types in the refusal."""
        self.logical_form_type = logical_form_type
        self.category_type = category_type
        super().__init__(
            f"logical-form type {language.print_type(logical_form_type)!r} does not "
            f"match category type {language.print_type(category_type)!r}"
        )


def check_logical_form_category(
    language: _BridgeLanguage,
    erasure: Erasure,
    logical_form: Term,
    category: Term,
    /,
) -> Term:
    """Return the category after checking its erased logical-form type."""
    category_type = erasure(category)
    inferred = language.checker.check(
        logical_form,
        language.lexicon,
        language.rules,
    )

    if inferred.residual or inferred.scheme is None:
        msg = (
            "logical form has no principal type to compare with category type "
            f"{language.print_type(category_type)!r}"
        )
        raise ValueError(msg)

    constrained = language.checker.check(
        logical_form,
        language.lexicon,
        language.rules,
        constraints={(): category_type},
    )

    if not constrained.is_well_typed or constrained.scheme is None:
        raise CategoryTypeMismatchError(
            inferred.scheme.body,
            category_type,
            language=language,
        )

    return category
