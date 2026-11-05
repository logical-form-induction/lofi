"""The documents a publication of a trained point's Layers records reads.

A selection names the evaluator roles whose predictions are published, the
retained split they are published over and the Layers formalism slug their
labels are drawn from. A tokenization mapping joins each item, by its stable
identifier, to the Layers tokenization record its token positions index into.
"""

import didactic.api as dx

from lofi._checks import ensure_name

__all__ = ["PublishSelection", "TokenizationMapping", "TokenizationRecord"]


class PublishSelection(dx.Model, extra="forbid"):
    """Declares which predictions of a trained point are published.

    Parameters
    ----------
    evaluators : tuple[str, ...]
        The evaluator roles of the run whose predictions are published, each
        written to a records file of its own.
    split : str
        The retained split whose items are published.
    formalism : str
        The Layers slug naming the formalism the published labels are drawn
        from.
    """

    evaluators: tuple[str, ...]
    split: str
    formalism: str

    @dx.validates("evaluators")
    @classmethod
    def _names_distinct_roles(cls, roles: tuple[str, ...]) -> tuple[str, ...]:
        """Validate a nonempty run of distinct, named evaluator roles."""
        if not roles:
            msg = "a publish selection names no evaluator role"
            raise ValueError(msg)

        seen: set[str] = set()

        for role in roles:
            ensure_name("a published evaluator role", role)

            if role in seen:
                msg = f"a publish selection names the evaluator role {role!r} twice"
                raise ValueError(msg)

            seen.add(role)

        return roles

    @dx.validates("split")
    @classmethod
    def _names_a_split(cls, split: str) -> str:
        """Validate the published split's name."""
        return ensure_name("a published split", split)

    @dx.validates("formalism")
    @classmethod
    def _names_a_formalism(cls, formalism: str) -> str:
        """Validate the published labels' formalism slug."""
        return ensure_name("a published formalism", formalism)


class TokenizationRecord(dx.Model, extra="forbid"):
    """Joins one item to the Layers tokenization its spans index into.

    Parameters
    ----------
    item : str
        The item's stable identifier.
    tokenization : str
        The identifier of the Layers tokenization record.
    """

    item: str
    tokenization: str

    @dx.validates("item")
    @classmethod
    def _names_an_item(cls, item: str) -> str:
        """Validate the joined item's identifier."""
        return ensure_name("a tokenization mapping's item", item)

    @dx.validates("tokenization")
    @classmethod
    def _names_a_tokenization(cls, tokenization: str) -> str:
        """Validate the joined tokenization's identifier."""
        return ensure_name("a tokenization mapping's tokenization", tokenization)


class TokenizationMapping(dx.Model, extra="forbid"):
    """Declares the Layers tokenization of each item a publication may write.

    Parameters
    ----------
    tokenizations : tuple[TokenizationRecord, ...]
        One record per item, each item named once.
    """

    tokenizations: tuple[TokenizationRecord, ...]

    @dx.validates("tokenizations")
    @classmethod
    def _joins_each_item_once(
        cls, records: tuple[TokenizationRecord, ...]
    ) -> tuple[TokenizationRecord, ...]:
        """Validate that no item is joined to two tokenizations."""
        seen: set[str] = set()

        for record in records:
            if record.item in seen:
                msg = f"a tokenization mapping names the item {record.item!r} twice"
                raise ValueError(msg)

            seen.add(record.item)

        return records
