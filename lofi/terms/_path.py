"""Path operations over interned terms."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi._native import Term, TermStore
    from lofi.terms._ports import Path

__all__ = ["at_path", "positions", "replace_at"]


def _child(term: Term, path: Path, index: int) -> Term:
    """Return one child or report where a path failed."""
    if index < 0 or index >= len(term.args):
        msg = f"path {path!r} reaches {term!r}, which has no child at index {index}"

        raise IndexError(msg)

    return term.args[index]


def at_path(term: Term, path: Path, /) -> Term:
    """Return the subterm addressed by a path."""
    current = term

    for index in path:
        current = _child(current, path, index)

    return current


def positions(term: Term, predicate: Callable[[Term], bool], /) -> tuple[Path, ...]:
    """Return matching paths in root-first, left-to-right preorder."""
    found: list[Path] = []
    pending: list[tuple[Path, Term]] = [((), term)]

    while pending:
        path, current = pending.pop()

        if predicate(current):
            found.append(path)

        pending.extend(
            ((*path, index), current.args[index])
            for index in range(len(current.args) - 1, -1, -1)
        )

    return tuple(found)


def replace_at(
    store: TermStore,
    term: Term,
    path: Path,
    value: Term,
    /,
) -> Term:
    """Return a term with one path functionally replaced."""
    if not path:
        return value

    ancestors: list[tuple[Term, int]] = []
    current = term

    for index in path:
        ancestors.append((current, index))
        current = _child(current, path, index)

    replacement = value

    for parent, index in reversed(ancestors):
        if replacement is parent.args[index]:
            replacement = parent

            continue

        arguments = (*parent.args[:index], replacement, *parent.args[index + 1 :])
        replacement = store.op(parent.op, *arguments, payload=parent.payload)

    return replacement
