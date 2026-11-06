"""The Markdown report renderer, installed as the ``markdown`` report format."""

from typing import TYPE_CHECKING, Final, final, override

from lofi.report._table import format_cell

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi._types import JsonValue

__all__ = ["MARKDOWN", "MarkdownRenderer"]


def _cell(value: JsonValue, precision: int | None, /) -> str:
    """Return one escaped Markdown cell."""
    return (
        format_cell(value, precision)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\n", "<br>")
    )


@final
class MarkdownRenderer:
    """Writes a report table as a pipe table, one line per row."""

    __slots__ = ()

    @property
    def suffix(self) -> str:
        """Return ``".md"``."""
        return ".md"

    def render(
        self,
        labels: Sequence[str],
        rows: Sequence[Sequence[JsonValue]],
        /,
        *,
        precision: int | None,
    ) -> bytes:
        """Return the table as Markdown.

        Parameters
        ----------
        labels : Sequence[str]
            One heading per column, in column order.
        rows : Sequence[Sequence[JsonValue]]
            One sequence of scalar cells per table row, in column order.
        precision : int | None
            Decimal places for floating cells, or None to write them in full.
        """
        lines = (
            "| " + " | ".join(_cell(label, None) for label in labels) + " |",
            "| " + " | ".join("---" for _ in labels) + " |",
            *(
                "| " + " | ".join(_cell(value, precision) for value in row) + " |"
                for row in rows
            ),
        )

        return ("\n".join(lines) + "\n").encode()

    @override
    def __repr__(self) -> str:
        return "MarkdownRenderer()"


MARKDOWN: Final = MarkdownRenderer()
"""The renderer the ``markdown`` entry of ``lofi.report_format`` names."""
