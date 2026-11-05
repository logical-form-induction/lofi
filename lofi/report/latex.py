"""The LaTeX report renderer, installed as the ``latex`` report format."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, final, override

from lofi.report._table import format_cell

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi._types import JsonValue

__all__ = ["LATEX", "LatexRenderer"]

_ESCAPES: Final = MappingProxyType(
    {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
)


def _cell(value: JsonValue, precision: int | None, /) -> str:
    """Return one escaped LaTeX cell."""
    return "".join(_ESCAPES.get(char, char) for char in format_cell(value, precision))


@final
class LatexRenderer:
    """Writes a report table as a ``tabular`` environment, one line per row."""

    __slots__ = ()

    @property
    def suffix(self) -> str:
        """Return ``".tex"``."""
        return ".tex"

    def render(
        self,
        labels: Sequence[str],
        rows: Sequence[Sequence[JsonValue]],
        /,
        *,
        precision: int | None,
    ) -> bytes:
        """Return the table as LaTeX.

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
            rf"\begin{{tabular}}{{{'l' * len(labels)}}}",
            " & ".join(_cell(label, None) for label in labels) + r" \\",
            r"\hline",
            *(
                " & ".join(_cell(value, precision) for value in row) + r" \\"
                for row in rows
            ),
            r"\end{tabular}",
        )

        return ("\n".join(lines) + "\n").encode()

    @override
    def __repr__(self) -> str:
        return "LatexRenderer()"


LATEX: Final = LatexRenderer()
"""The renderer the ``latex`` entry of ``lofi.report_format`` names."""
