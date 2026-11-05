"""Computes the annotator agreement ceiling of a registered judgment design.

The design is read through its registered reader, so its responses, the
judgments its document keeps and the factors its document declares, each a
bead expression over the item records, are the design's own. Responses are
grouped by the levels of the named factors, and the ceiling is computed over
the groups.
"""

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import TYPE_CHECKING

from lofi.judgment import AgreementCeiling, ReadDesign, split_half_ceiling
from lofi.run.fits import RegisteredDesigns

if TYPE_CHECKING:
    from lofi.judgment import JudgmentDesign

COEFFICIENT = (
    "the Spearman rank correlation between the group means of two disjoint "
    "halves of each group's annotators, over groups as the unit, averaged over "
    "every balanced split and stepped up to the full complement of annotators "
    "by the Spearman-Brown formula"
)
"""Describes the reported correlation."""


def grouped(
    design: JudgmentDesign, factors: tuple[str, ...]
) -> dict[tuple[str, ...], list[float]]:
    """Return the design's responses grouped by the levels of the named factors.

    Parameters
    ----------
    design : JudgmentDesign
        The read design.
    factors : tuple[str, ...]
        Design factors whose levels together identify a group.

    Returns
    -------
    dict[tuple[str, ...], list[float]]
        Response levels per group in row order.

    Raises
    ------
    ValueError
        If no factor is named or the design declares no factor of a name.
    """
    if not factors:
        msg = "a ceiling groups responses by at least one factor"
        raise ValueError(msg)
    unknown = sorted(set(factors) - set(design.factor_levels))
    if unknown:
        msg = (
            f"the design declares no factor {unknown!r}; it declares "
            f"{sorted(design.factor_levels)!r}"
        )
        raise ValueError(msg)
    columns = tuple(design.factors[name].tolist() for name in factors)
    groups: dict[tuple[str, ...], list[float]] = {}
    for row, response in enumerate(design.responses.tolist()):
        key = tuple(
            design.factor_levels[name][int(column[row])]
            for name, column in zip(factors, columns, strict=True)
        )
        groups.setdefault(key, []).append(float(response))
    return groups


def record(
    ceiling: AgreementCeiling,
    dataset: str,
    factors: tuple[str, ...],
    digests: dict[str, str],
    *,
    splits: int,
    level: float,
) -> dict[str, str | float | int | list[str] | dict[str, str]]:
    """Return the estimate and its provenance as JSON fields."""
    lower, upper = ceiling.interval
    return {
        "value": ceiling.value,
        "lower": lower,
        "upper": upper,
        "level": level,
        "splits": splits,
        "items": ceiling.items,
        "responses": ceiling.responses,
        "coefficient": COEFFICIENT,
        "dataset": dataset,
        "group_by": list(factors),
        "source_digests": digests,
    }


def main(argv: list[str] | None = None) -> int:
    """Compute and print the agreement ceiling of one registered design.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments without the program name, or ``None`` for
        ``sys.argv``.

    Returns
    -------
    int
        Zero after writing the report.
    """
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", help="the registered judgment design's tag")
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        metavar="FACTOR",
        help="a design factor identifying a group; repeat for each factor",
    )
    parser.add_argument(
        "--splits",
        type=int,
        required=True,
        help="how many balanced splits to average the correlation over",
    )
    parser.add_argument(
        "--level",
        type=float,
        required=True,
        help="the coverage the interval aims at, as a proportion",
    )
    parser.add_argument("--out", type=Path, default=None, help="the JSON record")
    arguments = parser.parse_args(argv)

    dataset: str = arguments.dataset
    factors = tuple(str(factor) for factor in arguments.group)
    match RegisteredDesigns().read(dataset):
        case ReadDesign(design=design, provenance=provenance):
            pass
        case _:
            msg = f"the design {dataset!r} records bounded responses, not ordinal ones"
            raise ValueError(msg)
    groups = grouped(design, factors)
    ceiling = split_half_ceiling(
        groups.values(), splits=arguments.splits, level=arguments.level
    )
    written = record(
        ceiling,
        dataset,
        factors,
        dict(provenance.corpus_digests),
        splits=arguments.splits,
        level=arguments.level,
    )
    lower, upper = ceiling.interval
    sys.stdout.write(
        f"{dataset}\n"
        f"  groups     {ceiling.items}\n"
        f"  responses  {ceiling.responses}\n"
        f"  ceiling    {ceiling.value:.4f}\n"
        f"  interval   [{lower:.4f}, {upper:.4f}] at {arguments.level:.2f} coverage\n"
    )
    if arguments.out is not None:
        destination: Path = arguments.out
        destination.write_text(
            json.dumps(written, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        sys.stdout.write(f"  written to {destination}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
