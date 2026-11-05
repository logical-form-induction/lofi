"""Response sampling from compiled programs.

Sampling and likelihood evaluation share the bound program, parameter layout,
row locations, and cutpoints.
"""

from typing import TYPE_CHECKING

import torch

from lofi.judgment._design import DesignCodec, JudgmentRow, RowColumns
from lofi.judgment.backends import latent
from lofi.judgment.ordinal import log_category_probabilities

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.judgment._design import JudgmentDesign
    from lofi.judgment.compiler import BoundProgram

__all__ = ["sample_responses", "simulated_design"]


def sample_responses(
    program: BoundProgram,
    params: Tensor,
    /,
    *,
    generator: torch.Generator,
) -> Tensor:
    """Return one response level per row, drawn at the given parameters.

    Parameters
    ----------
    program : BoundProgram
        Program bound to the design whose rows are sampled.
    params : Tensor
        Value of every drawn site, laid out as the program lays them out.
    generator : torch.Generator
        Source of the draws, so that a seed fixes the sample.

    Returns
    -------
    Tensor
        Zero-based levels, one per row, as integers.

    Raises
    ------
    TypeError
        If the vector holds values of a nonfloating kind.
    ValueError
        If the vector is not one axis of the layout's length.
    """
    location, thresholds = latent(program, params)
    probabilities = torch.exp(log_category_probabilities(location, thresholds))

    return torch.multinomial(probabilities, 1, generator=generator).squeeze(1)


def simulated_design(design: JudgmentDesign, responses: Tensor, /) -> JudgmentDesign:
    """Return a design with its recorded responses replaced by sampled ones.

    Every other column is kept: the participants, the items, the factors,
    the readouts and the alternatives of each row.

    Parameters
    ----------
    design : JudgmentDesign
        Design whose rows the responses were sampled over.
    responses : Tensor
        Zero-based levels, one per row of the design.

    Raises
    ------
    ValueError
        If the responses are not one integer per row.
    """
    if responses.is_floating_point() or responses.shape != (design.n_rows,):
        msg = (
            f"a design over {design.n_rows} rows takes one integer response per "
            f"row, and {responses.dtype} values of shape "
            f"{tuple(responses.shape)} were given"
        )

        raise ValueError(msg)

    codec = DesignCodec(
        design.response_levels,
        factors=tuple(design.factors),
        readouts=tuple(design.readouts),
        alternatives=design.alternatives_per_row,
    )
    levels = responses.tolist()

    return codec.encode(
        JudgmentRow(
            row.participant,
            row.item,
            level,
            RowColumns(
                dict(row.factors),
                dict(row.readouts),
                dict(row.alternative_readouts),
            ),
            row.alternatives,
        )
        for row, level in zip(codec.decode(design), levels, strict=True)
    )
