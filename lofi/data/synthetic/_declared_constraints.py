"""Declared surface-pattern constraints over a synthetic corpus."""

from typing import TYPE_CHECKING, Final

from lofi._types import ItemId
from lofi.data._constraint_sources import (
    ConstraintPattern,
    PinnedConstraintSource,
    ScoreQuantile,
)
from lofi.registry import ComponentPlugin
from lofi.spec.constraint_sources import (
    DeclaredConstraintSourceSettings,
    DeclaredConstraintSourceSpec,
)

if TYPE_CHECKING:
    from lofi.data.synthetic._gold_sources import ConstraintSourceContext

__all__ = ["PLUGIN"]


def _build(
    settings: DeclaredConstraintSourceSettings,
    context: ConstraintSourceContext,
    /,
) -> PinnedConstraintSource[str, str]:
    """Build the declared patterns under their domain's selection policy."""
    score_quantile = settings.score_quantile

    return PinnedConstraintSource(
        context.policy(settings.policy),
        tuple(
            ConstraintPattern(pattern.pattern, pattern.permitted)
            for pattern in settings.patterns
        ),
        score_quantile=(
            None
            if score_quantile is None
            else ScoreQuantile(
                score_quantile.quantile,
                {ItemId(item_id): score for item_id, score in score_quantile.scores},
            )
        ),
    )


PLUGIN: Final = ComponentPlugin(spec=DeclaredConstraintSourceSpec, build=_build)
"""The declared-pattern constraint source's plugin."""
