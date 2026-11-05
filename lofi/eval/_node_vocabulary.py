"""The node-vocabulary comparison over four persisted causal-probe sweeps."""

from typing import TYPE_CHECKING, Final, final

from lofi.eval._comparison import Reduction
from lofi.probe._node_vocabulary import NodeVocabularyComparison, NodeVocabularyRow
from lofi.probe._probing import read_probe_records
from lofi.registry import ComponentPlugin
from lofi.spec.reducers import (
    NodeVocabularyReducerSettings,
    NodeVocabularyReducerSpec,
    NodeVocabularySourceSettings,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.eval._reducers import ReductionContext

__all__ = ["PLUGIN"]


@final
class _NodeVocabularyReducer:
    """Read each arm's persisted sweep record and compare the four arms."""

    __slots__ = ("_context", "_settings")

    def __init__(
        self,
        settings: NodeVocabularyReducerSettings,
        context: ReductionContext,
    ) -> None:
        self._settings = settings
        self._context = context

    def _row(self, arm: NodeVocabularySourceSettings, /) -> NodeVocabularyRow:
        """Return one arm's row from the record its declared source holds.

        Raises
        ------
        ValueError
            If the source cannot be read, holds no record under the declared
            probe and name, or records another arm.
        """
        path = self._context.source(arm.source)

        try:
            records = read_probe_records(path)

        except (OSError, TypeError, ValueError) as refused:
            msg = f"the node-vocabulary source {path} cannot be read: {refused}"
            raise ValueError(msg) from refused

        named = records.get(arm.probe, {})

        if arm.record not in named:
            msg = (
                f"the node-vocabulary source {path} holds no record {arm.record!r} "
                f"under probe {arm.probe!r}"
            )
            raise ValueError(msg)

        row = NodeVocabularyRow.from_record(named[arm.record])

        if row.arm.value != arm.representation:
            msg = (
                f"the node-vocabulary source {path} records arm {row.arm.value!r}, "
                f"not the declared {arm.representation!r}"
            )
            raise ValueError(msg)

        return row

    def reduce(self, rows: Sequence[Mapping[str, JsonValue]], /) -> Reduction:
        """Return one summary row per arm in canonical arm order."""
        del rows
        comparison = NodeVocabularyComparison(
            self._row(arm) for arm in self._settings.arms
        )
        emitted: tuple[Mapping[str, JsonValue], ...] = tuple(
            {"reducer": "node-vocabulary", **summary}
            for summary in comparison.to_json()
        )

        return Reduction(emitted, ())


def _build(
    settings: NodeVocabularyReducerSettings,
    context: ReductionContext,
    /,
) -> _NodeVocabularyReducer:
    """Build the declared node-vocabulary comparison."""
    return _NodeVocabularyReducer(settings, context)


PLUGIN: Final = ComponentPlugin(spec=NodeVocabularyReducerSpec, build=_build)
"""The plugin the ``node-vocabulary`` reducer kind registers."""
